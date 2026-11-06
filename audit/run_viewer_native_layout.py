"""Read-only ComfyUI frontend proxy: exercise native graph/DOM-widget layout."""
import argparse
from html import unescape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import threading
import urllib.request
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = r'''
const out=document.createElement('pre');out.id='wdv3-native-checks';document.body.append(out);
let phase='startup',checks=[];
const report=value=>{out.textContent=JSON.stringify({...value,phase,checks})};report({status:'loading'});
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
try {
 let app;
 for(let i=0;i<250;i++){app=window.comfyAPI?.app?.app;if(app?.graph&&app?.canvas&&window.LiteGraph?.registered_node_types?.WDV3Preview)break;await pause(100)}
 if(!app?.graph||!window.LiteGraph?.registered_node_types?.WDV3Preview)throw Error('Comfy frontend not ready');
 await pause(2000);
 for(const nodeType of ['WDV3Preview','WDV3Inspect']) {
 phase='create-native-node';app.graph.clear();
 const node=window.LiteGraph.createNode(nodeType);node.pos=[100,100];app.graph.add(node);node.setSize([760,950]);
 app.canvas.ds.scale=.75;app.canvas.ds.offset=[0,0];
 const scores={general:{blue_eyes:.95,brown_hair:.85},character:{},ratings:{general:.9},caption:'blue_eyes, brown_hair',taglist:'blue eyes, brown hair'};
 const shapes=[['portrait',600,900],['landscape',1200,500],['square',700,700],['portrait-small',200,400]];
 const rows=shapes.map(([name,w,h],i)=>({image:{filename:'wdv3-native-'+name,subfolder:'',type:'temp'},image_size:[w,h],detail:{image_path:'C:/test/'+name+'.png',image_index:i,image_size:[w,h],full_image:scores,merged_tags:scores,detectors:{eyes:[{...scores,region_id:0,bbox:[10,10,100,100]}]}}}));
 const draw=()=>{app.graph.setDirtyCanvas(true,true);app.canvas.draw(true,true)};
 node.onExecuted({wdv3_results:rows,wdv3_total:[4]});draw();
 for(let i=0;i<100&&!node.wdv3Panel?.parentElement?.classList.contains('dom-widget');i++){draw();await pause(70)}
 const panel=node.wdv3Panel,widget=node.widgets.find(w=>w.name==='wdv3_results');
 app.canvas.selectNode(node);await pause(800);app.canvas.ds.scale=.75;app.canvas.ds.offset=[0,0];draw();await pause(300);
 if(!panel?.parentElement?.classList.contains('dom-widget'))throw Error('Native widget mount missing');
 function sample(label){const p=panel.getBoundingClientRect(),m=panel.parentElement.getBoundingClientRect();return{label,node:[...node.size],nodeWidth:node.width,renderingSize:[...node.renderingSize],nodeType,allWidgetWidths:node.widgets.map(w=>({name:w.name,width:w.width})),scale:app.canvas.ds.scale,widgetWidth:widget.width,widgetHeight:widget.computedHeight,panel:[p.width,p.height],mount:[m.width,m.height],cssPanel:[panel.clientWidth,panel.clientHeight],hostStyle:panel.parentElement.getAttribute('style'),panelStyle:panel.getAttribute('style')}}
 async function verify(index,label){phase=label;const select=panel.querySelector('select');select.focus();select.click();select.value=String(index);select.dispatchEvent(new Event('change',{bubbles:true}));draw();await pause(150);draw();await pause(150);const s=sample(label);checks.push(s);report({status:'running'});const expected=(node.size[0]-2*widget.margin)*app.canvas.ds.scale;if(Math.abs(s.panel[0]-expected)>2)throw Error('Panel shrank against native node: '+JSON.stringify({expected,...s}));}
 for(let repeat=0;repeat<2;repeat++)for(let index=0;index<shapes.length;index++)await verify(index,'frame-'+repeat+'-'+index);
 node.setSize([1000,1100]);draw();await verify(1,'resize-wide');
 node.setSize([520,900]);draw();await verify(0,'resize-narrow');
 app.canvas.ds.scale=1;draw();await verify(2,'zoom-100');
 app.canvas.ds.scale=.65;draw();await verify(3,'zoom-65');
 }
 report({status:'passed'});
}catch(error){report({status:'failed',message:String(error),stack:error.stack})}
'''


class Proxy(BaseHTTPRequestHandler):
    upstream = 'http://127.0.0.1:8188'

    def do_GET(self):
        if self.path == '/__wdv3_native_test.js':
            content, kind, code = SCRIPT.encode(), 'text/javascript', 200
        elif self.path.startswith('/view?') and 'wdv3-native-' in self.path:
            name = self.path.split('wdv3-native-', 1)[1].split('&', 1)[0]
            w, h = {'portrait': (600, 900), 'landscape': (1200, 500),
                    'square': (700, 700), 'portrait-small': (200, 400)}.get(name, (600, 900))
            content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="#32677c"/></svg>'.encode()
            kind, code = 'image/svg+xml', 200
        else:
            try:
                response = urllib.request.urlopen(self.upstream+self.path, timeout=15)
            except HTTPError as error:
                response = error
            except Exception:
                self.send_error(502)
                return
            with response:
                content, kind, code = response.read(), response.headers.get('Content-Type', 'application/octet-stream'), response.status
            if self.path == '/' or self.path.startswith('/?'):
                content = content.replace(b'</body>', b'<script type="module" src="/__wdv3_native_test.js"></script></body>')
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # Deliberately do not proxy POST/PUT/DELETE: no queue, preferences or user data writes.
    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', type=Path, required=True)
    args = parser.parse_args()
    work = ROOT/'temp/viewer-native'
    work.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Proxy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run([str(args.browser), '--headless', '--disable-gpu', '--no-first-run',
            '--no-default-browser-check', '--disable-extensions', '--disable-crash-reporter', '--disable-breakpad',
            f'--user-data-dir={work/"profile"}', '--dump-dom', '--virtual-time-budget=35000',
            '--window-size=1400,1100', f'http://127.0.0.1:{server.server_port}/'],
            capture_output=True, timeout=58, creationflags=subprocess.CREATE_NO_WINDOW)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    html = result.stdout.decode('utf-8', errors='replace')
    (work/'page.html').write_text(html, encoding='utf-8')
    (work/'browser.log').write_bytes(result.stderr)
    found = re.search(r'<pre id="wdv3-native-checks">(.*?)</pre>', html, re.S)
    if not found:
        raise AssertionError('Native frontend test did not start; see temp/viewer-native logs')
    report = json.loads(unescape(found.group(1)))
    destination = ROOT/'docs/verification/cascaded-regions/viewer-native-layout.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status':report['status'],'checks':len(report['checks']),'message':report.get('message')}, ensure_ascii=False))
    assert report['status'] == 'passed', report.get('message', report['status'])


if __name__ == '__main__':
    main()
