"""Serve real Results Viewer JS in an isolated DOM-widget mount, without models."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>WDV3 viewer layout regression</title>
<style>body{margin:20px;background:#15171b;color:#eee;font:14px sans-serif}button,select{margin:3px;padding:6px}#mock-node{display:flex;flex-direction:column;position:relative;width:760px;height:900px;background:#383838;border:1px solid #777;border-radius:8px}header{flex:0 0 52px;padding:8px;box-sizing:border-box}.mount{flex:1;display:flex;align-items:flex-start;width:100%;min-height:0;min-width:0}#measure{white-space:pre-wrap}pre{white-space:pre-wrap}</style></head>
<body><h1>Results Viewer layout regression</h1><button id="wider">Wider node</button><button id="narrower">Narrower node</button><button id="taller">Taller node</button><button id="rerun">Render results again</button><pre id="measure"></pre><pre id="checks">Manual mode</pre><div id="mock-node"><header>WDV3 Results Viewer — isolated widget mount</header><div class="mount" id="mount"></div></div>
<script type="module">
window.testApp={registerExtension(ext){window.viewerExtension=ext},graph:{setDirtyCanvas(){}}};
await import('/extensions/wdv3/results.js');
const root=document.querySelector('#mock-node'),mount=document.querySelector('#mount');
class Node {
 constructor(){this.size=[760,900];this.widgets=[]}
 setSize(size){this.size=[...size];root.style.width=size[0]+'px';root.style.height=size[1]+'px'}
 addDOMWidget(name,type,element,options){mount.append(element);const w={name,type,element,options};this.widgets.push(w);return w}
}
await window.viewerExtension.beforeRegisterNodeDef(Node,{name:'WDV3Preview'});
const node=new Node();
const score={general:{blue_eyes:.95,brown_hair:.8},character:{},ratings:{general:.9},caption:'blue_eyes, brown_hair',taglist:'blue eyes, brown hair'};
const shapes=[['portrait',600,900],['landscape',1200,500],['square',700,700],['portrait-small',200,400]];
const rows=shapes.map(([name,w,h],i)=>({image:{filename:name,subfolder:'',type:'temp'},image_size:[w,h],detail:{image_path:'C:/dataset/'+name+'.png',image_index:i,image_size:[w,h],full_image:score,merged_tags:score,detectors:{eyes:[{...score,region_id:0,bbox:[10,10,100,100]}]}}}));
const render=()=>{node.onExecuted({wdv3_results:rows,wdv3_total:[4]});measure()};
function measure(){const p=node.wdv3Panel,c=p?.querySelector('canvas'),m=mount.getBoundingClientRect(),r=p?.getBoundingClientRect(),v=c?.getBoundingClientRect();document.querySelector('#measure').textContent=JSON.stringify({node:node.size,mountWidth:m.width,mountHeight:m.height,panelWidth:r?.width,panelHeight:r?.height,panelLeft:r?.left,mountLeft:m.left,canvasWidth:v?.width,canvasHeight:v?.height,minHeight:node.widgets[0]?.options.getMinHeight?.()},null,2)}
document.querySelector('#wider').onclick=()=>{node.setSize([1000,node.size[1]]);measure()};
document.querySelector('#narrower').onclick=()=>{node.setSize([520,node.size[1]]);measure()};
document.querySelector('#taller').onclick=()=>{node.setSize([node.size[0],1200]);measure()};
document.querySelector('#rerun').onclick=render;
new ResizeObserver(measure).observe(mount);mount.addEventListener('change',()=>requestAnimationFrame(measure));
mount.addEventListener('load',measure,true);render();
if(new URLSearchParams(location.search).has('autorun')) {
 const checks=[], pause=()=>new Promise(resolve=>setTimeout(resolve,70));
 const near=(a,b)=>Math.abs(a-b)<1;
 async function verify(index){
  const choice=node.wdv3Panel.querySelector('select');choice.value=String(index);choice.dispatchEvent(new Event('change'));
  let canvas;
  for(let attempt=0;attempt<100;attempt++){await pause();canvas=node.wdv3Panel.querySelector('canvas');if(canvas?.width===shapes[index][1]&&canvas.height===shapes[index][2])break}
  const m=mount.getBoundingClientRect(),p=node.wdv3Panel.getBoundingClientRect(),c=canvas.getBoundingClientRect();
  if(!near(p.width,m.width)||!near(p.left,m.left)||!near(p.height,m.height))throw Error('Panel did not fill mount: '+JSON.stringify({index,mw:m.width,pw:p.width,mh:m.height,ph:p.height}));
  if(!near(c.width/c.height,shapes[index][1]/shapes[index][2]))throw Error('Image aspect ratio changed');
  if(c.width<p.width-50||c.width>p.width)throw Error('Canvas did not adapt to panel width');
  checks.push({image:shapes[index][0],node:[...node.size],panel:[p.width,p.height],canvas:[c.width,c.height]});
  const selects=node.wdv3Panel.querySelectorAll('select'),source=selects[1];source.value='2';source.dispatchEvent(new Event('change'));await pause();
  if(!near(node.wdv3Panel.getBoundingClientRect().width,p.width))throw Error('Region choice changed panel width');
 }
 try {
  for(let repeat=0;repeat<2;repeat++)for(let i=0;i<shapes.length;i++)await verify(i);
  document.querySelector('#wider').click();await verify(1);
  document.querySelector('#narrower').click();await verify(0);
  document.querySelector('#taller').click();await verify(2);
  document.querySelector('#rerun').click();await verify(3);
  if(node.size[0]!==520||node.size[1]!==1200)throw Error('Render reset user node size');
  if(node.widgets[0].options.getMinHeight()!==620||node.widgets[0].options.getHeight()!=='100%')throw Error('Native layout options missing');
  document.querySelector('#checks').textContent=JSON.stringify({status:'passed',checks},null,2);document.documentElement.dataset.testStatus='passed';
 }catch(error){document.querySelector('#checks').textContent=JSON.stringify({status:'failed',message:String(error),checks},null,2);document.documentElement.dataset.testStatus='failed'}
}
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/':
            content, kind = HTML.encode(), 'text/html; charset=utf-8'
        elif url.path == '/scripts/app.js':
            content, kind = b'export const app=window.testApp;', 'text/javascript'
        elif url.path == '/scripts/api.js':
            content, kind = b'export const api={apiURL:path=>path};', 'text/javascript'
        elif url.path.startswith('/extensions/wdv3/'):
            name = url.path.rsplit('/', 1)[-1]
            path = (ROOT/'web'/name).resolve()
            if path.parent != (ROOT/'web').resolve() or not path.is_file():
                self.send_error(404)
                return
            content, kind = path.read_bytes(), 'text/javascript'
        elif url.path == '/view':
            name = parse_qs(url.query).get('filename', ['portrait'])[0]
            width, height = {'portrait': (600, 900), 'landscape': (1200, 500),
                             'square': (700, 700), 'portrait-small': (200, 400)}.get(name, (600, 900))
            content = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
                f'<rect width="100%" height="100%" fill="#2f6679"/><path d="M0 0L{width} {height}M{width} 0L0 {height}" stroke="#91c9dc" stroke-width="5"/>'
                f'<text x="20" y="60" fill="white" font-size="36">{width} x {height}</text></svg>').encode()
            kind = 'image/svg+xml'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8789)
    args = parser.parse_args()
    server = HTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Viewer layout harness: http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()
