"""Portable standalone reports with original geometry, crops and safe HTML."""
import base64
from copy import deepcopy
import hashlib
from html import escape
from io import BytesIO
import json
from pathlib import Path
import uuid

import numpy as np
from PIL import Image, ImageDraw
import torch
from .Scripts.evidence import Bundle
from .comfy_evidence import formatted_text


def make_report(value):
    if value.kind not in ('results','evidence','regions','tag_sets'): raise ValueError('Inspect expects regions, evidence, tag_sets or results')
    data=value.data()
    if value.kind == 'tag_sets':
        from .Scripts.tag_postprocessing import local_decisions
        for frame in data['frames']:
            frame['decisions'] = local_decisions(frame)
    payload = dict(stage=value.kind, model=data.get('model'), frames=data['frames'],
                   source_fingerprint=value.fingerprint, schema_version=1)
    if 'postprocessing' in data:
        payload['postprocessing'] = data['postprocessing']
    return Bundle.pack('report', payload)


def preview_report(report, temp_directory):
    frames=report.data()['frames']; rows=[]; tensors=[]; tags=[]
    root=Path(temp_directory); root.mkdir(parents=True,exist_ok=True)
    for index, frame in enumerate(frames):
        encoded=frame.get('image_png')
        if not encoded: raise ValueError('Report image pixels are missing')
        pixels=base64.b64decode(encoded)
        stem='wdv3-inspect-'+hashlib.sha256(pixels).hexdigest()[:20]
        (root/(stem+'.png')).write_bytes(pixels)
        masks={}
        for name, regions in frame.get('detectors',{}).items():
            for region in regions:
                if region.get('mask_png'):
                    mask_name=stem+'-'+hashlib.sha256((name+str(region['region_id'])+region['mask_png']).encode()).hexdigest()[:12]+'.png'
                    (root/mask_name).write_bytes(base64.b64decode(region['mask_png']))
                    masks[f"{name}:{region['region_id']}"]={'filename':mask_name,'type':'temp','subfolder':''}
        detail=deepcopy(frame)
        detail.pop('image_png',None)
        for regions in detail.get('detectors',{}).values():
            for region in regions: region.pop('mask_png',None)
        detail['stage']=report.data()['stage']
        for record in [detail.setdefault('full_image',{})]+[r for rs in detail.get('detectors',{}).values() for r in rs]:
            record.setdefault('general',{}); record.setdefault('character',{}); record.setdefault('ratings',{})
            record.setdefault('taglist',', '.join(sorted(set(record['general'])|set(record['character']))))
        detail.setdefault('merged_tags',detail['full_image'])
        detail.setdefault('image_index',index)
        rows.append(dict(detail=detail,image={'filename':stem+'.png','type':'temp','subfolder':''},
                         image_size=frame['image_size'],region_masks=masks))
        with Image.open(BytesIO(pixels)) as original:
            marked=original.convert('RGB'); drawing=ImageDraw.Draw(marked)
            for regions in frame.get('detectors',{}).values():
                for region in regions: drawing.rectangle(region['bbox'],outline='#00ddff',width=2)
            tensors.append(torch.from_numpy(np.array(marked).astype(np.float32)/255).unsqueeze(0))
            marked.close()
        tags.append(formatted_text(detail))
    return rows,tensors,tags


HTML_SCRIPT = r'''
const DATA=JSON.parse(document.getElementById('report-data').textContent);
let index=0, page=0, selected=null;
const el=id=>document.getElementById(id), canvas=el('image'), ctx=canvas.getContext('2d'), img=new Image();
const frame=()=>DATA.frames[index], regions=()=>Object.entries(frame().detectors||{}).flatMap(([name,rs])=>rs.map(r=>({...r,name,key:name+':'+r.region_id})));
function draw(){if(!img.complete||!img.naturalWidth)return;canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.drawImage(img,0,0);for(const r of regions()){ctx.strokeStyle=selected===r.key?'#00ddff':'white';ctx.lineWidth=selected===r.key?4:1;const [x,y,x2,y2]=r.bbox;ctx.strokeRect(x,y,x2-x,y2-y)}}
img.onload=draw;
function load(){page=0;selected=null;img.src=frame().image_asset;el('tags').textContent=frame().display_tags||frame().merged_tags?.taglist||frame().full_image?.taglist||'';
const f=frame(),added=[],removed=[];if(f.merged_tags)for(const c of ['general','character']){for(const t of Object.keys(f.merged_tags[c]||{}))if(!Object.hasOwn(f.full_image?.[c]||{},t))added.push(t);for(const t of Object.keys(f.full_image?.[c]||{}))if(!Object.hasOwn(f.merged_tags[c]||{},t))removed.push(t)}el('changes').textContent=f.merged_tags?'Added: '+added.join(', ')+' | Removed: '+removed.join(', '):'';render()}
function render(){const query=el('search').value.toLowerCase(),status=el('status').value;
 let items=frame().decisions||Object.entries(frame().detectors||{}).flatMap(([source,rs])=>rs.flatMap(r=>['general','character'].flatMap(category=>Object.entries(r[category]||{}).map(([tag,score])=>({tag,score,category,source,region_id:r.region_id,status:'raw',reason:'raw_evidence'})))));
 items=items.filter(r=>(status==='all'||r.status===status)&&(r.tag+' '+r.source+' '+r.reason).toLowerCase().includes(query)&&(!selected||r.source+':'+r.region_id===selected));
 items.sort((a,b)=>b.score-a.score);page=Math.min(page,Math.max(0,Math.ceil(items.length/50)-1));el('rows').replaceChildren();
 for(const r of items.slice(page*50,(page+1)*50)){const tr=document.createElement('tr');const causes=(r.postprocessing?.causes||[]).map(c=>c.method+(c.method==='regex'?' ['+c.rule.id+'] '+c.rule.specific+' → '+c.rule.general:'')+': '+c.witness+' → '+r.tag).join('; ');for(const v of [r.tag,r.category,(100*r.score).toFixed(2)+'%',r.source+(r.region_id==null?'':' #'+r.region_id),r.status,r.reason+(causes?' · '+causes:'')]){const td=document.createElement('td');td.textContent=v;tr.append(td)}tr.onclick=()=>{if(r.region_id!=null){selected=r.source+':'+r.region_id;draw();render()}};el('rows').append(tr)}
 el('page').textContent=(page+1)+' / '+Math.max(1,Math.ceil(items.length/50))+' · '+items.length+' records';
 const record=selected?regions().find(r=>r.key===selected):frame().full_image;
 el('history').textContent=(record?.postprocessing_history||[]).map(h=>'Operation '+(h.operation_index+1)+'\nBefore: '+Object.keys(h.before).join(', ')+'\nAfter: '+Object.keys(h.after).join(', ')).join('\n\n')+(record?.region_uid?'\nCascade: '+(record.stage_path||[]).join(' / ')+'\nParent: '+record.parent_region_uid+'\nDetector input: '+JSON.stringify(record.detector_input_bbox)+'\nOwn: '+JSON.stringify({general:record.general,character:record.character})+'\nAggregate: '+JSON.stringify(record.aggregate||{})+'\nRouting: '+JSON.stringify(record.route_history||[]):'')+(frame().cascade?'\nExecution: '+JSON.stringify(frame().cascade.report||[]):'');
 el('assets').replaceChildren();for(const r of regions()){const a=document.createElement('a');a.href=r.crop_asset;a.textContent=r.name+' #'+r.region_id+' crop';el('assets').append(a);if(r.mask_asset){const m=document.createElement('a');m.href=r.mask_asset;m.textContent='mask';el('assets').append(m)}}}
canvas.onclick=e=>{const b=canvas.getBoundingClientRect(),x=(e.clientX-b.left)*canvas.width/b.width,y=(e.clientY-b.top)*canvas.height/b.height;
 const hits=regions().filter(r=>r.bbox[0]<=x&&x<r.bbox[2]&&r.bbox[1]<=y&&y<r.bbox[3]).sort((a,b)=>(a.bbox[2]-a.bbox[0])*(a.bbox[3]-a.bbox[1])-(b.bbox[2]-b.bbox[0])*(b.bbox[3]-b.bbox[1]));selected=hits.length?hits[(hits.findIndex(r=>r.key===selected)+1)%hits.length].key:null;draw();render()};
DATA.frames.forEach((f,i)=>{const o=document.createElement('option');o.value=i;o.textContent=f.image_path||'Image '+(i+1);el('frame').append(o)});
el('frame').onchange=()=>{index=Number(el('frame').value);load()};el('search').oninput=()=>{page=0;render()};el('status').onchange=()=>{page=0;render()};el('prev').onclick=()=>{page=Math.max(0,page-1);render()};el('next').onclick=()=>{page++;render()};el('reset').onclick=()=>{selected=null;draw();render()};load();
'''


def export_report(report, output_directory, prefix='report', save_json=True, save_html=True, save_txt=True):
    if report.kind!='report': raise ValueError('Export Report needs a report from Inspect')
    if not any((save_json,save_html,save_txt)): raise ValueError('Enable a report format')
    if not prefix.strip() or any(not(c.isalnum() or c in ' _-') for c in prefix): raise ValueError('Invalid report prefix')
    data=report.data()
    if save_txt and any('merged_tags' not in f for f in data['frames']):
        raise ValueError('Global TXT export requires a filtered full-image result; select Results in Inspect')
    root=Path(output_directory).expanduser().resolve()/('wdv3-'+prefix+'-'+uuid.uuid4().hex)
    root.mkdir(parents=True,exist_ok=False);assets=root/'assets';assets.mkdir()
    for index,frame in enumerate(data['frames']):
        frame['display_tags']=formatted_text(frame)
        image_bytes=base64.b64decode(frame.pop('image_png'))
        name=f'image-{index}.png';(assets/name).write_bytes(image_bytes);frame['image_asset']='assets/'+name
        with Image.open(BytesIO(image_bytes)) as image:
            for source, regions in frame.get('detectors',{}).items():
                for r in regions:
                    stem=f'{index}-'+hashlib.sha256((source+str(r['region_id'])).encode()).hexdigest()[:12]
                    with image.crop(r['bbox']) as crop:crop.save(assets/(stem+'.png'))
                    r['crop_asset']='assets/'+stem+'.png'
                    if r.get('mask_png'):
                        (assets/(stem+'-mask.png')).write_bytes(base64.b64decode(r.pop('mask_png')))
                        r['mask_asset']='assets/'+stem+'-mask.png'
    paths={}
    if save_json:
        target=root/'report.json';target.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8');paths['json']=str(target)
    if save_txt:
        target=root/'tags.txt';target.write_text('\n'.join(formatted_text(f) for f in data['frames']),encoding='utf8');paths['txt']=str(target)
    if save_html:
        payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
        title=escape(prefix)
        html=f'''<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{background:#20242c;color:#eee;font:14px sans-serif;max-width:1100px;margin:auto;padding:20px}}canvas{{max-width:100%;max-height:500px}}td,th{{padding:6px;border-bottom:1px solid #555;text-align:left}}button,input,select{{margin:6px;padding:6px}}#assets a{{display:inline-block;margin:8px;color:#8deaff}}</style></head><body>
<h1>{title}</h1><select id="frame"></select><button id="reset">All regions</button><div><canvas id="image"></canvas></div><pre id="tags"></pre><pre id="history" style="white-space:pre-wrap"></pre><div id="changes"></div>
<input id="search" placeholder="Search tag, source or reason"><select id="status"><option value="all">All</option><option value="accepted">Accepted</option><option value="rejected">Rejected</option></select>
<button id="prev">Previous</button><span id="page"></span><button id="next">Next</button><table><thead><tr><th>Tag</th><th>Category</th><th>Score</th><th>Source</th><th>Status</th><th>Reason</th></tr></thead><tbody id="rows"></tbody></table><div id="assets"></div>
<script type="application/json" id="report-data">{payload}</script><script>{HTML_SCRIPT}</script></body></html>'''
        target=root/'report.html';target.write_text(html,encoding='utf8');paths['html']=str(target)
    return paths
