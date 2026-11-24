"""Run the saved demonstration or compare its current TXT, without rewriting expectations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DEMO=ROOT/'examples/demonstrations/yolo_with_child_detector'


def compare(path):
    expected=(DEMO/'expected.txt').read_text(encoding='utf-8')
    actual=Path(path).read_text(encoding='utf-8')
    if actual != expected:
        old=set(t.strip() for t in expected.split(','));new=set(t.strip() for t in actual.split(','))
        raise ValueError(f'Result changed. Missing: {sorted(old-new)}; added: {sorted(new-old)}; exact text/order also checked')
    print(f'PASS: exact saved caption, {len(expected.split(","))} tags')


def request(server,path,data=None):
    req=urllib.request.Request(server.rstrip('/')+path,
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)


def run(server,timeout):
    graph=json.loads((DEMO/'workflow.json').read_text(encoding='utf-8'))
    definitions=request(server,'/object_info');prompt={}
    for node in graph['nodes']:
        kind=node['type'];schema=definitions[kind]['input']
        fields={**schema.get('required',{}),**schema.get('optional',{})}
        saved=node.get('widgets_values_named')
        if saved is None and node.get('widgets_values'):
            raise ValueError(f'Node {node["id"]} lacks named widget values; cannot safely reconstruct settings')
        values={k:v for k,v in (saved or {}).items() if k in fields}
        for slot in node.get('inputs',[]):
            if slot.get('link') is not None:
                link=next(row for row in graph['links'] if row[0]==slot['link'])
                values[slot['name']]=[str(link[1]),link[2]]
        prompt[str(node['id'])]={'class_type':kind,'inputs':values}
    destination=ROOT/'temp/saved-workflow-demo'/str(time.time_ns());destination.mkdir(parents=True)
    prompt['10']['inputs'].update(output_directory=str(destination),filename='actual',overwrite=False)
    if prompt['10']['inputs'].get('save_json') or not prompt['10']['inputs'].get('save_txt'):
        raise ValueError('Saved demonstration must save TXT only')
    ticket=request(server,'/prompt',{'prompt':prompt,'client_id':'wdv3-saved-workflow-demo'})
    if ticket.get('node_errors'):raise ValueError(json.dumps(ticket['node_errors'],ensure_ascii=False))
    prompt_id=ticket['prompt_id'];deadline=time.monotonic()+timeout
    print('Queued',prompt_id,flush=True)
    while time.monotonic()<deadline:
        history=request(server,'/history/'+prompt_id)
        if prompt_id in history:
            entry=history[prompt_id]
            if entry.get('status',{}).get('status_str')=='error':
                raise ValueError(json.dumps(entry.get('status'),ensure_ascii=False))
            compare(destination/'actual.txt')
            (destination/'history.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2),encoding='utf-8')
            return
        time.sleep(2)
    raise TimeoutError(f'Prompt {prompt_id} still pending; it was not cancelled. Check ComfyUI queue.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--actual-txt',type=Path);group.add_argument('--server')
    parser.add_argument('--timeout',type=float,default=1800)
    args=parser.parse_args()
    manifest=json.loads((DEMO/'manifest.json').read_text(encoding='utf-8'))
    for filename,key in [('workflow.json','workflow_sha256'),('expected.txt','expected_sha256')]:
        if hashlib.sha256((DEMO/filename).read_bytes()).hexdigest()!=manifest[key]:
            raise ValueError(f'{filename} differs from captured fixture')
    if args.actual_txt:compare(args.actual_txt)
    else:run(args.server,args.timeout)

if __name__=='__main__':
    try:main()
    except Exception as error:
        print('FAIL:',error,file=sys.stderr);raise SystemExit(1)
