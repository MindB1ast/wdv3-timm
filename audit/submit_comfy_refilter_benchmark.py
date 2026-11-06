"""Submit the same bounded graphs through the local ComfyUI API, without CLI startup per batch."""
import json
from pathlib import Path
import time
from urllib.request import Request,urlopen
import uuid

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/benchmarks/2026-11-05-test-yolo-wd'
OLD=BASE/'comfyui-pipeline'
OUT=BASE/'comfyui-pipeline-optimized'


def main():
    client=uuid.uuid4().hex
    previous=json.loads((OLD/'jobs.json').read_text(encoding='utf-8'))
    runs=[];started=time.time()*1000
    assert not list((OUT/'tags').glob('*.txt')),'Benchmark destination is already populated'
    for job in previous:
        graph=json.loads(Path(job['workflow_path']).read_text(encoding='utf-8'))
        for node in graph.values():
            if node['class_type']=='WDV3SaveResult':node['inputs']['output_directory']=str(OUT/'tags')
        path=OUT/'workflows'/Path(job['workflow_path']).name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
        request=Request('http://127.0.0.1:8188/prompt',method='POST',
                        data=json.dumps(dict(prompt=graph,client_id=client)).encode('utf-8'),
                        headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=30) as response:reply=json.load(response)
        assert not reply.get('node_errors'),reply
        runs.append(dict(start=job['start'],count=job['count'],prompt_id=reply['prompt_id']))
        (OUT/'submitted.json').write_text(json.dumps(dict(runs=runs,submission_start_ms=started,
                     submission_finished_ms=time.time()*1000),indent=2),encoding='utf-8')
    (OUT/'image-map.json').write_bytes((OLD/'image-map.json').read_bytes())
    print(json.dumps(dict(submitted=len(runs),images=sum(r['count'] for r in runs),
                         submission_seconds=(time.time()*1000-started)/1000,last=runs[-1]),indent=2),flush=True)


if __name__=='__main__':main()
