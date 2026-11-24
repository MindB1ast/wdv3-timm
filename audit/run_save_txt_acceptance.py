"""Validate and execute tags-only Save Result against installed ComfyUI."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys


def complete(coroutine):
    try: coroutine.send(None)
    except StopIteration as done: return done.value
    raise AssertionError('Unexpected asynchronous validation')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--comfy-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    work=root/'temp/save-txt-acceptance';work.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(work/'yolo');os.environ['MPLCONFIGDIR']=str(work/'mpl')
    sys.path[:0]=[str(args.comfy_root),str(root)];sys.argv=['save-txt-test','--cpu','--base-directory',str(work)]
    import comfy.options
    comfy.options.enable_args_parsing()
    from audit.test_comfy_nodes import PACKAGE
    import nodes
    import execution
    nodes.NODE_CLASS_MAPPINGS.update(PACKAGE.NODE_CLASS_MAPPINGS)
    cls=PACKAGE.NODE_CLASS_MAPPINGS['WDV3SaveResult']
    assert 'results_json' in cls.INPUT_TYPES()['optional']
    class TextSource:
        INPUT_TYPES=classmethod(lambda cls:{'required':{}})
        RETURN_TYPES=('STRING',)
        FUNCTION='run'
    nodes.NODE_CLASS_MAPPINGS['SaveTextFixture']=TextSource
    inputs=dict(tags=['text',0],output_directory=str(work),filename='tags-only',result_index=0,save_txt=True,save_json=False,overwrite=True)
    graph={'text':{'class_type':'SaveTextFixture','inputs':{}},'save':{'class_type':'WDV3SaveResult','inputs':inputs}}
    valid=complete(execution.validate_prompt('txt-only',graph,None))
    assert valid[0],valid
    output=cls.execute(output_directory=[str(work)],filename=['tags-only'],result_index=[0],
        save_txt=[True],save_json=[False],overwrite=[True],tags=['blue eyes, dress']).result
    assert Path(output[0]).read_text()=='blue eyes, dress' and output[1]==''
    inputs['save_json']=True
    assert not complete(execution.validate_prompt('missing-json',graph,None))[0]
    from audit.test_comfy_export import region_export_fixture
    region_cls=PACKAGE.NODE_CLASS_MAPPINGS['WDV3SaveRegionResults']
    region_fields=region_cls.INPUT_TYPES()['required']
    settings={name: entry[1]['default'] for name, entry in region_fields.items()
              if len(entry)>1 and 'default' in entry[1]}
    settings.update(output_directory=str(work),filename='regions-acceptance',write_mode='overwrite')
    region_prompt={'text':graph['text'],'save':{'class_type':'WDV3SaveRegionResults',
        'inputs':dict(settings,results_json=['text',0])}}
    valid=complete(execution.validate_prompt('region-export',region_prompt,None))
    assert valid[0],valid
    fixture=region_export_fixture()
    saved=region_cls.execute([json.dumps(fixture)],**{key:[value] for key,value in settings.items()}).result
    assert saved[1]=='written'
    [image]=json.loads(Path(saved[0]).read_text(encoding='utf-8'))['images']
    assert [(r['source'],r['general']) for r in image['regions']]==[('person',['shirt']),('face',['blue eyes'])]
    assert all('raw_scores' not in r and 'aggregate' not in r for r in image['regions'])
    assert set(('include_regions','include_general','include_character')) <= region_cls.INPUT_TYPES()['optional'].keys()
    region_prompt['save']['inputs'].update(include_regions=False,include_character=False)
    valid=complete(execution.validate_prompt('full-only-export',region_prompt,None))
    assert valid[0],valid
    saved=region_cls.execute([json.dumps(fixture)],include_regions=[False],include_character=[False],
        **{key:[value] for key,value in settings.items()}).result
    [image]=json.loads(Path(saved[0]).read_text(encoding='utf-8'))['images']
    assert image['full']=={'depth':0,'general':['1girl']} and 'regions' not in image
    print('PASS: compact regional JSON validates and keeps both source tag sets separate')
    from examples.build_comfy_workflows import build
    build(root)
    print('PASS: tags-only workflow validates and saves exact TXT without results_json')

if __name__=='__main__':main()
