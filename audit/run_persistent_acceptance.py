"""Real installed ComfyUI schemas/node API and local WD/YOLO persistent acceptance."""
import argparse
from contextlib import ExitStack
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(args.comfy_root.resolve()),str(root)]
    sys.argv=['persistent-acceptance','--cpu','--base-directory',str(root/'temp/persistent-comfy')]
    import comfy.options
    comfy.options.enable_args_parsing()
    spec=importlib.util.spec_from_file_location('wdv3_persistent_acceptance',root/'__init__.py',submodule_search_locations=[str(root)])
    package=importlib.util.module_from_spec(spec);sys.modules[spec.name]=package;spec.loader.exec_module(package)
    ui=importlib.import_module(spec.name+'.comfy_persistent_nodes')
    old_ui=importlib.import_module(spec.name+'.comfy_ui_nodes')
    post=importlib.import_module(spec.name+'.comfy_postprocessing_nodes')
    pipe=importlib.import_module(spec.name+'.comfy_pipeline_nodes')
    api=importlib.import_module(spec.name+'.Scripts.workflow_api')
    storage=importlib.import_module(spec.name+'.Scripts.artifact_store')
    captions=importlib.import_module(spec.name+'.Scripts.caption_files')
    runtime=importlib.import_module(spec.name+'.comfy_persistent_runtime')
    import torch
    from PIL import Image
    work=root/'temp/persistent-acceptance';work.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as name:
        dataset=Path(name)
        with Image.new('RGB',(8,8),'white') as image:image.save(dataset/'one.png')
        (dataset/'one.txt').write_bytes(b'blue eyes, watermark')
        store_cfg=ui.WDCalculationStore.execute(str(dataset)).result[0]
        from audit.test_comfy_nodes import runtime as recording_runtime
        cache_options=old_ui.WDTaggingOptions.execute(cache_mode='reuse',cache_dataset_root=str(dataset)).result[0]
        legacy_first=recording_runtime()
        collected=pipe.WDCollectEvidence.execute(torch.ones(1,6,8,3),legacy_first,options=cache_options).result[0]
        assert len(legacy_first.model.batches)==1
        legacy_second=recording_runtime(True)
        restored_raw=pipe.WDCollectEvidence.execute(torch.ones(1,6,8,3),legacy_second,
            options=dict(cache_options,general_threshold=1)).result[0]
        assert legacy_second.model.batches==[]
        assert restored_raw.data()['frames'][0]['full_image']==collected.data()['frames'][0]['full_image']
        assert pipe.WDCollectEvidence.INPUT_TYPES()['optional']['sources'][1]['display_name']=='sam_queries'
        loaded=ui.WDLoadCaptions.execute(store_cfg).result[0]
        edited=ui.WDEditCaptions.execute(loaded,'[{"remove":["watermark"],"add_after":["dataset_tag"]}]').result[0]
        export=ui.WDCaptionExportSettings.execute(write_mode='overwrite').result[0]
        preview=ui.WDPlanCaptionExport.execute(store_cfg,edited,export,'txt-edit').result[0]
        assert (dataset/'one.txt').read_bytes()==b'blue eyes, watermark'
        saved=ui.WDSaveCaptions.execute(preview)
        assert saved.ui['text'] and (dataset/'one.txt').read_text()=='blue eyes, dataset tag'
        # Presets retain new settings without moving the first five output indices.
        with patch.object(old_ui,'preset_directory',return_value=dataset/'presets'):
            old_ui.WDSaveConfig.execute('dataset',True,calculation_store=store_cfg,caption_export=export,
                                        caption_edit=[{'add_after':['extra']}],caption_read={'input_format':'comma_newline'})
            restored=old_ui.WDLoadConfig.execute('Saved: dataset').result
        assert len(restored)==10 and restored[5]==store_cfg and restored[9]==export
        # Real schema input typing and preflight before executing a prompt.
        import nodes,execution
        nodes.NODE_CLASS_MAPPINGS.update(package.NODE_CLASS_MAPPINGS)
        from audit.run_tag_redundancy_acceptance import complete
        prompt={
          'store':{'class_type':'WDV3CalculationStore','inputs':dict(dataset_root=str(dataset),store_path='',mode='reuse',refresh='')},
          'read':{'class_type':'WDV3LoadCaptions','inputs':dict(calculation_store=['store',0],input_format='comma_newline',input_separator=', ',escaped_parentheses=False,recursive=True)},
          'cfg':{'class_type':'WDV3CaptionExportSettings','inputs':dict(write_mode='overwrite',naming='stem_txt',in_place=True,target_root='',mapping_json='{}',input_format='comma_newline',input_separator=', ',escaped_parentheses=False)},
          'plan':{'class_type':'WDV3PlanCaptionExport','inputs':dict(calculation_store=['store',0],captions=['read',0],caption_export=['cfg',0],operation_id='prompt')},
          'save':{'class_type':'WDV3SaveCaptions','inputs':dict(caption_plan=['plan',0])}}
        assert complete(execution.validate_prompt('persistent',prompt,None))[0]
        # IMAGE path has no dependency on an eager model object.
        inference=dict(schema_version=1,wd={'identity':{'weights_sha256':'fixture'}},sources=[])
        value=ui.WDPlanCalculations.execute(store_cfg,inference,images=torch.ones(1,8,8,3),image_mapping_json='["one.png"]').result[0]
        backend=runtime.PersistentCalculator
        hooks={'wd':lambda images:[dict(general={'blue_eyes':.9,'skirt':.3},character={},ratings={'safe':.8}) for _ in images]}
        with patch.object(runtime,'PersistentCalculator',side_effect=lambda *a,**kw:backend(*a,hooks=hooks,**kw)):
            computed=ui.WDComputeMissing.execute(value).result
        refs=json.loads(computed[2]);raw=ui.WDLoadEvidence.execute(store_cfg,json.dumps(refs)).result[0]
        local=post.WDFilterTagSets.execute(raw).result[0]
        merged=post.WDMergeTagSets.execute(local).result[0]
        docs=ui.WDCombineCaptions.execute(merged).result[0]
        assert docs[0]['relative_path']=='one.png'
        # Direct API and UI preview/write produce identical bytes on fresh outputs.
        settings=ui.WDCaptionExportSettings.execute(write_mode='overwrite').result[0]
        plan_ui=ui.WDPlanCaptionExport.execute(store_cfg,docs,settings,'predictions').result[0]
        ui.WDSaveCaptions.execute(plan_ui)
        expected=(dataset/'one.txt').read_bytes()
        request=dict(schema_version=1,dataset_root=str(dataset),artifact_id=refs[0],options={})
        candidate=api.refilter(request)
        assert candidate['tags'][0].encode()==expected
        mode_results={}
        for mode in ('error','skip','append','overwrite'):
            target=dataset/'mode.txt';target.write_bytes(b'blue eyes, smile')
            cli_root=dataset/'cli';cli_root.mkdir(exist_ok=True)
            (cli_root/'mode.txt').write_bytes(b'blue eyes, smile')
            entries=[dict(relative_path='mode.png',document=captions.document_from_tags(['blue_eyes','long_hair']))]
            cfg=ui.WDCaptionExportSettings.execute(write_mode=mode,options={'tag_separator':' | '}).result[0]
            planned=ui.WDPlanCaptionExport.execute(store_cfg,entries,cfg,'mode-'+mode).result[0]
            outcome=json.loads(ui.WDSaveCaptions.execute(planned).result[0])
            cli_cfg=dict(cfg,in_place=False,target_root=str(cli_root))
            cli_plan=captions.plan_caption_export(storage.ArtifactStore(dataset),entries,operation_id='cli-'+mode,config=cli_cfg)
            request_file=dataset/'cli-request.json';request_file.write_text(json.dumps(dict(schema_version=1,dataset_root=str(dataset),plan=cli_plan)))
            reply=subprocess.run([sys.executable,'-m','Scripts.workflow_cli','caption-commit','--request',str(request_file)],
                                 cwd=root,capture_output=True,text=True,timeout=30)
            assert reply.returncode==(2 if mode=='error' else 0),reply.stderr+reply.stdout
            assert target.read_bytes()==(cli_root/'mode.txt').read_bytes()
            mode_results[mode]=outcome['files'][0]['status']
        # Resume and conflict use the identical UI entry point, no browser callbacks.
        entries=[dict(relative_path='resume.png',document=captions.document_from_tags(['one','two']))]
        cfg=ui.WDCaptionExportSettings.execute(write_mode='append').result[0]
        preview=ui.WDPlanCaptionExport.execute(store_cfg,entries,cfg,'resume').result[0]
        def fail(point,path):
            if point=='after_replace':raise OSError('injected crash after replace')
        interrupted=captions.commit_caption_export(storage.ArtifactStore(dataset),preview['plan'],fault=fail)
        assert interrupted['files'][0]['status']=='failed'
        resumed=json.loads(ui.WDSaveCaptions.execute(preview).result[0])
        assert resumed['files'][0]['status']=='unchanged'
        repeated=json.loads(ui.WDSaveCaptions.execute(preview).result[0])
        assert repeated['files'][0]['status']=='unchanged'
        (dataset/'resume.txt').write_bytes(b'manual edit')
        assert json.loads(ui.WDSaveCaptions.execute(preview).result[0])['files'][0]['status']=='conflict'
        summary=dict(schema_version=1,nodes=len(package.NODE_CLASS_MAPPINGS),real_comfy_schemas=True,
                     valid_prompt=True,txt_only=True,image_mapping=True,preset_roundtrip=True,
                     evidence_roundtrip=True,api_ui_bytes=True,browser_tested=False)
        summary.update(caption_modes=mode_results,custom_separator=True,append_resume=True,external_edit_conflict=True)
        summary.update(existing_collect_automatic_cache=True,visible_sam_port_labels=True)
    from examples.build_comfy_workflows import build
    build(root)
    definitions={name:dict(name=name,display_name=package.NODE_DISPLAY_NAME_MAPPINGS[name],description=cls.DESCRIPTION,
                  input=cls.INPUT_TYPES(),output=cls.RETURN_TYPES,output_name=cls.RETURN_NAMES) for name,cls in package.NODE_CLASS_MAPPINGS.items()}
    (root/'audit/fixtures/connection_nodes.json').write_text(json.dumps(definitions,ensure_ascii=False,indent=2),encoding='utf-8')
    for path in (root/'examples').glob('comfyui_wdv3*.json'):
        graph=json.loads(path.read_text(encoding='utf-8'));entries={n['id']:n for n in graph['nodes']}
        assert len(entries)==len(graph['nodes'])
        for _,source,index,target,slot,kind in graph['links']:
            assert entries[source]['outputs'][index]['type']==kind
            assert entries[target]['inputs'][slot]['type'] in (kind,'*')
    dest=root/'docs/verification/persistent-evidence';dest.mkdir(parents=True,exist_ok=True)
    (dest/'comfyui.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: persistent ComfyUI schemas/API, caption preview/commit, presets, IMAGE and example links')


if __name__=='__main__':main()
