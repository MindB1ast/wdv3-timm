import json, os, pathlib, shutil, subprocess, sys, uuid
root=pathlib.Path.cwd();out=root/'docs/verification/typed-contracts'
base=root/'temp'/('typed-mutations-'+uuid.uuid4().hex[:8]);base.mkdir()
source=(root/'Scripts/evaluation.py').read_text(encoding='utf-8')
cases=[('baseline',source,None),
       ('epsilon-zero',source.replace('METRIC_EPSILON = 1e-8','METRIC_EPSILON = 0.0'),'test_one_correct_tag_uses_epsilon_in_all_denominators'),
       ('empty-one',source.replace('    precision = tp / (tp + fp + METRIC_EPSILON)',
            '    if tp == fp == fn == 0:\n        return {"TP": 0, "FP": 0, "FN": 0, "precision": 1.0, "recall": 1.0, "f1_score": 1.0}\n    precision = tp / (tp + fp + METRIC_EPSILON)'), 'test_tp_fp_fn_and_empty_agree_across_image_tables'),
       ('mean-sample-f1',source.replace('    f1 = harmonic_f1(precision, recall)','    f1 = sum(sample["f1_score"] for sample in samples) / len(truth)'), 'test_summary_keeps_historical_order_of_aggregation')]
records=[]
driver='''import json, pathlib, sys, unittest
root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from Scripts import evaluation
assert pathlib.Path(evaluation.__file__).resolve()==root/'Scripts/evaluation.py'
suite=unittest.defaultTestLoader.discover(str(root/'audit'),'test_typed_contracts.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
record={'module':evaluation.__file__,'run':result.testsRun,'failures':[{'test':t.id(),'traceback':text} for t,text in result.failures],'errors':[{'test':t.id(),'traceback':text} for t,text in result.errors],'skipped':len(result.skipped)}
(root/'result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
raise SystemExit(0 if result.wasSuccessful() else 1)
'''
for name,code,expected in cases:
    assert name=='baseline' or code!=source
    folder=base/name;shutil.copytree(root/'Scripts',folder/'Scripts',ignore=shutil.ignore_patterns('__pycache__'))
    (folder/'Scripts/evaluation.py').write_text(code,encoding='utf-8')
    (folder/'audit/typecheck').mkdir(parents=True)
    shutil.copy2(root/'audit/test_typed_contracts.py',folder/'audit/test_typed_contracts.py')
    shutil.copy2(root/'audit/typecheck/positive.py',folder/'audit/typecheck/positive.py')
    (folder/'driver.py').write_text(driver,encoding='utf-8')
    run=subprocess.run([sys.executable,str(folder/'driver.py')],cwd=folder,capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8','MPLBACKEND':'Agg','YOLO_CONFIG_DIR':str(root/'temp/typed-yolo'),'MPLCONFIGDIR':str(root/'temp/typed-mpl')})
    (out/('mutation-'+name+'.log')).write_text(run.stdout+run.stderr,encoding='utf-8')
    assert (folder/'result.json').exists(),run.stderr
    record=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    record.update(case=name,exit=run.returncode,expected_failure=expected)
    if expected:
        assert run.returncode==1
        assert any(expected in failure['test'] and 'AssertionError' in failure['traceback'] for failure in record['failures']),record
        record['detected_by_behavior_assertion']=True
    else:assert run.returncode==0 and not record['errors'] and not record['failures']
    records.append(record);print(name,'verified',flush=True)
(out/'mutations.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
