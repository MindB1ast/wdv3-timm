"""Bounded mutation verification on a private source copy."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else 'docs/verification/fix-audited-tagging-defects')
OUTPUT.mkdir(parents=True, exist_ok=True)
SNAPSHOT = ROOT / 'temp' / ('fix-mutants-' + uuid.uuid4().hex[:8])
SNAPSHOT.mkdir(parents=True)
for directory in ('Scripts', 'audit'):
    (SNAPSHOT / directory).mkdir()
    for source in (ROOT / directory).glob('*.py'):
        shutil.copy2(source, SNAPSHOT / directory / source.name)
shutil.copy2(ROOT / 'detectors.json', SNAPSHOT / 'detectors.json')
if (ROOT / 'audit/fixtures').exists():
    shutil.copytree(ROOT / 'audit/fixtures', SNAPSHOT / 'audit/fixtures')
checker = """import importlib, json, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
module = importlib.import_module(sys.argv[1])
actual = Path(module.__file__).resolve()
expected = (Path.cwd() / (sys.argv[1].replace('.', '/') + '.py')).resolve()
assert actual == expected, (actual, expected)
with open('import-proof.jsonl', 'a', encoding='utf-8') as stream:
    stream.write(json.dumps({'cwd': str(Path.cwd()), 'module': str(actual)}) + '\\n')
suite = unittest.defaultTestLoader.discover(str(Path.cwd() / 'audit'), 'test*.py')
result = unittest.TextTestRunner(verbosity=1).run(suite)
assert result.testsRun >= 49, result.testsRun
raise SystemExit(0 if result.wasSuccessful() else 1)
"""
# The checker source needs one newline escape, not a literal backslash-n record delimiter.
checker = checker.replace("'\\\\n'", "'\\n'")
(SNAPSHOT / 'check_imports.py').write_text(checker, encoding='utf-8')
environment = os.environ.copy()
environment.update(PYTHONIOENCODING='utf-8', MPLBACKEND='Agg', HF_HUB_OFFLINE='1',
                   MPLCONFIGDIR=str(ROOT / 'temp/matplotlib-cache'),
                   YOLO_CONFIG_DIR=str(ROOT / 'temp/yolo-settings'),
                   PYTHONDONTWRITEBYTECODE='1')
configuration = (
    "from pathlib import Path; import tree_sitter_language_pack as p; "
    f"p.configure(p.PackConfig(cache_dir={str(ROOT / 'temp/parser-cache')!r})); "
    "from mutator.cli import main; main()"
)
reports = []
for filename, needle in [('preprocess.py', 'if regions and name in configs:'),
                         ('savingProc.py', 'append_tags and txt_path.exists()')]:
    path = SNAPSHOT / 'Scripts' / filename
    line = next(index for index, text in enumerate(path.read_text(encoding='utf-8').splitlines(), 1)
                if needle in text)
    module = 'Scripts.' + path.stem
    command = f'"{sys.executable}" check_imports.py {module}'
    baseline = subprocess.run([sys.executable, 'check_imports.py', module], cwd=SNAPSHOT,
                              env=environment, capture_output=True, text=True, timeout=90)
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    arguments = [sys.executable, '-c', configuration, '--root', str(SNAPSHOT),
                 '--no-coverage', '--mutate-all', '--max-workers', '1', '--lines', str(line),
                 '--test-command', command, 'Scripts/' + filename]
    try:
        run = subprocess.run(arguments, cwd=ROOT, env=environment, capture_output=True,
                             text=True, timeout=180)
        text = run.stdout + run.stderr
        code = run.returncode
    except subprocess.TimeoutExpired as exc:
        text, code = str(exc), 124
    (OUTPUT / ('mutator-' + path.stem + '.log')).write_text(text, encoding='utf-8')
    record = {'file': filename, 'line': line, 'tool_exit': code, 'snapshot': str(SNAPSHOT)}
    if code not in (0, 3):
        # Windows symlink overlays may be unavailable. Use the documented isolated-copy fallback.
        original = path.read_text(encoding='utf-8')
        if filename == 'preprocess.py':
            mutated = original.replace('if regions and name in configs:', 'if name in configs:', 1)
        else:
            mutated = original.replace('from .io_utils import atomic_write_text',
                "def atomic_write_text(path, text):\n    Path(path).write_text(text, encoding='utf-8')", 1)
        assert mutated != original
        try:
            path.write_text(mutated, encoding='utf-8')
            result = subprocess.run([sys.executable, 'check_imports.py', module], cwd=SNAPSHOT,
                                    env=environment, capture_output=True, text=True, timeout=90)
            log = result.stdout + result.stderr
            expected = ('FAIL: test_empty_detector_must_not_remove_full_tags'
                        if filename == 'preprocess.py' else
                        'FAIL: test_replace_failure_preserves_original_and_cleans_temp')
            record.update(method='controlled-copy-fallback', detected=result.returncode != 0 and expected in log,
                          test_exit=result.returncode)
            (OUTPUT / ('controlled-' + path.stem + '.log')).write_text(log, encoding='utf-8')
        finally:
            path.write_text(original, encoding='utf-8')
    else:
        record.update(method='mutator', detected=code == 0)
    reports.append(record)
    print(json.dumps(record), flush=True)
    assert record['detected'], record
shutil.copy2(SNAPSHOT / 'import-proof.jsonl', OUTPUT / 'mutation-import-proof.jsonl')
(OUTPUT / 'mutations.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
print('Mutation verification finished', flush=True)
