"""Run positive contracts and verify exact intentional negative diagnostics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', default='temp/typecheck')
    options = parser.parse_args()
    directory = ROOT / options.report_dir
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'pyright', '--pythonpath', sys.executable,
               '--pythonversion', f'{sys.version_info.major}.{sys.version_info.minor}', '--outputjson']
    negative_config = ROOT / 'temp/typed-negative.json'
    negative_config.parent.mkdir(parents=True, exist_ok=True)
    negative_config.write_text(json.dumps({'extends': '../pyrightconfig.json',
        'include': ['../audit/typecheck/negative.py'], 'exclude': [], 'strict': [], 'extraPaths': ['..']}), encoding='utf-8')
    summary = {}
    for name, arguments in [('positive', []), ('negative', ['--project', str(negative_config)])]:
        result = subprocess.run(command + arguments, cwd=ROOT, capture_output=True,
                                text=True, encoding='utf-8')
        assert result.stdout.strip().startswith('{'), (result.returncode, result.stderr)
        data = json.loads(result.stdout)
        (directory / (name + '.json')).write_text(result.stdout, encoding='utf-8')
        diagnostics = data['generalDiagnostics']
        if name == 'positive':
            assert result.returncode == 0, diagnostics
            assert data['summary']['errorCount'] == 0
        else:
            expected = {(6, 'reportGeneralTypeIssues'), (7, 'reportArgumentType'),
                        (8, 'reportTypedDictNotRequiredAccess')}
            actual = {(item['range']['start']['line'] + 1, item.get('rule'))
                      for item in diagnostics}
            assert result.returncode == 1, (result.returncode, result.stderr)
            assert len(diagnostics) == 3 and actual == expected, diagnostics
            assert all(Path(item['file']).name == 'negative.py' for item in diagnostics)
        summary[name] = data['summary']
    summary['interpreter'] = sys.executable
    summary['python'] = sys.version
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
