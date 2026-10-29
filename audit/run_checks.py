"""Run the read-only functional audit and emit a structured result."""
import argparse
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({'test': test.id(), 'status': 'passed'})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append({'test': test.id(), 'status': 'failed',
                             'reason': str(err[1])})

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append({'test': test.id(), 'status': 'error',
                             'reason': f'{err[0].__name__}: {err[1]}'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='temp/audit-test-results.json')
    parser.add_argument('--pattern', default='test*.py')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'audit'), args.pattern)
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    data = {'schema_version': 1, 'tests_run': result.testsRun,
            'passed': sum(t['status'] == 'passed' for t in result.records),
            'failures': len(result.failures), 'errors': len(result.errors),
            'skipped': len(result.skipped), 'tests': result.records}
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in data.items() if k != 'tests'}))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
