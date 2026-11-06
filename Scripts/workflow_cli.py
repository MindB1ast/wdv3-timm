"""python -m Scripts.workflow_cli <command> --request request.json"""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','execute','status','load','import','refilter','caption-plan','caption-commit'])
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args()
    try:
        from .workflow_api import dispatch
        with redirect_stdout(sys.stderr):
            result=dispatch(args.command,json.loads(args.request.read_text(encoding='utf-8-sig')))
        print(json.dumps(dict(schema_version=1,ok=True,result=result),ensure_ascii=False,allow_nan=False))
        if result.get('complete') is False or result.get('ready') is False:
            return 2
        if any(r.get('status') in ('conflict','failed') for r in result.get('files',[])):
            return 2
        return 0
    except Exception as exc:
        print(json.dumps(dict(schema_version=1,ok=False,error=dict(code=type(exc).__name__,message=str(exc))),ensure_ascii=False))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
