"""Standalone JSONL process. Run by filename to avoid importing legacy Scripts."""
import os
import sys

# Scripts/types.py must never shadow the standard library's types module when
# this file is launched directly. Keep our helpers after the standard library.
_directory = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry) != _directory]
sys.path.append(_directory)

from contextlib import redirect_stdout
import json

PROTOCOL_VERSION = 1


def main():
    backend = None
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if request["protocol_version"] != PROTOCOL_VERSION:
                raise ValueError("Unsupported SAM protocol version")
            action = request["action"]
            with redirect_stdout(sys.stderr):
                if action == "init":
                    if backend is not None:
                        raise ValueError("SAM worker already initialized")
                    from sam_runtime import create_backend
                    backend = create_backend(request["model"])
                    payload = {"provenance": backend.provenance}
                elif action == "segment" and backend is not None:
                    payload = {"records": backend.segment(request["images"], request["sources"], request["directory"]),
                               "metrics": backend.last_metrics}
                elif action == "offload" and backend is not None:
                    backend.offload()
                    payload = {}
                elif action == "load" and backend is not None:
                    backend.load()
                    payload = {}
                else:
                    raise ValueError("Invalid SAM action or uninitialized worker")
            response = {"status": "ok", **payload}
        except Exception as error:  # Process boundary: report failure, never a fake empty result.
            response = {"status": "error", "error": f"{type(error).__name__}: {error}"}
        response.update(protocol_version=PROTOCOL_VERSION, request_id=request.get("request_id"))
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
