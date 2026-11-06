"""Render the real viewer in headless Chromium and record DOM layout assertions."""
import argparse
from html import unescape
from http.server import HTTPServer
import json
from pathlib import Path
import re
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit.viewer_layout_harness import Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', type=Path, required=True)
    args = parser.parse_args()
    work = ROOT/'temp/viewer-layout'
    work.mkdir(parents=True, exist_ok=True)
    server = HTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        result = subprocess.run([str(args.browser), '--headless', '--disable-gpu', '--no-first-run',
            '--no-default-browser-check', '--disable-extensions', '--disable-crash-reporter',
            '--disable-breakpad', f'--user-data-dir={work / "profile"}',
            '--dump-dom', '--virtual-time-budget=12000', '--window-size=1280,1000',
            f'http://127.0.0.1:{server.server_port}/?autorun=1'],
            capture_output=True, timeout=45, creationflags=subprocess.CREATE_NO_WINDOW)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
    output = result.stdout.decode('utf-8', errors='replace')
    (work/'page.html').write_text(output, encoding='utf-8')
    (work/'browser.log').write_bytes(result.stderr)
    found = re.search(r'<pre id="checks">(.*?)</pre>', output, re.S)
    if not found:
        raise AssertionError(f'No browser layout results; see {work}/browser.log (exit {result.returncode})')
    report = json.loads(unescape(found.group(1)))
    destination = ROOT/'docs/verification/cascaded-regions/viewer-layout.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    assert report['status'] == 'passed', report
    print(f'PASS: {len(report["checks"])} frame/resize/rerender cases in real Chromium; panel fills mount, canvas aspect preserved.')


if __name__ == '__main__':
    main()
