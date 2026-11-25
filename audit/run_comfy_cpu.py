"""Run one real CPU regression against a disposable local ComfyUI API."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import psutil
import requests

from audit.comfy_cpu_cases import CASES, FIXTURES, MANIFEST, build_prompt, check_report

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare_models(scenario, cache):
    sources = json.loads((FIXTURES / 'models.json').read_text(encoding='utf8'))
    for name in ('wd', 'yolo', 'sam'):
        if name != 'wd' and not CASES[scenario][name]:
            continue
        source = sources[name]
        directory = cache / name
        directory.mkdir(parents=True, exist_ok=True)
        for filename, expected in source['sha256'].items():
            target = directory / filename
            if target.is_file() and sha256(target) == expected:
                continue
            print(f'Downloading {name}/{filename}', flush=True)
            temporary = target.with_suffix(target.suffix + '.part')
            try:
                subprocess.run([
                    'curl', '-q', '--fail', '--silent', '--show-error', '--location',
                    '--retry', '2', '--connect-timeout', '15', '--max-time', '600',
                    '--output', str(temporary), f"{source['base_url']}/{filename}",
                ], check=True, timeout=660)
                if sha256(temporary) != expected:
                    raise ValueError(f'SHA-256 mismatch: {name}/{filename}')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def wait_for(probe, timeout, before_probe=lambda: None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        before_probe()
        result = probe(max(.01, min(5, deadline - time.monotonic())))
        if result is not None:
            return result
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise TimeoutError(f'ComfyUI did not respond with the expected result within {timeout}s')


def result_from_history(history, prompt):
    result = history.get(prompt)
    if result is None:
        return None
    status = result['status']
    if status['status_str'] != 'success' or not status['completed']:
        raise RuntimeError(f'ComfyUI execution failed: {status.get("messages", status)}')
    return result


@contextlib.contextmanager
def server(comfy, work, port):
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'YOLO_CONFIG_DIR': str(work / 'yolo-settings'),
           'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'MPLCONFIGDIR': str(work / 'mpl'),
           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    with (work / 'comfy.log').open('w', encoding='utf8') as log:
        process = subprocess.Popen([
            sys.executable, '-u', str(comfy / 'main.py'), '--cpu', '--listen', '127.0.0.1',
            '--port', str(port), '--base-directory', str(work), '--disable-auto-launch',
            '--offline', '--disable-all-custom-nodes', '--whitelist-custom-nodes', 'wdv3-timm',
        ], cwd=comfy, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield process
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def run(args, metrics):
    input_path = FIXTURES / MANIFEST['image']
    if sha256(input_path) != MANIFEST['image_sha256']:
        raise ValueError('Control image SHA-256 mismatch')
    input_dir = args.work / 'input'
    input_dir.mkdir()
    shutil.copyfile(input_path, input_dir / input_path.name)
    custom_nodes = args.work / 'custom_nodes'
    custom_nodes.mkdir()
    (custom_nodes / 'wdv3-timm').symlink_to(ROOT, target_is_directory=True)

    metrics['stage'] = 'model_setup'
    start = time.monotonic()
    try:
        prepare_models(args.scenario, args.model_cache)
    finally:
        metrics['model_setup_seconds'] = round(time.monotonic() - start, 3)

    output = args.work / 'output'
    output.mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    with server(args.comfy_root, args.work, port) as process, requests.Session() as http:
        http.trust_env = False
        root_process = psutil.Process(process.pid)

        def sample():
            if process.poll() is not None:
                raise RuntimeError(f'ComfyUI exited with code {process.returncode}; see comfy.log')
            total = 0
            with contextlib.suppress(psutil.NoSuchProcess):
                for child in [root_process, *root_process.children(recursive=True)]:
                    with contextlib.suppress(psutil.NoSuchProcess):
                        total += child.memory_info().rss
            metrics['sampled_peak_process_tree_rss_mib'] = max(
                metrics.get('sampled_peak_process_tree_rss_mib', 0), round(total / (1024 ** 2), 1))

        def get(path, timeout):
            response = http.get(url + path, timeout=timeout)
            response.raise_for_status()
            return response.json()

        def catalog(timeout):
            try:
                return get('/object_info', timeout)
            except (requests.ConnectionError, requests.Timeout):
                return None

        metrics['stage'] = 'server_startup'
        start = time.monotonic()
        graph, export = build_prompt(args.scenario, wait_for(catalog, 120, sample),
                                     args.model_cache, output)
        metrics['startup_seconds'] = round(time.monotonic() - start, 3)
        (args.work / 'prompt.json').write_text(json.dumps(graph, indent=2), encoding='utf8')
        metrics['stage'] = 'prompt_execution'
        start = time.monotonic()
        response = http.post(url + '/prompt', json={'prompt': graph}, timeout=15)
        if not response.ok:
            raise RuntimeError(f'ComfyUI rejected prompt: {response.text}')
        submitted = response.json()
        if submitted.get('node_errors'):
            raise RuntimeError(f'ComfyUI rejected graph nodes: {submitted["node_errors"]}')
        prompt = metrics['prompt_id'] = submitted['prompt_id']
        try:
            result = wait_for(
                lambda timeout: result_from_history(get(f'/history/{prompt}', timeout), prompt),
                args.timeout or CASES[args.scenario]['timeout'], sample,
            )
        finally:
            metrics['execution_seconds'] = round(time.monotonic() - start, 3)
        sample()
        metrics['stage'] = 'check_report'
        report, = result['outputs'][export]['text']
        path = Path(report).resolve()
        if not path.is_relative_to(output):
            raise ValueError('Report is outside the run output directory')
        check_report(args.scenario, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario', choices=CASES)
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True, help='New, per-run diagnostics directory')
    parser.add_argument('--model-cache', type=Path, required=True, help='Private model download directory')
    parser.add_argument('--timeout', type=float, help='Override the inference timeout for local diagnostics')
    args = parser.parse_args()
    if args.timeout is not None and args.timeout <= 0:
        parser.error('--timeout must be positive')
    for name in ('comfy_root', 'work', 'model_cache'):
        setattr(args, name, getattr(args, name).resolve())
    if args.model_cache.is_relative_to(args.work):
        parser.error('--model-cache must be outside --work so artifacts never contain weights')
    args.work.mkdir(parents=True)
    metrics = {'scenario': args.scenario, 'stage': 'prepare', 'status': 'failed'}
    try:
        run(args, metrics)
        metrics.update(stage='complete', status='success')
    except Exception as error:
        metrics['error'] = str(error)
        raise
    finally:
        (args.work / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
        print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
