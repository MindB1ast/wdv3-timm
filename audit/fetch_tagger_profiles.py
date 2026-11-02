"""Snapshot published preprocessing at catalog revisions, without downloading weights."""
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


def main():
    root = Path(__file__).resolve().parents[1]
    entries = json.loads((root / 'Scripts/tagger_models.json').read_text(encoding='utf-8'))
    access = []
    for entry in entries:
        if entry['status'] != 'metadata-compatible':
            continue
        target = root / 'audit/fixtures/tagger-profiles' / entry['repo_id']
        target.mkdir(parents=True, exist_ok=True)
        for name in ('config.json', 'preprocess.json'):
            if name not in entry['files']:
                continue
            try:
                source = hf_hub_download(entry['repo_id'], name, revision=entry['revision'])
            except HfHubHTTPError as error:
                access.append({'repo_id': entry['repo_id'], 'file': name,
                               'status': error.response.status_code if error.response is not None else None,
                               'error': type(error).__name__})
                print(entry['repo_id'], name, type(error).__name__, flush=True)
                continue
            data = json.loads(Path(source).read_text(encoding='utf-8'))
            if name == 'config.json':
                # Tag order is checked separately against the model's actual CSV.
                data.pop('tags', None)
            (target / name).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        (target / 'source.json').write_text(json.dumps({
            'repo_id': entry['repo_id'], 'revision': entry['revision'],
            'config_omits': ['tags'], 'source': entry['source']}, indent=2), encoding='utf-8')
        print(entry['repo_id'], flush=True)
    (root / 'audit/fixtures/tagger-profiles/access.json').write_text(
        json.dumps(access, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
