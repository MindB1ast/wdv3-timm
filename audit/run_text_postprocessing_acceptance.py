"""Run the shipped text-only example against a local ComfyUI HTTP API."""
import argparse
import json
from pathlib import Path

from audit.run_comfy_cpu import result_from_history, wait_for
from audit.run_saved_workflow_demo import request

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/comfyui_wdv3_text_postprocessing.json'
ALLOWED = {'WDV3TagSetsFromText', 'WDV3RemoveRedundantTags',
           'WDV3ConditionalTagRules', 'WDV3TagSetsToText'}


def run(server):
    graph = json.loads(EXAMPLE.read_text(encoding='utf8'))
    catalog = request(server, '/object_info')
    links = {link[0]: link for link in graph['links']}
    prompt = {}
    for node in graph['nodes']:
        kind = node['type']
        if kind not in ALLOWED:
            raise ValueError(f'Text example contains an unexpected node: {kind}')
        schema = catalog[kind]['input']
        fields = {**schema.get('required', {}), **schema.get('optional', {})}
        sockets = {slot['name']: slot['link'] for slot in node['inputs']}
        values = dict(zip((name for name in fields if name not in sockets),
                          node['widgets_values'], strict=True))
        for name, link in sockets.items():
            if link is not None:
                values[name] = [str(links[link][1]), links[link][2]]
        prompt[str(node['id'])] = {'class_type': kind, 'inputs': values}
    ticket = request(server, '/prompt', {'prompt': prompt})
    if ticket.get('node_errors'):
        raise ValueError(ticket['node_errors'])
    prompt_id = ticket['prompt_id']
    result = wait_for(lambda _: result_from_history(request(server, f'/history/{prompt_id}'), prompt_id), 30)
    outputs = {key: result['outputs'][key]['text'] for key in ('3', '6')}
    if outputs != {'3': ['purple jacket'], '6': ['apron']}:
        raise AssertionError(outputs)
    print(f'PASS {prompt_id}: full={outputs["3"][0]!r}, region={outputs["6"][0]!r}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True)
    run(parser.parse_args().server)
