"""Build a separate ComfyUI demonstration from the pinned Region Tag Rules graph."""
import argparse
from copy import deepcopy
import json
from pathlib import PureWindowsPath
from uuid import NAMESPACE_URL, uuid5

if __package__:
    from .run_region_filter_outcomes import FIXTURES, HISTORICAL, scenarios
else:
    from run_region_filter_outcomes import FIXTURES, HISTORICAL, scenarios


def build_workflow():
    original = json.loads((HISTORICAL / 'workflow.json').read_bytes())
    templates = {node['id']: node for node in original['nodes']}
    graph = {**original, 'id': str(uuid5(NAMESPACE_URL, 'wdv3-timm/region-filter-outcomes')),
             'nodes': [], 'links': [], 'groups': [], 'extra': {}}
    nodes, links = graph['nodes'], graph['links']
    output_dir = str(PureWindowsPath(
        templates[103]['widgets_values_named']['output_directory']).with_name('filter_outcomes_demo'))

    def clone(template, x, y, title=None, **widgets):
        node = deepcopy(templates[template])
        node.update(id=len(nodes) + 1, pos=[x, y], order=len(nodes))
        for slot in node.get('inputs', []):
            slot['link'] = None
        for slot in node.get('outputs', []):
            slot['links'] = []
        if title:
            node['title'] = title
        if widgets:
            named = node['widgets_values_named']
            if set(widgets) - named.keys():
                raise ValueError(f'Unknown widgets for {node["type"]}: {set(widgets) - named.keys()}')
            named.update(widgets)
            node['widgets_values'] = list(named.values())
        nodes.append(node)
        return node

    def wire(source, output, target, port):
        source_index, outgoing = next((i, s) for i, s in enumerate(source['outputs'])
                                      if s['name'] == output)
        target_index, incoming = next((i, s) for i, s in enumerate(target['inputs'])
                                      if s['name'] == port)
        if outgoing['type'] != incoming['type'] or incoming['link'] is not None:
            raise ValueError(f'Cannot wire {source["id"]}:{output} to {target["id"]}:{port}')
        number = len(links) + 1
        outgoing['links'].append(number)
        incoming['link'] = number
        links.append([number, source['id'], source_index, target['id'], target_index, outgoing['type']])

    def note(x, y, title, text):
        node = clone(2000, x, y, title)
        node['widgets_values'] = [text]
        node['widgets_values_named'] = {'text': text}
        return node

    model = clone(1, 0, 340, download_missing=False)
    image = clone(3, 0, 550)
    options = clone(4, 0, 900)
    note(0, 0, 'WDV3 Region Tag Rules — Filter Outcomes',
         'Сравните full, regions и caption. В каждой ветке своё имя JSON. '
         f'Результаты новых прогонов: {output_dir}. '
         'Первый блок использует один YOLO-источник; второй — два параллельных. '
         'Точное равенство порогу проверяется отдельно на искусственных оценках.')

    def detector(x, y, title, source, overrides, previous=None):
        rules = clone(100, x, y, f'{title} — rules {source}', **overrides)
        yolo = clone(101, x + 440, y, f'{title} — YOLO {source}',
                     source_name=source, download_missing=False)
        wire(rules, 'rules', yolo, 'rules')
        if previous:
            wire(previous, 'detectors', yolo, 'previous_detectors')
        return yolo

    evidence = {}
    for suite, sources, y in (('single', ('person',), 0), ('dual', ('person_A', 'person_B'), 2300)):
        source = None
        for index, name in enumerate(sources):
            source = detector(750, y + index * 1070, 'Сбор evidence', name, {}, source)
        collect = clone(2, 1500, y + 160, f'Общий evidence — {suite}')
        for upstream, output, port in ((source, 'detectors', 'detectors'), (image, 'IMAGE', 'images'),
                                       (model, 'wd_model', 'wd_model'), (options, 'options', 'options')):
            wire(upstream, output, collect, port)
        evidence[suite] = collect, sources

    counts = {'single': 0, 'dual': 0}
    for suite, case in scenarios():
        if suite not in evidence:
            continue
        collect, names = evidence[suite]
        title = case['name']
        overrides = case.get('sources', {'person': case.get('rules', {})})
        x = 1950 + (counts[suite] % 3) * 1580
        pitch = 2440 if suite == 'dual' else 1410
        y = (10400 if suite == 'dual' else 0) + (counts[suite] // 3) * pitch
        counts[suite] += 1
        settings = json.dumps(overrides, ensure_ascii=False)
        note(x, y, title, f'{case["note"]}\n\nНастройки: {settings}')
        source = None
        for index, name in enumerate(names):
            source = detector(x, y + 260 + index * 1070, title, name, overrides.get(name, {}), source)
        apply = clone(102, x + 760, y + 260, f'{title} — Apply Filters')
        saver = clone(103, x + 1000, y + 260, f'{title} — Save Compact JSON',
                      filename=title, output_directory=output_dir)
        wire(collect, 'evidence', apply, 'evidence')
        wire(options, 'options', apply, 'options')
        wire(source, 'detectors', apply, 'detectors')
        wire(apply, 'results_json', saver, 'results_json')
    graph.update(last_node_id=len(nodes), last_link_id=len(links))
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check the checked-in graph without writing')
    args = parser.parse_args()
    target = FIXTURES / 'workflow.json'
    content = json.dumps(build_workflow(), ensure_ascii=False, indent=2) + '\n'
    if args.check:
        if target.read_text(encoding='utf-8') != content:
            parser.exit(1, 'Workflow does not match the manifest and graph template.\n')
        print(f'Workflow matches: {target}')
    else:
        target.write_text(content, encoding='utf-8')
        print(f'Wrote {target}')


if __name__ == '__main__':
    main()
