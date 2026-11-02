"""Regenerate the ComfyUI examples against the installed V3 API."""
import importlib.util
import json
from pathlib import Path
import sys


def build(root):
    spec = importlib.util.spec_from_file_location("wdv3_workflow_builder", root / "__init__.py",
                                                submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    classes = package.NODE_CLASS_MAPPINGS

    def node(number, kind, position, values=None):
        values = values or {}
        if kind == "LoadImage":
            inputs = {"required": {"image": (["tag_managing1.png"], {})}}
            output_types, output_names = ["IMAGE", "MASK"], ["IMAGE", "MASK"]
        elif kind == "PreviewAny":
            inputs = {"required": {"source": ("*", {})}}
            output_types, output_names = ["STRING"], ["STRING"]
        else:
            cls = classes[kind]
            inputs = cls.INPUT_TYPES()
            output_types, output_names = cls.RETURN_TYPES, cls.RETURN_NAMES
        sockets, widgets = [], []
        for section in ("required", "optional"):
            for name, data in inputs.get(section, {}).items():
                kind_or_choices = data[0]
                options = data[1] if len(data) > 1 else {}
                if kind_or_choices == "COMFY_AUTOGROW_V3":
                    template = options["template"]
                    nested = next(iter(template["input"]["required"].values()))[0]
                    for index in range(values.get("_count", 2)):
                        sockets.append({"name": f'{name}.{template["prefix"]}{index}', "type": nested, "link": None})
                    continue
                combo = isinstance(kind_or_choices, list) or kind_or_choices == "COMBO"
                choices = kind_or_choices if isinstance(kind_or_choices, list) else options.get("options", [])
                widget = combo or kind_or_choices in ("STRING", "INT", "FLOAT", "BOOLEAN")
                if widget and not options.get("forceInput"):
                    default = options.get("default")
                    if default is None:
                        default = choices[0] if combo else ""
                    widgets.append(values.get(name, default))
                else:
                    sockets.append({"name": name, "type": kind_or_choices, "link": None})
        if kind == "LoadImage":
            widgets.append("image")
        height = 430 if kind in ("WDV3Tagger", "WDV3Options", "WDV3RegionRules") else max(120, 70 + 32 * len(widgets))
        if kind == "WDV3RegionRules":
            height = 1000
        if kind == "WDV3Options":
            height = 640
        if kind == "LoadImage":
            height = 340
        return {"id": number, "type": kind, "pos": position,
                "size": [520, 760] if kind == "WDV3Preview" else [340, height],
                "flags": {}, "order": number - 1, "mode": 0, "inputs": sockets,
                "outputs": [{"name": name, "type": output, "links": None, "slot_index": slot}
                            for slot, (name, output) in enumerate(zip(output_names, output_types))],
                "properties": {"Node name for S&R": kind}, "widgets_values": widgets}

    def wire(graph, source, output_name, target, input_name):
        output = next(i for i, item in enumerate(source["outputs"]) if item["name"] == output_name)
        index = next(i for i, item in enumerate(target["inputs"]) if item["name"] == input_name)
        number = max((row[0] for row in graph["links"]), default=0) + 1
        graph["links"].append([number, source["id"], output, target["id"], index, source["outputs"][output]["type"]])
        source["outputs"][output]["links"] = (source["outputs"][output]["links"] or []) + [number]
        target["inputs"][index]["link"] = number

    def remove_nodes(graph, ids):
        graph["nodes"] = [n for n in graph["nodes"] if n["id"] not in ids]
        graph["links"] = [row for row in graph["links"] if row[1] not in ids and row[3] not in ids]
        active = {row[0] for row in graph["links"]}
        for n in graph["nodes"]:
            for slot in n["inputs"]:
                if slot["link"] not in active:
                    slot["link"] = None
            for slot in n["outputs"]:
                slot["links"] = [i for i in (slot["links"] or []) if i in active] or None

    def disconnect(graph, target, input_name):
        slot = next(item for item in target["inputs"] if item["name"] == input_name)
        old = slot["link"]
        graph["links"] = [row for row in graph["links"] if row[0] != old]
        for n in graph["nodes"]:
            for output in n["outputs"]:
                output["links"] = [i for i in (output["links"] or []) if i != old] or None
        slot["link"] = None

    def workflow(folder=False, sam=False, yolo=False):
        graph = {"nodes": [], "links": [], "groups": [], "config": {},
                 "extra": {"ds": {"scale": .65, "offset": [40, 40]}}, "version": .4}
        add = lambda kind, pos, values=None: graph["nodes"].append(node(len(graph["nodes"]) + 1, kind, pos, values)) or graph["nodes"][-1]
        def link(source, output_name, target, input_name):
            wire(graph, source, output_name, target, input_name)
        image = add("WDV3LoadFolder" if folder else "LoadImage", [20, 20],
                    {"limit": 0, "image": "tag_managing1.png", "directory": "input"})
        wd = add("WDV3ModelLoader", [20, 390])
        options = add("WDV3Options", [410, 20])
        tagger = add("WDV3TagFolder" if folder else "WDV3Tagger", [820, 20],
                     {"output_directory": "output/wdv3"} if folder else None)
        viewer = add("WDV3Preview", [1240, 20])
        link(wd, "wd_model", tagger, "wd_model")
        link(options, "options", tagger, "options")
        link(image, "files" if folder else "IMAGE", tagger, "files" if folder else "images")
        link(tagger, "results_json", viewer, "results_json")
        if not folder:
            link(image, "IMAGE", viewer, "images")
        if yolo or sam:
            rules = add("WDV3RegionRules", [410, 720])
            if yolo:
                detector = add("WDV3YOLODetector", [820, 530])
                link(rules, "rules", detector, "rules")
                link(detector, "detectors", tagger, "detectors")
            if sam:
                source = add("WDV3SAMSource", [20, 840])
                loader = add("WDV3SAMLoader", [20, 590])
                link(rules, "rules", source, "rules")
                link(loader, "sam_model", tagger, "sam_model")
                link(source, "sources", tagger, "sources")
                if not folder:
                    link(tagger, "sam_regions", viewer, "sam_regions")
        graph["last_node_id"] = len(graph["nodes"])
        graph["last_link_id"] = len(graph["links"])
        return graph

    def rule_values(rules):
        values = dict(general_threshold=rules.get("region_gen_threshold", .35),
                    character_threshold=rules.get("region_char_threshold", .75),
                    remove_from_full=", ".join(rules.get("remove_tags_from_full", [])),
                    remove_from_regions=", ".join(rules.get("remove_tags_from_region", [])),
                    add_to_regions=", ".join(tag if score == 1 else f'{tag}:{score}' for tag,score in rules.get("add_tags_to_region", {}).items()),
                    require_full_confirmation=", ".join(rules.get("exclude_from_region", [])),
                    only_from_source=", ".join(rules.get("only_from_source", [])),
                    allowed_tags=", ".join(rules.get("allowed_tags", [])))
        if rules.get('full_image_only_tags') is not None:
            values['full_image_only_tags']=', '.join(rules['full_image_only_tags'])
        return values

    def multi_workflow(sam):
        graph = workflow(sam=sam, yolo=not sam)
        # Replace the single-source branch with independently editable sources.
        remove_nodes(graph, {6, 7})
        by_id = {n["id"]: n for n in graph["nodes"]}
        by_id[4]["pos"] = [1240, 20]
        by_id[5]["pos"] = [1660, 20]
        if sam:
            by_id[8]["pos"] = [820, 20]
        preset = json.loads((root / "configs" / ("multi_sam.json" if sam else "multi_yolo.json")).read_text())
        rows = preset["sam_sources"] if sam else preset["detectors"]
        combine = node(10, "WDV3CombineSAMSources" if sam else "WDV3CombineDetectors", [820, 500], {"_count": len(rows)})
        graph["nodes"].append(combine)
        for index, row in enumerate(rows):
            x = 20 + 390 * index
            rule = node(11 + index * 2, "WDV3RegionRules", [x, 650], rule_values(row.get("rules", row)))
            if sam:
                values = {**row, "source_name": row["name"]}
                source = node(12 + index * 2, "WDV3SAMSource", [x, 1320], values)
                input_name, output_name = f"sources.source{index}", "sources"
            else:
                values = dict(model=row["model_path"], source_name=row["name"], confidence=row["confidence"],
                              classes=", ".join(str(value) for value in (row.get("classes") or [])))
                source = node(12 + index * 2, "WDV3YOLODetector", [x, 1320], values)
                input_name, output_name = f"detectors.detector{index}", "detectors"
            rule["title"] = row["name"] + " / tag rules"
            source["title"] = row["name"] + (" / SAM prompt and view" if sam else " / YOLO")
            graph["nodes"].extend([rule, source])
            wire(graph, rule, "rules", source, "rules")
            wire(graph, source, output_name, combine, input_name)
        wire(graph, combine, "sources" if sam else "detectors", by_id[4],
             "sources" if sam else "detectors")
        graph["last_node_id"] = max(n["id"] for n in graph["nodes"])
        graph["last_link_id"] = max(row[0] for row in graph["links"])
        return graph

    def config_workflow(save):
        graph = multi_workflow(sam=True) if save else workflow(sam=True)
        if save:
            remove_nodes(graph, {1, 2, 4, 5, 8, 9})
            by_id = {n["id"]: n for n in graph["nodes"]}
            target = node(20, "WDV3SaveConfig", [1240, 20], {"preset_name": "my_multi_sam"})
            preview = node(21, "PreviewAny", [1660, 20])
            model = node(22, "WDV3SAMLoader", [820, 20])
            graph["nodes"].extend([target, preview, model])
            wire(graph, by_id[3], "options", target, "options")
            wire(graph, by_id[10], "sources", target, "sources")
            wire(graph, model, "sam_model", target, "sam_model")
            wire(graph, target, "config_path", preview, "source")
        else:
            remove_nodes(graph, {3, 6, 7, 8})
            by_id = {n["id"]: n for n in graph["nodes"]}
            target = node(10, "WDV3LoadConfig", [410, 20], {"preset": "Example: multi_sam"})
            graph["nodes"].append(target)
            wire(graph, target, "options", by_id[4], "options")
            wire(graph, target, "sources", by_id[4], "sources")
            wire(graph, target, "sam_model", by_id[4], "sam_model")
        graph["last_node_id"] = max(n["id"] for n in graph["nodes"])
        graph["last_link_id"] = max(row[0] for row in graph["links"])
        return graph

    variants = {
        "comfyui_wdv3_pipeline.json": workflow(),
        "comfyui_wdv3_pipeline_yolo.json": workflow(yolo=True),
        "comfyui_wdv3_sam.json": workflow(sam=True),
        "comfyui_wdv3_folder.json": workflow(folder=True),
        "comfyui_wdv3_folder_sam.json": workflow(folder=True, sam=True),
        "comfyui_wdv3_multi_yolo.json": multi_workflow(sam=False),
        "comfyui_wdv3_multi_sam.json": multi_workflow(sam=True),
        "comfyui_wdv3_save_config.json": config_workflow(save=True),
        "comfyui_wdv3_from_config.json": config_workflow(save=False),
    }
    pipeline = workflow(yolo=True)
    by_id = {n['id']: n for n in pipeline['nodes']}
    remove_nodes(pipeline, {4, 5})
    collect_node = node(20, 'WDV3CollectEvidence', [820, 20])
    filtered = node(21, 'WDV3ApplyFilters', [1230, 20])
    inspect = node(22, 'WDV3Inspect', [1640, 20])
    export = node(23, 'WDV3ExportReport', [2400, 20], {'output_directory': 'output/wdv3'})
    face_rules = node(24, 'WDV3RegionRules', [20, 720], {'add_to_regions':'face', 'require_full_confirmation':'portrait, upper_body'})
    face_rules['title'] = 'Face / tag rules'
    face = node(25, 'WDV3YOLODetector', [820, 1000], {'model':'face_yolov9c.pt','source_name':'face'})
    pipeline['nodes'].extend([collect_node, filtered, inspect, export, face_rules, face])
    for src, output, dst, target in [(by_id[1], 'IMAGE', collect_node, 'images'),
        (by_id[2], 'wd_model', collect_node, 'wd_model'), (by_id[3], 'options', collect_node, 'options'),
        (by_id[7], 'detectors', face, 'previous_detectors'), (face_rules, 'rules', face, 'rules'),
        (face, 'detectors', collect_node, 'detectors'), (collect_node, 'evidence', filtered, 'evidence'),
        (by_id[3], 'options', filtered, 'options'), (face, 'detectors', filtered, 'detectors'),
        (filtered, 'results', inspect, 'results'), (inspect, 'report', export, 'report')]:
        wire(pipeline, src, output, dst, target)
    pipeline['last_node_id'] = 25
    pipeline['last_link_id'] = max(row[0] for row in pipeline['links'])
    variants['comfyui_wdv3_evidence.json'] = pipeline
    for name, graph in variants.items():
        path = root / "examples" / name
        path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1])
