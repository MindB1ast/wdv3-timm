// Pure node-definition catalog. No canvas, model loading, or backend calls.
const widgetTypes = new Set(["STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"]);

export function normalizeNodeDefinition(definition, id = definition.name) {
    const inputs = [];
    for (const section of ["required", "optional"]) {
        const entries = definition.input?.[section] || {};
        const order = definition.input_order?.[section] || Object.keys(entries);
        for (const name of order) {
            const spec = entries[name];
            if (!Array.isArray(spec)) continue;
            const [type, options = {}] = spec;
            if (type === "COMFY_AUTOGROW_V3") {
                const template = options.template;
                for (const [innerName, inner] of Object.entries(template?.input?.required || {})) {
                    const prefix = `${name}.${template.prefix || innerName}`;
                    inputs.push({name: `${prefix}0`, type: inner[0], index: inputs.length,
                        optional: section === "optional", prefix});
                }
            } else if (typeof type === "string" && (!widgetTypes.has(type) || options.forceInput)) {
                inputs.push({name, type, index: inputs.length, optional: section === "optional"});
            }
        }
    }
    return {id, displayName: definition.display_name || id, inputs,
        outputs: (definition.output || []).map((type, index) => ({
            name: definition.output_name?.[index] || String(type), type, index,
        }))};
}

export function buildCatalog(definitions) {
    const entries = Array.isArray(definitions)
        ? definitions.map(def => [def.name, def]) : Object.entries(definitions);
    return new Map(entries.map(([id, def]) => [id, normalizeNodeDefinition(def, id)]));
}

export function compatibleTypes(a, b) {
    if (typeof a !== "string" || typeof b !== "string") return false;
    const left = a.toUpperCase().split(",").map(t => t.trim());
    const right = b.toUpperCase().split(",").map(t => t.trim());
    return left.includes("*") || right.includes("*") || left.some(t => right.includes(t));
}

export function catalogCoverage(catalog) {
    const types = new Map();
    for (const node of catalog.values()) {
        if (!node.id.startsWith("WDV3")) continue;
        for (const direction of ["inputs", "outputs"]) {
            for (const slot of node[direction]) {
                if (!slot.type?.startsWith("WDV3_")) continue;
                if (!types.has(slot.type)) types.set(slot.type, {type: slot.type, producers: [], consumers: []});
                types.get(slot.type)[direction === "inputs" ? "consumers" : "producers"]
                    .push(`${node.id}.${slot.name}`);
            }
        }
    }
    return [...types.values()].sort((a, b) => a.type.localeCompare(b.type));
}

const producerPriority = {
    WDV3_MODEL: ['WDV3ModelLoader'],
    WDV3_OPTIONS: ['WDV3Options', 'WDV3LoadConfig'],
    WDV3_RULES: ['WDV3RegionRules'],
    WDV3_DETECTORS: ['WDV3YOLODetector', 'WDV3CombineDetectors', 'WDV3LoadConfig', 'WDV3DetectorLoader'],
    WDV3_SAM_MODEL: ['WDV3SAMLoader', 'WDV3LoadConfig'],
    WDV3_SAM_SOURCES: ['WDV3SAMSource', 'WDV3CombineSAMSources', 'WDV3LoadConfig'],
    WDV3_SAM_REGIONS: ['WDV3SAMSegment', 'WDV3Tagger'],
    WDV3_FILES: ['WDV3LoadFolder'],
    WDV3_EVIDENCE: ['WDV3CollectEvidence'],
    WDV3_RESULTS: ['WDV3ApplyFilters', 'WDV3MergeResults'],
    WDV3_REGIONS: ['WDV3CollectEvidence'],
    WDV3_REPORT: ['WDV3Inspect'],
    IMAGE: ['LoadImage'],
};

export function portRole(name) {
    if (name === 'results_json') return 'results_json';
    if (name === 'tags') return 'tags';
    if (name.endsWith('_path')) return 'path';
    if (name.endsWith('_directory') || name === 'directory') return 'directory';
    return null;
}

export function isWDV3Context(context) {
    return context.nodeId?.startsWith('WDV3') || context.type?.startsWith('WDV3_');
}

export function suggestConnections(context, catalog) {
    if (!isWDV3Context(context)) return [];
    const wanted = context.direction === 'input' ? 'outputs' : 'inputs';
    const role = portRole(context.name);
    const preferred = wanted === 'outputs' ? producerPriority[context.type] || [] : [];
    const candidates = [];
    for (const node of catalog.values()) {
        for (const slot of node[wanted]) {
            if (!compatibleTypes(context.type, slot.type)) continue;
            // Keep all generic strings in native search. WDV3's short list is semantic.
            if (context.type === 'STRING' && role && portRole(slot.name) !== role) continue;
            const rank = preferred.indexOf(node.id);
            candidates.push({nodeId: node.id, displayName: node.displayName, ...slot,
                direction: wanted === 'outputs' ? 'output' : 'input',
                priority: rank < 0 ? preferred.length + 1 : rank,
                label: `${node.displayName} · ${slot.name}`});
        }
    }
    const unique = new Map(candidates.map(c => [`${c.nodeId}:${c.direction}:${c.name}`, c]));
    return [...unique.values()].sort((a,b) => a.priority-b.priority ||
        a.displayName.localeCompare(b.displayName) || a.nodeId.localeCompare(b.nodeId) || a.index-b.index);
}
