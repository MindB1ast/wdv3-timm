// Visible labels only. Serialized names, slot indices and API keys stay stable.
export function portLabel(nodeId, name, type) {
    if (type === 'WDV3_SAM_SOURCES') return name === 'previous_sources' ? 'previous_sam_queries' : 'sam_queries';
    if (name === 'sources' && nodeId === 'WDV3RemoveRedundantTags') return 'detector_names';
    if (name === 'sources_json' && nodeId === 'WDV3InferenceSettings') return 'region_detectors_json';
    if (name === 'source_name') return nodeId === 'WDV3YOLODetector' ? 'detector_name' : 'sam_query_name';
    if (name === 'refresh') return 'refresh_selection';
    if (name === 'dataset_root' && nodeId === 'WDV3CalculationStore') return 'dataset_root';
    return null;
}

export function labelPorts(node, nodeId = node.comfyClass || node.type) {
    for (const slot of [...(node.inputs || []), ...(node.outputs || [])]) {
        const label = portLabel(nodeId, slot.name, slot.type);
        if (label) slot.label = label;
    }
    for (const widget of node.widgets || []) {
        const label = portLabel(nodeId, widget.name);
        if (label) { widget.label = label; widget.options = {...widget.options, label}; }
    }
}
