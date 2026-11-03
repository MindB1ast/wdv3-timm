// YOLO and SAM are independent. Text sources belong only to the live SAM path.
export function refreshTaggerInputs(node, definition) {
    const linked = name => node.inputs?.some(input => input.name === name && input.link != null);
    const sam = linked('sam_model');
    const sources = linked('sources');
    let changed = false;
    function visibility(name, type, visible) {
        const index = node.inputs?.findIndex(input => input.name === name) ?? -1;
        if (!visible && index >= 0 && node.inputs[index].link == null) {
            node.removeInput(index);
            changed = true;
        } else if (visible && index < 0) {
            node.addInput(name, type, {tooltip:definition.input?.optional?.[name]?.[1]?.tooltip});
            changed = true;
        }
    }
    // Keep existing connected ports when restoring or reconnecting a workflow.
    visibility('sources', 'WDV3_SAM_SOURCES', sam || sources);
    visibility('sam_regions', 'WDV3_SAM_REGIONS', !(sam || sources));
    const sourceInput = node.inputs?.find(input => input.name === 'sources');
    if (sourceInput) sourceInput.tooltip = sam
        ? definition.input?.optional?.sources?.[1]?.tooltip
        : 'SAM sources требуют sam_model. Подключите Load SAM 3.1 или отключите sources для тегирования без SAM.';
    if (changed) node.setDirtyCanvas(true, true);
}
