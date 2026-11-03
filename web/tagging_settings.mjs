// Old workflows can restore a blank/DOM-widget placeholder into the new INT slot.
export function samBatchSize(value) {
    if ((typeof value !== 'number' && typeof value !== 'string') ||
        (typeof value === 'string' && !value.trim())) return 1;
    const number = Number(value);
    return Number.isSafeInteger(number) ? Math.min(16, Math.max(1, number)) : 1;
}

export function repairTaggingSettings(node) {
    const widget = node.widgets?.find(widget => widget.name === 'sam_batch_size');
    if (!widget) return;
    widget.value = samBatchSize(widget.value);
    // Resolve at submission time as well, including subsequent user edits.
    widget.serializeValue = () => samBatchSize(widget.value);
}
