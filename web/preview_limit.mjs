// Older workflows can restore an empty string into the numeric widget.
export function previewLimit(value) {
    if (value == null || (typeof value === "string" && !value.trim())) return 16;
    const number = Number(value);
    return Number.isFinite(number) ? Math.min(256, Math.max(0, Math.trunc(number))) : 16;
}

export function repairPreviewLimit(node) {
    const widget = node.widgets?.find(widget => widget.name === "preview_limit");
    if (!widget) return;
    widget.value = previewLimit(widget.value);
    // Read the current value at submission time, including after editing it.
    widget.serializeValue = () => previewLimit(widget.value);
}
