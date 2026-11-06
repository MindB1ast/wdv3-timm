// Size against the DOM-widget mount, never against the current image's contents.
export function addViewerPanel(node, panel, inspect = false) {
    panel.style.cssText = 'display:block;width:100%;max-width:100%;min-width:0;min-height:0;height:100%;'
        + 'align-self:stretch;flex:1 1 0%;overflow:auto;scrollbar-gutter:stable;padding:10px;'
        + 'background:#20242c;color:#eee;font:13px sans-serif;box-sizing:border-box;border-radius:8px';
    const minimum = inspect ? [620, 980] : [460, 620];
    const widget = node.addDOMWidget('wdv3_results', 'wdv3_results', panel, {
        serialize: false,
        getMinHeight: () => minimum[1],
        getHeight: () => '100%',
    });
    // ComfyUI's node inspector draws this same widget in a narrow legacy canvas
    // and writes that canvas width onto it. The graph DOM mount then reuses it.
    // This panel always spans its graph node, regardless of the inspector width.
    Object.defineProperty(widget, 'width', {
        configurable: true,
        get: () => node.size[0],
        set: () => {},
    });
    // Modern widgets distribute available height through computeLayoutSize;
    // only older frontends need the fixed computeSize fallback.
    if (typeof widget.computeLayoutSize !== 'function') {
        widget.computeSize = () => [...minimum];
    }
    return widget;
}

export const viewerContentStyle = 'display:block;width:100%;max-width:100%;min-width:0;box-sizing:border-box;overflow-wrap:anywhere';
export const viewerCanvasStyle = 'display:block;margin:0 auto;width:100%;max-width:100%;height:auto;cursor:crosshair;touch-action:manipulation';
