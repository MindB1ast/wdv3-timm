// Old DOM-widget placeholders may occupy newly appended widget slots.
const blank = value => value == null || (typeof value === 'string' && !value.trim());
export const childRouteDefault = value => blank(value) ? 'separate' : value;
export const tagEnabledDefault = value => blank(value) ? true : value;

export function repairRegionRuleDefaults(node) {
    for (const [name, normalize] of [['child_tag_route', childRouteDefault], ['tag_enabled', tagEnabledDefault]]) {
        const widget = node.widgets?.find(widget => widget.name === name);
        if (!widget) continue;
        widget.value = normalize(widget.value);
        widget.serializeValue = () => normalize(widget.value);
    }
}
