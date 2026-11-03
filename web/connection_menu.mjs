import {compatibleTypes, isWDV3Context, suggestConnections} from './connection_suggestions.mjs';

const installed = Symbol.for('wdv3.connectionSuggestions');
const slotInstalled = Symbol.for('wdv3.slotSuggestions');

export function connectionContext(node, direction, reference) {
    const slots = direction === 'input' ? node?.inputs : node?.outputs;
    if (!slots) return null;
    const index = typeof reference === 'number' ? reference :
        slots.findIndex(slot => slot.name === (typeof reference === 'string' ? reference : reference?.name));
    const slot = slots[index];
    return slot ? {node, nodeId: node.type || node.comfyClass, direction, index,
        name: slot.name, type: slot.type} : null;
}

function liveSlot(node, direction, candidate) {
    const slots = direction === 'input' ? node.inputs : node.outputs;
    return slots?.findIndex(slot => (slot.name === candidate.name ||
        (candidate.prefix && slot.name.startsWith(candidate.prefix))) &&
        compatibleTypes(slot.type, candidate.type) && (direction !== 'input' || slot.link == null)) ?? -1;
}

export function createConnectedNode(context, candidate, {litegraph, canvas, event}) {
    // Resolve names again: autogrow and sam_regions can change indices after opening a menu.
    const current = connectionContext(context.node, context.direction, context.name);
    const graph = context.node.graph;
    if (!current || !graph || !compatibleTypes(current.type, candidate.type)) return null;
    if (current.direction === 'input' && context.node.inputs[current.index].link != null) return null;
    const created = litegraph?.createNode?.(candidate.nodeId);
    if (!created) return null;
    graph.beforeChange?.();
    try {
        graph.add(created);
        const index = liveSlot(created, candidate.direction, candidate);
        if (index < 0) { graph.remove(created); return null; }
        const x = event?.canvasX ?? canvas.graph_mouse?.[0] ?? 0;
        const y = event?.canvasY ?? canvas.graph_mouse?.[1] ?? 0;
        created.pos = [candidate.direction === 'output' ? x - (created.size?.[0] || 300) - 30 : x + 30, y];
        const link = current.direction === 'input'
            ? created.connect(index, context.node, current.index)
            : context.node.connect(current.index, created, index);
        if (!link) { graph.remove(created); return null; }
        canvas.setDirty?.(true, true);
        return created;
    } catch (error) {
        if (created.graph === graph) graph.remove(created);
        throw error;
    } finally {
        graph.afterChange?.();
    }
}

function menuItems(context, dependencies, event) {
    return suggestConnections(context, dependencies.catalog()).map(candidate => ({
        content: candidate.label,
        callback: () => { createConnectedNode(context, candidate, {...dependencies, event}); },
    }));
}

export function installCanvasSuggestions(canvas, dependencies) {
    if (canvas?.[installed]) return true;
    if (!canvas || typeof canvas.showConnectionMenu !== 'function' ||
        typeof canvas.showSearchBox !== 'function' || !dependencies.litegraph?.ContextMenu) return false;
    const originalMenu = canvas.showConnectionMenu;
    const originalSearch = canvas.showSearchBox;
    const deps = {...dependencies, canvas};
    const nativeSearch = (event, options) => originalSearch.call(canvas, event, options);
    function show(context, event, fallback, searchOptions) {
        if (!context || !isWDV3Context(context)) return fallback();
        const items = menuItems(context, deps, event);
        if (!items.length) return fallback();
        const limit = Math.max(1, dependencies.limit?.() || 5);
        const visible = items.slice(0, limit);
        if (items.length > limit) visible.push({content:'Все совместимые ноды WDV3',
            submenu:{options:items}});
        visible.push(null, {content:'Поиск всех нод…', callback:() => nativeSearch(event, searchOptions)},
            {content:'Штатное меню ComfyUI…', callback:fallback});
        return new dependencies.litegraph.ContextMenu(visible, {event,
            title:`${context.name} · ${context.type}`});
    }
    canvas.showConnectionMenu = function(options = {}) {
        const from = options.nodeFrom != null;
        const context = connectionContext(from ? options.nodeFrom : options.nodeTo,
            from ? 'output' : 'input', from ? options.slotFrom : options.slotTo);
        const searchOptions = from
            ? {node_from:context?.node, slot_from:context?.node.outputs[context.index], type_filter_in:context?.type}
            : {node_to:context?.node, slot_from:context?.node.inputs[context.index], type_filter_out:context?.type};
        return show(context, options.e,
            () => originalMenu.call(this, {...options, showSearchBox:nativeSearch}), searchOptions);
    };
    canvas.showSearchBox = function(event, options = {}) {
        const from = options.node_from != null;
        const node = from ? options.node_from : options.node_to;
        const context = connectionContext(node, from ? 'output' : 'input',
            from ? options.slot_from : options.slot_to ?? options.slot_from);
        return show(context, event, () => originalSearch.call(this, event, options), options);
    };
    canvas[installed] = true;
    return true;
}

export function installSlotSuggestions(Type, dependencies) {
    if (Object.hasOwn(Type.prototype, slotInstalled)) return;
    const previous = Type.prototype.getExtraSlotMenuOptions;
    Type.prototype.getExtraSlotMenuOptions = function(slot) {
        const existing = previous?.call(this, slot) || [];
        const context = connectionContext(this, slot.input ? 'input' : 'output', slot.slot);
        if (!context || !isWDV3Context(context)) return existing;
        const candidates = suggestConnections(context, dependencies.catalog());
        if (!candidates.length) return existing;
        return [...existing, {content:slot.input ? 'WDV3: подключить источник' : 'WDV3: подключить к ноде',
            submenu:{options:candidates.map(candidate => ({
                content:candidate.label,
                callback:(_item, _options, event) => {
                    const canvas = dependencies.canvas();
                    if (canvas) createConnectedNode(context, candidate, {...dependencies, canvas, event});
                },
            }))}}];
    };
    Type.prototype[slotInstalled] = true;
}
