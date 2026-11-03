import {app} from '../../scripts/app.js';
import {buildCatalog} from './connection_suggestions.mjs';
import {installCanvasSuggestions, installSlotSuggestions} from './connection_menu.mjs';

const definitions = new Map();
let catalog;
const currentCatalog = () => catalog ??= buildCatalog(Object.fromEntries(definitions));
let warned = false;

app.registerExtension({
    name:'wdv3.connection-suggestions',
    beforeRegisterNodeDef(Type, definition) {
        definitions.set(definition.name, definition);
        catalog = undefined;
        if (definition.name.startsWith('WDV3')) installSlotSuggestions(Type, {
            catalog:currentCatalog, canvas:() => app.canvas,
            get litegraph() { return globalThis.LiteGraph; },
        });
    },
    setup() {
        if (!installCanvasSuggestions(app.canvas, {
            catalog:currentCatalog, litegraph:globalThis.LiteGraph,
            limit:() => app.ui.settings.getSettingValue('Comfy.NodeSuggestions.number', 5),
        }) && !warned) {
            warned = true;
            console.warn('WDV3: connection menu API unavailable; native ComfyUI suggestions remain active.');
        }
    },
});
