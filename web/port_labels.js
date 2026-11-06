import {app} from '../../scripts/app.js';
import {labelPorts, portLabel} from './port_labels.mjs';

app.registerExtension({
    name: 'wdv3.readable-port-labels',
    beforeRegisterNodeDef(Type, data) {
        if (!data.name?.startsWith('WDV3')) return;
        for (const section of ['required', 'optional']) {
            for (const [name, spec] of Object.entries(data.input?.[section] || {})) {
                const label = portLabel(data.name, name, spec[0]);
                if (label) (spec[1] ??= {}).display_name = label;
            }
        }
        for (const event of ['onNodeCreated', 'onConfigure', 'onConnectionsChange']) {
            const previous = Type.prototype[event];
            Type.prototype[event] = function (...args) {
                const result = previous?.apply(this, args);
                queueMicrotask(() => labelPorts(this, data.name));
                return result;
            };
        }
    },
});
