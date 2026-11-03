import { app } from "../../scripts/app.js";
import { migrateTaggerOptions } from "./tagger_options.mjs";
import { refreshTaggerInputs } from "./tagger_input_state.mjs";

app.registerExtension({
    name: "wdv3.tagger-inputs",
    beforeConfigureGraph(graph) { migrateTaggerOptions(graph); },
    beforeRegisterNodeDef(Type, data) {
        if (data.name !== "WDV3Tagger") return;
        for (const event of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
            const previous = Type.prototype[event];
            Type.prototype[event] = function (...args) {
                const result = previous?.apply(this, args);
                // Link updates finish after ComfyUI invokes these callbacks.
                queueMicrotask(() => refreshTaggerInputs(this, data));
                return result;
            };
        }
    },
});
