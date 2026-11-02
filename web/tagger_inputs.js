import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "wdv3.tagger-inputs",
    beforeRegisterNodeDef(Type, data) {
        if (data.name !== "WDV3Tagger") return;
        const refresh = node => {
            const regions = node.inputs?.findIndex(input => input.name === "sam_regions") ?? -1;
            const direct = node.inputs?.some(input =>
                ["detectors", "sam_model", "sources"].includes(input.name) && input.link != null);
            // Preserve existing connected branches when restoring saved workflows.
            if (direct && regions >= 0 && node.inputs[regions].link == null) {
                node.removeInput(regions);
                node.setDirtyCanvas(true, true);
            } else if (!direct && regions < 0) {
                node.addInput("sam_regions", "WDV3_SAM_REGIONS");
                node.setDirtyCanvas(true, true);
            }
        };
        for (const event of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
            const previous = Type.prototype[event];
            Type.prototype[event] = function (...args) {
                const result = previous?.apply(this, args);
                // Link updates finish after ComfyUI invokes these callbacks.
                queueMicrotask(() => refresh(this));
                return result;
            };
        }
    },
});
