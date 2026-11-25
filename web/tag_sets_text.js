import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "wdv3.tagSetsText",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WDV3TagSetsToText") return;
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const text = message?.text?.[0] ?? "";
            const widget = this.widgets?.find(item => item.name === "result")
                ?? this.addWidget("text", "result", "", () => {}, { serialize: false });
            if (widget.inputEl) widget.inputEl.readOnly = true;
            widget.value = text;
            this.setDirtyCanvas(true, true);
        };
    },
});
