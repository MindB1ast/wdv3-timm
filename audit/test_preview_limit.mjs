import assert from "node:assert/strict";
import { previewLimit, repairPreviewLimit } from "../web/preview_limit.mjs";

for (const empty of ["", " ", null, undefined, NaN]) assert.equal(previewLimit(empty), 16);
assert.equal(previewLimit("32"), 32);
assert.equal(previewLimit(4), 4);
assert.equal(previewLimit(0), 1);
assert.equal(previewLimit(300), 256);
const widget = {name: "preview_limit", value: "", serializeValue: () => ""};
const node = {widgets: [{name: "other", value: ""}, widget]};
repairPreviewLimit(node);
assert.equal(widget.value, 16);
assert.equal(widget.serializeValue(), 16);
widget.value = 48;
assert.equal(widget.serializeValue(), 48);
widget.value = "";
assert.equal(widget.serializeValue(), 16);
assert.equal(node.widgets[0].value, "");
console.log("preview_limit restore and submission tests passed");
