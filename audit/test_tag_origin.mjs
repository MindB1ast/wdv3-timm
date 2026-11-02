import assert from "node:assert/strict";
import { mergedTagOrigin } from "../web/tag_origin.mjs";

const detail = {
    full_image: {general: {shared: .4, full: .8}},
    detectors: {"person:detail": [{region_id: 0}]},
    merged_tags: {provenance: {
        new: {source: "person:detail:0", model_score: .9},
        shared: {source: "person:detail:0", model_score: .7},
        full: {source: "full", model_score: .8},
        rule: {source: "manual_rule", assigned_score: 1, region_source: "person:detail:0"},
        prefix: {source: "manual_rule", assigned_score: 1},
    }},
};
assert.deepEqual(mergedTagOrigin(detail, "general", "new"), {key:"person:detail:0", label:"person:detail #0", added:true});
assert.equal(mergedTagOrigin(detail, "general", "shared").added, false);
assert.equal(mergedTagOrigin(detail, "general", "full").label, "Full image");
assert.equal(mergedTagOrigin(detail, "general", "rule").added, true);
assert.equal(mergedTagOrigin(detail, "general", "rule").label, "person:detail #0 · rule");
assert.equal(mergedTagOrigin(detail, "general", "prefix").added, false);
assert.equal(mergedTagOrigin({}, "general", "unknown").added, false);
console.log("Merged tag origins: new tags, higher scores, manual region rules and missing metadata passed");
