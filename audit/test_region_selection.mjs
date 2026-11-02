import assert from "node:assert/strict";
import { imagePoint, regionsAt, nextRegion } from "../web/region_selection.mjs";

const regions = [
    {key:"person_detail:0", option:2, bbox:[20,10,80,90]},
    {key:"person_context:0", option:3, bbox:[20,10,80,90]},
    {key:"cup:0", option:4, bbox:[30,40,50,60]},
];
const point = imagePoint(175, 250, {left:100, top:100, width:200, height:400}, 100, 200);
assert.deepEqual(point, {x:37.5, y:75});
const hits = regionsAt(regions, 40, 50);
assert.deepEqual(hits.map(r => r.option), [4,2,3]);
assert.equal(nextRegion(hits, null, false).option, 4);
assert.equal(nextRegion(hits, 4, true).option, 2);
assert.equal(nextRegion(hits, 2, true).option, 3);
assert.equal(nextRegion(hits, 3, true).option, 4);
assert.equal(nextRegion([], 3, true), null);
assert.deepEqual(regionsAt(regions, 80, 90), []);
assert.deepEqual(regionsAt(regions, 40, 50, region => region.option === 3).map(r => r.option), [3]);
console.log("Region picking: coordinate scaling, overlap cycling and mask filtering passed");
