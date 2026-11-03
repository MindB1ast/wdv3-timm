import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {buildCatalog, catalogCoverage, normalizeNodeDefinition} from "../web/connection_suggestions.mjs";

const definitions = JSON.parse(readFileSync(process.argv[2] || new URL('./fixtures/connection_nodes.json', import.meta.url)));
const catalog = buildCatalog(definitions);
if (process.argv[2]) {
    const fixture=buildCatalog(JSON.parse(readFileSync(new URL('./fixtures/connection_nodes.json',import.meta.url))));
    assert.deepEqual([...catalog], [...fixture], 'Real backend port contracts differ from the checked catalog');
}
assert.equal(catalog.get('WDV3LoadConfig').outputs.find(p => p.name === 'sam_model').index, 3);
assert.deepEqual(catalog.get('WDV3CombineSAMSources').inputs.map(p => [p.name,p.type]),
    [['sources.source0','WDV3_SAM_SOURCES']]);
const forced = normalizeNodeDefinition({name:'Test', input:{required:{text:['STRING',{}],
    data:['STRING',{forceInput:true}], tensor:['IMAGE',{}]}}, output:['MASK']});
assert.deepEqual(forced.inputs.map(p=>p.name),['data','tensor']);
assert.deepEqual(forced.inputs.map(p=>p.index),[0,1]);
const coverage = catalogCoverage(catalog);
assert.equal(coverage.length,12);
for (const type of coverage) {
    assert.ok(type.producers.length, `No producer for ${type.type}`);
    assert.ok(type.consumers.length, `No consumer for ${type.type}`);
}
console.log(`Catalog: ${catalog.size} nodes, ${coverage.length} custom types, complete port coverage`);
