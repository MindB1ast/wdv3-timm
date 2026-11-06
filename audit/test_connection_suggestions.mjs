import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {buildCatalog, suggestConnections} from '../web/connection_suggestions.mjs';
const definitions=JSON.parse(readFileSync(new URL('./fixtures/connection_nodes.json',import.meta.url)));
const catalog=buildCatalog(definitions);
const suggestions=(name,type,direction='input',nodeId='WDV3Tagger')=>suggestConnections({name,type,direction,nodeId},catalog);
for(const [name,type,first] of [
    ['wd_model','WDV3_MODEL','WDV3ModelLoader'],['options','WDV3_OPTIONS','WDV3Options'],
    ['sam_model','WDV3_SAM_MODEL','WDV3SAMLoader'],['sources','WDV3_SAM_SOURCES','WDV3SAMSource'],
    ['detectors','WDV3_DETECTORS','WDV3YOLODetector'],['rules','WDV3_RULES','WDV3RegionRules'],
    ['files','WDV3_FILES','WDV3LoadFolder'],['evidence','WDV3_EVIDENCE','WDV3CollectEvidence'],
    ['results','WDV3_RESULTS','WDV3ApplyFilters'],['regions','WDV3_REGIONS','WDV3CollectEvidence'],
    ['report','WDV3_REPORT','WDV3Inspect'],
    ['tag_sets','WDV3_TAG_SETS','WDV3FilterTagSets'],
    ['postprocessing','WDV3_POSTPROCESSING','WDV3PostprocessingConfig']])assert.equal(suggestions(name,type)[0].nodeId,first);
assert.equal(suggestions('sam_model','WDV3_SAM_MODEL').find(c=>c.nodeId==='WDV3LoadConfig').index,3);
assert.equal(suggestions('regions','WDV3_REGIONS')[0].index,1);
assert.ok(suggestions('previous_sources','WDV3_SAM_SOURCES').some(c=>c.nodeId==='WDV3CombineSAMSources'));
assert.ok(suggestions('rules','WDV3_RULES','output').some(c=>c.nodeId==='WDV3YOLODetector'&&c.name==='rules'));
assert.deepEqual(suggestions('results','WDV3_RESULTS','output').filter(c=>c.nodeId==='WDV3MergeResults').map(c=>c.name),['left','right']);
const future={name:'WDV3Future',input:{required:{value:['WDV3_FUTURE',{}]}},output:['WDV3_FUTURE'],output_name:['value']};
const expanded=buildCatalog({...definitions,WDV3Future:future});
assert.equal(suggestConnections({nodeId:'WDV3New',name:'value',type:'WDV3_FUTURE',direction:'input'},expanded)[0].nodeId,'WDV3Future');
assert.deepEqual(suggestions('options','WDV3_OPTIONS'),suggestions('options','WDV3_OPTIONS'));
assert.deepEqual(suggestions('text','STRING','input','ThirdParty'),[]);
for(const node of catalog.values()) {
    for(const slot of node.outputs.filter(slot=>slot.name==='results_json'&&slot.type==='STRING')) {
        const choices=suggestions(slot.name,slot.type,'output',node.id);
        assert.equal(choices[0].nodeId,'WDV3Preview',`${node.id} must recommend Results Viewer first`);
        assert.equal(choices[0].name,'results_json');
        assert.ok(!choices.some(c=>c.name==='tags'));
    }
}
const viewerSources=suggestions('results_json','STRING','input','WDV3Preview');
assert.equal(viewerSources[0].nodeId,'WDV3TagFolder');
for(const id of ['WDV3TagFolder','WDV3Tagger','WDV3MergeTagSets','WDV3ApplyFilters','WDV3MergeResults'])
    assert.ok(viewerSources.some(c=>c.nodeId===id&&c.name==='results_json'),`Missing viewer source: ${id}`);
for(const name of ['tags','results_json','txt_path'])assert.ok(suggestions(name,'STRING').every(c=>name==='txt_path'?c.name.endsWith('_path'):c.name===name));
catalog.set('LoadImage',{id:'LoadImage',displayName:'Load Image',inputs:[],outputs:[{name:'IMAGE',type:'IMAGE',index:0},{name:'MASK',type:'MASK',index:1}]});
assert.equal(suggestions('images','IMAGE')[0].nodeId,'LoadImage');
assert.ok(suggestions('mask','MASK').some(c=>c.nodeId==='LoadImage'&&c.index===1));
console.log('Connection suggestions: all roles, priorities, exact outputs, reverse consumers and future types passed');
