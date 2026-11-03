import assert from 'node:assert/strict';
import {refreshTaggerInputs} from '../web/tagger_input_state.mjs';

const definition={input:{optional:{sources:['WDV3_SAM_SOURCES',{tooltip:'SAM requests'}],
    sam_regions:['WDV3_SAM_REGIONS',{tooltip:'Existing masks'}]}}};
function node() {
    return {inputs:['images','wd_model','options','detectors','sam_model','sources','sam_regions'].map(name=>({name,link:null})),
        removeInput(index){this.inputs.splice(index,1);},addInput(name,type,options){this.inputs.push({name,type,...options,link:null});},
        setDirtyCanvas(){}};
}
const input=(n,name)=>n.inputs.find(i=>i.name===name);
const fresh=node();refreshTaggerInputs(fresh,definition);
assert.equal(input(fresh,'sources'),undefined);
assert.ok(input(fresh,'sam_regions'));
input(fresh,'detectors').link=1;refreshTaggerInputs(fresh,definition);
assert.ok(input(fresh,'sam_regions')); // YOLO may combine with precomputed SAM regions
assert.equal(input(fresh,'sources'),undefined);
input(fresh,'sam_model').link=2;refreshTaggerInputs(fresh,definition);
assert.ok(input(fresh,'sources'));assert.equal(input(fresh,'sam_regions'),undefined);
input(fresh,'sources').link=3;
input(fresh,'sam_model').link=null;refreshTaggerInputs(fresh,definition);
assert.equal(input(fresh,'sources').link,3); // never silently remove a saved connection
assert.match(input(fresh,'sources').tooltip,/sam_model/);
input(fresh,'sources').link=null;refreshTaggerInputs(fresh,definition);
assert.equal(input(fresh,'sources'),undefined);assert.ok(input(fresh,'sam_regions'));
const saved=node();input(saved,'sam_regions').link=4;input(saved,'sam_model').link=5;
refreshTaggerInputs(saved,definition);
assert.equal(input(saved,'sam_regions').link,4);assert.equal(input(saved,'sam_model').link,5);
const snapshot=JSON.stringify(fresh.inputs);refreshTaggerInputs(fresh,definition);
assert.equal(JSON.stringify(fresh.inputs),snapshot);
console.log('Tagger input dependencies: full-image, SAM sources, YOLO with saved masks and existing links passed');
