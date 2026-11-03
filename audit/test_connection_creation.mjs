import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {buildCatalog, suggestConnections} from '../web/connection_suggestions.mjs';
import {connectionContext, createConnectedNode, installSlotSuggestions} from '../web/connection_menu.mjs';

const catalog=buildCatalog(JSON.parse(readFileSync(new URL('./fixtures/connection_nodes.json',import.meta.url))));
const graph={nodes:[],links:[],changes:[],add(node){node.graph=this;this.nodes.push(node);},
    remove(node){this.nodes=this.nodes.filter(n=>n!==node);node.graph=null;},
    beforeChange(){this.changes.push('before');},afterChange(){this.changes.push('after');}};
function makeNode(type) {
    const definition=catalog.get(type);
    return {type,size:[300,100],inputs:definition.inputs.map(p=>({...p,link:null})),
        outputs:definition.outputs.map(p=>({...p})),
        connect(output,target,input){
            if(target.inputs[input].link!=null)return null;
            const link={source:this.type,output,target:target.type,input};
            graph.links.push(link);target.inputs[input].link=link;return link;
        }};
}
const litegraph={createNode:makeNode};
const canvas={graph_mouse:[500,100],setDirty(){}};
const deps={litegraph,canvas};
function choose(node,name,producer) {
    const context=connectionContext(node,'input',name);
    const candidate=suggestConnections(context,catalog).find(c=>c.nodeId===producer);
    return createConnectedNode(context,candidate,deps);
}
const tagger=makeNode('WDV3Tagger');graph.add(tagger);
const config=choose(tagger,'sam_model','WDV3LoadConfig');
assert.equal(graph.links.at(-1).output,3);
assert.equal(config.outputs[3].name,'sam_model');
const before=graph.nodes.length;
assert.equal(choose(tagger,'sam_model','WDV3SAMLoader'),null);
assert.equal(graph.nodes.length,before);
const inspector=makeNode('WDV3Inspect');graph.add(inspector);
choose(inspector,'regions','WDV3CollectEvidence');assert.equal(graph.links.at(-1).output,1);
assert.equal(graph.links.at(-1).input,inspector.inputs.findIndex(p=>p.name==='regions'));
// A menu can survive a dynamic input being removed before its selection.
const context=connectionContext(tagger,'input','sam_regions');
const candidate=suggestConnections(context,catalog)[0];
tagger.inputs=tagger.inputs.filter(p=>p.name!=='sam_regions');
assert.equal(createConnectedNode(context,candidate,deps),null);
tagger.inputs.unshift({name:'sam_regions',type:'WDV3_SAM_REGIONS',link:null});
createConnectedNode(context,candidate,deps);assert.equal(graph.links.at(-1).input,0);
// Reverse direction uses the selected consumer port, including left/right.
const filtered=makeNode('WDV3ApplyFilters');graph.add(filtered);
const outputContext=connectionContext(filtered,'output','results');
const right=suggestConnections(outputContext,catalog).find(c=>c.nodeId==='WDV3MergeResults'&&c.name==='right');
createConnectedNode(outputContext,right,deps);assert.equal(graph.links.at(-1).input,1);
const sam=makeNode('WDV3SAMSource');graph.add(sam);
const samOutput=connectionContext(sam,'output','sources');
const combine=suggestConnections(samOutput,catalog).find(c=>c.nodeId==='WDV3CombineSAMSources');
const combined=createConnectedNode(samOutput,combine,deps);
assert.equal(combined.inputs[0].name,'sources.source0');
assert.equal(graph.changes.length%2,0);
for(let i=0;i<graph.changes.length;i+=2)assert.deepEqual(graph.changes.slice(i,i+2),['before','after']);
const count=graph.nodes.length;
createConnectedNode(outputContext,{...right,name:'missing',prefix:undefined},deps);
assert.equal(graph.nodes.length,count); // stale candidate is rolled back
class Type {}
let oldCalls=0;
Type.prototype.getExtraSlotMenuOptions=()=>{oldCalls++;return [{content:'previous extension'}];};
installSlotSuggestions(Type,{catalog:()=>catalog,canvas:()=>canvas,litegraph});
const handler=Type.prototype.getExtraSlotMenuOptions;
installSlotSuggestions(Type,{catalog:()=>catalog,canvas:()=>canvas,litegraph});
assert.equal(Type.prototype.getExtraSlotMenuOptions,handler);
const instance=Object.assign(new Type(),makeNode('WDV3Tagger'));graph.add(instance);
const modelIndex=instance.inputs.findIndex(p=>p.name==='wd_model');
const info={input:instance.inputs[modelIndex],slot:modelIndex};
const menu=instance.getExtraSlotMenuOptions(info);
assert.equal(menu[0].content,'previous extension');assert.equal(oldCalls,1);
assert.match(menu[1].submenu.options[0].content,/Load Tagger Model/);
assert.equal(graph.nodes.length,count+1); // opening/cancelling a menu does not create nodes
// A newly added autogrow input resolves by its actual current name, not an old index.
combined.inputs.push({name:'sources.source1',type:'WDV3_SAM_SOURCES',link:null});
choose(combined,'sources.source1','WDV3SAMSource');
assert.equal(graph.links.at(-1).input,1);
console.log('Connection creation: exact slots, occupied inputs, rollback, dynamic ports, reverse wiring and change transactions passed');
