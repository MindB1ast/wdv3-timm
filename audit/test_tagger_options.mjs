import assert from "node:assert/strict";
import {migrateTaggerOptions} from "../web/tagger_options.mjs";
const legacy=()=>({nodes:[{id:1,type:"WDV3Tagger",pos:[500,0],inputs:[],
    widgets_values:[.4,.8,3,false,false,"remove","prefix","suffix",2," | "]}],links:[]});
const graph=legacy();migrateTaggerOptions(graph);
assert.deepEqual(graph.nodes[0].widgets_values,[]);
assert.deepEqual(graph.nodes[1].widgets_values,[.4,.8,2,3,false,false,"remove","prefix","suffix",""," | ",2]);
assert.equal(graph.nodes[0].inputs[0].link,graph.links[0][0]);
const snapshot=JSON.stringify(graph);migrateTaggerOptions(graph);assert.equal(JSON.stringify(graph),snapshot);
const linked=legacy();linked.nodes.push({id:2,type:"WDV3Options",widgets_values:[.6,.9],outputs:[{links:[1]}]});
linked.nodes[0].inputs=[{name:"options",link:1}];linked.links=[[1,2,0,1,0,"WDV3_OPTIONS"]];
migrateTaggerOptions(linked);
assert.equal(linked.nodes[2].widgets_values[0],.6);
assert.equal(linked.nodes[2].widgets_values[11],2);
assert.deepEqual(linked.nodes[1].widgets_values,[.6,.9]);
assert.equal(linked.links.length,1);
const config=legacy();config.nodes.push({id:2,type:"WDV3LoadConfig"});config.nodes[0].inputs=[{name:"options",link:1}];
config.links=[[1,2,0,1,0,"WDV3_OPTIONS"]];migrateTaggerOptions(config);
assert.equal(config.nodes.length,2);assert.equal(config.nodes[0].inputs[0].link,1);
console.log("Tagger settings migration preserves literals, formatting, shared options and connections");
const collected={nodes:[{id:1,type:'WDV3CollectEvidence',inputs:[],widgets_values:[4]}],links:[]};
migrateTaggerOptions(collected);
assert.deepEqual(collected.nodes[0].widgets_values,[]);
assert.equal(collected.nodes[1].type,'WDV3Options');
assert.equal(collected.nodes[1].widgets_values[11],4);
const migrated=JSON.stringify(collected);migrateTaggerOptions(collected);assert.equal(JSON.stringify(collected),migrated);
const controlled={nodes:[{id:1,type:'WDV3CollectEvidence',inputs:[{name:'options',link:1}],widgets_values:[4]},
    {id:2,type:'WDV3Options',widgets_values:[.35,.75,2,8,true,true,'','','','',', ',2],outputs:[{links:[1]}]}],
    links:[[1,2,0,1,0,'WDV3_OPTIONS']]};
migrateTaggerOptions(controlled);
assert.equal(controlled.nodes.length,2);assert.deepEqual(controlled.nodes[0].widgets_values,[]);
assert.equal(controlled.nodes[1].widgets_values[11],2);
