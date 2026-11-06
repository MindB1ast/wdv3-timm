// Serialized widget order in Tagging Settings; append new settings only.
const defaults=[.35,.75,2,8,true,true,"","","","",", ",1];

export function migrateTaggerOptions(graph) {
    if(!Array.isArray(graph?.nodes)||!Array.isArray(graph.links))return;
    let nodeId=Math.max(Number(graph.last_node_id)||0,...graph.nodes.map(n=>Number(n.id)||0));
    let linkId=Math.max(Number(graph.last_link_id)||0,...graph.links.map(l=>Number(l[0])||0));
    for(const tagger of [...graph.nodes]) {
        const collecting=tagger.type==="WDV3CollectEvidence";
        if((!collecting&&tagger.type!=="WDV3Tagger")||!Array.isArray(tagger.widgets_values)||tagger.widgets_values.length<(collecting?1:8))continue;
        const old=tagger.widgets_values;
        const input=(tagger.inputs??=[]).find(i=>i.name==="options");
        const edge=graph.links.find(l=>l[0]===input?.link);
        const source=edge&&graph.nodes.find(n=>n.id===edge[1]);
        const batchIndex=collecting?0:8;
        const samBatch=typeof old[batchIndex]==="number"?old[batchIndex]:1;
        // Existing options already controlled tagging. Preserve that connection;
        // clone a Settings node only to carry a formerly local SAM batch size.
        if(edge&&(source?.type!=="WDV3Options"||source.widgets_values?.length>=12||samBatch===1)) {
            tagger.widgets_values=[];
            continue;
        }
        const values=[...defaults];
        if(source?.type==="WDV3Options") {
            source.widgets_values?.forEach((value,index)=>{if(index<values.length)values[index]=value;});
        } else if(!collecting) {
            const positions=[0,1,3,4,5,6,7,8];
            old.slice(0,8).forEach((value,index)=>{values[positions[index]]=value;});
            if(typeof old[9]==="string")values[10]=old[9];
        }
        values[11]=samBatch;
        const settingsId=++nodeId, connection=++linkId;
        let slot=tagger.inputs.findIndex(i=>i.name==="options");
        if(slot<0){slot=tagger.inputs.length;tagger.inputs.push({name:"options",type:"WDV3_OPTIONS",link:null});}
        if(edge) {
            graph.links=graph.links.filter(l=>l[0]!==edge[0]);
            for(const output of source.outputs||[])output.links=output.links?.filter(id=>id!==edge[0])??null;
        }
        graph.nodes.push({id:settingsId,type:"WDV3Options",pos:[(tagger.pos?.[0]||0)-410,tagger.pos?.[1]||0],
            size:[360,640],flags:{},order:settingsId-1,mode:0,inputs:[],
            outputs:[{name:"options",type:"WDV3_OPTIONS",links:[connection],slot_index:0}],
            properties:{"Node name for S&R":"WDV3Options"},widgets_values:values});
        graph.links.push([connection,settingsId,0,tagger.id,slot,"WDV3_OPTIONS"]);
        tagger.inputs[slot].link=connection;
        tagger.widgets_values=[];
    }
    graph.last_node_id=nodeId;graph.last_link_id=linkId;
}
