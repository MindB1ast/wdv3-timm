import { app } from "../../scripts/app.js";
import { ruleWidgetValues, taggingWidgetValues } from "./rule_presets.mjs";
import { repairTaggingSettings } from "./tagging_settings.mjs";
const presets={
    "Full image":{},"Generic region":{exclude_from_region:["close-up","cowboy_shot","full_body"]},
    "Person":{remove_tags_from_full:["1girl","1boy"],add_tags_to_region:{person:1},exclude_from_region:["close-up","cowboy_shot","full_body"]},
    "Face":{remove_tags_from_full:["face"],remove_tags_from_region:["full_body"],add_tags_to_region:{face:1},exclude_from_region:["portrait","upper_body"]},
    "Custom":null
};
app.registerExtension({name:"wdv3.rule-presets",async beforeRegisterNodeDef(Type,data){
    if(!["WDV3RegionRules","WDV3Options"].includes(data.name))return;
    const fullSettings=data.name==="WDV3Options";
    const configured=Type.prototype.onConfigure;
    Type.prototype.onConfigure=function(){const result=configured?.apply(this,arguments);if(fullSettings)repairTaggingSettings(this);this.setSize([Math.max(360,this.size[0]),Math.max(fullSettings?640:1000,this.size[1])]);return result;};
    const created=Type.prototype.onNodeCreated;
    Type.prototype.onNodeCreated=function(){
        const result=created?.apply(this,arguments),node=this;
        if(fullSettings)repairTaggingSettings(node);
        const compositionDefault=node.widgets?.find(w=>w.name==="full_image_only_tags")?.value;
        const panel=document.createElement("div");panel.style.cssText="padding:6px;color:#ddd;background:#242832;font:12px sans-serif";
        const select=document.createElement("select");
        for(const name of fullSettings?["Full image","Custom"]:Object.keys(presets)){const o=document.createElement("option");o.textContent=name;o.value=name;select.append(o)}select.value="Custom";
        const apply=document.createElement("button");apply.textContent="Apply preset";
        const file=document.createElement("input");file.type="file";file.accept=".json,application/json";file.style.width="100%";
        const choice=document.createElement("select"),load=document.createElement("button");load.textContent="Import selected detector";load.disabled=true;
        const note=document.createElement("div");note.textContent="Literal tags: blue eyes = blue_eyes; comma/newline lists. Regex: separate fields, one full-name pattern per line. Import detectors.json below.";
        let detectors=[];
        const applyRules=rules=>{
            const values=fullSettings?taggingWidgetValues(rules):ruleWidgetValues(rules);
            if(!fullSettings&&values.full_image_only_tags===undefined)values.full_image_only_tags=compositionDefault??"";
            for(const [name,value]of Object.entries(values)){
                const w=node.widgets?.find(w=>w.name===name);if(!w)continue;
                w.value=value;
                w.callback?.(w.value);
            }
            app.graph.setDirtyCanvas(true,true);
        };
        apply.onclick=()=>{try{if(presets[select.value]){applyRules(presets[select.value]);note.textContent="Preset applied. Review all fields before running."}}catch(error){note.textContent=error.message}};
        file.onchange=async()=>{try{
            const data=JSON.parse(await file.files[0].text());detectors=Array.isArray(data)?data:data.detectors;
            if(!Array.isArray(detectors)||!detectors.every(row=>row&&typeof row.name==="string"))throw new Error("Expected detector array or config.detectors");
            choice.replaceChildren();detectors.forEach((d,i)=>{const o=document.createElement("option");o.value=i;o.textContent=d.name;choice.append(o)});load.disabled=!detectors.length;
            note.textContent="Choose a detector, then import its rules. No other nodes will be changed.";
        }catch(error){load.disabled=true;note.textContent=String(error.message)}};
        load.onclick=()=>{try{const d=detectors[Number(choice.value)];if(!d)return;applyRules(d.rules||d);select.value="Custom";
            note.textContent=`Imported ${d.name}. Assign separately in YOLO: model=${d.model_path??"not provided"}, confidence=${d.confidence??"default"}, classes=${JSON.stringify(d.classes??null)}. Assigned tag scores preserved.`}catch(error){note.textContent=error.message}};
        if(fullSettings){note.textContent="Full-image settings. Per-source presets and detector import are in Region Tag Rules.";panel.append(select,apply,note);}
        else panel.append(select,apply,file,choice,load,note);
        this.addDOMWidget("wdv3_presets","wdv3_presets",panel,{serialize:false});
        this.setSize([Math.max(360,this.size[0]),Math.max(fullSettings?640:1000,this.size[1])]);
        return result;
    };
}});
