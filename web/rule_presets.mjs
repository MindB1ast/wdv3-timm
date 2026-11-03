const mapping={region_gen_threshold:"general_threshold",region_char_threshold:"character_threshold",
    remove_tags_from_full:"remove_from_full",remove_tags_from_region:"remove_from_regions",
    add_tags_to_region:"add_to_regions",exclude_from_region:"require_full_confirmation",
    only_from_source:"only_from_source",allowed_tags:"allowed_tags",full_image_only_tags:"full_image_only_tags"};

export function ruleWidgetValues(rules){
    if(!rules||typeof rules!=="object"||Array.isArray(rules))throw new Error("Rules must be an object");
    const out={};
    for(const [key,name] of Object.entries(mapping)){
        const value=rules[key];
        if(key==="full_image_only_tags"&&value==null)continue;
        if(name.endsWith("threshold")){
            const number=value??(name==="general_threshold"?.35:.75);
            if(typeof number!=="number"||!Number.isFinite(number)||number<0||number>1)throw new Error(`${key}: expected a number 0..1`);
            out[name]=number;
        }else if(key==="add_tags_to_region"){
            const additions=value??{};
            if(typeof additions!=="object"||Array.isArray(additions))throw new Error("add_tags_to_region: expected tag/score object");
            out[name]=Object.entries(additions).map(([tag,score])=>{
                if(!tag.trim()||typeof score!=="number"||!Number.isFinite(score)||score<0||score>1)throw new Error("Invalid assigned tag score");
                return score===1?tag:`${tag}:${score}`;
            }).join(", ");
        }else{
            const tags=value??[];
            if(!Array.isArray(tags)||!tags.every(tag=>typeof tag==="string"))throw new Error(`${key}: expected a list of tags`);
            out[name]=[...new Set(tags.map(tag=>tag.trim().replaceAll(" ","_")).filter(Boolean))].join(", ");
        }
    }
    for(const field of ["remove_tags_from_full","remove_tags_from_region","exclude_from_region",
                        "only_from_source","allowed_tags","full_image_only_tags"]){
        const key=field+"_regex", patterns=rules[key]??[];
        if(!Array.isArray(patterns)||!patterns.every(p=>typeof p==="string"))throw new Error(`${key}: expected patterns`);
        out[key]=patterns.join("\n");
    }
    return out;
}

export function taggingWidgetValues(options={}){
    const values={general_threshold:.35,character_threshold:.75,image_batch_size:2,region_batch_size:8,
        replace_underscores:true,escape_parentheses:true,remove_tags:"",prefix_tags:"",suffix_tags:"",
        remove_tags_regex:[],tag_separator:", ",sam_batch_size:1,...options};
    if(!Array.isArray(values.remove_tags_regex)||!values.remove_tags_regex.every(p=>typeof p==="string"))throw new Error("remove_tags_regex: expected patterns");
    values.remove_tags_regex=values.remove_tags_regex.join("\n");
    for(const [name,value] of Object.entries(values)){
        if(name.endsWith("threshold")&&(typeof value!=="number"||!Number.isFinite(value)||value<0||value>1))throw new Error(`${name}: expected 0..1`);
        if(name.endsWith("batch_size")&&(!Number.isInteger(value)||value<1))throw new Error(`${name}: expected a positive integer`);
        if(["replace_underscores","escape_parentheses"].includes(name)&&typeof value!=="boolean")throw new Error(`${name}: expected boolean`);
        if(["remove_tags","prefix_tags","suffix_tags","tag_separator"].includes(name)&&typeof value!=="string")throw new Error(`${name}: expected text`);
    }
    return values;
}
