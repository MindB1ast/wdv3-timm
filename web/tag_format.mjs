// Match the backend's final formatting without parsing the rendered output.
export function formattedTags(result, options={}) {
    const names=new Set([...Object.keys(result.general||{}),...Object.keys(result.character||{})]);
    const canonical=value=>value.trim().replaceAll(" ","_");
    const ordered=[...new Set((result.caption||"").split(/[,\r\n]+/).map(canonical).filter(tag=>names.has(tag)))];
    ordered.push(...[...names].filter(tag=>!ordered.includes(tag)).sort());
    return ordered.map(tag=>{
        if(options.replace_underscores!==false)tag=tag.replaceAll("_"," ");
        if(options.escape_parentheses!==false)tag=tag.replaceAll("(","\\(").replaceAll(")","\\)");
        return tag;
    }).join(options.tag_separator??", ");
}
