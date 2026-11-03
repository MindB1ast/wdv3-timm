import assert from "node:assert/strict";
import { formattedTags } from "../web/tag_format.mjs";
const result={general:{long_hair:.8,blue_eyes:.9},character:{"hero_(series)":.9},caption:"blue_eyes, hero_(series), long_hair"};
assert.equal(formattedTags(result,{tag_separator:" | ",replace_underscores:false}),"blue_eyes | hero_\\(series\\) | long_hair");
assert.equal(formattedTags(result,{tag_separator:" ",escape_parentheses:false}),"blue eyes hero (series) long hair");
assert.equal(formattedTags({general:{a:1,b:1}},{tag_separator:""}),"ab");
assert.equal(formattedTags({}, {tag_separator:" | "}),"");
assert.equal(formattedTags({general:{solo:1}},{tag_separator:"; "}),"solo");
console.log("Custom tag formatting passed");
