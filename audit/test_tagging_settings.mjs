import assert from 'node:assert/strict';
import {samBatchSize, repairTaggingSettings} from '../web/tagging_settings.mjs';

for (const invalid of ['', ' ', null, undefined, NaN, Infinity, false, [], {}, ', ', 'not-a-number', 1.5]) {
    assert.equal(samBatchSize(invalid), 1);
}
for (const [value, expected] of [[1,1], [4,4], ['8',8], ['2.0',2], [0,1], [-1,1], [32,16]]) {
    assert.equal(samBatchSize(value), expected);
}
const separator={name:'tag_separator',value:' | '};
const widget={name:'sam_batch_size',value:'',serializeValue:()=>''};
const node={widgets:[separator,widget]};
repairTaggingSettings(node);
assert.equal(widget.value,1);
assert.equal(widget.serializeValue(),1);
assert.equal(separator.value,' | ');
widget.value=4;assert.equal(widget.serializeValue(),4);
widget.value='';assert.equal(widget.serializeValue(),1);
widget.value='8';assert.equal(widget.serializeValue(),8);
repairTaggingSettings(node);assert.equal(widget.value,8);
repairTaggingSettings({widgets:[]});
console.log('Tagging Settings: legacy blanks, integer submission, live edits and separator preservation passed');
