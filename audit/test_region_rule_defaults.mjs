import assert from 'node:assert/strict';
import {childRouteDefault, tagEnabledDefault, repairRegionRuleDefaults} from '../web/region_rule_defaults.mjs';

for (const value of ['', '  ', null, undefined]) {
    assert.equal(childRouteDefault(value), 'separate');
    assert.equal(tagEnabledDefault(value), true);
}
assert.equal(childRouteDefault('merge_to_parent'), 'merge_to_parent');
assert.equal(childRouteDefault('invalid'), 'invalid');
assert.equal(tagEnabledDefault(false), false);
assert.equal(tagEnabledDefault('invalid'), 'invalid');
const route = {name:'child_tag_route', value:'', serializeValue:()=>''};
const enabled = {name:'tag_enabled', value:'', serializeValue:()=>''};
const literal = {name:'add_to_regions', value:'person'};
repairRegionRuleDefaults({widgets:[literal, route, enabled]});
assert.equal(route.value, 'separate');
assert.equal(enabled.value, true);
assert.equal(literal.value, 'person');
route.value='merge_to_parent'; enabled.value=false;
assert.equal(route.serializeValue(), 'merge_to_parent');
assert.equal(enabled.serializeValue(), false);
route.value=''; enabled.value=null;
assert.equal(route.serializeValue(), 'separate');
assert.equal(enabled.serializeValue(), true);
repairRegionRuleDefaults({widgets:[]});
console.log('Region Tag Rules: legacy restore/submission defaults and explicit values preserved');
