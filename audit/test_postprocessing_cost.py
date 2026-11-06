"""Prevent vocabulary-sized rule compilation while preserving full explanations."""
from copy import deepcopy
from collections.abc import Mapping, Sequence
from unittest.mock import patch
import unittest

from audit.test_tag_postprocessing import evidence_fixture,pipeline,Bundle


class PostprocessingCostTests(unittest.TestCase):
    def test_bundle_is_opaque_to_comfy_recursive_memory_sizing(self):
        value=Bundle.pack('rules',{'general_threshold':.3,'nested':{'tag':.5}})
        self.assertNotIsInstance(value,(Mapping,Sequence))
        with patch.object(Bundle,'data',side_effect=AssertionError('Memory sizing must not decode JSON')):
            self.assertEqual(value._comfy_cache_tensors(),())
        self.assertEqual(dict(value),value.data())
        self.assertEqual(dict(value.items()),value.data())
        self.assertEqual(list(value.values()),list(value.data().values()))
        self.assertEqual(value.get('general_threshold'),.3)
        self.assertEqual(value.get('missing','fallback'),'fallback')
        self.assertIn('nested',value)

    def evaluate(self,count):
        data=evidence_fixture().data()
        data['frames']=data['frames'][:1]
        for raw in [data['frames'][0]['full_image']]+[r for rows in data['frames'][0]['detectors'].values() for r in rows]:
            raw['general'].update({f'below_{i}':.001 for i in range(count)})
        evidence=Bundle.pack('evidence',data)
        with patch.object(pipeline,'rule_matcher',wraps=pipeline.rule_matcher) as matcher:
            result=pipeline.apply_filters(evidence)
        return evidence,result,matcher.call_count

    def test_rule_compilation_is_independent_of_vocabulary_size(self):
        _,small,small_calls=self.evaluate(5)
        evidence,large,large_calls=self.evaluate(1000)
        self.assertEqual(small_calls,large_calls)
        a,b=small.data()['frames'][0],large.data()['frames'][0]
        self.assertEqual(a['merged_tags'],b['merged_tags'])
        self.assertEqual(b['full_image']['decisions']['general']['below_999']['reason'],'below_threshold')
        self.assertEqual(sum(d['tag']=='below_999' for d in b['decisions']),4)
        before=evidence.payload
        b['full_image']['raw_scores']['general'].clear()
        self.assertEqual(evidence.payload,before)
        self.assertIn('below_999',large.data()['frames'][0]['full_image']['raw_scores']['general'])

    def test_regex_diagnostics_include_rejected_tags_and_manual_additions(self):
        data=evidence_fixture().data();data['frames']=data['frames'][:1]
        data['frames'][0]['source_settings'][0]['remove_tags_from_full_regex']=['skirt']
        data['frames'][0]['source_settings'][0]['remove_tags_from_region_regex']=['below']
        local=pipeline.filter_tag_sets(Bundle.pack('evidence',data),
            {'prefix_tags':'manual','remove_tags_regex':['manual|below']})
        before=local.payload;result=pipeline.merge_tag_sets(local)
        frame=result.data()['frames'][0]
        below=next(d for d in frame['decisions'] if d['source']=='full' and d['tag']=='below')
        self.assertEqual(below['reason'],'below_threshold')
        self.assertEqual(below['matched_rules'],[dict(kind='regex',pattern='manual|below',action='remove_tags',source='global')])
        self.assertEqual(local.payload,before)
        self.assertEqual(frame['full_image']['decisions']['general']['below']['matched_rules'],below['matched_rules'])


if __name__=='__main__':unittest.main()
