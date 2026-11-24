"""Conditional policies on immutable accepted snapshots."""
from copy import deepcopy
import json
import unittest
from audit.test_tag_postprocessing import evidence_fixture, pipeline, post, Bundle, config, reports


def rule(id='apron', **kw):
    return dict(id=id, all_of=['apron'], remove=['dress'], **kw)


def policy(rules=None, **kw):
    return dict(operation='conditional_remove', rules=[rule()] if rules is None else rules, **kw)


def fixture():
    data = evidence_fixture().data()
    for frame in data['frames']:
        frame['full_image']['general'].update(apron=.9, dress=.8, skirt=.7)
        for i, row in enumerate(frame['detectors']['clothes']):
            row['general'] = dict(apron=.9 if i == 0 else .1, dress=.8, skirt=.7)
    return Bundle.pack('evidence', data)


class ConditionalTests(unittest.TestCase):
    def setUp(self):
        self.raw = fixture()
        self.local = pipeline.filter_tag_sets(self.raw)

    def apply(self, rules=None, **kw):
        return post.conditional_remove(self.local, policy(rules, **kw))

    def test_schema(self):
        invalid = [dict(schema_version=True), dict(scope='bad'), dict(sources='clothes'),
                   dict(unknown=True), dict(rules={}), dict(rules=[{}]),
                   dict(rules=[rule(),rule()]), dict(rules=[dict(id='x', remove=['a'])]),
                   dict(rules=[dict(id='x', all_of=['a'], remove=[])]),
                   dict(rules=[dict(id='x', all_of='a', remove=['b'])]),
                   dict(rules=[dict(id='x', all_of=['a,b'], remove=['b'])]),
                   dict(rules=[dict(id='x', all_of=['a b'], none_of=['a_b'], remove=['c'])]),
                   dict(rules=[dict(id='x', any_of=['a'], none_of=['a'], remove=['b'])]),
                   dict(rules=[dict(id='x', all_of=['a'], remove=['b'], typo=True)])]
        original = self.local.to_json()
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                post.conditional_remove(self.local, {**policy(), **value})
        self.assertEqual(original, self.local.to_json())
        normalized = post.validate_operation(policy([dict(id='x', all_of=['a b','a_b'],remove=['c'])]))
        self.assertEqual(normalized['rules'][0]['all_of'], ['a_b'])

    def test_locality_sources_scopes_and_preservation(self):
        original = self.local.to_json()
        out = self.apply(sources=['clothes']).data()
        frame = out['frames'][0]
        self.assertIn('dress',frame['full_image']['general'])
        self.assertNotIn('dress',frame['detectors']['clothes'][0]['general'])
        self.assertIn('dress',frame['detectors']['clothes'][1]['general'])
        self.assertEqual(out['frames'][1]['detectors']['clothes'], [])
        self.assertEqual(out['evidence'],self.local.data()['evidence'])
        for name, rows in frame['detectors'].items():
            for i, row in enumerate(rows):
                for field in ('raw_scores','ratings','bbox'):
                    self.assertEqual(row[field], self.local.data()['frames'][0]['detectors'][name][i][field])
        full=self.apply(scope='full').data()['frames'][0]
        self.assertNotIn('dress',full['full_image']['general'])
        self.assertIn('dress',full['detectors']['clothes'][0]['general'])
        self.assertNotIn('dress',self.apply(scope='all').data()['frames'][1]['full_image']['general'])
        with self.assertRaisesRegex(ValueError,'Unknown'):
            self.apply(sources=['clothes','missing'])
        self.assertEqual(original,self.local.to_json())
        self.assertEqual(self.apply([]).data()['frames'],self.local.data()['frames'])

    def test_snapshot_cycles_order_and_all_causes(self):
        rules=[rule(),dict(id='dress',all_of=['dress'],remove=['skirt']),
               dict(id='second',any_of=['apron','missing'],none_of=['absent'],remove=['dress'])]
        a=self.apply(rules,scope='all')
        self.assertEqual(a,self.apply(list(reversed(rules)),scope='all'))
        row=a.data()['frames'][0]['full_image']
        self.assertEqual(set(row['general']),{'apron','solo'})
        causes=row['decisions']['general']['dress']['postprocessing']['causes']
        self.assertEqual([c['rule_id'] for c in causes],['apron','second'])
        self.assertEqual(causes[0]['evidence']['general'],{'apron':.9})
        self.assertEqual(row['decisions']['general']['dress']['postprocessing']['score'],.8)
        sequential=post.apply_chain(self.local,[policy([rules[0]],scope='full'),policy([rules[1]],scope='full')])
        self.assertIn('skirt',sequential.data()['frames'][0]['full_image']['general'])
        cyclic=self.apply([rule(),dict(id='reverse',all_of=['dress'],remove=['apron'])],scope='full')
        self.assertNotIn('apron',cyclic.data()['frames'][0]['full_image']['general'])

    def test_conditions_characters_manual_and_literal(self):
        for conditions, removed in [({'none_of':['absent']},True),({'all_of':['apron'],'any_of':[]},True),
             ({'all_of':['apron'],'any_of':['missing']},False),({'none_of':['apron']},False),
             ({'all_of':['.*']},False),({'all_of':['apr']},False),({'all_of':['hero_(series)']},True)]:
            out=self.apply([dict(id='x',remove=['dress'],**conditions)],scope='full')
            self.assertEqual('dress' not in out.data()['frames'][0]['full_image']['general'],removed)
        out=self.apply([dict(id='x',all_of=['manual'],remove=['hero_(series)'])])
        self.assertNotIn('hero_(series)',out.data()['frames'][0]['detectors']['face'][0]['character'])
        data=self.local.data();data['frames'][0]['full_image']['general']['.*']=.8
        literal=post.conditional_remove(Bundle.pack('tag_sets',data),policy([dict(id='literal',all_of=['.*'],remove=['dress'])],scope='full'))
        self.assertNotIn('dress',literal.data()['frames'][0]['full_image']['general'])

    def test_chain_config_merge_and_reports(self):
        chain=[policy(scope='regions')]
        restored=config.validate_config(config.config_data(postprocessing=chain))['postprocessing']
        out=post.apply_chain(self.local,restored)
        self.assertEqual(out,Bundle.from_json(out.to_json()))
        result=pipeline.merge_tag_sets(out)
        self.assertIn('dress',result.data()['frames'][0]['merged_tags']['general'])
        self.assertEqual(pipeline.merge_results(result,result).data()['frames'],result.data()['frames'])
        with self.assertRaisesRegex(ValueError,'postprocessing'):
            pipeline.merge_results(result,pipeline.merge_tag_sets(self.local))
        report=reports.make_report(out).data()
        self.assertIn('postprocessing_history',report['frames'][0]['detectors']['clothes'][0])
        redundancy=dict(explicit_enabled=True,relations=[dict(specific='dress',general='skirt')],scope='full')
        a=post.apply_chain(self.local,[policy(scope='full'),redundancy])
        b=post.apply_chain(self.local,[redundancy,policy(scope='full')])
        self.assertIn('skirt',a.data()['frames'][0]['full_image']['general'])
        self.assertNotIn('skirt',b.data()['frames'][0]['full_image']['general'])

if __name__ == '__main__': unittest.main()
