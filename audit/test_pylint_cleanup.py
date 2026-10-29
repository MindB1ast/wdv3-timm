"""Behavior contracts pinned before the Pylint refactor."""
import json
from pathlib import Path
from unittest.mock import patch

from test_current_behavior import AuditFixture
from Scripts import analitycs
from Scripts.tag_manager import TagManager


def analytics_cases():
    truth = {name: {'eyes', 'skirt', 'jacket'} for name in ('a', 'b', 'c')}
    first = {name: {'eyes', 'skirt', 'wrong'} for name in truth}
    second = {name: {'skirt', 'jacket'} for name in truth}
    errors = {name: analitycs.analyze_tagging_errors(truth, values)
              for name, values in [('first', first), ('second', second)]}
    tags = {name: analitycs.analyze_tag_performance(truth, values).sort_values('tag').reset_index(drop=True)
            for name, values in [('first', first), ('second', second)]}
    metrics = {name: analitycs.calculate_tag_metrics(truth, values)
               for name, values in [('first', first), ('second', second)]}
    figures = {
        'images': analitycs.create_interactive_image_metrics_chart(errors['first'], 'first', 2),
        'tags': analitycs.create_interactive_tag_metrics_chart(tags['first'], 'first', 2),
        'combined_metrics': analitycs.create_combined_metrics_chart(metrics),
        'combined_images': analitycs.create_combined_image_metrics_chart(errors, 2),
        'combined_tags': analitycs.create_combined_tag_metrics_chart(tags, 2),
        'counts': analitycs.create_interactive_tag_count_chart(truth, {'first': first, 'second': second}),
    }
    with patch('plotly.graph_objects.Figure.show'):
        improved = analitycs.analyze_tag_improvements(tags['first'], tags['second'], interactive=True)
        worsened = analitycs.analyze_tag_deteriorations(tags['first'], tags['second'], interactive=True)
    return {'figures': {name: json.loads(figure.to_json()) for name, figure in figures.items()},
            'improved': json.loads(improved.sort_values('tag').reset_index(drop=True).to_json(orient='split')),
            'worsened': json.loads(worsened.sort_values('tag').reset_index(drop=True).to_json(orient='split'))}


class PylintRefactorContracts(AuditFixture):
    def test_analytics_charts_and_change_tables_preserve_baseline(self):
        baseline = json.loads((Path(__file__).parent / 'fixtures/analytics-before.json').read_text(encoding='utf-8'))
        self.assertEqual(analytics_cases(), baseline)

    def test_tag_editing_and_priority_sort_preserve_stats(self):
        target = self.image.with_suffix('.txt')
        target.write_text('skirt, blue eyes, jacket', encoding='utf-8')
        manager = TagManager(str(self.directory))
        self.assertEqual(manager.replace_tag('blue eyes', 'yellow eyes'), 1)
        self.assertEqual(manager.remove_tags('jacket'), 1)
        self.assertEqual(manager.add_tags(add_tags_before='solo'), 1)
        self.assertEqual(manager.sort_tags(importance=['yellow eyes', 'solo']), 1)
        self.assertEqual(manager.image_tags[str(target)]['tags'], ['yellow eyes', 'solo', 'skirt'])
        self.assertEqual(dict(manager.get_top_tags()), {'yellow eyes': 1, 'solo': 1, 'skirt': 1})
        self.assertEqual(manager.save_changes(), 1)
        self.assertEqual(target.read_text(encoding='utf-8'), 'yellow eyes, solo, skirt')



class ModelLoadingContracts(AuditFixture):
    def test_known_yolo_download_has_timeout_and_closes_response(self):
        from Scripts.model import load_yolo_model
        with patch('Scripts.model.requests.get') as request, patch('Scripts.model.YOLO') as loader:
            response = request.return_value.__enter__.return_value
            response.iter_content.return_value = [b'first', b'second']
            result = load_yolo_model('face_yolov9c.pt', self.directory)
            self.assertIs(result, loader.return_value)
            self.assertEqual(request.call_args.kwargs, {'stream': True, 'timeout': 30})
            response.raise_for_status.assert_called_once()
            request.return_value.__exit__.assert_called_once()
        self.assertEqual((self.directory / 'face_yolov9c.pt').read_bytes(), b'firstsecond')

    def test_custom_local_yolo_weights_do_not_access_network(self):
        from Scripts.model import load_yolo_model
        weights = self.directory / 'special.pth'
        weights.write_bytes(b'weights')
        with patch('Scripts.model.requests.get') as request, patch('Scripts.model.YOLO') as loader:
            self.assertIs(load_yolo_model('special', self.directory), loader.return_value)
            loader.assert_called_once_with(str(weights))
            request.assert_not_called()

    def test_shared_label_parser_preserves_categories_and_names(self):
        from Scripts.wdv3_timm import load_labels_csv
        dictionary = self.directory / 'labels.csv'
        dictionary.write_text('name,category\ngeneral_tag,0\ncharacter_(series),4\nsafe,9\n', encoding='utf-8')
        labels = load_labels_csv(dictionary)
        self.assertEqual(labels.names, ['general_tag', 'character_(series)', 'safe'])
        self.assertEqual(labels.general, [0])
        self.assertEqual(labels.character, [1])
        self.assertEqual(labels.rating, [2])
