"""Composable local tag operations and portable settings, without inference."""
import json
from comfy_api.latest import io
from .comfy_input_help import string_input
from .comfy_pipeline_nodes import WDApplyFilters, TAG_SETS
from .comfy_evidence import filter_tag_sets, merge_tag_sets, outputs
from .Scripts.configs import DetectorConfig
from .Scripts.tag_postprocessing import remove_redundant, operation_from_fields, apply_chain, validate_chain

POSTPROCESSING = io.Custom('WDV3_POSTPROCESSING')


class WDFilterTagSets(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3FilterTagSets', display_name='WDV3 Filter Tag Sets', category='WDV3/Postprocessing',
            description="Применяет локальные пороги и правила к evidence и сохраняет отдельный набор тегов полного кадра и каждого региона. Это вход в цепочку постобработки перед глобальным объединением. После операций подключите tag_sets к Merge Tag Sets; для просмотра локальных наборов — к Inspect.",
            inputs=WDApplyFilters.define_schema().inputs + [POSTPROCESSING.Input('postprocessing', optional=True,
                tooltip='Необязательная сохранённая цепочка из Postprocessing Config / Load Pipeline Config. Применяется к локальным наборам без инференса.')],
            outputs=[TAG_SETS.Output('tag_sets', tooltip='Отдельные принятые наборы full и каждого региона до merge. Подключите к операции, Merge Tag Sets или Inspect.')])

    @classmethod
    def execute(cls, evidence, options=None, detectors=None, sources=None, postprocessing=None):
        configs = None
        if detectors is not None or sources is not None:
            configs = list(detectors[0]) if detectors else []
            configs += [DetectorConfig(s['name'], 'sam3', **s.get('rules', {})) for s in (sources or [])]
        return io.NodeOutput(apply_chain(filter_tag_sets(evidence, options, configs), postprocessing or []))


class WDRemoveRedundantTags(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3RemoveRedundantTags', display_name='WDV3 Remove Redundant Tags', category='WDV3/Postprocessing',
            description="Удаляет избыточные general-теги внутри каждого выбранного локального набора. Доступны точные пары, отдельные regex-семейства и текстовая эвристика; protected_tags защищает нужные теги. Исходные оценки и области сохраняются, а Inspect показывает причины и before/after. Все методы выключены по умолчанию.",
            inputs=[TAG_SETS.Input('tag_sets', tooltip='Принятые локальные теги из Filter Tag Sets / предыдущей операции. Raw-оценки сохраняются.'),
                    io.Boolean.Input('explicit_enabled', default=False, tooltip='Включить точные пары specific → general. Оба тега должны быть приняты в одном наборе.'),
                    io.Boolean.Input('textual_enabled', default=False, tooltip='Необязательная эвристика: black_skirt вытесняет skirt по целым компонентам; skirted — нет. Может удалить полезный общий тег.'),
                    string_input('relations_json', default='[]', multiline=True, tooltip='JSON-массив, например [{"specific":"miniskirt","general":"skirt"}]. Полные буквальные имена, пробелы = _. Без regex/подстрок; циклы запрещены.'),
                    string_input('protected_tags', default='', multiline=True, tooltip='Сохранить эти теги при обоих методах. Список через запятые/переносы; пробел внутри имени = _.'),
                    io.Combo.Input('scope', options=['all', 'full', 'regions'], default='all', tooltip='Full, отдельные regions или оба. Разные наборы не сравниваются.'),
                    string_input('sources', display_name='detector_names', default='', tooltip='Имена региональных источников через запятую/перенос; пусто — все. Не ограничивает full. Неизвестное имя — ошибка.'),
                    io.Boolean.Input('regex_enabled', default=False, optional=True, tooltip='Включить отдельные regex-отношения. Шаблон specific сопоставляется с полным каноническим именем; general — шаблон замены с захваченными группами.'),
                    string_input('regex_relations_json', default='[]', multiline=True, optional=True,
                        tooltip='JSON-массив правил с id, specific (regex) и general (шаблон замены). Например specific=(?:black|white)_(?P<base>.+), general=\\g<base>. В JSON обратный слеш удваивается. Удаляется только уже принятый общий тег в том же наборе.')],
            outputs=[TAG_SETS.Output('tag_sets', tooltip='Новые наборы с историей before/after для следующей операции, Merge Tag Sets или Inspect.')])

    @classmethod
    def execute(cls, tag_sets, explicit_enabled=False, textual_enabled=False, relations_json='[]',
                protected_tags='', scope='all', sources='', regex_enabled=False, regex_relations_json='[]'):
        operation = operation_from_fields(relations_json, protected_tags, scope, sources,
                                          explicit_enabled, textual_enabled, regex_enabled, regex_relations_json)
        return io.NodeOutput(remove_redundant(tag_sets, operation))


class WDMergeTagSets(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3MergeTagSets', display_name='WDV3 Merge Tag Sets', category='WDV3/Postprocessing',
            description="Собирает локальные наборы полного кадра и регионов после постобработки в общий caption. Применяет правила переноса, владения, композиции и финальное форматирование. Повторы объединяются только в общем описании; отдельные региональные записи сохраняются в results и results_json.",
            inputs=[TAG_SETS.Input('tag_sets', tooltip='Локальные наборы после операций. Здесь применяются глобальное отсечение, композиция и формат вывода.')],
            outputs=WDApplyFilters.define_schema().outputs)

    @classmethod
    def execute(cls, tag_sets):
        result = merge_tag_sets(tag_sets)
        tags, details = outputs(result)
        return io.NodeOutput(result, tags, details)


class WDPostprocessingConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3PostprocessingConfig', display_name='WDV3 Postprocessing Config', category='WDV3/Settings',
            description="Создаёт переносимую упорядоченную цепочку операций из JSON-описания. Подключите postprocessing к Save Pipeline Config для сохранения или к Filter Tag Sets для применения к готовым evidence. Подходит для хранения и повторного использования правил без загрузки изображений и моделей.",
            inputs=[string_input('postprocessing_json', default='[]', multiline=True,
                tooltip='JSON-массив операций schema_version=1, operation=remove_redundant. Сохраните через Save Pipeline Config или примените через Filter Tag Sets. Без пикселей/оценок.')],
            outputs=[POSTPROCESSING.Output('postprocessing', tooltip='Проверенная цепочка для Save Pipeline Config или Filter Tag Sets.')])

    @classmethod
    def execute(cls, postprocessing_json='[]'):
        return io.NodeOutput(validate_chain(json.loads(postprocessing_json)))


NODE_CLASSES = [WDFilterTagSets, WDRemoveRedundantTags, WDMergeTagSets, WDPostprocessingConfig]
