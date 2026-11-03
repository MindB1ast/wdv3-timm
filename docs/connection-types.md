# Типы соединений WDV3

Матрица построена по фактическим legacy/V3 definitions 22 зарегистрированных нод (ComfyUI 0.43.2).
Проверка: `node audit/test_connection_catalog.mjs`; текущий каталог можно передать вторым аргументом.

| Тип | Производители (node_id.port) | Потребители (node_id.port) |
| --- | --- | --- |
| WDV3_DETECTORS | WDV3DetectorLoader.detectors, WDV3YOLODetector.detectors, WDV3CombineDetectors.detectors, WDV3LoadConfig.detectors | WDV3Tagger.detectors, WDV3YOLODetector.previous_detectors, WDV3TagFolder.detectors, WDV3CombineDetectors.detectors.detector0, WDV3SaveConfig.detectors, WDV3CollectEvidence.detectors, WDV3ApplyFilters.detectors |
| WDV3_EVIDENCE | WDV3CollectEvidence.evidence | WDV3ApplyFilters.evidence, WDV3Inspect.evidence |
| WDV3_FILES | WDV3LoadFolder.files | WDV3TagFolder.files |
| WDV3_MODEL | WDV3ModelLoader.wd_model | WDV3Tagger.wd_model, WDV3TagFolder.wd_model, WDV3CollectEvidence.wd_model |
| WDV3_OPTIONS | WDV3Options.options, WDV3LoadConfig.options | WDV3Tagger.options, WDV3TagFolder.options, WDV3SaveConfig.options, WDV3CollectEvidence.options, WDV3ApplyFilters.options |
| WDV3_REGIONS | WDV3CollectEvidence.regions | WDV3Inspect.regions |
| WDV3_REPORT | WDV3Inspect.report | WDV3ExportReport.report |
| WDV3_RESULTS | WDV3ApplyFilters.results, WDV3MergeResults.results | WDV3MergeResults.left, WDV3MergeResults.right, WDV3Inspect.results |
| WDV3_RULES | WDV3RegionRules.rules | WDV3YOLODetector.rules, WDV3SAMSource.rules |
| WDV3_SAM_MODEL | WDV3SAMLoader.sam_model, WDV3LoadConfig.sam_model | WDV3Tagger.sam_model, WDV3SAMSegment.sam_model, WDV3TagFolder.sam_model, WDV3SaveConfig.sam_model, WDV3CollectEvidence.sam_model |
| WDV3_SAM_REGIONS | WDV3Tagger.sam_regions, WDV3SAMSegment.sam_regions | WDV3Tagger.sam_regions, WDV3Preview.sam_regions |
| WDV3_SAM_SOURCES | WDV3SAMSource.sources, WDV3CombineSAMSources.sources, WDV3LoadConfig.sources | WDV3Tagger.sources, WDV3SAMSource.previous_sources, WDV3SAMSegment.sources, WDV3TagFolder.sources, WDV3CombineSAMSources.sources.source0, WDV3SaveConfig.sources, WDV3CollectEvidence.sources, WDV3ApplyFilters.sources |

Для всех 12 пользовательских типов есть производитель и потребитель. IMAGE/MASK остаются стандартными типами ComfyUI; строковые значения различаются рекомендациями по назначению порта, без изменения STRING в API.
