# Типы подключений WDV3

Каталог обновлён из реальных legacy/V3-схем установленного ComfyUI.

| Тип | Производители | Потребители |
|---|---|---|
| WDV3_CALCULATION_PLAN | WDV3PlanCalculations.plan | WDV3ComputeMissingEvidence.plan |
| WDV3_CALCULATION_STORE | WDV3LoadConfig.calculation_store, WDV3CalculationStore.calculation_store | WDV3SaveConfig.calculation_store, WDV3DatasetFiles.calculation_store, WDV3PlanCalculations.calculation_store, WDV3SaveEvidence.calculation_store, WDV3LoadEvidence.calculation_store, WDV3ImportCalculations.calculation_store, WDV3LoadCaptions.calculation_store, WDV3PlanCaptionExport.calculation_store |
| WDV3_CAPTION_EDIT | WDV3LoadConfig.caption_edit, WDV3CaptionEditSettings.caption_edit | WDV3SaveConfig.caption_edit, WDV3EditCaptions.caption_edit |
| WDV3_CAPTION_EXPORT | WDV3LoadConfig.caption_export, WDV3CaptionExportSettings.caption_export | WDV3SaveConfig.caption_export, WDV3PlanCaptionExport.caption_export |
| WDV3_CAPTION_PLAN | WDV3PlanCaptionExport.caption_plan | WDV3SaveCaptions.caption_plan |
| WDV3_CAPTION_READ | WDV3LoadConfig.caption_read, WDV3CaptionReadSettings.caption_read | WDV3SaveConfig.caption_read, WDV3LoadCaptions.caption_read |
| WDV3_CAPTIONS | WDV3LoadCaptions.captions, WDV3EditCaptions.captions, WDV3CombineCaptions.captions | WDV3EditCaptions.captions, WDV3CombineCaptions.captions, WDV3PlanCaptionExport.captions |
| WDV3_DETECTORS | WDV3DetectorLoader.detectors, WDV3YOLODetector.detectors, WDV3CombineDetectors.detectors, WDV3LoadConfig.detectors | WDV3Tagger.detectors, WDV3YOLODetector.previous_detectors, WDV3TagFolder.detectors, WDV3CombineDetectors.detectors.detector0, WDV3SaveConfig.detectors, WDV3CollectEvidence.detectors, WDV3ApplyFilters.detectors, WDV3FilterTagSets.detectors |
| WDV3_EVIDENCE | WDV3CollectEvidence.evidence, WDV3ComputeMissingEvidence.evidence, WDV3LoadEvidence.evidence, WDV3ImportCalculations.evidence | WDV3ApplyFilters.evidence, WDV3Inspect.evidence, WDV3FilterTagSets.evidence, WDV3SaveEvidence.evidence |
| WDV3_FILES | WDV3LoadFolder.files, WDV3DatasetFiles.files | WDV3TagFolder.files, WDV3PlanCalculations.files |
| WDV3_INFERENCE | WDV3LoadConfig.inference, WDV3InferenceSettings.inference | WDV3SaveConfig.inference, WDV3PlanCalculations.inference |
| WDV3_MODEL | WDV3ModelLoader.wd_model | WDV3Tagger.wd_model, WDV3TagFolder.wd_model, WDV3CollectEvidence.wd_model |
| WDV3_OPTIONS | WDV3Options.options, WDV3LoadConfig.options | WDV3Tagger.options, WDV3TagFolder.options, WDV3SaveConfig.options, WDV3CollectEvidence.options, WDV3ApplyFilters.options, WDV3FilterTagSets.options, WDV3PlanCalculations.options, WDV3CaptionExportSettings.options |
| WDV3_POSTPROCESSING | WDV3LoadConfig.postprocessing, WDV3PostprocessingConfig.postprocessing | WDV3SaveConfig.postprocessing, WDV3FilterTagSets.postprocessing |
| WDV3_REGIONS | WDV3CollectEvidence.regions | WDV3Inspect.regions |
| WDV3_REPORT | WDV3Inspect.report | WDV3ExportReport.report |
| WDV3_RESULTS | WDV3ApplyFilters.results, WDV3MergeResults.results, WDV3MergeTagSets.results | WDV3MergeResults.left, WDV3MergeResults.right, WDV3Inspect.results, WDV3CombineCaptions.results |
| WDV3_RULES | WDV3RegionRules.rules | WDV3YOLODetector.rules, WDV3SAMSource.rules |
| WDV3_SAM_MODEL | WDV3SAMLoader.sam_model, WDV3LoadConfig.sam_model | WDV3Tagger.sam_model, WDV3SAMSegment.sam_model, WDV3TagFolder.sam_model, WDV3SaveConfig.sam_model, WDV3CollectEvidence.sam_model |
| WDV3_SAM_REGIONS | WDV3Tagger.sam_regions, WDV3SAMSegment.sam_regions | WDV3Tagger.sam_regions, WDV3Preview.sam_regions |
| WDV3_SAM_SOURCES | WDV3SAMSource.sources, WDV3CombineSAMSources.sources, WDV3LoadConfig.sources | WDV3Tagger.sources, WDV3SAMSource.previous_sources, WDV3SAMSegment.sources, WDV3TagFolder.sources, WDV3CombineSAMSources.sources.source0, WDV3SaveConfig.sources, WDV3CollectEvidence.sources, WDV3ApplyFilters.sources, WDV3FilterTagSets.sources |
| WDV3_TAG_SETS | WDV3FilterTagSets.tag_sets, WDV3RemoveRedundantTags.tag_sets | WDV3Inspect.tag_sets, WDV3RemoveRedundantTags.tag_sets, WDV3MergeTagSets.tag_sets |

Для всех 22 пользовательских типов есть производитель и потребитель. IMAGE/MASK остаются стандартными типами; STRING различается рекомендациями по назначению порта.
