"""Explicit raw evidence, filtering, merging, inspection and report stages."""
from contextlib import ExitStack
import json
from pathlib import Path
import folder_paths
from comfy_api.latest import io

from .comfy_ui_nodes import MODEL, OPTIONS, DETECTORS, SAM_MODEL, SAM_SOURCES, RULES, progress
from .comfy_processing import tensor_frame
from .comfy_evidence import collect, apply_filters, outputs, merge_results, cached_segment
from .Scripts.configs import DetectorConfig
from .Scripts.evidence import Bundle

EVIDENCE=io.Custom('WDV3_EVIDENCE')
RESULTS=io.Custom('WDV3_RESULTS')
REGIONS=io.Custom('WDV3_REGIONS')
REPORT=io.Custom('WDV3_REPORT')


class WDCollectEvidence(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3CollectEvidence',display_name='WDV3 Collect Evidence',category='WDV3/Pipeline',
            description='Collect all WD scores before thresholds. Reuses cached inference when only rules change.',
            inputs=[io.Image.Input('images'),MODEL.Input('wd_model'),OPTIONS.Input('options',optional=True),
                    DETECTORS.Input('detectors',optional=True),SAM_MODEL.Input('sam_model',optional=True),
                    SAM_SOURCES.Input('sources',optional=True),io.Int.Input('sam_batch_size',default=1,min=1,max=16)],
            outputs=[EVIDENCE.Output('evidence'),REGIONS.Output('regions')])
    @classmethod
    def execute(cls,images,wd_model,sam_batch_size=1,options=None,detectors=None,sam_model=None,sources=None):
        sources=sources or None
        if (sam_model is None)!=(sources is None): raise ValueError('SAM needs both sam_model and sources')
        from .comfy_config import unique_names
        unique_names([c.name for c in (detectors[0] if detectors else [])]+[s['name'] for s in (sources or [])])
        with ExitStack() as stack:
            frames=[stack.enter_context(tensor_frame(t)) for t in images]
            regions=cached_segment(frames,wd_model,sam_model,sources,sam_batch_size,progress) if sources else None
            evidence=collect(frames,wd_model,options=options,detectors=detectors,sam_regions=regions,progress=progress)
        geometry=evidence.data()
        for frame in geometry['frames']:
            frame.pop('full_image',None)
            for rs in frame['detectors'].values():
                for r in rs:
                    for category in ['general','character','ratings']:r.pop(category,None)
        return io.NodeOutput(evidence,Bundle.pack('regions',geometry))


class WDApplyFilters(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3ApplyFilters',display_name='WDV3 Apply Filters',category='WDV3/Pipeline',
            inputs=[EVIDENCE.Input('evidence'),OPTIONS.Input('options',optional=True),
                    DETECTORS.Input('detectors',optional=True),SAM_SOURCES.Input('sources',optional=True)],
            outputs=[RESULTS.Output('results'),io.String.Output('tags',is_output_list=True),io.String.Output('results_json',is_output_list=True)])
    @classmethod
    def execute(cls,evidence,options=None,detectors=None,sources=None):
        configs=None
        if detectors is not None or sources is not None:
            configs=list(detectors[0]) if detectors else []
            configs += [DetectorConfig(s['name'],'sam3',**s.get('rules',{})) for s in (sources or [])]
        results=apply_filters(evidence,options,configs);tags,details=outputs(results)
        return io.NodeOutput(results,tags,details)


class WDMergeResults(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3MergeResults',display_name='WDV3 Merge Results',category='WDV3/Pipeline',
            inputs=[RESULTS.Input('left'),RESULTS.Input('right')],
            outputs=[RESULTS.Output('results'),io.String.Output('tags',is_output_list=True),io.String.Output('results_json',is_output_list=True)])
    @classmethod
    def execute(cls,left,right):
        result=merge_results(left,right);tags,details=outputs(result)
        return io.NodeOutput(result,tags,details)


class WDInspect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3Inspect',display_name='WDV3 Inspect',category='WDV3/Pipeline',is_output_node=True,
            description='Connect exactly one stage. Inspect raw scores, filtered decisions, regions and source attribution.',
            inputs=[RESULTS.Input('results',optional=True),EVIDENCE.Input('evidence',optional=True),REGIONS.Input('regions',optional=True)],
            outputs=[REPORT.Output('report'),io.Image.Output('images',is_output_list=True),io.String.Output('tags',is_output_list=True)])
    @classmethod
    def execute(cls,results=None,evidence=None,regions=None):
        from .comfy_reports import make_report,preview_report
        values=[v for v in (results,evidence,regions) if v is not None]
        if len(values)!=1: raise ValueError('Connect exactly one Inspect stage: results, evidence or regions')
        report=make_report(values[0]);rows,images,tags=preview_report(report,folder_paths.get_temp_directory())
        return io.NodeOutput(report,images,tags,ui={'wdv3_results':rows,'wdv3_total':[len(rows)]})


class WDExportReport(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3ExportReport',display_name='WDV3 Export Report',category='WDV3/Output',
            is_output_node=True,not_idempotent=True,
            inputs=[REPORT.Input('report'),io.String.Input('output_directory',default=str(Path(folder_paths.get_output_directory())/'wdv3')),
                    io.String.Input('prefix',default='report'),io.Boolean.Input('save_json',default=True),
                    io.Boolean.Input('save_html',default=True),io.Boolean.Input('save_txt',default=True)],
            outputs=[io.String.Output('json_path'),io.String.Output('html_path'),io.String.Output('txt_path')])
    @classmethod
    def execute(cls,report,output_directory,prefix,save_json,save_html,save_txt):
        from .comfy_reports import export_report
        paths=export_report(report,output_directory,prefix,save_json,save_html,save_txt)
        return io.NodeOutput(paths.get('json',''),paths.get('html',''),paths.get('txt',''),ui={'text':list(paths.values())})


NODE_CLASSES=[WDCollectEvidence,WDApplyFilters,WDMergeResults,WDInspect,WDExportReport]
