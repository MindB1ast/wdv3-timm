"""Exercise the real ComfyUI RAM-pressure walker against an opaque Bundle."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
COMFY=Path(r'C:\Users\annak\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI')
sys.path[:0]=[str(ROOT),str(COMFY)]
sys.argv=['ram-sizing-acceptance','--cpu']
import comfy.options
comfy.options.enable_args_parsing()
from comfy_execution import caching
from Scripts.evidence import Bundle


def main():
    value=Bundle.pack('evidence',{'scores':{f'tag_{i}':.001 for i in range(10861)}})
    cache=caching.RAMPressureCache(caching.CacheKeySetInputSignature)
    cache.cache={'entry':SimpleNamespace(outputs=[[value]])}
    cache.used_generation={'entry':0};cache.generation=1;cache.timestamps={'entry':0};cache.children={}
    with patch.object(caching,'virtual_memory_available',return_value=0), \
         patch.object(Bundle,'data',side_effect=AssertionError('Comfy RAM scanning decoded the Bundle')) as decode:
        released=cache.ram_release(1)
    assert decode.call_count==0 and released>0 and not cache.cache
    report=dict(real_comfy_ram_walker=True,bundle_json_decodes=decode.call_count,entry_evicted=True)
    path=ROOT/'temp/postprocessing-ram-sizing.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
