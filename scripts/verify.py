"""Run every acceptance suite and write machine-readable results from pytest XML."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('outputs/acceptance'))
    parser.add_argument('--require-ml',action='store_true',help='Fail if the real HF/PEFT integration is skipped')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    suites=[]
    for test in sorted((root/'tests').glob('test_*.py')):
        xml=out/(test.stem+'.xml')
        started=time.monotonic()
        with (out/(test.stem+'.log')).open('w',encoding='utf-8') as log:
            proc=subprocess.run([sys.executable,'-m','pytest',str(test),'-q',f'--junitxml={xml}'],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=False)
        parsed=ET.parse(xml).getroot() if xml.exists() else None
        nodes=parsed.findall('testsuite') if parsed is not None else []
        stats={k:sum(int(s.get(k,'0')) for s in nodes) for k in ('tests','failures','errors','skipped')}
        suites.append({'suite':test.stem,'exit_code':proc.returncode,'seconds':round(time.monotonic()-started,3),**stats})
        print(json.dumps(suites[-1]),flush=True)
    counts={k:sum(s[k] for s in suites) for k in ('tests','failures','errors','skipped')}
    result={'environment':{'python':platform.python_version(),'platform':platform.platform()},'suites':suites,
            'counts':counts,'passed':counts['tests']-counts['failures']-counts['errors']-counts['skipped'],
            'hf_peft_integration':next((s for s in suites if s['suite']=='test_hf_integration'),None),
            'qwen_4b_inference_run':False,'cuda_training_run':False,'bird_benchmark_run':False,
            'note':'This command validates implementation, not pretrained model quality. Tiny architecture tests are not benchmark results.'}
    (out/'acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    failed=any(s['exit_code'] for s in suites)
    if args.require_ml and result['hf_peft_integration']['skipped']: failed=True
    return int(failed)

if __name__=='__main__': raise SystemExit(main())
