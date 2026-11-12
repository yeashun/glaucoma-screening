"""Explicit manual entry point; no queue, polling, or automatic GPU launch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--enable-gpu',action='store_true')
    parser.add_argument('--gpu',help='Physical GPU allocated by the user; no default')
    parser.add_argument('--run',help='A prepared configuration name, for example P2_AGE_s42')
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--cache-root')
    parser.add_argument('--split-plan')
    parser.add_argument('--batch-size',type=int)
    args=parser.parse_args()
    if not args.enable_gpu or args.gpu is None or not args.run:
        parser.error('Training remains disabled until a GPU is allocated and --enable-gpu --gpu --run are explicit.')
    if not args.gpu.isdigit():parser.error('Specify exactly one physical GPU index')
    here=Path(__file__).resolve().parent
    config_path=here/'configs'/f'{args.run}.json'
    if not config_path.is_file():parser.error('Unknown prepared configuration')
    config=json.loads(config_path.read_text())
    protocol=json.loads((here/'protocol.json').read_text())
    prerequisite={'P2_AGE_s43':'P2_AGE_s42','P3_GAMMA_s43':'P3_GAMMA_s42','P5_C512_s42':'P5_C448_s42'}.get(args.run)
    if prerequisite:
        record=Path(args.output_root)/prerequisite/'complete.json'
        if not record.exists():parser.error(f'Complete and review {prerequisite} before this conditional run')
        score=json.loads(record.read_text())['best_metrics']
        baseline=protocol['baseline_validation_s42']
        if args.run=='P3_GAMMA_s43':passed=score['auroc']>baseline['auroc'] and score['ap']>baseline['ap']
        elif args.run=='P5_C512_s42':passed=score['auroc']>=baseline['auroc']+.0005 and score['ap']>=baseline['ap']-.01
        else:passed=score['auroc']>baseline['auroc'] and score['ap']>=baseline['ap']
        if not passed:parser.error('The preregistered validation condition for this follow-up run was not met')
    if args.batch_size:
        if config['effective_batch']%args.batch_size:parser.error('Physical batch must divide effective batch 32')
        config['batch_size']=args.batch_size
    output=Path(args.output_root)/args.run
    if args.split_plan:
        fold=json.loads(Path(args.split_plan).read_text())['outer_fold']
        output=output/f'fold{fold}'
    output.mkdir(parents=True,exist_ok=True)
    effective=output/'effective_config.json'
    if effective.exists() and json.loads(effective.read_text())!=config:
        raise ValueError('An existing run has a different configuration')
    effective.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    environment=os.environ.copy()
    environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=args.gpu,
                       OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    # No network weight download: reuse the project's existing pretrained cache.
    project=here.parent
    pretrained=project/'pretrained' if (project/'pretrained').is_dir() else project/'glaucoma_project/pretrained'
    environment['TORCH_HOME']=str(pretrained)
    weights=pretrained/'hub/checkpoints/convnext_tiny-983f1562.pth'
    if not weights.is_file():
        raise FileNotFoundError('Copy the frozen ImageNet pretrained checkpoint first; automatic download is disabled')
    command=[sys.executable,str(here/'train.py'),'--enable-gpu','--physical-gpu',args.gpu,
             '--config',str(effective),'--data-root',args.data_root,'--run',str(output)]
    if args.cache_root:command+=['--cache-root',args.cache_root]
    if args.split_plan:command+=['--split-plan',args.split_plan]
    raise SystemExit(subprocess.call(command,env=environment))


if __name__=='__main__':main()
