"""共享单图推理接口；整集预测与耗时核验使用同一 Predictor。"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from image_backend import activate
IMAGE_BACKEND = activate()
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from model import FundusClassifier
from utils import configure, fuse, preprocess, read_csv, sha256, write_json

PACKAGE = Path(__file__).resolve().parents[1]


class Predictor:
    def __init__(self, package=PACKAGE, device='cuda:0'):
        configure()
        self.package, self.device = Path(package), torch.device(device)
        self.image_backend = IMAGE_BACKEND
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --device cpu only for a functional check')
        selection = json.loads((self.package / 'model/selection.json').read_text(encoding='utf-8'))
        stack_path = self.package / 'model/stack.json'
        if sha256(stack_path) != selection['stack_sha256']:
            raise ValueError('Frozen stacking parameter hash differs')
        self.stack = json.loads(stack_path.read_text(encoding='utf-8'))
        assert self.stack['model_order'] == ['C00', 'candidate']
        self.models = []
        for record in selection['models']:
            path = self.package / 'model' / f"{record['name']}_s42.pt"
            if sha256(path) != record['export_sha256']:
                raise ValueError(f'Frozen weight hash differs: {path.name}')
            state = torch.load(path, map_location='cpu', weights_only=False)
            assert state['config']['use_age'] is False and state['config']['image_size'] == 384
            model = FundusClassifier(state['config'], pretrained=False)
            model.load_state_dict(state['model'], strict=True)
            self.models.append(model.to(self.device).eval().requires_grad_(False))
        self.age = torch.zeros((1, 2), device=self.device)
        self.graph_forward = None
        if self.device.type == 'cuda' and os.environ.get('GLAUCOMA_DISABLE_CUDA_GRAPH') != '1':
            from graph_runtime import GraphForward
            self.graph_forward = GraphForward(self.models, self.age)

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict_path(self, path):
        # No image/probability cache: every call reads and decodes the source file.
        image = preprocess(path).unsqueeze(0).to(self.device)
        if self.graph_forward is not None:
            scores = self.graph_forward(image).cpu().numpy()
        else:
            with torch.autocast('cuda', dtype=torch.float16, enabled=self.device.type == 'cuda'):
                outputs = [model(image, self.age) for model in self.models]
            scores = torch.cat(outputs).float().sigmoid().cpu().numpy()
        probability = float(fuse(scores, self.stack))
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid probability')
        return probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, help='包含test.csv及test/的官方数据目录')
    parser.add_argument('--output', required=True)
    parser.add_argument('--package', default=str(PACKAGE))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, help='仅小样本功能检查，不能作为完整提交')
    args = parser.parse_args()
    data, output = Path(args.data_root), Path(args.output)
    if output.exists():
        raise FileExistsError('拒绝覆盖既有CSV，请选择新路径')
    rows = read_csv(data / 'test.csv')
    assert len(rows) == 5855 and len({r['Image_name'] for r in rows}) == 5855
    if args.limit is not None:
        assert 0 < args.limit <= len(rows)
        rows = rows[:args.limit]
    paths = {p.name.casefold(): p for p in (data / 'test').iterdir() if p.is_file()}
    assert all(row['Image_name'].casefold() in paths for row in rows)
    predictor = Predictor(args.package, args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.partial')
    started = time.monotonic()
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Image_name', 'Probability', 'Pred_Label'])
        for i, row in enumerate(rows):
            value = predictor.predict_path(paths[row['Image_name'].casefold()])
            writer.writerow([row['Image_name'], repr(value), int(value >= .5)])
            handle.flush()
            if (i + 1) % 100 == 0 or i + 1 == len(rows):
                print(f'{i + 1}/{len(rows)}', flush=True)
    temporary.replace(output)
    write_json(output.with_suffix('.metadata.json'), {'rows': len(rows), 'full_test': len(rows) == 5855,
               'batch_size': 1, 'precision': 'fp16' if predictor.device.type == 'cuda' else 'fp32',
               'submission_sha256': sha256(output), 'elapsed_s': time.monotonic() - started})


if __name__ == '__main__':
    main()
