"""5张预热，100张batch=1串行，连续3轮，计入CSV写出。"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from predict import Predictor
from utils import compare_probabilities, read_csv, sha256, write_json
import numpy as np
import torch
import PIL


def image_paths(folder, count):
    paths = sorted((p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')),
                   key=lambda p: p.name.casefold())
    if len(paths) != count or len({p.name.casefold() for p in paths}) != count:
        raise ValueError(f'{folder}必须包含{count}张文件名唯一的图像，实际{len(paths)}张')
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('submission')
    parser.add_argument('pictures_100')
    parser.add_argument('pictures_5')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default=str(ROOT / 'cost_output'))
    parser.add_argument('--demo', action='store_true', help='仅使用主办方公开10张示例开发脚本，不作正式100张验收')
    args = parser.parse_args()
    sample_count = 10 if args.demo else 100
    measured, warmup = image_paths(args.pictures_100, sample_count), image_paths(args.pictures_5, 5)
    assert not {p.name.casefold() for p in measured} & {p.name.casefold() for p in warmup}
    rows = read_csv(args.submission)
    assert len(rows) == 5855 and len({r['Image_name'].casefold() for r in rows}) == 5855
    reference = {r['Image_name'].casefold(): r for r in rows}
    assert all(p.name.casefold() in reference for p in measured)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'cost_predictions.csv').exists():
        raise FileExistsError('请使用新的输出目录，保留既有测时记录')
    predictor = Predictor(ROOT, args.device)
    for path in warmup:
        predictor.predict_path(path)
    predictor.synchronize()
    durations, probabilities, comparisons = [], [], []
    names = [reference[p.name.casefold()]['Image_name'] for p in measured]
    with (output / 'cost_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Round', 'Image_name', 'Probability', 'Pred_Label'])
        for repeat in range(3):
            times, values = [], []
            for name, path in zip(names, measured):
                predictor.synchronize()
                started = time.perf_counter()
                p = predictor.predict_path(path)
                writer.writerow([repeat + 1, name, repr(p), int(p >= .5)])
                handle.flush()
                predictor.synchronize()
                times.append((time.perf_counter() - started) * 1000)
                values.append(p)
            durations.append(times)
            probabilities.append(values)
            comparisons.append(compare_probabilities(names, values, reference))
            print(json.dumps({'round': repeat + 1, 'mean_ms': float(np.mean(times)),
                              'consistency': comparisons[-1]}, ensure_ascii=False), flush=True)
    per_image = np.asarray(durations).mean(axis=0)
    mean_ms = float(per_image.mean())
    inconsistent_names = sorted({n for c in comparisons for n in c['inconsistent_names']})
    device_name = torch.cuda.get_device_name(predictor.device) if predictor.device.type == 'cuda' else 'CPU'
    result = {'samples': sample_count, 'warmup': 5, 'rounds': 3, 'batch_size': 1,
              'development_demo': args.demo, 'official_protocol_100x3': not args.demo,
              'mean_ms': mean_ms, 'per_image_three_round_mean_ms': per_image.tolist(),
              'round_mean_ms': np.asarray(durations).mean(axis=1).tolist(),
              'gpu': device_name, 'is_t4': 'T4' in device_name,
              'precision': 'fp16' if predictor.device.type == 'cuda' else 'fp32',
              'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'pillow': PIL.__version__, 'pillow_module': PIL.__file__,
              'cuda_graph_enabled': predictor.graph_forward is not None,
              'image_backend': predictor.image_backend,
              'submission_sha256': sha256(args.submission), 'comparisons': comparisons,
              'inconsistent_images': len(inconsistent_names), 'inconsistent_names': inconsistent_names,
              'max_absolute_probability_error': max(c['max_absolute_probability_error'] for c in comparisons),
              'repeat_max_absolute_probability_difference': float(np.ptp(probabilities, axis=0).max()),
              'consistency_passed': not inconsistent_names,
              'within_100ms_on_this_host': mean_ms <= 100,
              'official_t4_acceptance': '以主办方指定环境复测为准，非T4结果不作T4达标证明',
              'timing_includes': ['read', 'decode', 'crop', 'resize', 'normalize', 'two_forwards', 'fusion', 'csv_write_flush']}
    write_json(output / 'cost_result.json', result)
    conclusion = (f'{"[DEMO] 仅脚本开发示例，非正式验收。" if args.demo else "[CONCLUSION]"} '
                  f'结论：{sample_count}张样本、batch=1串行、连续3轮；先按每张3轮取均值，再对{sample_count}张取平均；'
                  f'最终平均耗时 {mean_ms:.3f} ms；预测结果与submission.csv比对{sample_count}张，'
                  f'不一致 {len(inconsistent_names)} 张；最大概率绝对误差 '
                  f"{result['max_absolute_probability_error']:.9g}；推理精度 {result['precision']}；硬件 {device_name}。")
    print(conclusion, flush=True)
    (output / 'conclusion.txt').write_text(conclusion + '\n', encoding='utf-8')
    if inconsistent_names:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
