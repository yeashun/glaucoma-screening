# 复现指南

## 环境与可复现范围

历史正式训练使用Python3.11.17、torch2.14.1+cu130、torchvision0.29.1+cu130及RTX4090。比赛目标推理环境为Python3.12、torch2.10.0、torchvision0.25.0、CUDA12.8和T4。本仓库固定2.10/0.25作为CPU检查目标，不能把它解释为所有历史训练都在这个环境完成。

先安装与设备匹配的PyTorch组合，再安装`requirements-dev.txt`。不应让pip覆盖已有的正确CUDA构建。没有真实数据、训练权重时可以运行根目录合成测试和ACRIMA合成测试；不能直接得到官网98.13分。

## 比赛数据目录

取得主办方授权后，在本地准备：

```text
competition/data/official/
  train.csv
  valid.csv
  test.csv
  train/ ...
  valid/ ...
  test/ ...
```

训练与验证CSV含`Image_name,True_Label,Age`接口，测试按公开任务说明提供文件名与可用元数据。实际文件名和字段以授权数据为准。数据、运行产物、权重、实际清单和预测均被`.gitignore`排除。

以下命令从`competition/`执行。数据根目录也可通过各训练入口的`--data-root`显式指定；CPU审计工具使用`GLAUCOMA_DATA_ROOT`覆盖默认目录。

```bash
python tools/download_weights.py
python v2_experiment/build_cache.py --data-root data/official \
  --cache-root cache/384 --size 384 --workers 4
CUDA_VISIBLE_DEVICES=0 python v2_experiment/train.py \
  --config configs/C00_s42.json --data-root data/official \
  --cache-root cache/384 --run runs/C00_s42
python v3_experiment/run_gpu.py --enable-gpu --gpu 0 \
  --run P3_GAMMA_s42 --data-root data/official \
  --cache-root cache/384 --output-root runs/V3
```

GPU由运行者指定，仓库不会连接服务器或自动选择其他GPU。缓存只用于等价的确定性图像阶段，随机训练增强仍在线执行；更改输入尺寸需要从原图重建缓存。

## 审计、OOF与融合

```bash
python v3_experiment/audit_images.py --workers 4
python v3_experiment/analyze_audit.py
python v3_experiment/prepare_folds.py
```

审计结果写入忽略的`results/`目录。感知近似只是复核候选，不能直接当患者ID或修改标签的依据。`prepare_folds.py`只用训练样本及训练内重复风险边形成五个外折，每个外折另有内部验证。

对C00、Gamma分别运行5个外折。下面仅演示fold0，其余折按各自计划运行：

```bash
python v3_experiment/run_gpu.py --enable-gpu --gpu 0 \
  --run OOF_C00_s42 --data-root data/official --output-root runs/V3 \
  --cache-root cache/384 --split-plan results/V3_0/cpu/OOF_plans/fold0.json
python v3_experiment/predict_oof.py --help
python v3_experiment/fit_model_stack.py \
  --oof-root C00=runs/V3/OOF_C00_s42 \
  --oof-root candidate=runs/V3/OOF_P3_GAMMA_s42 \
  --valid-predictions C00=runs/C00_s42/best_valid.csv \
  --valid-predictions candidate=runs/V3/P3_GAMMA_s42/best_valid.csv \
  --data-root data/official --output results/C00_Gamma_stack
```

外折训练后须使用`predict_oof.py`导出各自外折预测，并保留生成的metadata；本节不提供官方数据的既有分区清单。融合入口要求五折覆盖、checkpoint与分区来源核对通过，不能用官方验证预测冒充训练OOF。

## 私有训练产物与推理

仓库不分发比赛训练后的权重。拥有原始授权交付目录的运行者，可在仓库根目录执行：

```bash
python tools/import_private_artifacts.py --source /private/final-submission
# 若同时持有原交付的可再分发图像处理运行时，可显式导入它：
python tools/import_private_artifacts.py --source /private/final-submission --native-runtime
python competition/deployment/code/predict.py \
  --data-root /authorized/official --output /work/new_submission.csv
```

脚本只复制本地已有授权产物，校验微调权重与冻结融合SHA；不连接服务器、不下载比赛权重。产物放入忽略目录。只加载自己训练或可信来源的checkpoint。

公开版默认使用系统Pillow。原生行并行C源码、原文及许可证位于`competition/deployment/native_image_source/`；已授权的原离线运行时可通过上面的显式选项恢复。没有匹配的原生构建与硬件环境时，不保证达到报告的80.203ms。

## 规则计时

```bash
cd competition/deployment
python cost.py submission.csv /authorized/pictures_100 /authorized/pictures_5 \
  --output-dir /work/cost_check
# 公开示例只有10张时，仅做开发检查：
python cost.py submission.csv /authorized/demo_10 /authorized/pictures_5 \
  --demo --output-dir /work/cost_demo
```

默认严格模式要求100张和5张不重叠的预热图。图像串行处理，执行3轮；读取至写出均计时，加载/初始化/预热除外。cost与predict共用加载和预测路径，不一致会返回失败。主办方隐藏100图不向选手公开，开发示例或自选100图不能代替正式T4验收。

## 独立ACRIMA模块

见[该模块说明](../research/acrima/README.md)。必须从其独立目录执行；两边具有同名Python包，不要将两个包目录同时加入PYTHONPATH。

## 公开代码整理说明

原工作目录的服务器运维脚本、GPU排队/切卡记录、真实数据清单、PDF/XLSX提交文件和二进制依赖均未纳入公开仓库。保留了主要训练和推理实现，统一为UTF-8，修复旧基线入口的优化器缩进；CPU审计数据路径改为可配置。源码来源和发布SHA见[source-provenance.json](source-provenance.json)。
