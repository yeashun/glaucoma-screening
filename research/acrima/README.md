# 独立ACRIMA研究基线

这是一套与比赛项目隔离的研究流程，来自另一份项目目录。**ACRIMA没有进入比赛模型的训练、医学预训练或融合拟合。** 不应把本模块结果与比赛官网分数直接比较。

实现冻结EfficientNet-B0特征＋Logistic回归、B0分阶段微调和轻度增强三种方案。训练内部固定分组三折用于模型/超参数/epoch选择；独立校准集拟合温度和操作阈值，最终测试在选择和校准后执行。相似性风险组不是患者ID，患者级独立性尚未验证。

原固定划分的训练/校准/测试规模为493/106/106。三折选择E1（冻结B0＋Logistic，C=1.0）；独立测试AUROC0.984493、梯形PR-AUC0.988731、AP0.988819。校准后NLL0.198045、ECE0.027537。仅保留[聚合结果](../../docs/results/acrima.json)，不发布图像、实际分区清单或模型。

## 安装与运行

先按仓库根目录说明安装依赖，然后在本目录运行。自行取得原数据源的合法使用权限，准备数据清单，格式由`glaucoma/data.py`和`glaucoma/protocol.py`定义；配置中`data.root`指定本地整理目录。

```bash
cd research/acrima
python -m glaucoma doctor
python -m glaucoma prepare
CUDA_VISIBLE_DEVICES=0 python -m glaucoma cv
CUDA_VISIBLE_DEVICES=0 python -m glaucoma fit
CUDA_VISIBLE_DEVICES=0 python -m glaucoma calibrate
CUDA_VISIBLE_DEVICES=0 python -m glaucoma evaluate
```

默认配置`configs/acrima_4090.json`。首次训练使用TorchVision公开ImageNet权重；可通过配置中的`weights_path`指定本地原始预训练文件。最终测试有产物指纹与一次测试锁；变动模型或校准参数时不要复用既有测试目录搜索结果。

完成fit和calibrate后，可对新的无标签图像运行：

```bash
python -m glaucoma predict --images /authorized/image.jpg --predictions /work/predictions.csv
```

## 合成测试

```bash
python -m pytest -q
```

原项目合成测试随代码保留：临时创建随机图像和随机权重，覆盖指标、协议、分组、校准、冻结BN、样本保留和端到端流程；不需要真实数据或联网下载，不会产生ACRIMA科学实验结论。`requirements-dev.txt`包含测试依赖。
