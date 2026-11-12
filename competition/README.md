# 比赛训练与交付代码

从本目录执行训练命令，详见[复现指南](../docs/reproduction.md)。

| 目录/入口 | 用途 |
| --- | --- |
| `train.py`、`glaucoma/` | E00等初始骨干、数据、ROI和训练实现 |
| `v2_experiment/train.py` | C00、正则与输入尺寸对照；最终C00来自该阶段 |
| `v3_experiment/run_gpu.py` | 显式指定GPU的Gamma、Age、EMA及OOF训练入口 |
| `v3_experiment/fit_model_stack.py` | 完整训练OOF的Logistic融合 |
| `experiments/E_AUC01/` | 排序损失微调的独立实验，未晋级最终方案 |
| `deployment/` | 最终结构、共享单图Predictor与严格计时脚本 |

配置保留实验超参数。原始epoch、真实样本分区、OOF预测、微调权重和比赛提交CSV不在公开仓库；应从合法数据与训练产物重新生成，不能用合成测试代替正式指标。
