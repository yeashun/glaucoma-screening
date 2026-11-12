# 共享推理与计时

`code/predict.py`和`cost.py`调用同一个Predictor；最终组合为两路ConvNeXt-Tiny与冻结Logistic融合。推理需要本地`model/C00_s42.pt`、`Gamma_s42.pt`、`selection.json`、`stack.json`，公开仓库不提供这些比赛训练产物。

请先从根目录使用`tools/import_private_artifacts.py --source /private/final-submission`，或自行生成同格式可信产物；模型与融合哈希由selection校验。`code/image_backend.py`在没有可选原生运行时的时候使用标准Pillow；不伪造或回填历史预测。

```bash
python code/predict.py --data-root /authorized/official --output /work/submission.csv
python cost.py submission.csv /authorized/pictures_100 /authorized/pictures_5 --output-dir /work/cost
```

80.203ms是原离线并行Pillow运行时、RTX4090及自选100图的历史辅助测量，不是标准Pillow路径或T4的保证。原生运行时只缓存依赖库和固定操作图，每次推理仍重新读取图片。
