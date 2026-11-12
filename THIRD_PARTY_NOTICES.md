# Third-party notices

The MIT license at the repository root covers the project's original code and documentation. It does not relicense datasets, third-party packages, upstream model weights, or the Pillow-derived C files.

| Component | Use | License / notice |
| --- | --- | --- |
| TorchVision | ConvNeXt-Tiny and EfficientNet model builders and general ImageNet initialization | BSD-3-Clause; full text in `licenses/torchvision_LICENSE.txt` |
| Original ConvNeXt project | Architecture provenance | MIT; full text in `licenses/convnext_v1_LICENSE.txt` |
| Pillow | Image decoding, resizing and padding; modified resampling/crop source excerpts | MIT-CMU; full text in `licenses/Pillow_LICENSE.txt` |
| PyTorch, NumPy, SciPy, scikit-learn and remaining dependencies | Installed dependencies, not vendored | Their respective upstream licenses apply |

`competition/deployment/native_image_source/` preserves the original notices and the original and modified Pillow C files. The modifications schedule independent rows and copies; the pixel arithmetic is retained. No native shared libraries or model weights are distributed in this repository.

Official ImageNet weight URLs are downloaded from `download.pytorch.org`, with hash checking. Competition data, ACRIMA images, split manifests containing real sample identifiers, per-image labels and predictions, and trained artifacts are not redistributed here. Dataset access must be obtained from the original authorized source under its terms.
