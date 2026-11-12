# 保持原整数像素运算的行并行源码

基于Pillow12.3.0。`.original`为上游原文，`.c`为修改后源码；版权和MIT-CMU许可证保持原文，完整许可见根目录`licenses/Pillow_LICENSE.txt`。

- `Resample.c`：只对8bit水平/垂直重采样的独立输出行并行，保留系数、22bit定点舍入及每像素累加顺序。
- `Crop.c`：新建且裁剪范围完全在源图内时，并行复制大图的独立行；其他路径保持原实现。
- 原交付加载器固定使用4线程；图片仍batch=1串行，不跨图自适应。

原构建为Linux x86_64 GCC/OpenMP，`-O3 -DNDEBUG -fopenmp`，没有fast-math或强制AVX。JPEG解码链接原Pillow的libjpeg-turbo3.1.4.1（libjpeg ABI62），分别编译CPython3.11、3.12；图像库和工具链不同可能改变像素或运行表现，不能仅凭同一源码假定完全一致。

公开仓库保留源码而不分发平台二进制；可自行从[Pillow上游源码](https://pypi.org/project/pillow/12.3.0/#files)覆盖同名文件后重建。原源码包SHA256为`3b8182a766685eaa002637e28b4ec8d6b18819a0c71f579bf0dbaa5830297cce`。任何新构建都应重新核验张量、概率、标签、排序和端到端耗时。拥有原交付运行时的用户可通过根目录导入脚本恢复该依赖；公开版默认标准Pillow仍能执行功能检查。
