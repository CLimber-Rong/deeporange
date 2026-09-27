# DeepOrange 本地训练与测试（Windows版）

这是 **Tk 图形界面程序**，入口为 `main.py`，训练和测试共用一个窗口。`base/trainer_*.py` 是训练库，不是需要单独运行的命令行脚本。训练任务使用冻结的 MobileNetV2 α=0.5 提取特征，再训练两类分类头；类别固定为 `橙子` 和 `非橙子`。

## 基于Windows的程序启动

1. 在本目录双击 `train.bat`。它会优先使用随项目附带的 `resources/runtime/python.exe`（Python 3.10.6）；如果该运行时不存在，则调用系统的 `py -3.10`。`predictor.bat` 也会打开同一个训练与测试窗口。
2. 若没有附带运行时，先安装 64 位 Python 3.10，并在本目录执行 `py -3.10 -m pip install -r requirements.txt`，再双击 `train.bat`。也可从任意位置执行 `D:\路径\本地训练\resources\runtime\python.exe D:\路径\本地训练\main.py`。
3. NVIDIA GPU 加速需要适配 TensorFlow 2.10 的 CUDA 11.2 和 cuDNN 8.1；未配置 GPU 时会使用 CPU，仍可训练。首次打开和首次提取特征可能需要等待。

本项目附带运行时中的版本为 TensorFlow 2.10.0、NumPy 1.26.4、Pillow 12.3.0、Matplotlib 3.10.9、scikit-learn 1.7.2。`requirements.txt` 已与它对齐；JAX 不是本程序依赖，Windows 也无需另装 `tensorflow-gpu` 包。

## 选择训练器与验证集

编辑本目录的 `trainer_config.json`，保存后重新启动程序：

```json
{"trainer": "original_full_data"}
```

| 值 | 代码 | 验证集与训练行为 |
| --- | --- | --- |
| `original`（默认） | `base/trainer_original.py` | 沿用原版 Adam、20 轮、批大小 16、学习率 0.001、100 隐藏单元和 0.2 dropout。数据量至少 20 张时按类别及采集组抽取约 15% 作验证，不参与训练。 |
| `original_full_data` | `base/trainer_original_full_data.py` | 继承原版参数与导出格式，只关闭验证集划分；所有输入图片都进入训练，结果中没有验证指标。需要保留全部配比时选此项。 |
| `optimized` | `base/trainer_optimized.py` | 可选 Adam、SGD、RMSprop、AdamW；同样划约 15% 验证集，并使用验证损失早停和检查点。 |

这些设置是代码中的本地实现。导出文件采用已有的 `tm-object-classifier` 元数据及 TensorFlow.js Layers 模型格式；实际官网上传流程仍需用导出模型做一次验证。`original_full_data` 没有内部验证准确率，若需评估请使用项目根目录独立的 `验证集/`、`测试集/`，勿把它们加入训练来源。

## 训练步骤

1. 启动后切到「训练」页。程序会优先找到上一级项目根目录，因为那里已有 `橙子/`、`非橙子/`；也可点「浏览」手选包含这两个子目录的父目录，或选择 `tm-object-project v1` 的 JSON。**不要选单个 `橙子/` 目录。**本目录 `resources/datasets/` 目前为空，无需复制训练图片。
2. 确认数据源、模型基底 `model/basemodels/model.json` 和配置中的训练器。建议先点「数据体检」检查类别和图片数量。训练页默认选择「标准训练」，对应原版的 20 轮、批大小 16、学习率 0.001、100 隐藏单元；选择「自定义」后才使用手填参数。原版只支持 Adam，界面相应只提供 Adam。
3. 点「开始训练」，等待特征提取、分类头训练和导出结束。运行日志会显示训练/验证样本数：全量版应显示验证 `0`。默认导出目录为 `model/models_keras/`，模型文件包括 `metadata.json`、`model.json`、`weights.bin` 和训练曲线。训练器还会在本目录生成 `橙子识别项目.识物模型.zip`。
4. 「测试」页可选择刚导出的模型目录或 ZIP，并选测试图片做本地评估。训练与测试互斥。`验证集/`、`测试集/` 是独立评估素材，程序不会自动纳入训练。

训练输出不会修改上一级项目根目录的 `橙子/`、`非橙子/`，也不会改动原始素材。训练前若要更新数据集，先按根目录 [README.md](../README.md) 运行数据生成脚本。
