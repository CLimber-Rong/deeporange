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
| `original` | `base/trainer_original.py` | 沿用原版 Adam、20 轮、批大小 16、学习率 0.001、100 隐藏单元和 0.2 dropout。数据量至少 20 张时按类别及采集组抽取约 15% 作验证，不参与训练。 |
| `original_full_data` | `base/trainer_original_full_data.py` | 继承原版训练器与导出格式，只关闭验证集划分；所有输入图片都进入训练，结果中没有验证指标。需要保留全部配比时选此项。 |
| `optimized` | `base/trainer_optimized.py` | 可选 Adam、SGD、RMSprop、AdamW；同样划约 15% 验证集，并使用验证损失早停和检查点。 |
| `optimized_full_data` | `base/trainer_optimized_full_data.py` | 继承 `optimized` 的优化器选择、训练和导出方式，关闭内部验证集划分；所有输入图片参与训练，没有内部验证指标，也不会触发依赖验证损失的早停与检查点。 |
| `optimized_2` | `base/trainer_optimized_2.py` | 使用 Adam、并行解码及预取来提取特征；按类别及采集组划约 20% 内部验证集，使用验证损失早停、最佳权重检查点和学习率下调。非正方形图片等比缩放后在白底居中。 |
| `optimized_2_full_data` | `base/trainer_optimized_2_full_data.py` | 继承 `optimized_2` 的训练和导出方式，关闭内部验证集划分；所有输入图片参与训练，因此没有内部验证指标，也不会触发依赖验证损失的早停与学习率下调。 |

`trainer_config.json` 是必需的，程序会按文件中的 `trainer` 加载训练器；文件缺失或值错误时不会自动改选其他训练器。界面默认的「标准训练」预设为 20 轮、批大小 16、学习率 0.001、100 隐藏单元；「快速验证」为 10 轮、批大小 32；「精细训练」为 120 轮、批大小 8、学习率 0.0003、256 隐藏单元。`optimized` 和 `optimized_full_data` 还可选择「强正则」预设。只有「自定义」模式才按手填参数训练；选择 `full_data` 只改变内部验证集划分，不改变所选训练预设。

批量比较时，每次修改 `trainer_config.json` 后都要重启程序，再在界面选择训练预设。默认模型目录和 `橙子识别项目.识物模型.zip` 会被后续训练覆盖；每轮完成后，将 ZIP 复制为能辨认训练器、预设及轮次的不同文件名，再用同一独立验证集测试。`optimized_2` 及其全量版只支持 Adam，导出的模型会记录留白缩放方式；本程序的分类测试和单图/批量识别会按该方式处理图片，其他训练器仍按中心裁剪处理。浏览器平台若固定使用中心裁剪，上传此类模型后仍需单独验证其效果。

这些设置是代码中的本地实现。导出文件采用已有的 `tm-object-classifier` 元数据及 TensorFlow.js Layers 模型格式；实际官网上传流程仍需用导出模型做一次验证。三个 `full_data` 版本都没有内部验证准确率，若需评估请使用项目根目录独立的 `验证集/`、`测试集/`，勿把它们加入训练来源。

## 当前数据快照的批量对比

六种训练器与 flash、std、pro 三档位的 18 个模型 ZIP 存在 `model/deeporange-v1-*.zip`；每组验证结果存在 `result/<模型名>-result/`，包含逐样本 CSV、指标图表和已经运行批处理导出的错判图片。顶层结果沿用当前测试页的模型加载口径，`基底一致验证/` 使用训练时的 TFJS 特征提取图复测。每组的训练参数与实际轮次见结果目录中的 `训练记录.json`。原 4950 张验证图的结果归档于 `result/历史验证_4950/`。

`batch_compare.py` 是这次固定数据快照的命令行入口。在本目录运行 `resources/runtime/python.exe batch_compare.py all` 可依次准备冻结特征、训练和执行两套评估。脚本会核对训练集 1001/1011 张、验证集 100/100 张的快照，以及已生成模型和结果的哈希；数据或模型发生变化时应先归档旧产物。`.batch_cache/` 只放临时特征和中间文件，`all` 成功结束时会清理；单独运行阶段命令后，应在本轮工作结束时清理它和 `.batch_*.log`。选样清单、回收清单已存放在 `result/验证集筛选记录/`。当前验证集负样本按已有 18 个模型的两套结果精选，适合复核已知薄弱点；浏览器实际使用的特征提取基底仍需上传确认。

## 训练步骤

1. 启动后切到「训练」页。程序优先使用上一级项目根目录中的 `橙子/`、`非橙子/`；也可点「浏览」手选包含这两个子目录的父目录，或选择 `tm-object-project v1` 的 JSON。**不要选单个 `橙子/` 目录。**`resources/datasets/` 是备用数据源位置，无需为默认流程复制训练图片。
2. 确认数据源、模型基底 `model/basemodels/model.json` 和配置中的训练器。建议先点「数据体检」检查类别和图片数量。训练页默认选择「标准训练」，对应原版的 20 轮、批大小 16、学习率 0.001、100 隐藏单元；选择「自定义」后才使用手填参数。原版只支持 Adam，界面相应只提供 Adam。
3. 点「开始训练」，等待特征提取、分类头训练和导出结束。运行日志会显示训练/验证样本数：全量版应显示验证 `0`。默认导出目录为 `model/models_keras/`，本次写入的模型文件为 `metadata.json`、`model.json`、`weights.bin`。训练器会在本目录生成 `橙子识别项目.识物模型.zip`，ZIP 只包含上述三个模型文件；训练曲线及 `optimized_2` 有内部验证集时生成的验证评测 PNG 单独保存在本目录，文件名带唯一时间戳，不会随 ZIP 打包。界面上的「导出训练结果」另存训练摘要、曲线和数据体检报告，不是导出模型的必经步骤。
4. 「测试」页选择刚导出的模型目录或 ZIP。「图像分类」模式要选择包含**两个类别子目录**的父目录（例如根目录的 `验证集/`），再点「开始测试」；「单图/批量识别」模式可选单张图片或图片目录。训练与测试互斥，`验证集/`、`测试集/` 不会自动纳入训练。分类模式按子目录名排序确定类别顺序，并按该顺序解释模型输出列；程序只检查类别数量，因此两个子目录的顺序必须与模型 `metadata.json` 的标签顺序一致。本项目导出的标签顺序为「橙子、非橙子」。

分类测试完成后点「一键导出」，先选择一个存放报告的目录；程序在其中新建 `导出_分类_年月日_时分秒/`，包含指标文本、图表以及 `额外_逐样本预测结果/逐样本预测.csv`。CSV 的末列「文件名」对应每张测试图片。在 CSV 同目录还会生成 `导出预测错误图片.bat`；需要图片时再双击它，会按真实类别建立 `预测错误的橙子/` 和 `预测错误的非橙子/` 并复制两类错判图片。批处理中的源图片绝对路径是在导出时写入的，移动或删除原图片后需重新导出；再次运行会覆盖目标目录里的同名文件。此批处理只随**分类**报告生成，不会自动执行。

训练输出不会修改上一级项目根目录的 `橙子/`、`非橙子/`，也不会改动原始素材。训练前若要更新数据集，先按根目录 [README.md](../README.md) 运行数据生成脚本。

代码与目录边界：`train.bat` 和 `predictor.bat` 都启动 `main.py`；`main.py` 负责 Tk 界面、训练器选择、测试和报告导出，训练任务交给 `base/trainer_original.py`、`base/trainer_original_full_data.py`、`base/trainer_optimized.py`、`base/trainer_optimized_full_data.py`、`base/trainer_optimized_2.py` 或 `base/trainer_optimized_2_full_data.py`。训练器将目录或项目 JSON 加载为两类样本，图片经 224×224 预处理后，由冻结的 MobileNetV2 提取特征，再训练分类头；原版和 `optimized` 使用中心裁剪，`optimized_2` 使用留白缩放。三个全量版分别继承各自的基础训练器并把验证比例改为零。`model/basemodels/` 存放特征提取器，`model/models_keras/` 是默认分类头导出目录；`resources/runtime/` 是附带的 Windows Python 与依赖，`resources/datasets/` 是备用数据目录，`resources/testsets/` 是识别页默认指向的示例图片。`base/sdd.py` 供识别模式解压 ZIP；`base/predictor.py` 不由启动脚本单独调用。测试报告存到「一键导出」时所选位置，`result/` 并非固定输出路径。
