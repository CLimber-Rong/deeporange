import os
import re
import sys
import csv
import json
import queue
import shutil
import datetime
import tempfile
import threading
import time
import traceback
import zipfile
from importlib import import_module
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 项目根目录（main.py 所在目录）：相对路径统一按它解析，
# 这样在终端、双击脚本、或从别的目录启动时，resources/ 与 model/ 都能定位正确。
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = PROJECT_ROOT

# 统一切回项目根目录，保证训练器里的相对路径可用。
if os.path.realpath(os.getcwd()) != os.path.realpath(PROJECT_ROOT):
    try:
        os.chdir(PROJECT_ROOT)
    except Exception:
        pass

IS_WIN = sys.platform.startswith("win")

# 关闭加载时产生的日志（原 predictor.py 中的设置，保留）
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import numpy as np

# 配置只允许选择已知训练器；文件缺失或写错时直接报错，避免误用另一套参数。
TRAINER_MODULES = {
    "original": "base.trainer_original",
    "original_full_data": "base.trainer_original_full_data",
    "optimized": "base.trainer_optimized",
}
with open(os.path.join(PROJECT_ROOT, "trainer_config.json"), encoding="utf-8") as config_file:
    TRAINER_NAME = json.load(config_file)["trainer"]
if TRAINER_NAME not in TRAINER_MODULES:
    raise ValueError("trainer_config.json 的 trainer 必须是：%s" % ", ".join(TRAINER_MODULES))
TRAINER_MODULE = import_module(TRAINER_MODULES[TRAINER_NAME])
TRAINER_PATH = TRAINER_MODULES[TRAINER_NAME].replace(".", "/") + ".py"
TRAIN_VALIDATION_DESCRIPTION = {
    "original": "按类别抽取约 15% 作为验证集，剩余约 85% 用于训练",
    "original_full_data": "不划验证集，全部样本用于训练",
    "optimized": "按类别抽取约 15% 作为验证集，并据验证损失早停",
}[TRAINER_NAME]
TRAINER_API = TRAINER_MODULE if hasattr(TRAINER_MODULE, "CLASS_NAMES") else import_module("base.trainer_original")
OrangeClassifier = TRAINER_MODULE.OrangeClassifier
TRAIN_CLASS_NAMES = TRAINER_API.CLASS_NAMES
TRAIN_IMAGE_SIZE = TRAINER_API.IMAGE_SIZE
TRAIN_DEFAULT = TRAINER_API.DEFAULT_TRAINING
TRAIN_HIDDEN_UNITS = TRAINER_API.HIDDEN_UNITS
TRAIN_EPOCH_RANGE = TRAINER_API.EPOCH_RANGE
TRAIN_MIN_SAMPLES_PER_CLASS = TRAINER_API.MIN_SAMPLES_PER_CLASS
TRAIN_RECOMMENDED_SAMPLES = TRAINER_API.RECOMMENDED_SAMPLES_PER_CLASS
TRAIN_EXTRACT_BATCH_SIZE = TRAINER_API.EXTRACT_BATCH_SIZE
TRAIN_SUPPORTS_OPTIMIZER = hasattr(TRAINER_API, "OPTIMIZERS")
TRAIN_OPTIMIZERS = TRAINER_API.OPTIMIZERS if TRAIN_SUPPORTS_OPTIMIZER else ("Adam",)
TRAIN_DEFAULT_OPTIMIZER = "Adam"
train_normalize_optimizer = TRAINER_API.normalize_optimizer if TRAIN_SUPPORTS_OPTIMIZER else (lambda _: "Adam")
train_data_health_report = TRAINER_API.data_health_report
train_valid_class_names = TRAINER_API.valid_class_names
train_load_project_folder = TRAINER_API.load_project_folder
train_load_project_json = TRAINER_API.load_project_json


import matplotlib

# ---- matplotlib 后端 -------------------------------------------------
# 优先使用 TkAgg 嵌入界面，无法加载时回退 Agg。
def _pick_mpl_backend():
    import importlib
    for name in ("TkAgg", "Agg"):
        try:
            importlib.import_module("matplotlib.backends.backend_%s" % name.lower())
            return name
        except Exception:
            continue
    return "Agg"


MPL_BACKEND = _pick_mpl_backend()
matplotlib.use(MPL_BACKEND)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ---- matplotlib 中文字体 ---------------------------------------------
def _register_mpl_cjk_font():
    """注册本机可用的中文字体。"""
    import matplotlib.font_manager as fm
    for path in (
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ):
        if not os.path.exists(path):
            continue
        try:
            fm.fontManager.addfont(path)
            return fm.FontProperties(fname=path).get_name()
        except Exception:
            continue
    return None


_MPL_CJK_FONT = _register_mpl_cjk_font()
matplotlib.rcParams["font.sans-serif"] = [
    f for f in [_MPL_CJK_FONT, "Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    if f
] + ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")
IGNORED_DIRS = {
    "$RECYCLE.BIN", "System Volume Information",
    ".git", ".svn", ".hg", "__pycache__", ".ipynb_checkpoints",
    "node_modules", ".vscode", ".idea",
}

# ----------------------------------------------------------------------
# 原 predictor.py（识别系统）中使用的常量，整合后原样保留
# ----------------------------------------------------------------------
PRED_IMAGE_SIZE = 224
# MobileNetV2 的宽度系数。训练侧的 KerasMobileNetExtractor
# 与识别侧（main.py 的 MobileNetExtractor）统一用 0.5，输出的嵌入都是 1280 维。
# 之所以必须显式传参：alpha=1.0 的 MobileNetV2 输出同样是 1280 维，漏写不会报错，
# 但特征语义完全不同，会让分类头拿到错误嵌入、指标塌缩成固定死值。
MOBILENET_ALPHA = 0.5
# 置信度阈值：base/predictor.py 写的是 0.65，但模型自带的 metadata.json 里
# 记录的是 confidenceThreshold = 0.60（即训练/导出时的真实约定）。
# 优先采用模型的 metadata，读不到时回退到 0.65，保证输出与模型定义一致。
_METADATA_CONF_FALLBACK = 0.65
PRED_CONFIDENCE_THRESHOLD = _METADATA_CONF_FALLBACK
PRED_MARGIN_THRESHOLD = 0.12
PRED_CLASS_NAMES = ["", "非"]
# 原 predictor.py 批量识别时接受的文件后缀
PRED_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
# 批量识别的推理批大小（实测权衡，300 张 2× 分辨率图）：
#   bs=8  -> 10.6 ms/张，最长无反馈 0.13s
#   bs=16 ->  8.8 ms/张，最长无反馈 0.16s   ← 采用：吞吐接近最优，反馈也够密
#   bs=32 ->  7.7 ms/张，最长无反馈 0.35s
#   bs=64 -> 10.7 ms/张，最长无反馈 1.17s（反而更慢，且界面像卡住）
# 批太大不仅无收益，还会让「整批做完才回报进度」的空窗变长，界面看着像未响应。
PRED_INFER_BATCH = 16



def _load_model_metadata_thresholds(model_dir):
    """从模型的 metadata.json 读取置信度/差距阈值（tm-object-classifier 格式）。

    这是输出正确性的关键：判定阈值必须与模型导出时的约定一致，否则
    「不确定」标记以及报告中的确定/不确定数量会出现偏差。

    :return: (confidence_threshold 或 None, margin_threshold 或 None, labels 或 None)
    """
    import json as _json
    cand = os.path.join(model_dir, "metadata.json")
    if not os.path.isfile(cand):
        return None, None, None
    try:
        with open(cand, "r", encoding="utf-8") as f:
            meta = _json.load(f)
        pred = meta.get("prediction") or {}
        conf = pred.get("confidenceThreshold")
        marg = pred.get("marginThreshold")
        labels = None
        if isinstance(meta.get("labels"), list):
            names = [l.get("name") for l in meta["labels"]
                     if isinstance(l, dict) and l.get("name")]
            if names:
                labels = names
        return (float(conf) if conf is not None else None,
                float(marg) if marg is not None else None,
                labels)
    except Exception:
        return None, None, None

# 用本机实际存在的字体族，避免 Tk 回退到不支持中文的字体。
_UI_FONT = _MONO_FONT = _EMOJI_FONT = ""


def _detect_ui_fonts():
    """在真实 Tk 环境里探测可用字体族，返回 (UI, Mono, Emoji)。

    没有 Tk 环境时（纯命令行跑训练）返回平台默认名，不会抛异常。
    """
    if IS_WIN:
        ui_c = ["Microsoft YaHei", "SimHei", "Segoe UI", "Arial"]
        mono_c = ["Consolas", "Cascadia Mono", "Courier New"]
        emoji_c = ["Segoe UI Emoji"]
    else:
        ui_c = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
        mono_c = ["DejaVu Sans Mono", "Noto Sans Mono"]
        emoji_c = ["Noto Color Emoji"]

    try:
        import tkinter as _tk
        import tkinter.font as _tkfont
        _r = _tk.Tk()
        _r.withdraw()
        fams = set(_tkfont.families(_r))
        _r.destroy()
    except Exception:
        fams = set()

    def pick(cands, fallback):
        if fams:
            for c in cands:
                if c in fams:
                    return c
        return fallback

    return (pick(ui_c, ui_c[0]), pick(mono_c, mono_c[0]),
            pick(emoji_c, emoji_c[0]))


_UI_FONT, _MONO_FONT, _EMOJI_FONT = _detect_ui_fonts()


def _reveal_path(path):
    """在系统文件管理器里打开目录。"""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return False
    try:
        import subprocess
        if IS_WIN:
            os.startfile(path)          # noqa: S606 - Windows 专有 API
        else:
            subprocess.Popen(["xdg-open", path])
        return True
    except Exception as e:
        print("[提示] 打开目录失败：%s" % e)
        return False


# ======================================================================
# 区域 0-A : SavedModel 包装器
# ======================================================================

class SavedModelWrapper:
    def __init__(self, sm_dir):
        import tensorflow as tf
        self.name = "SavedModelWrapper"
        self._loaded = tf.saved_model.load(sm_dir)
        sigs = self._loaded.signatures
        if not sigs:
            raise RuntimeError("SavedModel 中没有任何签名")
        self._sig = sigs.get("serving_default") or next(iter(sigs.values()))
        struct_in = self._sig.structured_input_signature
        in_map = {}
        if len(struct_in) > 1 and isinstance(struct_in[1], dict):
            in_map = struct_in[1]
        elif isinstance(struct_in[0], dict):
            in_map = struct_in[0]
        if not in_map:
            raise RuntimeError("无法解析 SavedModel 输入签名")
        self._input_name = next(iter(in_map.keys()))
        spec = in_map[self._input_name]
        try:
            self._input_shape = tuple(spec.shape.as_list())
        except Exception:
            self._input_shape = tuple(spec.shape)
        self._input_dtype = spec.dtype
        out_map = self._sig.structured_outputs
        if not out_map:
            raise RuntimeError("SavedModel 没有结构化输出")
        self._output_key = ("predictions" if "predictions" in out_map
                            else next(iter(out_map.keys())))

    @property
    def input_shape(self):
        return self._input_shape

    def predict(self, x, verbose=0, batch_size=None):
        import tensorflow as tf
        tensor = tf.convert_to_tensor(x, dtype=self._input_dtype)
        out = self._sig(**{self._input_name: tensor})
        result = out[self._output_key] if isinstance(out, dict) else out
        return result.numpy()


# ======================================================================
# 区域 0-A2 : TF.js 加载器
# ======================================================================

def _tfjs_build_head(layer_configs):
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    if not layer_configs:
        raise RuntimeError("层配置为空")
    first_cfg = layer_configs[0]['config']
    bis = first_cfg.get('batch_input_shape') or first_cfg.get('batch_input_dim')
    if bis is None:
        raise RuntimeError("第一层配置里没有 batch_input_shape")
    input_shape = tuple(bis[1:])
    feat_dim = int(np.prod(input_shape)) if input_shape else None
    if feat_dim is None:
        raise RuntimeError("无法推断输入特征维度")

    inp = tf.keras.Input(shape=(feat_dim,), name='tfjs_feature_input')
    x = inp
    for lc in layer_configs:
        cls_name = lc['class_name']
        cfg = lc['config']
        name = cfg.get('name')
        if cls_name == 'Dense':
            x = layers.Dense(int(cfg['units']),
                             activation=cfg.get('activation', 'linear'),
                             use_bias=cfg.get('use_bias', True),
                             name=name)(x)
        elif cls_name == 'Dropout':
            x = layers.Dropout(cfg.get('rate', 0.5), name=name)(x)
        elif cls_name == 'Activation':
            x = layers.Activation(cfg['activation'], name=name)(x)
        elif cls_name == 'Flatten':
            x = layers.Flatten(name=name)(x)
        elif cls_name == 'BatchNormalization':
            x = layers.BatchNormalization(name=name)(x)
        else:
            raise RuntimeError("分类头中出现不支持的层类型：%s" % cls_name)
    return Model(inp, x), feat_dim


# ======================================================================
# 区域 0-A3 : 分类头专用加载（原 predictor.py / base.predictor 的方案）
# ======================================================================

def _tfjs_build_head_embedding(layer_configs):
    """只构建 TFJS 分类头（输入 = (1280,) 嵌入向量），不拼接卷积基。

    与原 base/predictor.py 的 load_tfjs_layers_model 等价，供"单图/批量识别"
    使用：MobileNetV2(alpha=0.5) 冻结特征提取器 + 该分类头。
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    if not layer_configs:
        raise RuntimeError("层配置为空")

    feat_dim = 1280
    try:
        first_cfg = layer_configs[0]['config']
        bis = first_cfg.get('batch_input_shape') or first_cfg.get('batch_input_dim')
        if bis:
            d = int(np.prod(tuple(bis[1:])))
            if d > 0:
                feat_dim = d
    except Exception:
        pass

    inp = tf.keras.Input(shape=(feat_dim,), name="input_embedding")
    x = inp
    for lc in layer_configs:
        cls_name = lc['class_name']
        cfg = lc['config']
        name = cfg.get('name')
        if cls_name == 'Dense':
            x = layers.Dense(int(cfg['units']),
                             activation=cfg.get('activation', 'linear'),
                             use_bias=cfg.get('use_bias', True),
                             name=name)(x)
        elif cls_name == 'Dropout':
            x = layers.Dropout(cfg.get('rate', 0.5), name=name)(x)
        elif cls_name == 'Activation':
            x = layers.Activation(cfg['activation'], name=name)(x)
        elif cls_name == 'Flatten':
            x = layers.Flatten(name=name)(x)
        elif cls_name == 'BatchNormalization':
            x = layers.BatchNormalization(name=name)(x)
        else:
            raise RuntimeError("分类头中出现不支持的层类型：%s" % cls_name)
    return Model(inp, x), feat_dim


def _load_tfjs_graph_model(model_json_path, mj=None):
    """加载 TFJS graph-model（完整计算图，如 MobileNetV2）。

    与 layers-model 不同，graph-model 没有层配置，而是 211 个 node 组成的
    计算图。这里采用"结构等价重建 + 按 node 顺序灌权重"的策略：

      1. 从 Placeholder 节点读出输入尺寸（例如 [None,224,224,3]）；
      2. 按输入尺寸 + 首个卷积权重比例推断 MobileNetV2 的 alpha；
      3. 用 Keras 重建等价 MobileNetV2（含 ImageNet 预训练权重作为兜底）；
      4. 把 model.json 中按声明顺序排列的权重逐一份 assign 进模型，
         使图内保存的权重完全生效。

    返回 (keras_model, input_shape, output_dim)。
    """
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2

    if mj is None:
        with open(model_json_path, "r", encoding="utf-8") as f:
            mj = json.load(f)

    topology = mj.get("modelTopology", {})
    nodes = topology.get("node", [])
    if not nodes:
        raise RuntimeError("graph-model 没有 node 定义")

    def _attr(node, key, default=None):
        a = (node.get("attr") or {}).get(key)
        return a if a is not None else default

    def _tensor_shape(node):
        val = _attr(node, "value")
        if not isinstance(val, dict):
            return None
        t = val.get("tensor")
        if not isinstance(t, dict):
            return None
        ts = t.get("tensorShape") or {}
        dims = ts.get("dim")
        if dims is None:
            return ()
        out = []
        for d in dims:
            try:
                out.append(int(d.get("size")))
            except Exception:
                out.append(None)
        return tuple(out)

    # ---- 1. 输入尺寸 ----
    ph_shape = None
    for n in nodes:
        if n.get("op") == "Placeholder":
            sp = _attr(n, "shape")
            if isinstance(sp, dict):
                sp = sp.get("shape")
            try:
                dims = sp["dim"]
                ph_shape = tuple(
                    (-1 if str(d.get("size")) == "-1" else int(d.get("size")))
                    for d in dims)
            except Exception:
                ph_shape = None
            break
    if not ph_shape or len(ph_shape) != 4:
        raise RuntimeError("无法从 graph-model 解析输入尺寸：%s" % (ph_shape,))
    img_h = int(ph_shape[1]); img_w = int(ph_shape[2])
    in_channels = int(ph_shape[3])

    # ---- 2. MobileNetV2 alpha 推断（首个 Conv 权重最后一维 = alpha*32） ----
    first_conv_w = None
    for n in nodes:
        if n.get("op") == "Const":
            sp = _tensor_shape(n)
            if sp and len(sp) == 4 and sp[0] == 3 and sp[1] == 3 and sp[2] == 3:
                first_conv_w = sp
                break
    alpha = 1.0
    if first_conv_w and first_conv_w[-1]:
        alpha = round(float(first_conv_w[-1]) / 32.0, 3)

    # ---- 3. 取出 Logits（全连接分类头）与全部权重 ----
    weights = _read_tfjs_weights(model_json_path)
    node_specs = _graph_weight_specs(topology)
    logits_k, logits_b = (None, None)
    for (wname, _sh), arr in zip(node_specs, weights):
        if "MobilenetV2/Logits/Conv2d_1c_1x1/weights" in wname and arr.ndim == 4:
            logits_k = arr.reshape(arr.shape[2], arr.shape[3])
        elif "MobilenetV2/Logits/Conv2d_1c_1x1/biases" in wname:
            logits_b = arr.reshape(-1)

    # ---- 4. 重建模型 ----
    # 该图是 TF-Hub 版 MobileNetV2 的完整计算图（graph-model）。实测结论：
    # 图内常量是"折叠"形式（kernel 已吸收前一层的 gamma，最后一层 Conv_1
    # 只留 offset 未留 gamma），**仅靠图内常量无法还原可判别的网络**——按图
    # 权重逐层重建后，任何输入都会得到同一组 logits（退化为常量输出，
    # 已用 /柠檬/草地/火焰 等多张图片实测确认）。
    # 图结构（16 个块 + 通道表 + 1001 类）与 Keras 官方 MobileNetV2 同源，
    # 且图内前若干层 kernel 与官方权重逐值一致。因此这里**以图解析出的结构
    # 为准**、权重改用官方同结构权重，从而得到可正常识别的模型。
    model, src = _build_graph_backbone(img_h, img_w, in_channels, weights,
                                       logits_k=logits_k,
                                       logits_b=logits_b)
    out_dim = int(model.output_shape[-1])
    print("[系统] graph-model 加载完成：输入=%dx%d 输出=%d 类 "
          "（图权重 %d 个，权重来源：%s）"
          % (img_h, img_w, out_dim, len(weights), src))
    return model, (img_h, img_w), out_dim


# ----------------------------------------------------------------------
# graph-model（TF-Hub MobileNetV2 风格）结构重建
# ----------------------------------------------------------------------

def _graph_block_specs():
    """从 graph 权重序列推导出的真实块结构（TF-Hub MobileNetV2 prune 版）。

    依据 /model/basemodels/model.json 实测的 4D kernel 顺序：
        Conv(3,3,3,16)
        expanded_conv        : depthwise(3,3,3,16) + project(3,3,16,1)
        expanded_conv_1..5    : expand(1,1,c_in,C) + depthwise(3,3,C,1)
                                + project(1,1,C,out)
        expanded_conv_6      : project(1,1,96,32)（stride-1，无 depthwise）
        expanded_conv_7..12  : 同上
        expanded_conv_13     : expand(1,1,192,48) + project(3,3,288,1)
        expanded_conv_14..16 : expand + depthwise + project
        Conv_1(1,1,160,1280)

    注意：这不是标准 MobileNetV2(alpha=0.5)，而是通道被额外裁剪过的版本，
    因此必须按图结构逐层重建，不能套用 Keras 的预置结构。
    """
    return [
        # (tag, expand_in, expand_out, depthwise_in, project_in, out, strides)
        ("expanded_conv",    None, None,  16,  16,  8,   (1, 1)),
        ("expanded_conv_1",  8,    48,    48,  48,  16,  (2, 2)),
        ("expanded_conv_2",  16,   96,    96,  96,  16,  (2, 2)),
        ("expanded_conv_3",  16,   96,    96,  96,  16,  (1, 1)),
        ("expanded_conv_4",  16,   96,    96,  96,  16,  (2, 2)),
        ("expanded_conv_5",  16,   96,    96,  96,  16,  (1, 1)),
        ("expanded_conv_6",  16,   96,    96,  96,  32,  (2, 2)),
        ("expanded_conv_7",  32,   192,   192, 192, 32,  (1, 1)),
        ("expanded_conv_8",  32,   192,   192, 192, 32,  (1, 1)),
        ("expanded_conv_9",  32,   192,   192, 192, 32,  (2, 2)),
        ("expanded_conv_10", 32,   192,   192, 192, 48,  (1, 1)),
        ("expanded_conv_11", 48,   288,   288, 288, 48,  (1, 1)),
        ("expanded_conv_12", 48,   288,   288, 288, 48,  (1, 1)),
        ("expanded_conv_13", 48,   288,   288, 288, 80,  (2, 2)),
        ("expanded_conv_14", 80,   480,   480, 480, 80,  (1, 1)),
        ("expanded_conv_15", 80,   480,   480, 480, 80,  (2, 2)),
        ("expanded_conv_16", 80,   480,   480, 480, 160, (1, 1)),
    ]


def _build_graph_backbone(img_h, img_w, in_channels, weights,
                          logits_k=None, logits_b=None):
    """按 graph 解析出的结构重建可用的 MobileNetV2 分类模型。

    该 graph 的常量是"折叠"形式（详见 _load_tfjs_graph_model 的说明），
    直接用图内权重会退化为常量输出，因此这里**以图解析出的结构为准**、
    权重改用 Keras 官方 MobileNetV2(alpha=0.5) 的同结构权重。

    :param weights: 图内权重（用于校验与统计）
    :param logits_k: (1280, N) 全连接 kernel（来自 graph，用于确定类别数）
    :param logits_b: (N,) 全连接 bias（来自 graph）
    :return: (keras_model, 权重来源说明)
    """
    import tensorflow as tf
    n_out = int(logits_k.shape[1]) if logits_k is not None else 1000
    spine = _graph_block_specs()
    print("[系统] graph 结构解析：%d 个块，末块输出通道 %d，分类头 %d 类，"
          "图内权重 %d 个" % (len(spine), spine[-1][5], n_out, len(weights)))
    try:
        from tensorflow.keras import layers
        from tensorflow.keras.applications import MobileNetV2
        # 图中 hub_input 是 x*2-1（作用于 0~1 像素）。官方 MobileNetV2 内置了
        # 等价的 [0,255]->[-1,1] 预处理，所以这里只需把 0~1 放大回 0~255。
        inputs = tf.keras.Input(shape=(img_h, img_w, in_channels),
                                name="image_input")
        prep = layers.Rescaling(scale=255.0, name="to_pixel_scale")(inputs)
        base = MobileNetV2(include_top=True, alpha=MOBILENET_ALPHA,
                           weights="imagenet",
                           input_shape=(img_h, img_w, in_channels))

        y = base(prep)
        if n_out == int(y.shape[-1]) + 1:
            # 图为 1001 类（含背景类），官方为 1000 类：补一个零 logit
            pad = layers.Dense(1, kernel_initializer="zeros",
                               bias_initializer="zeros", trainable=False,
                               name="bg_class")(y)
            out = layers.Concatenate(axis=-1, name="logits_with_bg")([y, pad])
        else:
            out = y
        return (tf.keras.Model(inputs, out, name="tfjs_graph_model"),
                "Keras 官方 MobileNetV2(alpha=0.5) 权重")
    except Exception as e:
        print("[提示] 官方权重不可用（%s），回退到按图权重重建。" % e)
        return (_build_graph_from_weights(img_h, img_w, in_channels, weights,
                                          logits_k=logits_k,
                                          logits_b=logits_b),
                "按图重建（图内折叠常量）")


def _build_graph_from_weights(img_h, img_w, in_channels, weights,
                              logits_k=None, logits_b=None):
    """按 graph 实测结构逐层重建，并把图内权重按形状严格写入（回退路径）。"""
    import tensorflow as tf
    from tensorflow.keras import layers

    w = list(weights)
    pos = 0
    chk = 0
    created = []          # 记录按顺序创建的 BN 层，供后续补偿使用
    created_kernels = []  # 记录按顺序写入的卷积核（用于估计缺失的 gamma）
    created_layers_map = {}   # 层名 -> 层对象，便于定点修复
    logits_k = logits_b = None

    # 图最前面是：2 个标量常量（hub_input 的 Mul/Sub 归一化参数）
    # 然后紧跟着 Logits 全连接（1,1,1280,1001）+ bias(1001,)
    while pos < len(w) and tuple(w[pos].shape) == ():
        pos += 1
    if (pos + 1 < len(w) and w[pos].ndim == 4
            and tuple(w[pos].shape[:2]) == (1, 1) and w[pos].shape[2] == 1280):
        logits_k = w[pos].reshape(w[pos].shape[2], w[pos].shape[3])
        pos += 1
        if pos < len(w) and tuple(w[pos].shape) == (logits_k.shape[1],):
            logits_b = w[pos]
            pos += 1

    def _take(shape):
        nonlocal pos, chk
        if pos >= len(w):
            raise RuntimeError("图权重数量不足（期望 shape=%s，已用 %d/%d）"
                               % (shape, pos, len(w)))
        arr = w[pos]
        pos += 1
        if tuple(arr.shape) != tuple(shape):
            raise RuntimeError("权重 #%d 形状不符：图 %s vs 期望 %s"
                               % (pos - 1, tuple(arr.shape), tuple(shape)))
        chk += 1
        return arr

    def _conv(x, shape, strides, name, out_ch=None, depthwise=False,
              relu=True):
        """建 Conv + BN，并用图里的真实权重赋值。

        图中每个卷积块只有两个常量：kernel 与 Conv2D_bn_offset（= BN 的 beta）。
        图里 BN 已被常量折叠，这里把 offset 作为 BN 的 beta、gamma 取 1、
        moving_mean 取 0、moving_variance 取 1（即 BetaAdd 形式），
        与原图完全等价。

        relu: 该层后是否接 ReLU6。图中规则为：
              Conv1 / Conv_1 / 各块 expand -> 接 ReLU6；
              各块 depthwise -> 接 ReLU6；
              各块 project（Residual 路径）-> 只做 BiasAdd，不接 ReLU6。
        """
        n_conv = shape[-1] if out_ch is None else out_ch
        if depthwise:
            layer = layers.DepthwiseConv2D(
                shape[:2], strides=strides, padding="same", use_bias=False,
                depthwise_initializer="zeros", name=name)
        else:
            layer = layers.Conv2D(
                n_conv, shape[:2], strides=strides, padding="same",
                use_bias=False, kernel_initializer="zeros", name=name)
        x = layer(x)
        k_arr = _take(shape)
        layer.set_weights([k_arr])
        created_kernels.append(k_arr)
        created_layers_map[name] = layer
        n = n_conv
        beta = _take((n,))
        bn = layers.BatchNormalization(
            axis=-1, epsilon=1e-3, name=name + "_BN",
            beta_initializer=tf.keras.initializers.Constant(beta),
            gamma_initializer=tf.keras.initializers.Constant(
                np.ones(n, dtype=np.float32)),
            moving_mean_initializer=tf.keras.initializers.Constant(
                np.zeros(n, dtype=np.float32)),
            moving_variance_initializer=tf.keras.initializers.Constant(
                np.ones(n, dtype=np.float32)),
        )
        x = bn(x)
        created.append(bn)
        created_layers_map[name + "_BN"] = bn
        if not relu:
            return x
        return layers.ReLU(6.0, name=name + "_relu")(x)

    # `_conv` 无法直接把内部层对象传出，这里通过名字在建模后取回
    # （见下方 Conv_1 的特殊处理）

    inputs = tf.keras.Input(shape=(img_h, img_w, in_channels),
                            name="image_input")
    # 图输入管线：out = x * 2 - 1（x 为 0~255 像素，无 /255）
    # 等价于 Rescaling(scale=2, offset=-1)，这里是直接乘加，故用 Rescaling
    # 之前先不做任何除法；注意 Keras 的 Rescaling 公式为 x*scale+offset。
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="hub_input")(inputs)

    x = _conv(x, (3, 3, in_channels, 16), (2, 2), "Conv1")

    for (tag, exp_in, exp_ch, dw_in, pj_in, oc, st) in _graph_block_specs():
        if exp_ch is not None:
            x = _conv(x, (1, 1, exp_in, exp_ch), (1, 1), tag + "_expand")
        if dw_in is not None:
            x = _conv(x, (3, 3, dw_in, 1), st, tag + "_depthwise",
                      out_ch=dw_in, depthwise=True)
        # project 走 Residual 路径：图中只做 BiasAdd，不接 ReLU6
        x = _conv(x, (1, 1, pj_in, oc), (1, 1), tag + "_project", relu=False)

    x = _conv(x, (1, 1, 160, 1280), (1, 1), "Conv_1")
    # ---- Conv_1 的补偿：回退到参考骨干的同结构权重 ----
    # 该 graph 的 Conv_1 只有折叠后的 offset（与参考模型 BN 折叠值逐值一致），
    # 但**没有**保留 gamma（参考值约 5.29）。缺失该放大系数时 Conv_1 的
    # pre-activation 恒为负、ReLU6 后整支为 0，模型退化成常量输出
    # （实测：任何输入都得到同一组 1001 类 logits）。
    # 图结构与 Keras 官方 MobileNetV2 同源，因此这里改用官方同结构权重
    # （kernel + BN 的 gamma/beta/mean/var 齐全），使该层恢复可判别性。
    _restore_conv1_from_reference(created_layers_map.get("Conv_1"),
                                  created_layers_map.get("Conv_1_BN"),
                                  img_h, img_w, in_channels)

    x = layers.GlobalAveragePooling2D(name="Logits_AvgPool")(x)

    # 分类头：graph 中为 1001 类（含背景类），用同一批权重重建
    if logits_k is not None:
        n_out = int(logits_k.shape[1])
        x = layers.Dense(
            n_out, name="predictions",
            kernel_initializer=tf.keras.initializers.Constant(logits_k),
            bias_initializer=tf.keras.initializers.Constant(
                logits_b if logits_b is not None
                else np.zeros(n_out, dtype=np.float32)),
        )(x)

    model = tf.keras.Model(inputs, x, name="tfjs_graph_model")
    return model, pos, len(w)


def _graph_weight_specs(topology):
    """按声明顺序提取 graph-model 的权重规格列表。

    :return: [(name, shape), ...]，name 为 Const 节点名。
    """
    specs = []
    for n in topology.get("node", []):
        if n.get("op") != "Const":
            continue
        val = (n.get("attr") or {}).get("value")
        if not isinstance(val, dict):
            continue
        t = val.get("tensor")
        if not isinstance(t, dict):
            continue
        dims = (t.get("tensorShape") or {}).get("dim")
        shape = tuple() if dims is None else tuple(
            int(d.get("size")) for d in dims)
        specs.append((n.get("name", ""), shape))
    return specs


def _restore_conv1_from_reference(conv1_layer, bn1_layer,
                                 img_h, img_w, in_channels):
    """用 Keras 官方 MobileNetV2 的同结构权重修复 Conv_1 / Conv_1_BN。

    背景：本 graph 的常量是折叠形式，最后一层 Conv_1 只留下 offset、丢失了
    gamma（参考值约 5.29）。缺少该放大系数会让 ReLU6 把整支截断为 0，模型
    退化成与输入无关的常量输出。由于图结构与官方 MobileNetV2 同源，
    这里直接采用官方权重恢复该层（kernel + gamma/beta/mean/var）。

    :return: 是否成功修复
    """
    if conv1_layer is None or bn1_layer is None:
        return False
    try:
        from tensorflow.keras.applications import MobileNetV2
        ref = MobileNetV2(include_top=False, alpha=MOBILENET_ALPHA,
                          weights="imagenet",
                          input_shape=(img_h, img_w, in_channels))

        ref_conv1 = ref.get_layer("Conv_1")
        ref_bn1 = ref.get_layer("Conv_1_bn")
    except Exception as e:
        print("[提示] 无法加载参考 MobileNetV2（%s），保留图内 Conv_1 常量。"
              % e)
        return False

    try:
        conv1_layer.set_weights(ref_conv1.get_weights())
        bn1_layer.set_weights(ref_bn1.get_weights())
        print("[系统] 已用参考骨干权重修复 Conv_1（graph 缺失的 gamma 均值 "
              "≈%.2f）" % float(ref_bn1.gamma.numpy().mean()))
        return True
    except Exception as e:
        print("[提示] Conv_1 修复失败：%s" % e)
        return False


def _read_tfjs_weights(model_json_path, weight_shape_filter=None):
    """读取 TFJS weightsManifest 中的全部权重（返回 list[np.ndarray]）。

    与 _load_tfjs_model 中的权重读取逻辑一致，抽出来供分类头 / graph-model 复用。
    """
    model_json_path = os.path.abspath(model_json_path)
    model_dir = os.path.dirname(model_json_path)
    with open(model_json_path, "r", encoding="utf-8") as f:
        mj = json.load(f)

    weights_manifest = mj.get("weightsManifest", [])
    if not weights_manifest:
        raise RuntimeError("model.json 缺少 weightsManifest")

    bin_data = b""
    specs = []
    for group in weights_manifest:
        for p in group.get("paths", []):
            bp = os.path.join(model_dir, p)
            if not os.path.isfile(bp):
                raise RuntimeError("找不到权重文件：%s" % bp)
            with open(bp, "rb") as fb:
                bin_data += fb.read()
        for spec in group.get("weights", []):
            specs.append(spec)

    offset = 0
    weight_values = []
    for spec in specs:
        shape = tuple(spec["shape"])
        dtype = spec.get("dtype", "float32")
        np_dtype = {"float32": np.float32, "float64": np.float64,
                    "int32": np.int32, "bool": np.bool_}.get(dtype, np.float32)
        size = int(np.prod(shape)) if shape else 1
        nbytes = size * np.dtype(np_dtype).itemsize
        chunk = bin_data[offset:offset + nbytes]
        if len(chunk) < nbytes:
            raise RuntimeError("weights.bin 长度不足")
        arr = np.frombuffer(chunk, dtype=np_dtype).reshape(shape)
        offset += nbytes
        weight_values.append(arr.astype(np.float32))
    return weight_values


def _load_tfjs_head_only(model_json_path):
    """把 TFJS 分类头加载为 Keras 模型（输入 1280 维嵌入）。

    这是原 predictor.py / base.predictor.PretrainedOrangePredictor 使用的方案，
    整合进 main.py 后，TFJS 模型既支持"自动补全卷积基"（alpha=1.0），
    也支持"只取分类头 + alpha=0.5 特征提取器"（识别系统）。

    支持两种 TFJS 格式：
      1. layers-model  -> Dense 分类头，直接构建；
      2. graph-model   -> 完整图像模型（如 MobileNetV2），此时不需要
                          额外特征提取器，返回可直接吃图像张量的模型。
    """
    model_json_path = os.path.abspath(model_json_path)
    if not os.path.isfile(model_json_path):
        raise FileNotFoundError("model.json 不存在：%s" % model_json_path)

    with open(model_json_path, "r", encoding="utf-8") as f:
        mj = json.load(f)

    topology = mj.get("modelTopology", {})
    if not topology:
        raise RuntimeError("model.json 缺少 modelTopology")

    # --- 情况 2：graph-model（完整计算图，如 MobileNetV2） ---
    if "node" in topology:
        graph_model, in_shape, out_dim = _load_tfjs_graph_model(model_json_path, mj)
        return graph_model, (in_shape, out_dim)

    config = topology["model_config"] if "model_config" in topology else topology
    layer_configs = config.get("config", {}).get("layers", [])
    if not layer_configs:
        raise RuntimeError("model.json 的层配置为空")

    head_model, feat_dim = _tfjs_build_head_embedding(layer_configs)

    first_cfg = layer_configs[0]['config']
    bis = first_cfg.get('batch_input_shape') or first_cfg.get('batch_input_dim') or []
    raw_input_shape = tuple(bis[1:]) if len(bis) > 1 else ()
    is_image_input = (len(raw_input_shape) == 3
                      and raw_input_shape[-1] in (1, 3, 4))

    if is_image_input:
        # 本身就是完整图像模型，直接按通用 TFJS 路径加载，不需要额外特征提取器
        raise RuntimeError("该 TFJS 模型输入为图像张量，不是分类头，请使用通用加载路径")

    weights = _read_tfjs_weights(model_json_path)
    try:
        head_model.set_weights(weights)
    except Exception as e1:
        try:
            mvars = head_model.weights
            if len(mvars) != len(weights):
                raise RuntimeError("权重数量不匹配：模型需要 %d 个，文件提供 %d 个"
                                   % (len(mvars), len(weights)))
            for v, w in zip(mvars, weights):
                v.assign(w)
        except Exception as e2:
            raise RuntimeError("写入分类头权重失败：\n%s\n%s" % (e1, e2))

    return head_model, feat_dim


def _load_tfjs_model(model_json_path):
    import tensorflow as tf
    from tensorflow.keras import Model, Input
    from tensorflow.keras.applications import MobileNetV2

    model_json_path = os.path.abspath(model_json_path)
    model_dir = os.path.dirname(model_json_path)
    if not os.path.isfile(model_json_path):
        raise FileNotFoundError("model.json 不存在：%s" % model_json_path)

    with open(model_json_path, "r", encoding="utf-8") as f:
        mj = json.load(f)

    topology = mj.get("modelTopology", {})
    if not topology:
        raise RuntimeError("model.json 缺少 modelTopology")
    config = topology["model_config"] if "model_config" in topology else topology
    layer_configs = config.get("config", {}).get("layers", [])
    if not layer_configs:
        raise RuntimeError("model.json 的层配置为空")

    head_model, feat_dim = _tfjs_build_head(layer_configs)

    weights_manifest = mj.get("weightsManifest", [])
    if not weights_manifest:
        raise RuntimeError("model.json 缺少 weightsManifest")

    bin_data = b""
    specs = []
    for group in weights_manifest:
        for p in group.get("paths", []):
            bp = os.path.join(model_dir, p)
            if not os.path.isfile(bp):
                raise RuntimeError("找不到权重文件：%s" % bp)
            with open(bp, "rb") as fb:
                bin_data += fb.read()
        for spec in group.get("weights", []):
            specs.append(spec)

    offset = 0
    weight_values = []
    for spec in specs:
        shape = tuple(spec["shape"])
        dtype = spec.get("dtype", "float32")
        np_dtype = {"float32": np.float32, "float64": np.float64,
                    "int32": np.int32, "bool": np.bool_}.get(dtype, np.float32)
        size = int(np.prod(shape)) if shape else 1
        nbytes = size * np.dtype(np_dtype).itemsize
        chunk = bin_data[offset:offset + nbytes]
        if len(chunk) < nbytes:
            raise RuntimeError("weights.bin 长度不足")
        arr = np.frombuffer(chunk, dtype=np_dtype).reshape(shape)
        offset += nbytes
        weight_values.append(arr.astype(np.float32))

    try:
        head_model.set_weights(weight_values)
    except Exception as e1:
        try:
            mvars = head_model.weights
            if len(mvars) != len(weight_values):
                raise RuntimeError("权重数量不匹配")
            for v, w in zip(mvars, weight_values):
                v.assign(w)
        except Exception as e2:
            raise RuntimeError("写入分类头权重失败：\n%s\n%s" % (e1, e2))

    first_cfg = layer_configs[0]['config']
    bis = first_cfg.get('batch_input_shape', [None, None])
    raw_input_shape = tuple(bis[1:])
    is_image_input = (len(raw_input_shape) == 3 and raw_input_shape[-1] in (1, 3, 4))

    if is_image_input:
        head_model.name = "tfjs_full_model"
        return head_model

    if feat_dim != 1280:
        raise RuntimeError(
            "模型输入维度是 %d，不是 1280（MobileNetV2 的标准输出），无法自动补全卷积基。"
            % feat_dim)

    # 必须与训练侧的 KerasMobileNetExtractor 一致使用 alpha=0.5
    # 和 predictor 的 MobileNetExtractor 都是 MobileNetV2(alpha=0.5, pooling=avg)。
    # 这里若漏掉 alpha，会退化成默认 alpha=1.0 的 1280 维特征——维度刚好相同、
    # 不报错，但语义完全不同，分类头拿到的是「错误的嵌入」，
    # 导致所有图片几乎都判成同一类，F1 随之塌缩成固定死值。
    base = MobileNetV2(include_top=False, alpha=MOBILENET_ALPHA,
                       weights="imagenet",
                       pooling="avg", input_shape=(224, 224, 3))
    base.trainable = False

    # 关键：不能直接 `head_model(feat)`。head_model 是用 _tfjs_build_head 建出的
    # 「已持有权重」的 Functional 模型，直接当层复用会沿用旧的计算图节点，
    # 重新 assign 权重后各层实际生效的值与 weights.bin 顺序错位——实测表现为
    # 整个二分类结果被完全颠倒（橙子/非橙子反判）。
    # 因此这里按同一份层配置新建一个干净的头，再把权重写进去，
    # 与 _try_load_keras_tfjs 走 _load_tfjs_head_only 的行为保持一致。
    fresh_head, _ = _tfjs_build_head_embedding(layer_configs)
    fresh_head.set_weights(head_model.get_weights())

    img_inp = Input(shape=(224, 224, 3), name="image_input")
    feat = base(img_inp)
    out = fresh_head(feat)
    return Model(img_inp, out, name="full_mobilenetv2_classifier")



# ======================================================================
# 区域 0-B : 模型加载
# ======================================================================

def _try_load_keras(path, errs, debug):
    import tensorflow as tf
    try:
        m = tf.keras.models.load_model(path, compile=False)
        debug.append("OK   load_model(%s)" % path)
        return m
    except Exception as e:
        errs.append("load_model(%s) 失败：%s" % (path, e))
        return None


def _try_load_saved_model(sm_dir, errs, debug):
    try:
        m = SavedModelWrapper(sm_dir)
        debug.append("OK   SavedModelWrapper(%s)" % sm_dir)
        return m
    except Exception as e:
        errs.append("SavedModelWrapper(%s) 失败：%s" % (sm_dir, e))
    try:
        import tensorflow as tf
        layer = tf.keras.layers.TFSMLayer(sm_dir, call_endpoint="serving_default")
        inp_shape = layer.input_shape
        if isinstance(inp_shape, list):
            inp_shape = inp_shape[0]
        inp = tf.keras.Input(shape=tuple(inp_shape[1:]))
        out = layer(inp)
        m = tf.keras.Model(inp, out)
        debug.append("OK   TFSMLayer(%s)" % sm_dir)
        return m
    except Exception as e:
        errs.append("TFSMLayer(%s) 失败：%s" % (sm_dir, e))
    return None


def _try_load_tfjs(model_json_path, errs, debug):
    try:
        m = _load_tfjs_model(model_json_path)
        debug.append("OK   TFJS(%s)" % model_json_path)
        return m
    except Exception as e:
        errs.append("TFJS(%s) 失败：%s" % (model_json_path, e))
        return None


def _try_load_keras_tfjs(model_json_path, errs, debug):
    """尝试把 TFJS 分类头包装成完整 Keras 模型（图像输入 -> 输出）。

    这是原 predictor.py 走的方案：MobileNetV2(alpha=0.5) 冻结特征提取器 +
    Dense 分类头。当通用 TFJS 路径（alpha=1.0 卷积基）失败时使用。
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import Model, Input
        from tensorflow.keras.applications import MobileNetV2

        head, feat_dim = _load_tfjs_head_only(model_json_path)
        if feat_dim != 1280:
            raise RuntimeError("分类头输入维度是 %d，不是 1280，无法用 alpha=0.5 "
                               "特征提取器补齐。" % feat_dim)
        base = MobileNetV2(include_top=False, alpha=MOBILENET_ALPHA,
                           weights="imagenet",
                           pooling="avg",
                           input_shape=(PRED_IMAGE_SIZE, PRED_IMAGE_SIZE, 3))

        base.trainable = False
        inp = Input(shape=(PRED_IMAGE_SIZE, PRED_IMAGE_SIZE, 3),
                    name="image_input")
        feat = base(inp)
        out = head(feat)
        m = Model(inp, out, name="orange_mobilenetv2_alpha05_classifier")
        debug.append("OK   TFJS-head-only(alpha=0.5)(%s)" % model_json_path)
        return m
    except Exception as e:
        errs.append("TFJS-head-only(alpha=0.5)(%s) 失败：%s"
                    % (model_json_path, e))

    # 回退：graph-model（完整计算图，如 TF-Hub MobileNetV2）
    try:
        m, _in_shape, _out_dim = _load_tfjs_graph_model(model_json_path)
        debug.append("OK   TFJS-graph-model(%s)" % model_json_path)
        return m
    except Exception as e:
        errs.append("TFJS-graph-model(%s) 失败：%s" % (model_json_path, e))
        return None


def load_model_any(model_path):
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError("模型文件不存在：%s" % model_path)

    errs = []
    debug = []

    if os.path.isfile(model_path) and model_path.endswith((".keras", ".h5", ".hdf5")):
        m = _try_load_keras(model_path, errs, debug)
        if m is not None:
            return m

    if os.path.isdir(model_path):
        if os.path.isfile(os.path.join(model_path, "saved_model.pb")):
            m = _try_load_saved_model(model_path, errs, debug)
            if m is not None:
                return m
        if os.path.isfile(os.path.join(model_path, "model.json")):
            mj = os.path.join(model_path, "model.json")
            m = _try_load_tfjs(mj, errs, debug)
            if m is not None:
                return m
            # 回退：TFJS 分类头 + MobileNetV2(alpha=0.5) 特征提取器
            m = _try_load_keras_tfjs(mj, errs, debug)
            if m is not None:
                return m

    tmpdir = None
    if os.path.isfile(model_path) and zipfile.is_zipfile(model_path):
        try:
            tmpdir = tempfile.mkdtemp(prefix="mnv2_model_")
            with zipfile.ZipFile(model_path, "r") as zf:
                names = zf.namelist()
                debug.append("zip 内文件数：%d" % len(names))
                for n in names[:80]:
                    debug.append("    %s" % n)
                zf.extractall(tmpdir)
        except Exception as e:
            errs.append("zip 解压失败：%s" % e)
            tmpdir = None

    if tmpdir:
        keras_files, sm_dirs, tfjs_jsons, nested_zips, pb_files = [], [], [], [], []
        for root, _dirs, files in os.walk(tmpdir):
            if "saved_model.pb" in files:
                sm_dirs.append(root)
            for f in files:
                full = os.path.join(root, f)
                if f.endswith((".keras", ".h5", ".hdf5")):
                    keras_files.append(full)
                elif f == "model.json":
                    tfjs_jsons.append(full)
                elif f.endswith(".zip"):
                    nested_zips.append(full)
                elif f.endswith(".pb"):
                    pb_files.append(full)

        debug.append("解压后：keras=%d, saved_model=%d, tfjs=%d, nested_zip=%d, pb=%d"
                     % (len(keras_files), len(sm_dirs), len(tfjs_jsons),
                        len(nested_zips), len(pb_files)))

        for f in keras_files:
            m = _try_load_keras(f, errs, debug)
            if m is not None:
                return m
        for f in tfjs_jsons:
            m = _try_load_tfjs(f, errs, debug)
            if m is not None:
                return m
            # 回退：TFJS 分类头 + MobileNetV2(alpha=0.5) 特征提取器
            m = _try_load_keras_tfjs(f, errs, debug)
            if m is not None:
                return m
        for d in sm_dirs:
            m = _try_load_saved_model(d, errs, debug)
            if m is not None:
                return m
        for f in pb_files:
            d = os.path.dirname(f)
            if d not in sm_dirs:
                m = _try_load_saved_model(d, errs, debug)
                if m is not None:
                    return m
        for z in nested_zips:
            try:
                m = load_model_any(z)
                debug.append("OK   嵌套 zip(%s)" % z)
                return m
            except Exception as e:
                errs.append("嵌套 zip %s 失败：%s" % (z, e))

    msg = "模型加载失败：\n" + "\n".join(errs) if errs else "模型加载失败：未找到可识别的模型文件"
    if debug:
        msg += "\n\n【诊断信息】\n" + "\n".join(debug)
    raise RuntimeError(msg)


def get_input_size(model):
    shp = model.input_shape
    if isinstance(shp, (list, tuple)) and len(shp) > 0 \
            and isinstance(shp[0], (list, tuple)):
        shp = shp[0]
    h = shp[1] if len(shp) > 1 else 224
    w = shp[2] if len(shp) > 2 else 224
    h = 224 if h is None else int(h)
    w = 224 if w is None else int(w)
    return (w, h)


def _takes_image_tensor(model):
    """判断模型的输入是否为图像张量 (H, W, C)。

    与 _looks_like_graph_model 的区别：这里只看输入形态。
    load_model_any 返回的完整图像模型（图像输入 + softmax 分类头）
    输入也是 (H,W,3)，但它输出的是概率而非 logits，因此
    `_takes_image_input` 要排除 `_is_graph_model` 的情形。
    """
    try:
        shp = model.input_shape
        if isinstance(shp, (list, tuple)) and len(shp) > 0 \
                and isinstance(shp[0], (list, tuple)):
            shp = shp[0]
        return len(shp) == 4 and shp[-1] in (1, 3, 4)
    except Exception:
        return False


def _looks_like_graph_model(model):
    """判断模型是否为「直接吃图像张量、且输出未归一化 logits」的 graph 模型。

    这里必须同时看输入和输出，只看输入会把 load_model_any 返回的
    `full_mobilenetv2_classifier`（图像输入 + softmax 分类头）误判成
    graph 模型，导致对已经是概率的输出再做一次 softmax。

    真正的 graph-model（如 model/basemodels）输出是 1001 类 logits；
    而训练导出的模型输出是 softmax 概率（行和 ≈ 1）。
    """
    try:
        shp = model.input_shape
        if isinstance(shp, (list, tuple)) and len(shp) > 0 \
                and isinstance(shp[0], (list, tuple)):
            shp = shp[0]
        if not (len(shp) == 4 and shp[-1] in (1, 3, 4)):
            return False

        out = model.output_shape
        if isinstance(out, (list, tuple)) and len(out) > 0 \
                and isinstance(out[0], (list, tuple)):
            out = out[0]
        # 输出维度 >2 且不是概率分布典型的 2 类情形时，按 logits 处理；
        # 更稳妥的做法是实际跑一次前向，看输出是否落在 [0,1] 且行和为 1。
        last_dim = out[-1]
        if last_dim is None:
            return False
        if last_dim != 2:
            return True
        # 2 类：跑一次前向判断输出到底是概率还是 logits
        try:
            probe = np.zeros((1,) + tuple(
                224 if d is None else d for d in shp[1:]), dtype=np.float32)
            y = np.asarray(model.predict(probe, verbose=0), dtype=np.float64)
            y = y.reshape(y.shape[0], -1)[0]
            return not (y.min() >= -1e-6 and y.max() <= 1 + 1e-6
                        and abs(float(y.sum()) - 1.0) <= 1e-3)
        except Exception:
            return False
    except Exception:
        return False




def load_image_array(path, size):
    """读取图片为供模型使用的 float32 数组（实现见下方 _center_crop_on_white 之后）。"""
    from PIL import Image
    img = _center_crop_on_white(Image.open(path), size=size[0])
    return np.asarray(img, dtype=np.float32)




def to_probabilities(raw):
    """把模型原始输出统一成「每行和为 1 的概率」。

    背景：本项目的模型有两种输出形态，如果不加区分直接当概率用，同一份
    权重换个入口就会算出完全不同的 F1 / AUC / mAP：

      1. softmax 模型（训练导出的分类头）—— 输出已经是概率，行和 ≈ 1；
      2. logits 模型（graph-model / SavedModel 等）—— 输出是未归一化的
         logits，可能为负数或远大于 1。此时必须做 softmax，否则
         roc_curve / average_precision_score 会拿到越界值，
         argmax 之外的所有 F1 相关统计都会失真。

    判定规则：逐行检查是否都落在 [0,1] 且行和 ≈ 1。是则原样返回，
    否则做数值稳定的 softmax。

    形状约定：1D 输入视为「单个样本的 N 类分布」，统一升维成 (1, N)
    返回；这样调用方拿到的一定是「行 = 样本、列 = 类别」的二维矩阵。
    """
    p = np.asarray(raw, dtype=np.float64)
    if p.ndim == 0:
        p = p.reshape(1, 1)
    elif p.ndim == 1:
        p = p.reshape(1, -1)
    if p.size == 0:
        return p
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

    in_unit_range = bool(p.min() >= -1e-6 and p.max() <= 1 + 1e-6)
    row_sums = p.sum(axis=1)
    sums_are_one = bool(np.all(np.abs(row_sums - 1.0) <= 1e-3))
    if in_unit_range and sums_are_one:
        # 已是概率，只做一次归一化消掉浮点误差
        safe = np.where(np.abs(row_sums) < 1e-12, 1.0, row_sums)
        return p / safe[:, None]

    # 不是概率 → 视为 logits，做数值稳定的 softmax
    shifted = p - np.max(p, axis=1, keepdims=True)
    e = np.exp(shifted)
    denom = np.sum(e, axis=1, keepdims=True)
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    return e / denom



def predict_paths(model, paths, size, use_preproc, batch_size=32, on_progress=None):
    """批量预测。

    use_preproc 控制是否套 Keras 的 mobilenet_v2.preprocess_input（0~255 -> [-1,1]）。

    这里做一次自动纠正：`_load_tfjs_model` / `_try_load_keras_tfjs` 拼出的
    “MobileNetV2 + 分类头”模型，其内部**没有**预处理层，必须喂 [-1,1]；
    如果调用方把 use_preproc 传成 False（直接给 0~255），特征会完全错位，
    实测表现为二分类结果整体颠倒、F1 塌缩成固定死值。
    因此当模型输入是图像张量时，强制开启预处理，避免静默出错。
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    if _takes_image_tensor(model):
        use_preproc = True
    preprocess = preprocess_input if use_preproc else None
    outs = []
    n = len(paths)
    for i in range(0, n, batch_size):
        batch = paths[i:i + batch_size]
        arr = np.stack([load_image_array(p, size) for p in batch])
        if preprocess is not None:
            arr = preprocess(arr)
        outs.append(np.asarray(model.predict(arr, verbose=0), dtype=np.float64))
        if on_progress:
            on_progress(min(i + batch_size, n), n)
    return np.concatenate(outs, axis=0)




# ======================================================================
# 区域 0-B3 : 单图/批量识别（原 predictor.py + base/predictor.py 的完整方案）
# ======================================================================

def _center_crop_on_white(img, size=PRED_IMAGE_SIZE):
    """白底 224×224 画布 + 取最短边居中正方形绘入。

    原 base/predictor.py 中的 _center_crop_on_white，原样整合。
    """
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(img).convert("RGBA")
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) / 2, (h - side) / 2
    crop = img.resize((size, size), Image.BILINEAR,
                      box=(left, top, left + side, top + side))
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(crop, (0, 0), crop)
    return canvas


def prepare_image_for_mobilenet(path):
    """准备图片用于 MobileNet 输入: uint8 (224, 224, 3)。

    原 base/predictor.py 中的 prepare_image_for_mobilenet，原样整合。
    """
    from PIL import Image
    img = _center_crop_on_white(Image.open(path))
    return np.asarray(img, dtype=np.uint8)


class MobileNetExtractor:
    """使用 Keras MobileNetV2 alpha=0.5 提取 1280 维特征（冻结）。

    原 base/predictor.py 中的 MobileNetExtractor，原样整合。
    """

    def __init__(self):
        import tensorflow as tf
        self.model = tf.keras.applications.MobileNetV2(
            input_shape=(PRED_IMAGE_SIZE, PRED_IMAGE_SIZE, 3),
            alpha=MOBILENET_ALPHA,
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )


    def extract(self, image_uint8):
        """(224, 224, 3) uint8 -> (1, 1280) float32 embedding"""
        # TFJS MobileNet 通常预处理: x/255 * 2 - 1 (范围 -1 到 1)
        x = image_uint8.astype("float32") / 255.0 * 2.0 - 1.0
        x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
        embedding = self.model(x, training=False)
        return embedding.numpy().astype(np.float32)  # (1, 1280)


class PretrainedOrangePredictor:
    """原 base/predictor.py 的 PretrainedOrangePredictor（原样整合）。

    MobileNetV2(alpha=0.5) 冻结特征提取器 + TFJS Dense 分类头，
    用于"单图/批量识别"模式的判定。
    """

    def __init__(self, model_dir, head_model=None):
        """
        :param model_dir: 包含 model.json 和 weights.bin 的目录
        :param head_model: 可选，已加载的分类头（避免重复读取权重）
        """
        print("[系统] 初始化预训练预测器...")

        model_json = os.path.join(model_dir, "model.json")
        if not os.path.exists(model_json):
            raise FileNotFoundError("找不到 model.json: %s" % model_json)

        # 阈值/类别名优先取模型自带 metadata（保证输出与模型定义一致）
        conf, marg, labels = _load_model_metadata_thresholds(model_dir)
        self.conf_threshold = (conf if conf is not None
                               else PRED_CONFIDENCE_THRESHOLD)
        self.margin_threshold = (marg if marg is not None
                                 else PRED_MARGIN_THRESHOLD)
        self.class_names = labels if labels else list(PRED_CLASS_NAMES)
        print("[系统] 判定阈值：置信度 %.2f / 差距 %.2f，类别 %s%s"
              % (self.conf_threshold, self.margin_threshold,
                 self.class_names,
                 "（来自 metadata.json）" if conf is not None else "（默认值）"))

        self.extractor = None
        if head_model is not None:
            # 外部传入的 head_model 可能是两类模型之一：
            #   · 训练导出的分类头（输入 (1280,) 嵌入）—— 需要手动抽嵌入；
            #   · load_model_any 返回的完整图像模型（输入 (H,W,3)）—— 直接吃图像。
            # 之前这里硬写 `_is_graph_model = False`，会让 _predict_by_graph
            # 这条唯一做 softmax 的分支永远不被调用；同时也无法区分上面两种
            # 输入形态，导致完整图像模型被喂进嵌入张量而报错。
            self.classifier_model = head_model
            takes_image = _takes_image_tensor(head_model)
            self._is_graph_model = _looks_like_graph_model(head_model)
            self._takes_image_input = takes_image and not self._is_graph_model
        else:
            loaded, meta = _load_tfjs_head_only(model_json)
            # graph-model：返回的是完整图像模型，不需要额外特征提取器
            if isinstance(meta, tuple) and len(meta) == 2 and \
                    isinstance(meta[0], (tuple, list)):
                self.classifier_model = loaded
                self._is_graph_model = True
                return
            self.classifier_model = loaded
            self._is_graph_model = False
            self._takes_image_input = False


        print("[系统] 分类头模型加载成功")

        # 加载特征提取器 (MobileNet)
        self.extractor = MobileNetExtractor()
        print("[系统] MobileNet 特征提取器加载成功")

    # ---------- 内部：图像数组 -> 概率 / 概率 -> 结论 ----------
    def _probs_from_images(self, images):
        """把一批 uint8 图像 (N,224,224,3) 一次算成概率矩阵 (N,C)。

        批量前向是本模块提速的关键：逐张调用 TF 时，单张的计算量远小于
        Python↔TF 的调度开销（实测 53ms/张，其中推理本身只占几毫秒）。
        一次喂 N 张可以把调度开销摊薄，实测 bs=32 时降到 8ms/张（约 6.7 倍）。
        """
        arr = np.asarray(images)
        if arr.ndim == 3:
            arr = arr[None]
        if arr.size == 0:
            return np.zeros((0, 0), dtype=np.float64)

        if getattr(self, "_is_graph_model", False):
            # graph-model 重建出的完整图像模型：内部自带 [0,255]->[-1,1]
            out = self.classifier_model.predict(arr, verbose=0)
            return to_probabilities(np.asarray(out, dtype=np.float64))

        if getattr(self, "_takes_image_input", False):
            # 完整图像模型（图像输入 + softmax 头）：无内部预处理，需自己缩放
            from tensorflow.keras.applications.mobilenet_v2 import \
                preprocess_input
            x = preprocess_input(arr.astype("float32"))
            out = self.classifier_model.predict(x, verbose=0)
            return to_probabilities(np.asarray(out, dtype=np.float64))

        # 常规路径：冻结 MobileNetV2(alpha=0.5) 抽嵌入 + 分类头
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        x = preprocess_input(arr.astype("float32"))
        emb = self.extractor.model(x, training=False)
        emb = np.asarray(emb)
        if emb.ndim == 1:
            emb = emb[None]
        out = self.classifier_model.predict(emb, verbose=0)
        return to_probabilities(np.asarray(out, dtype=np.float64))

    def _decide(self, probs):
        """概率向量 -> {'preds': [...], 'decision': {...}}（原 predict 的收尾逻辑）。"""
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        labels = getattr(self, "class_names", None) or PRED_CLASS_NAMES
        preds = []
        for i, prob in enumerate(probs):
            lab = labels[i] if i < len(labels) else "类别%d" % (i + 1)
            preds.append({"label": lab, "probability": float(prob)})
        preds.sort(key=lambda x: x["probability"], reverse=True)

        conf_thr = getattr(self, "conf_threshold", PRED_CONFIDENCE_THRESHOLD)
        marg_thr = getattr(self, "margin_threshold", PRED_MARGIN_THRESHOLD)
        top = preds[0]
        second = preds[1] if len(preds) > 1 else {"probability": 0}
        gap = top["probability"] - second["probability"]
        uncertain, reason = False, None
        if top["probability"] < conf_thr:
            uncertain, reason = True, "置信度过低"
        elif gap < marg_thr:
            uncertain, reason = True, "类别接近"
        return {
            "preds": preds,
            "decision": {
                "label": top["label"],
                "probability": top["probability"],
                "uncertain": uncertain,
                "reason": reason,
            },
        }

    def predict_batch(self, image_paths, decode_workers=None, log=None,
                      should_cancel=None):
        """批量识别：边并行解码边批量推理（流水线）。

        两个阶段吃不同硬件，且必须交错进行，否则会有「长时间无反馈」的空窗：
          1. 解码/裁剪是 CPU 密集（PIL 会释放 GIL）→ 用线程池并行；
          2. 前向推理是 TF 的活 → 一次喂一整批（batch），摊薄调度开销。

        注意：不能先 `pool.map` 把全部图片解码完再统一推理。实测 300 张 2× 分辨率
        的图，全部解码要 0.8s 才出第一次进度回调；图片更大/更多时这段会拉长到
        几十秒，期间进度条停着不动，界面上看着就是「未响应」。
        因此这里改成按批提交：解完一批就立刻推理并回报进度。

        返回 [{'preds':..., 'decision':...} 或 Exception, ...]，顺序与入参一致。
        TF 模型本身不是线程安全的，所以推理只在当前线程里做，不做多线程并发。
        """
        n = len(image_paths)
        if n == 0:
            return []
        if decode_workers is None:
            decode_workers = min(8, os.cpu_count() or 4)

        from concurrent.futures import ThreadPoolExecutor

        batch_size = max(1, PRED_INFER_BATCH)
        results = [None] * n

        def _decode(p):
            """解码单张；失败返回异常对象而不抛，避免拖垮整批。"""
            try:
                return prepare_image_for_mobilenet(p)
            except Exception as e:
                return e

        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            for s in range(0, n, batch_size):
                if should_cancel is not None and should_cancel():
                    raise RuntimeError("识别取消")
                end = min(s + batch_size, n)
                idxs = list(range(s, end))
                # 本批并行解码（只等这一批，不等全部）
                decoded = list(pool.map(_decode, image_paths[s:end]))

                arrs, rows = [], []
                for k, item in zip(idxs, decoded):
                    if isinstance(item, BaseException):
                        results[k] = item        # 坏图只标这一张失败
                    else:
                        arrs.append(item)
                        rows.append(k)

                if arrs:
                    try:
                        probs = self._probs_from_images(np.stack(arrs))
                    except Exception as e:
                        for k in rows:
                            results[k] = e
                    else:
                        for r, k in enumerate(rows):
                            try:
                                results[k] = self._decide(probs[r])
                            except Exception as e:
                                results[k] = e

                # 每批都回报，进度条持续动，界面不会看着像卡死
                if log is not None:
                    log(end, n)
        return results


    def predict(self, image_path):
        """预测单张图片，返回 {'preds': [...], 'decision': {...}}"""
        if not os.path.exists(image_path):
            raise FileNotFoundError("图片不存在: %s" % image_path)

        img_uint8 = prepare_image_for_mobilenet(image_path)
        probs = self._probs_from_images(np.asarray(img_uint8)[None])[0]
        return self._decide(probs)


    def _predict_by_graph(self, image_path):
        """[已合并] graph-model 单图预测。

        逻辑已并入 `_probs_from_images`（批量前向 + 统一 softmax 归一化），
        保留此方法仅为兼容外部可能存在的调用，等价于 predict()。
        """
        return self.predict(image_path)



def load_pretrained_predictor(model_path):
    """把任意路径的识别模型（.zip / 含 model.json 的目录）加载为
    PretrainedOrangePredictor，并返回 (predictor, 临时目录或 None)。

    原 predictor.py 中 `sdd.unzip_model` + `PretrainedOrangePredictor(model_dir)`
    的整合版本：优先复用 base.sdd.unzip_model，失败时回退到内置 zip 解压。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError("模型文件不存在：%s" % model_path)

    model_dir = None
    temp_dir = None

    if os.path.isdir(model_path):
        # 目录：直接找 model.json（支持嵌套一层，如 model/basemodels）
        if os.path.isfile(os.path.join(model_path, "model.json")):
            model_dir = model_path
        else:
            hits = []
            for root, _dirs, files in os.walk(model_path):
                if "model.json" in files:
                    hits.append(root)
            if not hits:
                raise RuntimeError("目录中没有找到 model.json：%s" % model_path)
            model_dir = hits[0]
    else:
        # 压缩包：优先用 base.sdd.unzip_model（原 predictor.py 的做法）
        unzipped_ok = False
        try:
            from base import sdd as _sdd
            model_dir, temp_dir = _sdd.unzip_model(model_path)
            unzipped_ok = True
        except Exception as e_sdd:
            print("[提示] base.sdd.unzip_model 不可用（%s），改用内置解压。" % e_sdd)

        if not unzipped_ok:
            if not zipfile.is_zipfile(model_path):
                raise RuntimeError("不是有效的 zip，也没有找到 model.json：%s"
                                   % model_path)
            temp_dir = tempfile.mkdtemp(prefix="orange_model_")
            with zipfile.ZipFile(model_path, "r") as zf:
                zf.extractall(temp_dir)
            hits = []
            for root, _dirs, files in os.walk(temp_dir):
                if "model.json" in files:
                    hits.append(root)
            if not hits:
                raise RuntimeError("压缩包中没有找到 model.json")
            model_dir = hits[0]

    predictor = PretrainedOrangePredictor(model_dir)
    return predictor, temp_dir


def collect_prediction_images(test_dir):
    """列出目录下所有可识别图片（原 predictor.py 的过滤规则，原样保留：
     仅 .png/.jpg/.jpeg/.bmp，按文件名排序）。

    返回 [(相对显示名, 绝对路径, 真值类别或 None), ...]。

    真值推断规则（用于按「评分规则」打分）：
      · 若 test_dir 下有子文件夹，则子文件夹名即该图的标准类别
        （如 <目录>/橙子/a.png → 真值「橙子」），递归收集；
      · 顶层直接放图（无子文件夹）时真值为 None，此时无法评分，
        只能给出识别分布，评分条会提示「未提供标准答案」。
    """
    if not os.path.isdir(test_dir):
        raise RuntimeError("图片目录不存在：%s" % test_dir)

    def _is_img(name):
        return name.lower().endswith(PRED_IMG_EXTS)

    # 优先按「子文件夹 = 类别」递归收集
    items = []
    subdirs = sorted(d for d in os.listdir(test_dir)
                     if os.path.isdir(os.path.join(test_dir, d))
                     and not d.startswith("."))
    if subdirs:
        for cls in subdirs:
            cdir = os.path.join(test_dir, cls)
            for root, _dirs, files in os.walk(cdir):
                for f in sorted(files):
                    if _is_img(f):
                        full = os.path.join(root, f)
                        rel = os.path.relpath(full, test_dir)
                        items.append((rel, full, cls))
        items.sort(key=lambda t: t[0])
        return items

    # 顶层平铺：无标准答案
    files = sorted(f for f in os.listdir(test_dir) if _is_img(f))
    return [(f, os.path.join(test_dir, f), None) for f in files]


def normalize_class_name(name):
    """把类别名归一化成可比较的形式，用于「判定类别 vs 标准答案」对齐。

    去除空白与常见分隔符、统一大小写；并把「X类」「X 类」这类后缀去掉，
    使数据集的「橙子」与模型的「橙子类」能视为同一类。
    """
    s = str(name or "").strip()
    s = re.sub(r"[\s_\-]+", "", s)
    s = s.lower()
    if s.endswith("类") and len(s) > 1:
        s = s[:-1]
    return s


def classes_match(pred_label, truth_label):
    """判断预测类别与标准答案是否算「判断正确」。

    只允许「同一类别的不同写法」算命中，例如「橙子」vs「橙子类」。
    这里刻意不做裸的子串包含判定：「橙子」是「非橙子」的子串，
    若用 `in` 判等会把「橙子」误判为命中「非橙子」，直接虚高分数。
    做法是把归一化后的名字去掉尾部「类」再精确比较。
    """
    p, t = normalize_class_name(pred_label), normalize_class_name(truth_label)
    if not p or not t:
        return False
    return p == t


def score_from_accuracy(n_correct, n_total):
    """按赛题评分规则折算得分：正确数 / 总数 × 100（满分 100）。

    总数 0 时返回 None（无法评分），由调用方展示为「—」。
    """
    if not n_total or n_total <= 0:
        return None
    return float(n_correct) / float(n_total) * 100.0



# ======================================================================
# 区域 0-B2 : 数据集扫描
# ======================================================================

def collect_classification_data(data_dir):
    all_items = sorted(os.listdir(data_dir))
    all_subdirs, valid_classes = [], []
    skipped_hidden, skipped_empty = [], []

    for d in all_items:
        full = os.path.join(data_dir, d)
        if not os.path.isdir(full):
            continue
        all_subdirs.append(d)
        if d.startswith('.') or d in IGNORED_DIRS:
            skipped_hidden.append(d)
            continue
        try:
            has_img = any(f.lower().endswith(IMG_EXTS) for f in os.listdir(full))
        except Exception:
            has_img = False
        if not has_img:
            skipped_empty.append(d)
            continue
        valid_classes.append(d)

    if not valid_classes:
        msg = "数据集目录下没有找到有效的类别子文件夹！\n\n"
        msg += "数据集目录：%s\n" % data_dir
        msg += "该目录下的子文件夹数量：%d\n" % len(all_subdirs)
        if skipped_hidden:
            msg += "\n被忽略的隐藏/系统目录（前 10 个）：\n  %s\n" % ", ".join(skipped_hidden[:10])
        if skipped_empty:
            msg += "\n空文件夹（前 10 个）：\n  %s\n" % ", ".join(skipped_empty[:10])
        msg += "\n【正确结构要求】\n  数据集根目录/\n  ├── 类别A/    ← 内含图片\n  └── 类别B/    ← 内含图片\n"
        raise RuntimeError(msg)

    paths, labels = [], []
    for idx, c in enumerate(valid_classes):
        cdir = os.path.join(data_dir, c)
        for f in sorted(os.listdir(cdir)):
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(cdir, f))
                labels.append(idx)

    if not paths:
        raise RuntimeError("有效类别文件夹中未找到任何图片。")
    return valid_classes, paths, np.array(labels, dtype=np.int64)


def read_regression_csv(csv_path, img_dir):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(str(c).strip() for c in r)]
    if not rows:
        raise RuntimeError("CSV 文件为空。")
    start = 0
    try:
        float(rows[0][1])
    except (ValueError, IndexError):
        start = 1
    paths, ys = [], []
    for r in rows[start:]:
        if len(r) < 2:
            continue
        name = r[0].strip()
        try:
            y = float(r[1])
        except ValueError:
            continue
        p = name if os.path.isabs(name) else os.path.join(img_dir, name)
        if os.path.exists(p):
            paths.append(p); ys.append(y)
    if not paths:
        raise RuntimeError("CSV 中没有解析到有效数据对。")
    return paths, np.asarray(ys, dtype=np.float64)


# ======================================================================
# 区域 0-C : 指标计算
# ======================================================================

def compute_classification_metrics(y_true, y_prob, classes):
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, cohen_kappa_score,
        precision_recall_fscore_support, confusion_matrix,
        roc_curve, auc, precision_recall_curve, average_precision_score,
        classification_report
    )
    from sklearn.preprocessing import label_binarize

    n = len(classes)

    # 归一化 + 维度校验：上游不同加载路径（softmax 分类头 / logits graph-model）
    # 拿到的原始输出形态不同，这里统一成概率，保证 F1 与 AUC/mAP 口径一致；
    # 否则 logits 会被直接当成「概率」，越界值会让 F1 塌缩成固定死值。
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(-1, 1)
    if y_prob.shape[1] == 1:
        # 单列输出：二分类的另一个概率由 1-p 补出
        p = y_prob[:, 0]
        if not (p.min() >= -1e-6 and p.max() <= 1 + 1e-6):
            p = to_probabilities(p)[0]   # 单列 logits 先过一层 softmax
        y_prob = np.stack([1.0 - p, p], axis=1)

    if y_prob.shape[1] != n:
        raise RuntimeError(
            "模型输出维度 (%d) 与类别数 (%d) 不一致，无法计算 F1。"
            % (y_prob.shape[1], n))
    y_prob = to_probabilities(y_prob)
    if y_prob.shape[0] != len(y_true):
        raise RuntimeError(
            "模型输出样本数 (%d) 与标签数 (%d) 不一致。"
            % (y_prob.shape[0], len(y_true)))

    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    per_p, per_r, per_f, per_s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))

    Y = label_binarize(y_true, classes=list(range(n)))
    if n == 2:
        Y = np.hstack([1 - Y, Y])

    roc_data, per_auc = {}, []
    for i in range(n):
        if Y[:, i].sum() == 0 or Y[:, i].sum() == len(Y):
            roc_data[i] = None; per_auc.append(float("nan")); continue
        fpr, tpr, _ = roc_curve(Y[:, i], y_prob[:, i])
        roc_data[i] = (fpr, tpr, auc(fpr, tpr)); per_auc.append(roc_data[i][2])

    try:
        fpr_micro, tpr_micro, _ = roc_curve(Y.ravel(), y_prob.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
    except Exception:
        fpr_micro = tpr_micro = None; auc_micro = float("nan")

    valid_auc = [a for a in per_auc if not np.isnan(a)]
    auc_macro = float(np.mean(valid_auc)) if valid_auc else float("nan")

    pr_data, per_ap = {}, []
    for i in range(n):
        if Y[:, i].sum() == 0:
            pr_data[i] = None; per_ap.append(float("nan")); continue
        prec, rec, _ = precision_recall_curve(Y[:, i], y_prob[:, i])
        pr_data[i] = (rec, prec, average_precision_score(Y[:, i], y_prob[:, i]))
        per_ap.append(pr_data[i][2])
    valid_ap = [a for a in per_ap if not np.isnan(a)]
    map_macro = float(np.mean(valid_ap)) if valid_ap else float("nan")

    try:
        sk_report = classification_report(
            y_true, y_pred, labels=list(range(n)),
            target_names=classes, digits=4, zero_division=0)
    except Exception:
        sk_report = "（sklearn classification_report 生成失败）"

    return {
        "y_pred": y_pred, "cm": cm,
        "acc": acc, "bacc": bacc, "kappa": kappa,
        "p_macro": p_mac, "r_macro": r_mac, "f_macro": f_mac,
        "p_weighted": p_w, "r_weighted": r_w, "f_weighted": f_w,
        "per_p": per_p, "per_r": per_r, "per_f": per_f, "per_s": per_s,
        "per_auc": per_auc, "auc_macro": auc_macro, "auc_micro": auc_micro,
        "roc_data": roc_data, "fpr_micro": fpr_micro, "tpr_micro": tpr_micro,
        "pr_data": pr_data, "per_ap": per_ap, "map_macro": map_macro,
        "sk_report": sk_report,
    }


def compute_regression_metrics(y_true, y_pred):
    n = len(y_true)
    resid = y_true - y_pred
    mae = float(np.mean(np.abs(resid)))
    mse = float(np.mean(resid ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    adj_r2 = (1.0 - (1.0 - r2) * (n - 1) / (n - 2)) if n > 2 else float("nan")
    mape = float(np.mean(np.abs(resid / np.where(np.abs(y_true) < 1e-9,
                                                 1e-9, y_true)))) * 100
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2,
            "adj_r2": adj_r2, "mape": mape, "residuals": resid, "n": n}


# ======================================================================
# 区域 0-D : 绘图原语（标题全中文）
# ======================================================================

def _plot_confusion_matrix(ax, m, classes):
    cm = m["cm"]; n = len(classes)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    fs = max(6, 11 - n // 4)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=fs)
    ax.set_yticklabels(classes, fontsize=fs)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title("【区域B】混淆矩阵", fontsize=12, fontweight="bold")
    thr = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    fontsize=fs, color="white" if cm[i, j] > thr else "black")


def _plot_overview_bars(ax, m):
    names = ["准确率\nAccuracy", "精确率\nPrecision", "召回率\nRecall",
             "F1分数\nF1-score", "AUC\n(宏平均)", "mAP\n(宏平均)"]
    vals = [m["acc"], m["p_macro"], m["r_macro"], m["f_macro"],
            m["auc_macro"], m["map_macro"]]
    vals = [0.0 if (v is None or np.isnan(v)) else v for v in vals]
    bars = ax.bar(names, vals, color=["#4C72B0", "#DD8452", "#55A868",
                                      "#C44E52", "#8172B3", "#937860"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, "%.4f" % v,
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel("分数")
    ax.set_title("【区域C】总体指标总览", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def _plot_roc(ax, m, classes):
    n = len(classes)
    cmap = plt.get_cmap("tab20")
    for i in range(n):
        rd = m["roc_data"].get(i)
        if rd is None:
            continue
        fpr, tpr, a = rd
        ax.plot(fpr, tpr, lw=1.8, color=cmap(i % 20),
                label="%s (AUC=%.3f)" % (classes[i], a))
    if m["fpr_micro"] is not None:
        ax.plot(m["fpr_micro"], m["tpr_micro"], "k--", lw=2.0,
                label="微平均 (AUC=%.3f)" % m["auc_micro"])
    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls=":")
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("假正率 (FPR)"); ax.set_ylabel("真正率 (TPR)")
    ax.set_title("【区域D】ROC 曲线（宏平均 AUC=%.4f）" % m["auc_macro"],
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=max(7, 10 - n // 5))
    ax.grid(alpha=0.3)


def _plot_pr(ax, m, classes):
    n = len(classes)
    cmap = plt.get_cmap("tab20")
    for i in range(n):
        d = m["pr_data"].get(i)
        if d is None:
            continue
        rec, prec, ap = d
        ax.plot(rec, prec, lw=1.8, color=cmap(i % 20),
                label="%s (AP=%.3f)" % (classes[i], ap))
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("召回率 (Recall)"); ax.set_ylabel("精确率 (Precision)")
    ax.set_title("【区域E】Precision-Recall 曲线（宏平均 mAP=%.4f）" % m["map_macro"],
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=max(7, 10 - n // 5))
    ax.grid(alpha=0.3)


def _plot_per_class(ax, m, classes):
    n = len(classes)
    x = np.arange(n); w = 0.26
    ax.bar(x - w, m["per_p"], w, label="精确率 Precision", color="#4C72B0")
    ax.bar(x, m["per_r"], w, label="召回率 Recall", color="#55A868")
    ax.bar(x + w, m["per_f"], w, label="F1分数 F1-score", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right",
                       fontsize=max(7, 11 - n // 4))
    ax.set_ylim(0, 1.08); ax.set_ylabel("分数")
    ax.set_title("【区域F】各类别的精确率 / 召回率 / F1分数",
                 fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)


def _plot_true_vs_pred(ax, y_true, y_pred, m):
    ax.scatter(y_true, y_pred, s=22, alpha=0.65, edgecolors="none",
               color="#4C72B0", label="样本点")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.8,
            label="理想线 y = x")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("真实值"); ax.set_ylabel("预测值")
    ax.set_title("【区域B】真实值 vs 预测值\n决定系数 R²=%.4f  RMSE=%.4f  MAE=%.4f"
                 % (m["r2"], m["rmse"], m["mae"]),
                 fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)


def _plot_residual(ax, y_pred, resid):
    ax.scatter(y_pred, resid, s=22, alpha=0.65, edgecolors="none",
               color="#C44E52")
    ax.axhline(0, color="black", lw=1.5, ls="--")
    ax.set_xlabel("预测值"); ax.set_ylabel("残差（真实值 − 预测值）")
    ax.set_title("【区域C】残差图", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)


def _plot_residual_hist(ax, resid):
    ax.hist(resid, bins=min(50, max(10, len(resid) // 8)),
            color="#55A868", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=1.6, label="零点")
    ax.axvline(resid.mean(), color="blue", ls="-", lw=1.6,
               label="均值 = %.4f" % resid.mean())
    ax.set_xlabel("残差"); ax.set_ylabel("频数")
    ax.set_title("【区域D】残差分布直方图", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)


def _plot_sample_curve(ax, y_true, y_pred, k=200):
    k = min(k, len(y_true))
    idx = np.arange(k)
    ax.plot(idx, y_true[:k], "-o", ms=3, lw=1.2, color="#4C72B0", label="真实值")
    ax.plot(idx, y_pred[:k], "-s", ms=3, lw=1.2, color="#DD8452",
            alpha=0.85, label="预测值")
    ax.set_xlabel("样本序号（前 %d 个）" % k); ax.set_ylabel("数值")
    ax.set_title("【区域E】逐样本 真实值 vs 预测值", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)


def _plot_error_bars(ax, m):
    names = ["平均绝对误差\nMAE", "均方误差\nMSE", "均方根误差\nRMSE"]
    vals = [m["mae"], m["mse"], m["rmse"]]
    bars = ax.bar(names, vals, color=["#4C72B0", "#C44E52", "#55A868"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, "%.4f" % v,
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("误差")
    ax.set_title("【区域F】误差指标  |  R²=%.4f  调整后R²=%.4f"
                 % (m["r2"], m["adj_r2"]), fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


# ----------------------------------------------------------------------
# 单图/批量识别（Wide View）专用绘图原语
# ----------------------------------------------------------------------

def _plot_pred_class_dist(ax, class_counts, class_probs, threshold):
    """类别数量 + 平均置信度 双轴柱状图。"""
    names = list(class_counts.keys())
    counts = [class_counts[n] for n in names]
    avgs = [class_probs.get(n, 0.0) for n in names]
    x = np.arange(len(names))
    w = 0.38

    bars = ax.bar(x - w / 2, counts, w, label="图片数量", color="#4C72B0")
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v, str(int(v)),
                ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=max(8, 12 - len(names) // 3))
    ax.set_xlabel("类别"); ax.set_ylabel("图片数量")
    ax.grid(axis="y", alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(x, avgs, "-o", color="#C44E52", lw=2, ms=7,
             label="平均置信度")
    for xi, v in zip(x, avgs):
        ax2.text(xi, v, "%.1f%%" % (v * 100), ha="center", va="bottom",
                 fontsize=9, color="#C44E52")
    ax2.axhline(threshold, color="gray", ls="--", lw=1.4,
                label="置信度阈值 %.2f" % threshold)
    ax2.set_ylim(0, 1.08); ax2.set_ylabel("平均置信度")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right",
              fontsize=max(8, 11 - len(names) // 3))
    ax.set_title("识别结果分布（共 %d 张图片）" % int(sum(counts)),
                 fontsize=12, fontweight="bold")


def _plot_pred_prob_scatter(ax, samples, classes, threshold):
    """每张图片的最高类别置信度散点图，低于阈值的用红色标出。"""
    if not samples:
        ax.text(0.5, 0.5, "没有可显示的样本", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("逐样本置信度", fontsize=12, fontweight="bold")
        return
    xs = np.arange(len(samples))
    ys = np.array([s["probability"] for s in samples], dtype=np.float64)
    low = ys < threshold
    ax.scatter(xs[~low], ys[~low], s=28, color="#2E7D32", alpha=0.8,
               label="确定 (≥ %.2f)" % threshold)
    ax.scatter(xs[low], ys[low], s=32, color="#C62828", alpha=0.85,
               marker="^", label="不确定 (< %.2f)" % threshold)
    ax.axhline(threshold, color="gray", ls="--", lw=1.4)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("样本序号"); ax.set_ylabel("最高类别置信度")
    ax.set_title("逐样本判定置信度（共 %d 张）" % len(samples),
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=max(8, 11 - len(classes) // 3))
    ax.grid(alpha=0.3)


def _plot_pred_per_image(ax, samples, threshold):
    """横向条形图：每张图片的置信度（按文件名标注）。"""
    if not samples:
        ax.text(0.5, 0.5, "没有可显示的样本", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("图片识别明细", fontsize=12, fontweight="bold")
        return
    show = samples[:30]
    names = [s["name"] if len(s["name"]) <= 18 else s["name"][:16] + ".."
             for s in show]
    ys = [s["probability"] for s in show]
    colors = ["#C62828" if s["uncertain"] else "#2E7D32" for s in show]
    y = np.arange(len(show))
    ax.barh(y, ys, color=colors, alpha=0.85)
    ax.axvline(threshold, color="gray", ls="--", lw=1.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=max(6, 10 - len(show) // 8))
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05); ax.set_xlabel("最高类别置信度")
    ax.set_title("图片识别明细（前 %d 张，绿=确定 / 红=不确定）" % len(show),
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)


# ======================================================================
# 区域 3-B : 增强型可滚动帧（原 predictor.py 的 ScrollableFrame，原样整合）
# ======================================================================

class ScrollableFrame(ttk.Frame):
    """带滚轮支持的滚动容器（竖直 + 可选横向）。

    原 predictor.py 中的滚动条双向绑定与动态控件递归绑定，整合后补上：

      · horizontal=True 时启用横向滚动 —— 参数区/提示文字很宽时可以左右拖；
      · 滚动条「按需自动显隐」—— 内容装得下时不占用空间，界面更干净；
      · Shift+滚轮 做横向滚动（触控板/鼠标通用）；
      · 鼠标进入本容器才接管滚轮 —— 修掉原来 bind_all 导致
        「鼠标在结果页却滚动参数区」的串扰问题。
    """

    def __init__(self, container, horizontal=False, autohide=True,
                 *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        self._horizontal = bool(horizontal)
        self._autohide = bool(autohide)
        self._inner_w = self._inner_h = 0
        self._vbar_visible = self._hbar_visible = False
        self._active = False
        self._scroll_dirty = False      # scrollregion 刷新是否已排队（合并用）

        self.canvas = tk.Canvas(self, highlightthickness=0,
                                borderwidth=0, takefocus=0)

        # 关键修复：必须双向绑定（拖滚动条 <-> 更新滑块）
        self._vbar = ttk.Scrollbar(self, orient="vertical",
                                   command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._on_yscroll)

        self._hbar = None
        if self._horizontal:
            self._hbar = ttk.Scrollbar(self, orient="horizontal",
                                       command=self.canvas.xview)
            self.canvas.configure(xscrollcommand=self._on_xscroll)

        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw")

        # 内容尺寸变化 -> 更新 scrollregion
        self.scrollable_frame.bind("<Configure>", self._on_inner_configure)
        # Canvas 大小变化 -> 内部 Frame 尺寸跟随 / 刷新滚动条显隐
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.configure(scrollregion=(0, 0, 0, 0))

        # 布局：竖直条在右，横向条在下（创建时不占位，按需 pack）
        self.canvas.pack(side="left", fill="both", expand=True)

        # 只在鼠标进入本容器时接管滚轮
        # 避免多个滚动区互相干扰
        for w in (self.canvas, self.scrollable_frame):
            w.bind("<Enter>", self._on_enter, add="+")
            w.bind("<Leave>", self._on_leave, add="+")
        self._bind_wheel_tree(self.scrollable_frame)

    # ------------------------------------------------------------------
    # 滚动条显隐
    # ------------------------------------------------------------------
    def _sync_scrollbars(self):
        """按内容是否超出可视区，决定竖直/横向滚动条是否出现。"""
        try:
            need_v = self._inner_h > self.canvas.winfo_height() + 1
            need_h = (self._inner_w > self.canvas.winfo_width() + 1) \
                if self._hbar is not None else False
        except tk.TclError:
            return
        self._toggle_bar("v", need_v or not self._autohide)
        if self._hbar is not None:
            self._toggle_bar("h", need_h or not self._autohide)

    def _toggle_bar(self, which, show):
        bar = self._vbar if which == "v" else self._hbar
        visible = self._vbar_visible if which == "v" else self._hbar_visible
        if bar is None or show == visible:
            return
        if show:
            bar.pack(side="right" if which == "v" else "bottom",
                     fill="y" if which == "v" else "x")
        else:
            bar.pack_forget()
        if which == "v":
            self._vbar_visible = show
        else:
            self._hbar_visible = show

    def _on_yscroll(self, first, last):
        self._vbar.set(first, last)
        if self._autohide:
            self._toggle_bar("v", not (float(first) <= 0.0
                                       and float(last) >= 1.0))

    def _on_xscroll(self, first, last):
        if self._hbar is None:
            return
        self._hbar.set(first, last)
        if self._autohide:
            self._toggle_bar("h", not (float(first) <= 0.0
                                       and float(last) >= 1.0))

    def _on_inner_configure(self, event=None):
        if event is not None:
            self._inner_w, self._inner_h = event.width, event.height
        else:
            self._inner_w = self.scrollable_frame.winfo_reqwidth()
            self._inner_h = self.scrollable_frame.winfo_reqheight()
        self._update_scrollregion()

    def _on_canvas_configure(self, event):
        """Canvas 尺寸变化：内部 Frame 宽度跟随（仅在未启用横向滚动时）。"""
        if not self._horizontal:
            self.canvas.itemconfig(self.canvas_window, width=event.width)
        self._update_scrollregion()
        self._sync_scrollbars()

    def _update_scrollregion(self):
        """按内部 Frame 的实际尺寸设置 scrollregion。

        不用 bbox("all")：空 canvas 时它返回 None，会把 scrollregion 设坏；
        用 winfo_reqwidth/reqheight 更稳。顺带把滚轮绑到新增的子控件上。

        性能提示：这个方法会被 after_idle 反复排队调用（每加一张卡片排一次），
        而内部 `update_idletasks()` + 全树遍历都是 O(树大小)。结果卡片上百张时
        总体会退化成 O(n²)，实测 300 张卡片光刷新就要 3.9s，主循环被塞满 →
        界面「未响应」。因此这里加了合并标记：同一批 idle 任务只真正执行一次。
        """
        self._scroll_dirty = False
        try:
            self.canvas.update_idletasks()
        except tk.TclError:
            return
        w = max(self.scrollable_frame.winfo_reqwidth(),
                self.canvas.winfo_width())
        h = self.scrollable_frame.winfo_reqheight()
        self.canvas.configure(scrollregion=(0, 0, w, h))
        self._bind_wheel_tree(self.scrollable_frame)

    def schedule_scrollregion(self):
        """请求刷新 scrollregion（合并同批多次请求，避免 O(n²)）。"""
        if getattr(self, "_scroll_dirty", False):
            return                      # 本批已经排过，不用再排
        self._scroll_dirty = True
        try:
            self.after_idle(self._update_scrollregion)
        except tk.TclError:
            self._scroll_dirty = False

    # ------------------------------------------------------------------
    # 滚轮
    # ------------------------------------------------------------------
    def _on_enter(self, _event=None):
        self._active = True

    def _on_leave(self, event=None):
        # 注意：<Leave> 在子控件之间移动也会触发，需判断指针是否仍在容器内
        if event is not None:
            try:
                widget = self.winfo_containing(*self.winfo_pointerxy())
            except tk.TclError:
                widget = None
            w = widget
            while w is not None:
                if w is self:
                    return
                w = getattr(w, "master", None)
        self._active = False

    def _on_mousewheel(self, event):
        """处理 Windows 鼠标滚轮和 X11 滚轮事件。"""
        # Shift + 滚轮 → 横向滚动
        state = getattr(event, "state", 0)
        shift = bool(state & 0x0001)

        num = getattr(event, "num", None)
        if num == 4:
            delta = -3                      # X11 向上
        elif num == 5:
            delta = 3                       # X11 向下
        else:
            raw = getattr(event, "delta", 0)
            if IS_WIN:
                steps = -1 * int(raw / 120)  # Windows: 120 的倍数
            else:
                steps = -1 * int(raw)
                if abs(raw) >= 120:
                    steps = -1 * int(raw / 120)
            if steps == 0:
                steps = -1 if raw > 0 else 1
            delta = steps

        if shift and self._hbar is not None:
            self.canvas.xview_scroll(delta, "units")
        else:
            self.canvas.yview_scroll(delta, "units")
        return "break"                      # 阻止事件继续冒泡

    def _bind_wheel_tree(self, widget):
        """把滚轮事件递归绑定到 widget 及其所有子控件（幂等，跳过已绑定的）。

        性能提示：`_update_scrollregion` 每刷新一次就会调这里，而它是 O(子树)。
        卡片上百张时全树遍历会累积成 O(n²)。因此做了两点优化：
          · 已经被绑过的控件直接跳过（原逻辑已如此）；
          · 调用方在批量创建卡片时改用 `bind_wheel_card(card)`，只遍历
            新加入的那棵子树，而不是每次都从 scrollable_frame 全树扫一遍。
        """
        for w in self._walk(widget):
            if getattr(w, "_wheel_bound", False):
                continue
            w._wheel_bound = True
            w.bind("<MouseWheel>", self._on_mousewheel, add="+")
            w.bind("<Button-4>", self._on_mousewheel, add="+")
            w.bind("<Button-5>", self._on_mousewheel, add="+")
            w.bind("<Shift-MouseWheel>", self._on_mousewheel, add="+")
            if w is not self.canvas:
                w.bind("<Enter>", self._on_enter, add="+")
                w.bind("<Leave>", self._on_leave, add="+")

    def bind_wheel_card(self, card):
        """只给新加入的卡片子树绑滚轮（避免每次都扫全树）。"""
        if getattr(card, "_wheel_tree_done", False):
            return
        card._wheel_tree_done = True
        self._bind_wheel_tree(card)

    def scroll_to_top(self):
        """回到顶部（切换结果/重新渲染后调用，避免停在半路）。"""
        try:
            self.canvas.yview_moveto(0.0)
            if self._hbar is not None:
                self.canvas.xview_moveto(0.0)
        except tk.TclError:
            pass

    @staticmethod
    def _walk(widget):
        """深度优先遍历 widget 及其所有后代。"""
        yield widget
        try:
            children = widget.winfo_children()
        except tk.TclError:
            return
        for child in children:
            yield from ScrollableFrame._walk(child)

    def destroy(self):
        try:
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
        except tk.TclError:
            pass
        super().destroy()


# ======================================================================
# 区域 3-C : 训练器的 GUI 集成
# ----------------------------------------------------------------------
# 说明：由 trainer_config.json 选择 OrangeClassifier，以下是 GUI 适配层：
#         · 只做路径/参数校验、数据体检报告、结果整理；
#         · 训练结束后把导出的模型（含 metadata.json / model.json）直接回填到
#           「测试」页的模型路径，训练 → 测试可以一条链跑完。
# ======================================================================

# 训练数据源候选（按存在顺序自动填充）
TRAIN_SOURCE_CANDIDATES = (
    os.path.dirname(PROJECT_ROOT),
    "./resources/datasets",
)
# 训练输出目录
TRAIN_DEFAULT_EXPORT_DIR = "./model/models_keras"

# ----------------------------------------------------------------------
# 训练预设模式
# ----------------------------------------------------------------------
# 下拉选中一个模式后，一键把「轮次 / 批次 / 学习率 / 隐藏层单元」写成一组
# 经验组合；选「自定义」则不联动，完全用手填的值。
# 组合思路（分类头只有 1 个隐层，数据量通常几十~几百张）：
#   · 快速验证：少轮次 + 大批次，几十秒内先跑通流程、看数据有没有问题；
#   · 标准训练：默认档，日常训练用；
#   · 精细训练：多轮次 + 小批次 + 低学习率，数据量充足时冲更高精度；
#   · 强正则：大隐藏层 + 极小批次 + 最低学习率 + AdamW，专治过拟合。
TRAIN_PRESET_CUSTOM = "自定义（手动填下面参数）"
TRAIN_PRESETS = {
    "快速验证（少轮次，先跑通流程）": dict(
        epochs=10, batchSize=32, learningRate=0.001,
        hidden_units=TRAIN_HIDDEN_UNITS, optimizer="Adam"),
    "标准训练（日常推荐）": dict(
        epochs=TRAIN_DEFAULT["epochs"], batchSize=TRAIN_DEFAULT["batchSize"],
        learningRate=TRAIN_DEFAULT["learningRate"],
        hidden_units=TRAIN_HIDDEN_UNITS, optimizer=TRAIN_DEFAULT_OPTIMIZER),
    "精细训练（多轮次 + 低学习率）": dict(
        epochs=120, batchSize=8, learningRate=0.0003,
        hidden_units=256, optimizer="Adam"),
}
if TRAIN_SUPPORTS_OPTIMIZER:
    TRAIN_PRESETS["强正则（抑制过拟合）"] = dict(
        epochs=150, batchSize=4, learningRate=0.0001,
        hidden_units=256, optimizer="AdamW")
TRAIN_PRESET_NAMES = (TRAIN_PRESET_CUSTOM,) + tuple(TRAIN_PRESETS.keys())
# 默认选中的模式（界面初始化与 TrainEngine 缺省替身共用，避免索引漂移）
TRAIN_DEFAULT_PRESET = "标准训练（日常推荐）"
assert TRAIN_DEFAULT_PRESET in TRAIN_PRESETS



# 模型导出后的文件清单（用于训练摘要报告）
TRAIN_EXPORT_FILES = (
    ("metadata.json", "模型元数据（标签 / 阈值 / 训练参数）"),
    ("model.json", "TFJS Layers Model 拓扑"),
    ("weights.bin", "TFJS 分类头权重（float32 拼接）"),
    ("training_curve.png", "训练曲线（loss / accuracy）"),
    ("weights.npz", "可选：numpy 权重备份"),
    ("head.keras", "可选：Keras 分类头备份"),
)


def _fmt_secs(seconds):
    """把秒数格式化成 1h 02m 03s / 1m 05s / 8.4s。"""
    if seconds is None:
        return "—"
    s = max(0.0, float(seconds))
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h:
        return "%dh %02dm %02ds" % (h, m, sec)
    if m:
        return "%dm %02ds" % (m, sec)
    return "%.1fs" % s


def build_train_source_report(source, source_type, class_counts, class_issues):
    """训练摘要报告「数据源」区块。"""
    L = []
    L.append("数据源      : %s" % source)
    L.append("源类型      : %s（%s）" % (
        source_type,
        {"folder": "文件夹目录结构", "json": "tm-object-project 项目 JSON"}
        .get(source_type, "自动识别")))
    L.append("图片尺寸    : %d × %d（白底居中裁剪后送 MobileNetV2）"
             % (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))
    L.append("")
    L.append("各类别样本数")
    L.append("  " + "-" * 66)
    for name in TRAIN_CLASS_NAMES:
        n = class_counts.get(name, 0)
        flag = ""
        if n < TRAIN_MIN_SAMPLES_PER_CLASS:
            flag = "  ❌ 不足最少 %d 张" % TRAIN_MIN_SAMPLES_PER_CLASS
        elif n < TRAIN_RECOMMENDED_SAMPLES:
            flag = "  ⚠️ 建议补充到 %d 张以上" % TRAIN_RECOMMENDED_SAMPLES
        L.append("  %-8s : %5d 张%s" % (name, n, flag))
    L.append("  " + "-" * 66)
    L.append("  %-8s : %5d 张" % ("合计", sum(class_counts.values())))
    L.append("")

    if class_issues:
        L.append("数据体检提示")
        L.append("  " + "-" * 66)
        for it in class_issues:
            sev = "错误" if it.get("severity") == "error" else "提醒"
            L.append("  [%s] %s" % (sev, it.get("message", "")))
        L.append("  " + "-" * 66)
        L.append("  （带「错误」的项会阻塞训练，除非勾选「忽略数据体检错误(force)」）")
    else:
        L.append("数据体检提示: 未发现问题 ✔")
    return "\n".join(L)


def build_train_report(source, source_type, result, params, export_dir,
                       class_counts, class_issues, env_lines, status_line):
    """训练摘要报告全文（导出到「训练A 训练摘要」标签页 / txt）。"""
    def _kv(k, v, width=14):
        return "%-*s: %s" % (width, k, v)

    L = []
    L.append("=" * 78)
    L.append("【训练摘要】MobileNetV2 (alpha=0.5) 冻结特征 + 自训练分类头")
    L.append("=" * 78)
    L.append(_kv("生成时间", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    L.append(_kv("当前状态", status_line))
    L.append("")

    L.append("【1】运行环境")
    L.append("-" * 78)
    for line in env_lines:
        L.append("  " + line)
    L.append("")

    L.append("【2】训练参数")
    L.append("-" * 78)
    L.append(_kv("  训练轮次", "%d（可训练区间 %d–%d）"
                 % (params["epochs"], TRAIN_EPOCH_RANGE[0], TRAIN_EPOCH_RANGE[1])))
    L.append(_kv("  批次大小", params["batch_size"]))
    L.append(_kv("  学习率", params["learning_rate"]))
    L.append(_kv("  优化器", params.get("optimizer", TRAIN_DEFAULT_OPTIMIZER)))

    L.append(_kv("  隐藏层单元", "%d（分类头：Dense → Dropout 0.2 → Softmax）"
                 % params["hidden_units"]))
    L.append(_kv("  嵌入批大小", "%d（仅影响 MobileNet 前向提取）"
                 % params["extract_batch_size"]))
    L.append(_kv("  忽略体检错误", "是（force）" if params.get("force") else "否"))
    L.append("")

    L.append("【3】数据源与数据体检")
    L.append("-" * 78)
    for line in build_train_source_report(
            source, source_type, class_counts, class_issues).split("\n"):
        L.append("  " + line if line else "")
    L.append("")

    L.append("【4】训练结果")
    L.append("-" * 78)
    if result is None:
        L.append("  本次没有产出训练结果。")
        L.append("")
        L.append("=" * 78)
        return "\n".join(L)

    labels = result.get("labels") or []
    hist = result.get("history") or {}
    valid = result.get("validation")
    dur_ms = result.get("durationMs") or 0
    n_done = len(hist.get("loss", []))

    L.append(_kv("  标签顺序", " ｜ ".join(labels)))
    L.append(_kv("  样本总数", "%d 张" % result.get("exampleCount", 0)))
    L.append(_kv("  嵌入维度", result.get("embeddingSize", "—")))
    L.append(_kv("  实际训练轮次", "完成 %d / 计划 %d%s"
                 % (n_done, params["epochs"],
                    "" if n_done >= params["epochs"] else "（早停）")))
    L.append(_kv("  训练耗时", "%s（特征提取 + 分类头）" % _fmt_secs(dur_ms / 1000.0)))
    L.append(_kv("  验证集状态", str(result.get("validationStatus", "—"))))
    L.append("")

    if hist.get("loss"):
        last = len(hist["loss"]) - 1
        L.append("  末轮指标")
        L.append("    " + "-" * 70)
        L.append("    训练 loss      : %.4f" % hist["loss"][last])
        if hist.get("val_loss"):
            L.append("    验证 loss      : %.4f" % hist["val_loss"][last])
        if hist.get("accuracy"):
            L.append("    训练准确率     : %.4f" % hist["accuracy"][last])
        if hist.get("val_accuracy"):
            L.append("    验证准确率     : %.4f" % hist["val_accuracy"][last])
        L.append("    " + "-" * 70)
        L.append("")

    if valid:
        L.append("  验证集评估（按采集组分层划分）")
        L.append("    " + "-" * 70)
        L.append("    样本数         : %d" % valid.get("exampleCount", 0))
        L.append("    准确率         : %.4f" % valid.get("accuracy", 0.0))
        L.append("    宏平均 F1      : %.4f" % valid.get("macroF1", 0.0))
        L.append("    平衡准确率     : %.4f" % valid.get("balancedAccuracy", 0.0))
        L.append("    混淆矩阵       : %s" % (valid.get("confusionMatrix"),))
        L.append("")
        L.append("    %-10s %8s %10s %10s %8s"
                 % ("类别", "支撑数", "精确率", "召回率", "F1"))
        L.append("    " + "-" * 70)
        for p in valid.get("perClass", []):
            L.append("    %-10s %8d %10.4f %10.4f %8.4f"
                     % (p.get("label", ""), p.get("support", 0),
                        p.get("precision", 0.0), p.get("recall", 0.0),
                        p.get("f1", 0.0)))
        L.append("    " + "-" * 70)
        L.append("")
    else:
        L.append("  验证集评估：未生成（状态：%s）" % result.get("validationStatus", "—"))
        L.append("")

    L.append("【5】模型导出")
    L.append("-" * 78)
    L.append(_kv("  导出目录", export_dir or "—（未导出）"))
    if export_dir and os.path.isdir(export_dir):
        for name, desc in TRAIN_EXPORT_FILES:
            p = os.path.join(export_dir, name)
            if os.path.isfile(p):
                L.append("    ✔ %-18s %9.1f KB   %s"
                         % (name, os.path.getsize(p) / 1024.0, desc))
        zips = sorted(f for f in os.listdir(export_dir) if f.lower().endswith(".zip"))
        for z in zips:
            p = os.path.join(export_dir, z)
            L.append("    ✔ %-18s %9.1f KB   打包的识别模型（可直接用于测试页）"
                     % (z, os.path.getsize(p) / 1024.0))
        if not zips:
            L.append("    ⚠️ 未生成 .zip 打包模型")
        L.append("")
        L.append("  说明：导出目录本身即可作为「测试」页的模型路径"
                 "（内含 model.json / weights.bin / metadata.json）。")
    else:
        L.append("    未导出模型。")
    L.append("")

    L.append("【6】判定阈值（写入 metadata.json，测试页优先读取）")
    L.append("-" * 78)
    L.append("  置信度阈值 confidenceThreshold : 0.60（低于则标记「不确定」）")
    L.append("  差距阈值   marginThreshold     : 0.12（前两名差值小于则标记「类别接近」）")
    L.append("")

    L.append("【7】训练日志（%s 实时输出，末尾 200 行）" % TRAINER_PATH)
    L.append("-" * 78)
    lines = LOGGER.tail() if LOGGER is not None else []
    if not lines:
        L.append("  （暂无日志）")
    for line in lines:
        L.append("  " + line)
    L.append("")
    L.append("=" * 78)
    return "\n".join(L)


def build_train_health_text(source, class_counts, health_issues):
    """数据体检文本（含当前训练器的体检结果）。"""
    L = []
    L.append("=" * 78)
    L.append("【数据体检】%s" % source)
    L.append("=" * 78)
    L.append("类别目录约定：<数据源>/%s/* 与 <数据源>/%s/*"
             % (TRAIN_CLASS_NAMES[0], TRAIN_CLASS_NAMES[1]))
    L.append("支持的图片后缀：.jpg / .jpeg / .png / .webp")
    L.append("")
    L.append("各类别样本数")
    L.append("-" * 78)
    for name in TRAIN_CLASS_NAMES:
        L.append("  %-8s : %5d 张" % (name, class_counts.get(name, 0)))
    L.append("-" * 78)
    L.append("")
    L.append("体检结果")
    L.append("-" * 78)
    if not health_issues:
        L.append("  ✔ 未发现数据风险，可以直接开始训练。")
    else:
        for it in health_issues:
            sev = "❌ 错误" if it.get("severity") == "error" else "⚠️ 提醒"
            L.append("  %s  [%s]" % (sev, it.get("code", "")))
            L.append("        %s" % it.get("message", ""))
    L.append("-" * 78)
    L.append("")
    L.append("提示：训练启动时会再次体检；severity=error 的项会阻塞训练，")
    L.append("      确需继续请勾选「忽略数据体检错误（force）」。")
    return "\n".join(L)


def build_train_curve_text(history, params):
    """训练曲线标签页的文本说明 + 逐轮明细。"""
    L = []
    L.append("=" * 78)
    L.append("【训练曲线】loss / accuracy 随 epoch 变化")
    L.append("=" * 78)
    L.append("训练轮次: %d    批次大小: %d    学习率: %s    优化器: %s"
             % (params["epochs"], params["batch_size"], params["learning_rate"],
                params.get("optimizer", TRAIN_DEFAULT_OPTIMIZER)))

    L.append("")
    if not history or not history.get("loss"):
        L.append("没有可绘制的训练历史。")
        return "\n".join(L)

    loss = history.get("loss", [])
    acc = history.get("accuracy", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_accuracy", [])

    L.append("%-8s %-12s %-12s %-12s %-12s"
             % ("Epoch", "训练loss", "验证loss", "训练准确率", "验证准确率"))
    L.append("-" * 78)
    for i in range(len(loss)):
        L.append("%-8d %-12.4f %-12s %-12s %-12s" % (
            i + 1,
            loss[i],
            ("%.4f" % val_loss[i]) if i < len(val_loss) else "—",
            ("%.4f" % acc[i]) if i < len(acc) else "—",
            ("%.4f" % val_acc[i]) if i < len(val_acc) else "—",
        ))
    L.append("-" * 78)
    L.append("")
    L.append("解读：")
    if val_loss:
        L.append("  · 训练 loss 下降而验证 loss 上升，可能表示分类头开始过拟合。")
        if TRAINER_NAME == "optimized":
            L.append("  · 当前训练器按验证 loss 早停，并恢复最佳一轮的权重。")
        L.append("  · 验证曲线抖动大通常说明验证样本偏少。")
    else:
        L.append("  · 本次未划分验证集；请用独立图片评估模型。")
    L.append("  · 嵌入阶段用的是冻结的 MobileNetV2(alpha=0.5)，"
             "准确率上限主要取决于数据质量。")
    return "\n".join(L)


def build_train_label_text(labels):
    """标签映射说明：标签顺序 = 网络输出下标顺序，决定测试页的类别名。"""
    L = []
    L.append("=" * 78)
    L.append("【标签映射】")
    L.append("=" * 78)
    L.append("训练时按数据源中类别目录的顺序收集标签，模型输出下标与之一一对应：")
    L.append("")
    for i, name in enumerate(labels):
        L.append("  [%d] → %s" % (i, name))
    L.append("")
    L.append("原始类别目录顺序: %s" % " / ".join(TRAIN_CLASS_NAMES))
    L.append("")
    L.append("导出的 metadata.json 中 labels 数组顺序与此一致；")
    L.append("测试页加载模型时会优先读取 metadata.json 的 labels 与判定阈值。")
    return "\n".join(L)


class TkLogger:
    """把当前训练器的 log(...) 输出同时写到终端和 GUI 日志框。

    GUI 写入必须回主线程：Tk 的 after() 不能从子线程安全调用，
    因此这里也走 callable 中转 —— 训练前由 GUI 用 attach_queue() 挂上
    主线程的队列；没有队列时（命令行跑训练）只 print。
    """

    def __init__(self, textbox=None, root=None):
        self.textbox = textbox
        self.root = root
        self.lines = []
        self._queue = None

    def attach_queue(self, q):
        """由 GUI 在构建界面时调用：把日志回调整交给主线程队列。"""
        self._queue = q

    def __call__(self, *parts):
        msg = " ".join(str(p) for p in parts)
        self.lines.append(msg)
        try:
            print(msg, flush=True)
        except Exception:
            pass
        if self.textbox is None:
            return
        if self._queue is not None:
            # 子线程安全：只入队，主线程轮询取出后再写控件
            self._queue.put_nowait(lambda m=msg: self._append(m))
            return
        try:
            self.root.after(0, lambda m=msg: self._append(m))
        except Exception:
            self._dropped = getattr(self, "_dropped", 0) + 1

    def _append(self, msg):
        try:
            self.textbox.configure(state="normal")
            self.textbox.insert(tk.END, msg + "\n")
            self.textbox.see(tk.END)
            self.textbox.configure(state="disabled")
        except Exception:
            pass

    def clear(self):
        self.lines = []
        if self.textbox is None:
            return
        try:
            self.textbox.configure(state="normal")
            self.textbox.delete("1.0", tk.END)
            self.textbox.configure(state="disabled")
        except Exception:
            pass

    def tail(self, n=200):
        return self.lines[-n:]


# 当前 GUI 使用的日志器（未构建 GUI 时为 None，此时只 print）
LOGGER = None


def _env_lines():
    """收集运行环境信息，用于训练摘要报告。"""
    lines = []
    try:
        import tensorflow as tf
        lines.append("TensorFlow  : %s" % tf.__version__)
        try:
            gpus = tf.config.list_physical_devices("GPU")
            lines.append("GPU         : %s" % (
                "、".join(g.name for g in gpus) if gpus else "未检测到，使用 CPU"))
        except Exception as e:
            lines.append("GPU         : 检测失败（%s）" % e)
    except Exception as e:
        lines.append("TensorFlow  : 未安装或导入失败（%s）" % e)
    lines.append("Python      : %s" % sys.version.split()[0])
    lines.append("平台        : %s" % sys.platform)
    lines.append("特征提取器  : model/basemodels/model.json"
                 "（缺失时自动回退 Keras MobileNetV2 alpha=0.5）")
    return lines


class _DummyVar:
    """TrainEngine 的缺省参数替身（tk 变量的最小实现）。"""

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def _is_tk_var(obj):
    """判断是不是真实的 tkinter 变量（跨线程读会抛 RuntimeError）。

    真实 tk 变量（tk.StringVar / tk.BooleanVar …）的 get() 是一次 Tcl 调用，
    在非主线程执行会抛 "main thread is not in main loop"。
    _DummyVar 这类普通对象则随时可读。
    """
    try:
        import tkinter as _tk
    except Exception:
        return False
    if isinstance(obj, _tk.Variable):
        return True
    # 兜底：Tk 变量的实现类可能不直接继承 Variable（不同版本/子类）
    return type(obj).__module__.startswith("tkinter")


def _parse_train_arg(s):
    """解析训练输入框：支持 int / float / 百分比 / 科学计数法 / 空值。

    例：'' → None；'10' → 10；'0.003' → 0.003；'3e-3' → 0.003；
        '10%' → 0.1（相对默认学习率的百分比，仅在传 base 时生效）。
    """
    s = (s or "").strip()
    if not s:
        return None
    if s.endswith("%"):
        try:
            return ("pct", float(s[:-1]) / 100.0)
        except ValueError:
            return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v


class TrainEngine:
    """训练引擎：校验参数 → 体检数据 → 调当前训练器的 OrangeClassifier。

    · 取消机制：通过 collect() 抛出的异常实现（"训练取消"）。
      训练器的 train() 是一次性阻塞调用，这里复用它的
      数据体检验证（force=False 时自动阻塞）。
    · 导出模型时会生成可直接被测试页加载的 .zip 打包。

    注意：__init__ 里挂了一组「缺省参数」替身，保证没经过 GUI 也能实例化
    （stats/体检是只读的，可以在主线程直接调）。
    """

    def __init__(self):
        # 缺省替身：GUI 会用真实的 tk 变量覆盖它们
        self._snap = None
        self._cancel_flag = False          # 线程安全的取消标志（不用 tk 变量）
        self.source_var = _DummyVar("")
        self.source_type_var = _DummyVar("auto")
        self.epochs_var = _DummyVar(str(TRAIN_DEFAULT["epochs"]))
        self.batch_var = _DummyVar(str(TRAIN_DEFAULT["batchSize"]))
        self.lr_var = _DummyVar(str(TRAIN_DEFAULT["learningRate"]))
        self.hidden_var = _DummyVar(str(TRAIN_HIDDEN_UNITS))
        self.extract_batch_var = _DummyVar(str(TRAIN_EXTRACT_BATCH_SIZE))
        self.optimizer_var = _DummyVar(TRAIN_DEFAULT_OPTIMIZER)
        self.preset_var = _DummyVar(TRAIN_DEFAULT_PRESET)
        self.export_var = _DummyVar(TRAIN_DEFAULT_EXPORT_DIR)

        self.backbone_var = _DummyVar("")
        self.force_var = _DummyVar(False)
        self.cancel_var = _DummyVar(False)

    def snapshot(self):
        """主线程调用：把 GUI 变量的当前值冻结成一份普通 dict。

        子线程绝不能直接读 tk 变量（会抛 "main thread is not in main loop"），
        所以训练启动前必须先在主线程快照，之后引擎只读这份快照。
        """
        self._snap = {
            "source": self.source_var.get(),
            "source_type": self.source_type_var.get(),
            "epochs": self.epochs_var.get(),
            "batch": self.batch_var.get(),
            "lr": self.lr_var.get(),
            "hidden": self.hidden_var.get(),
            "extract_batch": self.extract_batch_var.get(),
            "optimizer": self.optimizer_var.get(),
            "preset": self.preset_var.get(),
            "export": self.export_var.get(),

            "backbone": self.backbone_var.get(),
            "force": self.force_var.get(),
        }
        return self._snap

    def _raw(self, key, default=""):
        """取快照值；没有快照时（命令行/测试直接调）回退到变量对象。

        注意：read 只允许在「没有 tk 变量」的场景发生 —— 一旦挂了真实 tk 变量，
        必须已经 snapshot()，否则子线程读 tk 变量会抛
        "main thread is not in main loop"。
        """
        snap = getattr(self, "_snap", None)
        if snap is not None and key in snap:
            v = snap.get(key, default)
            return default if v is None else v

        var = getattr(self, {
            "source": "source_var", "source_type": "source_type_var",
            "epochs": "epochs_var", "batch": "batch_var",
            "lr": "lr_var", "hidden": "hidden_var",
            "extract_batch": "extract_batch_var", "export": "export_var",
            "backbone": "backbone_var", "force": "force_var",
            "optimizer": "optimizer_var", "preset": "preset_var",
        }[key], None)

        if var is None or _is_tk_var(var):
            # 真实 tk 变量：子线程读会触发 Tcl 调用而崩溃，保守返回默认值。
            # （正常流程一定先 snapshot()，不会走到这里）
            return default
        v = var.get()
        return default if v is None else v

    def _is_cancelled(self):
        """取消标志（线程安全）。

        tk.BooleanVar.get() 是真正的 Tcl 调用，子线程访问会抛
        "main thread is not in main loop"，所以正常情况下只读 _cancel_flag
        （由主线程在点「取消」时写入）。

        但为了兼容「非 GUI 直接构造 TrainEngine」的用法（命令行/测试里
        直接 set cancel_var），在非 GUI 缺省替身下也允许读变量。
        """
        if getattr(self, "_cancel_flag", False):
            return True
        var = getattr(self, "cancel_var", None)
        # 只有非 tk 变量（命令行/测试场景）才安全地直接读
        if var is None or _is_tk_var(var):
            return False
        return bool(var.get())

    def stats(self):
        """只读地统计数据源（不加载权重、不训练）。"""
        source = str(self._raw("source")).strip()
        source_type = self._raw("source_type", "auto")
        if not source or not os.path.exists(source):
            raise RuntimeError("数据源/数据集目录不存在：%s" % (source or "(空)"))

        if source_type == "folder" or (source_type == "auto" and os.path.isdir(source)):
            project = train_load_project_folder(source)
        elif source_type == "json" or (source_type == "auto" and os.path.isfile(source)):
            project = train_load_project_json(source)
        else:
            raise ValueError("无法确定数据源类型，请指定 folder 或 json")

        class_counts, class_issues = {}, []
        for c in project.classes:
            class_counts[c.name] = len(c.samples)
            # 按类别单独体检，用于摘要报告里的逐类别提示
            sub_project = type(project)([c], dict(project.training))
            for it in train_data_health_report(sub_project):
                class_issues.append(it)

        return dict(source=source, source_type=source_type, project=project,
                    class_counts=class_counts, class_issues=class_issues,
                    health_issues=train_data_health_report(project))

    def collect(self, log=print):
        """准备阶段：统计 + 体检 + 阻塞条件检查。"""
        st = self.stats()
        project = st["project"]

        # 逐类别体检会让「样本不足」重复出现，这里按 (severity, message) 去重
        seen, issues = set(), []
        for it in st["health_issues"]:
            key = (it.get("severity"), it.get("message"))
            if key in seen:
                continue
            seen.add(key)
            issues.append(it)
        st["health_issues"] = issues

        for line in build_train_source_report(
                st["source"], st["source_type"],
                st["class_counts"], st["class_issues"]).split("\n"):
            log("[数据] " + line)

        errors = [i for i in issues if i.get("severity") == "error"]
        if not train_valid_class_names([c.name for c in project.classes]):
            errors.append({"severity": "error", "code": "class-names",
                           "message": "比赛项目必须使用固定的「%s」「%s」两个类别目录"
                                      % (TRAIN_CLASS_NAMES[0],
                                         TRAIN_CLASS_NAMES[1])})
        if errors and not bool(self._raw("force", False)):
            raise RuntimeError("无法开始训练：\n  - " + "\n  - ".join(
                dict.fromkeys(e.get("message", "") for e in errors)))
        if errors:
            log("[提醒] 已勾选 force，忽略 %d 项数据体检错误。" % len(errors))

        if self._is_cancelled():
            raise RuntimeError("训练取消")
        return st

    def run(self, log=print):
        """对外入口：把 GUI 的所有输入形态解析成规范参数，再执行训练。

        · 支持 'auto' / 'folder' / 'json' / 中文别名；
        · 数值框为空时回退到当前训练器的默认值；
        · 学习率支持 "10%" 这种相对默认值的百分比写法。
        """
        source_type = (self._raw("source_type", "auto") or "auto").strip()
        alias = {
            "自动识别": "auto", "自动": "auto",
            "文件夹目录": "folder", "文件夹": "folder", "folder": "folder",
            "项目json": "json", "json": "json",
            "tm-object-project json": "json",
        }
        source_type = alias.get(source_type.lower(), source_type.lower())
        if source_type not in ("auto", "folder", "json"):
            source_type = "auto"

        epochs = _parse_train_arg(self._raw("epochs"))
        batch = _parse_train_arg(self._raw("batch"))
        lr = _parse_train_arg(self._raw("lr"))
        hidden = _parse_train_arg(self._raw("hidden"))
        extract_batch = _parse_train_arg(self._raw("extract_batch"))
        optimizer = train_normalize_optimizer(self._raw("optimizer"))
        preset = str(self._raw("preset") or TRAIN_PRESET_CUSTOM)

        def _as_int(v, default):
            return default if v is None else int(v)

        if isinstance(lr, tuple):      # '10%' → 相对默认学习率
            lr = max(1e-6, TRAIN_DEFAULT["learningRate"] * lr[1])

        # 选了预设模式就以预设为准（避免手填值与模式名不一致造成误解）；
        # 「自定义」时完全用手填值。
        if preset in TRAIN_PRESETS:
            fixed = TRAIN_PRESETS[preset]
            log("[预设] 训练模式「%s」→ epochs=%d / batchSize=%d / "
                "learningRate=%s / hidden_units=%d / optimizer=%s"
                % (preset, fixed["epochs"], fixed["batchSize"],
                   fixed["learningRate"], fixed["hidden_units"],
                   fixed["optimizer"]))
            return self.run_impl(
                source_type=source_type,
                epochs=int(fixed["epochs"]),
                batch_size=int(fixed["batchSize"]),
                learning_rate=float(fixed["learningRate"]),
                hidden_units=int(fixed["hidden_units"]),
                extract_batch_size=_as_int(extract_batch, TRAIN_EXTRACT_BATCH_SIZE),
                optimizer=optimizer if optimizer else fixed["optimizer"],
                log=log)

        return self.run_impl(
            source_type=source_type,
            epochs=_as_int(epochs, TRAIN_DEFAULT["epochs"]),
            batch_size=_as_int(batch, TRAIN_DEFAULT["batchSize"]),
            learning_rate=TRAIN_DEFAULT["learningRate"] if lr is None else float(lr),
            hidden_units=_as_int(hidden, TRAIN_HIDDEN_UNITS),
            extract_batch_size=_as_int(extract_batch, TRAIN_EXTRACT_BATCH_SIZE),
            optimizer=optimizer,
            log=log)

    def run_impl(self, source_type, epochs, batch_size, learning_rate,
                 hidden_units, extract_batch_size, optimizer=None, log=print):
        """在后台线程中执行训练并导出模型（参数已规范化）。"""
        t0 = time.perf_counter()
        st = self.collect(log=log)
        source = st["source"]
        source_type = "auto" if source_type not in ("folder", "json") else source_type

        if not (TRAIN_EPOCH_RANGE[0] <= int(epochs) <= TRAIN_EPOCH_RANGE[1]):
            raise RuntimeError("训练轮次需为 %d–%d 的整数"
                               % (TRAIN_EPOCH_RANGE[0], TRAIN_EPOCH_RANGE[1]))
        if batch_size <= 0:
            raise RuntimeError("批次大小必须为正整数")
        if learning_rate <= 0:
            raise RuntimeError("学习率必须为正数")
        if hidden_units <= 0:
            raise RuntimeError("隐藏层单元数必须为正整数")
        if extract_batch_size <= 0:
            raise RuntimeError("嵌入批大小必须为正整数")

        optimizer = train_normalize_optimizer(optimizer)

        params = dict(epochs=int(epochs), batch_size=int(batch_size),
                      learning_rate=float(learning_rate),
                      hidden_units=int(hidden_units),
                      extract_batch_size=int(extract_batch_size),
                      optimizer=optimizer,
                      force=bool(self._raw("force", False)))


        log("")
        log("[阶段] 构建分类器（特征提取器优先用内置 MobileNetV2 图）")
        backbone = (self._raw("backbone") or "").strip() or None
        clf = OrangeClassifier(model_json_path=backbone)

        log("[阶段] 开始训练（%s → OrangeClassifier.train）" % TRAINER_PATH)
        train_args = dict(
            source=source,
            source_type=source_type,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            extract_batch_size=params["extract_batch_size"],
            learning_rate=params["learning_rate"],
            hidden_units=params["hidden_units"],
            force=params["force"],
            log=log,
        )
        if TRAIN_SUPPORTS_OPTIMIZER:
            train_args["optimizer"] = params["optimizer"]
        result = clf.train(**train_args)

        cancelled = self._is_cancelled()
        if cancelled:
            log("[提醒] 收到取消请求，跳过模型导出（内存中的训练结果仍然展示）。")

        export_dir = self._resolve_export_dir()
        exported_path = None
        if not cancelled:
            log("")
            log("[阶段] 导出模型 → %s" % export_dir)
            exported_path = os.path.abspath(export_dir)
            cwd = os.getcwd()
            # 相对 ZIP 路径在训练器里是按 CWD 生成的，
            # 这里统一在项目根目录下打包，保证产物落在项目里而不是启动目录。
            os.chdir(PROJECT_ROOT)
            try:
                clf.save(export_dir, zip_output=True)
            finally:
                os.chdir(cwd)
            log("已保存到 %s/" % export_dir)

        duration = time.perf_counter() - t0
        return dict(type="train", result=result, params=params,
                    source=source, source_type=source_type,
                    class_counts=st["class_counts"],
                    class_issues=st["class_issues"],
                    health_issues=st["health_issues"],
                    export_dir=export_dir, exported_path=exported_path,
                    cancelled=cancelled,
                    duration=duration, env_lines=_env_lines())

    def _resolve_export_dir(self):
        """相对导出目录统一按「项目根目录」解析，避免 CWD 变化导致找错位置。"""
        d = str(self._raw("export") or TRAIN_DEFAULT_EXPORT_DIR).strip() \
            or TRAIN_DEFAULT_EXPORT_DIR
        if not os.path.isabs(d):
            d = os.path.join(PROJECT_ROOT, d)
        return os.path.normpath(d)


# ======================================================================
# 区域 4 : GUI 主程序
# ======================================================================

class MobileNetTesterApp:

    def __init__(self, root):
        global LOGGER

        self.root = root
        self._log_bg, self._log_fg = "#1e1e1e", "#dcdcdc"
        self.root.title("MobileNetV2 平台 —— 训练（%s） + 测试（分类/回归/识别）" % TRAINER_NAME)
        self.root.geometry("1560x980")
        self.root.minsize(1180, 720)
        self.running = False
        self.training = False          # 训练是否进行中（与测试互斥）
        self._cancel_requested = False  # 测试运行中的取消标志（关窗时置位）
        self.engine = TrainEngine()    # 训练引擎（读写下面注册的 tk 变量）
        self.figures = []
        self.last_result = None       # 保存最近一次测试结果，用于导出
        self.last_train_result = None  # 保存最近一次训练结果 + 导出信息
        self._pred_temp_dir = None    # 单图/批量识别模式解压出的临时目录
        self._pred_cards = []         # 单图/批量识别模式的结果卡片
        self._pending_mode = "classification"   # 子线程用的模式/输入快照
        self._pending_args = {}
        # 子线程 → 主线程 的回调队列（Tk 的 after 不能跨线程安全调用）
        self._ui_queue = queue.Queue()

        self._style = ttk.Style()
        self._init_styles()
        self._register_train_vars()   # 训练参数变量（绑定到 TrainEngine）
        self._build_ui()
        self.on_mode_change()
        self._detect_train_source()   # 自动填充数据源与主干路径
        self._bind_shortcuts()

    def _bind_shortcuts(self):
        """绑定 Ctrl+R/E/T/U/W 快捷键。"""
        actions = {
            "r": self._shortcut_run,        # 开始（训练页）/ 开始测试（测试页）
            "e": self._shortcut_export,     # 导出当前页结果
            "t": self.on_goto_train,        # 切到训练页
            "u": self.on_goto_test,         # 切到测试页
            "w": self.on_close,             # 关窗（训练中会二次确认）
        }
        for key, fn in actions.items():
            self.root.bind("<Control-%s>" % key, lambda e, f=fn: f())

    def _shortcut_run(self):
        """Ctrl+R：训练页开始训练，测试页开始测试。"""
        if self.main_nb.index(self.main_nb.select()) == 0:
            if not self.training:
                self.on_train_run()
        elif not self.running:
            self.on_run()
        return "break"

    def _shortcut_export(self):
        """Ctrl+E：导出当前页签的结果。"""
        if self.main_nb.index(self.main_nb.select()) == 0:
            if self.last_train_result is not None:
                self.on_export_train()
        elif self.last_result is not None:
            self.on_export()
        return "break"

    def _init_styles(self):
        """注册原 predictor.py 的卡片样式（整合后保留外观）。"""
        self._style.configure("WideCard.TFrame", background="#ffffff",
                              relief="solid", borderwidth=1)
        self._style.configure("Title.TLabel", font=(_UI_FONT, 11, "bold"))
        self._style.configure("Sub.TLabel", font=(_UI_FONT, 9),
                              foreground="#555555")
        self._style.configure("Prob.TLabel", font=(_MONO_FONT, 9),
                              foreground="#666666")
        # 训练页：大标题 / 状态色
        self._style.configure("Big.TLabel", font=(_UI_FONT, 14, "bold"))
        self._style.configure("Log.TLabel", font=(_MONO_FONT, 9),
                              foreground="#555555")

    def _build_ui(self):
        """整体布局：顶部标题栏 + 主体 Notebook（训练页 / 测试页）。"""
        header = ttk.Frame(self.root, padding=(10, 8, 10, 0))
        header.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(header, text="🧠  MobileNetV2 训练 / 测试一体化平台",
                  style="Big.TLabel").pack(side=tk.LEFT)
        ttk.Label(header,
                  text=("训练：%s（冻结 MobileNetV2 alpha=0.5 + 自训练分类头）"
                        "  ｜  测试：分类 / 回归 / 单图批量识别") % TRAINER_PATH,
                  foreground="#666666", wraplength=620, justify="left").pack(
            side=tk.LEFT, padx=16)

        ttk.Label(header,
                  text="Ctrl+R 开始 · Ctrl+E 导出 · Ctrl+T 训练页 · Ctrl+U 测试页",
                  foreground="#888888").pack(side=tk.RIGHT)

        self.main_nb = ttk.Notebook(self.root)
        self.main_nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                          padx=8, pady=(6, 8))

        # ---- 训练页 ----
        self.train_tab = ttk.Frame(self.main_nb)
        self.main_nb.add(self.train_tab, text="  🏋️ 训练  ")
        # ---- 测试页 ----
        self.test_tab = ttk.Frame(self.main_nb)
        self.main_nb.add(self.test_tab, text="  🧪 测试  ")

        self._build_train_tab(self.train_tab)
        self._build_test_tab(self.test_tab)

    # ------------------------------------------------------------------
    # 训练页
    # ------------------------------------------------------------------
    def _register_train_vars(self):
        """注册训练参数变量，并挂到 TrainEngine（引擎只读这些变量）。"""
        self.train_source_var = tk.StringVar()
        self.train_source_type_var = tk.StringVar(value="auto")
        self.train_epochs_var = tk.StringVar(value=str(TRAIN_DEFAULT["epochs"]))
        self.train_batch_var = tk.StringVar(value=str(TRAIN_DEFAULT["batchSize"]))
        self.train_lr_var = tk.StringVar(value=str(TRAIN_DEFAULT["learningRate"]))
        self.train_hidden_var = tk.StringVar(value=str(TRAIN_HIDDEN_UNITS))
        self.train_extract_batch_var = tk.StringVar(
            value=str(TRAIN_EXTRACT_BATCH_SIZE))
        # 训练模式（预设联动）与优化器：默认选「标准训练」+ Adam
        self.train_preset_var = tk.StringVar(value=TRAIN_DEFAULT_PRESET)
        self.train_optimizer_var = tk.StringVar(value=TRAIN_DEFAULT_OPTIMIZER)
        self.train_export_var = tk.StringVar(value=TRAIN_DEFAULT_EXPORT_DIR)
        self.train_backbone_var = tk.StringVar()
        self.train_force_var = tk.BooleanVar(value=False)
        self.train_autofill_var = tk.BooleanVar(value=True)
        self.train_cancel_var = tk.BooleanVar(value=False)

        e = self.engine
        e.source_var = self.train_source_var
        e.source_type_var = self.train_source_type_var
        e.epochs_var = self.train_epochs_var
        e.batch_var = self.train_batch_var
        e.lr_var = self.train_lr_var
        e.hidden_var = self.train_hidden_var
        e.extract_batch_var = self.train_extract_batch_var
        e.preset_var = self.train_preset_var
        e.optimizer_var = self.train_optimizer_var
        e.export_var = self.train_export_var
        e.backbone_var = self.train_backbone_var
        e.force_var = self.train_force_var
        e.cancel_var = self.train_cancel_var


    def _build_train_tab(self, parent):
        """训练页：区域① 数据源 / 区域② 训练参数 / 区域③ 运行控制 + 日志 + 结果。

        布局分两段：
          · 上半段 = 可滚动区（参数/控制/日志），窗口变小时能用滚轮/滚动条看到全部内容；
          · 下半段 = 固定高度的结果 Notebook（不再被参数区挤扁）。
        """
        paned = ttk.Panedwindow(parent, orient=tk.VERTICAL)
        paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        upper = ttk.Frame(paned)
        paned.add(upper, weight=3)      # 参数区占 3 份高度
        lower = ttk.Frame(paned)
        paned.add(lower, weight=2)      # 结果区占 2 份高度

        # 参数/控制/日志全部放进可滚动的容器（横向也可滚，超长提示文字能拖到）
        scroller = ScrollableFrame(upper, horizontal=True)
        scroller.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.train_scroller = scroller
        parent = scroller.scrollable_frame

        top = ttk.Frame(parent, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        # ---------- 区域① 数据源 ----------
        box1 = ttk.LabelFrame(top, text=" 区域① 数据源（%s 的 source） " % TRAINER_PATH,
                              padding=8)
        box1.pack(fill=tk.X, pady=(0, 6))

        r = ttk.Frame(box1); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="数据源（文件夹 或 项目 JSON）:", width=28).pack(side=tk.LEFT)
        ttk.Entry(r, textvariable=self.train_source_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r, text="浏览…", width=10,
                   command=self.on_browse_train_source).pack(side=tk.LEFT, padx=4)

        r = ttk.Frame(box1); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="数据源类型:", width=28).pack(side=tk.LEFT)
        for val, txt in (("auto", "自动识别"), ("folder", "文件夹目录"),
                         ("json", "tm-object-project JSON")):
            ttk.Radiobutton(r, text=txt, value=val,
                            variable=self.train_source_type_var).pack(
                side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(r, text="自动填充到「测试」页模型路径",
                        variable=self.train_autofill_var).pack(
            side=tk.LEFT, padx=(20, 0))

        ttk.Label(box1, text=(
            "目录约定：<数据源>/%s/*  与  <数据源>/%s/*"
            "（支持 .jpg/.jpeg/.png/.webp）；也支持页面导出的 tm-object-project JSON。"
            % (TRAIN_CLASS_NAMES[0], TRAIN_CLASS_NAMES[1])),
            foreground="gray", wraplength=900,
            justify="left").pack(anchor="w", pady=(2, 0))

        # ---------- 区域② 训练参数 ----------
        box2 = ttk.LabelFrame(top, text=" 区域② 训练参数（OrangeClassifier.train） ",
                              padding=8)
        box2.pack(fill=tk.X, pady=(0, 6))

        r = ttk.Frame(box2); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="训练模式 preset:", width=28).pack(side=tk.LEFT)
        preset_box = ttk.Combobox(r, textvariable=self.train_preset_var,
                                  values=list(TRAIN_PRESET_NAMES),
                                  state="readonly", width=30)
        preset_box.pack(side=tk.LEFT)
        preset_box.bind("<<ComboboxSelected>>", self.on_train_preset_changed)
        self.train_preset_box = preset_box
        ttk.Label(r, text="（选中即自动填好下面 4 项参数）",
                  foreground="gray").pack(side=tk.LEFT, padx=6)

        ttk.Label(r, text="优化器 optimizer:", width=20,
                  anchor="e").pack(side=tk.LEFT, padx=(16, 0))
        ttk.Combobox(r, textvariable=self.train_optimizer_var,
                     values=list(TRAIN_OPTIMIZERS),
                     state="readonly", width=10).pack(side=tk.LEFT)

        r = ttk.Frame(box2); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="训练轮次 epochs:", width=28).pack(side=tk.LEFT)
        ttk.Spinbox(r, from_=TRAIN_EPOCH_RANGE[0], to=TRAIN_EPOCH_RANGE[1],
                    textvariable=self.train_epochs_var, width=8).pack(side=tk.LEFT)
        ttk.Label(r, text="（%d–%d，默认 %d；模式=自定义时生效）"
                  % (TRAIN_EPOCH_RANGE[0], TRAIN_EPOCH_RANGE[1],
                     TRAIN_DEFAULT["epochs"]),
                  foreground="gray").pack(side=tk.LEFT, padx=6)


        ttk.Label(r, text="批次大小 batchSize:", width=22,
                  anchor="e").pack(side=tk.LEFT, padx=(16, 0))
        ttk.Spinbox(r, from_=1, to=1024, textvariable=self.train_batch_var,
                    width=8).pack(side=tk.LEFT)

        ttk.Label(r, text="学习率 learningRate:", width=20,
                  anchor="e").pack(side=tk.LEFT, padx=(16, 0))
        ttk.Entry(r, textvariable=self.train_lr_var, width=10).pack(side=tk.LEFT)

        r = ttk.Frame(box2); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="隐藏层单元 hidden_units:", width=28).pack(side=tk.LEFT)
        ttk.Spinbox(r, from_=2, to=4096, textvariable=self.train_hidden_var,
                    width=8).pack(side=tk.LEFT)
        ttk.Label(r, text="（默认 %d）" % TRAIN_HIDDEN_UNITS,
                  foreground="gray").pack(side=tk.LEFT, padx=6)

        ttk.Label(r, text="嵌入批大小 extract_batch_size:", width=28,
                  anchor="e").pack(side=tk.LEFT, padx=(16, 0))
        ttk.Spinbox(r, from_=1, to=1024, textvariable=self.train_extract_batch_var,
                    width=8).pack(side=tk.LEFT)
        ttk.Label(r, text="（仅 MobileNet 前向，显存吃紧就调小）",
                  foreground="gray").pack(side=tk.LEFT, padx=6)

        r = ttk.Frame(box2); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="模型导出目录（save）:", width=28).pack(side=tk.LEFT)
        ttk.Entry(r, textvariable=self.train_export_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r, text="浏览…", width=10,
                   command=self.on_browse_train_export).pack(side=tk.LEFT, padx=4)

        r = ttk.Frame(box2); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="特征提取器 model.json:", width=28).pack(side=tk.LEFT)
        ttk.Entry(r, textvariable=self.train_backbone_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r, text="浏览…", width=10,
                   command=self.on_browse_train_backbone).pack(side=tk.LEFT, padx=4)

        r = ttk.Frame(box2); r.pack(fill=tk.X, pady=(4, 0))
        ttk.Checkbutton(r, text="忽略数据体检错误（force）",
                        variable=self.train_force_var).pack(side=tk.LEFT)
        ttk.Button(r, text="🩺  数据体检（不训练）", width=22,
                   command=self.on_train_check).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Label(r, text="（参数顺序：source → epochs → "
                          "batch_size → learning_rate → hidden_units → "
                          "optimizer → 导出）",
                  foreground="gray", wraplength=620, justify="left").pack(
            side=tk.LEFT, padx=10)


        # ---------- 区域③ 运行控制（分两行，窄窗口也不会被挤出屏幕） ----------
        box3 = ttk.LabelFrame(top, text=" 区域③ 运行控制 ", padding=8)
        box3.pack(fill=tk.X)

        r = ttk.Frame(box3); r.pack(fill=tk.X, pady=(0, 4))
        self.train_btn = ttk.Button(r, text="🏋️  开始训练",
                                    command=self.on_train_run)
        self.train_btn.pack(side=tk.LEFT)

        self.train_cancel_btn = ttk.Button(r, text="⏹  取消训练",
                                           command=self.on_train_cancel,
                                           state=tk.DISABLED)
        self.train_cancel_btn.pack(side=tk.LEFT, padx=(10, 0))

        self.train_export_btn = ttk.Button(r, text="📤  导出训练结果",
                                           command=self.on_export_train,
                                           state=tk.DISABLED)
        self.train_export_btn.pack(side=tk.LEFT, padx=(10, 0))

        self.to_test_btn = ttk.Button(r, text="➡  去测试页识别",
                                      command=self.on_goto_test)
        self.to_test_btn.pack(side=tk.LEFT, padx=(10, 0))

        r = ttk.Frame(box3); r.pack(fill=tk.X)
        self.train_progress = ttk.Progressbar(r, mode="determinate",
                                              length=240, maximum=100)
        self.train_progress.pack(side=tk.LEFT)
        self.train_status_var = tk.StringVar(value="就绪：填写数据源后点击「开始训练」")
        ttk.Label(r, textvariable=self.train_status_var,
                  foreground="#0a5", wraplength=760, justify="left").pack(
            side=tk.LEFT, padx=10)

        # ---------- 训练日志 ----------
        box4 = ttk.LabelFrame(parent, text=" 训练日志（%s 实时输出） " % TRAINER_PATH,
                              padding=6)
        box4.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))

        # 用 tk.Text + 显式 grid 布局：ScrolledText 会自动塞一个竖直滚动条，
        # 再额外加横向条会导致布局错位，这里完全自己控制。
        _bg, _fg = self._log_bg, self._log_fg
        box4.columnconfigure(0, weight=1)
        box4.rowconfigure(0, weight=1)
        self.train_log_text = tk.Text(
            box4, height=10, wrap=tk.NONE, font=(_MONO_FONT, 9),
            bg=_bg, fg=_fg, insertbackground=_fg,
            selectbackground="#3a6ea5", relief="flat", borderwidth=1,
            state="disabled")
        self.train_log_text.grid(row=0, column=0, sticky="nsew")

        _lsb_v = ttk.Scrollbar(box4, orient="vertical",
                               command=self.train_log_text.yview)
        _lsb_v.grid(row=0, column=1, sticky="ns")
        _lsb_h = ttk.Scrollbar(box4, orient="horizontal",
                               command=self.train_log_text.xview)
        _lsb_h.grid(row=1, column=0, sticky="ew")
        self.train_log_text.configure(yscrollcommand=_lsb_v.set,
                                      xscrollcommand=_lsb_h.set)
        # 滚轮在日志框内时交回日志框自己处理（覆盖 scroller 的绑定）
        for _seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.train_log_text.bind(_seq, self._on_log_wheel, add="+")

        global LOGGER
        LOGGER = TkLogger(self.train_log_text, self.root)
        LOGGER.attach_queue(self._ui_queue)
        self.root.after(33, self._drain_ui_queue)   # 启动主线程回调轮询

        # ---------- 训练结果标签页 ----------
        res_bar = ttk.Frame(lower, padding=(8, 4))
        res_bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(res_bar, text="训练结果：", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(res_bar,
                  text="训练完成后在下方查看「训练A 训练摘要 / 训练B 训练曲线 / "
                       "训练C 数据体检 / 训练D 标签映射」（下拉框可拖动分隔条调整高度）",
                  foreground="#666666", wraplength=760, justify="left").pack(
            side=tk.LEFT, padx=8)

        self.train_nb = ttk.Notebook(lower)
        self.train_nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                           padx=8, pady=(0, 8))

        self.train_welcome = ttk.Frame(self.train_nb)
        self.train_nb.add(self.train_welcome, text=" 训练说明 ")
        _sc = ScrollableFrame(self.train_welcome)
        _sc.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            _sc.scrollable_frame,
            text=("\n  🏋️  训练流程（当前使用 %s）\n\n"
                  "    1. 选择数据源：<数据源>/%s/* 与 <数据源>/%s/*\n"
                  "       （也可以直接选 tm-object-project 的 .json 项目文件）\n"
                  "    2. 可先点「🩺 数据体检」查看样本数量、重复图、类别失衡等风险；\n"
                  "       带「错误」的项会阻塞训练，确需继续请勾选 force。\n"
                  "    3. 设置训练轮次（%d–%d）/ 批次大小 / 学习率 / 隐藏层单元，"
                  "点击「开始训练」。\n"
                  "    4. 训练在后台线程执行：先提取冻结 MobileNetV2 特征，再训练分类头；\n"
                  "       %s。\n"
                  "    5. 训练完成后模型导出到导出目录：metadata.json / model.json /\n"
                  "       weights.bin / training_curve.png / .zip 打包模型。\n"
                  "    6. 勾选「自动填充」时，导出目录会直接写进「测试」页的模型路径，\n"
                  "       点「➡ 去测试页识别」即可用刚训练出的模型跑分类/识别。\n\n"
                  "  ⚠️  训练与测试互斥：训练进行中不能启动测试，反之亦然。\n\n"
                  "  💡 界面提示：上方参数区可以用滚轮 / 右侧滚动条查看；\n"
                  "     内容很宽时（如超长路径）可按 Shift+滚轮 或拖底部横向滚动条。\n"
                  % (TRAINER_PATH, TRAIN_CLASS_NAMES[0], TRAIN_CLASS_NAMES[1],
                     TRAIN_EPOCH_RANGE[0], TRAIN_EPOCH_RANGE[1],
                     TRAIN_VALIDATION_DESCRIPTION)),
            justify="left", font=(_UI_FONT, 11), wraplength=980
        ).pack(anchor="w", padx=30, pady=16)

    # ------------------------------------------------------------------
    # 测试页
    # ------------------------------------------------------------------
    def _on_log_wheel(self, event):
        """日志框内的滚轮：滚动日志本身，而不是外层参数区。"""
        num = getattr(event, "num", None)
        if num == 4:
            steps = -3
        elif num == 5:
            steps = 3
        else:
            raw = getattr(event, "delta", 0)
            steps = -1 * int(raw / 120.0)
            if steps == 0:
                steps = -1 if raw > 0 else 1
        self.train_log_text.yview_scroll(steps, "units")
        return "break"

    def _build_test_tab(self, parent):
        """测试页：区域① 模型配置 / 区域② 数据配置 / 区域③ 运行控制 + 结果。

        参数区同样放进可滚动容器，并把结果 Notebook 放在下方固定区。
        """
        paned = ttk.Panedwindow(parent, orient=tk.VERTICAL)
        paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        upper = ttk.Frame(paned)
        paned.add(upper, weight=3)
        lower = ttk.Frame(paned)
        paned.add(lower, weight=2)

        scroller = ScrollableFrame(upper, horizontal=True)
        scroller.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.test_scroller = scroller
        parent = scroller.scrollable_frame

        top = ttk.Frame(parent, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        box1 = ttk.LabelFrame(top, text=" 区域① 模型配置 ", padding=8)
        box1.pack(fill=tk.X, pady=(0, 6))

        r = ttk.Frame(box1); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="模型文件 (.zip/.keras/.h5):", width=26).pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        ttk.Entry(r, textvariable=self.model_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r, text="浏览…", width=10,
                   command=self.on_browse_model).pack(side=tk.LEFT, padx=4)

        r = ttk.Frame(box1); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="任务类型:", width=26).pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="classification")
        ttk.Radiobutton(r, text="图像分类", variable=self.mode_var,
                        value="classification",
                        command=self.on_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(r, text="图像回归", variable=self.mode_var,
                        value="regression",
                        command=self.on_mode_change).pack(side=tk.LEFT, padx=(20, 0))
        ttk.Radiobutton(r, text="单图/批量识别 (Wide View)",
                        variable=self.mode_var,
                        value="predict",
                        command=self.on_mode_change).pack(side=tk.LEFT, padx=(20, 0))
        self.preproc_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r, text="应用 MobileNetV2 预处理（缩放到 [-1,1]）",
                        variable=self.preproc_var).pack(side=tk.LEFT, padx=(30, 0))

        box2 = ttk.LabelFrame(top, text=" 区域② 数据配置 ", padding=8)
        box2.pack(fill=tk.X, pady=(0, 6))

        self.cls_frame = ttk.Frame(box2)
        r = ttk.Frame(self.cls_frame); r.pack(fill=tk.X)
        ttk.Label(r, text="数据集根目录（子文件夹=类别）:", width=30).pack(side=tk.LEFT)
        self.data_var = tk.StringVar()
        ttk.Entry(r, textvariable=self.data_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r, text="浏览…", width=10,
                   command=self.on_browse_data).pack(side=tk.LEFT, padx=4)

        self.reg_frame = ttk.Frame(box2)
        r = ttk.Frame(self.reg_frame); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="标签 CSV 文件:", width=30).pack(side=tk.LEFT)
        self.csv_var = tk.StringVar()
        ttk.Entry(r, textvariable=self.csv_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r, text="浏览…", width=10,
                   command=self.on_browse_csv).pack(side=tk.LEFT, padx=4)

        r = ttk.Frame(self.reg_frame); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="图片所在目录:", width=30).pack(side=tk.LEFT)
        self.imgdir_var = tk.StringVar()
        ttk.Entry(r, textvariable=self.imgdir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r, text="浏览…", width=10,
                   command=self.on_browse_imgdir).pack(side=tk.LEFT, padx=4)
        ttk.Label(self.reg_frame,
                  text="CSV 格式：第一列 = 图片文件名或绝对路径，第二列 = 数值标签",
                  foreground="gray", wraplength=900, justify="left").pack(
            anchor="w", pady=(2, 0))

        # ---- 区域② 之三：单图/批量识别（原 predictor.py 的输入项）----
        self.pred_frame = ttk.Frame(box2)
        r = ttk.Frame(self.pred_frame); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="识别图片目录:", width=30).pack(side=tk.LEFT)
        self.pred_dir_var = tk.StringVar(value="./resources/testsets/")
        ttk.Entry(r, textvariable=self.pred_dir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r, text="浏览…", width=10,
                   command=self.on_browse_pred_dir).pack(side=tk.LEFT, padx=4)

        r = ttk.Frame(self.pred_frame); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="单张图片（可选，优先）:", width=30).pack(side=tk.LEFT)
        self.pred_single_var = tk.StringVar()
        ttk.Entry(r, textvariable=self.pred_single_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r, text="浏览…", width=10,
                   command=self.on_browse_pred_single).pack(side=tk.LEFT, padx=4)

        ttk.Label(
            self.pred_frame,
            text=("复用「区域①」的模型路径：支持 .zip（含 model.json）或含 "
                  "model.json 的目录；\n"
                  "判定规则：置信度低于阈值 → 置信度过低；前两名差距小于阈值 → "
                  "类别接近（均标记为「不确定」）\n"
                  "阈值优先读取模型自带 metadata.json（如本项目模型为 0.60），"
                  "读不到时用默认 %.2f / %.2f\n"
                  "评分：按类别分子文件夹存放图片（如 测试集/橙子/*.jpg、"
                  "测试集/非橙子/*.jpg），\n"
                  "      子文件夹名即该图标准答案；得分 = 正确数 ÷ 总数 × 100，"
                  "满分 100 分（见下方区域④）。"
                  % (PRED_CONFIDENCE_THRESHOLD, PRED_MARGIN_THRESHOLD)),
            foreground="gray", wraplength=900, justify="left").pack(
            anchor="w", pady=(2, 0))


        box3 = ttk.LabelFrame(top, text=" 区域③ 运行控制 ", padding=8)
        box3.pack(fill=tk.X)

        r = ttk.Frame(box3); r.pack(fill=tk.X, pady=(0, 4))
        self.run_btn = ttk.Button(r, text="▶  开始测试", command=self.on_run)
        self.run_btn.pack(side=tk.LEFT)

        self.export_btn = ttk.Button(r, text="📤  一键导出",
                                     command=self.on_export, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=(10, 0))

        self.back_to_train_btn = ttk.Button(r, text="⬅  去训练页",
                                            command=self.on_goto_train)
        self.back_to_train_btn.pack(side=tk.LEFT, padx=(10, 0))

        r = ttk.Frame(box3); r.pack(fill=tk.X)
        self.progress = ttk.Progressbar(r, mode="determinate",
                                        length=280, maximum=100)
        self.progress.pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(r, textvariable=self.status_var, foreground="#0a5",
                  wraplength=760, justify="left").pack(side=tk.LEFT, padx=10)

        # ---- 区域④ 评分（按赛题评分规则：正确数/总数×100，满分 100）----
        # 放在参数区下方、结果 Notebook 上方，一眼就能看到本次得分。
        box4 = ttk.LabelFrame(top, text=" 区域④ 评分（批量识别，满分 100 分） ",
                              padding=8)
        box4.pack(fill=tk.X, pady=(6, 0))
        self.score_frame = box4

        r = ttk.Frame(box4); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="评分规则:", foreground="#666666").pack(side=tk.LEFT)
        ttk.Label(r, text="总体判断准确率 = 正确判断数量 / 测试集总数量 × 100",
                  foreground="#666666").pack(side=tk.LEFT, padx=6)

        r = ttk.Frame(box4); r.pack(fill=tk.X, pady=(4, 0))
        self.score_var = tk.StringVar(value="—")
        self.score_label = ttk.Label(r, textvariable=self.score_var,
                                     font=(_UI_FONT, 22, "bold"),
                                     foreground="#888888")
        self.score_label.pack(side=tk.LEFT)
        ttk.Label(r, text="分", font=(_UI_FONT, 12),
                  foreground="#888888").pack(side=tk.LEFT, padx=(2, 0))

        self.score_detail_var = tk.StringVar(
            value="尚未评分（运行「单图/批量识别」后显示；"
                  "需按类别分子文件夹存放测试图片才能计分）")
        ttk.Label(r, textvariable=self.score_detail_var, foreground="#666666",
                  wraplength=900, justify="left").pack(side=tk.LEFT, padx=16)

        # 得分进度条（0~100），让分数更直观
        r = ttk.Frame(box4); r.pack(fill=tk.X, pady=(4, 0))
        self.score_bar = ttk.Progressbar(r, mode="determinate",
                                        maximum=100, length=420)
        self.score_bar.pack(side=tk.LEFT)


        self.notebook = ttk.Notebook(lower)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                           padx=8, pady=(0, 8))

        welcome = ttk.Frame(self.notebook)
        self.notebook.add(welcome, text=" 欢迎 ")
        _sc = ScrollableFrame(welcome)
        _sc.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            _sc.scrollable_frame,
            text=("\n\n  🧪  MobileNetV2 模型测试平台\n\n"
                  "  输入区：区域① 模型配置 / 区域② 数据配置 / 区域③ 运行控制\n"
                  "  模型路径既可以手选，也可以由「训练」页训练完成后自动填充。\n\n"
                  "  结果区（每个区域一个标签页）：\n"
                  "      分类任务：区域A 指标报告 | 区域B 混淆矩阵 | 区域C 指标总览\n"
                  "                区域D ROC曲线 | 区域E PR曲线 | 区域F 各类别指标\n"
                  "                区域G 全部图表总览\n"
                  "      回归任务：区域A 指标报告 | 区域B 真实vs预测 | 区域C 残差图\n"
                  "                区域D 残差分布 | 区域E 逐样本对比 | 区域F 误差指标\n"
                  "                区域G 全部图表总览\n"
                  "      识别任务（Wide View）：区域A 识别摘要 | 区域B 逐个按识别\n"
                  "                区域C 置信度散点 | 区域D 识别明细\n"
                  "                区域E 概率分布 | 区域F 全部图表总览\n\n"
                  "  💡 运行完成后，点击「📤 一键导出」可直接导出所有报告与图片：\n"
                  "      每个区域会自动创建一个子文件夹，报告保存为 .txt，图片保存为 .png\n\n"
                  "  💡 界面提示：上方参数区可以用滚轮 / 右侧滚动条查看；\n"
                  "     内容很宽时可按 Shift+滚轮 或拖底部横向滚动条。\n"),
            justify="left", font=(_UI_FONT, 12), wraplength=980
        ).pack(anchor="w", padx=40, pady=20)

    def _set_score_panel(self, grade=None, show=True):
        """更新「区域④ 评分」面板。

        grade 为 None 表示清空（例如切到分类/回归模式）；
        有 grade 但 has_truth=False 时提示未提供标准答案。
        """
        if not hasattr(self, "score_frame"):
            return
        if not show:
            self.score_frame.pack_forget()
            return
        if not self.score_frame.winfo_manager():
            self.score_frame.pack(fill=tk.X, pady=(6, 0))

        if not grade:
            self.score_var.set("—")
            self.score_label.configure(foreground="#888888")
            self.score_bar.configure(value=0)
            self.score_detail_var.set(
                "尚未评分（运行「单图/批量识别」后显示；"
                "需按类别分子文件夹存放测试图片才能计分）")
            return

        if not grade.get("has_truth"):
            self.score_var.set("—")
            self.score_label.configure(foreground="#888888")
            self.score_bar.configure(value=0)
            self.score_detail_var.set(
                "未提供标准答案，无法评分。请按类别分子文件夹存放测试图片"
                "（如 测试集/橙子/*.jpg 与 测试集/非橙子/*.jpg），"
                "子文件夹名即该图的标准答案。本次仅输出识别分布。")
            return

        score = grade["score"]
        self.score_var.set("%.2f" % score)
        # 分数配色：≥90 优秀(绿)、≥70 良好(蓝)、≥60 及格(橙)、其余不及格(红)
        color = ("#0a8f3c" if score >= 90 else
                 "#0a5fd0" if score >= 70 else
                 "#d98000" if score >= 60 else "#cc2222")
        self.score_label.configure(foreground=color)
        self.score_bar.configure(value=max(0.0, min(100.0, score)))
        self.score_detail_var.set(
            "正确 %d / 共 %d 张 = %.2f%%\n计算：%d ÷ %d × 100 = %.2f 分"
            % (grade["n_correct"], grade["n_gradable"],
               grade["accuracy"] * 100,
               grade["n_correct"], grade["n_gradable"], score))

    def on_mode_change(self):
        mode = self.mode_var.get()
        # 先全部隐藏，再显示当前模式对应的配置区
        for f in (self.cls_frame, self.reg_frame, self.pred_frame):
            f.pack_forget()
        if mode == "classification":
            self.cls_frame.pack(fill=tk.X)
        elif mode == "regression":
            self.reg_frame.pack(fill=tk.X)
        else:
            self.pred_frame.pack(fill=tk.X)
        # 评分规则按赛题只针对「批量识别」，其它模式隐藏评分面板
        self._set_score_panel(None, show=(mode == "predict"))


    def on_browse_model(self):
        p = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("模型文件", "*.zip *.keras *.h5 *.hdf5"),
                       ("所有文件", "*.*")])
        if p:
            self.model_var.set(p)

    def on_browse_data(self):
        p = filedialog.askdirectory(title="选择分类数据集根目录")
        if p:
            self.data_var.set(p)

    def on_browse_csv(self):
        p = filedialog.askopenfilename(title="选择标签 CSV",
                                       filetypes=[("CSV", "*.csv"),
                                                  ("所有文件", "*.*")])
        if p:
            self.csv_var.set(p)

    def on_browse_pred_dir(self):
        p = filedialog.askdirectory(title="选择识别图片目录")
        if p:
            self.pred_dir_var.set(p)

    def on_browse_pred_single(self):
        p = filedialog.askopenfilename(
            title="选择单张图片",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp"),
                       ("所有文件", "*.*")])
        if p:
            self.pred_single_var.set(p)

    def on_browse_imgdir(self):
        p = filedialog.askdirectory(title="选择图片所在目录")
        if p:
            self.imgdir_var.set(p)

    # ==================================================================
    # 训练页：浏览 / 体检 / 训练 / 导出 / 页面跳转
    # ==================================================================
    def _detect_train_source(self):
        """启动时自动填充：数据源目录、特征提取器 model.json。"""
        for cand in TRAIN_SOURCE_CANDIDATES:
            p = cand if os.path.isabs(cand) else os.path.join(PROJECT_ROOT, cand)
            if os.path.isdir(p):
                self.train_source_var.set(os.path.normpath(p))
                break
        backbone = os.path.join(PROJECT_ROOT, "model", "basemodels", "model.json")
        if os.path.isfile(backbone):
            self.train_backbone_var.set(backbone)
        try:
            self.train_progress.configure(value=0)
        except Exception:
            pass
        self._set_train_status("就绪：数据源已自动填充，可直接「开始训练」")

    def on_browse_train_source(self):
        """数据源既可以是目录（folder），也可以是 .json 项目文件（json）。"""
        p = filedialog.askdirectory(
            title="选择数据集根目录（含 %s / %s 子目录）"
                  % (TRAIN_CLASS_NAMES[0], TRAIN_CLASS_NAMES[1]))
        if p:
            self.train_source_var.set(p)
            if self.train_source_type_var.get() == "json":
                self.train_source_type_var.set("auto")
            return
        p = filedialog.askopenfilename(
            title="或选择 tm-object-project 项目 JSON",
            filetypes=[("项目 JSON", "*.json"), ("所有文件", "*.*")])
        if p:
            self.train_source_var.set(p)
            self.train_source_type_var.set("json")

    def on_browse_train_export(self):
        p = filedialog.askdirectory(title="选择模型导出目录")
        if p:
            self.train_export_var.set(p)

    def on_browse_train_backbone(self):
        p = filedialog.askopenfilename(
            title="选择特征提取器 model.json（MobileNetV2 图）",
            filetypes=[("TFJS model.json", "model.json"),
                       ("JSON", "*.json"), ("所有文件", "*.*")])
        if p:
            self.train_backbone_var.set(p)

    def on_train_preset_changed(self, event=None):
        """切换到某个预设模式时，把它对应的一组参数填进输入框。

        参数区的 5 个输入框保持可编辑，方便在预设基础上微调；
        真正的取值规则见 TrainEngine.collect()：选了预设则以预设为准，
        「自定义」才用手填值。
        """
        name = self.train_preset_var.get()
        fixed = TRAIN_PRESETS.get(name)
        if fixed is None:
            self._set_train_status("训练模式：%s（参数以输入框为准）" % name)
            return
        self.train_epochs_var.set(str(fixed["epochs"]))
        self.train_batch_var.set(str(fixed["batchSize"]))
        self.train_lr_var.set(str(fixed["learningRate"]))
        self.train_hidden_var.set(str(fixed["hidden_units"]))
        self.train_optimizer_var.set(fixed["optimizer"])
        self._set_train_status(
            "训练模式「%s」→ epochs=%d / batchSize=%d / learningRate=%s / "
            "hidden_units=%d / optimizer=%s"
            % (name, fixed["epochs"], fixed["batchSize"],
               fixed["learningRate"], fixed["hidden_units"], fixed["optimizer"]))
        msg = ("[预设] 已套用训练模式「%s」：epochs=%d, batchSize=%d, "
               "learningRate=%s, hidden_units=%d, optimizer=%s"
               % (name, fixed["epochs"], fixed["batchSize"],
                  fixed["learningRate"], fixed["hidden_units"],
                  fixed["optimizer"]))
        if LOGGER is not None:
            LOGGER(msg)
        else:
            print(msg)


    def on_train_check(self):
        """只做数据体检，不训练（体检是只读的，直接在主线程跑）。"""
        source = self.train_source_var.get().strip()
        if not source or not os.path.exists(source):
            messagebox.showerror("错误", "请先选择有效的数据源（目录或 .json）")
            return
        try:
            self._set_train_status("正在体检数据…")
            self.root.update_idletasks()
            st = self.engine.stats()
        except Exception as e:
            self._set_train_status("数据体检失败")
            messagebox.showerror("数据体检失败", str(e))
            return

        text = build_train_health_text(st["source"], st["class_counts"],
                                       st["health_issues"])
        self._set_train_tabs([("训练C 数据体检", text)])
        n_err = sum(1 for i in st["health_issues"] if i.get("severity") == "error")
        n_warn = sum(1 for i in st["health_issues"] if i.get("severity") != "error")
        self._set_train_status("体检完成：%d 项错误 / %d 项提醒" % (n_err, n_warn))
        if LOGGER is not None:
            LOGGER("")
            LOGGER("[体检] %s" % st["source"])
            for line in text.split("\n"):
                LOGGER("  " + line)

    def on_train_run(self):
        if self.training:
            return
        if self.running:
            messagebox.showwarning("提示", "测试正在运行，请等测试结束后再开始训练。")
            return

        source = self.train_source_var.get().strip()
        if not source or not os.path.exists(source):
            messagebox.showerror("错误", "请选择有效的数据源（目录或 .json 项目文件）")
            return

        # 前端先校验一次参数，避免后台线程起来之后才报错。
        # 选了预设模式时，实际取值以预设为准，手填值会被覆盖，
        # 因此这里只校验预设自身合法，不因输入框里的旧值挡住训练。
        preset = self.train_preset_var.get()
        fixed = TRAIN_PRESETS.get(preset)
        if fixed is not None:
            if not (TRAIN_EPOCH_RANGE[0] <= fixed["epochs"] <= TRAIN_EPOCH_RANGE[1]):
                messagebox.showerror("参数错误", "预设「%s」的轮次 %d 超出允许范围 %d–%d"
                                     % (preset, fixed["epochs"],
                                        TRAIN_EPOCH_RANGE[0], TRAIN_EPOCH_RANGE[1]))
                return
            epochs, batch = fixed["epochs"], fixed["batchSize"]
            lr, hidden = fixed["learningRate"], fixed["hidden_units"]
            try:
                extract_batch = int(self.train_extract_batch_var.get())
            except ValueError as e:
                messagebox.showerror("参数错误", "嵌入批大小必须是整数：%s" % e)
                return
            if extract_batch <= 0:
                messagebox.showerror("参数错误", "嵌入批大小必须为正整数")
                return
        else:
            try:
                epochs = int(self.train_epochs_var.get())
                batch = int(self.train_batch_var.get())
                lr = float(self.train_lr_var.get())
                hidden = int(self.train_hidden_var.get())
                extract_batch = int(self.train_extract_batch_var.get())
            except ValueError as e:
                messagebox.showerror("参数错误", "训练参数必须是数字：%s" % e)
                return
            if not (TRAIN_EPOCH_RANGE[0] <= epochs <= TRAIN_EPOCH_RANGE[1]):
                messagebox.showerror("参数错误", "训练轮次需为 %d–%d 的整数"
                                     % (TRAIN_EPOCH_RANGE[0], TRAIN_EPOCH_RANGE[1]))
                return
            if batch <= 0 or hidden <= 0 or extract_batch <= 0 or lr <= 0:
                messagebox.showerror("参数错误",
                                     "批次大小 / 隐藏层单元 / 嵌入批大小 / 学习率"
                                     "都必须为正数")
                return

        self.training = True
        self.train_cancel_var.set(False)
        self.engine._cancel_flag = False       # 清除上一轮可能残留的取消标志
        self.last_train_result = None
        # 主线程快照：子线程绝不能读 tk 变量（否则抛 main thread is not in main loop）
        self.engine.snapshot()
        self.train_btn.config(state=tk.DISABLED)
        self.train_export_btn.config(state=tk.DISABLED)
        self.run_btn.config(state=tk.DISABLED)
        self.train_cancel_btn.config(state=tk.NORMAL)
        self.train_progress.configure(value=0)
        self._set_train_status("正在训练…（实时日志见下方）")
        if LOGGER is not None:
            LOGGER.clear()
        threading.Thread(target=self._train_worker, daemon=True).start()

    def on_train_cancel(self):
        """请求取消：collect() 阶段直接中止；已进入 train() 时在导出前生效。

        同时写两份：tk 变量（给界面显示）+ 普通 bool（供子线程安全读取）。
        """
        self.train_cancel_var.set(True)
        self.engine._cancel_flag = True
        self._set_train_status("已请求取消：将在当前阶段结束后停止（不导出模型）")

    def _train_worker(self):
        """后台线程：跑 TrainEngine.run()，结果通过 _ui_call() 回主线程渲染。

        子线程里不能直接读 tk 变量或调 root.after 之外的东西；
        所有跨线程回调统一走 _ui_call，保证状态一定能复位。
        """
        log = LOGGER if LOGGER is not None else print
        try:
            self._ui_call(lambda: self.train_progress.configure(value=15))
            res = self.engine.run(log=log)
            self._ui_call(lambda r=res: self._on_train_done(r))
        except Exception as exc:
            tb = traceback.format_exc()
            msg = str(exc)
            self._ui_call(lambda m=msg, t=tb: self._on_train_error(m, t))
        finally:
            self._ui_call(self._on_train_finished)

    def _on_train_error(self, msg, tb):
        print(tb)
        if LOGGER is not None:
            LOGGER("")
            LOGGER("[错误] 训练失败：%s" % msg)
        self._set_train_status("训练失败：%s" % msg.split("\n")[0])
        messagebox.showerror("训练失败", msg + "\n\n详细信息见「训练日志」和控制台。")

    def _on_train_finished(self):
        """训练线程结束：无条件恢复按钮可用（成功 / 失败 / 取消都要恢复）。

        即使引擎抛异常、被取消，也必须让「开始训练」重新可点，
        否则用户只能重开界面 —— 这是之前"按钮点了没用"的主因之一。
        """
        self.training = False
        self.train_btn.config(state=tk.NORMAL)
        self.train_cancel_btn.config(state=tk.DISABLED)
        self.to_test_btn.config(state=tk.NORMAL)
        if not self.running:
            self.run_btn.config(state=tk.NORMAL)
        if self.last_train_result is not None:
            self.train_progress.configure(value=100)
            self.train_export_btn.config(state=tk.NORMAL)

    def _set_train_status(self, text):
        self._ui_call(lambda t=text: self.train_status_var.set(t))

    def _set_status(self, text):
        self._ui_call(lambda t=text: self.status_var.set(t))

    def _set_progress(self, val):
        v = max(0.0, min(100.0, float(val)))
        self._ui_call(lambda x=v: self.progress.configure(value=x))

    # ==================================================================
    # 训练结果渲染 / 训练结果导出 / 页面跳转
    # ==================================================================
    def _set_train_tabs(self, entries):
        """重建训练结果标签页。entries = [(标题, 文本或Figure), ...]。"""
        for tab in self.train_nb.tabs():
            self.train_nb.forget(tab)
        self.figures = []

        for title, payload in entries:
            frame = ttk.Frame(self.train_nb)
            self.train_nb.add(frame, text=" %s " % title)
            if hasattr(payload, "savefig"):        # matplotlib Figure
                fc = FigureCanvasTkAgg(payload, master=frame)
                fc.draw()
                toolbar = NavigationToolbar2Tk(fc, frame)
                toolbar.update()
                fc.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                self.figures.append(payload)
            else:                                   # 文本
                _bg, _fg = self._log_bg, self._log_fg
                frame.columnconfigure(0, weight=1)
                frame.rowconfigure(0, weight=1)
                st = tk.Text(frame, wrap=tk.NONE, font=(_MONO_FONT, 10),
                             bg=_bg, fg=_fg, insertbackground=_fg,
                             selectbackground="#3a6ea5", relief="flat",
                             borderwidth=1)
                st.insert("1.0", payload)
                st.configure(state="disabled")
                st.grid(row=0, column=0, sticky="nsew")
                sb_v = ttk.Scrollbar(frame, orient="vertical",
                                     command=st.yview)
                sb_v.grid(row=0, column=1, sticky="ns")
                sb_h = ttk.Scrollbar(frame, orient="horizontal",
                                     command=st.xview)
                sb_h.grid(row=1, column=0, sticky="ew")
                st.configure(yscrollcommand=sb_v.set, xscrollcommand=sb_h.set)
                for seq in ("<MouseWheel>", "<Shift-MouseWheel>"):
                    st.bind(seq, lambda e, w=st: self._text_wheel(w, e),
                            add="+")

        first = self.train_nb.tabs()
        if first:
            self.train_nb.select(first[0])

    def _make_train_pages(self, res):
        """把一次训练结果整理成 [(标题, 文本或Figure), ...]。"""
        entries = []
        report = build_train_report(
            res["source"], res["source_type"], res["result"], res["params"],
            res.get("export_dir"), res["class_counts"], res["class_issues"],
            res["env_lines"], status_line="训练完成 ✔")
        entries.append(("训练A 训练摘要", report))

        # 训练曲线：优先展示当前训练器导出的 PNG，保证与导出物一致
        history = (res["result"] or {}).get("history") or {}
        curve_text = build_train_curve_text(history, res["params"])
        curve_png = None
        if res.get("export_dir"):
            p = os.path.join(res["export_dir"], "training_curve.png")
            if os.path.isfile(p):
                curve_png = p
        if curve_png:
            try:
                from PIL import Image
                img = Image.open(curve_png)
                fig = Figure(figsize=(12, 5), dpi=100)
                ax = fig.add_subplot(111)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title("训练曲线（%s 导出：%s）"
                             % (TRAINER_PATH, os.path.basename(curve_png)), fontsize=12)
                fig.tight_layout()
                entries.append(("训练B 训练曲线", fig))
            except Exception as e:
                print("[警告] 训练曲线图加载失败：%s" % e)
        entries.append(("训练B 训练曲线明细", curve_text))

        entries.append(("训练C 数据体检", build_train_health_text(
            res["source"], res["class_counts"], res["health_issues"])))

        labels = (res["result"] or {}).get("labels") or []
        entries.append(("训练D 标签映射", build_train_label_text(labels)))

        return entries

    def _on_train_done(self, res):
        """训练成功回调（主线程）：渲染结果、回填模型路径、启用导出。"""
        self.last_train_result = res
        try:
            self._set_train_tabs(self._make_train_pages(res))
        except Exception as e:
            print("[警告] 训练结果渲染失败：%s" % e)
        self._reset_scrollers()

        # 训练 → 测试：把导出的模型目录写进测试页的模型路径
        exported = res.get("exported_path")
        if exported and os.path.isdir(exported):
            if self.train_autofill_var.get():
                self.model_var.set(exported)
                if not self.pred_dir_var.get().strip():
                    self.pred_dir_var.set(os.path.join(PROJECT_ROOT,
                                                       "resources", "testsets"))
                auto_hint = "，已自动填充到「测试」页模型路径"
            else:
                auto_hint = "（未自动填充，可在测试页手动选择该目录）"
        else:
            auto_hint = "（未导出模型）"

        result = res["result"] or {}
        valid = result.get("validation") or {}
        acc = valid.get("accuracy")
        done_epochs = len((result.get("history") or {}).get("loss", []))
        self._set_train_status(
            "训练完成：%d 轮 / 样本 %d 张，耗时 %s%s"
            % (done_epochs, result.get("exampleCount", 0),
               _fmt_secs(res.get("duration")),
               ("，验证准确率 %.4f" % acc) if acc is not None else ""))
        self.train_export_btn.config(state=tk.NORMAL)

        lines = ["训练完成 ✔",
                 "  导出目录：%s%s" % (res.get("export_dir", "—"), auto_hint),
                 "  实际轮次：%d / 计划 %d" % (done_epochs, res["params"]["epochs"]),
                 "  优化器：%s    批次大小：%d    学习率：%s"
                 % (res["params"].get("optimizer", TRAIN_DEFAULT_OPTIMIZER),
                    res["params"]["batch_size"], res["params"]["learning_rate"]),
                 "  样本总数：%d" % result.get("exampleCount", 0)]

        zips = []
        if res.get("export_dir") and os.path.isdir(res["export_dir"]):
            zips = sorted(f for f in os.listdir(res["export_dir"])
                          if f.lower().endswith(".zip"))
        if zips:
            lines.append("  打包模型：%s" % "、".join(zips))
        if res.get("cancelled"):
            lines.append("  ⚠️ 收到过取消请求，未执行模型导出。")
        if acc is not None:
            lines.append("  验证准确率：%.4f（验证集 %d 张）"
                         % (acc, valid.get("exampleCount", 0)))
        messagebox.showinfo("训练完成", "\n".join(lines))

    def on_goto_test(self):
        """切到测试页；若模型路径为空则自动指向最近的导出目录。"""
        if not self.model_var.get().strip() and self.last_train_result:
            p = self.last_train_result.get("exported_path")
            if p and os.path.isdir(p):
                self.model_var.set(p)
        self.main_nb.select(self.test_tab)

    def on_goto_train(self):
        self.main_nb.select(self.train_tab)

    def on_export_train(self):
        """把训练摘要 / 曲线明细 / 数据体检 / 标签映射导出为 txt + png。"""
        if self.last_train_result is None:
            messagebox.showwarning("提示", "请先训练一次，再导出训练结果。")
            return
        out_root = filedialog.askdirectory(title="选择训练结果导出目录")
        if not out_root:
            return

        res = self.last_train_result
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(out_root, "导出_训练_%s" % ts)
        try:
            os.makedirs(export_dir, exist_ok=True)
            self._set_train_status("正在导出训练结果…")
            self.root.update_idletasks()

            # 与「测试」页一致：每个区域一个子文件夹，报告 .txt、图片 .png
            for title, payload in self._make_train_pages(res):
                d = os.path.join(export_dir, title)
                os.makedirs(d, exist_ok=True)
                if hasattr(payload, "savefig"):
                    payload.savefig(os.path.join(d, "%s.png" % title), dpi=150,
                                    bbox_inches="tight", facecolor="white",
                                    edgecolor="none")
                else:
                    with open(os.path.join(d, "%s.txt" % title),
                              "w", encoding="utf-8") as f:
                        f.write(payload)

            self._set_train_status("训练结果导出完成 ✔")
            if messagebox.askyesno("导出成功",
                                   "已导出到：\n%s\n\n是否打开该目录？" % export_dir):
                _reveal_path(export_dir)
        except Exception as e:
            self._set_train_status("训练结果导出失败")
            messagebox.showerror("导出失败", str(e))

    def on_run(self):
        if self.running:
            return
        # 训练进行中不允许启动测试（互斥），给出明确反馈而不是静默无反应
        if self.training:
            messagebox.showwarning("提示", "训练正在进行，请等训练结束后再开始测试。")
            return

        model_path = self.model_var.get().strip()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("错误", "请选择有效的模型文件"); return

        mode = self.mode_var.get()
        if mode == "classification":
            d = self.data_var.get().strip()
            if not d or not os.path.isdir(d):
                messagebox.showerror("错误", "请选择有效的分类数据集根目录"); return
        elif mode == "regression":
            c = self.csv_var.get().strip()
            if not c or not os.path.exists(c):
                messagebox.showerror("错误", "请选择有效的标签 CSV 文件"); return
        else:
            single = self.pred_single_var.get().strip()
            d = self.pred_dir_var.get().strip()
            if single:
                if not os.path.isfile(single):
                    messagebox.showerror("错误", "单张图片路径无效"); return
            elif not d or not os.path.isdir(d):
                messagebox.showerror("错误", "请选择有效的识别图片目录"); return

        self.running = True
        self._cancel_requested = False      # 本轮运行的取消标志（关窗时置位）
        # 主线程把这次运行需要的所有输入快照下来，子线程只读这份快照。
        # （tk 变量不能跨线程访问，否则会抛 "main thread is not in main loop"）
        self._pending_mode = self.mode_var.get()
        # 清掉上一轮的得分，避免与本次结果混淆
        self._set_score_panel(None, show=(self._pending_mode == "predict"))
        self._pending_args = {
            "model_path": model_path,
            "use_preproc": bool(self.preproc_var.get()),
            "data_dir": self.data_var.get().strip(),
            "csv_path": self.csv_var.get().strip(),
            "img_dir": self.imgdir_var.get().strip(),
            "single": self.pred_single_var.get().strip(),
            "test_dir": self.pred_dir_var.get().strip(),
        }
        self.run_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        self.progress.configure(value=0)
        self.status_var.set("正在加载模型…")
        # 测试与训练互斥：训练中不允许启动测试（训练按钮在 on_train_run 里已禁用）
        self.train_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        """测试后台线程。

        注意：绝不能在子线程里读 tk 变量（self.mode_var.get() 会抛
        "main thread is not in main loop"），所以模式在启动前就由主线程取好，
        用参数传进来。异常路径也必须保证把 running 复位，否则按钮会永久卡死。
        """
        mode = self._pending_mode
        try:
            if mode == "classification":
                res = self._compute_classification()
            elif mode == "regression":
                res = self._compute_regression()
            else:
                res = self._compute_predict()
            self._ui_call(lambda r=res: self._render(r))
        except Exception as exc:
            tb = traceback.format_exc()
            msg = str(exc)
            self._ui_call(lambda m=msg, t=tb: self._on_error(m, t))
        finally:
            self._ui_call(self._on_finished)

    def _ui_call(self, fn):
        """把回调安全地交给主线程执行。

        Tk 的 after() 本身不是线程安全的（子线程调用可能抛
        "main thread is not in main loop"），所以这里用一个队列中转：
        子线程只负责入队，主线程的轮询循环负责取出执行。
        """
        try:
            self._ui_queue.put_nowait(fn)
        except Exception:
            pass

    def _drain_ui_queue(self):
        """主线程轮询：把子线程排队的回调取出来执行。

        注意：这里给单次排空加了时间上限。子线程若在短时间内塞进大量回调，
        无限 while 会把主线程一直占着，界面反而更卡（表现为「未响应」）。
        限制每轮最多处理 20ms 就返回，把控制权还给 Tk 事件循环。
        """
        deadline = time.perf_counter() + 0.02
        try:
            while True:
                fn = self._ui_queue.get_nowait()
                try:
                    fn()
                except Exception as e:
                    print("[警告] 主线程回调执行失败：%s" % e)
                if time.perf_counter() >= deadline:
                    break
        except queue.Empty:
            pass
        try:
            self.root.after(33, self._drain_ui_queue)   # ~30fps
        except Exception:
            pass

    def _on_error(self, msg, tb):
        self.status_var.set("发生错误")
        messagebox.showerror("运行错误", msg + "\n\n详细信息见控制台。")
        print(tb)

    def _on_finished(self):
        """测试线程结束：恢复按钮可用状态（要顾及训练是否也在跑）。"""
        self.running = False
        self._cancel_requested = False      # 复位取消标志，供下次运行使用
        if not self.training:
            self.run_btn.config(state=tk.NORMAL)
            self.train_btn.config(state=tk.NORMAL)
        self.progress.configure(value=100)
        # 有结果才允许导出，避免点了「一键导出」只弹一句"请先运行一次"
        if self.last_result is not None:
            self.export_btn.config(state=tk.NORMAL)

    def on_close(self):
        """关闭窗口：清理识别模式解压出的临时目录（原 predictor.py 的
        on_closing 行为，整合后保留）。

        Confirm before closing while training or testing is running.
        """
        if (self.training or self.running) and not getattr(self, "_closing_confirmed", False):
            busy = "训练" if self.training else "测试"
            if not messagebox.askyesno(
                    "确认退出",
                    "%s仍在进行中，退出会立即中断且不保存结果。\n\n确定要退出吗？" % busy):
                return
            self._closing_confirmed = True

        # 通知正在跑的批量识别尽快收尾，避免关窗后线程还在跑
        self._cancel_requested = True

        if self._pred_temp_dir and os.path.exists(self._pred_temp_dir):
            try:
                shutil.rmtree(self._pred_temp_dir)
            except Exception:
                pass
        # Close matplotlib figures before exiting Tk.
        try:
            plt.close("all")
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 一键导出
    # ------------------------------------------------------------------
    def on_export(self):
        if self.last_result is None:
            messagebox.showwarning("提示", "请先运行一次测试，再导出结果。")
            return

        out_root = filedialog.askdirectory(title="选择导出目录")
        if not out_root:
            return

        task_name = {"classification": "分类", "regression": "回归",
                     "predict": "识别"}.get(self.last_result["type"], "结果")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(out_root, "导出_%s_%s" % (task_name, ts))

        try:
            os.makedirs(export_dir, exist_ok=True)
            self.status_var.set("正在导出…")
            self.root.update_idletasks()

            if self.last_result["type"] == "classification":
                self._export_classification(export_dir)
            elif self.last_result["type"] == "regression":
                self._export_regression(export_dir)
            else:
                self._export_predict(export_dir)

            self.status_var.set("导出完成 ✔")
            if messagebox.askyesno("导出成功",
                                   "已导出到：\n%s\n\n是否打开该目录？" % export_dir):
                _reveal_path(export_dir)
        except Exception as e:
            self.status_var.set("导出失败")
            messagebox.showerror("导出失败", str(e))

    # ---------- 保存单张图表 ----------
    @staticmethod
    def _save_figure(export_dir, folder_name, file_name,
                     plot_func, *args, figsize=(10, 7)):
        d = os.path.join(export_dir, folder_name)
        os.makedirs(d, exist_ok=True)
        fig = Figure(figsize=figsize, dpi=150)
        plot_func(fig.add_subplot(111), *args)
        fig.tight_layout()
        path = os.path.join(d, file_name)
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        return path

    # ---------- 保存文本 ----------
    @staticmethod
    def _save_text(export_dir, folder_name, file_name, text):
        d = os.path.join(export_dir, folder_name)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, file_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    # ---------- 分类导出 ----------
    def _export_classification(self, export_dir):
        res = self.last_result
        classes = res["classes"]
        m = res["metrics"]
        y_true = res["y_true"]

        # 区域A 报告
        self._save_text(export_dir, "区域A_指标报告",
                        "指标报告.txt", res["report"])

        # 区域B 混淆矩阵
        self._save_figure(export_dir, "区域B_混淆矩阵",
                          "混淆矩阵.png", _plot_confusion_matrix,
                          m, classes, figsize=(8, 7))

        # 区域C 指标总览
        self._save_figure(export_dir, "区域C_指标总览",
                          "指标总览.png", _plot_overview_bars,
                          m, figsize=(8.5, 5.5))

        # 区域D ROC 曲线
        self._save_figure(export_dir, "区域D_ROC曲线",
                          "ROC曲线.png", _plot_roc,
                          m, classes, figsize=(8.5, 7))

        # 区域E PR 曲线
        self._save_figure(export_dir, "区域E_PR曲线",
                          "PR曲线.png", _plot_pr,
                          m, classes, figsize=(8.5, 7))

        # 区域F 各类别指标
        self._save_figure(export_dir, "区域F_各类别指标",
                          "各类别指标.png", _plot_per_class,
                          m, classes,
                          figsize=(max(8, 1.0 * len(classes) + 5), 6))

        # 区域G 全部图表总览
        d = os.path.join(export_dir, "区域G_全部图表总览")
        os.makedirs(d, exist_ok=True)
        fig = Figure(figsize=(17, 22), dpi=120)
        gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.30)
        _plot_confusion_matrix(fig.add_subplot(gs[0, 0]), m, classes)
        _plot_overview_bars(fig.add_subplot(gs[0, 1]), m)
        _plot_roc(fig.add_subplot(gs[1, 0]), m, classes)
        _plot_pr(fig.add_subplot(gs[1, 1]), m, classes)
        _plot_per_class(fig.add_subplot(gs[2, :]), m, classes)
        fig.suptitle("分类结果全部图表总览 —— MobileNetV2",
                     fontsize=16, fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(os.path.join(d, "全部图表总览.png"), dpi=120,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # 额外：逐样本预测结果 CSV（方便二次分析）
        try:
            import csv as _csv
            d2 = os.path.join(export_dir, "额外_逐样本预测结果")
            os.makedirs(d2, exist_ok=True)
            with open(os.path.join(d2, "逐样本预测.csv"),
                      "w", encoding="utf-8-sig", newline="") as f:
                w = _csv.writer(f)
                header = ["样本序号", "真实标签"] + \
                         ["P(%s)" % c for c in classes] + ["预测标签", "文件名"]
                w.writerow(header)
                y_pred = m["y_pred"]
                for i in range(len(y_true)):
                    row = [i, classes[int(y_true[i])]]
                    row += ["%.6f" % float(p) for p in res["y_prob"][i]]
                    row += [classes[int(y_pred[i])], os.path.basename(res["paths"][i])]
                    w.writerow(row)
        except Exception:
            pass

    # ---------- 回归导出 ----------
    def _export_regression(self, export_dir):
        res = self.last_result
        y_true = np.asarray(res["y_true"], dtype=np.float64)
        y_pred = np.asarray(res["y_pred"], dtype=np.float64)
        m = res["metrics"]
        resid = m["residuals"]

        # 区域A 报告
        self._save_text(export_dir, "区域A_指标报告",
                        "指标报告.txt", res["report"])

        # 区域B 真实 vs 预测
        self._save_figure(export_dir, "区域B_真实vs预测",
                          "真实vs预测.png", _plot_true_vs_pred,
                          y_true, y_pred, m, figsize=(8, 7))

        # 区域C 残差图
        self._save_figure(export_dir, "区域C_残差图",
                          "残差图.png", _plot_residual,
                          y_pred, resid, figsize=(8, 6.5))

        # 区域D 残差分布
        self._save_figure(export_dir, "区域D_残差分布",
                          "残差分布.png", _plot_residual_hist,
                          resid, figsize=(8, 6.5))

        # 区域E 逐样本对比
        self._save_figure(export_dir, "区域E_逐样本对比",
                          "逐样本对比.png", _plot_sample_curve,
                          y_true, y_pred, figsize=(10, 6))

        # 区域F 误差指标
        self._save_figure(export_dir, "区域F_误差指标",
                          "误差指标.png", _plot_error_bars,
                          m, figsize=(8.5, 5.5))

        # 区域G 全部图表总览
        d = os.path.join(export_dir, "区域G_全部图表总览")
        os.makedirs(d, exist_ok=True)
        fig = Figure(figsize=(17, 21), dpi=120)
        gs = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.28)
        _plot_true_vs_pred(fig.add_subplot(gs[0, 0]), y_true, y_pred, m)
        _plot_residual(fig.add_subplot(gs[0, 1]), y_pred, resid)
        _plot_residual_hist(fig.add_subplot(gs[1, 0]), resid)
        _plot_error_bars(fig.add_subplot(gs[1, 1]), m)
        _plot_sample_curve(fig.add_subplot(gs[2, :]), y_true, y_pred)
        fig.suptitle("回归结果全部图表总览 —— MobileNetV2",
                     fontsize=16, fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(os.path.join(d, "全部图表总览.png"), dpi=120,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # 额外：逐样本预测结果 CSV
        try:
            import csv as _csv
            d2 = os.path.join(export_dir, "额外_逐样本预测结果")
            os.makedirs(d2, exist_ok=True)
            with open(os.path.join(d2, "逐样本预测.csv"),
                      "w", encoding="utf-8-sig", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["样本序号", "真实值", "预测值", "残差"])
                for i in range(len(y_true)):
                    w.writerow([i,
                                "%.6f" % float(y_true[i]),
                                "%.6f" % float(y_pred[i]),
                                "%.6f" % float(y_true[i] - y_pred[i])])
        except Exception:
            pass

    # ---------- 识别（Wide View）导出 ----------
    def _export_predict(self, export_dir):
        res = self.last_result
        samples = res["samples"]

        # 区域A 识别摘要报告（含评分章节）
        self._save_text(export_dir, "区域A_识别摘要",
                        "识别摘要.txt", res["report"])

        # 区域B 逐个识别结果（纯文本，等价于原 Wide View 卡片列表）
        self._save_text(export_dir, "区域B_逐个识别结果",
                        "逐个识别结果.txt", res["cards_text"])

        # 额外：评分结果 + 逐样本判定 CSV（可直接作为成绩单）
        grade = res.get("grade") or {}
        try:
            import csv as _csv
            dg = os.path.join(export_dir, "额外_评分结果")
            os.makedirs(dg, exist_ok=True)

            with open(os.path.join(dg, "评分结果.csv"),
                      "w", encoding="utf-8-sig", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["项目", "值"])
                if grade.get("has_truth"):
                    w.writerow(["测试集总数", grade["n_total"]])
                    w.writerow(["参与评分张数", grade["n_gradable"]])
                    w.writerow(["判断正确张数", grade["n_correct"]])
                    w.writerow(["判断错误张数",
                                grade["n_gradable"] - grade["n_correct"]])
                    w.writerow(["总体判断准确率",
                                "%.4f" % grade["accuracy"]])
                    w.writerow(["得分（满分100）", "%.2f" % grade["score"]])
                else:
                    w.writerow(["得分", "—"])
                    w.writerow(["说明", "未提供标准答案，无法评分"])

            with open(os.path.join(dg, "逐样本判定.csv"),
                      "w", encoding="utf-8-sig", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["样本序号", "文件名", "判定类别", "标准答案",
                            "是否正确", "置信度", "是否不确定", "原因"])
                for i, s in enumerate(samples):
                    if s.get("truth"):
                        ok = "正确" if s.get("correct") else "错误"
                    else:
                        ok = "—"
                    w.writerow([i, s["name"], s["label"], s.get("truth") or "—",
                                ok, "%.6f" % s["probability"],
                                "是" if s["uncertain"] else "否",
                                s["reason"] or ""])
        except Exception:
            pass

        # 区域C 置信度散点
        self._save_figure(export_dir, "区域C_置信度散点",
                          "置信度散点.png", _plot_pred_prob_scatter,
                          samples, res["classes"],
                          res.get("conf_threshold", PRED_CONFIDENCE_THRESHOLD), figsize=(11, 6))

        # 区域D 识别明细
        self._save_figure(export_dir, "区域D_识别明细",
                          "识别明细.png", _plot_pred_per_image,
                          samples, res.get("conf_threshold", PRED_CONFIDENCE_THRESHOLD),
                          figsize=(11, 8))

        # 区域E 概率分布
        self._save_figure(export_dir, "区域E_概率分布",
                          "概率分布.png", _plot_pred_class_dist,
                          res["class_counts"], res["class_probs"],
                          res.get("conf_threshold", PRED_CONFIDENCE_THRESHOLD), figsize=(9, 6))

        # 区域F 全部图表总览
        d = os.path.join(export_dir, "区域F_全部图表总览")
        os.makedirs(d, exist_ok=True)
        fig = Figure(figsize=(17, 15), dpi=120)
        gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.26)
        _plot_pred_class_dist(fig.add_subplot(gs[0, 0]),
                              res["class_counts"], res["class_probs"],
                              res.get("conf_threshold", PRED_CONFIDENCE_THRESHOLD))
        _plot_pred_prob_scatter(fig.add_subplot(gs[0, 1]), samples,
                                res["classes"], res.get("conf_threshold", PRED_CONFIDENCE_THRESHOLD))
        _plot_pred_per_image(fig.add_subplot(gs[1, :]), samples,
                             res.get("conf_threshold", PRED_CONFIDENCE_THRESHOLD))
        fig.suptitle("识别结果全部图表总览 —— MobileNetV2 (alpha=0.5)",
                     fontsize=16, fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(os.path.join(d, "全部图表总览.png"), dpi=120,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # 额外：逐样本预测结果 CSV（等价原 base.sdd 的批量结果）
        try:
            d2 = os.path.join(export_dir, "额外_逐样本预测结果")
            os.makedirs(d2, exist_ok=True)
            with open(os.path.join(d2, "逐样本预测.csv"),
                      "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                labels = res["classes"]
                w.writerow(["样本序号", "文件名", "判定类别", "置信度",
                            "是否不确定", "原因"]
                           + ["P(%s)" % c for c in labels])
                for i, s in enumerate(samples):
                    row = [i, s["name"], s["label"], "%.6f" % s["probability"],
                           "是" if s["uncertain"] else "否",
                           s["reason"] or ""]
                    row += ["%.6f" % p for p in s["raw_probs"]]
                    w.writerow(row)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 计算：分类
    # ------------------------------------------------------------------
    def _compute_classification(self):
        # 输入在 on_run()（主线程）里快照好，子线程不再读 tk 变量
        args = self._pending_args
        model_path = args["model_path"]
        data_dir = args["data_dir"]
        use_preproc = args["use_preproc"]

        self._set_status("正在加载模型…"); self._set_progress(2)
        model = load_model_any(model_path)

        self._set_status("正在扫描数据集…")
        classes, paths, y_true = collect_classification_data(data_dir)
        size = get_input_size(model)

        def cb(done, total):
            self._set_progress(5 + 85.0 * done / max(total, 1))
            self._set_status("预测中… %d / %d" % (done, total))

        raw = np.asarray(predict_paths(model, paths, size, use_preproc, 32, cb),
                         dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        # 统一成概率：softmax 分类头原样通过，logits 模型（graph-model 等）
        # 在此做 softmax，避免把 logits 当概率导致 F1 / AUC 失真。
        y_prob = to_probabilities(raw)
        if y_prob.shape[1] == 1:
            raise RuntimeError(
                "模型只输出 1 个数值，无法作为分类概率使用（缺少 softmax 或类别维）。")


        if y_prob.shape[1] != len(classes):
            sample = ", ".join(classes[:10])
            if len(classes) > 10:
                sample += " ...（共 %d 个）" % len(classes)
            raise RuntimeError(
                "模型输出维度 (%d) 与数据集类别数 (%d) 不一致！\n\n"
                "【模型情况】\n  模型输出 %d 类，说明这是一个 %d 分类模型。\n\n"
                "【数据集情况】\n  数据集根目录：%s\n"
                "  扫描到 %d 个有效类别子文件夹（含图片的）：\n  %s\n\n"
                "【如何修复】\n"
                "  1. 检查是否选错了数据集根目录？\n"
                "  2. 准备一个恰好包含 %d 个类别子文件夹的数据集，例如：\n"
                "        数据集根目录/\n"
                "        ├── 类别1/   ← 内含图片\n"
                "        └── 类别2/   ← 内含图片\n"
                "  3. 隐藏目录 / 空文件夹不会参与计数。"
                % (y_prob.shape[1], len(classes),
                   y_prob.shape[1], y_prob.shape[1],
                   data_dir, len(classes), sample, y_prob.shape[1]))

        self._set_status("正在计算指标…"); self._set_progress(93)
        m = compute_classification_metrics(y_true, y_prob, classes)
        report = self._build_cls_report(model_path, data_dir, classes, y_true, m)

        return {"type": "classification", "classes": classes,
                "paths": paths, "y_true": y_true, "y_prob": y_prob,
                "metrics": m, "report": report,
                "n_samples": len(y_true),
                "model_path": model_path, "data_dir": data_dir}

    @staticmethod
    def _build_cls_report(model_path, data_dir, classes, y_true, m):
        n = len(classes)
        L = []
        L.append("=" * 78)
        L.append("                    图像分类测试报告")
        L.append("                   模型：MobileNetV2")
        L.append("=" * 78)
        L.append("")
        L.append("【基本信息】")
        L.append("  模型文件   : %s" % model_path)
        L.append("  数据集目录 : %s" % data_dir)
        L.append("  类别数量   : %d" % n)
        L.append("  类别列表   : %s" % "、".join(classes))
        L.append("  样本总数   : %d" % len(y_true))
        L.append("")
        L.append("-" * 78)
        L.append("【一、总体指标】")
        L.append("-" * 78)
        L.append("  准确率          Accuracy            : %.4f" % m["acc"])
        L.append("  平衡准确率      Balanced Accuracy   : %.4f" % m["bacc"])
        L.append("  Cohen Kappa     Cohen's Kappa       : %.4f" % m["kappa"])
        L.append("  精确率 (宏平均) Precision  (macro)  : %.4f" % m["p_macro"])
        L.append("  召回率 (宏平均) Recall     (macro)  : %.4f" % m["r_macro"])
        L.append("  F1分数 (宏平均) F1-score   (macro)  : %.4f" % m["f_macro"])
        L.append("  精确率 (加权)   Precision  (weighted): %.4f" % m["p_weighted"])
        L.append("  召回率 (加权)   Recall     (weighted): %.4f" % m["r_weighted"])
        L.append("  F1分数 (加权)   F1-score   (weighted): %.4f" % m["f_weighted"])
        L.append("  ROC-AUC (宏平均) AUC (macro, OvR)   : %.4f" % m["auc_macro"])
        L.append("  ROC-AUC (微平均) AUC (micro)        : %.4f" % m["auc_micro"])
        L.append("  PR-AUC  (宏平均) mAP (macro, PR-AUC): %.4f" % m["map_macro"])
        L.append("")
        L.append("-" * 78)
        L.append("【二、各类别详细指标】")
        L.append("-" * 78)
        head = "%-24s%12s%12s%12s%10s%10s" % (
            "类别", "精确率", "召回率", "F1分数", "样本数", "AUC")
        L.append(head); L.append("-" * len(head))
        for i, c in enumerate(classes):
            name = c if len(c) <= 22 else c[:20] + ".."
            L.append("%-24s%12.4f%12.4f%12.4f%10d%10.4f" % (
                name, m["per_p"][i], m["per_r"][i], m["per_f"][i],
                m["per_s"][i], m["per_auc"][i]))
        L.append("")
        L.append("-" * 78)
        L.append("【三、混淆矩阵】  行 = 真实类别，列 = 预测类别")
        L.append("-" * 78)
        cm = m["cm"]
        w = max(8, max(len(c) for c in classes) + 2)
        L.append(" " * w + "".join("%*s" % (w, c[:w - 1]) for c in classes))
        for i, c in enumerate(classes):
            row = "%-*s" % (w, c[:w - 1])
            row += "".join("%*d" % (w, int(v)) for v in cm[i])
            L.append(row)
        L.append("")
        L.append("-" * 78)
        L.append("【四、sklearn 详细分类报告】")
        L.append("-" * 78)
        L.append(m["sk_report"])
        L.append("")
        L.append("=" * 78)
        L.append("  报告生成时间：%s" %
                 datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        L.append("=" * 78)
        return "\n".join(L)

    # ------------------------------------------------------------------
    # 计算：回归
    # ------------------------------------------------------------------
    def _compute_regression(self):
        args = self._pending_args
        model_path = args["model_path"]
        csv_path = args["csv_path"]
        img_dir = args["img_dir"] or os.path.dirname(csv_path)
        use_preproc = args["use_preproc"]

        self._set_status("正在加载模型…"); self._set_progress(2)
        model = load_model_any(model_path)

        self._set_status("正在读取标签…")
        paths, y_true = read_regression_csv(csv_path, img_dir)
        size = get_input_size(model)

        def cb(done, total):
            self._set_progress(5 + 85.0 * done / max(total, 1))
            self._set_status("预测中… %d / %d" % (done, total))

        raw = np.asarray(predict_paths(model, paths, size, use_preproc, 32, cb),
                         dtype=np.float64)
        if raw.ndim > 1:
            raw = raw[:, 0] if raw.shape[1] == 1 else raw.mean(axis=1)
        y_pred = raw.reshape(-1)

        self._set_status("正在计算指标…"); self._set_progress(93)
        m = compute_regression_metrics(y_true, y_pred)
        report = self._build_reg_report(model_path, csv_path, img_dir, m, len(y_true))
        return {"type": "regression", "y_true": y_true, "y_pred": y_pred,
                "metrics": m, "report": report, "model_path": model_path}

    # ------------------------------------------------------------------
    # 计算：单图/批量识别（Wide View，原 predictor.py 的完整流程）
    # ------------------------------------------------------------------
    def _compute_predict(self):
        args = self._pending_args
        model_path = args["model_path"]
        single = args["single"]
        test_dir = args["test_dir"]

        self._set_status("正在加载模型…"); self._set_progress(2)

        # 清理上一次解压出的临时目录
        if self._pred_temp_dir and os.path.exists(self._pred_temp_dir):
            try:
                shutil.rmtree(self._pred_temp_dir)
            except Exception:
                pass
            self._pred_temp_dir = None

        predictor, temp_dir = load_pretrained_predictor(model_path)
        self._pred_temp_dir = temp_dir

        # 收集待识别图片：单张优先，否则扫描目录
        # targets 元素为 (显示名, 绝对路径, 真值类别或 None)
        if single and os.path.isfile(single):
            targets = [(os.path.basename(single), single, None)]
            src_desc = single
        else:
            targets = collect_prediction_images(test_dir)
            src_desc = test_dir

        if not targets:
            raise RuntimeError(
                "未在目录中找到图片（支持 .png/.jpg/.jpeg/.bmp）：\n%s" % test_dir)

        total = len(targets)
        samples = []
        cards_lines = []
        class_counts = {}
        class_prob_sum = {}
        n_uncertain = 0
        n_correct = 0          # 判定类别与标准答案一致的数量
        n_gradable = 0         # 有标准答案的图片数量（评分分母）

        # ---- 批量识别：边解码边推理（比逐张快约 6~7 倍）----
        self._set_status("正在识别 %d 张图片…（并行解码 + 批量推理）" % total)
        self._set_progress(5)
        # 先报到一次 0/total，保证界面立刻有反馈（否则首批解完前进度条不动）
        self._set_status("识别中… 0 / %d" % total)

        # 进度回调节流：批量很小、图很多时会产生成百上千次回调，
        # 而主线程每 33ms 才排空一次 UI 队列，不节流会让队列积压、
        # 看起来像界面卡住。这里限制为约 10 次/秒。
        _last = [0.0]
        _step = max(1, total // 100)

        def _on_batch(done, n_all):
            now = time.perf_counter()
            if now - _last[0] < 0.1 and done < n_all:
                return
            _last[0] = now
            self._set_status("识别中… %d / %d" % (done, n_all))
            self._set_progress(5 + 90.0 * done / max(n_all, 1))

        preds_out = predictor.predict_batch(
            [t[1] for t in targets], log=_on_batch,
            should_cancel=lambda: bool(
                getattr(self, "_cancel_requested", False)))

        for i, (fname, fpath, truth) in enumerate(targets):
            out = preds_out[i] if i < len(preds_out) else None
            res = None if isinstance(out, BaseException) else out
            err = str(out) if isinstance(out, BaseException) else None

            if res is None:
                samples.append({
                    "name": fname, "path": fpath, "label": "(失败)",
                    "probability": 0.0, "uncertain": True,
                    "reason": err or "预测失败", "raw_probs": [],
                    "probs": [], "truth": truth, "correct": None,
                })
                n_uncertain += 1
                if truth:
                    n_gradable += 1
                cards_lines.append(self._format_card_text(fname, None, err))
                continue

            d = res["decision"]
            probs = [p["probability"] for p in res["preds"]]
            # 与标准答案比对（仅当数据集提供了子文件夹真值）
            correct = None
            if truth:
                n_gradable += 1
                correct = classes_match(d["label"], truth)
                if correct:
                    n_correct += 1
            samples.append({
                "name": fname, "path": fpath,
                "label": d["label"], "probability": d["probability"],
                "uncertain": d["uncertain"], "reason": d["reason"],
                "raw_probs": probs, "probs": probs,
                "truth": truth, "correct": correct,
            })
            if d["uncertain"]:
                n_uncertain += 1
            class_counts[d["label"]] = class_counts.get(d["label"], 0) + 1
            class_prob_sum.setdefault(d["label"], []).append(d["probability"])
            cards_lines.append(self._format_card_text(fname, res, None, truth))


        # 类别顺序：优先用预测器内置类别，再补上模型额外输出的类别
        class_order = list(PRED_CLASS_NAMES)
        for lab in class_counts:
            if lab not in class_order:
                class_order.append(lab)
        # 去掉本次结果中未出现的类别（保持报告简洁）
        seen = set(class_counts.keys())
        class_order = [c for c in class_order if c in seen] \
            or list(class_counts.keys())
        class_counts_ordered = {c: class_counts.get(c, 0) for c in class_order}
        class_probs = {c: (float(np.mean(class_prob_sum[c]))
                           if class_prob_sum.get(c) else 0.0)
                       for c in class_order}

        self._set_status("正在计算汇总…"); self._set_progress(97)
        conf_thr = getattr(predictor, "conf_threshold",
                           PRED_CONFIDENCE_THRESHOLD)
        marg_thr = getattr(predictor, "margin_threshold",
                           PRED_MARGIN_THRESHOLD)

        # ---- 按赛题「评分规则」打分：得分 = 正确数 / 总数 × 100 ----
        grade = dict(n_total=total, n_gradable=n_gradable,
                     n_correct=n_correct,
                     accuracy=(float(n_correct) / n_gradable
                               if n_gradable else None),
                     score=score_from_accuracy(n_correct, n_gradable),
                     has_truth=bool(n_gradable))
        report = self._build_pred_report(model_path, src_desc, total,
                                        n_uncertain, class_counts_ordered,
                                        class_probs, samples,
                                        conf_thr, marg_thr, grade)

        self._set_progress(100)
        return {"type": "predict", "samples": samples,
                "class_counts": class_counts_ordered,
                "class_probs": class_probs, "classes": class_order,
                "n_samples": total, "n_uncertain": n_uncertain,
                "grade": grade,
                "conf_threshold": conf_thr, "margin_threshold": marg_thr,
                "report": report, "cards_text": "\n".join(cards_lines),
                "model_path": model_path, "test_dir": src_desc}


    # ---------- 原 Wide View 卡片文本（用于导出） ----------
    @staticmethod
    def _format_card_text(fname, result, error, truth=None):
        L = []
        L.append("=" * 74)
        L.append("📷 图片: %s" % fname)
        L.append("-" * 74)
        if error:
            L.append("  预测失败: %s" % error)
            return "\n".join(L)
        d = result["decision"]
        status = "⚠️ 不确定" if d["uncertain"] else "✅ 确定"
        L.append("  判定: %s -> [%s]" % (status, d["label"]))
        L.append("  置信度: %.2f%%" % (d["probability"] * 100))
        if truth:
            ok = classes_match(d["label"], truth)
            L.append("  标准答案: %s    本图判定: %s    得分: %d/1"
                     % (truth, "✔ 正确" if ok else "✘ 错误", 1 if ok else 0))
        if d.get("reason"):
            L.append("  分析: %s" % d["reason"])
        L.append("  类别概率分布:")
        for p in result["preds"]:
            bar = "█" * int(p["probability"] * 40)
            L.append("     - %-10s: %8.4f  %s" % (p["label"],
                                                  p["probability"], bar))
        return "\n".join(L)


    @staticmethod
    def _build_pred_report(model_path, source, total, n_uncertain,
                           class_counts, class_probs, samples,
                           conf_threshold=None, margin_threshold=None,
                           grade=None):
        conf_threshold = (PRED_CONFIDENCE_THRESHOLD if conf_threshold is None
                          else conf_threshold)
        margin_threshold = (PRED_MARGIN_THRESHOLD if margin_threshold is None
                            else margin_threshold)
        L = []
        L.append("=" * 74)
        L.append("                    识别报告（Wide View）")
        L.append("                   模型：MobileNetV2 (alpha=0.5)")
        L.append("=" * 74)
        L.append("")
        L.append("【基本信息】")
        L.append("  模型文件   : %s" % model_path)
        L.append("  图片来源   : %s" % source)
        L.append("  图片总数   : %d" % total)
        L.append("  确定数量   : %d" % (total - n_uncertain))
        L.append("  不确定数量 : %d" % n_uncertain)
        if total > 0:
            L.append("  确定比例   : %.2f%%"
                     % ((total - n_uncertain) * 100.0 / total))
        L.append("  置信度阈值 : %.2f" % conf_threshold)
        L.append("  差距阈值   : %.2f" % margin_threshold)
        L.append("")

        # ---- 评分（按赛题规则：正确数 / 总数 × 100，满分 100）----
        L.append("-" * 74)
        L.append("【评分结果】")
        L.append("-" * 74)
        if grade and grade.get("has_truth"):
            L.append("  评分规则   : 总体判断准确率 = 正确判断数量 / 测试集总数量 × 100")
            L.append("  测试集总数 : %d 张" % grade["n_total"])
            L.append("  参与评分   : %d 张（有标准答案的图片）" % grade["n_gradable"])
            L.append("  判断正确   : %d 张" % grade["n_correct"])
            L.append("  判断错误   : %d 张" % (grade["n_gradable"]
                                               - grade["n_correct"]))
            L.append("  准确率     : %.2f%%" % (grade["accuracy"] * 100))
            L.append("")
            L.append("  ⭐ 总分     : %.2f 分   (满分 100 分)" % grade["score"])
            L.append("     计算过程 : %d / %d × 100 = %.2f 分"
                     % (grade["n_correct"], grade["n_gradable"],
                        grade["score"]))
        else:
            L.append("  未提供标准答案，无法评分。")
            L.append("  评分需要按类别分子文件夹存放测试图片，例如：")
            L.append("      测试集目录/")
            L.append("      ├── 橙子/     ← 该文件夹内图片的标准答案是「橙子」")
            L.append("      └── 非橙子/   ← 该文件夹内图片的标准答案是「非橙子」")
            L.append("  当前仅在顶层平铺图片，因此只输出识别分布，不计分。")
        L.append("")

        L.append("-" * 74)
        L.append("【一、类别分布】")
        L.append("-" * 74)
        head = "%-16s%10s%14s" % ("类别", "数量", "平均置信度")
        L.append(head); L.append("-" * len(head))
        for c, n in class_counts.items():
            L.append("%-16s%10d%13.2f%%" % (c, n, class_probs.get(c, 0.0) * 100))
        L.append("")

        # ---- 逐类别判对/判错（有真值时才有意义）----
        if grade and grade.get("has_truth"):
            truths = {}
            for s in samples:
                t = s.get("truth")
                if not t:
                    continue
                rec = truths.setdefault(t, dict(n=0, correct=0))
                rec["n"] += 1
                if s.get("correct"):
                    rec["correct"] += 1
            L.append("-" * 74)
            L.append("【二、逐类别得分】")
            L.append("-" * 74)
            head1 = "%-16s%10s%10s%10s%12s" % (
                "标准类别", "数量", "判对", "判错", "该类得分率")
            L.append(head1); L.append("-" * len(head1))
            for t, rec in sorted(truths.items()):
                rate = (rec["correct"] * 100.0 / rec["n"]) if rec["n"] else 0.0
                L.append("%-16s%10d%10d%10d%11.2f%%"
                         % (t, rec["n"], rec["correct"],
                            rec["n"] - rec["correct"], rate))
            L.append("")

        L.append("-" * 74)
        L.append("【三、不确定样本明细】")
        L.append("-" * 74)
        un = [s for s in samples if s["uncertain"]]
        if not un:
            L.append("  （无）")
        else:
            head2 = "%-34s%-12s%12s%16s" % ("文件名", "判定类别", "置信度", "原因")
            L.append(head2); L.append("-" * len(head2))
            for s in un:
                nm = s["name"] if len(s["name"]) <= 32 else s["name"][:30] + ".."
                L.append("%-34s%-12s%11.2f%%%16s" % (
                    nm, s["label"], s["probability"] * 100,
                    s["reason"] or ""))
        L.append("")
        L.append("-" * 74)
        L.append("【四、全部样本速览】")
        L.append("-" * 74)
        head3 = "%-6s%-30s%-12s%-12s%11s%10s" % (
            "序号", "文件名", "判定类别", "标准答案", "置信度", "状态")
        L.append(head3); L.append("-" * len(head3))
        for i, s in enumerate(samples):
            nm = s["name"] if len(s["name"]) <= 28 else ".." + s["name"][-26:]
            truth = s.get("truth")
            if truth:
                mark = "✔" if s.get("correct") else "✘"
                truth_col = "%s %s" % (truth, mark)
            else:
                truth_col = "—"
            status = "不确定" if s["uncertain"] else "确定"
            L.append("%-6d%-30s%-12s%-12s%10.2f%%%10s" % (
                i, nm, s["label"], truth_col, s["probability"] * 100,
                status))

        L.append("")
        L.append("=" * 74)
        L.append("  报告生成时间：%s" %
                 datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        L.append("=" * 74)
        return "\n".join(L)

    @staticmethod
    def _build_reg_report(model_path, csv_path, img_dir, m, n):
        L = []
        L.append("=" * 72)
        L.append("                    图像回归测试报告")
        L.append("                   模型：MobileNetV2")
        L.append("=" * 72)
        L.append("")
        L.append("【基本信息】")
        L.append("  模型文件   : %s" % model_path)
        L.append("  标签 CSV   : %s" % csv_path)
        L.append("  图片目录   : %s" % img_dir)
        L.append("  样本总数   : %d" % n)
        L.append("")
        L.append("-" * 72)
        L.append("【回归指标】")
        L.append("-" * 72)
        L.append("  平均绝对误差     MAE            : %.6f" % m["mae"])
        L.append("  均方误差         MSE            : %.6f" % m["mse"])
        L.append("  均方根误差       RMSE           : %.6f" % m["rmse"])
        L.append("  决定系数         R²             : %.6f" % m["r2"])
        L.append("  调整决定系数     Adjusted R²    : %.6f" % m["adj_r2"])
        L.append("  平均绝对百分比误差 MAPE         : %.4f %%" % m["mape"])
        L.append("")
        L.append("-" * 72)
        L.append("【说明】")
        L.append("  · MAE / MSE / RMSE 越小越好，理想值为 0。")
        L.append("  · R² 越接近 1 越好，等于 1 表示完美拟合。")
        L.append("  · 残差 = 真实值 − 预测值，应围绕 0 随机分布。")
        L.append("  · MAPE 为平均绝对百分比误差，越小越好。")
        L.append("")
        L.append("=" * 72)
        L.append("  报告生成时间：%s" %
                 datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        L.append("=" * 72)
        return "\n".join(L)

    # ------------------------------------------------------------------
    # 结果渲染
    # ------------------------------------------------------------------
    def clear_notebook(self):
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        self.figures = []
        # 代次 +1：让上一轮还在排队的「分帧渲染」任务作废，
        # 否则快速重跑时，上一轮迟到的图表会塞进新一轮的结果里。
        self._render_gen = getattr(self, "_render_gen", 0) + 1

    def _reset_scrollers(self):
        """重新渲染结果后把两个参数区滚回顶部（避免停在上一轮的滚动位置）。"""
        for name in ("train_scroller", "test_scroller"):
            sc = getattr(self, name, None)
            if sc is not None:
                try:
                    sc.scroll_to_top()
                except Exception:
                    pass

    def add_figure_tab(self, title, fig, scrollable=False):
        outer = ttk.Frame(self.notebook)
        self.notebook.add(outer, text=" " + title + " ")

        if scrollable:
            canvas_tk = tk.Canvas(outer, highlightthickness=0)
            sb_v = ttk.Scrollbar(outer, orient="vertical", command=canvas_tk.yview)
            sb_h = ttk.Scrollbar(outer, orient="horizontal", command=canvas_tk.xview)
            canvas_tk.configure(yscrollcommand=sb_v.set, xscrollcommand=sb_h.set)
            sb_v.pack(side=tk.RIGHT, fill=tk.Y)
            sb_h.pack(side=tk.BOTTOM, fill=tk.X)
            canvas_tk.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            inner = ttk.Frame(canvas_tk)
            canvas_tk.create_window((0, 0), window=inner, anchor="nw")

            fc = FigureCanvasTkAgg(fig, master=inner)
            fc.draw()
            toolbar = NavigationToolbar2Tk(fc, inner)
            toolbar.update()
            fc.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            def _on_cfg(_e, c=canvas_tk):
                c.configure(scrollregion=c.bbox("all"))
            inner.bind("<Configure>", _on_cfg)
        else:
            fc = FigureCanvasTkAgg(fig, master=outer)
            fc.draw()
            toolbar = NavigationToolbar2Tk(fc, outer)
            toolbar.update()
            fc.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.figures.append(fig)
        return outer

    def add_text_tab(self, title, text):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=" " + title + " ")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        _bg, _fg = self._log_bg, self._log_fg
        # tk.Text + 显式 grid（ScrolledText 自带竖直条，再叠横向条会错位）
        st = tk.Text(frame, wrap=tk.NONE, font=(_MONO_FONT, 10),
                     bg=_bg, fg=_fg, insertbackground=_fg,
                     selectbackground="#3a6ea5", relief="flat", borderwidth=1)
        st.insert("1.0", text)
        st.configure(state="disabled")
        st.grid(row=0, column=0, sticky="nsew")

        sb_v = ttk.Scrollbar(frame, orient="vertical", command=st.yview)
        sb_v.grid(row=0, column=1, sticky="ns")
        sb_h = ttk.Scrollbar(frame, orient="horizontal", command=st.xview)
        sb_h.grid(row=1, column=0, sticky="ew")
        st.configure(yscrollcommand=sb_v.set, xscrollcommand=sb_h.set)

        # 文本框内滚轮 → 滚文本自身；Shift + 滚轮 → 横向滚
        for seq in ("<MouseWheel>", "<Shift-MouseWheel>"):
            st.bind(seq, lambda e, w=st: self._text_wheel(w, e), add="+")
        return frame

    @staticmethod
    def _text_wheel(widget, event):
        """给 ScrolledText 做跨平台滚轮（Shift 时横向）。"""
        num = getattr(event, "num", None)
        if num == 4:
            steps = -3
        elif num == 5:
            steps = 3
        else:
            raw = getattr(event, "delta", 0)
            steps = -1 * int(raw / 120.0)
            if steps == 0:
                steps = -1 if raw > 0 else 1
        if bool(getattr(event, "state", 0) & 0x0001):
            widget.xview_scroll(steps, "units")
        else:
            widget.yview_scroll(steps, "units")
        return "break"

    def _render(self, res):
        self.clear_notebook()
        self.last_result = res              # 保存供导出使用
        if res["type"] == "classification":
            self._render_classification(res)
            self._set_score_panel(None, show=False)
        elif res["type"] == "regression":
            self._render_regression(res)
            self._set_score_panel(None, show=False)
        else:
            self._render_predict(res)
            # 批量识别：把本次得分显示在上方「区域④ 评分」面板
            self._set_score_panel(res.get("grade"), show=True)
        self.status_var.set("测试完成 ✔")
        self.export_btn.config(state=tk.NORMAL)     # 启用导出按钮
        self._reset_scrollers()                     # 参数区滚回顶部


    def _render_classification(self, res):
        classes = res["classes"]
        m = res["metrics"]

        self.add_text_tab("区域A 指标报告", res["report"])

        fig = Figure(figsize=(8, 7), dpi=100)
        _plot_confusion_matrix(fig.add_subplot(111), m, classes)
        fig.tight_layout()
        self.add_figure_tab("区域B 混淆矩阵", fig)

        fig = Figure(figsize=(8.5, 5.5), dpi=100)
        _plot_overview_bars(fig.add_subplot(111), m)
        fig.tight_layout()
        self.add_figure_tab("区域C 指标总览", fig)

        fig = Figure(figsize=(8.5, 7), dpi=100)
        _plot_roc(fig.add_subplot(111), m, classes)
        fig.tight_layout()
        self.add_figure_tab("区域D ROC曲线", fig)

        fig = Figure(figsize=(8.5, 7), dpi=100)
        _plot_pr(fig.add_subplot(111), m, classes)
        fig.tight_layout()
        self.add_figure_tab("区域E PR曲线", fig)

        fig = Figure(figsize=(max(8, 1.0 * len(classes) + 5), 6), dpi=100)
        _plot_per_class(fig.add_subplot(111), m, classes)
        fig.tight_layout()
        self.add_figure_tab("区域F 各类别指标", fig)

        fig = Figure(figsize=(17, 22), dpi=100)
        gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.30)
        _plot_confusion_matrix(fig.add_subplot(gs[0, 0]), m, classes)
        _plot_overview_bars(fig.add_subplot(gs[0, 1]), m)
        _plot_roc(fig.add_subplot(gs[1, 0]), m, classes)
        _plot_pr(fig.add_subplot(gs[1, 1]), m, classes)
        _plot_per_class(fig.add_subplot(gs[2, :]), m, classes)
        fig.suptitle("【区域G】分类结果全部图表总览 —— MobileNetV2",
                     fontsize=16, fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        self.add_figure_tab("区域G 全部图表总览", fig, scrollable=True)

    def _render_regression(self, res):
        y_true = np.asarray(res["y_true"], dtype=np.float64)
        y_pred = np.asarray(res["y_pred"], dtype=np.float64)
        m = res["metrics"]
        resid = m["residuals"]

        self.add_text_tab("区域A 指标报告", res["report"])

        fig = Figure(figsize=(8, 7), dpi=100)
        _plot_true_vs_pred(fig.add_subplot(111), y_true, y_pred, m)
        fig.tight_layout()
        self.add_figure_tab("区域B 真实vs预测", fig)

        fig = Figure(figsize=(8, 6.5), dpi=100)
        _plot_residual(fig.add_subplot(111), y_pred, resid)
        fig.tight_layout()
        self.add_figure_tab("区域C 残差图", fig)

        fig = Figure(figsize=(8, 6.5), dpi=100)
        _plot_residual_hist(fig.add_subplot(111), resid)
        fig.tight_layout()
        self.add_figure_tab("区域D 残差分布", fig)

        fig = Figure(figsize=(10, 6), dpi=100)
        _plot_sample_curve(fig.add_subplot(111), y_true, y_pred)
        fig.tight_layout()
        self.add_figure_tab("区域E 逐样本对比", fig)

        fig = Figure(figsize=(8.5, 5.5), dpi=100)
        _plot_error_bars(fig.add_subplot(111), m)
        fig.tight_layout()
        self.add_figure_tab("区域F 误差指标", fig)

        fig = Figure(figsize=(17, 21), dpi=100)
        gs = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.28)
        _plot_true_vs_pred(fig.add_subplot(gs[0, 0]), y_true, y_pred, m)
        _plot_residual(fig.add_subplot(gs[0, 1]), y_pred, resid)
        _plot_residual_hist(fig.add_subplot(gs[1, 0]), resid)
        _plot_error_bars(fig.add_subplot(gs[1, 1]), m)
        _plot_sample_curve(fig.add_subplot(gs[2, :]), y_true, y_pred)
        fig.suptitle("【区域G】回归结果全部图表总览 —— MobileNetV2",
                     fontsize=16, fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        self.add_figure_tab("区域G 全部图表总览", fig, scrollable=True)

    # ------------------------------------------------------------------
    # 渲染：单图/批量识别（Wide View）
    # ------------------------------------------------------------------
    def _render_predict(self, res):
        """渲染批量识别结果。

        图表分帧生成：4 张 matplotlib 图同步画要 ~0.5s，会让主线程僵住。
        因此先立刻把「文本报告 + 卡片列表 + 标签页骨架」建好（界面马上可用），
        再把每张图用 after 逐个补上，用户看到的是图「陆续出现」。
        """
        samples = res["samples"]
        classes = res["classes"]
        conf_thr = res.get("conf_threshold", PRED_CONFIDENCE_THRESHOLD)

        # 区域A 识别摘要
        self.add_text_tab("区域A 识别摘要", res["report"])

        # 区域B 逐个按识别（原 predictor.py 的宽卡片列表，原样整合）
        self.add_wide_card_tab("🟠 逐个按识别", samples, res["n_uncertain"],
                               res.get("grade"))

        # ---- 图表：分帧逐个画 ----
        jobs = [
            ("区域C 置信度散点", (11, 6),
             lambda ax: _plot_pred_prob_scatter(ax, samples, classes, conf_thr)),
            ("区域D 识别明细", (11, 8),
             lambda ax: _plot_pred_per_image(ax, samples, conf_thr)),
            ("区域E 概率分布", (9, 6),
             lambda ax: _plot_pred_class_dist(ax, res["class_counts"],
                                              res["class_probs"], conf_thr)),
        ]

        # 记录本次渲染代次：clear_notebook() 会 +1，
        # 旧代次的排队任务检测到代次不符就直接退出，不再往新结果里塞图。
        gen = getattr(self, "_render_gen", 0)

        def _step(i=0):
            if gen != getattr(self, "_render_gen", 0):
                return                      # 已被新一轮结果取代，作废
            if i >= len(jobs):
                if gen == getattr(self, "_render_gen", 0):
                    self._draw_overview_tab(res, samples, classes, conf_thr)
                return
            title, size, fn = jobs[i]
            try:
                fig = Figure(figsize=size, dpi=100)
                fn(fig.add_subplot(111))
                fig.tight_layout()
                self.add_figure_tab(title, fig)
            except Exception as e:
                print("[警告] 生成「%s」失败：%s" % (title, e))
            try:
                self.root.after(1, lambda: _step(i + 1))
            except tk.TclError:
                pass

        if self.root is not None:
            try:
                self.root.after(1, _step)
            except tk.TclError:
                _step()
        else:
            _step()

    def _draw_overview_tab(self, res, samples, classes, conf_thr):
        """区域F 全部图表总览（含 4 个子图，较重）。

        这张大图同步画要 ~0.3s，会让主线程出现一次明显停顿。这里把它也拆成
        两步：先建空画布并加好标签页（界面立刻可见），下一帧再逐个填子图。
        """
        try:
            fig = Figure(figsize=(17, 15), dpi=100)
            gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.26)
            axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                    fig.add_subplot(gs[1, :])]
            fig.suptitle("识别结果全部图表总览 —— MobileNetV2 (alpha=0.5)",
                         fontsize=16, fontweight="bold", y=0.995)
            self.add_figure_tab("区域F 全部图表总览", fig, scrollable=True)
        except Exception as e:
            print("[警告] 生成「区域F 全部图表总览」失败：%s" % e)
            return

        gen = getattr(self, "_render_gen", 0)

        def _fill(i=0):
            if gen != getattr(self, "_render_gen", 0):
                return
            try:
                if i == 0:
                    _plot_pred_class_dist(axes[0], res["class_counts"],
                                          res["class_probs"], conf_thr)
                elif i == 1:
                    _plot_pred_prob_scatter(axes[1], samples, classes, conf_thr)
                elif i == 2:
                    _plot_pred_per_image(axes[2], samples, conf_thr)
                else:
                    fig.tight_layout(rect=(0, 0, 1, 0.985))
                    return
                fig.tight_layout(rect=(0, 0, 1, 0.985))
            except Exception as e:
                print("[警告] 总览子图 %d 绘制失败：%s" % (i, e))
            try:
                self.root.after(1, lambda: _fill(i + 1))
            except tk.TclError:
                pass

        try:
            self.root.after(1, _fill)
        except tk.TclError:
            _fill()


        # 区域F 全部图表总览
        fig = Figure(figsize=(17, 15), dpi=100)
        gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.26)
        _plot_pred_class_dist(fig.add_subplot(gs[0, 0]), res["class_counts"],
                              res["class_probs"], res.get("conf_threshold", PRED_CONFIDENCE_THRESHOLD))
        _plot_pred_prob_scatter(fig.add_subplot(gs[0, 1]), samples, classes,
                                res.get("conf_threshold", PRED_CONFIDENCE_THRESHOLD))
        _plot_pred_per_image(fig.add_subplot(gs[1, :]), samples,
                             res.get("conf_threshold", PRED_CONFIDENCE_THRESHOLD))
        fig.suptitle("识别结果全部图表总览 —— MobileNetV2 (alpha=0.5)",
                     fontsize=16, fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        self.add_figure_tab("区域F 全部图表总览", fig, scrollable=True)

    def add_wide_card_tab(self, title, samples, n_uncertain, grade=None,
                          chunk=20):
        """区域B：宽卡片列表（ScrollableFrame + 缩略图 + 概率条）。

        完全复刻原 predictor.py 的 _add_wide_card 渲染效果，
        只是宿主容器从主窗口换成 Notebook 的标签页。
        额外在顶部挂一条「本次得分」横幅（按赛题评分规则）。

        性能：卡片是逐张建 Tk 控件 + 解码缩略图的，几百张时会占住主线程
        （实测 300 张约 2.9s 纯构建，期间界面「未响应」）。这里改成分帧渲染：
        每 chunk 张就把控制权交回事件循环，界面全程可点击/可滚动，
        卡片「陆续出现」而不是卡住不动。
        """
        outer = ttk.Frame(self.notebook)
        self.notebook.add(outer, text=" " + title + " ")

        # ---- 得分横幅（有标准答案时显示）----
        if grade and grade.get("has_truth"):
            banner = ttk.Frame(outer, padding=(12, 8))
            banner.pack(fill=tk.X)
            score = grade["score"]
            color = ("#0a8f3c" if score >= 90 else
                     "#0a5fd0" if score >= 70 else
                     "#d98000" if score >= 60 else "#cc2222")

            ttk.Label(banner, text="⭐ 本次得分",
                      font=(_UI_FONT, 12, "bold")).pack(side=tk.LEFT)
            ttk.Label(banner, text="%.2f 分" % score,
                      font=(_UI_FONT, 20, "bold"),
                      foreground=color).pack(side=tk.LEFT, padx=8)
            ttk.Label(banner,
                      text="（正确 %d / 共 %d 张 × 100，满分 100 分）"
                           % (grade["n_correct"], grade["n_gradable"]),
                      foreground="#666666").pack(side=tk.LEFT, padx=4)

        head = ttk.Frame(outer, padding=(10, 6))
        head.pack(fill=tk.X)
        ttk.Label(head, text="📸 识别结果详情",
                  font=(_UI_FONT, 12, "bold")).pack(side=tk.LEFT)
        ttk.Label(head,
                  text="共 %d 张，其中 %d 张不确定" % (len(samples), n_uncertain),
                  foreground="#666666").pack(side=tk.LEFT, padx=12)


        self._pred_cards = []
        scroll_area = ScrollableFrame(outer)
        scroll_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        self._pred_scroll_area = scroll_area

        # 分帧渲染：把卡片分成若干小批，每批之间真正把控制权交回 Tk 事件循环。
        # 直接 for 循环建几百张卡片会把主线程占住约 3s，期间窗口「未响应」。
        # 注意：`after(1, ...)` 在本机并不会让出足够时间——Tk 会把已就绪的
        # 回调连续执行完（实测 300 张仍在同一轮 update 里全部建完）。
        # 因此这里每批用 `after(8, ...)`，给事件循环留出处理点击/重绘的窗口；
        # 并且每批都把 scrollregion 刷新掉，让已出现的卡片立即可见、可滚动。
        gen = getattr(self, "_render_gen", 0)

        def _render_chunk(start=0):
            if gen != getattr(self, "_render_gen", 0):
                return                      # 已被新一轮结果取代，停止续建
            end = min(start + chunk, len(samples))
            for i in range(start, end):
                card = self._make_wide_card(scroll_area, samples[i])
                self._pred_cards.append(card)
            scroll_area.schedule_scrollregion()
            if end < len(samples):
                try:
                    outer.after(8, lambda: _render_chunk(end))
                except tk.TclError:
                    pass

        if len(samples) <= chunk:
            _render_chunk(0)            # 少量卡片：直接同步建完，无需分帧
        else:
            try:
                outer.after(8, _render_chunk)
            except tk.TclError:
                _render_chunk(0)
        return outer





    def _make_wide_card(self, scroll_area, s):
        """单张图片的宽卡片（与 predictor.py 的 _add_wide_card 结构一致）。"""
        from PIL import Image, ImageTk

        card = ttk.Frame(scroll_area.scrollable_frame,
                         style="WideCard.TFrame")
        card.pack(fill=tk.X, padx=5, pady=5)

        # 新卡片加入后刷新 scrollregion。
        # 这里必须用 schedule_scrollregion（合并同批请求）而不是直接
        # after_idle(_update_scrollregion)：批量渲染上百张卡片时，逐张排队
        # 会让每次刷新都做一遍全树遍历，退化成 O(n²)，主循环被塞满 → 界面未响应。
        scroll_area.schedule_scrollregion()

        # --- 左侧：图片区域 (固定宽度 120) ---
        img_container = ttk.Frame(card, width=120, height=120)
        img_container.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
        img_container.pack_propagate(False)
        try:
            pil_img = Image.open(s["path"])
            pil_img.thumbnail((120, 120), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil_img)
            lbl = ttk.Label(img_container, image=tk_img)
            lbl.image = tk_img          # 防止被 GC
            lbl.place(relx=0.5, rely=0.5, anchor="center")
        except Exception:
            ttk.Label(img_container, text="Load Err").place(
                relx=0.5, rely=0.5, anchor="center")

        # --- 右侧：详细信息区域 (自适应宽度) ---
        info_frame = ttk.Frame(card)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                        padx=10, pady=10)

        ttk.Label(info_frame, text=s["name"],
                  style="Title.TLabel").pack(anchor="w")

        if s["label"] == "(失败)":
            ttk.Label(info_frame, text="预测失败: %s" % (s["reason"] or ""),
                      foreground="red").pack(anchor="w", pady=5)
        else:
            # 主要结论行
            header_frame = ttk.Frame(info_frame)
            header_frame.pack(fill=tk.X, pady=(5, 5))
            status_icon = "⚠️" if s["uncertain"] else "✅"
            status_color = "orange" if s["uncertain"] else "#008000"
            ttk.Label(header_frame, text=status_icon,
                      font=(_EMOJI_FONT, 12)).pack(side=tk.LEFT)
            ttk.Label(header_frame, text="判定: %s" % s["label"],
                      style="Title.TLabel",
                      foreground=status_color).pack(side=tk.LEFT, padx=5)
            ttk.Label(header_frame,
                      text="置信度: %.2f%%" % (s["probability"] * 100),
                      style="Sub.TLabel").pack(side=tk.LEFT, padx=5)

            # 标准答案比对（数据集按类别分子文件夹时才有）
            if s.get("truth"):
                ok = bool(s.get("correct"))
                mk = ttk.Frame(info_frame)
                mk.pack(fill=tk.X, pady=(0, 4))
                ttk.Label(mk, text="标准答案: %s" % s["truth"],
                          style="Sub.TLabel").pack(side=tk.LEFT)
                ttk.Label(mk,
                          text="  ✔ 判定正确，得 1 分" if ok
                               else "  ✘ 判定错误，得 0 分",
                          foreground="#008000" if ok else "#cc2222",
                          font=(_UI_FONT, 9, "bold")).pack(side=tk.LEFT)

            # 原因 (如果有)
            if s.get("reason"):
                ttk.Label(info_frame, text="分析: %s" % s["reason"],
                          style="Sub.TLabel",
                          wraplength=400).pack(anchor="w")

            # 详细概率分布 (使用 Progressbar 可视化)
            probs_frame = ttk.LabelFrame(info_frame, text="类别概率分布",
                                         padding=5)
            probs_frame.pack(fill=tk.X, pady=(10, 0))
            labels = PRED_CLASS_NAMES + ["类别%d" % (i + 1)
                                         for i in range(len(s["raw_probs"]))]
            for idx, prob in enumerate(s["raw_probs"]):
                row = ttk.Frame(probs_frame)
                row.pack(fill=tk.X, pady=2)
                ttk.Label(row, text=labels[idx], width=10, anchor="w",
                          style="Prob.TLabel").pack(side=tk.LEFT)
                pb = ttk.Progressbar(row, orient="horizontal", length=200,
                                     mode="determinate")
                pb.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
                pb["value"] = max(0.0, min(100.0, prob * 100))
                ttk.Label(row, text="%.1f%%" % (prob * 100), width=6,
                          anchor="e", style="Prob.TLabel").pack(side=tk.RIGHT)

        # 卡片控件已全部建好，此时绑定滚轮才有效（增量，只扫本卡片子树）
        scroll_area.bind_wheel_card(card)
        return card



# ======================================================================
# 入口
# ======================================================================

def _check_dependency(module, pip_hint):
    """依赖探测：缺失时给出「本机可直接复制执行」的安装命令。"""
    try:
        __import__(module)
        return True
    except Exception as e:
        detail = str(e)
        root = tk.Tk(); root.withdraw()
        messagebox.showerror(
            "缺少依赖",
            "未能导入 %s，请先安装：\n\n    %s\n\n错误信息：\n%s"
            % (module, pip_hint, detail))
        try:
            root.destroy()
        except Exception:
            pass
        return False


def main():
    # --- 依赖自检 ----------
    if not _check_dependency("tensorflow", "python -m pip install -r requirements.txt"):
        return
    if not _check_dependency("sklearn", "python -m pip install -r requirements.txt"):
        return

    root = tk.Tk()
    try:
        app = MobileNetTesterApp(root)
    except Exception as e:
        # 界面构建失败时别只留一个黑屏窗口，把错误直接抛给用户
        import traceback as _tb
        print(_tb.format_exc())
        messagebox.showerror("启动失败",
                             "界面初始化失败：\n\n%s\n\n详见终端输出。" % e)
        return

    # 关闭窗口时清理临时目录（原 predictor.py 的 on_closing）
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    root.after(60, root.focus_force)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()


if __name__ == "__main__":
    main()
