from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import shutil
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any

import numpy as np
from PIL import Image, ImageOps

# 配置常量
CLASS_NAMES = ["橙子", "非橙子"]
IMAGE_SIZE = 224
DEFAULT_TRAINING = dict(epochs=20, batchSize=16, learningRate=0.001)
HIDDEN_UNITS = 100
DROPOUT_RATE = 0.2
VALIDATION_SEED = 1592594996
VALIDATION_FRACTION = 0.15
VALIDATION_MIN_TOTAL = 20
CONFIDENCE_THRESHOLD = 0.65
MARGIN_THRESHOLD = 0.12
EPOCH_RANGE = (5, 200)

# 数据体检阈值
MIN_SAMPLES_PER_CLASS = 5
RECOMMENDED_SAMPLES_PER_CLASS = 20
IMBALANCE_RATIO = 2
NEAR_DUP_HAMMING = 6
CONSECUTIVE_CAPTURE_MS = 2500
HASH_W = HASH_H = 8

MOBILENET_OUTPUT_NODE = "module_apply_default/MobilenetV2/Logits/AvgPool"
EMBEDDING_SIZE = 1280
EXTRACT_BATCH_SIZE = 64  # 仅用于 MobileNet 前向提取嵌入；与训练分类头的 batchSize 无关，显存吃紧就调小
DECODE_WORKERS = min(8, os.cpu_count() or 4)  # 并行做图片解码/缩放（CPU 密集）的线程数
STEPS_PER_EXECUTION = 32  # 一次性下发多个训练 step 给 TF，减少小模型多轮次训练时的调度开销


# 数据结构
@dataclass
class Sample:
    id: str
    data: bytes
    source: str = "upload"
    created_at: Optional[str] = None
    capture_group_id: Optional[str] = None


@dataclass
class ClassData:
    id: str
    name: str
    samples: list = field(default_factory=list)


@dataclass
class Project:
    classes: list
    training: dict = field(default_factory=lambda: dict(DEFAULT_TRAINING))


@dataclass
class Example:
    image: bytes
    label: str
    group_id: Optional[str]


# 图像处理工具
def _center_crop_on_white(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """白底 224×224 画布 + 取最短边居中正方形绘入。"""
    img = ImageOps.exif_transpose(img).convert("RGBA")
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) / 2, (h - side) / 2
    crop = img.resize((size, size), Image.BILINEAR, box=(left, top, left + side, top + side))
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(crop, (0, 0), crop)
    return canvas


def prepare_upload_jpeg(path: Path, quality: int = 90) -> bytes:
    """模拟页面上传图片处理。"""
    canvas = _center_crop_on_white(Image.open(path))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def decode_for_model(data: bytes) -> np.ndarray:
    """训练/预测时送进 MobileNet 前的处理 → uint8 (224,224,3)。"""
    return np.asarray(_center_crop_on_white(Image.open(io.BytesIO(data))), dtype=np.uint8)


# 数据加载
def load_project_folder(root: str) -> Project:
    """从目录结构加载: root/橙子/*, root/非橙子/*"""
    classes = []
    for idx, name in enumerate(CLASS_NAMES):
        d = Path(root) / name
        if not d.is_dir():
            raise FileNotFoundError(f"找不到类别目录: {d}")
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in
                       {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"})
        samples = [Sample(id=f"sample-{idx}-{i:04d}", data=prepare_upload_jpeg(p)) for i, p in enumerate(files)]
        classes.append(ClassData(id=f"class-{'ab'[idx]}", name=name, samples=samples))
    return Project(classes)


def load_project_json(path: str) -> Project:
    """读取页面导出的 tm-object-project JSON。"""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("format") != "tm-object-project" or obj.get("formatVersion") != 1:
        raise ValueError("项目格式或版本不受支持（需要 tm-object-project v1）")

    classes = []
    for c in obj["classes"]:
        samples = []
        for s in c["samples"]:
            m = re.match(r"^data:image/jpeg;base64,([A-Za-z0-9+/]+={0,2})$", s["dataUrl"])
            if not m:
                raise ValueError(f"样本 {s.get('id')} 的 dataUrl 不是 base64 JPEG")
            samples.append(Sample(
                id=s["id"],
                data=base64.b64decode(m.group(1)),
                source=s.get("source", "upload"),
                created_at=s.get("createdAt"),
                capture_group_id=s.get("captureGroupId")
            ))
        classes.append(ClassData(id=c["id"], name=c["name"], samples=samples))

    training = {**DEFAULT_TRAINING, **obj.get("training", {})}
    return Project(classes, training)


#  数据健康检查与分组
def _norm(name: str) -> str:
    return unicodedata.normalize("NFKC", name).strip()


def valid_class_names(names) -> bool:
    n = [_norm(x) for x in names]
    return len(n) == len(CLASS_NAMES) and len(set(n)) == len(CLASS_NAMES) and all(c in n for c in CLASS_NAMES)


def _parse_ms(iso: Optional[str]) -> Optional[float]:
    if not iso: return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000
    except ValueError:
        return None


def perceptual_hash(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((HASH_W, HASH_H), Image.BILINEAR)
    px = np.asarray(img, dtype=np.int64).reshape(-1, 3)
    gray = np.floor((px[:, 0] * 299 + px[:, 1] * 587 + px[:, 2] * 114) / 1000 + 0.5).astype(np.int64)
    mean = gray.sum() / gray.size
    out = bytearray(1 + gray.size // 8)
    l = int(math.floor(mean + 0.5)) & 0xFF
    out[0] = l ^ (l >> 1)
    for i, g in enumerate(gray):
        if g >= mean: out[1 + i // 8] |= 1 << (i % 8)
    return bytes(out)


def _hamming(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def _dup_groups(items):
    buckets: dict = {}
    for cid, s in items:
        key = hashlib.sha1(s.data).digest() + len(s.data).to_bytes(8, "big")
        buckets.setdefault(key, []).append((cid, s))
    return [g for g in buckets.values() if len(g) > 1]


def data_health_report(project: Project) -> list:
    issues = []
    for c in project.classes:
        name = c.name.strip() or "未命名类别"
        n = len(c.samples)
        if n < MIN_SAMPLES_PER_CLASS:
            issues.append(dict(severity="error", code="insufficient-samples",
                               message=f"“{name}”只有 {n} 张图片，至少需要 {MIN_SAMPLES_PER_CLASS} 张才能训练。"))
        elif n < RECOMMENDED_SAMPLES_PER_CLASS:
            issues.append(dict(severity="warning", code="below-recommended-samples",
                               message=f"“{name}”有 {n} 张图片，建议补充到 {RECOMMENDED_SAMPLES_PER_CLASS} 张以上。"))

        # 连续拍摄近似重复检查
        cams = [(s, _parse_ms(s.created_at), i) for i, s in enumerate(c.samples) if s.source == "camera"]
        cams = sorted([x for x in cams if x[1] is not None], key=lambda x: (x[1], x[2]))
        chain = None
        for (a, ta, _), (b, tb, _) in zip(cams, cams[1:]):
            if tb - ta <= CONSECUTIVE_CAPTURE_MS and a.data != b.data \
                    and _hamming(perceptual_hash(a.data), perceptual_hash(b.data)) <= NEAR_DUP_HAMMING:
                if chain and chain[-1] is a:
                    chain.append(b)
                else:
                    chain = [a, b]
                    issues.append(
                        dict(severity="warning", code="consecutive-camera-near-duplicate", _chain=chain, message=""))
            else:
                chain = None

        for it in issues:
            if it["code"] == "consecutive-camera-near-duplicate" and not it["message"]:
                it[
                    "message"] = f"“{name}”中有 {len(it.pop('_chain'))} 张连续采集图片非常相似，建议只保留有明显变化的画面。"

    counts = [len(c.samples) for c in project.classes]
    if len(counts) >= 2:
        lo, hi = min(counts), max(counts)
        if hi > 0 and (hi >= MIN_SAMPLES_PER_CLASS if lo == 0 else hi / lo >= IMBALANCE_RATIO):
            issues.append(dict(severity="warning", code="class-imbalance",
                               message=f"类别样本数量差异较大（最少 {lo} 张，最多 {hi} 张），建议保持各类别数量接近。"))

    for g in _dup_groups([(c.id, s) for c in project.classes for s in c.samples]):
        if len({cid for cid, _ in g}) >= 2:
            issues.append(dict(severity="error", code="cross-class-exact-duplicate",
                               message="同一张图片出现在多个类别中，可能造成标签冲突。"))
    return issues


def build_examples(project: Project) -> list:
    examples = []
    for c in project.classes:
        parent: dict = {}

        def find(x, parent=parent):
            r = parent.get(x, x)
            if r == x: return x
            root = find(r)
            parent[x] = root
            return root

        def union(a, b, parent=parent, find=find):
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra

        base = [s.capture_group_id if s.capture_group_id
                else (f"legacy-camera-{c.id}" if s.source == "camera" else f"image-{s.id}")
                for s in c.samples]

        first_by_data: dict = {}
        for s, g in zip(c.samples, base):
            parent.setdefault(g, g)
            if s.data in first_by_data:
                union(first_by_data[s.data], g)
            else:
                first_by_data[s.data] = g

        for s, g in zip(c.samples, base):
            examples.append(Example(image=s.data, label=c.name.strip(), group_id=find(g)))
    return examples


#  随机数与验证集划分
def mulberry32(seed: int) -> Callable[[], float]:
    M = 0xFFFFFFFF
    t = seed & M

    def imul(a, b): return (a * b) & M

    def rng() -> float:
        nonlocal t
        t = (t + 1831565813) & M
        e = t
        e = imul(e ^ (e >> 15), e | 1)
        e ^= (e + imul(e ^ (e >> 7), e | 61)) & M
        return ((e ^ (e >> 14)) & M) / 4294967296

    return rng


def _shuffle(items: list, rng) -> None:
    for n in range(len(items) - 1, 0, -1):
        r = math.floor(rng() * (n + 1))
        items[n], items[r] = items[r], items[n]


def _pick_validation_groups(groups: list, target: int, rng) -> set:
    r = list(groups)
    _shuffle(r, rng)
    r.sort(key=lambda g: len(g[1]))
    chosen, acc = set(), 0
    for key, idx in r:
        if len(chosen) >= len(groups) - 1: break
        if abs(target - acc - len(idx)) < abs(target - acc):
            chosen.add(key)
            acc += len(idx)
    if not chosen: chosen.add(r[0][0])
    return chosen


def group_stratified_split(examples, class_idx, num_classes, validation_split, seed=VALIDATION_SEED):
    if validation_split <= 0: return list(range(len(class_idx))), [], "disabled"
    per_class = [dict() for _ in range(num_classes)]
    for i, c in enumerate(class_idx):
        gid = examples[i].group_id
        gid = gid.strip() if isinstance(gid, str) else None
        key = f"group:{gid}" if gid else f"sample:{i}"
        per_class[c].setdefault(key, []).append(i)

    if any(len(m) < 2 for m in per_class): return list(range(len(class_idx))), [], "insufficient-groups"

    rng = mulberry32(seed)
    train, val = [], []
    for m in per_class:
        groups = list(m.items())
        n = sum(len(ix) for _, ix in groups)
        chosen = _pick_validation_groups(groups, max(1, math.ceil(n * validation_split)), rng)
        for key, ix in groups:
            (val if key in chosen else train).extend(ix)
    return train, val, "available"


#  模型构建与训练
def load_tfjs_weights(model_json_path: str):
    spec = json.loads(Path(model_json_path).read_text(encoding="utf-8"))
    if spec.get("format") != "graph-model": raise ValueError("不是 tfjs graph-model")
    base = Path(model_json_path).parent
    weights = {}
    for group in spec["weightsManifest"]:
        buf = b"".join((base / p).read_bytes() for p in group["paths"])
        off = 0
        for w in group["weights"]:
            if w.get("quantization"): raise NotImplementedError("量化权重未实现")
            dt = np.dtype(w["dtype"]).newbyteorder("<")
            count = int(np.prod(w["shape"], dtype=np.int64)) if w["shape"] else 1
            weights[w["name"]] = np.frombuffer(buf, dtype=dt, count=count, offset=off).reshape(w["shape"]).copy()
            off += count * dt.itemsize
        if off != len(buf): raise ValueError(f"权重分片大小不匹配")
    return {n["name"]: n for n in spec["modelTopology"]["node"]}, weights


def _enable_gpu_memory_growth(tf) -> None:
    """TF 默认启动就占满整张卡；改成按需分配，这样 nvidia-smi 里看到的才是真实用量。"""
    try:
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError:
        pass  # GPU 已经初始化过，来不及设置


class TfjsMobileNetExtractor:
    def __init__(self, model_json_path: str):
        import tensorflow as tf
        _enable_gpu_memory_growth(tf)
        self.tf = tf
        self.nodes, weights = load_tfjs_weights(model_json_path)
        self.consts = {k: tf.constant(v) for k, v in weights.items()}
        # 急切模式下 _forward 里的 cache 字典会把每一层的输出都留到函数返回，
        # 显存 ≈ 所有层激活之和 × batch。tf.function 图执行会在用完后立刻回收缓冲区。
        self._infer = tf.function(
            self._infer_impl,
            input_signature=[tf.TensorSpec([None, IMAGE_SIZE, IMAGE_SIZE, 3], tf.uint8)],
        )

    @staticmethod
    def _ints(node, key):
        return [int(i) for i in node["attr"][key]["list"]["i"]]

    @staticmethod
    def _str(node, key):
        return base64.b64decode(node["attr"][key]["s"]).decode()

    def _forward(self, x):
        tf = self.tf
        cache = {"images": x}

        def ev(name):
            if name in cache: return cache[name]
            node = self.nodes[name]
            op = node["op"]
            if op == "Const":
                out = self.consts[name]
            else:
                a = [ev(i) for i in node.get("input", [])]
                if op == "Mul":
                    out = a[0] * a[1]
                elif op == "Sub":
                    out = a[0] - a[1]
                elif op == "Add":
                    out = a[0] + a[1]
                elif op == "BiasAdd":
                    out = tf.nn.bias_add(a[0], a[1])
                elif op == "Relu6":
                    out = tf.nn.relu6(a[0])
                elif op == "Identity":
                    out = a[0]
                elif op == "Squeeze":
                    out = tf.squeeze(a[0], axis=self._ints(node, "squeeze_dims"))
                elif op == "_FusedConv2D":
                    st = self._ints(node, "strides")
                    out = tf.nn.conv2d(a[0], a[1], strides=st, padding=self._str(node, "padding"))
                    fused = [base64.b64decode(s).decode() for s in node["attr"]["fused_ops"]["list"]["s"]]
                    extra = a[2:]
                    for f in fused:
                        if f == "BiasAdd":
                            out = tf.nn.bias_add(out, extra.pop(0))
                        elif f == "Relu6":
                            out = tf.nn.relu6(out)
                        elif f == "Relu":
                            out = tf.nn.relu(out)
                        else:
                            raise NotImplementedError(f"未实现的融合算子 {f}")
                elif op == "DepthwiseConv2dNative":
                    out = tf.nn.depthwise_conv2d(a[0], a[1], strides=self._ints(node, "strides"),
                                                 padding=self._str(node, "padding"))
                elif op == "AvgPool":
                    out = tf.nn.avg_pool2d(a[0], ksize=self._ints(node, "ksize"), strides=self._ints(node, "strides"),
                                           padding=self._str(node, "padding"))
                else:
                    raise NotImplementedError(f"未实现的算子 {op} ({name})")
            cache[name] = out
            return out

        return ev(MOBILENET_OUTPUT_NODE)

    def _infer_impl(self, images_uint8):
        tf = self.tf
        x = tf.cast(images_uint8, tf.float32) / 255.0 * 2.0 - 1.0
        y = self._forward(x)
        return tf.reshape(y, [tf.shape(y)[0], -1])

    def __call__(self, images_uint8: np.ndarray) -> np.ndarray:
        return self._infer(images_uint8).numpy().astype(np.float32)


class KerasMobileNetExtractor:
    def __init__(self, weights: Optional[str] = "imagenet"):
        import tensorflow as tf
        _enable_gpu_memory_growth(tf)
        self.tf = tf
        self.model = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), alpha=0.5,
                                                       include_top=False, weights=weights, pooling="avg")
        # 用 tf.function 包一层，批大小设为 None（动态），这样最后一个不满批不会触发重新 trace
        self._infer = tf.function(
            lambda x: self.model(x, training=False),
            input_signature=[tf.TensorSpec([None, IMAGE_SIZE, IMAGE_SIZE, 3], tf.float32)],
        )

    def __call__(self, images_uint8: np.ndarray) -> np.ndarray:
        x = images_uint8.astype("float32") / 255.0 * 2.0 - 1.0
        return self._infer(x).numpy().astype(np.float32)


def make_extractor(model_json: Optional[str]):
    if model_json and Path(model_json).is_file():
        try:
            shards = json.loads(Path(model_json).read_text(encoding="utf-8"))["weightsManifest"][0]["paths"]
            if all((Path(model_json).parent / p).is_file() for p in shards):
                print(f"[特征提取器] 使用 tfjs 图: {model_json}")
                return TfjsMobileNetExtractor(model_json)
        except Exception:
            pass
        print(f"[警告] 找不到权重分片，改用 Keras MobileNetV2(alpha=0.5, imagenet)")
    else:
        print("[警告] 未提供 model.json，改用 Keras MobileNetV2(alpha=0.5, imagenet)。")
    return KerasMobileNetExtractor()


def extract_embeddings(extractor, images: list, batch_size: int = EXTRACT_BATCH_SIZE,
                       log=print, max_workers: int = DECODE_WORKERS) -> np.ndarray:
    """提取嵌入。

    这一步真正的瓶颈往往不是 GPU，而是 PIL 图片解码/缩放（CPU）：原来的写法是
    "解码一个 batch → GPU 推理 → 解码下一个 batch"，GPU 和 CPU 互相干等。
    这里做两点优化：
    1. 一个 batch 内的图片用线程池并行解码（PIL 的 resize/decode 会释放 GIL，多核能吃满）；
    2. 用一个后台线程提前解码下一个 batch，和当前 batch 的 GPU 前向计算重叠执行。
    """
    n = len(images)
    if n == 0:
        return np.zeros((0, EMBEDDING_SIZE), dtype=np.float32)
    batches = [images[i:i + batch_size] for i in range(0, n, batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as decode_pool, \
         ThreadPoolExecutor(max_workers=1) as prefetcher:

        def decode_batch(batch):
            return np.stack(list(decode_pool.map(decode_for_model, batch)))

        out, done = [], 0
        pending = prefetcher.submit(decode_batch, batches[0])
        for i, batch in enumerate(batches):
            decoded = pending.result()
            if i + 1 < len(batches):
                pending = prefetcher.submit(decode_batch, batches[i + 1])
            out.append(extractor(decoded))
            done += len(batch)
            log(f"[提取嵌入] {done}/{n}")

    emb = np.concatenate(out, axis=0)
    if emb.ndim != 2: raise RuntimeError("嵌入形状异常")
    return emb


def build_head(embedding_size: int, num_classes: int, hidden_units: int, learning_rate: float):
    import tensorflow as tf
    K = tf.keras

    def var_scaling(fan_in):
        return K.initializers.TruncatedNormal(mean=0.0, stddev=math.sqrt(1.0 / max(1, fan_in)))

    model = K.Sequential([
        K.layers.Input(shape=(embedding_size,)),
        K.layers.Dense(hidden_units, activation="relu", kernel_initializer=var_scaling(embedding_size),
                       bias_initializer="zeros"),
        K.layers.Dropout(DROPOUT_RATE),
        K.layers.Dense(num_classes, activation="softmax", kernel_initializer=var_scaling(hidden_units),
                       bias_initializer="zeros"),
    ])
    compile_kwargs = dict(
        optimizer=K.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    try:
        # steps_per_execution：一次 Python/图调度里连续跑多个 step。
        # 这个分类头很小（100 个隐藏单元），单步计算量远小于 Python↔TF 的调度开销，
        # epochs 又可以到 200，调度开销会被放大很多倍，这个参数能明显缩短训练分类头的耗时。
        model.compile(**compile_kwargs, steps_per_execution=STEPS_PER_EXECUTION)
    except TypeError:
        model.compile(**compile_kwargs)  # 旧版本 TF 不支持该参数时回退
    return model


def collect_labels(examples) -> list:
    labels, seen = [], set()
    for e in examples:
        lab = e.label.strip()
        if not lab: raise ValueError("Every training image must have a non-empty label.")
        if lab not in seen:
            labels.append(lab)
            seen.add(lab)
    if len(labels) < 2: raise ValueError("Training requires at least two different classes.")
    return labels


def _safe_div(a, b): return a / b if b > 0 else 0


def validation_report(labels, actual, predicted) -> dict:
    k = len(labels)
    cm = [[0] * k for _ in range(k)]
    for a, p in zip(actual, predicted): cm[a][p] += 1
    support = [sum(row) for row in cm]
    predicted_n = [sum(cm[r][c] for r in range(k)) for c in range(k)]
    per_class = []
    for i, name in enumerate(labels):
        precision = _safe_div(cm[i][i], predicted_n[i])
        recall = _safe_div(cm[i][i], support[i])
        per_class.append(dict(label=name, support=support[i], precision=precision, recall=recall,
                              f1=_safe_div(2 * precision * recall, precision + recall)))
    correct = sum(cm[i][i] for i in range(k))
    return dict(exampleCount=len(actual), support=support, confusionMatrix=cm, perClass=per_class,
                accuracy=correct / len(actual), macroF1=sum(p["f1"] for p in per_class) / k,
                balancedAccuracy=sum(p["recall"] for p in per_class) / k)


def _register_cjk_font() -> Optional[str]:
    """给 matplotlib 找一个能显示中文的字体，避免图里中文变成方框。找不到就跳过（返回 None）。"""
    import matplotlib.font_manager as fm
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            return fm.FontProperties(fname=path).get_name()
    return None


def plot_training_curves(history: dict, out_path: str) -> None:
    """把 fit() 返回的逐轮 loss/accuracy 画成训练曲线图（左：loss，右：accuracy）并保存为 PNG。"""
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境（服务器/命令行）下也能画图存文件
    import matplotlib.pyplot as plt

    font_name = _register_cjk_font()
    if font_name:
        plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False  # 中文字体下负号别显示成方块

    loss = history.get("loss", [])
    if not loss:
        return
    epochs = range(1, len(loss) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_loss.plot(epochs, loss, label="训练 loss")
    if "val_loss" in history:
        ax_loss.plot(epochs, history["val_loss"], label="验证 loss")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.set_title("损失曲线"); ax_loss.legend(); ax_loss.grid(alpha=0.3)

    if "accuracy" in history:
        ax_acc.plot(epochs, history["accuracy"], label="训练准确率")
        if "val_accuracy" in history:
            ax_acc.plot(epochs, history["val_accuracy"], label="验证准确率")
        ax_acc.set_ylim(0, 1.02)
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("准确率曲线"); ax_acc.legend(); ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# 主分类器类
class OrangeClassifier:
    validation_fraction = VALIDATION_FRACTION

    def __init__(self, model_json_path: Optional[str] = None):
        self.extractor = make_extractor(model_json_path)
        self.model = None
        self.labels = None
        self.training_result = None

    def _load_project(self, source: str, source_type: str = "auto") -> Project:
        if source_type == "folder" or (source_type == "auto" and Path(source).is_dir()):
            return load_project_folder(source)
        elif source_type == "json" or (source_type == "auto" and Path(source).is_file()):
            return load_project_json(source)
        else:
            raise ValueError("无法确定数据源类型，请指定 folder 或 json")

    def train(self, source: str, source_type: str = "auto",
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None, extract_batch_size: int = EXTRACT_BATCH_SIZE,
              learning_rate: Optional[float] = None, hidden_units: int = HIDDEN_UNITS,
              force: bool = False, log: Callable = print) -> Dict[str, Any]:
        """
        训练模型。
        :param source: 文件夹路径或 JSON 文件路径
        :param source_type: 'folder', 'json' 或 'auto'
        :param epochs: 训练轮次
        :param batch_size: 批次大小
        :param learning_rate: 学习率
        :param hidden_units: 隐藏层单元数
        :param force: 是否忽略数据健康检查错误
        :param log: 日志打印函数
        """
        project = self._load_project(source, source_type)

        # 更新训练参数
        if epochs is not None: project.training["epochs"] = epochs
        if batch_size is not None: project.training["batchSize"] = batch_size
        if learning_rate is not None: project.training["learningRate"] = learning_rate

        # 打印基本信息
        for c in project.classes:
            log(f"类别 {c.name}: {len(c.samples)} 张")

        # 数据健康检查
        health_issues = data_health_report(project)
        for i in health_issues:
            log(f"[{'错误' if i['severity'] == 'error' else '提醒'}] {i['message']}")

        # 检查阻塞条件
        blockers = []
        if not valid_class_names([c.name for c in project.classes]):
            blockers.append("比赛项目必须使用固定的“橙子”和“非橙子”两个类别")
        for c in project.classes:
            if len(c.samples) < MIN_SAMPLES_PER_CLASS:
                blockers.append(f"为“{c.name}”再添加 {MIN_SAMPLES_PER_CLASS - len(c.samples)} 张图片")
        ep = project.training["epochs"]
        if not (isinstance(ep, int) and EPOCH_RANGE[0] <= ep <= EPOCH_RANGE[1]):
            blockers.append(f"训练轮次需为 {EPOCH_RANGE[0]}–{EPOCH_RANGE[1]} 的整数")
        blockers += [i["message"] for i in health_issues if i["severity"] == "error"]

        if blockers and not force:
            raise RuntimeError("无法开始训练：\n  - " + "\n  - ".join(dict.fromkeys(blockers)))

        # 构建示例
        examples = build_examples(project)
        t = project.training

        # 执行训练
        labels = collect_labels(examples)
        label_index = {l: i for i, l in enumerate(labels)}
        class_idx = [label_index[e.label.strip()] for e in examples]

        validation_split = self.validation_fraction if len(examples) >= VALIDATION_MIN_TOTAL else 0.0

        t_extract_start = time.perf_counter()
        log("[阶段] 提取嵌入（MobileNetV2 alpha=0.5，冻结）")
        emb = extract_embeddings(self.extractor, [e.image for e in examples], batch_size=extract_batch_size, log=log)
        dim = emb.shape[1]
        t_extract_end = time.perf_counter()

        tr, va, status = group_stratified_split(examples, class_idx, len(labels), validation_split)
        log(f"[验证集] 状态={status} 训练={len(tr)} 验证={len(va)}")

        y_all = np.eye(len(labels), dtype=np.float32)[np.asarray(class_idx)]
        x_tr, y_tr = emb[tr], y_all[tr]
        x_va, y_va = (emb[va], y_all[va]) if va else (None, None)

        model = build_head(dim, len(labels), hidden_units, t["learningRate"])
        log("[阶段] 训练分类头")

        fit = model.fit(x_tr, y_tr, epochs=t["epochs"], batch_size=min(t["batchSize"], len(tr)), shuffle=True,
                        validation_data=(x_va, y_va) if va else None, verbose=2)
        t_fit_end = time.perf_counter()

        history = {k: [float(v) for v in vs if np.isfinite(v)] for k, vs in fit.history.items()}

        validation = None
        if va:
            pred = np.argmax(model.predict(x_va, verbose=0), axis=1).tolist()
            validation = validation_report(labels, [class_idx[i] for i in va], pred)

        duration_ms = int((t_fit_end - t_extract_start) * 1000)

        self.model = model
        self.labels = labels
        self.training_result = dict(
            model=model, labels=labels, embeddingSize=dim, exampleCount=len(examples),
            epochsCompleted=t["epochs"], durationMs=duration_ms, history=history,
            validationStatus=status, validation=validation,
            training={
                "epochs": t["epochs"],
                "batchSize": t["batchSize"],
                "learningRate": t["learningRate"]
            }
        )

        h = history
        log(f"\n训练完成：{t['epochs']} 轮，标签顺序={labels}")
        log(f"  训练准确率 {h['accuracy'][-1]:.4f}" + (
            f"  验证准确率 {h['val_accuracy'][-1]:.4f}" if "val_accuracy" in h else ""))
        if validation:
            log(f"  验证集 {validation['exampleCount']} 张  accuracy={validation['accuracy']:.4f}")
        log(f"  耗时：提取嵌入 {t_extract_end - t_extract_start:.1f}s ｜ 训练分类头 {t_fit_end - t_extract_end:.1f}s "
            f"｜ 总计 {duration_ms / 1000:.1f}s")

        return self.training_result

    def predict(self, image_path: str) -> List[Dict[str, float]]:
        """
        预测单张图片。
        :return: 排序后的概率列表 [{'label': '...', 'probability': 0.xx}, ...]
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 train()")

        image_bytes = prepare_upload_jpeg(Path(image_path))
        emb = extract_embeddings(self.extractor, [image_bytes], log=lambda *_: None)
        prob = self.model.predict(emb, verbose=0)[0]

        preds = sorted(({"label": l, "probability": float(p)} for l, p in zip(self.labels, prob)),
                       key=lambda d: -d["probability"])
        return preds

    @staticmethod
    def decide(preds: List[Dict[str, float]],
               confidence: float = CONFIDENCE_THRESHOLD,
               margin: float = MARGIN_THRESHOLD) -> Dict[str, Any]:
        """
        根据预测结果做出决策。
        :return: {'label': ..., 'probability': ..., 'uncertain': bool, 'reason': ...}
        """
        if not preds: return None
        top = preds[0]
        gap = top["probability"] - (preds[1]["probability"] if len(preds) > 1 else 0)
        reason = "low-confidence" if top["probability"] < confidence else ("small-margin" if gap < margin else None)
        return dict(label=top["label"], probability=top["probability"], margin=gap,
                    uncertain=reason is not None, reason=reason)

    def save(self, out_dir: str, zip_output: bool = True):
        def _save_tfjs_layers_model(model, w1, b1, w2, b2, out: Path):
            """导出 TFJS Layers Model 格式的 model.json + weights.bin"""

            embedding_size = w1.shape[0]  # 应为 1280
            hidden_units = w1.shape[1]  # 应为 100
            num_classes = w2.shape[1]  # 应为 2

            #  model.json 结构（严格匹配目标格式）
            model_json = {
                "modelTopology": {
                    "class_name": "Sequential",
                    "config": {
                        "name": "sequential_1",
                        "layers": [
                            {
                                "class_name": "Dense",
                                "config": {
                                    "units": hidden_units,
                                    "activation": "relu",
                                    "use_bias": True,
                                    "kernel_initializer": {
                                        "class_name": "VarianceScaling",
                                        "config": {
                                            "scale": 1,
                                            "mode": "fan_in",
                                            "distribution": "normal",
                                            "seed": None
                                        }
                                    },
                                    "bias_initializer": {
                                        "class_name": "Zeros",
                                        "config": {}
                                    },
                                    "kernel_regularizer": None,
                                    "bias_regularizer": None,
                                    "activity_regularizer": None,
                                    "kernel_constraint": None,
                                    "bias_constraint": None,
                                    "name": "dense_Dense1",
                                    "trainable": True,
                                    "batch_input_shape": [None, embedding_size],
                                    "dtype": "float32"
                                }
                            },
                            {
                                "class_name": "Dropout",
                                "config": {
                                    "rate": DROPOUT_RATE,
                                    "noise_shape": None,
                                    "seed": None,
                                    "name": "dropout_Dropout1",
                                    "trainable": True
                                }
                            },
                            {
                                "class_name": "Dense",
                                "config": {
                                    "units": num_classes,
                                    "activation": "softmax",
                                    "use_bias": True,
                                    "kernel_initializer": {
                                        "class_name": "VarianceScaling",
                                        "config": {
                                            "scale": 1,
                                            "mode": "fan_in",
                                            "distribution": "normal",
                                            "seed": None
                                        }
                                    },
                                    "bias_initializer": {
                                        "class_name": "Zeros",
                                        "config": {}
                                    },
                                    "kernel_regularizer": None,
                                    "bias_regularizer": None,
                                    "activity_regularizer": None,
                                    "kernel_constraint": None,
                                    "bias_constraint": None,
                                    "name": "dense_Dense2",
                                    "trainable": True
                                }
                            }
                        ]
                    },
                    "keras_version": "tfjs-layers 4.22.0",
                    "backend": "tensor_flow.js"
                },
                "weightsManifest": [
                    {
                        "paths": ["weights.bin"],
                        "weights": [
                            {
                                "name": "dense_Dense1/kernel",
                                "shape": [embedding_size, hidden_units],
                                "dtype": "float32"
                            },
                            {
                                "name": "dense_Dense1/bias",
                                "shape": [hidden_units],
                                "dtype": "float32"
                            },
                            {
                                "name": "dense_Dense2/kernel",
                                "shape": [hidden_units, num_classes],
                                "dtype": "float32"
                            },
                            {
                                "name": "dense_Dense2/bias",
                                "shape": [num_classes],
                                "dtype": "float32"
                            }
                        ]
                    }
                ],
                "format": "layers-model",
                "generatedBy": "TensorFlow.js tfjs-layers v4.22.0",
                "convertedBy": None
            }

            (out / "model.json").write_text(
                json.dumps(model_json, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            #  写入 weights.bin（按顺序拼接 float32 权重）
            with open(out / "weights.bin", "wb") as f:
                for w in [w1, b1, w2, b2]:
                    f.write(np.asarray(w, dtype="<f4").tobytes())


        """保存模型和元数据（Teachable Machine 兼容格式）。"""
        if self.training_result is None:
            raise RuntimeError("没有可保存的训练结果")

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        model = self.training_result["model"]

        # 保存 Keras 原始模型（可选备份）
        model.save(out / "head.keras")
        w1, b1, w2, b2 = [w.numpy() if hasattr(w, "numpy") else np.asarray(w) for w in model.weights]
        np.savez(out / "head_weights.npz",
                 dense_kernel=w1, dense_bias=b1,
                 dense_1_kernel=w2, dense_1_bias=b2)

        # 生成 metadata.json（严格匹配 tm-object-classifier 格式）
        LABEL_COLORS = ["#3157D5", "#F47A5A", "#88D498", "#F5A623", "#D0021B"]
        labels_meta = []
        for i, name in enumerate(self.training_result["labels"]):
            labels_meta.append({
                "id": f"model-label-{i + 1}",
                "name": name,
                "color": LABEL_COLORS[i % len(LABEL_COLORS)]
            })

        t = self.training_result.get("training", None)  # 见下方说明
        meta = {
            "format": "tm-object-classifier",
            "formatVersion": 1,
            "name": "橙子识别项目",
            "createdAt": datetime.now(timezone.utc)
                         .isoformat(timespec="milliseconds")
                         .replace("+00:00", "Z"),
            "imageSize": IMAGE_SIZE,
            "labels": labels_meta,
            "training": {
                "epochs": t["epochs"],
                "batchSize": t["batchSize"],
                "learningRate": t["learningRate"]
            },
            "prediction": {
                "confidenceThreshold": 0.6,
                "marginThreshold": 0.12
            },
            "featureExtractor": "MobileNet v2 alpha 0.5 embedding"
        }
        (out / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 生成 model.json（TFJS Layers Model 格式）+ weights.bin
        _save_tfjs_layers_model(model, w1, b1, w2, b2, out)

        # 生成训练曲线图（loss / accuracy 随 epoch 变化），随模型一起导出
        try:
            history = self.training_result.get("history") or {}
            if history.get("loss"):
                plot_training_curves(history, str(out / "training_curve.png"))
                print(f"训练曲线已保存: {out / 'training_curve.png'}")
        except Exception as e:
            print(f"[警告] 训练曲线图生成失败：{e}")

        print(f"已保存到 {out_dir}/")

        try:
            os.remove(f"{out}/head.keras")
        except: pass
        try:
            os.remove(f"{out}/head_weights.npz")
        except: pass
        try:
            os.remove(f"{out}/weights.npz")
        except: pass
        if zip_output:
            zip_path = shutil.make_archive("橙子识别项目.识物模型", "zip", out)
            size_mb = os.path.getsize(zip_path) / 1024 / 1024
            print(f"已打包: {zip_path} ({size_mb:.1f} MB)")

