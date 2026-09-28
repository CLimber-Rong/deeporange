from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any

import numpy as np
from PIL import Image, ImageOps

from .model_export import ModelExport

# 配置常量
CLASS_NAMES = ["橙子", "非橙子"]
IMAGE_SIZE = 224
DEFAULT_TRAINING = dict(epochs=20, batchSize=16, learningRate=0.001)
HIDDEN_UNITS = 100
DROPOUT_RATE = 0.2
VALIDATION_SEED = 123
VALIDATION_FRACTION = 0.2
VALIDATION_MIN_TOTAL = 20
CONFIDENCE_THRESHOLD = 0.6
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
def _letterbox_on_white(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """白底 224×224 画布 + 整张图等比缩放后居中贴入（不裁剪，保留完整图像内容）。"""
    img = ImageOps.exif_transpose(img).convert("RGBA")
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    left, top = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (left, top), resized)
    return canvas


def prepare_upload_jpeg(path: Path, quality: int = 90) -> bytes:
    """模拟页面上传图片处理。"""
    canvas = _letterbox_on_white(Image.open(path))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def decode_for_model(data: bytes) -> np.ndarray:
    """训练/预测时送进 MobileNet 前的处理 → uint8 (224,224,3)。"""
    return np.asarray(_letterbox_on_white(Image.open(io.BytesIO(data))), dtype=np.uint8)


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


def build_head(embedding_size: int, num_classes: int, hidden_units: int, learning_rate: float,
               dropout_rate: float = DROPOUT_RATE, l2_reg: float = 0.0):
    import tensorflow as tf
    K = tf.keras

    def var_scaling(fan_in):
        return K.initializers.TruncatedNormal(mean=0.0, stddev=math.sqrt(1.0 / max(1, fan_in)))

    reg = K.regularizers.l2(l2_reg) if l2_reg > 0 else None

    model = K.Sequential([
        K.layers.Input(shape=(embedding_size,)),
        K.layers.Dense(hidden_units, activation="relu", kernel_initializer=var_scaling(embedding_size),
                       bias_initializer="zeros", kernel_regularizer=reg, name="dense_Dense1"),
        K.layers.Dropout(dropout_rate, name="dropout_Dropout1"),
        K.layers.Dense(num_classes, activation="softmax", kernel_initializer=var_scaling(hidden_units),
                       bias_initializer="zeros", kernel_regularizer=reg, name="dense_Dense2"),
    ])
    compile_kwargs = dict(
        optimizer=K.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    try:
        model.compile(**compile_kwargs, steps_per_execution=STEPS_PER_EXECUTION)
    except TypeError:
        model.compile(**compile_kwargs)
    return model


def load_head_tfjs_model(model_dir: str) -> Dict[str, Any]:
    d = Path(model_dir)
    spec = json.loads((d / "model.json").read_text(encoding="utf-8"))
    if spec.get("format") != "layers-model":
        raise ValueError(f"{d / 'model.json'} 不是 layers-model 格式")
    manifest = spec["weightsManifest"][0]["weights"]
    buf = (d / "weights.bin").read_bytes()
    off = 0
    arrays = {}
    for w in manifest:
        dt = np.dtype(w["dtype"]).newbyteorder("<")
        count = int(np.prod(w["shape"], dtype=np.int64)) if w["shape"] else 1
        arrays[w["name"]] = np.frombuffer(buf, dtype=dt, count=count, offset=off).reshape(w["shape"]).copy()
        off += count * dt.itemsize
    if off != len(buf):
        raise ValueError(f"{d / 'weights.bin'} 大小与 model.json 中的 weightsManifest 不匹配")
    layers = spec["modelTopology"]["config"]["layers"]
    dense1_name = layers[0]["config"]["name"]
    dropout_rate = layers[1]["config"]["rate"]
    dense2_name = layers[2]["config"]["name"]
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    labels = [l["name"] for l in meta["labels"]]
    return dict(
        w1=arrays[f"{dense1_name}/kernel"], b1=arrays[f"{dense1_name}/bias"],
        w2=arrays[f"{dense2_name}/kernel"], b2=arrays[f"{dense2_name}/bias"],
        dropout_rate=dropout_rate, labels=labels,
    )


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
    """计算验证集的整体与分类别评测指标：混淆矩阵、各类别 precision/recall/F1，
    以及宏平均（macro）和按样本数加权平均（weighted）的 precision/recall/F1。"""
    k = len(labels)
    cm = [[0] * k for _ in range(k)]
    for a, p in zip(actual, predicted): cm[a][p] += 1
    support = [sum(row) for row in cm]
    predicted_n = [sum(cm[r][c] for r in range(k)) for c in range(k)]
    total = sum(support)
    per_class = []
    for i, name in enumerate(labels):
        precision = _safe_div(cm[i][i], predicted_n[i])
        recall = _safe_div(cm[i][i], support[i])
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class.append(dict(label=name, support=support[i], precision=precision, recall=recall, f1=f1))
    correct = sum(cm[i][i] for i in range(k))
    macro_precision = sum(p["precision"] for p in per_class) / k
    macro_recall = sum(p["recall"] for p in per_class) / k
    macro_f1 = sum(p["f1"] for p in per_class) / k
    weighted_precision = _safe_div(sum(p["precision"] * p["support"] for p in per_class), total)
    weighted_recall = _safe_div(sum(p["recall"] * p["support"] for p in per_class), total)
    weighted_f1 = _safe_div(sum(p["f1"] * p["support"] for p in per_class), total)
    return dict(
        exampleCount=len(actual), support=support, confusionMatrix=cm, perClass=per_class,
        accuracy=_safe_div(correct, total),
        macroPrecision=macro_precision, macroRecall=macro_recall, macroF1=macro_f1,
        weightedPrecision=weighted_precision, weightedRecall=weighted_recall, weightedF1=weighted_f1,
        balancedAccuracy=macro_recall,  # 二分类下等价于宏平均召回率；保留旧字段名以兼容已有调用方
    )


def _register_cjk_font() -> Optional[str]:
    """给 matplotlib 找一个能显示中文的字体，避免图里中文变成方框。找不到就跳过（返回 None）。"""
    import matplotlib.font_manager as fm
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
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


def plot_validation_report(validation: dict, labels: list, out_path: str) -> None:
    """把 validation_report() 的结果（混淆矩阵 + 各类别 precision/recall/F1）画成图并保存为 PNG。
    不再单独产出评测结果的 JSON，图里已经包含全部数字。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font_name = _register_cjk_font()
    if font_name:
        plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False

    cm = np.asarray(validation["confusionMatrix"], dtype=np.int64)
    k = len(labels)

    fig, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=(11, 4.6))

    # 左：混淆矩阵热力图
    vmax = int(cm.max()) if cm.size else 1
    im = ax_cm.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    ax_cm.set_xticks(range(k)); ax_cm.set_xticklabels(labels)
    ax_cm.set_yticks(range(k)); ax_cm.set_yticklabels(labels)
    ax_cm.set_xlabel("预测类别"); ax_cm.set_ylabel("真实类别")
    ax_cm.set_title("混淆矩阵")
    for i in range(k):
        for j in range(k):
            color = "white" if cm[i, j] > vmax / 2 else "black"
            ax_cm.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    # 右：各类别 precision / recall / F1 分组柱状图
    x = np.arange(k)
    width = 0.25
    precision = [p["precision"] for p in validation["perClass"]]
    recall = [p["recall"] for p in validation["perClass"]]
    f1 = [p["f1"] for p in validation["perClass"]]
    ax_bar.bar(x - width, precision, width, label="precision")
    ax_bar.bar(x, recall, width, label="recall")
    ax_bar.bar(x + width, f1, width, label="F1")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(labels)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title("各类别评测指标"); ax_bar.legend(); ax_bar.grid(axis="y", alpha=0.3)
    for xi, vals in zip(x, zip(precision, recall, f1)):
        for off, v in zip((-width, 0, width), vals):
            ax_bar.text(xi + off, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"验证集 {validation['exampleCount']} 张 ｜ accuracy={validation['accuracy']:.4f} ｜ "
        f"macroF1={validation['macroF1']:.4f} ｜ macro召回率={validation['macroRecall']:.4f} ｜ "
        f"加权F1={validation['weightedF1']:.4f}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
              dropout_rate: float = DROPOUT_RATE, l2_reg: float = 0.0,
              force: bool = False, log: Callable = print,
              validation_source: Optional[str] = None,
              validation_source_type: str = "auto",
              validation_split: Optional[float] = None,
              early_stop_patience: int = 5,
              reduce_lr_factor: float = 0.5, reduce_lr_patience: int = 2,
              reduce_lr_min: float = 1e-6) -> Dict[str, Any]:
        project = self._load_project(source, source_type)
        t = self._apply_training_overrides(project, epochs, batch_size, learning_rate)

        def build_model(dim, num_labels):
            model = build_head(dim, num_labels, hidden_units, t["learningRate"],
                               dropout_rate=dropout_rate, l2_reg=l2_reg)
            required_shape = (EMBEDDING_SIZE, hidden_units)
            actual_shape = tuple(model.layers[0].kernel.shape)
            if actual_shape != required_shape:
                raise RuntimeError(
                    f"第一层（dense_Dense1）权重形状为 {actual_shape}，"
                    f"但要求必须是 {required_shape}"
                    f"（嵌入维度实际为 {dim}，需为 {EMBEDDING_SIZE}；"
                    f"隐藏单元数实际为 {actual_shape[1]}，需为 {hidden_units}）。"
                )
            log(f"[校验] 第一层权重形状 {actual_shape} 符合要求 {required_shape}")
            return model

        return self._run_training(project, build_model, extract_batch_size, force, log,
                                  validation_source, validation_source_type, validation_split,
                                  early_stop_patience, reduce_lr_factor, reduce_lr_patience, reduce_lr_min)

    def load_pretrained(self, model_dir: str, learning_rate: float) -> None:
        d = load_head_tfjs_model(model_dir)
        embedding_size, hidden_units = d["w1"].shape
        model = build_head(embedding_size, len(d["labels"]), hidden_units, learning_rate,
                           dropout_rate=d["dropout_rate"])
        model.layers[0].set_weights([d["w1"], d["b1"]])
        model.layers[2].set_weights([d["w2"], d["b2"]])
        self.model = model
        self.labels = d["labels"]

    def finetune(self, model_dir: str, source: str, source_type: str = "auto",
                epochs: Optional[int] = None,
                batch_size: Optional[int] = None, extract_batch_size: int = EXTRACT_BATCH_SIZE,
                learning_rate: Optional[float] = None,
                force: bool = False, log: Callable = print,
                validation_source: Optional[str] = None,
                validation_source_type: str = "auto",
                validation_split: Optional[float] = None,
                early_stop_patience: int = 5,
                reduce_lr_factor: float = 0.5, reduce_lr_patience: int = 2,
                reduce_lr_min: float = 1e-6) -> Dict[str, Any]:
        project = self._load_project(source, source_type)
        effective_lr = learning_rate if learning_rate is not None else 1e-4
        t = self._apply_training_overrides(project, epochs, batch_size, effective_lr)

        def build_model(dim, num_labels):
            self.load_pretrained(model_dir, t["learningRate"])
            loaded_dim = int(self.model.layers[0].kernel.shape[0])
            if loaded_dim != dim:
                raise RuntimeError(f"嵌入维度不匹配：已保存模型为 {loaded_dim}，当前特征提取器输出为 {dim}")
            if len(self.labels) != num_labels:
                raise RuntimeError(f"类别数量不匹配：已保存模型为 {len(self.labels)}，当前数据为 {num_labels}")
            log(f"[微调] 已从 {model_dir} 加载模型，标签={self.labels}，学习率={t['learningRate']}")
            return self.model

        return self._run_training(project, build_model, extract_batch_size, force, log,
                                  validation_source, validation_source_type, validation_split,
                                  early_stop_patience, reduce_lr_factor, reduce_lr_patience, reduce_lr_min)

    def _apply_training_overrides(self, project: Project, epochs: Optional[int],
                                  batch_size: Optional[int], learning_rate: Optional[float]) -> dict:
        if epochs is not None: project.training["epochs"] = epochs
        if batch_size is not None: project.training["batchSize"] = batch_size
        if learning_rate is not None: project.training["learningRate"] = learning_rate
        return project.training

    def _run_training(self, project: Project, build_model: Callable, extract_batch_size: int,
                      force: bool, log: Callable,
                      validation_source: Optional[str], validation_source_type: str,
                      validation_split_override: Optional[float],
                      early_stop_patience: int, reduce_lr_factor: float,
                      reduce_lr_patience: int, reduce_lr_min: float) -> Dict[str, Any]:
        for c in project.classes:
            log(f"类别 {c.name}: {len(c.samples)} 张")

        health_issues = data_health_report(project)
        for i in health_issues:
            log(f"[{'错误' if i['severity'] == 'error' else '提醒'}] {i['message']}")

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

        examples = build_examples(project)
        t = project.training

        labels = collect_labels(examples)
        label_index = {l: i for i, l in enumerate(labels)}
        class_idx = [label_index[e.label.strip()] for e in examples]

        val_examples = val_class_idx = None
        if validation_source:
            val_project = self._load_project(validation_source, validation_source_type)
            if not valid_class_names([c.name for c in val_project.classes]):
                raise RuntimeError("验证集类别名称必须同样是“橙子”和“非橙子”")
            for c in val_project.classes:
                log(f"验证集类别 {c.name}: {len(c.samples)} 张")
            val_examples = build_examples(val_project)
            if not val_examples:
                raise RuntimeError("验证集为空，请检查 validation_source")
            unknown = sorted({e.label.strip() for e in val_examples} - set(label_index))
            if unknown:
                raise RuntimeError(f"验证集包含训练集里没有的类别：{unknown}")
            val_class_idx = [label_index[e.label.strip()] for e in val_examples]

        if val_examples is not None:
            validation_split = 0.0
        else:
            frac = validation_split_override if validation_split_override is not None else self.validation_fraction
            validation_split = frac if len(examples) >= VALIDATION_MIN_TOTAL else 0.0

        t_extract_start = time.perf_counter()
        log("[阶段] 提取嵌入（MobileNetV2 alpha=0.5，冻结）")
        y_all = np.eye(len(labels), dtype=np.float32)[np.asarray(class_idx)]

        if val_examples is not None:
            status = "external"
            x_tr = extract_embeddings(self.extractor, [e.image for e in examples],
                                      batch_size=extract_batch_size, log=log)
            y_tr = y_all
            dim = x_tr.shape[1]
            log(f"[验证集] 状态=external（独立数据集，不从训练集里划分）训练={len(examples)} 验证={len(val_examples)}")
            log("[阶段] 提取验证集嵌入（独立数据集）")
            x_va = extract_embeddings(self.extractor, [e.image for e in val_examples],
                                      batch_size=extract_batch_size, log=log)
            y_va = np.eye(len(labels), dtype=np.float32)[np.asarray(val_class_idx)]
            va_actual = val_class_idx
        else:
            emb = extract_embeddings(self.extractor, [e.image for e in examples],
                                     batch_size=extract_batch_size, log=log)
            dim = emb.shape[1]
            tr, va, status = group_stratified_split(examples, class_idx, len(labels), validation_split)
            log(f"[验证集] 状态={status} 训练={len(tr)} 验证={len(va)}")
            x_tr, y_tr = emb[tr], y_all[tr]
            x_va, y_va = (emb[va], y_all[va]) if va else (None, None)
            va_actual = [class_idx[i] for i in va] if va else None
        t_extract_end = time.perf_counter()
        has_val = x_va is not None

        model = build_model(dim, len(labels))

        log("[阶段] 训练分类头")

        callbacks = []
        ckpt_dir = ckpt_path = None
        if has_val:
            import tensorflow as tf
            ckpt_dir = tempfile.mkdtemp(prefix="orange_ckpt_")
            ckpt_path = os.path.join(ckpt_dir, "best.weights.h5")
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=ckpt_path, monitor="val_loss", mode="min",
                    save_best_only=True, save_weights_only=True,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", mode="min", patience=early_stop_patience, restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", mode="min", factor=reduce_lr_factor, patience=reduce_lr_patience,
                    min_lr=reduce_lr_min, verbose=1,
                ),
            ]

        fit = model.fit(x_tr, y_tr, epochs=t["epochs"], batch_size=min(t["batchSize"], len(x_tr)), shuffle=True,
                        validation_data=(x_va, y_va) if has_val else None, verbose=2, callbacks=callbacks)
        t_fit_end = time.perf_counter()

        if ckpt_path and os.path.exists(ckpt_path):
            model.load_weights(ckpt_path)
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        stopped_early = len(fit.history.get("loss", [])) < t["epochs"]
        if stopped_early:
            log(f"[早停] 验证 loss 连续 {early_stop_patience} 轮没有改善，训练在第 {len(fit.history['loss'])} 轮提前停止，"
                f"已恢复验证 loss 最低那一轮的权重")

        history = {k: [float(v) for v in vs if np.isfinite(v)] for k, vs in fit.history.items()}

        validation = None
        if has_val:
            pred = np.argmax(model.predict(x_va, verbose=0), axis=1).tolist()
            validation = validation_report(labels, va_actual, pred)

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
            v = validation
            log(f"  验证集（{v['exampleCount']} 张，{status}）：")
            log(f"    accuracy={v['accuracy']:.4f}  macroF1={v['macroF1']:.4f}  "
                f"macro召回率={v['macroRecall']:.4f}  macro精确率={v['macroPrecision']:.4f}")
            log(f"    加权F1={v['weightedF1']:.4f}  加权召回率={v['weightedRecall']:.4f}  "
                f"加权精确率={v['weightedPrecision']:.4f}")
            for pc in v["perClass"]:
                log(f"    · {pc['label']:<6} support={pc['support']:<5d} "
                    f"precision={pc['precision']:.4f}  recall={pc['recall']:.4f}  f1={pc['f1']:.4f}")
            log(f"    混淆矩阵（行=真实类别，列=预测类别，顺序={labels}）：")
            for name, row in zip(labels, v["confusionMatrix"]):
                log(f"      {name:<6}" + "  ".join(f"{x:5d}" for x in row))
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
            import tensorflow as tf

            def _serialize(obj):
                if obj is None:
                    return None
                d = tf.keras.utils.serialize_keras_object(obj)
                return {"class_name": d["class_name"], "config": d.get("config", {})}

            def _dense_layer_json(layer, batch_input_shape=None):
                cfg = {
                    "units": layer.units,
                    "activation": tf.keras.activations.serialize(layer.activation),
                    "use_bias": layer.use_bias,
                    "kernel_initializer": _serialize(layer.kernel_initializer),
                    "bias_initializer": _serialize(layer.bias_initializer),
                    "kernel_regularizer": _serialize(layer.kernel_regularizer),
                    "bias_regularizer": _serialize(layer.bias_regularizer),
                    "activity_regularizer": _serialize(layer.activity_regularizer),
                    "kernel_constraint": _serialize(layer.kernel_constraint),
                    "bias_constraint": _serialize(layer.bias_constraint),
                    "name": layer.name,
                    "trainable": layer.trainable,
                }
                if batch_input_shape is not None:
                    cfg["batch_input_shape"] = batch_input_shape
                    cfg["dtype"] = layer.dtype
                return {"class_name": "Dense", "config": cfg}

            def _dropout_layer_json(layer):
                return {
                    "class_name": "Dropout",
                    "config": {
                        "rate": layer.rate,
                        "noise_shape": layer.noise_shape,
                        "seed": layer.seed,
                        "name": layer.name,
                        "trainable": layer.trainable,
                    }
                }

            dense1, dropout1, dense2 = model.layers
            batch_input_shape = [None if d is None else int(d) for d in dense1.input_shape]

            model_json = {
                "modelTopology": {
                    "class_name": model.__class__.__name__,
                    "config": {
                        "name": model.name,
                        "layers": [
                            _dense_layer_json(dense1, batch_input_shape=batch_input_shape),
                            _dropout_layer_json(dropout1),
                            _dense_layer_json(dense2),
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
                                "name": f"{dense1.name}/kernel",
                                "shape": list(w1.shape),
                                "dtype": "float32"
                            },
                            {
                                "name": f"{dense1.name}/bias",
                                "shape": list(b1.shape),
                                "dtype": "float32"
                            },
                            {
                                "name": f"{dense2.name}/kernel",
                                "shape": list(w2.shape),
                                "dtype": "float32"
                            },
                            {
                                "name": f"{dense2.name}/bias",
                                "shape": list(b2.shape),
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

            with open(out / "weights.bin", "wb") as f:
                for w in [w1, b1, w2, b2]:
                    f.write(np.asarray(w, dtype="<f4").tobytes())


        """保存模型和元数据（Teachable Machine 兼容格式）。"""
        if self.training_result is None:
            raise RuntimeError("没有可保存的训练结果")

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        model = self.training_result["model"]

        model.save(out / "head.keras")
        w1, b1, w2, b2 = [w.numpy() if hasattr(w, "numpy") else np.asarray(w) for w in model.weights]
        np.savez(out / "head_weights.npz",
                 dense_kernel=w1, dense_bias=b1,
                 dense_1_kernel=w2, dense_1_bias=b2)

        LABEL_COLORS = ["#3157D5", "#F47A5A", "#88D498", "#F5A623", "#D0021B"]
        labels_meta = []
        for i, name in enumerate(self.training_result["labels"]):
            labels_meta.append({
                "id": f"model-label-{i + 1}",
                "name": name,
                "color": LABEL_COLORS[i % len(LABEL_COLORS)]
            })

        t = self.training_result.get("training", None)
        meta = {
            "format": "tm-object-classifier",
            "formatVersion": 1,
            "name": "橙子识别项目",
            "imageSize": 224,
            "createdAt": datetime.now(timezone.utc)
                         .isoformat(timespec="milliseconds")
                         .replace("+00:00", "Z"),
            "labels": labels_meta,
            "training": {
                "epochs": t["epochs"],
                "batchSize": t["batchSize"],
                "learningRate": t["learningRate"]
            },
            "prediction": {
                "confidenceThreshold": CONFIDENCE_THRESHOLD,
                "marginThreshold": MARGIN_THRESHOLD
            },
            "featureExtractor": "MobileNet v2 alpha 0.5 embedding",
            "imagePreprocessing": "letterbox"
        }
        (out / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        _save_tfjs_layers_model(model, w1, b1, w2, b2, out)

        export = ModelExport(out)
        artifacts = {}
        try:
            history = self.training_result.get("history") or {}
            if history.get("loss"):
                curve = export.save_plot("training_curve", lambda path: plot_training_curves(history, path))
                artifacts["training_curve"] = str(curve)
                print(f"📈 训练曲线已保存: {curve}")
        except Exception as e:
            print(f"[警告] 训练曲线图生成失败：{e}")

        try:
            validation = self.training_result.get("validation")
            if validation:
                report = export.save_plot(
                    "validation_report",
                    lambda path: plot_validation_report(validation, self.training_result["labels"], path))
                artifacts["validation_report"] = str(report)
                print(f"📊 验证集评测图已保存: {report}")
        except Exception as e:
            print(f"[警告] 验证集评测图生成失败：{e}")

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
            zip_path = export.package()
            artifacts["zip"] = str(zip_path)
            size_mb = os.path.getsize(zip_path) / 1024 / 1024
            print(f"📦 已打包: {zip_path} ({size_mb:.1f} MB)")

        return artifacts
