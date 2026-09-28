"""Train and compare every trainer/preset pair on the fixed independent set.

Run with resources/runtime/python.exe batch_compare.py [prepare|train|evaluate|all].
The cached embeddings only avoid repeating the frozen MobileNetV2 forward pass;
each classifier head is fitted by its own, unmodified trainer implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = HERE / "model"
RESULT = HERE / "result"
CACHE = HERE / ".batch_cache"
SELECTION_DIR = RESULT / "验证集筛选记录"
SELECTION_FILE = SELECTION_DIR / "选样清单.json"
BACKBONE = MODEL / "basemodels" / "model.json"
TRAINERS = (
    "original", "original_full_data", "optimized", "optimized_full_data",
    "optimized_2", "optimized_2_full_data",
)
PRESETS = {
    "flash": dict(epochs=10, batch_size=32, learning_rate=0.001, hidden_units=100),
    "std": dict(epochs=20, batch_size=16, learning_rate=0.001, hidden_units=100),
    "pro": dict(epochs=120, batch_size=8, learning_rate=0.0003, hidden_units=256),
}
CLASSES = ("橙子", "非橙子")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def pairs():
    for trainer in TRAINERS:
        for preset in PRESETS:
            yield trainer, preset, f"deeporange-v1-{trainer.replace('_', '-')}-{preset}"


def paths(root):
    return [p for c in CLASSES for p in sorted((root / c).iterdir())
            if p.is_file() and p.suffix.lower() in EXTS]


def snapshot(items):
    h = hashlib.sha256()
    for path in items:
        h.update(str(path.relative_to(ROOT)).encode("utf-8"))
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def state():
    train, val = paths(ROOT), paths(ROOT / "验证集")
    if [sum(p.parent.name == c for p in train) for c in CLASSES] != [1001, 1011]:
        raise RuntimeError("训练集数量与本次固定样本快照不符，请先核对数据")
    if [sum(p.parent.name == c for p in val) for c in CLASSES] != [100, 100]:
        raise RuntimeError("验证集数量与本次固定样本快照不符，请先核对数据")
    return train, val, snapshot(train), snapshot(val)


def cache_file(kind, digest):
    return CACHE / f"{kind}-{digest[:16]}.npy"


def audit(train, val):
    digest_train = {hashlib.sha256(p.read_bytes()).digest(): p for p in train}
    overlaps = [(str(digest_train[h]), str(p)) for p in val
                if (h := hashlib.sha256(p.read_bytes()).digest()) in digest_train]
    (CACHE / "exact_overlap.json").write_text(
        json.dumps(overlaps, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练/验证完全相同文件: {len(overlaps)} 组", flush=True)
    for pair in overlaps[:10]:
        print(pair, flush=True)


def audit_near_duplicates():
    """Find likely cross-split copies after resizing or JPEG recompression."""
    from PIL import Image, ImageOps
    train, val, _, _ = state()

    def descriptor(path):
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image = ImageOps.fit(image, (64, 64), method=Image.Resampling.BILINEAR)
            grey = np.asarray(image.convert("L").resize((9, 8)), dtype=np.int16)
            bits = grey[:, 1:] > grey[:, :-1]
            dhash = sum(int(v) << i for i, v in enumerate(bits.flat))
            thumb = np.asarray(image.resize((16, 16)), dtype=np.int16)
            return dhash, thumb

    index = {}
    train_desc = []
    for i, path in enumerate(train):
        code, thumb = descriptor(path)
        train_desc.append((code, thumb))
        for part in range(4):
            index.setdefault((part, (code >> (16 * part)) & 0xffff), []).append(i)
    matches = []
    for path in val:
        code, thumb = descriptor(path)
        candidates = set()
        for part in range(4):
            candidates.update(index.get((part, (code >> (16 * part)) & 0xffff), ()))
        for i in candidates:
            original_code, original_thumb = train_desc[i]
            distance = (code ^ original_code).bit_count()
            if distance > 3:
                continue
            pixel_diff = float(np.mean(np.abs(thumb - original_thumb)))
            if pixel_diff <= 12:
                matches.append(dict(train=str(train[i]), validation=str(path),
                                    dhash_distance=distance, mean_pixel_diff=pixel_diff))
    (CACHE / "near_overlap.json").write_text(
        json.dumps(matches, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练/验证疑似近重复: {len(matches)} 组", flush=True)
    for item in matches[:10]:
        print(item, flush=True)


def prepare():
    os.chdir(HERE)
    CACHE.mkdir(exist_ok=True)
    train, val, train_hash, val_hash = state()
    audit(train, val)
    manifest = dict(train_sha256=train_hash, validation_sha256=val_hash,
                    train_counts=[1001, 1011], validation_counts=[100, 100])
    (CACHE / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, module_name in (("center", "trainer_original"),
                             ("letterbox", "trainer_optimized_2")):
        dest = cache_file(f"train-{key}", train_hash)
        if dest.is_file():
            print(f"复用 {dest}", flush=True)
            continue
        mod = importlib.import_module(f"base.{module_name}")
        project = mod.load_project_folder(str(ROOT))
        examples = mod.build_examples(project)
        if len(examples) != len(train):
            raise RuntimeError("训练样本顺序或数量异常")
        extractor = mod.make_extractor(str(BACKBONE))
        emb = mod.extract_embeddings(extractor, [e.image for e in examples],
                                     batch_size=mod.EXTRACT_BATCH_SIZE,
                                     log=lambda msg: print(f"[{key}] {msg}", flush=True))
        np.save(dest, emb)
        print(f"已保存训练特征 {dest}: {emb.shape}", flush=True)

    import main
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    base = main.MobileNetExtractor().model
    for key in ("center", "letterbox"):
        dest = cache_file(f"validation-{key}", val_hash)
        if dest.is_file():
            print(f"复用 {dest}", flush=True)
            continue
        layout = "center_crop" if key == "center" else "letterbox"
        batches = []
        for i in range(0, len(val), 32):
            image = np.stack([main.load_image_array(str(p), (224, 224), layout)
                              for p in val[i:i + 32]])
            batches.append(base(preprocess_input(image), training=False).numpy())
            print(f"[{key}] 验证特征 {min(i + 32, len(val))}/{len(val)}", flush=True)
        emb = np.concatenate(batches).astype("float32")
        np.save(dest, emb)
        print(f"已保存验证特征 {dest}: {emb.shape}", flush=True)


def ensure_snapshot():
    train, val, train_hash, val_hash = state()
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    if (train_hash, val_hash) != (manifest["train_sha256"], manifest["validation_sha256"]):
        raise RuntimeError("数据集已变化，需重新运行 prepare")
    return train, val, train_hash, val_hash


def train_one(trainer, preset, name):
    os.chdir(HERE)
    _, _, train_hash, _ = ensure_snapshot()
    target = MODEL / f"{name}.zip"
    log_dir = RESULT / f"{name}-result"
    log_dir.mkdir(parents=True, exist_ok=True)
    info_path = log_dir / "训练记录.json"
    if target.is_file() and info_path.is_file():
        previous = json.loads(info_path.read_text(encoding="utf-8"))
        if (previous.get("train_sha256") != train_hash or
                previous.get("zip_sha256") != hashlib.sha256(target.read_bytes()).hexdigest()):
            raise RuntimeError(f"现有模型与当前训练集或记录不匹配，请先归档: {target}")
        print(f"已完成 {name}", flush=True)
        return
    base_name = "trainer_optimized_2" if trainer.startswith("optimized_2") else (
        "trainer_optimized" if trainer.startswith("optimized") else "trainer_original")
    module_name = f"trainer_{trainer}"
    mod = importlib.import_module(f"base.{module_name}")
    base_mod = importlib.import_module(f"base.{base_name}")
    mode = "letterbox" if trainer.startswith("optimized_2") else "center"
    emb = np.load(cache_file(f"train-{mode}", train_hash), mmap_mode="r")
    if emb.shape != (2012, 1280):
        raise RuntimeError(f"训练特征形状异常: {emb.shape}")
    base_mod.make_extractor = lambda _: None
    base_mod.extract_embeddings = lambda _extractor, images, **_kwargs: emb if len(images) == 2012 else (
        (_ for _ in ()).throw(RuntimeError("训练特征与样本数不符")))
    clf = mod.OrangeClassifier(model_json_path=str(BACKBONE))
    params = PRESETS[preset]
    kwargs = dict(source=str(ROOT), source_type="folder", extract_batch_size=64,
                  force=False, log=lambda msg: print(msg, flush=True), **params)
    if base_name == "trainer_optimized":
        kwargs["optimizer"] = "Adam"
    started = time.time()
    result = clf.train(**kwargs)
    work = CACHE / "exports" / name
    artifacts = clf.save(str(work), zip_output=True)
    shutil.move(artifacts["zip"], target)
    artifacts["zip"] = str(target)
    with zipfile.ZipFile(target) as z:
        if set(z.namelist()) != {"metadata.json", "model.json", "weights.bin"}:
            raise RuntimeError(f"ZIP 文件结构异常: {target}")
    info = dict(name=name, trainer=trainer, preset=preset, params=params,
                train_sha256=train_hash, started=started, ended=time.time(),
                duration_seconds=time.time() - started, examples=result["exampleCount"],
                validation_status=result["validationStatus"],
                epochs_run=len(result["history"].get("loss", [])),
                history=result["history"], validation=result["validation"],
                zip_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                training_artifacts=artifacts)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成 {name}: {target}", flush=True)


def evaluate_one(trainer, preset, name):
    os.chdir(HERE)
    _, val, _, val_hash = ensure_snapshot()
    target = MODEL / f"{name}.zip"
    if not target.is_file():
        raise FileNotFoundError(target)
    dest = RESULT / f"{name}-result"
    metric_file = dest / "指标.json"
    if metric_file.is_file():
        previous = json.loads(metric_file.read_text(encoding="utf-8"))
        training = json.loads((dest / "训练记录.json").read_text(encoding="utf-8"))
        if (previous.get("validation_sha256") != val_hash or
                training.get("zip_sha256") != hashlib.sha256(target.read_bytes()).hexdigest()):
            raise RuntimeError(f"现有结果与当前模型或验证集不匹配，请先归档: {dest}")
        print(f"已评估 {name}", flush=True)
        return
    import main
    mode = "letterbox" if trainer.startswith("optimized_2") else "center"
    emb = np.load(cache_file(f"validation-{mode}", val_hash), mmap_mode="r")
    with zipfile.ZipFile(target) as z:
        extract_dir = CACHE / "eval" / name
        extract_dir.mkdir(parents=True, exist_ok=True)
        for member in ("metadata.json", "model.json", "weights.bin"):
            (extract_dir / member).write_bytes(z.read(member))
    metadata = json.loads((extract_dir / "metadata.json").read_text(encoding="utf-8"))
    labels = [entry["name"] for entry in metadata["labels"]]
    if labels != list(CLASSES):
        raise RuntimeError(f"模型标签顺序错误: {name}: {labels}")
    if metadata.get("imagePreprocessing", "center_crop") != ("letterbox" if mode == "letterbox" else "center_crop"):
        raise RuntimeError(f"模型预处理模式错误: {name}")
    head, width = main._load_tfjs_head_only(str(extract_dir / "model.json"))
    if width != 1280:
        raise RuntimeError(f"分类头输入维度异常: {name}")
    prob = np.concatenate([head(np.asarray(emb[i:i + 256]), training=False).numpy()
                           for i in range(0, len(val), 256)])
    y_true = np.array([CLASSES.index(p.parent.name) for p in val], dtype=np.int64)
    metrics = main.compute_classification_metrics(y_true, prob, list(CLASSES))
    report = main.MobileNetTesterApp._build_cls_report(
        str(target), str(ROOT / "验证集"), list(CLASSES), y_true, metrics)
    app = object.__new__(main.MobileNetTesterApp)
    app.last_result = dict(type="classification", classes=list(CLASSES),
                           paths=[str(p) for p in val], y_true=y_true,
                           y_prob=prob, metrics=metrics, report=report)
    app._export_classification(str(dest))
    bat = dest / "额外_逐样本预测结果" / "导出预测错误图片.bat"
    subprocess.run(["cmd.exe", "/d", "/c", str(bat)], check=True, cwd=bat.parent,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    cm = metrics["cm"].tolist()
    for label, expected in (("橙子", cm[0][1]), ("非橙子", cm[1][0])):
        folder = bat.parent / f"预测错误的{label}"
        if len([p for p in folder.iterdir() if p.is_file()]) != expected:
            raise RuntimeError(f"错误图片导出数不符: {name} {label} {expected}")
    summary = dict(name=name, trainer=trainer, preset=preset, validation_sha256=val_hash,
                   accuracy=float(metrics["acc"]), balanced_accuracy=float(metrics["bacc"]),
                   macro_f1=float(metrics["f_macro"]), weighted_f1=float(metrics["f_weighted"]),
                   kappa=float(metrics["kappa"]), auc_macro=float(metrics["auc_macro"]),
                   auc_orange=float(metrics["per_auc"][0]), ap_orange=float(metrics["per_ap"][0]),
                   macro_ap=float(metrics["map_macro"]), confusion_matrix=cm,
                   orange_precision=float(metrics["per_p"][0]),
                   orange_recall=float(metrics["per_r"][0]),
                   orange_f1=float(metrics["per_f"][0]),
                   nonorange_recall=float(metrics["per_r"][1]),
                   false_negative=cm[0][1], false_positive=cm[1][0])
    metric_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已评估 {name}: balanced={summary['balanced_accuracy']:.4f} "
          f"FN={cm[0][1]} FP={cm[1][0]}", flush=True)


def verify_first_model():
    """Check that cached feature evaluation matches the GUI model path."""
    import main
    name = "deeporange-v1-original-flash"
    target = MODEL / f"{name}.zip"
    model = main.load_model_any(str(target))
    val = paths(ROOT / "验证集")[:8]
    gui = main.predict_paths(model, [str(p) for p in val], (224, 224), True,
                             batch_size=8, image_preprocessing="center_crop")
    head, _ = main._load_tfjs_head_only(
        str(CACHE / "eval" / name / "model.json"))
    emb = np.load(cache_file("validation-center", state()[3]), mmap_mode="r")
    cached = head(np.asarray(emb[:8]), training=False).numpy()
    delta = float(np.max(np.abs(gui - cached)))
    print(f"GUI 与缓存特征测试最大概率差: {delta:.8f}", flush=True)
    if delta > 1e-4:
        raise RuntimeError("缓存验证路径与 GUI 预测结果不一致")


def verify_backbone():
    """Compare the frozen features used at training and in the GUI tester."""
    import main
    from base import trainer_original
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    train, _, train_hash, _ = ensure_snapshot()
    original = np.load(cache_file("train-center", train_hash), mmap_mode="r")[:16]
    base = main.MobileNetExtractor().model
    image = np.stack([trainer_original.decode_for_model(
        trainer_original.prepare_upload_jpeg(p)) for p in train[:16]])
    reconstructed = base(preprocess_input(image), training=False).numpy()
    dot = np.sum(original * reconstructed, axis=1)
    norm = np.linalg.norm(original, axis=1) * np.linalg.norm(reconstructed, axis=1)
    cosine = dot / np.maximum(norm, 1e-12)
    print(f"训练/测试 MobileNetV2 特征余弦相似度: min={cosine.min():.4f} "
          f"mean={cosine.mean():.4f} max={cosine.max():.4f}", flush=True)


def prepare_matched_validation():
    """Extract holdout features with the exact frozen graph used by training."""
    _, val, _, val_hash = ensure_snapshot()
    for key, module_name in (("center", "trainer_original"),
                             ("letterbox", "trainer_optimized_2")):
        dest = cache_file(f"validation-matched-{key}", val_hash)
        if dest.is_file():
            print(f"复用 {dest}", flush=True)
            continue
        mod = importlib.import_module(f"base.{module_name}")
        project = mod.load_project_folder(str(ROOT / "验证集"))
        examples = mod.build_examples(project)
        if len(examples) != len(val):
            raise RuntimeError("验证样本顺序或数量异常")
        extractor = mod.make_extractor(str(BACKBONE))
        features = mod.extract_embeddings(extractor, [e.image for e in examples],
                                          batch_size=mod.EXTRACT_BATCH_SIZE,
                                          log=lambda msg: print(f"[{key}] {msg}", flush=True))
        np.save(dest, features)
        print(f"已保存训练基底一致的验证特征 {dest}: {features.shape}", flush=True)


def compare_matched_first_model():
    import main
    _, val, _, val_hash = ensure_snapshot()
    name = "deeporange-v1-original-flash"
    head, _ = main._load_tfjs_head_only(
        str(CACHE / "eval" / name / "model.json"))
    features = np.load(cache_file("validation-matched-center", val_hash), mmap_mode="r")
    probabilities = np.concatenate([
        head(np.asarray(features[i:i + 256]), training=False).numpy()
        for i in range(0, len(val), 256)])
    truth = np.array([CLASSES.index(p.parent.name) for p in val], dtype=np.int64)
    metrics = main.compute_classification_metrics(truth, probabilities, list(CLASSES))
    print(f"训练基底一致评估 {name}: balanced={metrics['bacc']:.4f} "
          f"FN={metrics['cm'][0, 1]} FP={metrics['cm'][1, 0]} ", flush=True)


def evaluate_matched():
    """Measure all heads with the training graph and training image preparation."""
    import main
    _, val, _, val_hash = ensure_snapshot()
    truth = np.array([CLASSES.index(p.parent.name) for p in val], dtype=np.int64)
    features = {
        mode: np.load(cache_file(f"validation-matched-{mode}", val_hash), mmap_mode="r")
        for mode in ("center", "letterbox")
    }
    for trainer, preset, name in pairs():
        output = RESULT / f"{name}-result" / "基底一致指标.json"
        extracted = CACHE / "eval" / name / "model.json"
        head, _ = main._load_tfjs_head_only(str(extracted))
        mode = "letterbox" if trainer.startswith("optimized_2") else "center"
        matrix = features[mode]
        probabilities = np.concatenate([
            head(np.asarray(matrix[i:i + 256]), training=False).numpy()
            for i in range(0, len(val), 256)])
        metrics = main.compute_classification_metrics(truth, probabilities, list(CLASSES))
        cm = metrics["cm"].tolist()
        summary = dict(name=name, validation_sha256=val_hash,
                       backbone="model/basemodels/model.json",
                       image_path="trainer prepare_upload_jpeg + decode_for_model",
                       accuracy=float(metrics["acc"]),
                       balanced_accuracy=float(metrics["bacc"]),
                       macro_f1=float(metrics["f_macro"]),
                       orange_precision=float(metrics["per_p"][0]),
                       orange_recall=float(metrics["per_r"][0]),
                       orange_f1=float(metrics["per_f"][0]),
                       orange_ap=float(metrics["per_ap"][0]),
                       orange_auc=float(metrics["per_auc"][0]),
                       false_negative=cm[0][1], false_positive=cm[1][0],
                       confusion_matrix=cm)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        export_dir = output.parent / "基底一致验证"
        report = main.MobileNetTesterApp._build_cls_report(
            str(MODEL / f"{name}.zip"), str(ROOT / "验证集"),
            list(CLASSES), truth, metrics)
        app = object.__new__(main.MobileNetTesterApp)
        app.last_result = dict(type="classification", classes=list(CLASSES),
                               paths=[str(p) for p in val], y_true=truth,
                               y_prob=probabilities, metrics=metrics, report=report)
        app._export_classification(str(export_dir))
        bat = export_dir / "额外_逐样本预测结果" / "导出预测错误图片.bat"
        subprocess.run(["cmd.exe", "/d", "/c", str(bat)], check=True,
                       cwd=bat.parent, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
        for label, expected in (("橙子", cm[0][1]), ("非橙子", cm[1][0])):
            actual = len([p for p in (bat.parent / f"预测错误的{label}").iterdir()
                          if p.is_file()])
            if actual != expected:
                raise RuntimeError(f"基底一致错误图片导出数不符: {name} {label}")
        print(f"{name}: bacc={summary['balanced_accuracy']:.4f} "
              f"FN={summary['false_negative']} FP={summary['false_positive']}", flush=True)


def write_report():
    rows = []
    matched = {}
    for trainer, preset, name in pairs():
        directory = RESULT / f"{name}-result"
        metrics = json.loads((directory / "指标.json").read_text(encoding="utf-8"))
        matched[name] = json.loads((directory / "基底一致指标.json").read_text(encoding="utf-8"))
        training = json.loads((directory / "训练记录.json").read_text(encoding="utf-8"))
        training["training_artifacts"]["zip"] = str(MODEL / f"{name}.zip")
        (directory / "训练记录.json").write_text(
            json.dumps(training, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append((metrics, training))
    train, val, train_hash, val_hash = state()
    manifest = dict(train_sha256=train_hash, validation_sha256=val_hash)
    train_bytes = {hashlib.sha256(path.read_bytes()).digest() for path in train}
    exact = [path.name for path in val
             if hashlib.sha256(path.read_bytes()).digest() in train_bytes]
    near_path = CACHE / "near_overlap.json"
    audit_path = SELECTION_DIR / "审查记录.json"
    if near_path.is_file():
        near = json.loads(near_path.read_text(encoding="utf-8"))
        SELECTION_DIR.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(dict(manifest=manifest, near=near),
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    elif audit_path.is_file():
        audit_record = json.loads(audit_path.read_text(encoding="utf-8"))
        near = audit_record["near"] if audit_record["manifest"] == manifest else None
    else:
        near = None
    for metrics, training in rows:
        name = metrics["name"]
        artifacts = training.get("training_artifacts", {})
        target_dir = RESULT / f"{name}-result" / "训练过程"
        for key, title in (("training_curve", "训练曲线.png"),
                           ("validation_report", "内部验证.png")):
            source = artifacts.get(key)
            if source and Path(source).is_file():
                target_dir.mkdir(exist_ok=True)
                shutil.copy2(source, target_dir / title)

    def percent(value):
        return f"{100 * value:.2f}%"

    def label(metrics):
        name = metrics["name"]
        return f"[{name}](本地训练/result/{name}-result/)"

    selection = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
    quotas = selection["quotas"]
    red_names = selection["red_round_fruit_categories"]
    red_count = sum(quotas[name] for name in red_names)
    if ({item["filename"] for item in selection["images"]} !=
            {path.name for path in val if path.parent.name == "非橙子"}):
        raise RuntimeError("保留清单和验证集不一致")
    best_gui = max((m for m, _ in rows), key=lambda m: (m["balanced_accuracy"], m["macro_f1"]))
    best_matched = max(matched.values(), key=lambda m: (m["balanced_accuracy"], m["macro_f1"]))
    red_files = {item["filename"] for item in selection["images"]
                 if item["category"] in red_names}

    def red_false_positives(name, part=""):
        source = RESULT / f"{name}-result" / part / "额外_逐样本预测结果" / "逐样本预测.csv"
        with source.open(encoding="utf-8-sig", newline="") as file:
            return sum(row["文件名"] in red_files and row["预测标签"] == "橙子"
                       for row in csv.DictReader(file) if row["真实标签"] == "非橙子")

    near_names = {(Path(item["train"]).name, Path(item["validation"]).name)
                  for item in near or []}
    near_note = ("；近重复算法提示 1 组：`变换_excellent9_0060.jpg` 与 `santol_013.jpg`，"
                 "人工核对为外形相似的不同果实。"
                 if near_names == {("变换_excellent9_0060.jpg", "santol_013.jpg")}
                 else f"；近重复算法提示 {len(near)} 组，需逐一核对。" if near is not None
                 else "；近重复检查未完成。")

    lines = [
        "# 模型筛选报告",
        "",
        "## 1. 对比范围与口径",
        "",
        "- 训练集：橙子 1001 张、非橙子 1011 张；当前验证集：橙子 100 张、非橙子 100 张。六个训练器各用 flash、std、pro 三档位训练一次，共 18 个模型。模型未重新训练，只更换验证负样本。",
        f"- 非橙子从原 97 类、4850 张中按这 18 个模型两种口径的旧错判精选 100 张，涵盖 {len(quotas)} 类；红色圆形水果相关类别保留 {red_count} 张。逐图名单、类别配额与旧错判次数见 `本地训练/result/验证集筛选记录/选样清单.json`。高正确率类别仅留 1～2 张。",
        "- 这 100 张是用当前 18 个模型的旧错误挑出的**困难集**，新分数适合比较这些模型在已知薄弱点上的表现，不能视作未经筛选的独立泛化准确率。验证集仍不含普通柠檬和乒乓球，需另备素材检验。",
        "- flash：10 轮、批量 32、学习率 0.001、隐藏单元 100；std：20 轮、批量 16、学习率 0.001、隐藏单元 100；pro：最多 120 轮、批量 8、学习率 0.0003、隐藏单元 256。训练器的早停可减少实际轮次。",
        "- 所有结果都在同一套独立验证集上，以模型概率最大的一类作为预测类别。每个模型目录的顶层是当前本地测试页口径；`基底一致验证/` 则用训练时的冻结图和上传图片处理方式复测。两处均有指标、图表、逐样本 CSV 及已导出的错误图片。",
        f"- 训练集 SHA-256：`{manifest['train_sha256']}`；验证集 SHA-256：`{manifest['validation_sha256']}`。训练/验证字节完全相同图片 {len(exact)} 组" + near_note,
        "- 已知与训练集同源的 `banana_003.jpg`、`grapefruit_005.jpg`、`grapefruit_020.jpg` 未进入保留名单。旧版 4950 张验证结果已归档于 `本地训练/result/历史验证_4950/`。",
        "- 当前本地测试页加载 ZIP 时重建 Keras MobileNetV2；训练器提取特征时用项目附带的 TFJS 图。同一输入图上两者的特征余弦相似度抽样均值仅 0.9045。两套测试不可混为一个分数，浏览器实际使用哪种基底仍需上传确认。当前测试页口径抽样 8 张图，与 GUI 逐批预测的最大概率差为 0.0000003。运行环境为 TensorFlow 2.10.0，CPU。",
        "- 为批量比较，共享了冻结 MobileNetV2 的特征提取结果；表中训练耗时只计每组分类头训练、画图和导出，不含一次性特征提取。所有分类头仍由对应训练器训练。",
        "",
        "## 2. 当前本地测试页口径的分类性能",
        "",
        "表中漏判是橙子判为非橙子（100 张中的数量），误报是非橙子判为橙子（100 张中的数量）。",
        "",
        "| 模型与完整测试结果 | 总准确率 | 平衡准确率 | 橙子召回率 | 橙子精确率 | 橙子 F1 | 漏判 | 误报 | 非橙子召回率 | 宏 F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m, _ in rows:
        lines.append("| " + " | ".join((label(m), percent(m["accuracy"]),
                     percent(m["balanced_accuracy"]), percent(m["orange_recall"]),
                     percent(m["orange_precision"]), percent(m["orange_f1"]),
                     str(m["false_negative"]), str(m["false_positive"]),
                     percent(m["nonorange_recall"]), percent(m["macro_f1"]))) + " |")

    lines += [
        "",
        "### 概率排序、训练与产物",
        "",
        "| 模型 ZIP | 橙子 ROC AUC | 橙子 AP | 宏 AP | Kappa | 加权 F1 | 实际/上限轮次 | 内部验证 | 分类头与导出耗时 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for m, t in rows:
        name = m["name"]
        lines.append("| " + " | ".join((
            f"[{name}](本地训练/model/{name}.zip)",
            percent(m["auc_orange"]), percent(m["ap_orange"]),
            percent(m["macro_ap"]), f"{m['kappa']:.3f}",
            percent(m["weighted_f1"]),
            f"{t['epochs_run']}/{t['params']['epochs']}",
            "无" if t["validation_status"] == "disabled" else "有",
            f"{t['duration_seconds']:.1f} 秒")) + " |")

    lines += [
        "",
        "## 3. 训练基底一致的复测",
        "",
        "这一口径用 `model/basemodels/model.json` 的冻结 TFJS 图和训练器的图片上传处理方式，对同一 200 张验证图重新提取特征，再送入各 ZIP 的分类头。每个模型的 [结果目录](本地训练/result/) 下另有 `基底一致验证/`，含完整图表、CSV、批处理及已经导出的错判图片。",
        "",
        "| 模型与基底一致结果 | 总准确率 | 平衡准确率 | 橙子召回率 | 橙子精确率 | 橙子 F1 | 漏判 | 误报 | 橙子 AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m, _ in rows:
        name = m["name"]
        other = matched[name]
        lines.append("| " + " | ".join((
            f"[{name}](本地训练/result/{name}-result/基底一致验证/)",
            percent(other["accuracy"]), percent(other["balanced_accuracy"]),
            percent(other["orange_recall"]), percent(other["orange_precision"]),
            percent(other["orange_f1"]), str(other["false_negative"]),
            str(other["false_positive"]), percent(other["orange_ap"]))) + " |")

    lines += [
        "",
        "## 4. 怎样审查这些参数",
        "",
        "- **混淆矩阵、漏判、误报**：矩阵的行是真实类别，列是预测类别。漏判会错过橙子；误报会把葡萄柚等非橙子当成橙子。直接看两类错误张数最容易判断是否符合使用场景。",
        "- **总准确率**：全部判对的比例。两类各 100 张，恒判非橙子的准确率是 50%；当前准确率更容易直接比较，但仍需同时看两类错误。",
        "- **召回率**：某类真实图片被找回的比例。橙子召回率每相差 1 个百分点，约等于这套验证集上多找回或漏掉 1 张橙子；非橙子召回率越高，误报越少。",
        "- **橙子精确率**：所有预测为橙子的图片中，真正为橙子的比例；误报越多，它通常越低。",
        "- **F1 与宏 F1**：精确率和召回率的调和平均；宏 F1 把两类等权看待。当前两类各 100 张，因此加权 F1 与宏 F1 相同。",
        "- **平衡准确率**：两类召回率的简单平均；当前两类数量相同，与总准确率数值相同。",
        "- **ROC AUC**：改变判定阈值时，模型把橙子排在非橙子前面的能力；它不表示当前阈值下的漏判或误报数。",
        "- **AP（平均精确率）**：把不同召回位置的精确率加权汇总，反映 PR 表现。橙子 AP 的随机基线约为橙子占比 50%；宏 AP 是两类 AP 的平均。",
        "- **Kappa**：扣除类别频率下偶然一致后的分类一致性；0 附近表示与按频率猜测相近，越接近 1 越好。",
        "- **实际轮次与内部验证**：早停训练器可能未跑满上限；全量版不划内部验证集，训练图片更多，但没有内部早停依据。模型优劣以同一套独立验证集结果为准。",
        "",
        "## 5. 推荐供进一步判断的模型",
        "",
    ]
    by_name = {m["name"]: m for m, _ in rows}
    selections = [
        ("deeporange-v1-original-pro", "两套口径的平均表现最好，适合作为首选对照"),
        ("deeporange-v1-optimized-flash", "两套口径下的误报都比首选少，但橙子漏判更多"),
        ("deeporange-v1-optimized-std", "橙子漏判较少，两套口径都维持在 60% 及以上的准确率"),
        ("deeporange-v1-optimized-full-data-std", "训练基底一致口径误报最少；本地测试页口径仅作有条件备选"),
    ]
    for name, reason in selections:
        gui, consistent = by_name[name], matched[name]
        lines.append(
            f"- **{name}**：{reason}。当前测试页漏判 {gui['false_negative']} / 误报 {gui['false_positive']}，"
            f"平衡准确率 {percent(gui['balanced_accuracy'])}；训练基底一致复测漏判 "
            f"{consistent['false_negative']} / 误报 {consistent['false_positive']}，"
            f"平衡准确率 {percent(consistent['balanced_accuracy'])}。"
            f"[查看两套结果](本地训练/result/{name}-result/)。")
    lines += [
        "",
        f"当前测试页口径的最高准确率是 `{best_gui['name']}`（{percent(best_gui['balanced_accuracy'])}）；"
        f"训练基底一致口径的最高准确率是 `{best_matched['name']}`（{percent(best_matched['balanced_accuracy'])}）。"
        "两套排序不同，不能把它们混作一个分数。",
        "",
        f"红色圆形水果相关类别共 {red_count} 张。`original-pro` 在其中误报 "
        f"{red_false_positives('deeporange-v1-original-pro')}/{red_count} 张（本地测试页）、"
        f"{red_false_positives('deeporange-v1-original-pro', '基底一致验证')}/{red_count} 张（基底一致）；"
        f"`optimized-flash` 分别为 {red_false_positives('deeporange-v1-optimized-flash')}/{red_count} "
        f"和 {red_false_positives('deeporange-v1-optimized-flash', '基底一致验证')}/{red_count} 张。"
        "这组仍存在明显误报，训练数据可针对这些类别继续改进。",
        "",
        "这份验证集是按当前模型错误筛出的困难集，每组模型也只训练一次；分数和排序可能随独立新素材、随机初始化或上传平台预处理变化。"
        "最终选择前须确认浏览器实际使用的 MobileNetV2 基底和图片预处理方式，并补测普通柠檬、乒乓球。",
        "",
    ]
    errors = defaultdict(lambda: [0, 0])
    for metrics, _ in rows:
        directory = RESULT / f"{metrics['name']}-result"
        for index, part in enumerate(("", "基底一致验证")):
            csv_path = directory / part / "额外_逐样本预测结果" / "逐样本预测.csv"
            with csv_path.open(encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    if row["真实标签"] != row["预测标签"]:
                        errors[row["文件名"]][index] += 1
    positive = [path for path in val if path.parent.name == "橙子"]
    negative = [path for path in val if path.parent.name == "非橙子"]
    difficult_oranges = sorted(positive, key=lambda path: (
        -min(errors[path.name]), -sum(errors[path.name]), path.name))[:8]
    unanimous_negatives = [path for path in negative if errors[path.name] == [18, 18]]
    category_files = defaultdict(list)
    for item in selection["images"]:
        category_files[item["category"]].append(item["filename"])
    old_errors = defaultdict(lambda: [0, 0])
    archive = RESULT / "历史验证_4950"
    for metrics, _ in rows:
        directory = archive / f"{metrics['name']}-result"
        for index, part in enumerate(("", "基底一致验证")):
            csv_path = directory / part / "额外_逐样本预测结果" / "逐样本预测.csv"
            with csv_path.open(encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    if row["真实标签"] == "非橙子" and row["预测标签"] == "橙子":
                        old_errors[row["文件名"].rsplit("_", 1)[0]][index] += 1

    lines += [
        "## 6. 有关数据配比的通用调整建议",
        "",
        "以下每个数字均为 18 个模型中的错判数，左、右分别对应本地测试页与训练基底一致口径；两套结果单独计票。负样本 100 张按旧模型的错判挑选，属于困难集。旧版完整负样本的统计用于交叉核对类别趋势，不能把困难集误报率当作自然出现频率。",
        "",
        "### 经常漏判的橙子",
        "",
        f"有 {sum(min(errors[path.name]) >= 12 for path in positive)} 张橙子在两套口径下各被至少 12/18 个模型漏判；没有图片在两套口径下同时被 18/18 个模型漏判。下面列出共同错判最多的 8 张：",
        "",
        "| 验证图 | 本地测试页漏判 | 基底一致漏判 |",
        "| --- | ---: | ---: |",
    ]
    for path in difficult_oranges:
        gui_errors, matched_errors = errors[path.name]
        lines.append(f"| [{path.name}](<验证集/橙子/{path.name}>) | {gui_errors}/18 | {matched_errors}/18 |")
    lines += [
        "",
        "人工查看这些图，常见画面是树上的单果或果串、成堆售卖的果实、黄绿色果皮、带叶枝条，以及切面；两张旋转图带明显黑角。黑角、水印、主体太小和果种边界模糊的验证图应先审查标签与画面质量。若确为橙子，再用**不同原图**补相应场景的训练样本；不要把验证图或其裁剪、旋转版本加入训练集。",
        "",
        "### 经常误报的非橙子",
        "",
        f"当前困难集有 {len(unanimous_negatives)} 张负样本在两套口径下均被 18/18 个模型误报：" +
        "、".join(f"[{path.name}](验证集/非橙子/{path.name})" for path in unanimous_negatives) + "。",
        "",
        "下表展示值得优先核查的类别。当前列的分母是该类保留图片数 × 18；旧版完整列的分母是每类 50 张 × 18 = 900。每格按本地测试页 / 基底一致口径排列。",
        "",
        "| 非橙子类别 | 当前保留张数 | 当前困难集误报 | 旧版 50 张误报 |",
        "| --- | ---: | ---: | ---: |",
    ]
    priority_categories = (
        "grapefruit", "apricot", "taxus_baccata", "sea_buckthorn", "grenadilla",
        "chico", "pomegranate", "mabolo", "mango", "hog_plum")
    for category in priority_categories:
        filenames = category_files[category]
        current = [sum(errors[name][index] for name in filenames) for index in (0, 1)]
        previous = old_errors[category]
        lines.append(f"| `{category}` | {len(filenames)} | "
                     f"{current[0]}/{len(filenames) * 18} / {current[1]}/{len(filenames) * 18} | "
                     f"{previous[0]}/900 / {previous[1]}/900 |")
    lines += [
        "",
        "红色圆形水果仍需重点保留：`pomegranate` 在旧版 50 张上的误报为 391/900 / 120/900，当前 8 张为 125/144 / 89/144；`acerola`、`apple`、`jujube` 等在两种口径上的难度不同，应保留真实背景、近景和树上照片，不靠单一红色背景或重复增强来凑数。葡萄柚、杏、芒果等橙黄圆果及 `taxus_baccata`、`sea_buckthorn` 等树上小果也要覆盖。",
        "",
        "1. **正类先补真实场景。** 当前正类来源以合成图为主。第一轮可用约 80～120 张独立拍摄、可确认果种的树上果串、市场成堆、黄绿色或光照变化明显的橙子，替换同量重复感较强的合成图，保持正类总量约 1000 张。切面图只在能确认是橙子时纳入；不要用同一原图的多次旋转充数。",
        "2. **负类补清晰可辨的难例。** 在非橙子约 1000 张总量内，先试着替换约 60～100 张普遍容易判对、重复或背景过于单一的样本，分配给表中橙黄圆果、树上小果及红色圆果。优先选能从果蒂、表皮、切面或叶片区分的独立照片；外观无法稳定判别的孤立果实不宜硬贴负标签。正式评测重点柠檬、乒乓球的已有配额先保留。",
        "3. **保持类别与场景平衡。** 两类总量仍接近 1:1；两类都要有白底和真实背景、单果和多果、近景和树上场景。补图前先检查错标、同源近重复、水印及中心裁剪后的主体大小，更新来源清单和实际配额。",
        "4. **固定验证边界。** 本次验证图及同源图不可参与下一轮训练。现有困难集适合检查已发现的弱点；反复按它调样本和选模型会逐渐过拟合。最终定型前另备未参与调参的橙子、普通柠檬、乒乓球等独立测试图，一次性检查泛化表现。",
        "",
    ]
    (ROOT / "模型筛选报告.md").write_text("\n".join(lines), encoding="utf-8")
    default_zip = HERE / "橙子识别项目.识物模型.zip"
    if all((MODEL / f"{m['name']}.zip").is_file() for m, _ in rows):
        default_zip.unlink(missing_ok=True)
    print("已写入模型筛选报告.md", flush=True)


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "train", "evaluate", "verify", "backbone", "matched_prepare", "matched_check", "matched_evaluate", "audit", "report", "all"))
    parser.add_argument("--trainer", choices=TRAINERS)
    parser.add_argument("--preset", choices=PRESETS)
    args = parser.parse_args()
    if args.phase in ("prepare", "all"):
        prepare()
    selected = [(t, p, n) for t, p, n in pairs()
                if (args.trainer is None or t == args.trainer)
                and (args.preset is None or p == args.preset)]
    if args.phase in ("train", "all"):
        for item in selected:
            train_one(*item)
    if args.phase in ("evaluate", "all"):
        for item in selected:
            evaluate_one(*item)
    if args.phase == "verify":
        verify_first_model()
    if args.phase == "backbone":
        verify_backbone()
    if args.phase == "matched_prepare":
        prepare_matched_validation()
    if args.phase == "matched_check":
        compare_matched_first_model()
    if args.phase == "matched_evaluate":
        evaluate_matched()
    if args.phase == "audit":
        audit_near_duplicates()
    if args.phase == "all":
        audit_near_duplicates()
        prepare_matched_validation()
        evaluate_matched()
    if args.phase in ("report", "all"):
        write_report()
    if args.phase == "all":
        shutil.rmtree(CACHE)


if __name__ == "__main__":
    main_cli()
