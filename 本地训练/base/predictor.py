import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

from PIL import Image, ImageOps
import tensorflow as tf

#  配置常量 
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.65
MARGIN_THRESHOLD = 0.12
CLASS_NAMES = ["橙子", "非橙子"]


#  图像处理工具 
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


def prepare_image_for_mobilenet(path: str) -> np.ndarray:
    """准备图片用于 MobileNet 输入: uint8 (224, 224, 3)"""
    img = _center_crop_on_white(Image.open(path))
    return np.asarray(img, dtype=np.uint8)


#  MobileNet 特征提取器 (冻结) 
class MobileNetExtractor:
    """使用 Keras MobileNetV2 alpha=0.5 提取 1280 维特征"""

    def __init__(self):
        self.model = tf.keras.applications.MobileNetV2(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            alpha=0.5,
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )

    def extract(self, image_uint8: np.ndarray) -> np.ndarray:
        """
        :param image_uint8: (224, 224, 3) uint8
        :return: (1, 1280) float32 embedding
        """
        # TFJS MobileNet 通常预处理: x/255 * 2 - 1 (范围 -1 到 1)
        x = image_uint8.astype("float32") / 255.0 * 2.0 - 1.0
        x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
        embedding = self.model(x, training=False)
        return embedding.numpy().astype(np.float32)  # (1, 1280)


#  TFJS Layers 模型加载器 
def load_tfjs_layers_model(model_json_path: str) -> tf.keras.Model:
    """
    手动加载 TFJS layers-model 格式为 Keras 模型。
    支持 Dense, Dropout 等常见层。
    """
    with open(model_json_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)

    if spec.get("format") != "layers-model":
        raise ValueError("仅支持 layers-model 格式")

    model_dir = Path(model_json_path).parent

    # 1. 构建 Keras 模型结构
    inputs = tf.keras.Input(shape=(1280,), name="input_embedding")
    x = inputs

    layers_config = spec["modelTopology"]["config"]["layers"]

    for layer_cfg in layers_config:
        class_name = layer_cfg["class_name"]
        config = layer_cfg["config"]

        if class_name == "Dense":
            units = config["units"]
            activation = config["activation"]
            use_bias = config["use_bias"]

            # 获取权重名称以便后续赋值
            layer_name = config["name"]

            x = tf.keras.layers.Dense(
                units=units,
                activation=activation,
                use_bias=use_bias,
                name=layer_name
            )(x)

        elif class_name == "Dropout":
            rate = config["rate"]
            x = tf.keras.layers.Dropout(rate=rate)(x)

        else:
            print(f"[警告] 未实现的层类型: {class_name}，跳过或可能导致错误")

    outputs = x
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # 2. 加载权重
    weight_manifest = spec["weightsManifest"][0]
    bin_paths = weight_manifest["paths"]

    # 读取所有二进制分片
    raw_bytes = b""
    for p in bin_paths:
        raw_bytes += (model_dir / p).read_bytes()

    offset = 0
    weight_map = {}

    for w_info in weight_manifest["weights"]:
        name = w_info["name"]
        shape = w_info["shape"]
        dtype = np.dtype(w_info["dtype"]).newbyteorder("<")  # Little endian

        num_elements = int(np.prod(shape))
        byte_size = num_elements * dtype.itemsize

        data = np.frombuffer(raw_bytes, dtype=dtype, count=num_elements, offset=offset)
        weight_map[name] = data.reshape(shape)

        offset += byte_size

    # 3. 将权重设置到 Keras 模型中
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            kernel_name = f"{layer.name}/kernel"
            bias_name = f"{layer.name}/bias"

            if kernel_name in weight_map and bias_name in weight_map:
                layer.set_weights([weight_map[kernel_name], weight_map[bias_name]])
            else:
                raise ValueError(f"找不到层 {layer.name} 的权重: {kernel_name}, {bias_name}")

    return model


#  主预测器类 
class PretrainedOrangePredictor:
    def __init__(self, model_dir: str):
        """
        :param model_dir: 包含 model.json 和 weights.bin 的目录
        """
        print("[系统] 初始化预训练预测器...")

        # 加载分类头 (Dense 模型)
        model_json = os.path.join(model_dir, "model.json")
        if not os.path.exists(model_json):
            raise FileNotFoundError(f"找不到 model.json: {model_json}")

        self.classifier_model = load_tfjs_layers_model(model_json)
        print("[系统] 分类头模型加载成功")

        # 加载特征提取器 (MobileNet)
        self.extractor = MobileNetExtractor()
        print("[系统] MobileNet 特征提取器加载成功")

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        预测单张图片
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        # 1预处理图片
        img_uint8 = prepare_image_for_mobilenet(image_path)

        # 提取嵌入 (1, 1280)
        embedding = self.extractor.extract(img_uint8)

        # 分类预测
        probs = self.classifier_model.predict(embedding, verbose=0)[0]

        # 格式化结果
        labels = CLASS_NAMES
        preds = []
        for label, prob in zip(labels, probs):
            preds.append({"label": label, "probability": float(prob)})

        # 按概率降序排列
        preds.sort(key=lambda x: x["probability"], reverse=True)

        # 决策逻辑
        top = preds[0]
        second = preds[1] if len(preds) > 1 else {"probability": 0}

        gap = top["probability"] - second["probability"]
        uncertain = False
        reason = None

        if top["probability"] < CONFIDENCE_THRESHOLD:
            uncertain = True
            reason = "low-confidence"
        elif gap < MARGIN_THRESHOLD:
            uncertain = True
            reason = "small-margin"

        return {
            "preds": preds,
            "decision": {
                "label": top["label"],
                "probability": top["probability"],
                "uncertain": uncertain,
                "reason": reason
            }
        }


