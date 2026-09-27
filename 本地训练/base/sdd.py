import os
import tempfile
import zipfile
from pathlib import Path


def unzip_model(zip_path, extract_to=None):
    """
    解压 ZIP 模型文件。
    :param zip_path: ZIP 文件路径
    :param extract_to: 解压目标目录，如果为 None 则使用临时目录
    :return: 解压后的根目录路径
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP 文件不存在: {zip_path}")

    if extract_to is None:
        temp_dir = tempfile.mkdtemp(prefix="tfjs_model_")
        print(f"[系统] 创建临时目录: {temp_dir}")
    else:
        temp_dir = extract_to
        os.makedirs(temp_dir, exist_ok=True)

    print(f"[系统] 正在解压 {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    root_path = Path(temp_dir)
    model_json_files = list(root_path.rglob("model.json"))

    if not model_json_files:
        raise FileNotFoundError("在 ZIP 包中未找到 model.json")

    model_dir = model_json_files[0].parent
    print(f"[系统] 模型根目录定位到: {model_dir}")

    return str(model_dir), temp_dir if extract_to is None else None


def test_prediction(classifier, image_paths):
    """批量测试预测"""
    print("\n--- 开始预测测试 ---")
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"[跳过] 文件不存在: {img_path}")
            continue

        try:
            preds = classifier.predict(img_path)
            decision = classifier.decide(preds)

            status = "✅ 确定" if not decision['uncertain'] else "⚠️ 不确定"
            reason = f"({decision['reason']})" if decision['uncertain'] else ""

            print(f"\n📷 图片: {os.path.basename(img_path)}")
            print(f"   结果: {status} -> [{decision['label']}] {reason}")
            print(f"   置信度: {decision['probability']:.2%}")
            print(f"   详细概率:")
            for p in preds:
                bar = "█" * int(p['probability'] * 20)
                print(f"     - {p['label']:<8}: {p['probability']:.4f} {bar}")

        except Exception as e:
            print(f"[错误] 处理图片 {img_path} 时出错: {e}")

