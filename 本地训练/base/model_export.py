"""训练器共用的曲线图与官网模型包导出。"""

from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


class ModelExport:
    MODEL_FILES = ("metadata.json", "model.json", "weights.bin")
    ROOT = Path(__file__).resolve().parent.parent

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)

    def save_plot(self, name, render):
        while True:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = self.ROOT / f"{name}_{stamp}.png"
            try:
                path.touch(exist_ok=False)
                break
            except FileExistsError:
                continue
        try:
            render(str(path))
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path

    def package(self):
        files = [self.model_dir / name for name in self.MODEL_FILES]
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
        archive = self.ROOT / "橙子识别项目.识物模型.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as output:
            for path in files:
                output.write(path, arcname=path.name)
        return archive
