"""Deterministically transform white-background oranges and composite cutouts."""

from __future__ import annotations

from collections import Counter
from hashlib import blake2b
from pathlib import Path
import random
import re

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps


class OrangeObfuscator:
    IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
    SEED = 20260924
    SIZE = 320
    TRANSFORM_COUNT = 140
    MAX_BACKGROUND_USES = 12
    PURE_BACKGROUND_SHARE = 0.15

    def __init__(self, root: Path) -> None:
        self.root = root
        self.output_root = root / "数据源" / "橙子数据源"

    def images(self, directory: Path) -> list[Path]:
        return sorted(
            (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES),
            key=lambda path: path.name.casefold(),
        )

    def groups(self, name: str) -> dict[str, list[Path]]:
        directory = self.root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"缺少目录：{directory}")
        groups = {
            folder.name: self.images(folder)
            for folder in sorted(directory.iterdir(), key=lambda path: path.name.casefold())
            if folder.is_dir()
        }
        if not groups or not any(groups.values()):
            raise ValueError(f"目录中没有可用图片：{directory}")
        return groups

    def normalize_backgrounds(self) -> dict[str, list[Path]]:
        directory = self.root / "背景"
        if not directory.is_dir():
            raise FileNotFoundError(f"缺少目录：{directory}")

        backgrounds: dict[str, list[Path]] = {}
        for folder in sorted(directory.iterdir(), key=lambda path: path.name.casefold()):
            if not folder.is_dir():
                continue
            files = self.images(folder)
            pattern = re.compile(rf"{re.escape(folder.name)}_(\d{{3,}})$")
            used = {
                int(match.group(1))
                for path in files
                if (match := pattern.fullmatch(path.stem))
            }
            number = 1
            for path in files:
                if pattern.fullmatch(path.stem):
                    continue
                while number in used:
                    number += 1
                target = folder / f"{folder.name}_{number:03d}{path.suffix.lower()}"
                path.rename(target)
                used.add(number)
                number += 1
            backgrounds[folder.name] = self.images(folder)
            print(f"[背景] {folder.name}：{len(backgrounds[folder.name])} 张，文件名已规范化")

        if not backgrounds or not any(backgrounds.values()):
            raise ValueError(f"目录中没有可用背景：{directory}")
        return backgrounds

    def random_for(self, path: Path) -> random.Random:
        identity = f"{self.SEED}/{path.relative_to(self.root).as_posix()}".encode("utf-8")
        return random.Random(int.from_bytes(blake2b(identity, digest_size=8).digest(), "big"))

    def select_transforms(self, groups: dict[str, list[Path]]) -> dict[str, list[Path]]:
        total = sum(map(len, groups.values()))
        count = min(self.TRANSFORM_COUNT, total)
        quotas = {name: count * len(files) // total for name, files in groups.items()}
        remainder = count - sum(quotas.values())
        order = sorted(groups, key=lambda name: (-(count * len(groups[name]) % total), name.casefold()))
        for name in order[:remainder]:
            quotas[name] += 1
        return {
            name: sorted(
                sorted(files, key=lambda path: self.random_for(path).getrandbits(64))[:quotas[name]],
                key=lambda path: path.name.casefold(),
            )
            for name, files in groups.items()
        }

    def assign_backgrounds(self, backgrounds: dict[str, list[Path]], count: int) -> list[Path]:
        plain = backgrounds.get("纯色", [])
        textured = [path for name, files in backgrounds.items() if name != "纯色" for path in files]
        plain_count = min(round(count * self.PURE_BACKGROUND_SHARE), len(plain) * self.MAX_BACKGROUND_USES)
        plain_count = max(plain_count, count - len(textured) * self.MAX_BACKGROUND_USES)
        if plain_count > len(plain) * self.MAX_BACKGROUND_USES or count - plain_count > len(textured) * self.MAX_BACKGROUND_USES:
            raise ValueError("合格背景数量不足：无法在单张背景最多使用 12 次的限制内完成合成。")

        chooser = random.Random(self.SEED)

        plain_choices = iter(self.balanced_backgrounds(plain, plain_count, chooser))
        textured_choices = iter(self.balanced_backgrounds(textured, count - plain_count, chooser))
        slots = [True] * plain_count + [False] * (count - plain_count)
        chooser.shuffle(slots)
        return [next(plain_choices if is_plain else textured_choices) for is_plain in slots]

    def balanced_backgrounds(self, files: list[Path], needed: int, chooser: random.Random) -> list[Path]:
        result: list[Path] = []
        while len(result) < needed:
            cycle = files.copy()
            chooser.shuffle(cycle)
            result.extend(cycle[: needed - len(result)])
        return result

    def orange_mask(self, image: Image.Image) -> Image.Image:
        hsv = image.convert("HSV")
        saturation = hsv.getchannel("S").point(lambda value: 255 if value >= 80 else 0)
        brightness = hsv.getchannel("V").point(lambda value: 255 if value >= 35 else 0)
        mask = ImageChops.multiply(saturation, brightness)
        mask = mask.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
        mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.7))
        if not mask.getbbox():
            raise ValueError("无法识别橙子轮廓")
        return mask

    def transform(self, source: Path, rng: random.Random) -> Image.Image:
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB").filter(ImageFilter.MedianFilter(3))
        mask = self.orange_mask(image)
        crop = image.convert("RGBA")
        crop.putalpha(mask)
        crop = crop.crop(mask.getbbox())
        crop = crop.rotate(rng.uniform(-18, 18), Image.Resampling.BICUBIC, expand=True)
        side = round(self.SIZE * rng.uniform(0.58, 0.83))
        crop = ImageOps.contain(crop, (side, side), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.SIZE, self.SIZE), "white")
        x = max(0, min(self.SIZE - crop.width, (self.SIZE - crop.width) // 2 + rng.randint(-30, 30)))
        y = max(0, min(self.SIZE - crop.height, (self.SIZE - crop.height) // 2 + rng.randint(-30, 30)))
        canvas.paste(crop, (x, y), crop)
        return canvas

    def composite(self, source: Path, background: Path, rng: random.Random) -> Image.Image:
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        mask = self.orange_mask(image)
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.93, 1.07))
        subject = image.convert("RGBA")
        subject.putalpha(mask)
        subject = subject.crop(mask.getbbox())
        subject = subject.rotate(rng.uniform(-22, 22), Image.Resampling.BICUBIC, expand=True)
        side = round(self.SIZE * rng.uniform(0.48, 0.76))
        subject.thumbnail((side, side), Image.Resampling.LANCZOS)

        with Image.open(background) as raw:
            scene = ImageOps.exif_transpose(raw).convert("RGB")
        padded_size = round(self.SIZE * 1.5)
        scene = ImageOps.fit(scene, (padded_size, padded_size), Image.Resampling.LANCZOS, centering=(rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75)))
        scene = scene.rotate(rng.uniform(-12, 12), Image.Resampling.BICUBIC)
        margin = (padded_size - self.SIZE) // 2
        scene = scene.crop((margin, margin, margin + self.SIZE, margin + self.SIZE))
        scene = ImageEnhance.Brightness(scene).enhance(rng.uniform(0.9, 1.1))
        x = max(0, min(self.SIZE - subject.width, (self.SIZE - subject.width) // 2 + rng.randint(-34, 34)))
        y = max(0, min(self.SIZE - subject.height, (self.SIZE - subject.height) // 2 + rng.randint(-34, 34)))
        scene.paste(subject, (x, y), subject)
        return scene

    def output_directory(self, kind: str, group: str) -> Path:
        target = self.output_root / f"{kind}_{group}"
        if target.exists():
            unexpected = [path for path in target.iterdir() if not path.is_file() or not re.fullmatch(r"\d+\.jpg", path.name)]
            if unexpected:
                raise ValueError(f"输出目录含非本脚本文件，未覆盖：{target}")
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write_group(self, kind: str, group: str, files: list[Path], backgrounds: list[Path] | None = None) -> None:
        target = self.output_directory(kind, group)
        expected = {f"{index:04d}.jpg" for index in range(1, len(files) + 1)}
        for index, source in enumerate(files, 1):
            rng = self.random_for(source)
            image = self.transform(source, rng) if backgrounds is None else self.composite(source, backgrounds[index - 1], rng)
            image.save(target / f"{index:04d}.jpg", quality=92, subsampling=0)
        for old in target.iterdir():
            if old.name not in expected:
                old.unlink()
        print(f"[{kind}] {group}：{len(files)} 张 → {target}")

    def run(self) -> None:
        transform_groups = self.select_transforms(self.groups("待变换数据"))
        composite_groups = self.groups("待混淆数据")
        backgrounds = self.normalize_backgrounds()
        all_composites = [(group, path) for group, files in composite_groups.items() for path in files]
        assignments = self.assign_backgrounds(backgrounds, len(all_composites))
        counts = Counter(path.parent.name for path in assignments)
        print(f"[背景分配] {dict(sorted(counts.items()))}")

        for group, files in transform_groups.items():
            self.write_group("变换", group, files)
        offset = 0
        for group, files in composite_groups.items():
            self.write_group("混淆", group, files, assignments[offset:offset + len(files)])
            offset += len(files)
        print(f"完成：变换 {sum(map(len, transform_groups.values()))} 张，混淆 {len(all_composites)} 张；固定随机种子 {self.SEED}。")


if __name__ == "__main__":
    OrangeObfuscator(Path(__file__).resolve().parent).run()
