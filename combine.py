# -*- coding: utf-8 -*-
"""Copy images from the two source trees into flat, upload-ready folders."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import cos, pi
import os
from pathlib import Path
import shutil
import sys
import tempfile

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None


IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
)
PHASH_IMAGE_SIZE = 32
PHASH_DCT_SIZE = 8
PHASH_COSINES = tuple(
    tuple(
        cos(pi * (sample + 0.5) * frequency / PHASH_IMAGE_SIZE)
        for sample in range(PHASH_IMAGE_SIZE)
    )
    for frequency in range(PHASH_DCT_SIZE)
)
MANIFEST_FILENAME = "manifest.json"
MANIFEST_INDEX_FILENAME = ".combine_manifest_index.json"
MANIFEST_RANDOM_SEED = 20260924


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    perceptual_hash: int
    pixels: bytes


@dataclass(frozen=True)
class ManifestSource:
    path: Path
    count: int


class DatasetCombiner:
    CATEGORIES = (
        ("橙子数据源", "橙子"),
        ("非橙子数据源", "非橙子"),
    )

    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self) -> int:
        if Image is None or ImageOps is None:
            print("[错误] 数据整理和查重需要 Pillow，请先执行：python -m pip install Pillow")
            return 1

        try:
            manifest_path = self.root / "数据源" / "非橙子数据源" / MANIFEST_FILENAME
            self.load_manifest(manifest_path)
            self.load_manifest_index()
        except ValueError as error:
            print(f"[错误] manifest 配置无效：{error}")
            return 1

        total = 0
        missing = False

        for source_name, output_name in self.CATEGORIES:
            source_root = self.root / "数据源" / source_name
            output_root = self.root / output_name
            copied, source_exists = self.combine_category(source_root, output_root)
            total += copied
            missing |= not source_exists

        print(f"\n整理完成，共复制 {total} 张图片。")
        if missing:
            return 1

        datasets = tuple(
            (output_name, self.root / output_name)
            for _, output_name in self.CATEGORIES
        )
        return 0 if VisualDuplicateCleaner(datasets).run() else 1

    def combine_category(self, source_root: Path, output_root: Path) -> tuple[int, bool]:
        if not source_root.is_dir():
            print(f"[跳过] 未找到数据源目录：{source_root}")
            return 0, False

        try:
            manifest_sources = self.load_manifest(source_root / MANIFEST_FILENAME)
            previous_outputs = (
                self.load_manifest_index()
                if source_root.name == "非橙子数据源"
                else ()
            )
        except ValueError as error:
            print(f"[错误] manifest 配置无效：{error}")
            return 0, False

        if output_root.is_symlink():
            print(f"[错误] 输出目录不能是符号链接：{output_root}")
            return 0, False

        output_root.mkdir(parents=True, exist_ok=True)
        for name in previous_outputs:
            old_file = output_root / name
            if old_file.is_file() and not old_file.is_symlink():
                old_file.unlink()
        used_names: set[str] = set()
        copied = 0
        groups = sorted(
            (path for path in source_root.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        )

        for group in groups:
            group_copied, _ = self.copy_group(
                group, self.image_files(group), output_root, used_names
            )
            copied += group_copied

            print(f"[{source_root.name}] {group.name}: 复制 {group_copied} 张")

        if not groups:
            print(f"[{source_root.name}] 未找到来源子目录。")

        manifest_outputs: list[str] = []
        for source in manifest_sources:
            images = self.image_files(source.path)
            relative_path = source.path.relative_to(self.root.resolve()).as_posix()
            selected = sorted(
                sorted(
                    images,
                    key=lambda image: sha256(
                        f"{MANIFEST_RANDOM_SEED}:{relative_path}/{image.name}".encode(
                            "utf-8"
                        )
                    ).digest(),
                )[: source.count],
                key=self.path_sort_key,
            )
            group_copied, names = self.copy_group(
                source.path, selected, output_root, used_names
            )
            copied += group_copied
            manifest_outputs.extend(names)
            print(
                f"[manifest] {relative_path}: 选取 {source.count} 张，复制 {group_copied} 张"
            )

        normalized = self.normalize_output(output_root)
        if source_root.name == "非橙子数据源":
            self.save_manifest_index(manifest_outputs)
        print(f"[整理] {output_root.name}：输出图片已统一为正方形，修正 {normalized} 张已有图片。")
        return copied, True

    def load_manifest(self, manifest_path: Path) -> tuple[ManifestSource, ...]:
        if manifest_path.is_symlink():
            raise ValueError(f"manifest 不能是符号链接：{manifest_path}")
        if not manifest_path.exists():
            return ()
        if not manifest_path.is_file():
            raise ValueError(f"manifest 必须是普通文件：{manifest_path}")

        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{manifest_path}: {error}") from error
        if not isinstance(document, dict) or not isinstance(
            document.get("sources"), list
        ):
            raise ValueError("根节点必须包含 sources 数组")

        project_root = self.root.resolve()
        seen: set[str] = set()
        sources: list[ManifestSource] = []
        for index, entry in enumerate(document["sources"], start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"sources[{index}] 必须是对象")
            raw_path = entry.get("path")
            count = entry.get("count")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError(f"sources[{index}].path 必须是非空字符串")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"sources[{index}].count 必须是正整数")

            relative_path = Path(raw_path)
            if relative_path.is_absolute():
                raise ValueError(f"sources[{index}].path 必须相对项目根目录")
            source_path = (project_root / relative_path).resolve()
            if not source_path.is_relative_to(project_root):
                raise ValueError(f"sources[{index}].path 超出项目根目录")
            if not source_path.is_dir():
                raise ValueError(f"sources[{index}].path 不是目录：{raw_path}")

            path_key = str(source_path).casefold()
            if path_key in seen:
                raise ValueError(f"sources[{index}].path 重复：{raw_path}")
            seen.add(path_key)
            available = len(self.image_files(source_path))
            if count > available:
                raise ValueError(
                    f"sources[{index}] 需要 {count} 张，目录只有 {available} 张：{raw_path}"
                )
            sources.append(ManifestSource(source_path, count))
        return tuple(sources)

    def load_manifest_index(self) -> tuple[str, ...]:
        index_path = self.root / MANIFEST_INDEX_FILENAME
        if index_path.is_symlink():
            raise ValueError(f"输出索引不能是符号链接：{index_path}")
        if not index_path.exists():
            return ()
        if not index_path.is_file():
            raise ValueError(f"输出索引必须是普通文件：{index_path}")
        try:
            names = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{index_path}: {error}") from error
        if not isinstance(names, list) or any(
            not isinstance(name, str)
            or Path(name).name != name
            or Path(name).suffix.casefold() not in IMAGE_SUFFIXES
            for name in names
        ):
            raise ValueError(f"输出索引包含无效文件名：{index_path}")
        return tuple(names)

    def save_manifest_index(self, names: list[str]) -> None:
        index_path = self.root / MANIFEST_INDEX_FILENAME
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.root,
                suffix=".json", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(names, temporary, ensure_ascii=False, indent=2)
                temporary.write("\n")
            os.replace(temporary_path, index_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def copy_group(
        self,
        group: Path,
        images: list[Path],
        output_root: Path,
        used_names: set[str],
    ) -> tuple[int, list[str]]:
        names: list[str] = []
        prefix = f"{group.name}_"
        for source_file in images:
            target_name = (
                source_file.name
                if source_file.name.casefold().startswith(prefix.casefold())
                else f"{prefix}{source_file.name}"
            )
            target_name = self.unique_name(target_name, used_names)
            if not self.copy_image(source_file, output_root / target_name):
                continue
            used_names.add(target_name.casefold())
            names.append(target_name)
        return len(names), names

    @staticmethod
    def path_sort_key(path: Path) -> tuple[str, str]:
        return path.name.casefold(), path.name

    @classmethod
    def image_files(cls, directory: Path) -> list[Path]:
        return sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
            ),
            key=cls.path_sort_key,
        )

    def copy_image(self, source_file: Path, target_file: Path) -> bool:
        if target_file.is_symlink():
            print(f"[跳过] 目标文件是符号链接：{target_file}")
            return False

        try:
            with Image.open(source_file) as opened:
                if opened.width == opened.height:
                    opened.verify()
                    shutil.copy2(source_file, target_file)
                    return True

                cropped = self.center_crop(ImageOps.exif_transpose(opened))
                self.save_image(cropped, target_file, opened.format)
                return True
        except Exception as error:
            print(f"[跳过] 无法整理图片：{source_file}（{error}）")
            return False

    def normalize_output(self, output_root: Path) -> int:
        corrected = 0
        images = sorted(
            (
                path
                for path in output_root.iterdir()
                if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
            ),
            key=lambda path: path.name.casefold(),
        )
        for path in images:
            if path.is_symlink():
                print(f"[跳过] 输出图片是符号链接：{path}")
                continue
            corrected += int(self.crop_file(path))
        return corrected

    def crop_file(self, path: Path) -> bool:
        try:
            with Image.open(path) as opened:
                if opened.width == opened.height:
                    return False

                cropped = self.center_crop(ImageOps.exif_transpose(opened))
                self.save_image(cropped, path, opened.format)
                return True
        except Exception as error:
            print(f"[跳过] 无法中心裁剪图片：{path}（{error}）")
            return False

    @staticmethod
    def center_crop(image):
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return image.crop((left, top, left + side, top + side))

    @staticmethod
    def save_image(image, target: Path, image_format: str | None) -> None:
        image_format = (
            image_format
            or Image.registered_extensions().get(target.suffix.casefold(), "PNG")
        ).upper()
        if image_format == "JPG":
            image_format = "JPEG"
        if image_format == "JPEG" and image.mode not in {"L", "RGB"}:
            image = image.convert("RGB")

        options = {"quality": 95} if image_format in {"JPEG", "WEBP"} else {}
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=target.suffix, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            image.save(temporary_path, format=image_format, **options)
            os.replace(temporary_path, target)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def unique_name(name: str, used_names: set[str]) -> str:
        if name.casefold() not in used_names:
            return name

        path = Path(name)
        number = 2
        while True:
            candidate = f"{path.stem}_{number}{path.suffix}"
            if candidate.casefold() not in used_names:
                print(f"[提示] 目标文件名冲突，改为：{candidate}")
                return candidate
            number += 1


class VisualDuplicateCleaner:
    IMAGE_SIZE = PHASH_IMAGE_SIZE
    DCT_SIZE = PHASH_DCT_SIZE
    MAX_HASH_DISTANCE = 6
    MAX_PIXEL_DISTANCE = 18.0
    COSINES = PHASH_COSINES

    def __init__(self, datasets: tuple[tuple[str, Path], ...]) -> None:
        self.datasets = datasets
        self.output_roots = {root.absolute() for _, root in datasets}

    def run(self) -> bool:
        if Image is None or ImageOps is None:
            print("[错误] 查重需要 Pillow，请先执行：python -m pip install Pillow")
            return False

        records: dict[str, list[ImageRecord]] = {}
        removed = 0
        for label, root in self.datasets:
            records[label], count = self.remove_internal_duplicates(label, root)
            removed += count

        (positive_label, _), (negative_label, _) = self.datasets
        success, count = self.remove_cross_duplicates(
            records[positive_label], records[negative_label]
        )
        removed += count
        print(f"[查重] 完成，共删除 {removed} 张重复图片。")
        return success

    def remove_internal_duplicates(
        self, label: str, root: Path
    ) -> tuple[list[ImageRecord], int]:
        records = self.read_records(root)
        groups = self.duplicate_groups(records)
        survivors: list[ImageRecord] = []
        removed = 0

        print(f"[查重] {label}：检查 {len(records)} 张图片。")
        for group in groups:
            keeper = group[0]
            survivors.append(keeper)
            for duplicate in group[1:]:
                removed += int(
                    self.delete(
                        duplicate.path,
                        f"{label} 内部重复，保留 {keeper.path.name}",
                    )
                )
        return survivors, removed

    def remove_cross_duplicates(
        self, positive: list[ImageRecord], negative: list[ImageRecord]
    ) -> tuple[bool, int]:
        positive_matches = [
            [
                negative_index
                for negative_index, negative_record in enumerate(negative)
                if self.same_visual(record, negative_record)
            ]
            for record in positive
        ]
        negative_matches = [[] for _ in negative]
        for positive_index, matches in enumerate(positive_matches):
            for negative_index in matches:
                negative_matches[negative_index].append(positive_index)

        visited_positive: set[int] = set()
        visited_negative: set[int] = set()
        removed = 0

        for start_positive, matches in enumerate(positive_matches):
            if not matches or start_positive in visited_positive:
                continue

            positive_group: set[int] = set()
            negative_group: set[int] = set()
            queue = [("P", start_positive)]
            visited_positive.add(start_positive)

            while queue:
                side, index = queue.pop()
                if side == "P":
                    positive_group.add(index)
                    for negative_index in positive_matches[index]:
                        if negative_index not in visited_negative:
                            visited_negative.add(negative_index)
                            queue.append(("N", negative_index))
                else:
                    negative_group.add(index)
                    for positive_index in negative_matches[index]:
                        if positive_index not in visited_positive:
                            visited_positive.add(positive_index)
                            queue.append(("P", positive_index))

            choice = self.ask_cross_choice(
                [positive[index] for index in sorted(positive_group)],
                [negative[index] for index in sorted(negative_group)],
            )
            if choice is None:
                return False, removed

            if choice in {"P", "A"}:
                for index in positive_group:
                    removed += int(
                        self.delete(positive[index].path, "正负样本重复，按选择删除正样本")
                    )
            if choice in {"N", "A"}:
                for index in negative_group:
                    removed += int(
                        self.delete(negative[index].path, "正负样本重复，按选择删除负样本")
                    )

        if not any(positive_matches):
            print("[查重] 两个数据集之间未发现重复图片。")
        return True, removed

    def read_records(self, root: Path) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        if not root.is_dir():
            return records

        images = sorted(
            (
                path
                for path in root.iterdir()
                if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
            ),
            key=lambda path: path.name.casefold(),
        )
        for path in images:
            record = self.read_record(path)
            if record is not None:
                records.append(record)
        return records

    def read_record(self, path: Path) -> ImageRecord | None:
        try:
            with Image.open(path) as opened:
                corrected = ImageOps.exif_transpose(opened)
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                rgb = corrected.convert("RGB").resize(
                    (self.IMAGE_SIZE, self.IMAGE_SIZE), resampling
                )
                pixels = rgb.tobytes()
                gray = rgb.convert("L")
                gray_pixels = list(gray.getdata())
        except Exception as error:
            print(f"[跳过] 无法读取图片：{path}（{error}）")
            return None

        rows = [
            gray_pixels[offset : offset + self.IMAGE_SIZE]
            for offset in range(0, self.IMAGE_SIZE * self.IMAGE_SIZE, self.IMAGE_SIZE)
        ]
        row_coefficients = [
            [
                sum(
                    row[sample] * self.COSINES[frequency][sample]
                    for sample in range(self.IMAGE_SIZE)
                )
                for frequency in range(self.DCT_SIZE)
            ]
            for row in rows
        ]
        coefficients = [
            sum(
                row_coefficients[row][horizontal]
                * self.COSINES[vertical][row]
                for row in range(self.IMAGE_SIZE)
            )
            for horizontal in range(self.DCT_SIZE)
            for vertical in range(self.DCT_SIZE)
        ]
        middle = sorted(coefficients[1:])[len(coefficients[1:]) // 2]
        perceptual_hash = 0
        for coefficient in coefficients[1:]:
            perceptual_hash = (perceptual_hash << 1) | int(coefficient > middle)
        return ImageRecord(path, perceptual_hash, pixels)

    def duplicate_groups(
        self, records: list[ImageRecord]
    ) -> list[list[ImageRecord]]:
        groups: list[list[ImageRecord]] = []
        for record in records:
            for group in groups:
                if self.same_visual(record, group[0]):
                    group.append(record)
                    break
            else:
                groups.append([record])
        return groups

    def same_visual(self, left: ImageRecord, right: ImageRecord) -> bool:
        if (left.perceptual_hash ^ right.perceptual_hash).bit_count() > self.MAX_HASH_DISTANCE:
            return False
        difference = sum(
            abs(left_pixel - right_pixel)
            for left_pixel, right_pixel in zip(left.pixels, right.pixels)
        )
        return difference / len(left.pixels) <= self.MAX_PIXEL_DISTANCE

    def ask_cross_choice(
        self, positive: list[ImageRecord], negative: list[ImageRecord]
    ) -> str | None:
        print("\n[跨数据集重复] 检测到视觉内容相同：")
        for record in positive:
            print(f"  正样本：{record.path}")
        for record in negative:
            print(f"  负样本：{record.path}")

        while True:
            try:
                choice = input("删除正样本(P)、负样本(N)或两边都删除(A)：").strip().upper()
            except EOFError:
                print("[错误] 未获得有效选择，已停止查重。")
                return None
            if choice in {"P", "N", "A"}:
                return choice
            print("[提示] 请输入 P、N 或 A。")

    def delete(self, path: Path, reason: str) -> bool:
        if (
            path.parent.absolute() not in self.output_roots
            or path.parent.is_symlink()
            or path.is_symlink()
            or not path.is_file()
        ):
            return False
        path.unlink()
        print(f"[删除] {path}（{reason}）")
        return True


if __name__ == "__main__":
    sys.exit(DatasetCombiner(Path(__file__).resolve().parent).run())
