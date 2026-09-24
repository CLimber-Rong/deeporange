# -*- coding: utf-8 -*-
"""Copy images from the two source trees into flat, upload-ready folders."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi
from pathlib import Path
import shutil
import sys

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


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    perceptual_hash: int
    pixels: bytes


class DatasetCombiner:
    CATEGORIES = (
        ("橙子数据源", "橙子"),
        ("非橙子数据源", "非橙子"),
    )

    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self) -> int:
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

        output_root.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()
        copied = 0
        groups = sorted(
            (path for path in source_root.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        )

        for group in groups:
            group_copied = 0
            prefix = f"{group.name}_"
            images = sorted(
                (
                    path
                    for path in group.iterdir()
                    if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
                ),
                key=lambda path: path.name.casefold(),
            )

            for source_file in images:
                target_name = (
                    source_file.name
                    if source_file.name.casefold().startswith(prefix.casefold())
                    else f"{prefix}{source_file.name}"
                )
                target_name = self.unique_name(target_name, used_names)
                shutil.copy2(source_file, output_root / target_name)
                used_names.add(target_name.casefold())
                copied += 1
                group_copied += 1

            print(f"[{source_root.name}] {group.name}: 复制 {group_copied} 张")

        if not groups:
            print(f"[{source_root.name}] 未找到来源子目录。")
        return copied, True

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
