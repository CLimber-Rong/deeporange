# -*- coding: utf-8 -*-
"""Use the Wikimedia Commons API to collect openly licensed training images.

The script asks for one search term, then saves up to 100 validated images in
a folder with that term under the workspace root. Network requests are made
sequentially and are deliberately throttled to reduce load on each service.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from html import unescape
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

try:
    import requests
    from PIL import Image
except ImportError:
    print("缺少依赖，请先执行：python -m pip install requests pillow")
    raise SystemExit(1)


API_URL = "https://commons.wikimedia.org/w/api.php"
TARGET_IMAGE_COUNT = 100
API_PAGE_SIZE = 50
CANDIDATES_PER_TARGET_IMAGE = 3
MAX_API_PAGES = 12
MAX_CANDIDATES = API_PAGE_SIZE * MAX_API_PAGES
MAX_DOWNLOADS_PER_HOST = 50
MIN_IMAGE_WIDTH = 150
MIN_IMAGE_HEIGHT = 150
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_REDIRECTS = 5
REQUEST_TIMEOUT = (15, 45)
MIN_REQUEST_DELAY = 1.0
MAX_REQUEST_DELAY = 3.0

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36 "
    "FruitDatasetCollector/1.0"
)

IMAGE_HEADERS = {
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://commons.wikimedia.org/",
}

API_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://commons.wikimedia.org/",
}

ROBOTS_HEADERS = {
    "Accept": "text/plain,*/*;q=0.1",
}

SUPPORTED_FORMATS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}

INVALID_FOLDER_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
RESERVED_WINDOWS_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class CrawlStopped(RuntimeError):
    """Raised when a service asks the crawler to stop, such as with 403/429."""

    def __init__(self, url: str, status_code: int):
        super().__init__(f"HTTP {status_code}: {url}")
        self.url = url
        self.status_code = status_code


class CrawlError(RuntimeError):
    """Raised for an unrecoverable API or configuration error."""


class RequestGate:
    """A single sequential request gate with a random 1-3 second delay."""

    def __init__(self) -> None:
        self._last_request_at = 0.0

    def wait(self) -> None:
        if self._last_request_at:
            elapsed = time.monotonic() - self._last_request_at
            delay = random.uniform(MIN_REQUEST_DELAY, MAX_REQUEST_DELAY)
            if elapsed < delay:
                time.sleep(delay - elapsed)
        self._last_request_at = time.monotonic()


class HttpClient:
    """Wrap requests so every network request uses the same throttle."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )
        self.gate = RequestGate()

    def get(self, url: str, **kwargs: object) -> requests.Response:
        self.gate.wait()
        response = self.session.get(url, **kwargs)
        if response.status_code in (403, 429):
            response.close()
            raise CrawlStopped(url, response.status_code)
        return response


class RobotsPolicy:
    """Cache and apply robots.txt rules per origin before each download."""

    def __init__(self, client: HttpClient) -> None:
        self.client = client
        self._parsers: Dict[str, RobotFileParser] = {}
        self._blocked: Dict[str, str] = {}

    @staticmethod
    def origin(url: str) -> Optional[str]:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"

    def block_origin(self, url: str, reason: str) -> None:
        origin = self.origin(url)
        if origin:
            self._blocked[origin] = reason

    def allowed(self, url: str) -> bool:
        origin = self.origin(url)
        if origin is None:
            return False
        if origin in self._blocked:
            return False

        if origin not in self._parsers:
            robots_url = f"{origin}/robots.txt"
            try:
                with self.client.get(
                    robots_url,
                    headers=ROBOTS_HEADERS,
                    timeout=REQUEST_TIMEOUT,
                    allow_redirects=True,
                ) as response:
                    if response.status_code == 404:
                        parser = RobotFileParser()
                        parser.parse([])
                    elif response.status_code != 200:
                        self._blocked[origin] = (
                            f"robots.txt 返回 HTTP {response.status_code}"
                        )
                        return False
                    else:
                        parser = RobotFileParser()
                        parser.parse(response.text.splitlines())
            except CrawlStopped:
                self._blocked[origin] = "robots.txt 返回 403 或 429"
                raise
            except requests.RequestException as exc:
                self._blocked[origin] = f"robots.txt 无法访问: {exc}"
                return False

            self._parsers[origin] = parser

        return self._parsers[origin].can_fetch(USER_AGENT, url)


@dataclass
class ValidatedImage:
    data: bytes
    extension: str
    width: int
    height: int


def normalize_folder_name(search_term: str) -> str:
    """Make a safe Windows path component while preserving normal search terms."""

    name = INVALID_FOLDER_CHARS.sub("_", search_term.strip())
    name = name.rstrip(" .")
    if not name or name in {".", ".."}:
        raise ValueError("搜索词不能转换为有效的文件夹名称")

    base_name = name.split(".", 1)[0].upper()
    if base_name in RESERVED_WINDOWS_NAMES:
        name = f"_{name}"

    if len(name) > 120:
        name = name[:120].rstrip(" .")

    if name != search_term.strip():
        print(f"提示：搜索词包含路径不允许的字符，文件夹名称将使用：{name}")
    return name


def image_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_existing_hashes(output_dir: Path) -> Set[str]:
    hashes: Set[str] = set()
    for path in output_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        try:
            digest = hashlib.sha256()
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes.add(digest.hexdigest())
        except OSError:
            continue
    return hashes


def validate_image(data: bytes) -> Optional[ValidatedImage]:
    """Verify image bytes and keep only formats accepted by the target platform."""

    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()

        with Image.open(BytesIO(data)) as image:
            image_format = (image.format or "").upper()
            width, height = image.size
            if image_format not in SUPPORTED_FORMATS:
                return None
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                return None
            if width * height > MAX_IMAGE_PIXELS:
                return None
            image.load()
    except Exception:
        return None

    return ValidatedImage(
        data=data,
        extension=SUPPORTED_FORMATS[image_format],
        width=width,
        height=height,
    )


def collect_image_bytes(response: requests.Response) -> Optional[bytes]:
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
    if content_type.startswith("text/") or content_type in {
        "application/json",
        "application/xml",
    }:
        return None

    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            if int(content_length) > MAX_IMAGE_BYTES:
                return None
        except ValueError:
            pass

    data = bytearray()
    try:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            data.extend(chunk)
            if len(data) > MAX_IMAGE_BYTES:
                return None
    except requests.RequestException:
        return None
    return bytes(data)


def download_image(
    client: HttpClient,
    robots: RobotsPolicy,
    image_url: str,
) -> Optional[ValidatedImage]:
    """Download one image while checking robots.txt at every redirect target."""

    current_url = image_url
    try:
        for _ in range(MAX_REDIRECTS + 1):
            if not robots.allowed(current_url):
                return None

            with client.get(
                current_url,
                headers=IMAGE_HEADERS,
                timeout=REQUEST_TIMEOUT,
                stream=True,
                allow_redirects=False,
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location:
                        return None
                    current_url = urljoin(current_url, location)
                    continue
                if response.status_code != 200:
                    return None
                data = collect_image_bytes(response)
                if data is None:
                    return None
                return validate_image(data)
    except CrawlStopped as exc:
        robots.block_origin(current_url, f"HTTP {exc.status_code}")
        print(f"跳过站点 {robots.origin(current_url)}：收到 HTTP {exc.status_code}")
        return None
    except requests.RequestException:
        return None
    return None


def fetch_wikimedia_page(
    client: HttpClient,
    robots: RobotsPolicy,
    search_term: str,
    page: int,
    continuation: Optional[Dict[str, object]],
) -> Tuple[List[str], Optional[Dict[str, object]]]:
    # Wikimedia exposes this endpoint as an official API. Its generic robots
    # rules may disallow /w/ for crawlers even when the API is intended for use.
    params: Dict[str, object] = {
        "action": "query",
        "generator": "search",
        "gsrsearch": search_term,
        "gsrnamespace": 6,
        "gsrlimit": API_PAGE_SIZE,
        "prop": "imageinfo",
        "iiprop": "url|mime|size|width|height",
        "iiurlwidth": 1200,
        "format": "json",
        "formatversion": 2,
    }
    if continuation:
        params.update(continuation)

    try:
        with client.get(
            API_URL,
            params=params,
            headers=API_HEADERS,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            if response.status_code != 200:
                raise CrawlError(
                    f"Wikimedia Commons API 返回 HTTP {response.status_code}"
                )
            payload = response.json()
    except CrawlStopped:
        raise
    except (requests.RequestException, ValueError) as exc:
        raise CrawlError(f"Wikimedia Commons API 请求失败：{exc}") from exc

    urls: List[str] = []
    query = payload.get("query", {})
    pages = query.get("pages", []) if isinstance(query, dict) else []
    if isinstance(pages, dict):
        pages = list(pages.values())

    for result in pages:
        if not isinstance(result, dict):
            continue
        image_info = result.get("imageinfo", [])
        if not isinstance(image_info, list) or not image_info:
            continue
        info = image_info[0]
        if not isinstance(info, dict):
            continue
        mime = str(info.get("mime", "")).lower()
        if not mime.startswith("image/"):
            continue
        image_url = info.get("thumburl") or info.get("url")
        if not isinstance(image_url, str):
            continue
        image_url = unescape(image_url).strip()
        parsed = urlsplit(image_url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            urls.append(image_url)

    next_continuation = payload.get("continue")
    if isinstance(next_continuation, dict) and next_continuation:
        return urls, next_continuation
    return urls, None


def collect_candidates(
    client: HttpClient,
    robots: RobotsPolicy,
    search_term: str,
    target_count: int,
) -> List[str]:
    candidates: List[str] = []
    seen: Set[str] = set()
    continuation: Optional[Dict[str, object]] = None
    candidate_limit = min(
        max(API_PAGE_SIZE, target_count * CANDIDATES_PER_TARGET_IMAGE),
        MAX_CANDIDATES,
    )
    page_limit = min(
        MAX_API_PAGES,
        (candidate_limit + API_PAGE_SIZE - 1) // API_PAGE_SIZE,
    )

    for page in range(1, page_limit + 1):
        urls, continuation = fetch_wikimedia_page(
            client,
            robots,
            search_term,
            page,
            continuation,
        )
        added = 0
        for image_url in urls:
            if image_url not in seen:
                seen.add(image_url)
                candidates.append(image_url)
                added += 1
        print(f"已获取第 {page} 页候选结果，新增 {added} 个，累计 {len(candidates)} 个")

        if len(candidates) >= candidate_limit or continuation is None:
            break
        if not added:
            break

    return candidates[:candidate_limit]


def save_image(
    output_dir: Path,
    image: ValidatedImage,
    index: int,
    filename_prefix: str,
) -> Path:
    while True:
        output_path = output_dir / f"{filename_prefix}_image_{index:03d}{image.extension}"
        if not output_path.exists():
            break
        index += 1

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".image_",
            suffix=".part",
            dir=output_dir,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(image.data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    except OSError:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass
        raise
    return output_path


def read_search_terms() -> List[Tuple[str, str, int]]:
    search_terms: List[Tuple[str, str]] = []
    print("请输入搜索词，每输入一个后按回车继续；直接回车结束输入。")

    while True:
        search_term = input(
            f"请输入第 {len(search_terms) + 1} 个搜索词："
        ).strip()
        if not search_term:
            break

        try:
            folder_name = normalize_folder_name(search_term)
        except ValueError as exc:
            print(f"错误：{exc}，请重新输入。")
            continue
        search_terms.append((search_term, folder_name))

    configured_terms: List[Tuple[str, str, int]] = []
    print("搜索词输入完毕，现在设置每个搜索词的预期图片数量。")
    for search_term, folder_name in search_terms:
        while True:
            count_text = input(
                f"“{search_term}”预期搜索多少张图（直接回车默认 {TARGET_IMAGE_COUNT}）："
            ).strip()
            if not count_text:
                target_count = TARGET_IMAGE_COUNT
                break
            try:
                target_count = int(count_text)
            except ValueError:
                print("请输入正整数，或直接回车使用默认值。")
                continue
            if target_count <= 0:
                print("图片数量必须是大于 0 的整数。")
                continue
            break
        configured_terms.append((search_term, folder_name, target_count))

    return configured_terms


def crawl_search_term(
    client: HttpClient,
    robots: RobotsPolicy,
    crawler_dir: Path,
    search_term: str,
    folder_name: str,
    target_count: int,
) -> bool:
    output_dir = crawler_dir / folder_name
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"无法创建输出文件夹 {output_dir}：{exc}")
        return False

    print(f"输出文件夹：{output_dir}")
    try:
        candidates = collect_candidates(client, robots, search_term, target_count)
    except CrawlStopped:
        raise
    except CrawlError as exc:
        print(f"获取“{search_term}”的候选图片失败：{exc}")
        return False

    if not candidates:
        print(f"没有找到“{search_term}”的可用图片候选结果。")
        return False

    existing_hashes = load_existing_hashes(output_dir)
    host_attempts: Counter[str] = Counter()
    downloaded = 0
    next_index = 1

    for number, image_url in enumerate(candidates, start=1):
        if downloaded >= target_count:
            break

        origin = robots.origin(image_url)
        if origin is None:
            continue
        if host_attempts[origin] >= MAX_DOWNLOADS_PER_HOST:
            continue
        host_attempts[origin] += 1

        image = download_image(client, robots, image_url)
        if image is None:
            print(f"[{number}/{len(candidates)}] 跳过：图片损坏、尺寸过小或格式不支持")
            continue

        digest = image_sha256(image.data)
        if digest in existing_hashes:
            print(f"[{number}/{len(candidates)}] 跳过：与已有图片重复")
            continue

        try:
            saved_path = save_image(output_dir, image, next_index, folder_name)
        except OSError as exc:
            print(f"保存图片失败：{exc}")
            return False

        next_index += 1
        existing_hashes.add(digest)
        downloaded += 1
        print(
            f"[{downloaded}/{target_count}] 已保存 {saved_path.name} "
            f"({image.width}x{image.height})"
        )

    print(f"“{search_term}”本次新增图片：{downloaded} 张，保存位置：{output_dir}")
    if downloaded < target_count:
        print(
            f"提示：候选图片不足或部分图片未通过校验，未达到 {target_count} 张；"
            "可以稍后用其他搜索词再次运行。"
        )
    return downloaded > 0


def main() -> int:
    search_terms = read_search_terms()
    if not search_terms:
        print("没有输入搜索词，程序结束。")
        return 0

    crawler_dir = Path(__file__).resolve().parent
    print(f"共收到 {len(search_terms)} 个搜索词，将按顺序开始爬取。")
    print("数据源：Wikimedia Commons 官方 API（开放授权媒体平台）")
    print("请求为单线程执行，每次请求之间随机等待 1-3 秒。")

    client = HttpClient()
    robots = RobotsPolicy(client)
    completed = 0
    try:
        for index, (search_term, folder_name, target_count) in enumerate(
            search_terms,
            start=1,
        ):
            print(f"\n===== [{index}/{len(search_terms)}] 开始处理：{search_term} =====")
            print(f"本次预期图片数量：{target_count}")
            if crawl_search_term(
                client,
                robots,
                crawler_dir,
                search_term,
                folder_name,
                target_count,
            ):
                completed += 1
    except CrawlStopped as exc:
        print(f"已停止：服务返回 HTTP {exc.status_code}，不会继续请求该服务。")
        return 1

    print(f"批量处理结束：{completed}/{len(search_terms)} 个搜索词完成。")
    return 0 if completed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已由用户中断，已保存的图片会保留。")
        raise SystemExit(130)
