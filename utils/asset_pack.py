from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from utils import shared
except Exception:  # pragma: no cover - allows build scripts to import before app bootstrap
    shared = None


PACKS_DIRNAME = "packs"
OPTIONAL_PACKS_DIRNAME = "optional-packs"
SMART_OCR_PARAM_KEYS = (
    "vertical_ocr",
    "horizontal_ocr",
    "fallback_ocr",
    "page_ocr",
)


@dataclass(frozen=True)
class PackEntry:
    target_relpath: str
    source_candidates: tuple[str, ...]
    is_dir: bool = False


@dataclass(frozen=True)
class PackSpec:
    name: str
    archive_name: str
    description: str
    entries: tuple[PackEntry, ...]
    bundled_by_default: bool = False
    auto_extract_on_startup: bool = False


PACK_SPECS = {
    "core-models": PackSpec(
        name="core-models",
        archive_name="core-models.zip",
        description="默认文本检测与 OCR 核心模型",
        bundled_by_default=True,
        auto_extract_on_startup=True,
        entries=(
            PackEntry(
                "data/models/comictextdetector.pt",
                ("data/models/comictextdetector.pt",),
            ),
            PackEntry(
                "data/models/comictextdetector.pt.onnx",
                ("data/models/comictextdetector.pt.onnx",),
            ),
            PackEntry(
                "data/models/ocr_ar_48px.ckpt",
                ("data/models/ocr_ar_48px.ckpt",),
            ),
            PackEntry(
                "data/alphabet-all-v7.txt",
                ("data/alphabet-all-v7.txt",),
            ),
        ),
    ),
    "advanced-ocr": PackSpec(
        name="advanced-ocr",
        archive_name="advanced-ocr.zip",
        description="高级 OCR 模型包",
        bundled_by_default=False,
        auto_extract_on_startup=False,
        entries=(
            PackEntry(
                "data/models/mit32px_ocr.ckpt",
                ("data/models/mit32px_ocr.ckpt",),
            ),
            PackEntry(
                "data/models/mit48pxctc_ocr.ckpt",
                ("data/models/mit48pxctc_ocr.ckpt",),
            ),
            PackEntry(
                "data/alphabet-all-v5.txt",
                ("data/alphabet-all-v5.txt",),
            ),
            PackEntry(
                "data/models/manga-ocr-base",
                ("data/models/manga-ocr-base",),
                is_dir=True,
            ),
            PackEntry(
                "data/models/PaddleOCR-VL-For-Manga",
                ("data/models/PaddleOCR-VL-For-Manga",),
                is_dir=True,
            ),
        ),
    ),
}

MODULE_PACK_MAP = {
    "ctd": ("core-models",),
    "mit48px": ("core-models",),
    "mit32px": ("advanced-ocr",),
    "mit48px_ctc": ("advanced-ocr",),
    "manga_ocr": ("advanced-ocr",),
    "PaddleOCRVLManga": ("advanced-ocr",),
}


def default_program_root() -> Path:
    if shared is not None and getattr(shared, "PROGRAM_PATH", None):
        return Path(shared.PROGRAM_PATH)
    return Path(__file__).resolve().parents[1]


def get_pack_spec(pack_name: str) -> PackSpec:
    if pack_name not in PACK_SPECS:
        raise KeyError(f"Unknown asset pack: {pack_name}")
    return PACK_SPECS[pack_name]


def normalize_roots(root_paths: Iterable[Path | str] | Path | str | None) -> list[Path]:
    if root_paths is None:
        return [default_program_root()]
    if isinstance(root_paths, (str, Path)):
        return [Path(root_paths)]
    return [Path(root_path) for root_path in root_paths]


def pack_path(pack_name: str, root: Path | str | None = None) -> Path:
    spec = get_pack_spec(pack_name)
    base_root = Path(root) if root is not None else default_program_root()
    return base_root / PACKS_DIRNAME / spec.archive_name


def pack_search_paths(pack_name: str, root: Path | str | None = None) -> list[Path]:
    spec = get_pack_spec(pack_name)
    base_root = Path(root) if root is not None else default_program_root()
    archive_name = spec.archive_name
    search_paths = [
        base_root / PACKS_DIRNAME / archive_name,
        base_root / OPTIONAL_PACKS_DIRNAME / archive_name,
        base_root.parent / OPTIONAL_PACKS_DIRNAME / archive_name,
        base_root.parent / PACKS_DIRNAME / archive_name,
        base_root / archive_name,
        base_root.parent / archive_name,
    ]
    unique_paths = []
    seen = set()
    for path in search_paths:
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        unique_paths.append(path)
    return unique_paths


def locate_pack_path(pack_name: str, root: Path | str | None = None) -> Path | None:
    for candidate_path in pack_search_paths(pack_name, root):
        if candidate_path.exists():
            return candidate_path
    return None


def pack_exists(pack_name: str, root: Path | str | None = None) -> bool:
    return locate_pack_path(pack_name, root) is not None


def module_pack_names(module_name: str) -> tuple[str, ...]:
    return MODULE_PACK_MAP.get(module_name, ())


def _entry_target_path(entry: PackEntry, root: Path) -> Path:
    return root / Path(entry.target_relpath)


def required_paths_exist(pack_name: str, root: Path | str | None = None) -> bool:
    base_root = Path(root) if root is not None else default_program_root()
    spec = get_pack_spec(pack_name)
    for entry in spec.entries:
        if not _entry_target_path(entry, base_root).exists():
            return False
    return True


def pack_archive_is_complete(pack_name: str, root: Path | str | None = None) -> bool:
    archive_path = locate_pack_path(pack_name, root)
    if archive_path is None:
        return False

    spec = get_pack_spec(pack_name)
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive_names = {
                Path(name).as_posix().rstrip("/")
                for name in archive.namelist()
                if name and not name.endswith("/")
            }
    except zipfile.BadZipFile:
        return False

    for entry in spec.entries:
        target_path = Path(entry.target_relpath).as_posix().rstrip("/")
        if entry.is_dir:
            prefix = f"{target_path}/"
            if not any(name.startswith(prefix) for name in archive_names):
                return False
            continue
        if target_path not in archive_names:
            return False
    return True


def pack_ready(pack_name: str, root: Path | str | None = None) -> bool:
    return required_paths_exist(pack_name, root) or pack_archive_is_complete(pack_name, root)


def missing_packs_for_module(module_name: str, root: Path | str | None = None) -> list[str]:
    missing = []
    for pack_name in module_pack_names(module_name):
        if not required_paths_exist(pack_name, root):
            missing.append(pack_name)
    return missing


def ensure_pack_extracted(pack_name: str, root: Path | str | None = None) -> bool:
    base_root = Path(root) if root is not None else default_program_root()
    if required_paths_exist(pack_name, base_root):
        return False

    archive_path = locate_pack_path(pack_name, base_root)
    if archive_path is None:
        search_detail = "\n".join(
            f"- {candidate_path}" for candidate_path in pack_search_paths(pack_name, base_root)
        )
        raise FileNotFoundError(
            f"Asset pack archive not found for {pack_name}. Searched:\n{search_detail}"
        )

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(base_root)

    if not required_paths_exist(pack_name, base_root):
        raise FileNotFoundError(
            f"Asset pack archive for {pack_name} is incomplete: {archive_path}"
        )
    return True


def _resolve_entry_source(entry: PackEntry, roots: list[Path]) -> Path | None:
    for root in roots:
        for candidate in entry.source_candidates:
            source_path = root / Path(candidate)
            if source_path.exists():
                return source_path
    return None


def iter_pack_file_records(
    pack_name: str,
    root_paths: Iterable[Path | str] | Path | str | None,
    *,
    allow_missing: bool = False,
) -> list[tuple[Path, Path]]:
    roots = normalize_roots(root_paths)
    spec = get_pack_spec(pack_name)
    records: list[tuple[Path, Path]] = []
    missing_targets: list[str] = []

    for entry in spec.entries:
        source_path = _resolve_entry_source(entry, roots)
        if source_path is None:
            missing_targets.append(entry.target_relpath)
            continue

        if entry.is_dir:
            for child in sorted(source_path.rglob("*")):
                if not child.is_file():
                    continue
                archive_relpath = Path(entry.target_relpath) / child.relative_to(source_path)
                records.append((child, archive_relpath))
            continue

        records.append((source_path, Path(entry.target_relpath)))

    if missing_targets and not allow_missing:
        detail = "\n".join(f"- {target}" for target in missing_targets)
        raise FileNotFoundError(
            f"Missing source files for asset pack {pack_name}:\n{detail}"
        )

    return records


def compute_pack_signature(
    pack_name: str,
    root_paths: Iterable[Path | str] | Path | str | None,
    *,
    allow_missing: bool = False,
) -> str:
    digest = hashlib.sha256()
    payload = {
        "pack_name": pack_name,
        "records": [],
    }

    for source_path, archive_relpath in iter_pack_file_records(
        pack_name,
        root_paths,
        allow_missing=allow_missing,
    ):
        stat = source_path.stat()
        payload["records"].append(
            {
                "archive_relpath": archive_relpath.as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:16]
