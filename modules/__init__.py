from .ocr import OCR, OCRBase
from .textdetector import TEXTDETECTORS, TextDetectorBase
from utils.registry import Registry
from .base import BaseModule
from .base import DEFAULT_DEVICE, GPUINTENSIVE_SET, LOGGER, merge_config_module_params, \
    init_module_registries, init_textdetector_registries, init_inpainter_registries, init_ocr_registries
from pathlib import Path

from utils.asset_pack import module_pack_names, pack_exists

INPAINTERS = Registry('inpainters')


class InpainterBase(BaseModule):
    check_need_inpaint = False


GET_VALID_TEXTDETECTORS = lambda : list(TEXTDETECTORS.module_dict.keys())
GET_VALID_INPAINTERS = lambda : list(INPAINTERS.module_dict.keys())
GET_VALID_OCR = lambda : list(OCR.module_dict.keys())


def _get_module_class(registry: Registry, module_name: str):
    if not module_name:
        return None
    return registry.module_dict.get(module_name)


def _tracked_download_targets_ready(module_class) -> bool:
    download_specs = getattr(module_class, 'download_file_list', None) or []
    has_tracked_targets = False

    for download_spec in download_specs:
        save_files = download_spec.get('save_files')
        if save_files:
            has_tracked_targets = True
            if any(not Path(file_path).exists() for file_path in save_files):
                return False
            continue

        files = download_spec.get('files') or []
        save_dir = download_spec.get('save_dir')
        if save_dir and files:
            has_tracked_targets = True
            base_dir = Path(save_dir)
            if any(not (base_dir / file_name).exists() for file_name in files):
                return False
            continue

        if files:
            has_tracked_targets = True
            if any(not Path(file_path).exists() for file_path in files):
                return False

    return has_tracked_targets


def _special_module_ready(module_name: str):
    if module_name == 'ysgyolo':
        try:
            from .textdetector.detector_ysg import CKPT_LIST, update_ckpt_list

            update_ckpt_list()
            return any(Path(model_path).exists() for model_path in CKPT_LIST)
        except Exception:
            return False

    if module_name == 'one_ocr':
        try:
            from .ocr.ocr_oneocr import DLL_NAME, MODEL_NAME, ONE_OCR_PATH

            model_dir = Path(ONE_OCR_PATH)
            return (model_dir / DLL_NAME).exists() and (model_dir / MODEL_NAME).exists()
        except Exception:
            return False

    if module_name == 'paddle_ocr':
        paddle_ocr_dir = Path('data/models/paddle-ocr')
        if not paddle_ocr_dir.exists():
            return False
        return any(child.is_file() for child in paddle_ocr_dir.rglob('*'))

    return None


def _module_is_available(registry: Registry, module_name: str) -> bool:
    module_class = _get_module_class(registry, module_name)
    if module_class is None:
        return False

    special_ready = _special_module_ready(module_name)
    if special_ready is not None:
        return special_ready

    if _tracked_download_targets_ready(module_class):
        return True

    pack_names = module_pack_names(module_name)
    if len(pack_names) > 0:
        return all(pack_exists(pack_name) for pack_name in pack_names)

    download_specs = getattr(module_class, 'download_file_list', None) or []
    if len(download_specs) > 0:
        return False

    return module_name in {'smart_ocr', 'none_ocr'}


def get_available_textdetectors() -> list[str]:
    return [name for name in TEXTDETECTORS.module_dict.keys() if _module_is_available(TEXTDETECTORS, name)]


def get_available_ocr() -> list[str]:
    return [name for name in OCR.module_dict.keys() if _module_is_available(OCR, name)]


GET_AVAILABLE_TEXTDETECTORS = get_available_textdetectors
GET_AVAILABLE_OCR = get_available_ocr


MODULETYPE_TO_REGISTRIES = {
    'textdetector': TEXTDETECTORS,
    'ocr': OCR,
    'inpainter': INPAINTERS,
}

# TODO: use manga-image-translator as backend...
