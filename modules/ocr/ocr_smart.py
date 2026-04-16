from copy import deepcopy
from typing import Dict, List

import numpy as np

from .base import OCR, OCRBase, TextBlock, register_OCR
from ..base import soft_empty_cache
from utils.config import pcfg


SMART_OCR_OPTIONS = [
    'mit48px',
    'mit48px_ctc',
    'manga_ocr',
    'windows_ocr',
    'one_ocr',
    'paddle_ocr',
    'PaddleOCRVLManga',
    'paddle_vl',
    'llm_ocr',
    'google_vision',
    'bing_ocr',
    'stariver_ocr',
    'google_lens_exp',
    'none_ocr',
]


@register_OCR('smart_ocr')
class SmartOCR(OCRBase):
    params = {
        'vertical_ocr': {
            'type': 'selector',
            'options': SMART_OCR_OPTIONS,
            'value': 'mit48px',
            'description': 'OCR backend for vertical manga text blocks.',
        },
        'horizontal_ocr': {
            'type': 'selector',
            'options': SMART_OCR_OPTIONS,
            'value': 'windows_ocr',
            'description': 'OCR backend for horizontal text blocks.',
        },
        'fallback_ocr': {
            'type': 'selector',
            'options': SMART_OCR_OPTIONS,
            'value': 'manga_ocr',
            'description': 'Retry backend used when the primary result is empty or the primary backend is unavailable.',
        },
        'page_ocr': {
            'type': 'selector',
            'options': SMART_OCR_OPTIONS,
            'value': 'horizontal_ocr',
            'description': 'OCR backend for full-image OCR requests. Use "horizontal_ocr" to reuse the horizontal route.',
        },
        'retry_on_empty': {
            'type': 'checkbox',
            'value': True,
            'description': 'Retry once with fallback OCR when the first result is empty.',
        },
        'vertical_aspect_ratio_threshold': {
            'value': 1.25,
            'description': 'When detector orientation is missing, treat height / width above this threshold as vertical.',
        },
        'description': 'Route each text block to different OCR backends based on orientation.',
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.engines: Dict[str, OCRBase] = {}
        self.failed_engines = set()

    def unload_model(self, empty_cache=False):
        model_deleted = False
        for engine in self.engines.values():
            try:
                model_deleted = engine.unload_model(empty_cache=False) or model_deleted
            except Exception:
                continue
        self.engines.clear()
        self.failed_engines.clear()
        if empty_cache and model_deleted:
            soft_empty_cache()
        return model_deleted

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in {'vertical_ocr', 'horizontal_ocr', 'fallback_ocr', 'page_ocr'}:
            self.unload_model(empty_cache=False)

    def _params_for_engine(self, engine_name: str) -> dict:
        cfg_params = pcfg.module.ocr_params.get(engine_name, {})
        if cfg_params is None:
            return {}
        return deepcopy(cfg_params)

    def _resolve_engine_name(self, preferred_name: str) -> str:
        if preferred_name == 'horizontal_ocr':
            preferred_name = self.get_param_value('horizontal_ocr')
        if not preferred_name or preferred_name == self.name:
            return 'none_ocr'
        if preferred_name not in OCR.module_dict:
            return 'none_ocr'
        if preferred_name in self.failed_engines:
            return 'none_ocr'
        return preferred_name

    def _get_engine(self, preferred_name: str) -> OCRBase:
        engine_name = self._resolve_engine_name(preferred_name)
        if engine_name == 'none_ocr':
            return OCR.get('none_ocr')()

        if engine_name in self.engines:
            return self.engines[engine_name]

        engine_cls = OCR.get(engine_name)
        engine = engine_cls(**self._params_for_engine(engine_name))
        self.engines[engine_name] = engine
        return engine

    def _mark_engine_failed(self, engine_name: str, exc: Exception):
        if engine_name in {'', 'none_ocr'} or engine_name in self.failed_engines:
            return
        self.failed_engines.add(engine_name)
        self.engines.pop(engine_name, None)
        self.logger.warning(
            f'Smart OCR disabled backend {engine_name} for this run after failure: {exc}'
        )

    def _is_vertical_block(self, blk: TextBlock) -> bool:
        if blk.src_is_vertical is not None:
            return bool(blk.src_is_vertical)

        x1, y1, x2, y2 = blk.xyxy
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        return (height / width) >= float(self.get_param_value('vertical_aspect_ratio_threshold'))

    def _text_is_empty(self, text) -> bool:
        if text is None:
            return True
        if isinstance(text, str):
            return text.strip() == ''
        if isinstance(text, list):
            return all(self._text_is_empty(item) for item in text)
        return False

    def _run_engine_on_block(self, preferred_name: str, img: np.ndarray, blk: TextBlock) -> bool:
        engine_name = self._resolve_engine_name(preferred_name)
        if engine_name == 'none_ocr':
            blk.text = []
            return False

        try:
            engine = self._get_engine(engine_name)
            engine.run_ocr(img, [blk])
            return True
        except Exception as exc:
            self.logger.warning(f'Smart OCR backend {engine_name} failed on block {blk.xyxy}: {exc}')
            self._mark_engine_failed(engine_name, exc)
            blk.text = []
            return False

    def _route_engine_name(self, blk: TextBlock) -> str:
        if self._is_vertical_block(blk):
            return self.get_param_value('vertical_ocr')
        return self.get_param_value('horizontal_ocr')

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
        fallback_name = self.get_param_value('fallback_ocr')
        retry_on_empty = bool(self.get_param_value('retry_on_empty'))

        for blk in blk_list:
            primary_name = self._route_engine_name(blk)
            succeeded = self._run_engine_on_block(primary_name, img, blk)
            if succeeded and (not retry_on_empty or not self._text_is_empty(blk.get_text())):
                continue

            if not retry_on_empty:
                continue

            resolved_primary = self._resolve_engine_name(primary_name)
            resolved_fallback = self._resolve_engine_name(fallback_name)
            if resolved_fallback in {'none_ocr', resolved_primary}:
                continue

            self._run_engine_on_block(resolved_fallback, img, blk)

    def ocr_img(self, img: np.ndarray) -> str:
        page_engine = self._get_engine(self.get_param_value('page_ocr'))
        if not page_engine.all_model_loaded():
            page_engine.load_model()
        return page_engine.ocr_img(img)
