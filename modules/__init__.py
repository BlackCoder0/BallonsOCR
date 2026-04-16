from .ocr import OCR, OCRBase
from .textdetector import TEXTDETECTORS, TextDetectorBase
from .inpaint import INPAINTERS, InpainterBase
from .base import DEFAULT_DEVICE, GPUINTENSIVE_SET, LOGGER, merge_config_module_params, \
    init_module_registries, init_textdetector_registries, init_inpainter_registries, init_ocr_registries

GET_VALID_TEXTDETECTORS = lambda : list(TEXTDETECTORS.module_dict.keys())
GET_VALID_INPAINTERS = lambda : list(INPAINTERS.module_dict.keys())
GET_VALID_OCR = lambda : list(OCR.module_dict.keys())


MODULETYPE_TO_REGISTRIES = {
    'textdetector': TEXTDETECTORS,
    'ocr': OCR,
    'inpainter': INPAINTERS,
}

# TODO: use manga-image-translator as backend...
