import time
from typing import Union, List, Callable
import os.path as osp
from pathlib import Path

import numpy as np
from qtpy.QtCore import QThread, Signal, QObject, QTimer
from qtpy.QtWidgets import QFileDialog, QMessageBox, QWidget

from .funcmaps import get_maskseg_method
from utils.logger import logger as LOGGER
from utils.registry import Registry
from utils.imgproc_utils import enlarge_window, get_block_mask
from utils.io_utils import imread, text_is_empty
from modules.base import BaseModule, soft_empty_cache
from modules import INPAINTERS, TEXTDETECTORS, OCR, \
    GET_VALID_TEXTDETECTORS, GET_VALID_INPAINTERS, GET_VALID_OCR, \
    InpainterBase, TextDetectorBase, OCRBase, merge_config_module_params
from utils.textblock import TextBlock, sort_regions
from utils import shared
from utils.asset_pack import (
    SMART_OCR_PARAM_KEYS,
    ensure_pack_extracted,
    get_pack_spec,
    missing_packs_for_module,
    module_pack_names,
    pack_exists,
)
from utils.message import create_error_dialog, create_info_dialog
from .custom_widget import ImgtransProgressMessageBox, ParamComboBox
from .configpanel import ConfigPanel
from utils.proj_imgtrans import ProjImgTrans
from utils.config import pcfg, RunStatus
cfg_module = pcfg.module


class ModuleThread(QThread):

    finish_set_module = Signal()
    _failed_set_module_msg = '模块加载失败。'
    module_thread_stopped = Signal()

    def __init__(self, module_key: str, MODULE_REGISTER: Registry, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.job = None
        self.module: Union[TextDetectorBase, InpainterBase, OCRBase] = None
        self.module_register = MODULE_REGISTER
        self.module_key = module_key

        self.pipeline_pagekey_queue = []
        self.finished_counter = 0
        self.num_process_pages = 0
        self.imgtrans_proj: ProjImgTrans = None
        self.stop_requested = False

    def _set_module(self, module_name: str):
        old_module = self.module
        try:
            module: Union[TextDetectorBase, InpainterBase, OCRBase] \
                = self.module_register.module_dict[module_name]
            params = cfg_module.get_params(self.module_key)[module_name]
            if params is not None:
                self.module = module(**params)
            else:
                self.module = module()
            if not pcfg.module.load_model_on_demand:
                self.module.load_model()
            if old_module is not None:
                del old_module
        except Exception as e:
            self.module = old_module
            create_error_dialog(e, self._failed_set_module_msg)

        self.finish_set_module.emit()

    def pipeline_finished(self):
        if self.imgtrans_proj is None:
            return True
        elif self.finished_counter >= self.num_process_pages:
            return True
        return False

    def initImgtransPipeline(self, proj: ProjImgTrans):
        if self.isRunning():
            self.terminate()
        self.imgtrans_proj = proj
        self.finished_counter = 0
        self.pipeline_pagekey_queue.clear()

    def requestStop(self):
        self.stop_requested = True

    def run(self):
        if self.job is not None:
            self.job()
        self.job = None


class InpaintThread(ModuleThread):

    finish_inpaint = Signal(dict)
    inpainting = False    
    inpaint_failed = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__('inpainter', INPAINTERS, *args, **kwargs)

    @property
    def inpainter(self) -> InpainterBase:
        return self.module

    def setInpainter(self, inpainter: str):
        self.job = lambda : self._set_module(inpainter)
        self.start()

    def inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect=None):
        self.job = lambda : self._inpaint(img, mask, img_key, inpaint_rect)
        self.start()
    
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect=None):
        inpaint_dict = {}
        self.inpainting = True
        try:
            inpainted = self.inpainter.inpaint(img, mask)
            inpaint_dict = {
                'inpainted': inpainted,
                'img': img,
                'mask': mask,
                'img_key': img_key,
                'inpaint_rect': inpaint_rect
            }
            self.finish_inpaint.emit(inpaint_dict)
        except Exception as e:
            create_error_dialog(e, self.tr('图像修补失败。'), 'InpaintFailed')
            self.inpainting = False
            self.inpaint_failed.emit()
        self.inpainting = False


class TextDetectThread(ModuleThread):
    
    finish_detect_page = Signal(str)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__('textdetector', TEXTDETECTORS, *args, **kwargs)

    def setTextDetector(self, textdetector: str):
        self.job = lambda : self._set_module(textdetector)
        self.start()

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.module


class OCRThread(ModuleThread):

    finish_ocr_page = Signal(str)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__('ocr', OCR, *args, **kwargs)

    def setOCR(self, ocr: str):
        self.job = lambda : self._set_module(ocr)
        self.start()
    
    @property
    def ocr(self) -> OCRBase:
        return self.module


class ImgtransThread(QThread):

    pipeline_stopped = Signal()
    update_detect_progress = Signal(int)
    update_ocr_progress = Signal(int)
    update_inpaint_progress = Signal(int)

    finish_blktrans_stage = Signal(str, int)
    finish_blktrans = Signal(int, list)
    unload_modules = Signal(list)

    detect_counter = 0
    ocr_counter = 0
    inpaint_counter = 0

    def __init__(self, 
                 textdetect_thread: TextDetectThread,
                 ocr_thread: OCRThread,
                 inpaint_thread: InpaintThread,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.textdetect_thread = textdetect_thread
        self.ocr_thread = ocr_thread
        self.inpaint_thread = inpaint_thread
        self.job = None
        self.imgtrans_proj: ProjImgTrans = None
        self.stop_requested = False
        self.pages_to_process = None  # 需要处理的页面列表（用于继续运行模式）

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.textdetect_thread.textdetector

    @property
    def ocr(self) -> OCRBase:
        return self.ocr_thread.ocr

    @property
    def inpainter(self) -> InpainterBase:
        return self.inpaint_thread.inpainter

    def runImgtransPipeline(self, imgtrans_proj: ProjImgTrans, pages_to_process=None):
        self.imgtrans_proj = imgtrans_proj
        self.pages_to_process = pages_to_process  # 保存需要处理的页面列表
        self.num_pages = len(self.imgtrans_proj.pages)
        self.stop_requested = False
        # 创建处理索引到实际页面索引的映射
        self.process_idx_to_page_idx = {}
        self.job = self._imgtrans_pipeline
        self.start()
    
    def requestStop(self):
        """请求停止当前任务"""
        if self.isRunning():
            self.stop_requested = True

    def runBlktransPipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int, blk_ids: List[int], tgt_mask):
        self.job = lambda : self._blktrans_pipeline(blk_list, tgt_img, mode, blk_ids, tgt_mask)
        self.start()

    def _blktrans_pipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int, blk_ids: List[int], tgt_mask):
        if mode >= 0 and mode < 3:
            try:
                self.ocr_thread.module.run_ocr(tgt_img, blk_list, split_textblk=True)
            except Exception as e:
                create_error_dialog(e, self.tr('文字识别失败。'), 'OCRFailed')
            self.finish_blktrans.emit(mode, blk_ids)

        if mode > 1:
            im_h, im_w = tgt_img.shape[:2]
            progress_prod = 100. / len(blk_list) if len(blk_list) > 0 else 0
            for ii, blk in enumerate(blk_list):
                xyxy = enlarge_window(blk.xyxy, im_w, im_h)
                xyxy = np.array(xyxy)
                x1, y1, x2, y2 = xyxy.astype(np.int64)
                blk.region_inpaint_dict = None
                if y2 - y1 > 2 and x2 - x1 > 2:
                    im = np.copy(tgt_img[y1: y2, x1: x2])
                    maskseg_method = get_maskseg_method()
                    inpaint_mask_array, ballon_mask, bub_dict = maskseg_method(im, mask=tgt_mask[y1: y2, x1: x2])
                    mask = self.post_process_mask(inpaint_mask_array)
                    if mask.sum() > 0:
                        inpainted = self.inpaint_thread.inpainter.inpaint(im, mask)
                        blk.region_inpaint_dict = {'img': im, 'mask': mask, 'inpaint_rect': [x1, y1, x2, y2], 'inpainted': inpainted}
                    self.finish_blktrans_stage.emit('inpaint', int((ii+1) * progress_prod))
        self.finish_blktrans.emit(mode, blk_ids)

    def _imgtrans_pipeline(self):
        self.detect_counter = 0
        self.ocr_counter = 0
        self.inpaint_counter = 0
        
        # 如果指定了pages_to_process，只处理这些页面
        all_pages = list(self.imgtrans_proj.pages.keys())
        if self.pages_to_process is not None and len(self.pages_to_process) > 0:
            pages_to_iterate = self.pages_to_process
            self.num_pages = num_pages = len(self.pages_to_process)
            # 建立处理索引到实际页面索引的映射
            for process_idx, page_name in enumerate(pages_to_iterate):
                if page_name in all_pages:
                    self.process_idx_to_page_idx[process_idx] = all_pages.index(page_name)
            LOGGER.info(f'Processing specific pages: {len(pages_to_iterate)} pages')
        else:
            pages_to_iterate = all_pages
            self.num_pages = num_pages = len(self.imgtrans_proj.pages)
            # 处理索引等于实际页面索引
            for i in range(num_pages):
                self.process_idx_to_page_idx[i] = i
            LOGGER.info(f'Processing all {num_pages} pages')
        self.textdetect_thread.num_process_pages = self.num_pages
        self.ocr_thread.num_process_pages = self.num_pages
        self.inpaint_thread.num_process_pages = self.num_pages

        for imgname in pages_to_iterate:
            
            # 检查是否请求停止
            if self.stop_requested:
                LOGGER.info('Image extraction pipeline stopped by user')
                break
                
            img = self.imgtrans_proj.read_img(imgname)
            mask = blk_list = None
            need_save_mask = False
            blk_removed: List[TextBlock] = []
            if cfg_module.enable_detect:
                try:
                    mask, blk_list = self.textdetector.detect(img, self.imgtrans_proj)
                    need_save_mask = True
                except Exception as e:
                    create_error_dialog(e, self.tr('文本检测失败。'), 'TextDetectFailed')
                    blk_list = []
                self.detect_counter += 1
                if pcfg.module.keep_exist_textlines:
                    blk_list = self.imgtrans_proj.pages[imgname] + blk_list
                    blk_list = sort_regions(blk_list)
                    existed_mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                    if existed_mask is not None:
                        mask = np.bitwise_or(mask, existed_mask)
                self.imgtrans_proj.pages[imgname] = blk_list

                if mask is not None and not cfg_module.enable_ocr:
                    self.imgtrans_proj.save_mask(imgname, mask)
                    need_save_mask = False
                    
                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_DET)
                self.update_detect_progress.emit(self.detect_counter)

            if blk_list is None:
                blk_list = self.imgtrans_proj.pages[imgname] if imgname in self.imgtrans_proj.pages else []

            if cfg_module.enable_ocr:
                try:
                    self.ocr.run_ocr(img, blk_list)
                except Exception as e:
                    create_error_dialog(e, self.tr('文字识别失败。'), 'OCRFailed')
                self.ocr_counter += 1

                if pcfg.restore_ocr_empty:
                    blk_list_updated = []
                    for blk in blk_list:
                        text = blk.get_text()
                        if text_is_empty(text):
                            blk_removed.append(blk)
                        else:
                            blk_list_updated.append(blk)

                    if len(blk_removed) > 0:
                        blk_list.clear()
                        blk_list += blk_list_updated
                        
                        if mask is None:
                            mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                        if mask is not None:
                            inpainted = None
                            if not cfg_module.enable_inpaint:
                                inpainted = self.imgtrans_proj.load_inpainted_by_imgname(imgname)
                            for blk in blk_removed:
                                xywh = blk.bounding_rect()
                                blk_mask, xyxy = get_block_mask(xywh, mask, blk.angle)
                                x1, y1, x2, y2 = xyxy
                                if blk_mask is not None:
                                    mask[y1: y2, x1: x2] = 0
                                    if inpainted is not None:
                                        mskpnt = np.where(blk_mask)
                                        inpainted[y1: y2, x1: x2][mskpnt] = img[y1: y2, x1: x2][mskpnt]
                                    need_save_mask = True
                            if inpainted is not None and need_save_mask:
                                self.imgtrans_proj.save_inpainted(imgname, inpainted)
                            if need_save_mask:
                                self.imgtrans_proj.save_mask(imgname, mask)
                                need_save_mask = False

                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_OCR)
                self.update_ocr_progress.emit(self.ocr_counter)

            if need_save_mask and mask is not None:
                self.imgtrans_proj.save_mask(imgname, mask)
                need_save_mask = False
                        
            if cfg_module.enable_inpaint:
                if mask is None:
                    mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                if mask is not None:
                    try:
                        inpainted = self.inpainter.inpaint(img, mask, blk_list)
                        self.imgtrans_proj.save_inpainted(imgname, inpainted)
                    except Exception as e:
                        create_error_dialog(e, self.tr('图像修补失败。'), 'InpaintFailed')
                    
                self.inpaint_counter += 1
                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_INPAINT)
                self.update_inpaint_progress.emit(self.inpaint_counter)
            else:
                if len(blk_removed) > 0:
                    self.imgtrans_proj.load_mask_by_imgname

        if self.stop_requested:
            self.pipeline_stopped.emit()

    def detect_finished(self) -> bool:
        if self.imgtrans_proj is None:
            return True
        return self.detect_counter == self.num_pages or not cfg_module.enable_detect

    def ocr_finished(self) -> bool:
        if self.imgtrans_proj is None:
            return True
        return self.ocr_counter == self.num_pages or not cfg_module.enable_ocr

    def inpaint_finished(self) -> bool:
        if self.imgtrans_proj is None or not cfg_module.enable_inpaint:
            return True
        return self.inpaint_counter == self.num_pages or not cfg_module.enable_inpaint

    def run(self):
        if self.job is not None:
            self.job()
        self.job = None

    def recent_finished_index(self, ref_counter: int) -> int:
        if cfg_module.enable_detect:
            ref_counter = min(ref_counter, self.detect_counter)
        if cfg_module.enable_ocr:
            ref_counter = min(ref_counter, self.ocr_counter)
        if cfg_module.enable_inpaint:
            ref_counter = min(ref_counter, self.inpaint_counter)

        process_idx = ref_counter - 1
        # 将处理索引转换为实际页面索引
        if hasattr(self, 'process_idx_to_page_idx') and process_idx in self.process_idx_to_page_idx:
            return self.process_idx_to_page_idx[process_idx]
        return process_idx


def unload_modules(self, module_names):
    model_deleted = False
    for module in module_names:
        module: BaseModule = getattr(self, module)
        if module is None:
            continue
        model_deleted = model_deleted or module.unload_model()
    if model_deleted:
        soft_empty_cache()


class ModuleManager(QObject):
    imgtrans_proj: ProjImgTrans = None

    canvas_inpaint_finished = Signal(dict)
    inpaint_th_finished = Signal()

    imgtrans_pipeline_finished = Signal()
    blktrans_pipeline_finished = Signal(int, list)
    page_trans_finished = Signal(int)

    run_canvas_inpaint = False
    is_waiting_th = False
    block_set_inpainter = False

    def __init__(self, 
                 imgtrans_proj: ProjImgTrans,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imgtrans_proj = imgtrans_proj
        self.check_inpaint_fin_timer = QTimer(self)
        self.check_inpaint_fin_timer.timeout.connect(self.check_inpaint_th_finished)
        self.prompt_parent: QWidget = None
        self.selector_widgets = {
            'textdetector': [],
            'ocr': [],
            'inpainter': [],
        }

    def bindPromptParent(self, parent: QWidget):
        self.prompt_parent = parent

    def registerModuleSelector(self, module_key: str, selector_widget: QWidget):
        if module_key not in self.selector_widgets:
            raise KeyError(f'Unknown module key: {module_key}')
        if selector_widget is None or selector_widget in self.selector_widgets[module_key]:
            return
        self.selector_widgets[module_key].append(selector_widget)

    def setupThread(self, config_panel: ConfigPanel, imgtrans_progress_msgbox: ImgtransProgressMessageBox, ocr_postprocess: Callable = None):
        self.textdetect_thread = TextDetectThread()

        self.ocr_thread = OCRThread()

        self.inpaint_thread = InpaintThread()
        self.inpaint_thread.finish_inpaint.connect(self.on_finish_inpaint)

        self.progress_msgbox = imgtrans_progress_msgbox
        self.progress_msgbox.stop_clicked.connect(self.stopImgtransPipeline)

        self.imgtrans_thread = ImgtransThread(self.textdetect_thread, self.ocr_thread, self.inpaint_thread)
        self.imgtrans_thread.update_detect_progress.connect(self.on_update_detect_progress)
        self.imgtrans_thread.update_ocr_progress.connect(self.on_update_ocr_progress)
        self.imgtrans_thread.update_inpaint_progress.connect(self.on_update_inpaint_progress)
        self.imgtrans_thread.finish_blktrans_stage.connect(self.on_finish_blktrans_stage)
        self.imgtrans_thread.finish_blktrans.connect(self.on_finish_blktrans)
        self.imgtrans_thread.pipeline_stopped.connect(self.on_imgtrans_thread_stopped)

        self.inpaint_panel = None
        if not shared.EXTRACT_ONLY:
            self.inpaint_panel = inpainter_panel = config_panel.inpaint_config_panel
            inpainter_params = merge_config_module_params(cfg_module.inpainter_params, GET_VALID_INPAINTERS(), INPAINTERS.get)
            inpainter_panel.addModulesParamWidgets(inpainter_params)
            inpainter_panel.paramwidget_edited.connect(self.on_inpainterparam_edited)
            inpainter_panel.inpainter_changed.connect(self.setInpainter)
            inpainter_panel.needInpaintChecker.checker_changed.connect(self.on_inpainter_checker_changed)
            inpainter_panel.needInpaintChecker.checker.setChecked(cfg_module.check_need_inpaint)

        self.textdetect_panel = textdetector_panel = config_panel.detect_config_panel
        textdetector_params = merge_config_module_params(cfg_module.textdetector_params, GET_VALID_TEXTDETECTORS(), TEXTDETECTORS.get)
        textdetector_panel.addModulesParamWidgets(textdetector_params)
        textdetector_panel.paramwidget_edited.connect(self.on_textdetectorparam_edited)
        textdetector_panel.detector_changed.connect(self.setTextDetector)

        self.ocr_panel = ocr_panel = config_panel.ocr_config_panel
        ocr_params = merge_config_module_params(cfg_module.ocr_params, GET_VALID_OCR(), OCR.get)
        ocr_panel.addModulesParamWidgets(ocr_params)
        ocr_panel.paramwidget_edited.connect(self.on_ocrparam_edited)
        ocr_panel.ocr_changed.connect(self.setOCR)
        OCRBase.register_postprocess_hooks(ocr_postprocess)

        config_panel.unload_models.connect(self.unload_all_models)


    def unload_all_models(self):
        unload_modules(self, {'textdetector', 'inpainter', 'ocr'})

    @property
    def inpainter(self) -> InpainterBase:
        return self.inpaint_thread.inpainter

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.textdetect_thread.textdetector

    @property
    def ocr(self) -> OCRBase:
        return self.ocr_thread.ocr

    def inpainterBusy(self):
        return self.inpaint_thread.isRunning()

    def inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect = None, **kwargs):
        if self.inpaint_thread.isRunning():
            LOGGER.warning('Waiting for inpainting to finish')
            return
        self.inpaint_thread.inpaint(img, mask, img_key, inpaint_rect)

    def terminateRunningThread(self):
        if self.textdetect_thread.isRunning():
            self.textdetect_thread.quit()
        if self.ocr_thread.isRunning():
            self.ocr_thread.quit()
        if self.inpaint_thread.isRunning():
            self.inpaint_thread.quit()

    def check_inpaint_th_finished(self):
        if self.inpaint_thread.isRunning():
            return
        self.block_set_inpainter = False
        self.check_inpaint_fin_timer.stop()
        self.inpaint_th_finished.emit()

    def runImgtransPipeline(self, pages_to_process=None):
        if self.imgtrans_proj.is_empty:
            LOGGER.info('proj file is empty, nothing to do')
            self.progress_msgbox.hide()
            return
        if not self.ensureRuntimePacksReady():
            return
        self.last_finished_index = -1
        self.terminateRunningThread()
        
        if cfg_module.all_stages_disabled() and self.imgtrans_proj is not None and self.imgtrans_proj.num_pages > 0:
            for ii in range(self.imgtrans_proj.num_pages):
                self.page_trans_finished.emit(ii)
            self.imgtrans_pipeline_finished.emit()
            return
        
        self.progress_msgbox.detect_bar.setVisible(cfg_module.enable_detect)
        self.progress_msgbox.ocr_bar.setVisible(cfg_module.enable_ocr)
        self.progress_msgbox.inpaint_bar.setVisible(cfg_module.enable_inpaint)
        self.progress_msgbox.zero_progress()
        self.progress_msgbox.show()
        self.imgtrans_thread.runImgtransPipeline(self.imgtrans_proj, pages_to_process)
    
    def stopImgtransPipeline(self):
        """停止图像翻译流程"""
        LOGGER.info('Stopping image extraction pipeline...')
        self.imgtrans_thread.requestStop()

    def runBlktransPipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int, blk_ids: List[int], tgt_mask):
        if not self.ensureRuntimePacksReady(
            need_detect=False,
            need_ocr=mode >= 0 and mode < 3,
            need_inpaint=mode >= 2,
        ):
            return
        self.terminateRunningThread()
        self.progress_msgbox.hide_all_bars()
        if mode >= 0 and mode < 3:
            self.progress_msgbox.ocr_bar.show()
        if mode >= 2:
            self.progress_msgbox.inpaint_bar.show()
        self.progress_msgbox.zero_progress()
        self.progress_msgbox.show()
        self.imgtrans_thread.runBlktransPipeline(blk_list, tgt_img, mode, blk_ids, tgt_mask)

    def on_finish_blktrans_stage(self, stage: str, progress: int):
        if stage == 'ocr':
            self.progress_msgbox.updateOCRProgress(progress)
        elif stage == 'inpaint':
            self.progress_msgbox.updateInpaintProgress(progress)
        else:
            raise NotImplementedError(f'Unknown stage: {stage}')
        
    def on_finish_blktrans(self, mode: int, blk_ids: List):
        self.blktrans_pipeline_finished.emit(mode, blk_ids)
        self.progress_msgbox.hide()

    def on_update_detect_progress(self, progress: int):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        if 'detect' in shared.pbar:
            shared.pbar['detect'].update(1)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        self.progress_msgbox.updateDetectProgress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def on_update_ocr_progress(self, progress: int):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        if 'ocr' in shared.pbar:
            shared.pbar['ocr'].update(1)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        self.progress_msgbox.updateOCRProgress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def on_update_inpaint_progress(self, progress: int):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        if 'inpaint' in shared.pbar:
            shared.pbar['inpaint'].update(1)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        self.progress_msgbox.updateInpaintProgress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def progress(self):
        progress = {}
        num_pages = self.imgtrans_thread.num_pages
        if cfg_module.enable_detect:
            progress['detect'] = self.imgtrans_thread.detect_counter / num_pages
        if cfg_module.enable_ocr:
            progress['ocr'] = self.imgtrans_thread.ocr_counter / num_pages
        if cfg_module.enable_inpaint:
            progress['inpaint'] = self.imgtrans_thread.inpaint_counter / num_pages
        return progress

    def proj_finished(self):
        if self.imgtrans_thread.detect_finished() \
            and self.imgtrans_thread.ocr_finished() \
                and self.imgtrans_thread.inpaint_finished():
            return True
        return False

    def finishImgtransPipeline(self):
        if self.proj_finished():
            self.progress_msgbox.hide()
            self.imgtrans_pipeline_finished.emit()
    
    def on_imgtrans_thread_stopped(self):
        """线程完成时确保关闭进度对话框"""
        # 线程完成了，直接关闭窗口
        self.progress_msgbox.hide()
        self.imgtrans_pipeline_finished.emit()

    def _get_param_value(self, params: dict, param_key: str):
        if not isinstance(params, dict) or param_key not in params:
            return None
        value = params[param_key]
        if isinstance(value, dict):
            return value.get('value')
        return value

    def _current_module_name(self, module_key: str) -> str:
        thread = getattr(self, f'{module_key}_thread', None)
        module = getattr(thread, module_key, None) if thread is not None else None
        if module is not None:
            return module.name
        return getattr(cfg_module, module_key)

    def _set_widget_value(self, widget: QWidget, value):
        if widget is None or value is None:
            return
        widget.blockSignals(True)
        try:
            if hasattr(widget, 'setCurrentText'):
                widget.setCurrentText(str(value))
            elif hasattr(widget, 'setText'):
                widget.setText(str(value))
        finally:
            widget.blockSignals(False)

    def _revert_module_selectors(self, module_key: str, value: str):
        for selector_widget in self.selector_widgets.get(module_key, []):
            self._set_widget_value(selector_widget, value)

    def _get_module_class(self, module_name: str):
        if not module_name:
            return None
        for registry in (TEXTDETECTORS, OCR, INPAINTERS):
            module_class = registry.module_dict.get(module_name)
            if module_class is not None:
                return module_class
        return None

    def _module_download_targets_ready(self, module_name: str) -> bool:
        module_class = self._get_module_class(module_name)
        if module_class is None:
            return False

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

    def _missing_pack_names_for_module(self, module_name: str) -> List[str]:
        pack_names = missing_packs_for_module(module_name)
        if len(pack_names) == 0:
            return []
        if self._module_download_targets_ready(module_name):
            return []
        return pack_names

    def _unique_pack_names(self, pack_names: List[str]) -> List[str]:
        ordered_pack_names = []
        seen = set()
        for pack_name in pack_names:
            if not pack_name or pack_name in seen:
                continue
            seen.add(pack_name)
            ordered_pack_names.append(pack_name)
        return ordered_pack_names

    def _prompt_extract_packs(self, pack_names: List[str], reason: str) -> bool:
        pack_names = self._unique_pack_names(pack_names)
        if len(pack_names) == 0:
            return True

        missing_archives = [pack_name for pack_name in pack_names if not pack_exists(pack_name)]
        if len(missing_archives) > 0:
            missing_archive_lines = '\n'.join(
                f'- {get_pack_spec(pack_name).archive_name}'
                for pack_name in missing_archives
            )
            QMessageBox.warning(
                self.prompt_parent,
                self.tr('缺少模型压缩包'),
                self.tr('未在 packs/ 或 optional-packs/ 中找到所需模型压缩包。\n') +
                f'{reason}\n\n{missing_archive_lines}',
            )
            return False

        pack_lines = '\n'.join(
            f'- {get_pack_spec(pack_name).description} ({get_pack_spec(pack_name).archive_name})'
            for pack_name in pack_names
        )
        answer = QMessageBox.question(
            self.prompt_parent,
            self.tr('解压模型压缩包'),
            self.tr('当前功能需要额外的模型文件。\n') +
            f'{reason}\n\n{pack_lines}\n\n' +
            self.tr('现在解压吗？'),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False

        try:
            for pack_name in pack_names:
                extracted = ensure_pack_extracted(pack_name)
                if extracted:
                    LOGGER.info(f'Extracted asset pack: {pack_name}')
        except Exception as e:
            create_error_dialog(e, self.tr('解压模型压缩包失败。'))
            return False
        return True

    def _ensure_module_pack_ready(self, module_name: str, reason: str) -> bool:
        return self._prompt_extract_packs(self._missing_pack_names_for_module(module_name), reason)

    def _collect_smart_ocr_missing_packs(self) -> List[str]:
        smart_ocr_params = cfg_module.ocr_params.get('smart_ocr', {})
        pack_names = []
        for param_key in SMART_OCR_PARAM_KEYS:
            module_name = self._get_param_value(smart_ocr_params, param_key)
            if module_name in {None, '', 'none_ocr', 'horizontal_ocr'}:
                continue
            pack_names.extend(self._missing_pack_names_for_module(module_name))
        return self._unique_pack_names(pack_names)

    def ensureRuntimePacksReady(self, need_detect=None, need_ocr=None, need_inpaint=None) -> bool:
        if need_detect is None:
            need_detect = cfg_module.enable_detect
        if need_ocr is None:
            need_ocr = cfg_module.enable_ocr
        if need_inpaint is None:
            need_inpaint = cfg_module.enable_inpaint and not shared.EXTRACT_ONLY

        pack_names = []
        if need_detect:
            pack_names.extend(self._missing_pack_names_for_module(cfg_module.textdetector))
        if need_ocr:
            pack_names.extend(self._missing_pack_names_for_module(cfg_module.ocr))
            if cfg_module.ocr == 'smart_ocr':
                pack_names.extend(self._collect_smart_ocr_missing_packs())
        if need_inpaint:
            pack_names.extend(self._missing_pack_names_for_module(cfg_module.inpainter))

        return self._prompt_extract_packs(
            pack_names,
            self.tr('当前任务需要额外的模型文件。'),
        )

    def setInpainter(self, inpainter: str = None, prompt_missing: bool = True):
        
        if self.block_set_inpainter:
            return False
        
        if inpainter is None:
            inpainter = cfg_module.inpainter

        current_inpainter = self._current_module_name('inpainter')
        if prompt_missing and not self._ensure_module_pack_ready(
            inpainter,
            self.tr(f'修补模型：{inpainter}'),
        ):
            self._revert_module_selectors('inpainter', current_inpainter)
            return False
        
        if self.inpaint_thread.isRunning():
            self.block_set_inpainter = True
            create_info_dialog(self.tr('正在切换修补模型...'), modal=True, signal_slot_map_list=[{'signal': self.inpaint_th_finished, 'slot': 'done'}])
            self.check_inpaint_fin_timer.start(300)
            return False

        self.inpaint_thread.setInpainter(inpainter)
        return True

    def setTextDetector(self, textdetector: str = None, prompt_missing: bool = True):
        if textdetector is None:
            textdetector = cfg_module.textdetector
        current_textdetector = self._current_module_name('textdetector')
        if prompt_missing and not self._ensure_module_pack_ready(
            textdetector,
            self.tr(f'文本检测模型：{textdetector}'),
        ):
            self._revert_module_selectors('textdetector', current_textdetector)
            return False
        if self.textdetect_thread.isRunning():
            LOGGER.warning('Terminating a running text detection thread.')
            self.textdetect_thread.terminate()
        self.textdetect_thread.setTextDetector(textdetector)
        return True

    def setOCR(self, ocr: str = None, prompt_missing: bool = True):
        if ocr is None:
            ocr = cfg_module.ocr
        current_ocr = self._current_module_name('ocr')
        if prompt_missing and not self._ensure_module_pack_ready(
            ocr,
            self.tr(f'文字识别模型：{ocr}'),
        ):
            self._revert_module_selectors('ocr', current_ocr)
            return False
        if self.ocr_thread.isRunning():
            LOGGER.warning('Terminating a running OCR thread.')
            self.ocr_thread.terminate()
        self.ocr_thread.setOCR(ocr)
        return True

    def on_finish_inpaint(self, inpaint_dict: dict):
        if self.run_canvas_inpaint:
            self.canvas_inpaint_finished.emit(inpaint_dict)
            self.run_canvas_inpaint = False

    def canvas_inpaint(self, inpaint_dict):
        self.run_canvas_inpaint = True
        self.inpaint(**inpaint_dict)
    
    def on_inpainterparam_edited(self, param_key: str, param_content: dict):
        if self.inpainter is not None:
            self.updateModuleSetupParam(self.inpainter, param_key, param_content)
            cfg_module.inpainter_params[self.inpainter.name] = self.inpainter.params

    def on_textdetectorparam_edited(self, param_key: str, param_content: dict):
        if self.textdetector is not None:
            self.updateModuleSetupParam(self.textdetector, param_key, param_content)
            cfg_module.textdetector_params[self.textdetector.name] = self.textdetector.params

    def on_ocrparam_edited(self, param_key: str, param_content: dict):
        if self.ocr is not None:
            if self.ocr.name == 'smart_ocr' and param_key in SMART_OCR_PARAM_KEYS:
                selected_backend = param_content.get('content')
                if param_key == 'page_ocr' and selected_backend == 'horizontal_ocr':
                    selected_backend = self._get_param_value(self.ocr.params, 'horizontal_ocr')
                previous_value = self._get_param_value(self.ocr.params, param_key)
                if selected_backend not in {None, '', 'none_ocr', 'horizontal_ocr'}:
                    if not self._ensure_module_pack_ready(
                        selected_backend,
                        self.tr(f'Smart OCR 后端：{selected_backend}'),
                    ):
                        self._set_widget_value(param_content.get('widget'), previous_value)
                        return
            self.updateModuleSetupParam(self.ocr, param_key, param_content)
            cfg_module.ocr_params[self.ocr.name] = self.ocr.params

    def updateModuleSetupParam(self, 
                               module: InpainterBase,
                               param_key: str, param_content: dict):
            
        if param_content.get('flush', False):
            param_widget: ParamComboBox = param_content['widget']
            param_widget.blockSignals(True)
            current_item = param_widget.currentText()
            param_widget.clear()
            param_widget.addItems(module.flush(param_key))
            param_widget.setCurrentText(current_item)
            param_widget.blockSignals(False)
        elif param_content.get('select_path', False):
            dialog = QFileDialog()
            f = module.params[param_key].get('path_filter', None)
            p = dialog.getOpenFileUrl(self.parent(), filter=f)[0].toLocalFile()
            if osp.exists(p):
                param_widget: ParamComboBox = param_content['widget']
                param_widget.setCurrentText(p)
        else:
            module.updateParam(param_key, param_content['content'])

    def handle_page_changed(self):
        if not self.imgtrans_thread.isRunning():
            if self.inpaint_thread.inpainting:
                self.run_canvas_inpaint = False
                self.inpaint_thread.terminate()

    def on_inpainter_checker_changed(self, is_checked: bool):
        cfg_module.check_need_inpaint = is_checked
        InpainterBase.check_need_inpaint = is_checked
