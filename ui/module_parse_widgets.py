from typing import List, Callable

from modules import GET_AVAILABLE_TEXTDETECTORS, GET_AVAILABLE_OCR, DEFAULT_DEVICE, GPUINTENSIVE_SET
from utils.logger import logger as LOGGER
from .custom_widget import ConfigComboBox, ParamComboBox, ParamNameLabel
from utils.shared import CONFIG_COMBOBOX_LONG, size2width, CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_HEIGHT
from utils.config import pcfg

from qtpy.QtWidgets import QPlainTextEdit, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QCheckBox, QLineEdit, QGridLayout, QPushButton
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QDoubleValidator


PARAM_DISPLAY_NAME_MAP = {
    'device': '设备',
    'detect_size': '检测尺寸',
    'detect size': '检测尺寸',
    'det_rearrange_max_batches': '检测重排最大批次',
    'det_limit_side_len': '检测边长上限',
    'font size multiplier': '字号倍率',
    'font size max': '字号上限',
    'font size min': '字号下限',
    'font_size_multiplier': '字号倍率',
    'mask dilate size': '蒙版膨胀尺寸',
    'chunk_size': '批大小',
    'language': '语言',
    'language_hints': '语言提示',
    'confidence_level': '置信等级',
    'vertical_ocr': '竖排 OCR',
    'horizontal_ocr': '横排 OCR',
    'fallback_ocr': '回退 OCR',
    'page_ocr': '整页 OCR',
    'retry_on_empty': '空结果重试',
    'vertical_aspect_ratio_threshold': '竖排判定阈值',
    'expand_small_blocks': '小块扩展',
    'newline_handling': '换行处理',
    'reverse_line_order': '反转行顺序',
    'no_uppercase': '禁用大写',
    'response_method': '响应方式',
    'proxy': '代理',
    'delay': '请求间隔',
    'api_key': 'API Key',
    'provider': '服务提供方',
    'multiple_keys': '多个密钥',
    'endpoint': '接口地址',
    'model': '模型',
    'override_model': '覆盖模型名',
    'detail_level': '图像细节等级',
    'prompt': '提示词',
    'system_prompt': '系统提示词',
    'requests_per_minute': '每分钟请求数',
    'max_response_tokens': '最大回复 Token',
    'max_new_tokens': '最大生成 Token',
    'target_language': '目标语言',
    'ocr_version': 'OCR 版本',
    'use_angle_cls': '启用角度分类',
    'enable_mkldnn': '启用 MKL-DNN',
    'rec_batch_num': '识别批大小',
    'drop_score': '识别置信度阈值',
    'text_case': '文本大小写',
    'output_format': '输出格式',
    'User': '用户名',
    'Password': '密码',
    'refine': '结果优化',
    'filtrate': '结果过滤',
    'disable_skip_area': '禁用跳过区域',
    'detect_scale': '检测缩放',
    'merge_threshold': '合并阈值',
    'force_expand': '强制扩图',
    'low_accuracy_mode': '低精度模式',
    'update_token_btn': '刷新 Token',
    'expand_ratio': '扩展比例',
    'font_size_offset': '字号偏移',
    'font_size_min(set to -1 to disable)': '字号下限（-1 禁用）',
    'font_size_max(set to -1 to disable)': '字号上限（-1 禁用）',
    'label': '检测标签',
    'source text is vertical': '原文为竖排',
    'confidence threshold': '置信度阈值',
    'IoU threshold': 'IoU 阈值',
    'model path': '模型路径',
    'merge text lines': '合并文本行',
    'server_url': '服务地址',
    'prettifyMarkdown': '美化 Markdown',
    'visualize': '输出可视化',
    'NOTICE': '说明',
}

PARAM_DESCRIPTION_MAP = {
    'ComicTextDetector': 'ComicTextDetector 文本检测器',
    'OCRMIT32px': 'MIT OCR 模型',
    'OCR backend for vertical manga text blocks.': '竖排漫画文本块使用的 OCR 后端。',
    'OCR backend for horizontal text blocks.': '横排文本块使用的 OCR 后端。',
    'Retry backend used when the primary result is empty or the primary backend is unavailable.': '主 OCR 结果为空或不可用时使用的回退 OCR 后端。',
    'OCR backend for full-image OCR requests. Use "horizontal_ocr" to reuse the horizontal route.': '整页 OCR 请求使用的后端。选择 "horizontal_ocr" 可复用横排 OCR 路由。',
    'Retry once with fallback OCR when the first result is empty.': '首次识别结果为空时，使用回退 OCR 再重试一次。',
    'When detector orientation is missing, treat height / width above this threshold as vertical.': '当检测结果缺少方向信息时，高宽比超过该阈值将按竖排处理。',
    'Route each text block to different OCR backends based on orientation.': '根据文本块方向，将其分发到不同的 OCR 后端。',
}


def localize_param_text(text: str):
    if not isinstance(text, str):
        return text
    return PARAM_DISPLAY_NAME_MAP.get(text, PARAM_DESCRIPTION_MAP.get(text, text))


class ParamCheckGroup(QWidget):

    paramwidget_edited = Signal(str, dict)

    def __init__(self, param_key, check_group: dict, parent=None) -> None:
        super().__init__(parent=parent)
        self.param_key = param_key
        layout = QHBoxLayout(self)
        self.label2widget = {}
        for k, v in check_group.items():
            checker = QCheckBox(text=localize_param_text(k), parent=self)
            checker.setChecked(v)
            layout.addWidget(checker)
            self.label2widget[k] = checker
            checker.clicked.connect(self.on_checker_clicked)

    def on_checker_clicked(self):
        new_state_dict = {}
        w = QCheckBox()
        for k, w in self.label2widget.items():
            new_state_dict[k] = w.isChecked()
        self.paramwidget_edited.emit(self.param_key, new_state_dict)


class ParamLineEditor(QLineEdit):
    
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, force_digital, size='short', *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key
        self.setFixedWidth(size2width(size))
        self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.textChanged.connect(self.on_text_changed)

        if force_digital:
            validator = QDoubleValidator()
            self.setValidator(validator)

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.text())

class ParamEditor(QPlainTextEdit):
    
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key

        if param_key == 'chat sample':
            self.setFixedWidth(int(CONFIG_COMBOBOX_LONG * 1.2))
            self.setFixedHeight(200)
        else:
            self.setFixedWidth(CONFIG_COMBOBOX_LONG)
            self.setFixedHeight(100)
        # self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.textChanged.connect(self.on_text_changed)

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.text())

    def setText(self, text: str):
        self.setPlainText(text)

    def text(self):
        return self.toPlainText()


class ParamCheckerBox(QWidget):
    checker_changed = Signal(bool)
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.checker = QCheckBox()
        name_label = ParamNameLabel(localize_param_text(param_key))
        hlayout = QHBoxLayout(self)
        hlayout.addWidget(name_label)
        hlayout.addWidget(self.checker)
        hlayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.checker.stateChanged.connect(self.on_checker_changed)

    def on_checker_changed(self):
        is_checked = self.checker.isChecked()
        self.checker_changed.emit(is_checked)
        checked = 'true' if is_checked else 'false'
        self.paramwidget_edited.emit(self.param_key, checked)


class ParamCheckBox(QCheckBox):
    paramwidget_edited = Signal(str, bool)
    def __init__(self, param_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.stateChanged.connect(self.on_checker_changed)

    def on_checker_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.isChecked())


def get_param_display_name(param_key: str, param_dict: dict = None):
    if param_dict is not None and isinstance(param_dict, dict):
        if 'display_name' in param_dict:
            return localize_param_text(param_dict['display_name'])
    return localize_param_text(param_key)


SMART_OCR_SELECTOR_KEYS = {'vertical_ocr', 'horizontal_ocr', 'fallback_ocr', 'page_ocr'}


def get_selector_options(param_key: str, param_dict: dict) -> list:
    options = list(param_dict.get('options', []))
    if param_key not in SMART_OCR_SELECTOR_KEYS:
        return options

    available_ocr = GET_AVAILABLE_OCR()
    filtered_options = []
    for option in options:
        if option == 'horizontal_ocr' and param_key == 'page_ocr':
            filtered_options.append(option)
            continue
        if option in available_ocr:
            filtered_options.append(option)
    return filtered_options


class ParamPushButton(QPushButton):
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, param_dict: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.setText(get_param_display_name(param_key, param_dict))
        self.clicked.connect(self.on_clicked)

    def on_clicked(self):
        self.paramwidget_edited.emit(self.param_key, '')


class ParamWidget(QWidget):

    paramwidget_edited = Signal(str, dict)
    def __init__(self, params, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layout = QHBoxLayout(self)
        self.param_layout = param_layout = QGridLayout()
        param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(param_layout)
        layout.addStretch(-1)

        if 'description' in params:
            self.setToolTip(localize_param_text(params['description']))

        for ii, param_key in enumerate(params):
            if param_key == 'description' or param_key.startswith('__'):
                continue
            display_param_name = get_param_display_name(param_key)

            require_label = True
            is_str = isinstance(params[param_key], str)
            is_digital = isinstance(params[param_key], float) or isinstance(params[param_key], int)
            param_widget = None

            if isinstance(params[param_key], bool):
                param_widget = ParamCheckBox(param_key)
                val = params[param_key]
                param_widget.setChecked(val)
                param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

            elif is_str or is_digital:
                param_widget = ParamLineEditor(param_key, force_digital=is_digital)
                val = params[param_key]
                if is_digital:
                    val = str(val)
                param_widget.setText(val)
                param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

            elif isinstance(params[param_key], dict):
                param_dict = params[param_key]
                display_param_name = get_param_display_name(param_key, param_dict)
                value = params[param_key]['value']
                param_widget = None  # Ensure initialization
                param_type = param_dict['type'] if 'type' in param_dict else 'line_editor'
                flush_btn = param_dict.get('flush_btn', False)
                path_selector = param_dict.get('path_selector', False)
                param_size = param_dict.get('size', 'short')
                if param_type == 'selector':
                    if 'url' in param_key:
                        size = size2width('median')
                    else:
                        size = size2width(param_size)

                    options = get_selector_options(param_key, param_dict)
                    param_widget = ParamComboBox(
                        param_key, options, size=size, scrollWidget=scrollWidget, flush_btn=flush_btn, path_selector=path_selector)

                    param_widget.setCurrentText(str(value))
                    param_widget.setEditable(param_dict.get('editable', False))

                elif param_type == 'editor':
                    param_widget = ParamEditor(param_key)
                    param_widget.setText(value)

                elif param_type == 'checkbox':
                    param_widget = ParamCheckBox(param_key)
                    if isinstance(value, str):
                        value = value.lower().strip() == 'true'
                        params[param_key]['value'] = value
                    param_widget.setChecked(value)

                elif param_type == 'pushbtn':
                    param_widget = ParamPushButton(param_key, param_dict)
                    require_label = False

                elif param_type == 'line_editor':
                    param_widget = ParamLineEditor(param_key, force_digital=is_digital)
                    param_widget.setText(str(value))

                elif param_type == 'check_group':
                    param_widget = ParamCheckGroup(param_key, check_group=value)

                if param_widget is not None:
                    param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)
                    if 'description' in param_dict:
                        param_widget.setToolTip(localize_param_text(param_dict['description']))

            widget_idx = 0
            if require_label:
                param_label = ParamNameLabel(display_param_name)
                param_layout.addWidget(param_label, ii, 0)
                widget_idx = 1
            if param_widget is not None:
                pw_lo = None
                if hasattr(param_widget, 'flush_btn') or hasattr(param_widget, 'path_select_btn'):
                    pw_lo = QHBoxLayout()
                    pw_lo.addWidget(param_widget)
                if hasattr(param_widget, 'flush_btn'):
                    pw_lo.addWidget(param_widget.flush_btn)
                    param_widget.flushbtn_clicked.connect(self.on_flushbtn_clicked)
                if hasattr(param_widget, 'path_select_btn'):
                    pw_lo.addWidget(param_widget.path_select_btn)
                    param_widget.pathbtn_clicked.connect(self.on_pathbtn_clicked)
                if pw_lo is None:
                    param_layout.addWidget(param_widget, ii, widget_idx)
                else:
                    param_layout.addLayout(pw_lo, ii, widget_idx)
            else:
                v = params[param_key]
                raise ValueError(f"Failed to initialize widget for key-value pair: {param_key}-{v}")
            
    def on_flushbtn_clicked(self):
        paramw: ParamComboBox = self.sender()
        content_dict = {'content': '', 'widget': paramw, 'flush': True}
        self.paramwidget_edited.emit(paramw.param_key, content_dict)

    def on_pathbtn_clicked(self):
        paramw: ParamComboBox = self.sender()
        content_dict = {'content': '', 'widget': paramw, 'select_path': True}
        self.paramwidget_edited.emit(paramw.param_key, content_dict)

    def on_paramwidget_edited(self, param_key, param_content):
        content_dict = {'content': param_content, 'widget': self.sender()}
        self.paramwidget_edited.emit(param_key, content_dict)

class ModuleParseWidgets(QWidget):
    def addModulesParamWidgets(self, ocr_instance):
        self.params = ocr_instance.get_params()
        self.on_module_changed()

    def on_module_changed(self):
        self.updateModuleParamWidget()

    def updateModuleParamWidget(self):
        widget = ParamWidget(self.params, scrollWidget=self)
        layout = QVBoxLayout()
        layout.addWidget(widget)
        self.setLayout(layout)

class ModuleConfigParseWidget(QWidget):
    module_changed = Signal(str)
    paramwidget_edited = Signal(str, dict)
    def __init__(self, module_name: str, get_valid_module_keys: Callable, scrollWidget: QWidget, add_from: int = 1, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.get_valid_module_keys = get_valid_module_keys
        self.module_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.params_layout = QHBoxLayout()
        self.params_layout.setContentsMargins(0, 0, 0, 0)

        p_layout = QHBoxLayout()
        p_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.module_label = ParamNameLabel(module_name)
        p_layout.addWidget(self.module_label)
        p_layout.addWidget(self.module_combobox)
        p_layout.addStretch(-1)
        self.p_layout = p_layout

        layout = QVBoxLayout(self)
        self.param_widget_map = {}
        layout.addLayout(p_layout) 
        layout.addLayout(self.params_layout)
        layout.setSpacing(30)
        self.vlayout = layout

        self.visibleWidget: QWidget = None
        self.module_dict: dict = {}

    def addModulesParamWidgets(self, module_dict: dict):
        invalid_module_keys = []
        valid_modulekeys = self.get_valid_module_keys()

        num_widgets_before = len(self.param_widget_map)

        for module in module_dict:
            if module not in valid_modulekeys:
                invalid_module_keys.append(module)
                continue

            if module in self.param_widget_map:
                LOGGER.warning(f'duplicated module key: {module}')
                continue

            self.module_combobox.addItem(module)
            params = module_dict[module]
            if params is not None:
                self.param_widget_map[module] = None

        if len(invalid_module_keys) > 0:
            LOGGER.warning(F'Invalid module keys: {invalid_module_keys}')
            for ik in invalid_module_keys:
                module_dict.pop(ik)

        self.module_dict = module_dict

        num_widgets_after = len(self.param_widget_map)
        if num_widgets_before == 0 and num_widgets_after > 0:
            self.on_module_changed()
            self.module_combobox.currentTextChanged.connect(self.on_module_changed)

    def setModule(self, module: str):
        self.blockSignals(True)
        self.module_combobox.setCurrentText(module)
        self.updateModuleParamWidget()
        self.blockSignals(False)

    def updateModuleParamWidget(self):
        module = self.module_combobox.currentText()
        if self.visibleWidget is not None:
            self.visibleWidget.hide()
        if module in self.param_widget_map:
            widget: QWidget = self.param_widget_map[module]
            if widget is None:
                # lazy load widgets
                params = self.module_dict[module]
                widget = ParamWidget(params, scrollWidget=self)
                widget.paramwidget_edited.connect(self.paramwidget_edited)
                self.param_widget_map[module] = widget
                self.params_layout.addWidget(widget)
            else:
                widget.show()
            self.visibleWidget = widget

    def on_module_changed(self):
        self.updateModuleParamWidget()
        self.module_changed.emit(self.module_combobox.currentText())



class TextDetectConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_AVAILABLE_TEXTDETECTORS, scrollWidget = scrollWidget, *args, **kwargs)
        self.detector_changed = self.module_changed
        self.setDetector = self.setModule
        self.keep_existing_checker = QCheckBox(text=self.tr('保留现有文本线'))
        self.p_layout.insertWidget(2, self.keep_existing_checker)
        

class OCRConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_AVAILABLE_OCR, scrollWidget = scrollWidget, *args, **kwargs)
        self.ocr_changed = self.module_changed
        self.setOCR = self.setModule
        self.restoreEmptyOCRChecker = QCheckBox(self.tr('当 OCR 返回空文本时，删除并还原对应区域。'), self)
        self.restoreEmptyOCRChecker.clicked.connect(self.on_restore_empty_ocr)
        self.vlayout.addWidget(self.restoreEmptyOCRChecker)
        self.fontDetectChecker = QCheckBox(self.tr('字体检测'), self)
        self.fontDetectChecker.setChecked(pcfg.module.ocr_font_detect)
        self.fontDetectChecker.clicked.connect(self.on_fontdetect_changed)
        self.vlayout.addWidget(self.fontDetectChecker)

    def on_restore_empty_ocr(self):
        pcfg.restore_ocr_empty = self.restoreEmptyOCRChecker.isChecked()

    def on_fontdetect_changed(self):
        pcfg.module.ocr_font_detect = self.fontDetectChecker.isChecked()
