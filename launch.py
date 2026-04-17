from pathlib import Path
import sys
import argparse
import os.path as osp
import os
import importlib
import subprocess
from platform import platform

BRANCH = 'dev'
VERSION = '1.0.0'

python = sys.executable
git = os.environ.get('GIT', "git")
skip_install = False
index_url = os.environ.get('INDEX_URL', "")
QT_APIS = ['pyqt6', 'pyside6', 'pyqt5', 'pyside2']
stored_commit_hash = None

REQ_WIN = [
    'pywin32'
]

PATH_ROOT=Path(__file__).parent
PATH_FONTS=str(PATH_ROOT/'fonts')
FONT_EXTS = {'.ttf','.otf','.ttc','.pfb'}

IS_WIN7 = "Windows-7" in platform()

import utils.shared as shared # Earlier import of shared to use default for config_path argument

parser = argparse.ArgumentParser()
parser.add_argument("--reinstall-torch", action='store_true', help="launch.py argument: install the appropriate version of torch even if you have some version already installed")
parser.add_argument("--proj-dir", default='', type=str, help='Open project directory on startup')
if IS_WIN7:
    parser.add_argument("--qt-api", default='pyqt5', choices=QT_APIS, help='Set qt api')
else:
    parser.add_argument("--qt-api", default='pyqt6', choices=QT_APIS, help='Set qt api')
parser.add_argument("--debug", action='store_true')
parser.add_argument("--requirements", default='requirements.txt')
parser.add_argument("--headless", action='store_true', help='run without GUI')
parser.add_argument("--headless_continuous", action='store_true', help='like headless but will not exit after finishing extraction, prompts the user for new exec_dirs until user exits the program')
parser.add_argument("--exec_dirs", default='', help='extraction queue (project directories) separated by comma')
parser.add_argument("--ldpi", default=None, type=float, help='logical dots perinch')
parser.add_argument("--export-source-txt", action='store_true', help='save source to txt file once RUN completed')
parser.add_argument("--export-source-md", action='store_true', help='save source to markdown file once RUN completed')
parser.add_argument("--frozen", action='store_true', help='run without checking requirements')
parser.add_argument("--update", action='store_true', help="Update the repository before launching") # Add argument --update
parser.add_argument("--config_path", default=shared.CONFIG_PATH, help='Config file to use for extraction') # Named config_path to avoid conflict with existing name config
parser.add_argument('--nightly', action='store_true', help="Enable AMD Nightly ROCm")
args, _ = parser.parse_known_args()


def _set_param_value(params_dict, param_key, value):
    if params_dict is None or param_key not in params_dict:
        return
    param = params_dict[param_key]
    if isinstance(param, dict) and 'value' in param:
        param['value'] = value
    else:
        params_dict[param_key] = value


def _pick_first_matching_option(options, keywords):
    lowered_keywords = [kw.lower() for kw in keywords]
    for option in options:
        option_text = str(option).lower()
        if any(keyword in option_text for keyword in lowered_keywords):
            return option
    return None


def _get_module_param_value(params_dict, param_key):
    if not isinstance(params_dict, dict) or param_key not in params_dict:
        return None
    param = params_dict[param_key]
    if isinstance(param, dict):
        return param.get('value')
    return param


def _stariver_detector_ready(detector_params):
    username = _get_module_param_value(detector_params, 'User')
    password = _get_module_param_value(detector_params, 'Password')
    if not username or not password:
        return False
    username = str(username).strip()
    password = str(password).strip()
    placeholder_user = '填入你的用户名'
    placeholder_password = '填入你的密码。请注意，密码会明文保存，请勿在公共电脑上使用'
    return username != placeholder_user and password != placeholder_password


def apply_extract_mode_defaults():
    from utils.config import pcfg
    from utils.logger import logger as LOGGER
    from modules import OCR, TEXTDETECTORS, GET_AVAILABLE_OCR, GET_AVAILABLE_TEXTDETECTORS, merge_config_module_params

    pcfg.module.enable_detect = True
    pcfg.module.enable_ocr = True
    pcfg.module.enable_inpaint = False
    pcfg.module.keep_exist_textlines = False
    pcfg.module.update_finish_code()
    pcfg.show_source_text = True
    pcfg.show_trans_text = False
    pcfg.imgtrans_paintmode = False
    pcfg.let_autolayout_flag = False

    available_detectors = GET_AVAILABLE_TEXTDETECTORS()
    pcfg.module.textdetector_params = merge_config_module_params(
        pcfg.module.textdetector_params,
        available_detectors,
        TEXTDETECTORS.get,
    )
    preferred_detector_order = ['ctd', 'ysgyolo', 'stariver_ocr']
    current_detector = pcfg.module.textdetector
    current_detector_valid = current_detector in available_detectors
    current_detector_ready = True
    if current_detector == 'stariver_ocr':
        current_detector_ready = _stariver_detector_ready(
            pcfg.module.textdetector_params.get('stariver_ocr')
        )
    if available_detectors and (not current_detector_valid or not current_detector_ready):
        preferred_detector = next(
            (name for name in preferred_detector_order if name in available_detectors),
            available_detectors[0],
        )
        if preferred_detector != current_detector:
            LOGGER.info(
                f'Extraction mode switched detector from {current_detector} to {preferred_detector}.'
            )
        pcfg.module.textdetector = preferred_detector

    available_ocr = GET_AVAILABLE_OCR()
    pcfg.module.ocr_params = merge_config_module_params(pcfg.module.ocr_params, available_ocr, OCR.get)
    # Prefer OCR backends that normalize vertical manga text regions before recognition.
    preferred_ocr_order = [
        'smart_ocr',
        'mit48px',
        'mit48px_ctc',
        'manga_ocr',
        'paddle_ocr',
        'PaddleOCRVLManga',
        'paddle_vl',
        'windows_ocr',
        'macos_ocr',
    ]
    preferred_ocr = next((name for name in preferred_ocr_order if name in available_ocr), None)
    if preferred_ocr is None and pcfg.module.ocr not in available_ocr and available_ocr:
        preferred_ocr = available_ocr[0]
    if preferred_ocr is not None:
        pcfg.module.ocr = preferred_ocr

    for module_name, params in pcfg.module.ocr_params.items():
        if params is None:
            continue
        if module_name == 'smart_ocr':
            vertical_ocr = next((name for name in ['mit48px', 'mit48px_ctc', 'manga_ocr'] if name in available_ocr), None)
            horizontal_ocr = next((name for name in ['windows_ocr', 'one_ocr', 'paddle_ocr', 'mit48px'] if name in available_ocr), None)
            fallback_candidates = []
            if vertical_ocr is not None and vertical_ocr != horizontal_ocr:
                fallback_candidates.append(vertical_ocr)
            fallback_candidates.extend(['mit48px_ctc', 'mit48px', 'one_ocr', 'paddle_ocr', 'windows_ocr', 'none_ocr'])
            fallback_ocr = next((name for name in fallback_candidates if name in available_ocr and name != horizontal_ocr), None)

            if vertical_ocr is not None:
                _set_param_value(params, 'vertical_ocr', vertical_ocr)
            if horizontal_ocr is not None:
                _set_param_value(params, 'horizontal_ocr', horizontal_ocr)
                _set_param_value(params, 'page_ocr', 'horizontal_ocr')
            if fallback_ocr is not None:
                _set_param_value(params, 'fallback_ocr', fallback_ocr)
            else:
                _set_param_value(params, 'fallback_ocr', 'none_ocr')
            _set_param_value(params, 'retry_on_empty', True)
        elif module_name == 'paddle_ocr':
            _set_param_value(params, 'language', 'Chinese & English')
            _set_param_value(params, 'output_format', 'As Recognized')
        elif module_name == 'google_vision':
            _set_param_value(params, 'language_hints', 'zh-CN')
        elif module_name == 'ocr_llm_api':
            _set_param_value(params, 'language', 'Chinese (Simplified)')
        elif module_name in {'windows_ocr', 'macos_ocr'} and 'language' in params:
            language_param = params['language']
            options = language_param.get('options', []) if isinstance(language_param, dict) else []
            preferred_language = _pick_first_matching_option(options, ['chinese', '中文', 'zh'])
            if preferred_language is not None:
                _set_param_value(params, 'language', preferred_language)

    if preferred_ocr is not None:
        LOGGER.info(f'Extraction mode enabled. Preferred OCR set to {preferred_ocr}.')
    else:
        LOGGER.info(f'Extraction mode enabled. Using existing OCR setting: {pcfg.module.ocr}.')


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    if desc is not None:
        print(desc)

    if live:
        result = subprocess.run(command, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def run_pip(args, desc=None):
    if skip_install:
        return

    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {args} --prefer-binary{index_url_line} --disable-pip-version-check --no-warn-script-location', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=True)


def commit_hash():
    global stored_commit_hash

    if stored_commit_hash is not None:
        return stored_commit_hash

    try:
        stored_commit_hash = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        stored_commit_hash = "<none>"

    return stored_commit_hash


BT = None
APP = None

def restart():
    global BT
    print('restarting...\n')
    if BT:
        BT.close()
    os.execv(sys.executable, ['python'] + sys.argv)


def setup_locks():
    from utils.lock import RUNTIME_LOCKS
    from qtpy.QtCore import QMutex
    RUNTIME_LOCKS['model_loading'] = QMutex()


def main():

    if args.debug:
        os.environ['BALLOONTRANS_DEBUG'] = '1'

    os.environ['QT_API'] = args.qt_api

    commit = commit_hash()

    print('Python version: ', sys.version)
    print('Python executable: ', sys.executable)
    print(f'Version: {VERSION}')
    print(f'Branch: {BRANCH}')
    print(f"Commit hash: {commit}")

    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(APP_DIR)

    prepare_environment()

    from utils.zluda_config import enable_zluda_config
    enable_zluda_config()

    if args.update:
        if getattr(sys, 'frozen', False):
            print('Running as app, skipping update.')
        else:
            print('Checking for updates...')
            try:
                current_commit = commit_hash()
                run(f"{git} fetch origin {BRANCH}", desc="Fetching updates from git...", errdesc="Failed to fetch updates.")
                latest_commit = run(f"{git} rev-parse origin/{BRANCH}").strip()

                if current_commit != latest_commit:
                    print("New updates found. Updating repository...")
                    run(f"{git} pull origin {BRANCH}", desc="Updating repository...", errdesc="Failed to update repository.")
                    print("Repository updated. Restarting to apply updates...")
                    restart()
                    return
                else:
                    print("No updates found.")
            except Exception as e:
                print(f"Update check failed: {e}")
                print("Continuing with the current version.")


    from utils.logger import setup_logging, logger as LOGGER
    from utils.io_utils import find_all_files_recursive
    from utils import config as program_config

    from qtpy.QtCore import QTranslator, QLocale, Qt
    shared.args = args
    shared.DEFAULT_DISPLAY_LANG = QLocale.system().name().replace('en_CN', 'zh_CN')
    shared.HEADLESS = args.headless
    shared.HEADLESS_CONTINUOUS = args.headless_continuous
    shared.EXTRACT_ONLY = True
    shared.load_cache()
    program_config.load_config(args.config_path)
    config = program_config.pcfg

    if args.headless or args.headless_continuous:
        config.module.load_model_on_demand = True
        config.module.empty_runcache = False

    if sys.platform == 'win32':
        import ctypes
        myappid = u'BallonsOCR' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    import qtpy
    from qtpy.QtWidgets import QApplication
    from qtpy.QtGui import QIcon, QFontDatabase, QGuiApplication, QFont
    from qtpy import API, QT_VERSION

    LOGGER.info(f'QT_API: {API}, QT Version: {QT_VERSION}')

    shared.DEBUG = args.debug
    shared.USE_PYSIDE6 = API == 'pyside6'
    if qtpy.API_NAME[-1] == '6':
        shared.FLAG_QT6 = True
    else:
        shared.FLAG_QT6 = False
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True) #enable high dpi scaling
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True) #use high dpi icons
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    os.chdir(shared.PROGRAM_PATH)

    setup_logging(shared.LOGGING_PATH)

    app_args = sys.argv
    if args.headless or args.headless_continuous:
        app_args = sys.argv + ['-platform', 'offscreen']
    app = QApplication(app_args)
    app.setApplicationName('BallonsOCR')
    app.setApplicationVersion(VERSION)

    from modules.base import init_module_registries
    from modules.prepare_local_files import prepare_local_files_forall
    init_module_registries(['textdetector', 'ocr'])
    apply_extract_mode_defaults()
    prepare_local_files_forall()

    if not args.headless and not args.headless_continuous:
        ps = QGuiApplication.primaryScreen()
        shared.LDPI = ps.logicalDotsPerInch()
        shared.SCREEN_W = ps.geometry().width()
        shared.SCREEN_H = ps.geometry().height()

    lang = config.display_lang
    langp = osp.join(shared.TRANSLATE_DIR, lang + '.qm')
    if osp.exists(langp):
        translator = QTranslator()
        translator.load(lang, osp.dirname(osp.abspath(__file__)) + "/translate")
        app.installTranslator(translator)
    elif lang not in ('en_US', 'English'):
        LOGGER.warning(f'target display language file {langp} doesnt exist.')
    LOGGER.info(f'set display language to {lang}')

    # Fonts
    # Load custom fonts if they exist
    if osp.exists(PATH_FONTS):
        for fp in find_all_files_recursive(PATH_FONTS, FONT_EXTS):
            fnt_idx = QFontDatabase.addApplicationFont(fp)
            if fnt_idx >= 0:
                shared.CUSTOM_FONTS.append(QFontDatabase.applicationFontFamilies(fnt_idx)[0])

    if sys.platform == 'win32' and (args.headless or args.headless_continuous):
        # font database does not initialise on windows with qpa -offscreen:
        # whttps://github.com/dmMaze/BallonsTranslator/issues/519
        from qtpy.QtCore import QStandardPaths
        font_dir_list = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.FontsLocation)
        for fd in font_dir_list:
            fp_list = find_all_files_recursive(fd, FONT_EXTS)
            for fp in fp_list:
                fnt_idx = QFontDatabase.addApplicationFont(fp)

    if shared.FLAG_QT6:
        shared.FONT_FAMILIES = set(f for f in QFontDatabase.families())
    else:
        fdb = QFontDatabase()
        shared.FONT_FAMILIES = set(fdb.families())

    app_font = QFont('Microsoft YaHei UI')
    if not app_font.exactMatch() or sys.platform == 'darwin':
        app_font = app.font()
    app_font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias)
    QGuiApplication.setFont(app_font)
    shared.DEFAULT_FONT_FAMILY = app_font.family()
    shared.APP_DEFAULT_FONT = app_font.family()
    
    if args.ldpi:
        shared.LDPI = args.ldpi

    setup_locks()

    from ui.mainwindow import MainWindow
    ballontrans = MainWindow(app, config, open_dir=args.proj_dir, **vars(args))
    global BT
    BT = ballontrans
    BT.restart_signal.connect(restart)

    if not args.headless and not args.headless_continuous:
        if shared.SCREEN_W > 1707 and sys.platform == 'win32':   # higher than 2560 (1440p) / 1.5
            # https://github.com/dmMaze/BallonsTranslator/issues/220
            BT.comicTransSplitter.setHandleWidth(7)

        ballontrans.setWindowIcon(QIcon(shared.ICON_PATH))
        ballontrans.show()
        ballontrans.resetStyleSheet()
    sys.exit(app.exec())

def is_amd_gpu():
    try:
        if sys.platform == 'win32':
            # Windows: use wmic
            cmd = 'wmic path win32_VideoController get name'
            output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
            return any(keyword in output for keyword in ["AMD", "Radeon"])

        else:
            return False

    except Exception:
        return False

def supported_amd_nightly_gpu():
    try:
        if sys.platform == 'win32':
            # Windows: use wmic
            cmd = 'wmic path win32_VideoController get name'
            output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)

            if any(keyword in output for keyword in
                   ["RX 7900", "RX 7800", "RX 7700", "RX 7600", "PRO W7900", "PRO W7800", "PRO W7700"]):
                return "RDNA3"
            if any(keyword in output for keyword in
                   ["RX 9070", "RX 9060"]):
                return "RDNA4"
        else:
            return "None"

    except Exception:
        return "None"

def prepare_environment():

    try:
        import packaging
    except ModuleNotFoundError:
        run_pip(f"install packaging", "install packaging")

    from utils.package import check_req_file, check_reqs

    if getattr(sys, 'frozen', False):
        print('Running as app, skip dependency installation')
        return

    if args.frozen:
        return

    req_updated = False
    if sys.platform == 'win32':
        for req in REQ_WIN:
            if not check_reqs([req]):
                run_pip(f"install {req}", req)
                req_updated = True

    torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check")
    if args.nightly and is_amd_gpu():
        print('AMD GPU: Yes')
        amd_nightly_gpu = supported_amd_nightly_gpu()
        if amd_nightly_gpu == "None":
            Exception("No AMD Nightly GPU supported")
        if amd_nightly_gpu == "RDNA3":
            torch_command = os.environ.get('TORCH_COMMAND',
                                           "pip install https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torch-2.8.0a0%2Bgitfc14c65-cp312-cp312-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torchvision-0.24.0a0%2Bc85f008-cp312-cp312-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torchaudio-2.6.0a0%2B1a8f621-cp312-cp312-win_amd64.whl")
        if amd_nightly_gpu == "RDNA4":
            torch_command = os.environ.get('TORCH_COMMAND',
                                           "pip install https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torch-2.8.0a0%2Bgitfc14c65-cp312-cp312-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torchvision-0.24.0a0%2Bc85f008-cp312-cp312-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torchaudio-2.6.0a0%2B1a8f621-cp312-cp312-win_amd64.whl")
    if args.reinstall_torch or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)
        req_updated = True

    if not check_req_file(args.requirements):
        run_pip(f"install -r {args.requirements}", "requirements")
        req_updated = True

    if req_updated:
        import site
        importlib.reload(site)





if __name__ == '__main__':
    main()
