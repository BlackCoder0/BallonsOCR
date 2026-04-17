@echo off
setlocal
cd /d "%~dp0.."

python scripts/release/package_portable.py --allow-dirty --skip-tag --no-commit %*
