@echo off
cd /d "%~dp0"
uv run python mt\serve.py %*
