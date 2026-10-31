#!/bin/bash
# 웹 UI 실행 (설정은 .env). Windows는 run.bat.
cd "$(dirname "$0")" && exec uv run python mt/serve.py "$@"
