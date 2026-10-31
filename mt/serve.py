"""웹 UI 실행: uv run python mt/serve.py  (호스트·포트는 .env의 MT_HOST / MT_PORT, 기본 127.0.0.1:8196)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402

import config  # noqa: E402

if __name__ == "__main__":
    print(f"만화 번역 → http://{'127.0.0.1' if config.HOST in ('0.0.0.0', '') else config.HOST}:{config.PORT}")
    if config.HOST not in ("127.0.0.1", "localhost", "::1"):
        print(f"[경고] MT_HOST={config.HOST}: 인증이 없어 같은 네트워크의 누구나 원본을 받고, 번역(API 비용)을 돌리고, "
              "작품을 지울 수 있어요. 믿을 수 있는 네트워크에서만 쓰세요.")
    uvicorn.run("server:app", host=config.HOST, port=config.PORT, app_dir=str(Path(__file__).resolve().parent))
