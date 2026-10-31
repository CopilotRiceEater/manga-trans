"""설정: 프로젝트 루트의 .env(없으면 환경 변수)에서 읽는다. 항목 설명은 .env.example 참고."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# anime_tools(OCR 모델)는 ANIME_TOOLS_HOME이 없으면 현재 폴더 아래 models/에 받는다 → 프로젝트 폴더로 고정
os.environ.setdefault("ANIME_TOOLS_HOME", str(ROOT))


def _get(name, default=""):
    """빈 값(.env.example을 그대로 복사한 경우 등)은 기본값으로 본다."""
    return os.environ.get(name, "").strip() or default


# 번역 LLM ------------------------------------------------------------------
LLM_API = _get("MT_LLM_API", "ollama").lower()          # ollama | openai (OpenAI 호환 API 전반)
LLM_URL = _get("MT_LLM_URL", "https://ollama.com" if LLM_API == "ollama" else "https://api.openai.com/v1").rstrip("/")
LLM_KEY = _get("MT_LLM_KEY") or _get("OLLAMA_API_KEY" if LLM_API == "ollama" else "OPENAI_API_KEY")
LLM_MODEL = _get("MT_LLM_MODEL", "gemma4:31b")
LLM_WORKERS = int(_get("MT_LLM_WORKERS", "3"))
LLM_CTX = int(_get("MT_LLM_CTX", "32768"))              # ollama num_ctx (설정집 단계는 작품 전체 대사가 들어감)

# 그림 위 글자 지우기 ---------------------------------------------------------
INPAINT = _get("MT_INPAINT", "lama").lower()             # lama | moss | opencv
LAMA_URL = _get("MT_LAMA_URL",
                "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt")
COMFY_URL = _get("MT_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = _get("MT_COMFY_DIR")                         # 있으면 필요할 때 이 ComfyUI를 직접 띄운다
COMFY_PYTHON = _get("MT_COMFY_PYTHON")                   # 비우면 COMFY_DIR의 .venv 파이썬
MOSS_DIT = _get("MT_MOSS_DIT", "MOSS-Image-Base")
MOSS_TE = _get("MT_MOSS_TE", "int8/qwen35_9b_int8_convrot.safetensors")   # huggingface.co/sorryhyun/MOSS-Image-Qwen3.5-9B-int8
MOSS_VAE = _get("MT_MOSS_VAE", "MOSS-Image-Base")
MOSS_LORA = _get("MT_MOSS_LORA", "moss_i2i_sanitize_v3.safetensors")

# GPU 무거운 작업(MOSS) 직전에 실행할 셸 명령 — 예: 다른 GPU 서비스 내리기
BEFORE_GPU = _get("MT_BEFORE_GPU")

# 글꼴 ------------------------------------------------------------------------
FONT_DIR = ROOT / "fonts"
FONTS = {
    "dialogue": _get("MT_FONT_DIALOGUE", str(FONT_DIR / "NanumSquareRoundEB.ttf")),
    "shout": _get("MT_FONT_SHOUT", str(FONT_DIR / "DoHyeon-Regular.ttf")),
    "hand": _get("MT_FONT_HAND", str(FONT_DIR / "NanumPenScript-Regular.ttf")),
}
FONT_FALLBACK = _get("MT_FONT_FALLBACK", str(FONT_DIR / "GothicA1-Bold.ttf"))   # 글꼴에 없는 기호(…, ♪ 등)

# 웹 서버 ---------------------------------------------------------------------
HOST = _get("MT_HOST", "127.0.0.1")
PORT = int(_get("MT_PORT", "8196"))
BOOKS = Path(_get("MT_BOOKS", str(ROOT / "books")))
MODELS = ROOT / "models"
