"""그림 위 글자 지우기(인페인팅) 백엔드. config.INPAINT로 고른다.

- lama  : LaMa(big-lama, Apache-2.0) — 처음 쓸 때 models/lama/big-lama.pt(약 200MB)를 받는다. GPU 1GB 안팎, CPU도 가능.
- opencv: cv2.inpaint(Telea) — 모델 없음, 빠르지만 넓은 영역은 뭉개진다.
- moss  : MOSS-Image + sanitize LoRA를 ComfyUI로(comfy.py) — 가장 자연스럽지만 VRAM ~26GB와 ComfyUI 세팅이 필요.

clean(page_path, page, masks) → 지운 결과 이미지 경로(페이지와 같은 크기가 아니어도 됨: 합성할 때 맞춘다).
"""
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config

_lama = None


def _union(masks, shape):
    m = np.zeros(shape, bool)
    for mk in masks:
        m |= mk
    return m


def _lama_model():
    global _lama
    if _lama is None:
        import torch
        path = config.MODELS / "lama" / "big-lama.pt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[inpaint] LaMa 받는 중 → {path}")
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(config.LAMA_URL, timeout=60) as r, open(tmp, "wb") as f:   # 60초 무응답이면 실패
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(path)
        dev = "cpu"
        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 2**30 >= 1.5:   # GPU가 꽉 차 있으면 CPU
            dev = "cuda"
        _lama = (torch.jit.load(str(path), map_location=dev).eval(), dev)
    return _lama


def lama(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch
    model, dev = _lama_model()
    H, W = mask.shape
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    img = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    m = np.pad(mask.astype(np.float32), ((0, ph), (0, pw)))
    with torch.no_grad():
        t_img = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(dev)
        t_m = torch.from_numpy(m)[None, None].to(dev)
        out = model(t_img, (t_m > 0.5).float())
    out = out[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
    return out[:H, :W]


def clean(page_path: Path, page: Image.Image, masks, cache: Path) -> Path:
    """지운 결과를 cache 파일로 저장하고 경로를 돌려준다(이미 있으면 그대로)."""
    if cache.exists():
        return cache
    backend = config.INPAINT
    if backend == "moss":
        import shutil
        from comfy import ensure_running, sanitize
        ensure_running()
        out, dt = sanitize(page_path, f"mt/{page_path.parent.parent.name}_{page_path.stem}")
        shutil.copy(out, cache)
        print(f"  {page_path.name} MOSS sanitize {dt:.1f}s")
        return cache
    rgb = np.array(page.convert("RGB"))
    m = _union(masks, rgb.shape[:2])
    m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if backend == "opencv":
        out = cv2.inpaint(rgb, m.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    else:
        out = lama(rgb, m)
    Image.fromarray(out).save(cache)
    return cache
