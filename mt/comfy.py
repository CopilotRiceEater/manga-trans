"""ComfyUI로 MOSS-Image sanitize v3 실행 (그림 위 글자 지우기, config.INPAINT == "moss"일 때만).

필요: 네이티브 MOSS-Image 지원 ComfyUI + ComfyUI-MOSS-Toolkit(0.10+) + ComfyUI-Spectrum-KSampler,
models/loras에 moss_i2i_sanitize_v3.safetensors (huggingface.co/sorryhyun/MOSS-I2I-LORAs), MOSS-Image-Base, Qwen3.5 TE.
그래프는 sorryhyun 예제 moss-i2i-sanitize.json 그대로: Reference(task auto → 전체 크기 isolated ref),
고정 캡션 "no sfx, no dialogue, no watermark", Spectrum KSampler 30스텝 CFG 4, uncond block 22 skip.
"""
import json
import os
import tempfile
import urllib.parse
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

import config

URL = config.COMFY_URL
SANITIZE_CAPTION = "no sfx, no dialogue, no watermark"
_proc = None
_log = None
_TMP = Path(tempfile.mkdtemp(prefix="mt_comfy_"))    # 업로드·결과 임시 파일(프로세스당 하나)


def alive():
    try:
        urllib.request.urlopen(URL + "/system_stats", timeout=2)
        return True
    except OSError:
        return False


def ensure_running(timeout=180):
    """ComfyUI가 응답하지 않으면 MT_COMFY_DIR의 ComfyUI를 띄운다(설정 없으면 오류). 띄우기 전에 MT_BEFORE_GPU 실행."""
    global _proc
    if alive():
        return
    if not config.COMFY_DIR:
        raise RuntimeError(f"ComfyUI({URL})가 꺼져 있어요. 직접 켜거나 .env에 MT_COMFY_DIR을 지정하세요")
    if config.BEFORE_GPU:
        print(f"[comfy] MT_BEFORE_GPU 실행: {config.BEFORE_GPU}")
        try:
            r = subprocess.run(config.BEFORE_GPU, shell=True, timeout=300)
            if r.returncode:
                print(f"[comfy] MT_BEFORE_GPU 종료 코드 {r.returncode} (계속 진행)")
        except subprocess.TimeoutExpired:
            print("[comfy] MT_BEFORE_GPU 5분 초과 (계속 진행)")
    # 상대 경로면 cwd를 바꾼 뒤 경로가 겹치므로 절대 경로로. resolve()는 venv 파이썬 링크를 따라가 venv를 벗어나므로 쓰지 않는다
    cdir = Path(config.COMFY_DIR).absolute()
    py = config.COMFY_PYTHON or str(cdir / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python"))
    if os.sep in py or "/" in py:
        py = str(Path(py).absolute())
    port = str(urllib.parse.urlparse(URL).port or 8188)
    global _log
    _log = open(config.ROOT / "comfy.log", "ab")
    _proc = subprocess.Popen([py, "main.py", "--listen", "127.0.0.1", "--port", port],
                             cwd=cdir, stdout=_log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if alive():
            print(f"[comfy] 기동 {time.time() - t0:.0f}s")
            return
        if _proc.poll() is not None:
            raise RuntimeError("ComfyUI가 기동 중 종료됨 (comfy.log 확인)")
        time.sleep(2)
    raise TimeoutError("ComfyUI 기동 시간 초과")


def stop():
    """ensure_running이 띄운 ComfyUI만 내린다."""
    global _proc, _log
    if _proc and _proc.poll() is None:
        _proc.terminate()
        _proc.wait(30)
    _proc = None
    if _log:
        _log.close()
        _log = None


def _post(path, data, headers=None):
    req = urllib.request.Request(URL + path, data=data, headers=headers or {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def upload(path: Path) -> str:
    boundary = uuid.uuid4().hex
    name = f"mt_{path.stem}.png"
    buf = _TMP / name
    Image.open(path).convert("RGB").save(buf)
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + buf.read_bytes() + \
           f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    return _post("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})["name"]


def sanitize_graph(image_name, prefix, seed=20, steps=30, stream="bf16", resolution=1024):
    return {
        "17": {"class_type": "MOSSImageLoader", "inputs": {
            "dit": config.MOSS_DIT, "weight_dtype": "default",
            "text_encoder": config.MOSS_TE, "vae": config.MOSS_VAE,
            "text_device": "default", "int8_matmul": True, "stream": stream}},
        "23": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "3": {"class_type": "MOSSi2iReference", "inputs": {
            "model": ["17", 0], "vae": ["17", 2], "image": ["23", 0],
            "lora_name": config.MOSS_LORA, "strength": 1.0, "resolution": resolution,
            "ref_scale": 1, "task": "auto", "ref_cache": "gpu"}},
        "24": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["17", 1], "text": SANITIZE_CAPTION}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["17", 1], "text": ""}},
        "5": {"class_type": "MossSpectrumKSampler", "inputs": {
            "model": ["3", 0], "positive": ["24", 0], "negative": ["4", 0], "latent_image": ["3", 1],
            "seed": seed, "steps": steps, "cfg": 4.0, "sampler_name": "er_sde_cns", "scheduler": "normal",
            "denoise": 1.0, "shift": 5.0, "uncond_skip_layers": "22", "spectrum": True, "flex_window": 0.5,
            "compile": False}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["17", 2]}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}},
    }


def _fetch(info):
    """결과 이미지를 /view로 받아 임시 파일로(ComfyUI가 다른 PC에 있어도 동작)."""
    q = urllib.parse.urlencode({"filename": info["filename"], "subfolder": info.get("subfolder", ""),
                                "type": info.get("type", "output")})
    data = urllib.request.urlopen(f"{URL}/view?{q}", timeout=120).read()
    out = _TMP / info["filename"]
    out.write_bytes(data)
    return out


def run(graph, timeout=600):
    pid = _post("/prompt", json.dumps({"prompt": graph}).encode())["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = json.loads(urllib.request.urlopen(f"{URL}/history/{pid}", timeout=30).read())
        if pid in hist:
            h = hist[pid]
            if h["status"].get("status_str") == "error":
                raise RuntimeError(json.dumps(h["status"]["messages"])[-2000:])
            imgs = [i for o in h["outputs"].values() for i in o.get("images", [])]
            return [_fetch(i) for i in imgs], time.time() - t0
        time.sleep(0.5)
    raise TimeoutError(pid)


def sanitize(page: Path, prefix: str, **kw):
    """페이지 전체를 sanitize한 이미지(1024² 면적 상한, 원본 비율) 경로와 걸린 시간."""
    paths, dt = run(sanitize_graph(upload(page), prefix, **kw))
    return paths[0], dt


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        out, dt = sanitize(Path(p), f"mt/{Path(p).stem}_sanitize")
        print(f"{p} → {out}  {dt:.1f}s")
