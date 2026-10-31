"""만화 번역 웹 UI 서버.

  uv run python mt/serve.py      → http://127.0.0.1:8196 (호스트·포트는 .env의 MT_HOST / MT_PORT)

작품(book) = books/<id>/ (src/ 원본 PNG, ko/ 번역본, *.json 단계 캐시, meta.json 상태).
작업은 GPU 하나를 쓰므로 워커 스레드 1개가 큐를 차례로 처리한다(OCR 모델은 한 번 올려 재사용).
"""
import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from queue import Queue
from urllib.parse import quote, urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageOps
from starlette.background import BackgroundTask

import book as bk
import config

BOOKS = config.BOOKS
WEB = config.ROOT / "web"
EXPORT_FMT = {"png": ("PNG", ".png", {}), "avif": ("AVIF", ".avif", {"quality": 80}),
              "webp": ("WEBP", ".webp", {"quality": 90}), "jpg": ("JPEG", ".jpg", {"quality": 92})}
STAGES = [("ocr", "글자 찾기"), ("bible", "설정집"), ("translate", "번역"), ("render", "식자")]
THUMB_W = (240, 360, 720)
# 업로드 한도(압축 해제 후 기준)
MAX_PAGES = 3000
MAX_IMAGE_BYTES = 200 * 2**20
MAX_TOTAL_BYTES = 8 * 2**30
MAX_PIXELS = 60_000_000           # 한 장 약 6000×10000
MAX_ZIP_ENTRIES = 50_000          # 이미지 아닌 항목 포함

app = FastAPI()
_lock = threading.Lock()          # meta.json 쓰기
_jobs = threading.Lock()          # 큐 등록·삭제 (_pending 보호)
_pending = set()                  # 큐에 있거나 작업 중인 작품
_export = threading.Semaphore(2)
_queue: Queue = Queue()
_state = {"current": None}
_ocr = None
_LOOPBACK = config.HOST in ("127.0.0.1", "localhost", "::1")


@app.middleware("http")
async def _same_origin(request: Request, call_next):
    """다른 사이트가 브라우저를 거쳐 이 서버를 부르는 것(CSRF, DNS 리바인딩)을 막는다."""
    host = request.headers.get("host", "")
    origin = request.headers.get("origin")
    if origin and urlparse(origin).netloc != host:
        return JSONResponse({"detail": "다른 사이트에서 온 요청은 받지 않아요"}, 403)
    if _LOOPBACK and urlparse("//" + host).hostname not in ("127.0.0.1", "localhost", "::1"):
        return JSONResponse({"detail": "Host 헤더가 맞지 않아요"}, 403)
    if int(request.headers.get("content-length") or 0) > MAX_TOTAL_BYTES:   # 본문을 받기 전에 거절
        return JSONResponse({"detail": f"업로드가 너무 커요 (최대 {MAX_TOTAL_BYTES >> 30}GB)"}, 413)
    return await call_next(request)


# ---------- 메타 ----------

def meta_path(bid):
    return BOOKS / bid / "meta.json"


def load_meta(bid):
    d = BOOKS / bid
    m = bk.load_json(meta_path(bid), None)
    if m is None:   # 웹 UI 이전에 만든 작품(CLI) — 파일로 상태 추정
        done = (d / "ko" / "render.json").exists()
        m = {"id": bid, "title": bid, "created": d.stat().st_mtime, "status": "done" if done else "new"}
    m.setdefault("summary", bk.load_json(d / "ko" / "summary.json", None))
    ps = bk.pages(d) if (d / "src").exists() else []
    m["pages"], m["first"] = len(ps), (ps[0].name if ps else None)
    return m


def save_meta(bid, **kw):
    with _lock:
        m = bk.load_json(meta_path(bid), None) or {"id": bid}
        m.update(kw)
        bk.atomic_write(meta_path(bid), json.dumps(m, ensure_ascii=False, indent=1, default=str))
        return m


def book_dir(bid):
    if not re.fullmatch(r"[\w\-]+", bid) or not (BOOKS / bid).is_dir():
        raise HTTPException(404, "없는 작품")
    return BOOKS / bid


# ---------- 업로드 ----------

def _collect(files):
    """업로드에서 이미지 항목만 [(원래 이름, 크기, 읽기 함수)]로, 이름 자연 정렬. zip은 항목별로 읽는다."""
    entries = []
    for f in files:
        name = f.filename or ""
        if name.lower().endswith((".zip", ".cbz")):
            try:
                z = zipfile.ZipFile(f.file)
            except zipfile.BadZipFile:
                raise HTTPException(400, f"ZIP을 열 수 없어요: {name}")
            if len(z.infolist()) > MAX_ZIP_ENTRIES:
                raise HTTPException(413, f"ZIP 안 파일이 너무 많아요 (최대 {MAX_ZIP_ENTRIES}개): {name}")
            for info in z.infolist():
                n = info.filename
                if info.is_dir() or n.startswith("__MACOSX") or not n.lower().endswith(bk.IMAGE_EXT):
                    continue
                if info.flag_bits & 0x1:
                    raise HTTPException(400, f"암호가 걸린 ZIP은 안 돼요: {name}")
                entries.append((n, info.file_size, lambda z=z, info=info: z.read(info)))
        elif name.lower().endswith(bk.IMAGE_EXT):
            f.file.seek(0, 2)
            size = f.file.tell()
            f.file.seek(0)
            entries.append((name, size, f.file.read))
    if len(entries) > MAX_PAGES:
        raise HTTPException(413, f"이미지가 너무 많아요 (최대 {MAX_PAGES}장)")
    big = [n for n, size, _ in entries if size > MAX_IMAGE_BYTES]
    if big:
        raise HTTPException(413, f"이미지 한 장이 너무 커요 (최대 {MAX_IMAGE_BYTES >> 20}MB): {big[0]}")
    if sum(size for _, size, _ in entries) > MAX_TOTAL_BYTES:
        raise HTTPException(413, f"전체 크기가 너무 커요 (최대 {MAX_TOTAL_BYTES >> 30}GB)")
    return sorted(entries, key=lambda e: bk._natural_key(e[0]))


def _decode(name, read):
    try:
        im = Image.open(io.BytesIO(read()))
        if im.width * im.height > MAX_PIXELS:
            raise HTTPException(413, f"이미지 해상도가 너무 커요: {name} ({im.width}×{im.height})")
        return ImageOps.exif_transpose(im).convert("RGB")     # avif/webp/gif 등 → 내부는 전부 PNG
    except HTTPException:
        raise
    except Exception:   # noqa: BLE001 — 손상·가짜 이미지, 깨진 zip 항목 등
        raise HTTPException(400, f"이미지를 읽을 수 없어요: {name}")


@app.post("/api/books")
def create_book(title: str = Form(""), autorun: bool = Form(True), files: list[UploadFile] = File(...)):
    imgs = _collect(files)
    if not imgs:
        raise HTTPException(400, "이미지가 없어요 (png, jpg, webp, avif, zip/cbz)")
    bid = time.strftime("b%y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    BOOKS.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".up-", dir=BOOKS))   # 다 풀고 검증한 뒤에 작품으로 확정
    try:
        src = tmp / "src"
        src.mkdir()
        width = max(3, len(str(len(imgs))))
        for i, (name, _, read) in enumerate(imgs, 1):
            _decode(name, read).save(src / f"{i:0{width}d}.png")
        meta = {"id": bid, "title": title or Path(files[0].filename or bid).stem, "created": time.time(),
                "status": "new", "source": [n for n, _, _ in imgs]}
        (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.rename(BOOKS / bid)     # 메타까지 다 쓴 뒤에 한 번에 공개
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if autorun:
        enqueue(bid, "all")
    return load_meta(bid)


# ---------- 작업 큐 ----------

def enqueue(bid, mode):
    with _jobs:
        if bid in _pending:
            raise HTTPException(409, "이미 대기 중이거나 작업 중이에요")
        if not (BOOKS / bid).is_dir():
            raise HTTPException(404, "없는 작품")
        _pending.add(bid)
        try:
            save_meta(bid, status="queued", mode=mode, error=None, stage=None, progress=None)
            _queue.put((bid, mode))
        except BaseException:
            _pending.discard(bid)
            raise


def _worker():
    while True:
        bid, mode = _queue.get()
        try:
            _run_job(bid, mode)
        except Exception:   # noqa: BLE001 — 상태 기록까지 실패해도 워커는 살아 있어야 한다
            import traceback
            traceback.print_exc()
        finally:
            _state["current"] = None
            with _jobs:
                _pending.discard(bid)


def _run_job(bid, mode):
    global _ocr
    d = BOOKS / bid
    if not d.is_dir():
        return
    _state["current"] = bid
    t0 = time.time()
    try:
        if mode == "all":
            for f in ("ocr.json", "bible.json", "trans.json", "short.json"):
                (d / f).unlink(missing_ok=True)
            shutil.rmtree(d / "clean", ignore_errors=True)
        # all: 처음부터 / fix: OCR은 두고 설정집 고치기 + 빠진·일본어 남은 번역만 다시 + 식자 / render: 식자만
        stages = {"all": STAGES, "fix": STAGES[1:], "render": STAGES[3:]}[mode]
        for key, label in stages:
            save_meta(bid, status="running", stage=key, stage_label=label, progress=[0, 1])

            def prog(done, total, msg="", key=key):
                save_meta(bid, progress=[done, total], progress_msg=msg)
            if key == "ocr":
                if _ocr is None:
                    from ocr import OCR
                    _ocr = OCR()
                bk.stage_ocr(d, ocr=_ocr, progress=prog)
            elif key == "bible":
                bk.stage_bible(d, progress=prog, refresh=mode == "all")
            elif key == "translate":
                bk.stage_translate(d, progress=prog)
            else:
                summary = bk.stage_render(d, progress=prog)
                save_meta(bid, summary=summary)
        save_meta(bid, status="done", stage=None, finished=time.time(), elapsed=round(time.time() - t0, 1))
    except Exception as e:   # noqa: BLE001 — 실패는 작품 상태로 보여 준다
        import traceback
        traceback.print_exc()
        save_meta(bid, status="error", error=f"{type(e).__name__}: {e}")


threading.Thread(target=_worker, daemon=True).start()


@app.on_event("startup")
def _resume():
    """서버가 작업 도중 꺼졌던 작품은 다시 큐에 넣고, 업로드·내보내기 도중 꺼져 남은 임시 파일은 지운다."""
    BOOKS.mkdir(exist_ok=True)
    for d in BOOKS.glob(".up-*"):
        shutil.rmtree(d, ignore_errors=True)
    for f in BOOKS.glob("*/.export-*.zip"):
        f.unlink(missing_ok=True)
    for d in sorted(BOOKS.iterdir()):
        if d.name.startswith("."):
            continue
        m = bk.load_json(d / "meta.json", None)
        if m and m.get("status") in ("queued", "running"):
            enqueue(d.name, m.get("mode", "all"))


@app.on_event("shutdown")
def _shutdown():
    from comfy import stop
    stop()


# ---------- 조회 ----------

@app.get("/api/books")
def list_books():
    out = [load_meta(d.name) for d in BOOKS.iterdir()
           if d.is_dir() and not d.name.startswith(".") and (d / "src").is_dir()]
    return sorted(out, key=lambda m: -float(m.get("created") or 0))


@app.get("/api/books/{bid}")
def get_book(bid: str):
    d = book_dir(bid)
    m = load_meta(bid)
    report = bk.load_json(d / "ko" / "render.json", {})
    pages = []
    for p in bk.pages(d):
        blocks = [{"id": r["id"], "ja": r["ja"], "ko": r.get("used") or r["ko"], "kind": r["kind"], "mode": r["mode"],
                   "step": r.get("step"), "score": r.get("score")} for r in report.get(p.name, [])]
        ko = _ko_path(d, p.name)
        pages.append({"name": p.name, "translated": ko.exists(), "v": int(ko.stat().st_mtime) if ko.exists() else 0,
                      "blocks": blocks})
    m.update(pages_detail=pages, bible=bk.load_json(d / "bible.json", None), queue_current=_state["current"])
    return m


@app.post("/api/books/{bid}/run")
def run_book(bid: str, mode: str = Form("all")):
    book_dir(bid)
    if mode not in ("all", "fix", "render"):
        raise HTTPException(400, "mode는 all / fix / render")
    enqueue(bid, mode)
    return load_meta(bid)


@app.post("/api/books/{bid}/rename")
def rename_book(bid: str, title: str = Form(...)):
    book_dir(bid)
    return save_meta(bid, title=title)


@app.delete("/api/books/{bid}")
def delete_book(bid: str):
    d = book_dir(bid)
    with _jobs:
        if bid in _pending:
            raise HTTPException(409, "대기 중이거나 작업 중인 작품은 지울 수 없어요")
        shutil.rmtree(d)
    return {"ok": True}


def _ko_path(d, name):
    """번역본은 원본 확장자와 상관없이 <이름>.png (CLI로 jpg 등을 넣은 작품도)."""
    return d / "ko" / f"{Path(name).stem}.png"


@app.get("/api/books/{bid}/img/{kind}/{name}")
def page_image(bid: str, kind: str, name: str, w: int = 0):
    d = book_dir(bid)
    if kind not in ("src", "ko") or name not in {p.name for p in bk.pages(d)}:
        raise HTTPException(404)
    if w and w not in THUMB_W:
        raise HTTPException(400, f"w는 {THUMB_W} 중 하나")
    path = d / "src" / name if kind == "src" else _ko_path(d, name)
    if not path.exists():
        raise HTTPException(404)
    if not w:
        return FileResponse(path, headers={"Cache-Control": "no-cache"})
    thumb = d / "thumb" / f"{kind}_{w}_{Path(name).stem}.webp"
    if not thumb.exists() or thumb.stat().st_mtime < path.stat().st_mtime:
        thumb.parent.mkdir(exist_ok=True)
        im = Image.open(path).convert("RGB")
        im.thumbnail((w, w * 3))
        im.save(thumb, "WEBP", quality=82)
    return FileResponse(thumb, headers={"Cache-Control": "no-cache"})


@app.get("/api/books/{bid}/export")
def export(bid: str, fmt: str = "png", which: str = "ko"):
    d = book_dir(bid)
    if fmt not in EXPORT_FMT or which not in ("ko", "src"):
        raise HTTPException(400, "fmt: png / avif / webp / jpg")
    pil_fmt, ext, opts = EXPORT_FMT[fmt]
    files = [p for p in (_ko_path(d, p.name) for p in bk.pages(d)) if p.exists()] if which == "ko" else bk.pages(d)
    if not files:
        raise HTTPException(404, "번역본이 아직 없어요")
    if not _export.acquire(blocking=False):     # ZIP 만들기(CPU·디스크)를 동시에 2개까지
        raise HTTPException(429, "다른 내보내기가 진행 중이에요. 잠시 뒤에 다시 해 주세요")
    tmp = None
    try:
        fd, name = tempfile.mkstemp(suffix=".zip", prefix=".export-", dir=d)   # 메모리 대신 임시 파일
        os.close(fd)
        tmp = Path(name)
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            for f in files:
                if f.suffix.lower() == ext:
                    z.write(f, f.name)
                else:
                    b = io.BytesIO()
                    Image.open(f).convert("RGB").save(b, pil_fmt, **opts)
                    z.writestr(f.stem + ext, b.getvalue())
        title = re.sub(r'[\\/:*?"<>|]+', "_", load_meta(bid).get("title") or bid)
    except BaseException:
        if tmp:
            tmp.unlink(missing_ok=True)
        raise
    finally:
        _export.release()
    name = quote(f"{title}_{'번역' if which == 'ko' else '원본'}_{fmt}.zip")
    return FileResponse(tmp, media_type="application/zip", background=BackgroundTask(tmp.unlink, missing_ok=True),
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{name}"})


@app.get("/api/status")
def status():
    return {"current": _state["current"], "queued": _queue.qsize()}


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text(encoding="utf-8")
