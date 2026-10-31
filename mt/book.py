"""작품(여러 페이지) 단위 파이프라인. 단계별 결과를 books/<이름>/에 JSON으로 남겨 다시 돌릴 때 재사용한다.

  uv run python mt/book.py books/<이름> ocr        # src/*.png → ocr.json (+ debug/order_*.jpg)
  uv run python mt/book.py books/<이름> bible      # 전체 대사 → bible.json (줄거리·인물·호칭·용어)
  uv run python mt/book.py books/<이름> translate  # 설정집+앞뒤 문맥+페이지 이미지 → trans.json
  uv run python mt/book.py books/<이름> render     # → ko/*.png
"""
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from letter import FALLBACK
from ocr import Block
from order import merge_free, sort_blocks


IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".avif", ".bmp", ".gif", ".tif", ".tiff")


def pages(book: Path):
    return sorted((p for p in (book / "src").iterdir() if p.suffix.lower() in IMAGE_EXT),
                  key=lambda p: _natural_key(p.name))


def _noop(*a):
    pass


def load_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def save_json(path: Path, obj):
    atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=1, default=int))


def atomic_write(path: Path, text: str):
    """고유한 임시 파일에 쓰고 바꿔 끼운다(웹 UI가 쓰는 도중의 파일을 읽지 않게).
    Windows에서는 다른 쪽이 읽느라 열어 둔 파일을 잠깐 못 바꿀 수 있어 몇 번 다시 시도한다."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def blocks_from(d):
    return [Block(**{k: (tuple(v) if k in ("box", "bubble") and v else v) for k, v in b.items()}) for b in d]


def stage_ocr(book: Path, ocr=None, progress=_noop):
    """progress(완료 수, 전체, 메시지). ocr: 이미 올린 OCR 인스턴스(서버에서 재사용)."""
    if ocr is None:
        from ocr import OCR
        ocr = OCR()
    out, dbg = {}, book / "debug"
    dbg.mkdir(exist_ok=True)
    font = ImageFont.truetype(FALLBACK.path, 22, index=FALLBACK.index)
    plist = pages(book)
    for k, p in enumerate(plist):
        t0 = time.time()
        page = Image.open(p).convert("RGB")
        blocks, _ = ocr.read(page)
        blocks, ps = sort_blocks(np.asarray(page), blocks)
        blocks = merge_free(blocks)
        for i, b in enumerate(blocks):
            b.id = i
        out[p.name] = {"size": page.size, "panels": ps, "blocks": [asdict(b) for b in blocks]}
        # 순서 확인용: 칸 테두리 + 블록 번호
        vis = page.copy()
        d = ImageDraw.Draw(vis)
        for x0, y0, x1, y1 in ps:
            d.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 160, 255), width=3)
        for b in blocks:
            col = (230, 0, 0) if b.kind == "text_bubble" else (0, 170, 0)
            d.rectangle(b.box, outline=col, width=2)
            d.text((b.box[2] - 4, b.box[1] - 4), str(b.id), font=font, fill=col, anchor="rb",
                   stroke_width=3, stroke_fill=(255, 255, 255))
        vis.save(dbg / f"order_{p.stem}.jpg", quality=85)
        print(f"{p.name}: {len(blocks)} blocks, {len(ps)} panels  {time.time() - t0:.1f}s")
        progress(k + 1, len(plist), p.name)
    save_json(book / "ocr.json", out)


def stage_bible(book: Path, progress=_noop):
    from translate import make_bible
    ocr = load_json(book / "ocr.json")
    names = list(ocr)
    # 인물 파악용 그림: 대사 많은 페이지 위주로 최대 8장
    pick = sorted(sorted(names, key=lambda n: -len(ocr[n]["blocks"]))[:8], key=names.index)
    images = [(n, Image.open(book / "src" / n)) for n in pick]
    bible, dt = make_bible([(name, d["blocks"]) for name, d in ocr.items()], images)
    bible = validate_bible(bible, ocr)
    save_json(book / "bible.json", bible)
    progress(1, 1, "설정집")
    print(f"bible {dt:.1f}s")
    print(json.dumps(bible, ensure_ascii=False, indent=1))


def validate_bible(bible, ocr):
    """설정집 자동 검증: 인물은 원문 이름이 실제 글에 나와야 하고, 그 이름 하나만 덩그러니 있는 블록
    (작가명·크레딧)에만 나오면 인물이 아니라고 본다. 빠진 이름은 terms에 '표기만'으로 남긴다."""
    texts = [b["text"] for d in ocr.values() for b in d["blocks"] if b.get("text")]
    for k in ("characters", "address", "terms"):     # LLM이 형식을 틀리게 준 항목은 버리고, 값은 문자열로
        items = bible.get(k) if isinstance(bible.get(k), list) else []
        bible[k] = [{f: v if isinstance(v, str) else "" if v is None else json.dumps(v, ensure_ascii=False)
                     for f, v in x.items()} for x in items if isinstance(x, dict)]
    keep, dropped = [], []
    for c in bible["characters"]:
        names = [n for n in re.split(r"[/・()（）、, ]+", str(c.get("ja") or "")) if n]
        hits = [t for t in texts for n in names if n in t]
        credit_only = hits and all(t.strip() in names for t in hits)
        (keep if hits and not credit_only else dropped).append(c)
    bible["characters"] = keep
    if dropped:
        bible.setdefault("terms", []).extend(
            {"ja": c.get("ja"), "ko": c.get("ko"), "note": "인물 아님(대사에 안 나오거나 크레딧에만 나옴), 표기만 참고"}
            for c in dropped)
        names_ko = {c.get("ko") for c in dropped}
        bible["address"] = [a for a in bible.get("address", [])
                            if not any(n and n in (a.get("from", "") + a.get("to", "")) for n in names_ko)]
        print("설정집 검증: 인물에서 제외", [c.get("ja") for c in dropped])
    return bible


def shorten_cached(book: Path, name: str, bible):
    """engine 4단계용 단축 함수(페이지별 캐시: short.json)."""
    from translate import shorten
    cache = load_json(book / "short.json", {})

    def fn(items):
        page = cache.setdefault(name, {})
        key = lambda it: f"{it['id']}|{it['ko']}|{it['max_chars']}"
        need = [it for it in items if key(it) not in page]
        if need:
            got, dt = shorten(need, bible)
            print(f"  {name} 단축 요청 {len(need)}개 {dt:.1f}s")
            for it in need:
                page[key(it)] = got.get(it["id"], [])
            save_json(book / "short.json", cache)
        return {it["id"]: page.get(key(it), []) for it in items}
    return fn


def stage_translate(book: Path, workers=None, progress=_noop):
    from concurrent.futures import ThreadPoolExecutor
    from translate import script_text, translate
    workers = workers or config.LLM_WORKERS
    ocr = load_json(book / "ocr.json")
    bible = load_json(book / "bible.json")
    names = list(ocr)
    trans = load_json(book / "trans.json", {})

    def one(i):
        name = names[i]
        blocks = blocks_from(ocr[name]["blocks"])
        if not blocks:
            return name, [], 0.0
        prev = script_text([(names[i - 1], ocr[names[i - 1]]["blocks"])]) if i > 0 else ""
        nxt = script_text([(names[i + 1], ocr[names[i + 1]]["blocks"])]) if i + 1 < len(names) else ""
        dt = translate(Image.open(book / "src" / name), blocks, bible, prev, nxt)
        return name, [{"id": b.id, "kind": b.extra["kind"], "ko": b.extra["ko"]} | ({"missing": True} if b.extra["missing"] else {})
                      for b in blocks], dt

    t0 = time.time()
    # 아직 안 한 페이지 + 지난번에 번역이 빠진 블록이 있던 페이지
    todo = [i for i, n in enumerate(names) if n not in trans or any(t.get("missing") for t in trans[n])]
    with ThreadPoolExecutor(workers) as ex:
        for k, (name, items, dt) in enumerate(ex.map(one, todo)):
            trans[name] = items
            miss = [t["id"] for t in items if t.get("missing")]
            print(f"{name}: {len(items)} items  {dt:.1f}s" + (f"  번역 빠짐 {miss} (원문 유지)" if miss else ""))
            save_json(book / "trans.json", trans)
            progress(k + 1, len(todo), name)
    print(f"translate total {time.time() - t0:.1f}s ({len(todo)} pages, {workers} workers)")


def page_blocks(book, name, ocr, trans):
    blocks = blocks_from(ocr[name]["blocks"])
    by_id = {t["id"]: t for t in trans.get(name, [])}
    for b in blocks:
        t = by_id.get(b.id, {})
        b.extra.update(kind=t.get("kind"), ko=t.get("ko", ""))
    return blocks


def _clean_path(cdir: Path, name: str):
    return cdir / f"{Path(name).stem}_{config.INPAINT}.png"


def stage_render(book: Path, fonts=None, tag=None, progress=_noop):
    """fonts: {역할: 글꼴 파일} (dialogue / shout / hand). 없으면 기본 글꼴. tag가 있으면 ko_<tag>/에 저장."""
    import letter
    from run import plan_page, render_page
    ocr, trans = load_json(book / "ocr.json"), load_json(book / "trans.json", {})
    out = book / (f"ko_{tag}" if tag else "ko")
    faces = {role: letter.face(p) for role, p in {**letter.DEFAULT_FONTS, **(fonts or {})}.items()}
    out.mkdir(exist_ok=True)
    # 1차: 페이지마다 계획 → 작품 글자 크기 통계용 블록만 모으고, 그림 위 글은 지워 둔다(clean/ 캐시).
    # 페이지 이미지·마스크는 페이지 크기라 수백 쪽을 다 들고 있으면 RAM이 모자라서 2차에서 다시 계획한다.
    live, need = [], []
    cdir = book / "clean"
    for k, name in enumerate(ocr):
        page = Image.open(book / "src" / name).convert("RGB")
        plan = plan_page(np.asarray(page), page_blocks(book, name, ocr, trans))
        live += [b for b, _, m in plan if m != "keep"]
        masks = [it.mask for _, it, m in plan if m == "sanitize"]
        if masks:
            from inpaint import clean
            need.append(name)
            cdir.mkdir(exist_ok=True)
            print(f"그림 위 글 지우기({config.INPAINT}): {name}")
            clean(book / "src" / name, page, masks, _clean_path(cdir, name))
        del page, plan, masks
        progress(k + 1, 2 * len(ocr), name)
    from engine import book_base, glyph_est
    base = book_base(live)
    gs = sorted(glyph_est(b) for b in live)
    median_g = gs[len(gs) // 2] if gs else base / 0.75
    del live
    bible = load_json(book / "bible.json")
    print(f"기본 글자 크기 {base}px (하한 {round(base * 0.8)}px)")
    report = {}
    for k, name in enumerate(ocr):
        page = Image.open(book / "src" / name).convert("RGB")
        plan = plan_page(np.asarray(page), page_blocks(book, name, ocr, trans))
        cl = _clean_path(cdir, name) if name in need else None
        img, log = render_page(page, plan, cl, base, median_g, shorten_cached(book, name, bible), faces,
                               ocr[name].get("panels"))
        img.save(out / f"{Path(name).stem}.png")
        report[name] = log
        progress(len(ocr) + k + 1, 2 * len(ocr), name)
    save_json(out / "render.json", report)
    scores, unplaced = [], []
    for name, log in report.items():
        for r in log:
            if r["mode"] == "keep" and not r.get("unplaced"):
                continue
            if r.get("unplaced") or r.get("score") is None:      # 식자 영역이 안 나와 원문 그대로 둔 블록
                unplaced.append((name, r["id"]))
                continue
            scores.append(r["score"])
            used = "" if r["used"] == r["ko"] else f"  ⇒ {r['used']}"
            print(f"{name} [{r['id']:2d}] s{r['step']} {r['layout']:5s} {r['font_px']:3d}/{r['target']:<3d} "
                  f"점수 {r['score']:5.1f}  {r['ko']}{used}")
    bad = [(n, r["id"], r["overlap_px"], r["outside_px"]) for n, l in report.items() for r in l
           if r.get("overlap_px") or r.get("outside_px", 0) > 20]
    print(f"잉크 검사: 겹침/삐져나옴 {len(bad)}건 {bad[:10]}")
    scores.sort()
    steps = {k: sum(1 for l in report.values() for r in l if r.get('step') == k) for k in range(1, 7)}
    avg = sum(scores) / len(scores) if scores else 100.0
    print(f"\n블록 {len(scores)}개  평균 {avg:.1f}  최저 {scores[:5]}  단계별 {steps}")
    grown = sum(1 for l in report.values() for r in l if r.get("grow"))
    print(f"말풍선 키움 {grown}개" + (f"  식자 못 함 {unplaced}" if unplaced else ""))
    summary = {"blocks": len(scores), "unplaced": len(unplaced), "avg": round(avg, 1), "min": scores[:5], "steps": steps, "grown": grown,
               "ink_issues": len(bad), "base_px": base, "sanitized": need}
    save_json(out / "summary.json", summary)
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[2] not in ("ocr", "bible", "translate", "render"):
        sys.exit(__doc__)
    book, stage = Path(sys.argv[1]), sys.argv[2]
    if stage == "render":
        # 추가 인자: 역할=글꼴파일 (예: dialogue=.../Pretendard-ExtraBold.otf shout=... hand=...), tag=이름
        kv = dict(a.split("=", 1) for a in sys.argv[3:])
        tag = kv.pop("tag", None)
        stage_render(book, kv, tag)
    else:
        {"ocr": stage_ocr, "bible": stage_bible, "translate": stage_translate}[stage](book)
