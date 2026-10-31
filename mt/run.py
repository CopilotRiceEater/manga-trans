"""한 페이지씩 바로 번역하는 간이 경로(설정집·앞뒤 문맥 없음) + 페이지 렌더 함수(plan_page / render_page).
작품 단위 번역은 book.py(명령줄)나 웹 UI를 쓰세요.

  uv run python mt/run.py pages/*.webp [--out out/]

1) 식질 (ocr.py)  2) 번역 (translate.py, gemma4)
3) 흰 바탕 글(말풍선·설명문): 기계 지우기 + 식자 (letter.py)
4) 그림 위 글: 인페인팅(inpaint.py: LaMa 기본 / OpenCV / MOSS) 결과를 그 영역만 합성 + 외곽선 식자
5) 효과음·간판·기호: 원본 유지
"""
import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import expand
from engine import Placed, block_spec, book_base, glyph_est, ink_check, place, score, shrink_budget, try_chain
from letter import Interior, draw_layout, erase, interior, make_area
from ocr import OCR
from translate import translate


def free_text_region(rgb, box):
    """그림 위 글: 글 박스를 조금 넓힌 사각 영역을 sanitize 합성 마스크이자 식자 영역으로."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = box
    p = max(6, int(0.08 * max(x1 - x0, y1 - y0)))
    m = np.zeros((H, W), bool)
    m[max(0, y0 - p):min(H, y1 + p), max(0, x0 - p):min(W, x1 + p)] = True
    return Interior(m, (255, 255, 255), True)


def composite_sanitized(rgb, clean_path, masks):
    clean = np.asarray(Image.open(clean_path).convert("RGB").resize((rgb.shape[1], rgb.shape[0]), Image.LANCZOS))
    m = np.zeros(rgb.shape[:2], np.float32)
    for mk in masks:
        m[mk] = 1
    m = cv2.GaussianBlur(m, (0, 0), 2)[..., None]
    rgb[:] = (rgb * (1 - m) + clean * m).astype(np.uint8)


TRANSLATE_KINDS = ("dialogue", "text")


def plan_page(rgb, blocks):
    """블록마다 (block, interior, mode). mode: bubble(기계 지우기) / sanitize(MOSS 합성) / keep(그대로).
    말풍선 밖 글도 흰 바탕(설명문·나레이션 등)이면 기계 지우기로 처리하고, 그림 위일 때만 sanitize."""
    plan = []
    kept = []
    for b in blocks:
        ko = (b.extra.get("ko") or "").strip()
        # 번역이 원문과 같거나, 원문의 영문 표기(로고·작품명)가 번역에 그대로 남아 있으면 손대지 않는다
        if (b.extra.get("kind") not in TRANSLATE_KINDS or not ko or ko == b.text.strip()
                or _shares_logo(b.text, ko)):
            plan.append((b, None, "keep"))
            kept.append(b.box)
            continue
        plan.append((b, None, None))
    for i, (b, _, mode) in enumerate(plan):
        if mode == "keep":
            continue
        if any(_inside(b.box, k) for k in kept):     # 보존 블록 안에 든 조각(로고 옆 화수 등)도 보존
            plan[i] = (b, None, "keep")
            continue
        it = interior(rgb, b.box, b.bubble)
        if it.ok and b.bubble is None and it.mask.sum() > MAX_FREE_AREA * _area(b.box):
            it = None   # 말풍선 밖인데 지울 영역이 너무 넓다 = 흰 바탕이 아니라 밝은 그림에 번진 것
        plan[i] = (b, it, "bubble") if it is not None and it.ok else (b, free_text_region(rgb, b.box), "sanitize")
    return plan


MAX_FREE_AREA = 4.0


def _area(box):
    return max(1, (box[2] - box[0]) * (box[3] - box[1]))


def _inside(a, b, ratio=0.5):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0) >= ratio * _area(a)


def _shares_logo(ja, ko):
    """원문과 번역에 같은 4자 이상 영문·숫자 표기가 있으면(예: 영문 작품명 로고) 로고·작품명 블록으로 본다."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", ja)
    return any(w in ko for w in words)


def split_shared(plan, shape):
    """붙은 말풍선이나 한 말풍선 안 여러 덩어리처럼 내부 마스크를 공유하는 블록들의 식자 영역을
    '가장 가까운 원문 글 박스' 기준으로 나눈다. 지우기는 원래 마스크로, 식자만 나눈 영역으로 한다."""
    live = [(i, b, it) for i, (b, it, mode) in enumerate(plan) if mode != "keep"]
    out = {i: it for i, _, it in live}
    if len(live) < 2:
        return out
    union = np.zeros(shape, bool)
    for _, _, it in live:
        union |= it.mask
    ys, xs = np.nonzero(union)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    dists = []
    for _, b, _ in live:
        rect = np.ones((y1 - y0, x1 - x0), np.uint8)
        bx0, by0, bx1, by1 = b.box
        rect[max(0, by0 - y0):max(0, by1 - y0), max(0, bx0 - x0):max(0, bx1 - x0)] = 0
        dists.append(cv2.distanceTransform(rect, cv2.DIST_L2, 3))
    owner = np.argmin(np.stack(dists), axis=0)
    for k, (i, b, it) in enumerate(live):
        others = [j for j, (_, _, o) in enumerate(live) if j != k and (o.mask[y0:y1, x0:x1] & it.mask[y0:y1, x0:x1]).any()]
        if not others:
            continue
        m = it.mask.copy()
        m[y0:y1, x0:x1] &= owner == k
        out[i] = Interior(m, it.bg, it.ok)
    return out


def render_page(page: Image.Image, plan, clean_path=None, base=None, median_g=None, shorten_fn=None, faces=None,
                panels=None):
    """plan대로 지우고 식자한 이미지와 블록별 로그.

    sanitize 블록이 있으면 clean_path(sanitize 결과)가 필요. base/median_g는 작품 단위 크기 기준(없으면 이 페이지로 추정).
    shorten_fn(items) → {id: [후보]}: 1~4단계에서 실패한 블록만 모아 한 번 부른다(engine 5단계).
    panels: 칸 목록(말풍선 키우기가 칸 밖으로 못 나가게). 없으면 이 페이지에서 계산."""
    rgb = np.array(page.convert("RGB"))
    san = [it.mask for _, it, mode in plan if mode == "sanitize"]
    if san:
        composite_sanitized(rgb, clean_path, san)
    for b, it, mode in plan:
        if mode == "bubble":
            erase(rgb, it)
    live = [b for b, _, mode in plan if mode != "keep"]
    if base is None:
        base = book_base(live or [b for b, _, _ in plan])
        gs = sorted(glyph_est(b) for b in live) or [base / 0.75]
        median_g = gs[len(gs) // 2]
    if faces is None:
        from letter import DEFAULT_FONTS, face
        faces = {role: face(p) for role, p in DEFAULT_FONTS.items()}
    areas = split_shared(plan, rgb.shape[:2])
    jobs = {}
    for i, (b, it, mode) in enumerate(plan):
        if mode == "keep":
            continue
        area = make_area(areas[i], b.box)
        if area is not None:
            jobs[i] = (b, area, block_spec(b, base, median_g, mode, faces), mode)
    first = {i: try_chain(a, b.extra["ko"], sp) for i, (b, a, sp, _) in jobs.items()}
    placed = {i: Placed(r[0], r[1], jobs[i][0].extra["ko"]) for i, r in first.items() if r}
    # 4단계: 말풍선 키우기 — 안 들어가거나 목표 크기보다 많이 작게 들어간 말풍선
    expansions = grow_balloons(rgb, plan, areas, jobs, first, placed, panels)
    fails = [i for i in jobs if i not in placed]
    # 5단계: 단축 1차(예산 60%) → 그래도 최종 수단까지 간 블록은 2차(예산 40%)
    for ratio in (0.6, 0.4):
        todo = [i for i in fails if i not in placed or placed[i].step == 6]
        if not todo or not shorten_fn:
            break
        alts = shorten_fn([{"id": jobs[i][0].id, "ja": jobs[i][0].text, "ko": jobs[i][0].extra["ko"],
                            "max_chars": shrink_budget(jobs[i][1], jobs[i][2], ratio)} for i in todo])
        for i in todo:
            b, a, sp, _ = jobs[i]
            pl = place(a, b.extra["ko"], sp, alts.get(b.id, ()))
            if i not in placed or pl.step < 6 or pl.layout.size > placed[i].layout.size:
                placed[i] = pl
    for i in fails:
        if i not in placed:
            b, a, sp, _ = jobs[i]
            placed[i] = place(a, b.extra["ko"], sp)
    for ex in expansions.values():
        expand.draw(rgb, ex)
    img = Image.fromarray(rgb)
    log = []
    for i, (b, it, mode) in enumerate(plan):
        entry = {"id": b.id, "det": b.kind, "mode": mode, "kind": b.extra.get("kind"), "box": b.box,
                 "ja": b.text, "ko": b.extra.get("ko"), "font_px": 0}
        if i in expansions:
            entry["grow"] = expansions[i].grow
        if i in jobs:
            _, area, spec, _ = jobs[i]
            pl = placed[i]
            draw_layout(img, pl.layout, stroke=12 if mode == "sanitize" else 0)
            ov, out = ink_check(pl.layout)
            entry.update(overlap_px=ov, outside_px=out)
            entry.update(font_px=pl.layout.size, target=spec.target, step=pl.step, layout=pl.layout.mode, role=spec.role,
                         used=pl.text, score=score(pl, base))
        log.append(entry)
    return img, log


GROW_IF_BELOW = 0.85    # 들어가더라도 목표 크기의 이 비율보다 작으면 키워 본다


def grow_balloons(rgb, plan, areas, jobs, first, placed, panels):
    """말풍선 키우기(engine 4단계). 키운 블록은 placed에 step 4로 넣고 {i: Expansion}을 돌려준다."""
    if panels is None:
        from order import panels as find_panels
        panels = find_panels(rgb)
    H, W = rgb.shape[:2]
    # 다른 블록이 차지한 곳: 말풍선 내부 + 모든 글 박스(효과음·간판 포함)
    occupied = {}
    for i, (b, it, mode) in enumerate(plan):
        m = it.mask.copy() if it is not None else np.zeros((H, W), bool)
        x0, y0, x1, y1 = b.box
        m[y0:y1, x0:x1] = True
        occupied[i] = m
    out = {}
    for i, (b, area, spec, mode) in jobs.items():
        r = first[i]
        small = r is not None and r[0].size < GROW_IF_BELOW * spec.target
        it = plan[i][1]
        if mode != "bubble" or b.bubble is None or areas[i] is not it or not (r is None or small):
            continue        # 말풍선 안 대사만, 붙은 말풍선(공유 영역)은 제외
        cx, cy = (b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2
        panel = next((p for p in panels if p[0] <= cx < p[2] and p[1] <= cy < p[3]), None)
        allowed = expand.allowed_region((H, W), panel, [m for j, m in occupied.items() if j != i])
        for ex in expand.candidates(rgb, it, allowed):
            a2 = make_area(Interior(ex.interior, it.bg, True), b.box)
            r2 = try_chain(a2, b.extra["ko"], spec) if a2 else None
            if r2 and (r is None or r2[0].size >= min(spec.target, r[0].size * 1.15)):
                jobs[i] = (b, a2, spec, mode)
                placed[i] = Placed(r2[0], 4, b.extra["ko"], {"grow": ex.grow})
                out[i] = ex
                occupied[i] = cv2.dilate(ex.interior.astype(np.uint8), np.ones((2 * ex.width + 3,) * 2, np.uint8)).astype(bool)
                break
    return out


def process(ocr, path: Path, out_dir: Path):
    t = {}
    page = Image.open(path).convert("RGB")
    t0 = time.time()
    blocks, _ = ocr.read(page)
    t["ocr"] = time.time() - t0
    t["translate"] = translate(page, blocks) if blocks else 0.0
    plan = plan_page(np.asarray(page), blocks)
    t0 = time.time()
    clean = None
    if any(mode == "sanitize" for _, _, mode in plan):
        import config
        from inpaint import clean as make_clean
        clean = make_clean(path, page, [it.mask for _, it, m in plan if m == "sanitize"],
                           out_dir / f"{path.stem}_clean_{config.INPAINT}.png")
    t["sanitize"] = time.time() - t0
    t0 = time.time()
    img, log = render_page(page, plan, clean)
    t["letter"] = time.time() - t0
    out = out_dir / f"{path.stem}_ko.png"
    img.save(out)
    (out_dir / f"{path.stem}_ko.json").write_text(json.dumps({"times": t, "blocks": log}, ensure_ascii=False, indent=1), encoding="utf-8")
    return out, t, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ocr = OCR()
    print(f"load {time.time() - t0:.1f}s")
    for p in a.pages:
        out, t, log = process(ocr, p, a.out)
        print(f"== {p} → {out}  " + " ".join(f"{k}={v:.1f}s" for k, v in t.items()))
        for r in log:
            print(f"  [{r['id']}] {r['mode']:8s} {r['kind']:8s} px={r['font_px']:3d}  {r['ja']}  →  {r['ko']}")
