"""말풍선 안: 기계적으로 지우고 한글 배치(fit)·그리기(draw_layout). 어느 배치를 쓸지는 engine.py가 정한다.

지우기: bubble 박스 안에서 밝은 픽셀의 연결 성분 중 글 박스와 겹치는 것을 고르고, 구멍(=글자 획, 후리가나)을 메운
내부 마스크를 만든 뒤 바탕색으로 칠한다. 말풍선 테두리는 어두운 픽셀이라 마스크 밖이라 남는다.
배치: 주어진 크기 목록 중 줄들이 영역 안에 다 들어가는 가장 큰 크기를 찾는다(줄 수를 바꿔가며, 줄 폭이 고르게).
"""
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config

REF_H = 187 / 200   # 크기 기준: Noto Sans CJK KR Bold의 「가」 높이 / 글자 크기 (이 비율에 맞춰 모든 글꼴 크기를 보정)


@dataclass(frozen=True)
class Face:
    """식자 글꼴. scale은 「가」 높이를 REF_H에 맞추는 px 배율이라, 같은 size면 어느 글꼴이든 글자 크기가 같다."""
    path: str
    index: int = 0
    scale: float = 1.0


def _glyph_h(path, index):
    f = ImageFont.truetype(path, 200, index=index)
    x0, y0, x1, y1 = f.getbbox("가", anchor="ls")
    return y1 - y0


def face(path, index=None) -> Face:
    index = index if index is not None else (1 if str(path).endswith(".ttc") else 0)
    return Face(str(path), index, round(REF_H * 200 / _glyph_h(str(path), index), 3))


FALLBACK = face(config.FONT_FALLBACK)   # 글꼴에 없는 기호(…, ♪ 등)를 대신 그릴 글꼴
DEFAULT = FALLBACK
DEFAULT_FONTS = config.FONTS            # 역할별 기본 글꼴(대사 나눔스퀘어라운드 EB, 외침 도현, 손글씨 나눔펜)


def set_font(path, index=None):
    """기본(대사) 글꼴 교체."""
    global DEFAULT
    DEFAULT = face(path, index)


@dataclass
class Interior:
    mask: np.ndarray        # 페이지 크기 bool, 말풍선 내부(글자 획 포함)
    bg: tuple[int, int, int]
    ok: bool                # 흰 바탕으로 판정됐는지


def interior(rgb: np.ndarray, text_box, bubble_box, light_thr=None) -> Interior:
    H, W = rgb.shape[:2]
    bx0, by0, bx1, by1 = bubble_box or _pad(text_box, 0.15, W, H)
    tx0, ty0, tx1, ty1 = text_box
    crop = rgb[by0:by1, bx0:bx1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    # 글 박스 바깥 테두리 띠의 밝기로 바탕을 추정
    ring = np.zeros(gray.shape, bool)
    rx0, ry0, rx1, ry1 = tx0 - bx0, ty0 - by0, tx1 - bx0, ty1 - by0
    p = max(4, int(0.04 * max(rx1 - rx0, ry1 - ry0)))
    ring[max(0, ry0 - p):ry1 + p, max(0, rx0 - p):rx1 + p] = True
    ring[max(0, ry0):ry1, max(0, rx0):rx1] = False
    bgv = np.median(gray[ring]) if ring.any() else 255
    thr = light_thr or max(140, bgv - 45)
    # 흰 말풍선 = 세 채널 모두 밝음. 회색 밝기만 보면 피부·하늘 같은 연한 색이 섞여 들어온다.
    # 연한 피부(255,228,217)는 채도(max-min)로 거른다.
    c16 = crop.astype(np.int16)
    light = ((c16.min(axis=2) >= thr) & (c16.max(axis=2) - c16.min(axis=2) <= 28)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(light, 8)
    # 글 박스와 가장 많이 겹치는 밝은 성분
    # 글자 획이 말풍선 안 흰 영역을 여러 조각으로 가를 수 있어서, 글 박스와 의미 있게 겹치는 밝은 성분을 모두 합친다.
    # 단 bubble 박스 테두리에 닿는 성분은 말풍선 밖 배경이라 뺀다(bubble 박스가 없을 때만 해당).
    sub = lab[max(0, ry0):ry1, max(0, rx0):rx1]
    ids, cnt = np.unique(sub[sub > 0], return_counts=True)
    m = np.zeros(gray.shape, np.uint8)
    if len(ids):
        floor = max(30, 0.02 * cnt.max())
        for i, c in zip(ids, cnt):
            if c >= floor:
                m[lab == i] = 1
    # 구멍 메우기: 바깥에서 flood fill이 안 닿는 곳 = 내부
    filled = m.copy()
    pad = cv2.copyMakeBorder(1 - m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    ff = pad.copy()
    cv2.floodFill(ff, None, (0, 0), 2)
    holes = (ff[1:-1, 1:-1] == 1)
    filled[holes] = 1
    full = np.zeros((H, W), bool)
    full[by0:by1, bx0:bx1] = filled.astype(bool)
    bg_pix = crop[m.astype(bool)]
    bg = tuple(int(v) for v in np.median(bg_pix, 0)) if len(bg_pix) else (255, 255, 255)
    # 판정: 글 박스의 대부분이 내부여야 하고 바탕이 충분히 밝아야 함
    tb_gray = cv2.cvtColor(rgb[ty0:ty1, tx0:tx1], cv2.COLOR_RGB2GRAY)
    cover = full[ty0:ty1, tx0:tx1].mean() if tb_gray.size else 0
    ok = cover > 0.7 and np.median(tb_gray) > 190
    return Interior(full, bg, ok)


def _pad(box, r, W, H):
    x0, y0, x1, y1 = box
    dx, dy = int((x1 - x0) * r), int((y1 - y0) * r)
    return max(0, x0 - dx), max(0, y0 - dy), min(W, x1 + dx), min(H, y1 + dy)


def erase(rgb: np.ndarray, it: Interior) -> None:
    m = cv2.erode(it.mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    rgb[m] = it.bg


PUNCT = set("…!?！？♪~～.,、。・\u2026♡♥")


def _units(text, mode):
    """(조각, 앞에 공백 여부) 목록. word=어절 단위, char=음절 단위(문장부호는 앞 글자에 붙임)."""
    out = []
    for wi, word in enumerate(text.split()):
        if mode == "word":
            out.append((word, wi > 0))
            continue
        for ci, ch in enumerate(word):
            if ch in PUNCT and ci > 0:
                out[-1] = (out[-1][0] + ch, out[-1][1])
            else:
                out.append((ch, wi > 0 and ci == 0))
    return out


def _wrap(units, n_lines, widths, space_w):
    """units를 정확히 n_lines 줄로 나누되 가장 긴 줄이 가장 짧아지게(DP). 줄 폭이 고르면 말풍선에 잘 들어간다."""
    n = len(units)
    if n < n_lines:
        return None
    # pre[i]: units[0:i] 폭 합(공백 포함), 줄 첫 조각 앞 공백은 빼고 계산
    sp = [space_w if s and i else 0.0 for i, (_, s) in enumerate(units)]

    def width(a, b):  # units[a:b] 한 줄 폭
        return sum(widths[a:b]) + sum(sp[a + 1:b])

    INF = float("inf")
    best = [[INF] * (n + 1) for _ in range(n_lines + 1)]
    cut = [[0] * (n + 1) for _ in range(n_lines + 1)]
    best[0][0] = 0.0
    for k in range(1, n_lines + 1):
        for b in range(k, n - (n_lines - k) + 1):
            for a in range(k - 1, b):
                if best[k - 1][a] == INF:
                    continue
                v = max(best[k - 1][a], width(a, b))
                if v < best[k][b]:
                    best[k][b], cut[k][b] = v, a
    if best[n_lines][n] == INF:
        return None
    bounds, b = [], n
    for k in range(n_lines, 0, -1):
        a = cut[k][b]
        bounds.append((a, b))
        b = a
    bounds.reverse()
    return ["".join((" " if s and i else "") + u for i, (u, s) in enumerate(units[a:b])) for a, b in bounds]


def _center(mask_crop):
    dt = cv2.distanceTransform(mask_crop.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(dt >= 0.6 * dt.max())
    return xs.mean(), ys.mean(), dt.max()


def _place(cols, cx, w):
    """이 줄 높이에서 비어 있는 열 구간(cols) 중 cx에 가장 가까운, 폭 w가 들어가는 구간에 가운데 정렬한 x.
    휘거나 기운 말풍선에서 줄마다 중심이 따라가게 한다."""
    w = int(w) + 1
    d = np.diff(np.concatenate([[0], cols.astype(np.int8), [0]]))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    best = None
    for a, b in zip(starts, ends):
        if b - a < w:
            continue
        x = int(min(max(cx - w / 2, a), b - w))
        dist = abs(x + w / 2 - cx)
        if best is None or dist < best[0]:
            best = (dist, x)
    return None if best is None else best[1]


@dataclass
class Area:
    """식자 영역: 페이지 좌표 (x0, y0)에서 시작하는 crop 마스크들. safe=기본 여백, loose=좁은 여백."""
    x0: int
    y0: int
    safe: np.ndarray
    loose: np.ndarray
    cx: float
    cy: float


@dataclass
class Layout:
    text: str
    size: int
    mode: str               # word / char
    lines: list             # [(줄, x, y)] crop 좌표
    area: Area
    loose: bool = False
    face: Face = None


def make_area(it: Interior, text_box=None) -> Area | None:
    ys, xs = np.nonzero(it.mask)
    if not len(xs):
        return None
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    m = it.mask[y0:y1, x0:x1].astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    margin = max(3, int(0.06 * min(x1 - x0, y1 - y0)))
    safe = cv2.erode(m, k, iterations=margin).astype(bool)
    loose = cv2.erode(m, k, iterations=max(2, margin // 2)).astype(bool)
    if text_box is not None:
        # 원문 글자가 있던 박스는 말풍선 안이 보장되니 안전 영역에 포함하고, 중심도 원문 박스 중심으로.
        tx0, ty0, tx1, ty1 = text_box
        sl = (slice(max(0, ty0 - y0), max(0, ty1 - y0)), slice(max(0, tx0 - x0), max(0, tx1 - x0)))
        safe[sl] |= m[sl].astype(bool)
        loose[sl] |= m[sl].astype(bool)
        cx, cy = (tx0 + tx1) / 2 - x0, (ty0 + ty1) / 2 - y0
    else:
        if not safe.any():
            safe = m.astype(bool)
        cx, cy, _ = _center(safe)
    return Area(int(x0), int(y0), safe, loose, cx, cy)


_FONTS = {}
_CMAPS = {}


def font_at(size, fc: Face = None):
    """fc 글꼴의 '보정된' size px 폰트 객체."""
    fc = fc or DEFAULT
    px = max(4, round(size * fc.scale))
    key = (fc.path, fc.index, px)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(fc.path, px, index=fc.index)
    return _FONTS[key]


def _cmap(fc: Face):
    if fc.path not in _CMAPS:
        from fontTools.ttLib import TTFont
        t = TTFont(fc.path, fontNumber=fc.index, lazy=True)
        _CMAPS[fc.path] = set(t.getBestCmap() or {})
    return _CMAPS[fc.path]


def runs(text, fc: Face = None):
    """[(조각, 글꼴)] — 글꼴에 없는 글자(…, ♪, ♡ 등)는 대체 글꼴(config.FONT_FALLBACK)로 대체."""
    fc = fc or DEFAULT
    cm = _cmap(fc)
    out = []
    for ch in text:
        f = fc if (ord(ch) in cm or ch == " ") else FALLBACK
        if out and out[-1][1] == f:
            out[-1] = (out[-1][0] + ch, f)
        else:
            out.append((ch, f))
    return out


_METRICS = {}


def metrics(fc: Face = None):
    """size 대비 (윗선, 아랫선, 칸 폭). 받침 있는 음절까지 포함한 실제 잉크 범위라, 줄 간격은 이걸로 잡아야 안 겹친다.
    (size는 「가」 높이 기준이라 받침이 아래로 긴 글꼴—나눔스퀘어라운드 등—은 글자가 size보다 크다)"""
    fc = fc or DEFAULT
    if fc not in _METRICS:
        size = 100
        cm = _cmap(fc)
        sample = "".join(c for c in "가갉닭힣쀍뷁릝앍뭘ㅎ" if ord(c) in cm) or "가"
        f = font_at(size, fc)
        tops, bots = zip(*[(f.getbbox(c, anchor="ls")[1], f.getbbox(c, anchor="ls")[3]) for c in sample])
        width = max(f.getlength(c) for c in sample)
        _METRICS[fc] = (-min(tops) / size, max(0, max(bots)) / size, max(1.0, width / size))
    return _METRICS[fc]


GAP = 0.18        # 줄 사이 여백(size 대비)
GAP_TIGHT = 0.11  # 좁은 줄간격 단계


def text_len(text, size, fc: Face = None):
    return sum(font_at(size, f).getlength(seg) for seg, f in runs(text, fc))


def draw_text(draw, x, top, text, size, fc: Face = None, **kw):
    """글자 윗선(metrics 윗선)이 top에 오도록, 대체 글꼴 조각까지 같은 기준선에 그린다."""
    fc = fc or DEFAULT
    base = top + metrics(fc)[0] * size
    for seg, f in runs(text, fc):
        font = font_at(size, f)
        draw.text((x, base), seg, font=font, anchor="ls", **kw)
        x += font.getlength(seg)


def fit(area: Area, text: str, sizes, mode="word", loose=False, line_h=1.12, fc: Face = None) -> Layout | None:
    """sizes(큰 것부터) 중 mode 줄바꿈으로 영역 안에 들어가는 가장 큰 배치."""
    fc = fc or DEFAULT
    asc, desc, _ = metrics(fc)
    gap = GAP_TIGHT if line_h < 1.12 else GAP    # line_h<1.12 = 좁은 줄간격 단계
    safe = area.loose if loose else area.safe
    cx, cy = area.cx, area.cy
    units0 = _units(text, mode)
    for size in sizes:
        gh = (asc + desc) * size          # 한 줄 잉크 높이
        lh = gh + gap * size
        widths = [text_len(u, size, fc) for u, _ in units0]
        space_w = text_len(" ", size, fc)
        for n in range(1, len(units0) + 1):
            block_h = n * lh - gap * size
            if block_h > safe.shape[0]:
                break
            lines = _wrap(units0, n, widths, space_w)
            if not lines:
                continue
            top = cy - block_h / 2
            top = min(max(top, 0), safe.shape[0] - block_h)
            boxes = []
            for i, ln in enumerate(lines):
                w = text_len(ln, size, fc)
                ly0, ly1 = int(top + i * lh), int(top + i * lh + gh) + 1
                if ly0 < 0 or ly1 > safe.shape[0]:
                    break
                lx0 = _place(safe[ly0:ly1].all(axis=0), cx, w)
                if lx0 is None:
                    break
                boxes.append((ln, lx0, ly0))
            else:
                lay = Layout(text, size, mode, boxes, area, loose, fc)
                if n == 1 or _ink_ok(lay):     # 실제 잉크로 줄끼리 맞닿지 않는지 확인
                    return lay
    return None


def _ink_ok(lay):
    from engine import ink_check
    return ink_check(lay)[0] == 0


# 세로쓰기용 대체 글자. 번들 글꼴에는 세로형 문장부호(︙ ︐ ︒ ︵ 등)가 없어서(두부로 찍힘) 있는 글자로 바꾸고,
# 쉼표·마침표·괄호는 한국어 세로 식자 관례대로 뺀다(""). 물결표는 그대로 둔다.
VERT_MAP = {"…": "⋮", "‥": "⋮", "ー": "｜", "—": "｜", "―": "｜", "～": "~",
            ",": "", ".": "", "、": "", "。": "", "(": "", ")": "", "（": "", "）": ""}


def _vert_cells(word):
    """세로쓰기 칸: 음절 하나씩, 연속된 !?는 한 칸(가로로 묶음), 말줄임표 등은 세로형 글리프로."""
    cells = []
    for ch in word.replace("...", "…"):     # 마침표 셋은 말줄임표(마침표는 빼므로)
        ch = VERT_MAP.get(ch, ch)
        if not ch:
            continue
        if ch in "!?！？" and cells and all(c in "!?！？" for c in cells[-1]):
            cells[-1] += ch
        else:
            cells.append(ch)
    return cells


def fit_vert(area: Area, text: str, sizes, loose=False, col_gap=1.18, cell_h=1.12, fc: Face = None,
             max_cols=None) -> Layout | None:
    """한국어 세로쓰기: 칸을 위→아래로 쌓고 열은 오른쪽→왼쪽. 열은 어절 경계에서 나눈다(DP로 열 길이 고르게).
    max_cols: 열 수 상한(None이면 제한 없음)."""
    fc = fc or DEFAULT
    safe = area.loose if loose else area.safe
    words = text.split()
    asc, desc, cw_ratio = metrics(fc)
    for size in sizes:
        ch = size * (asc + desc + cell_h - 1.0)
        cw = int(size * cw_ratio)
        units = [(w, i > 0) for i, w in enumerate(words)]
        heights = [len(_vert_cells(w)) * ch for w in words]
        for ncol in range(1, min(len(words), max_cols or len(words)) + 1):
            cols = _wrap(units, ncol, heights, ch * 0.5)
            if not cols:
                continue
            col_cells = [[c for w in col.split() for c in (_vert_cells(w) + [" "])][:-1] for col in cols]
            total_w = cw + (ncol - 1) * cw * col_gap
            if total_w > safe.shape[1]:
                break
            right = min(max(area.cx + total_w / 2, total_w), safe.shape[1])
            boxes, ok = [], True
            for k, cells in enumerate(col_cells):
                h = sum(ch * (0.5 if c == " " else 1) for c in cells)
                x = int(right - cw - k * cw * col_gap)
                top = min(max(area.cy - h / 2, 0), safe.shape[0] - h)
                if top < 0 or x < 0 or not safe[int(top):int(top + h) + 1, x:x + cw + 1].all():
                    ok = False
                    break
                y = top
                for c in cells:
                    if c != " ":
                        w = text_len(c, size, fc)
                        boxes.append((c, int(x + (cw - w) / 2), int(y)))
                    y += ch * (0.5 if c == " " else 1)
            if ok:
                lay = Layout(text, size, "vert", boxes, area, loose, fc)
                if _ink_ok(lay):
                    return lay
    return None


def capacity(area: Area, size, line_h=1.12):
    """그 크기에서 영역에 대략 몇 음절이 들어가는지(행별 빈 폭 합 / 글자 폭)."""
    safe = area.safe
    asc, desc, _ = metrics()
    lh = max(1, int(size * (asc + desc + GAP)))
    total = 0
    for y in range(0, safe.shape[0] - size, lh):
        cols = safe[y:y + size].all(axis=0)
        total += int(cols.sum() // size)
    return total


def draw_layout(img: Image.Image, lay: Layout, color=(0, 0, 0), stroke=0, stroke_color=(255, 255, 255)):
    draw = ImageDraw.Draw(img)
    sw = max(1, lay.size * stroke // 100) if stroke else 0
    for ln, lx, ly in lay.lines:
        draw_text(draw, lay.area.x0 + lx, lay.area.y0 + ly, ln, lay.size, lay.face,
                  fill=color, stroke_width=sw, stroke_fill=stroke_color)
