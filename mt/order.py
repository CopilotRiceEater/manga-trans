"""읽는 순서: 칸(패널)을 재귀 XY-cut으로 나누고, 칸은 위→아래·오른쪽→왼쪽, 칸 안의 블록은 오른쪽→왼쪽·위→아래.

칸 경계는 '잉크가 거의 없는 가로/세로 띠'(흰 여백)로 찾는다. 말풍선이 칸 경계를 넘어가면 그 자리는 못 자르지만,
블록은 중심점으로 칸에 배정하니 순서가 크게 틀어지진 않는다.
"""
import cv2
import numpy as np

MIN_GAP = 6        # 여백 띠 최소 두께(px)
INK_FRAC = 0.004   # 이 비율 이하로 잉크가 있으면 빈 줄로 본다


def _runs(empty):
    """빈 줄(True) 구간 [(a, b)]."""
    d = np.diff(np.concatenate([[0], empty.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _segments(ink, axis, x0, y0):
    """ink(영역)를 axis 방향 여백으로 자른 조각 [(x0, y0, x1, y1)] (자를 데가 없으면 1개)."""
    prof = ink.mean(axis=1 - axis) if axis == 0 else ink.mean(axis=0)
    n = len(prof)
    gaps = [(a, b) for a, b in _runs(prof <= INK_FRAC) if b - a >= MIN_GAP and a > 0 and b < n]
    cuts, start = [], 0
    for a, b in gaps:
        cuts.append((start, a))
        start = b
    cuts.append((start, n))
    cuts = [(a, b) for a, b in cuts if b - a > 2 * MIN_GAP]
    h, w = ink.shape
    if axis == 0:
        return [(x0, y0 + a, x0 + w, y0 + b) for a, b in cuts]
    return [(x0 + a, y0, x0 + b, y0 + h) for a, b in cuts]


def panels(rgb, depth=0):
    """페이지를 칸들로 나눠 읽는 순서대로 [(x0, y0, x1, y1)]."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = gray < 200
    return _cut(ink, 0, 0, ink.shape[1], ink.shape[0], 0)


def _cut(ink, x0, y0, x1, y1, depth):
    region = ink[y0:y1, x0:x1]
    if depth > 6 or region.size == 0:
        return [(x0, y0, x1, y1)]
    rows = _segments(region, 0, x0, y0)          # 가로 여백 → 위에서 아래로
    if len(rows) > 1:
        return [p for r in rows for p in _cut(ink, *r, depth + 1)]
    cols = _segments(region, 1, x0, y0)          # 세로 여백 → 오른쪽에서 왼쪽으로
    if len(cols) > 1:
        return [p for c in reversed(cols) for p in _cut(ink, *c, depth + 1)]
    return [(x0, y0, x1, y1)]


def sort_blocks(rgb, blocks):
    """blocks(.box)를 읽는 순서로 정렬해 반환하고, 각 블록 extra['panel']에 칸 번호를 적는다."""
    ps = panels(rgb)

    def panel_of(b):
        cx, cy = (b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2
        for i, (x0, y0, x1, y1) in enumerate(ps):
            if x0 <= cx < x1 and y0 <= cy < y1:
                return i
        return len(ps)

    widths = sorted(max(1, b.box[2] - b.box[0]) for b in blocks) or [1]
    band = max(1, widths[len(widths) // 2] // 2)
    for b in blocks:
        b.extra["panel"] = panel_of(b)
    # 칸 안: 오른쪽 끝 기준 띠(오른쪽 먼저) → 위쪽 먼저
    return sorted(blocks, key=lambda b: (b.extra["panel"], -(b.box[2] // band), b.box[1])), ps


def merge_free(blocks):
    """말풍선 밖 글(text_free) 중 바짝 붙은 것끼리 한 블록으로 합친다(이미 읽는 순서로 정렬된 상태에서).
    손글씨 혼잣말이 세로 열마다 따로 잡히면 번역·식자가 조각나서 서로 겹치기 때문."""
    out = []
    for b in blocks:
        prev = next((p for p in reversed(out) if p.kind == "text_free"), None)
        if b.kind == "text_free" and prev is not None and _adjacent(prev.box, b.box):
            x0, y0, x1, y1 = prev.box
            bx0, by0, bx1, by1 = b.box
            prev.box = (min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1))
            prev.text = prev.text + b.text
            prev.det = max(prev.det, b.det)
            continue
        out.append(b)
    return out


def _adjacent(a, b, gap_ratio=0.6, overlap_ratio=0.4):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    aw, ah, bw, bh = ax1 - ax0, ay1 - ay0, bx1 - bx0, by1 - by0
    hgap = max(bx0 - ax1, ax0 - bx1, 0)
    vgap = max(by0 - ay1, ay0 - by1, 0)
    vover = min(ay1, by1) - max(ay0, by0)
    hover = min(ax1, bx1) - max(ax0, bx0)
    side = hgap <= gap_ratio * min(aw, bw) and vover >= overlap_ratio * min(ah, bh)
    stack = vgap <= gap_ratio * min(ah, bh) and hover >= overlap_ratio * min(aw, bw)
    return side or stack
