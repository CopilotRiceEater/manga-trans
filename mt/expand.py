"""말풍선 키우기: 번역문이 원래 말풍선에 안 들어가면 말풍선을 (주로 가로로) 부풀려 다시 그린다.

세로로 긴 일본어 말풍선에 가로쓰기 한국어를 넣을 때 커뮤니티 식자가 하는 방식.
- 모양: 원래 내부 마스크(꼬리 포함)를 타원 커널로 가로 rx, 세로 ry만큼 팽창 → 원래 모양을 닮은 채로 커진다.
- 테두리: 원래 테두리 굵기·색을 재서 새 내부 바깥에 같은 굵기로 다시 그린다(안쪽의 옛 테두리는 흰색으로 덮임).
- 한계: 같은 칸 안에서만, 다른 말풍선·글 블록과 겹치지 않을 때만, 폭 증가 최대 MAX_GROW. 그 안에서 가장 작게 키운 것을 쓴다.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from letter import Interior

GROW_STEPS = (0.10, 0.20, 0.30, 0.40)   # 말풍선 폭 대비 늘릴 비율(양옆 합)
V_RATIO = 0.25                         # 세로는 가로 증가량의 이 비율만
CLEAR = 4                              # 다른 말풍선·글과 띄울 최소 간격(px)


@dataclass
class Expansion:
    interior: np.ndarray    # 새 내부(페이지 크기 bool)
    width: int              # 테두리 굵기
    color: tuple
    bg: tuple
    grow: float
    tails: list = None      # [(꼬리 마스크, 꼬리 안쪽(흰) 마스크, dx, dy)] — 키운 만큼 바깥으로 옮겨 다시 붙인다


def find_tails(rgb, interior, width):
    """말풍선 꼬리: 테두리까지 포함한 말풍선 모양에서 '몸통'(굵은 열림 연산으로 남는 부분)을 뺀 가는 돌출부."""
    filled = cv2.dilate(interior.astype(np.uint8), np.ones((2 * width + 3,) * 2, np.uint8))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # 꼬리는 내부 마스크에 다 안 들어오는 경우가 많아 테두리 근처의 어두운 선까지 포함
    near = cv2.dilate(filled, np.ones((17, 17), np.uint8)).astype(bool)
    shape = filled.astype(bool) | (near & (gray < 110) & _connected_to(filled, gray < 110))
    ys, xs = np.nonzero(interior)
    k = max(5, int(0.3 * min(xs.max() - xs.min(), ys.max() - ys.min())))
    body = cv2.morphologyEx(shape.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    tail = shape & ~cv2.dilate(body, np.ones((5, 5), np.uint8)).astype(bool)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(tail.astype(np.uint8), 8)
    cy, cx = ys.mean(), xs.mean()
    out = []
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] < 25:
            continue
        m = lab == j
        # 진짜 꼬리는 안쪽이 흰 쐐기(말풍선 내부와 이어짐). 말풍선에 닿은 집중선·배경 선은 어두운 선뿐이라 제외
        if (m & interior).sum() < 12:
            continue
        tx, ty = cent[j]
        out.append((m, (tx - cx, ty - cy)))
    return out


def _connected_to(seed, dark):
    """seed(말풍선)에 닿아 있는 어두운 선 성분만."""
    n, lab = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
    touch = np.unique(lab[cv2.dilate(seed, np.ones((3, 3), np.uint8)).astype(bool) & dark])
    return np.isin(lab, touch[touch > 0])


def outline_of(rgb, interior):
    """원래 테두리 굵기(px)와 색: 내부 바깥으로 한 겹씩 넓혀 가며 어두운 픽셀이 대부분인 겹 수를 센다."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k = np.ones((3, 3), np.uint8)
    prev = interior.astype(np.uint8)
    width, pix = 0, []
    for _ in range(12):
        cur = cv2.dilate(prev, k)
        ring = (cur > 0) & (prev == 0)
        if not ring.any():
            break
        dark = gray[ring] < 110
        if dark.mean() < 0.5:
            break
        width += 1
        pix.append(rgb[ring][dark])
        prev = cur
    color = tuple(int(v) for v in np.median(np.concatenate(pix), 0)) if pix else (0, 0, 0)
    return max(1, width), color


def allowed_region(shape, panel, others):
    """키운 말풍선이 들어가도 되는 곳: 칸 안(테두리 안쪽으로 약간), 다른 말풍선·글 블록 주변 제외."""
    H, W = shape
    ok = np.zeros(shape, bool)
    x0, y0, x1, y1 = panel if panel else (0, 0, W, H)
    ok[y0 + 3:y1 - 3, x0 + 3:x1 - 3] = True
    for m in others:
        ok &= ~cv2.dilate(m.astype(np.uint8), np.ones((2 * CLEAR + 1, 2 * CLEAR + 1), np.uint8)).astype(bool)
    return ok


def candidates(rgb, it: Interior, allowed):
    """작은 것부터 키운 말풍선 후보(Expansion)들. 허용 영역을 벗어나는 후보는 건너뛴다."""
    ys, xs = np.nonzero(it.mask)
    if not len(xs):
        return
    bw = xs.max() - xs.min() + 1
    width, color = outline_of(rgb, it.mask)
    base = it.mask.astype(np.uint8)
    tails = find_tails(rgb, it.mask, width)
    for g in GROW_STEPS:
        rx = max(2, int(bw * g / 2))
        ry = max(1, int(rx * V_RATIO))
        new = cv2.dilate(base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1))).astype(bool)
        outer = cv2.dilate(new.astype(np.uint8), np.ones((2 * width + 1, 2 * width + 1), np.uint8)).astype(bool)
        grown = outer & ~cv2.dilate(base, np.ones((2 * width + 3, 2 * width + 3), np.uint8)).astype(bool)
        moved = []
        for m, (vx, vy) in tails:
            norm = max(1e-6, (vx * vx + vy * vy) ** 0.5)
            dx, dy = int(round(rx * vx / norm)), int(round(ry * vy / norm))
            mm = _shift(m, dx, dy)
            grown |= mm
            moved.append((m, dx, dy))
        if (grown & ~allowed).any():      # 새로 덮는 부분(옮긴 꼬리 포함)이 칸 밖이나 다른 말풍선·글에 닿으면 탈락
            break
        yield Expansion(new, width, color, it.bg, g, moved)


def _shift(m, dx, dy):
    out = np.zeros_like(m)
    H, W = m.shape
    ys, xs = np.nonzero(m)
    ys2, xs2 = ys + dy, xs + dx
    ok = (ys2 >= 0) & (ys2 < H) & (xs2 >= 0) & (xs2 < W)
    out[ys2[ok], xs2[ok]] = True
    return out


def draw(rgb, ex: Expansion):
    """새 테두리(검정 고리) → 새 내부(바탕색) 순으로 칠하고, 꼬리를 키운 만큼 옮겨 다시 붙인다.
    경계는 살짝 흐려 계단 현상을 줄인다."""
    snap = rgb.copy()
    inner = ex.interior.astype(np.float32)
    outer = cv2.dilate(ex.interior.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ex.width + 1, 2 * ex.width + 1))).astype(np.float32)
    inner = cv2.GaussianBlur(inner, (0, 0), 0.7)[..., None]
    outer = cv2.GaussianBlur(outer, (0, 0), 0.7)[..., None]
    col, bg = np.array(ex.color, np.float32), np.array(ex.bg, np.float32)
    out = rgb.astype(np.float32)
    out = out * (1 - outer) + col * outer
    out = out * (1 - inner) + bg * inner
    rgb[:] = out.clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(snap, cv2.COLOR_RGB2GRAY)
    ring = outer[..., 0] > 0.5
    for m, dx, dy in ex.tails or []:
        dst = _shift(m, dx, dy)
        light = _shift(m & (gray >= 110), dx, dy)          # 꼬리 안쪽(흰 부분)
        ys, xs = np.nonzero(m)
        ys2, xs2 = ys + dy, xs + dx
        ok = (ys2 >= 0) & (ys2 < rgb.shape[0]) & (xs2 >= 0) & (xs2 < rgb.shape[1])
        ys, xs, ys2, xs2 = ys[ok], xs[ok], ys2[ok], xs2[ok]
        put = ~ex.interior[ys2, xs2]                       # 새 말풍선 안쪽엔 옛 꼬리 선을 찍지 않는다
        rgb[ys2[put], xs2[put]] = snap[ys[put], xs[put]]
        # 꼬리가 붙는 자리의 새 테두리는 열어 준다(꼬리 안쪽과 말풍선 안쪽이 이어지게)
        opening = cv2.dilate(light.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ring & ~ex.interior
        rgb[opening] = ex.bg
        del dst
