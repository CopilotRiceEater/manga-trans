"""식자 엔진: 블록마다 자동 폴백 체인으로 배치를 고른다. 사람 개입 없이 항상 결과를 낸다.

크기 체계: 작품 기본 크기 base(원문 글자 크기 중앙값에서 추정), 하한 floor = 0.8·base.
원문이 눈에 띄게 큰 블록(외침·강조)만 target을 키운다(최대 3·base).

체인 (앞에서 성공하면 끝):
  1 어절 줄바꿈, target→floor, 기본 여백 (VERTICAL이면 세로로 긴 영역은 세로쓰기와 비교 — 기본 꺼짐)
  2 어절 줄바꿈, target→floor, 좁은 여백·줄간격 1.0
  3 한 어절짜리면 음절 단위, target→floor
  4 말풍선 키우기(expand.py) 후 1~3 재시도 — 번역문을 살리는 쪽이 먼저. 같은 칸·다른 말풍선과 안 겹칠 때만
  5 gemma 단축 후보(긴 것부터, 말투 바뀐 후보 제외)로 1~3 재시도  ← 실패 블록만 페이지당 1회 호출,
    그래도 실패하면 예산을 더 줄여 2차 요청
  6 최종: 가장 짧은 문장으로 음절 단위·좁은 여백, 하한 무시하고 8px까지. 그래도 안 되면 8px로 강제 배치
"""
import math
from dataclasses import dataclass, field

import re

from letter import Area, Face, Layout, capacity, fit, fit_vert, text_len

MIN_PX = 8
FLOOR_RATIO = 0.8
EMPH_RATIO = 1.35       # 원문 글자 크기가 중앙값의 이 배수 이상이면 강조
GLYPH_FROM_BOX = 0.75   # sqrt(박스 면적/글자 수)는 자간·행간 포함이라 실제 글자 크기로 줄임


def glyph_est(block):
    x0, y0, x1, y1 = block.box
    n = len(block.text.replace(" ", "")) or 1
    return math.sqrt(max(1, (x1 - x0) * (y1 - y0)) / n)


def book_base(blocks):
    """작품 기본 글자 크기(px). blocks: 작품 전체의 대사 블록."""
    gs = sorted(glyph_est(b) for b in blocks if b.text)
    return max(12, round(gs[len(gs) // 2] * GLYPH_FROM_BOX)) if gs else 24


@dataclass
class Spec:
    target: int
    floor: int
    face: Face = None
    role: str = "dialogue"


def block_spec(block, base, median_g, mode="bubble", faces=None):
    """크기와 역할(dialogue / shout / hand)을 정한다. faces: {역할: Face}, 없는 역할은 dialogue 글꼴."""
    g = glyph_est(block)
    target, role = base, "dialogue"
    if g >= EMPH_RATIO * median_g:
        target, role = min(3 * base, round(g * GLYPH_FROM_BOX)), "shout"
    if mode == "sanitize" and block.extra.get("kind") == "dialogue":
        role = "hand"        # 그림 위 손글씨 대사
    faces = faces or {}
    return Spec(target, max(MIN_PX, round(base * FLOOR_RATIO)), faces.get(role) or faces.get("dialogue"), role)


@dataclass
class Placed:
    layout: Layout
    step: int
    text: str
    notes: dict = field(default_factory=dict)


def _sizes(hi, lo):
    return range(hi, lo - 1, -1)


VERTICAL = False    # 한국어 세로쓰기 사용 여부 — 읽기 어색해서 기본 끔
VERT_GAIN = 1.25    # (켰을 때) 세로쓰기가 이만큼 더 크게 들어가야 가로 대신 쓴다
VERT_MAX_SYL = 24


def _tall(area: Area):
    return area.safe.shape[0] >= 2.0 * area.safe.shape[1]


def try_chain(area: Area, text: str, spec: Spec):
    """1~3단계. 성공하면 (Layout, 단계), 아니면 None.
    세로로 긴 영역(원래 세로쓰기 자리)은 1단계에서 세로쓰기도 재서 충분히 더 크면 세로쓰기를 쓴다."""
    sz = _sizes(spec.target, spec.floor)
    f = spec.face
    lay = fit(area, text, sz, "word", fc=f)
    if VERTICAL and _tall(area) and len(text.replace(" ", "")) <= VERT_MAX_SYL:
        floor_v = max(spec.floor, int(lay.size * VERT_GAIN)) if lay else spec.floor
        vert = fit_vert(area, text, _sizes(spec.target, floor_v), fc=f)
        if vert and (not lay or vert.size >= lay.size * VERT_GAIN):
            return vert, 1
    if lay:
        return lay, 1
    lay = fit(area, text, sz, "word", loose=True, line_h=1.0, fc=f)
    if lay:
        return lay, 2
    if len(text.split()) == 1:
        lay = fit(area, text, sz, "char", fc=f) or fit(area, text, sz, "char", loose=True, line_h=1.0, fc=f)
        if lay:
            return lay, 3
    return None


def final_fit(area: Area, text: str, spec: Spec) -> Layout:
    f, sz = spec.face, _sizes(spec.target, MIN_PX)
    cands = [fit(area, text, sz, "word", loose=True, line_h=1.0, fc=f),
             fit_vert(area, text, sz, loose=True, fc=f) if VERTICAL and _tall(area) else None,
             fit(area, text, sz, "char", loose=True, line_h=1.0, fc=f)]
    lay = max((c for c in cands if c), key=lambda c: c.size, default=None)
    if lay:
        return lay
    # 영역이 비정상적으로 작다: 원문 박스 중심에 최소 크기로 한 줄씩 강제 배치
    words = text.split() or [text]
    lines = [(w, int(area.cx - text_len(w, MIN_PX, f) / 2), int(area.cy - len(words) * MIN_PX / 2 + i * MIN_PX))
             for i, w in enumerate(words)]
    return Layout(text, MIN_PX, "forced", lines, area, True, f)


def shrink_budget(area: Area, spec: Spec, ratio=0.6):
    """단축 요청 시 알려줄 대략적인 글자 수 상한. capacity는 줄바꿈 손실을 무시해 과대평가라 보수적으로 깎는다."""
    return max(2, int(capacity(area, spec.floor) * ratio))


_POLITE = re.compile(r"(요|니다|니까|세요|십시오|죠|시죠)$")


def polite(s: str):
    """문장 끝 어미가 존댓말인지(기호 떼고 마지막 어절 기준)."""
    tail = re.sub(r"[\s…!?.,~♪♡—\-'\"“”’()]+$", "", s)
    return bool(_POLITE.search(tail))


def same_speech(orig: str, alt: str):
    return polite(orig) == polite(alt)


def place(area: Area, text: str, spec: Spec, alts=()):
    """체인 전체. alts는 gemma 단축 후보(없으면 4단계 건너뜀). 항상 Placed를 반환."""
    r = try_chain(area, text, spec)
    if r:
        return Placed(r[0], r[1], text)
    alts = [a for a in alts if a and a != text and same_speech(text, a)]   # 말투 바뀐 후보 탈락
    for alt in sorted(set(alts), key=len, reverse=True):
        r = try_chain(area, alt, spec)
        if r:
            return Placed(r[0], 5, alt, {"orig": text, "sub_step": r[1]})
    shortest = min([text, *[a for a in alts if a]], key=len)
    return Placed(final_fit(area, shortest, spec), 6, shortest, {"orig": text})


def score(placed: Placed, base: int):
    """블록 품질 점수(높을수록 좋음, 0~100). 크기 부족·음절 끊기·단축·강제 배치에 감점."""
    lay = placed.layout
    s = 100.0
    s -= max(0.0, (base * 0.95 - lay.size) / base) * 120
    if lay.mode == "char" and len(placed.text.split()) > 1:
        s -= 25
    if lay.mode == "forced":
        s -= 60
    s -= {1: 0, 2: 3, 3: 5, 4: 4, 5: 8, 6: 30}[placed.step]
    return round(max(0.0, s), 1)


def ink_check(lay: Layout):
    """실제로 그려 보고 (줄끼리 겹치거나 맞닿은 픽셀 수, 식자 영역 밖으로 나간 픽셀 수). 둘 다 0이어야 정상.
    맞닿음도 잡도록 각 줄 잉크를 글자 크기의 6%만큼 부풀려서 비교한다."""
    import cv2
    from PIL import Image, ImageDraw
    import numpy as np
    from letter import draw_text
    H, W = lay.area.loose.shape
    masks = []
    for ln, lx, ly in lay.lines:
        im = Image.new("L", (W, H), 0)
        draw_text(ImageDraw.Draw(im), lx, ly, ln, lay.size, lay.face, fill=255)
        masks.append(np.asarray(im) > 96)
    r = max(1, round(lay.size * 0.03))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    fat = [cv2.dilate(m.astype(np.uint8), k).astype(bool) for m in masks]
    overlap = 0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            overlap += int((fat[i] & fat[j]).sum())
    ink = np.any(masks, axis=0) if masks else np.zeros((H, W), bool)
    outside = int((ink & ~lay.area.loose).sum())
    return overlap, outside
