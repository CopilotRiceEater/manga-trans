"""식질: ogkalu comic-text-and-bubble-detector(RT-DETR-v2) + 만화 VL 리더(anime_tools, PaddleOCR-VL-1.6 파인튜닝).

검출기 클래스가 bubble / text_bubble(말풍선 안 글) / text_free(말풍선 밖 글)라 안/밖 분류가 같이 나온다.
(anime_tools 기본 검출기 AnimeText는 HF gated라 토큰 없이는 못 받음.)
박스는 원본 페이지 픽셀 [x0, y0, x1, y1].
"""
from dataclasses import dataclass, field

import numpy as np
import torch

import config  # noqa: F401 — anime_tools 모델 위치(ANIME_TOOLS_HOME)를 먼저 정한다
from PIL import Image

DET_REPO = "ogkalu/comic-text-and-bubble-detector"
LABELS = {0: "bubble", 1: "text_bubble", 2: "text_free"}


@dataclass
class Block:
    kind: str                      # text_bubble | text_free
    box: tuple[int, int, int, int]
    det: float
    text: str = ""
    score: float = 0.0
    bubble: tuple[int, int, int, int] | None = None   # 담고 있는 bubble 박스
    id: int = 0
    extra: dict = field(default_factory=dict)


def _iou_in(inner, outer):
    """inner 박스 면적 중 outer 안에 든 비율."""
    x0, y0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x1, y1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    a = max(0, x1 - x0) * max(0, y1 - y0)
    return a / max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))


NEED_VRAM_GB = 3.0   # 검출기 + 리더(PaddleOCR-VL 0.9B) 여유


def pick_device(device="auto"):
    """auto: GPU 여유 메모리가 NEED_VRAM_GB 이상이면 cuda, 아니면 cpu(다른 프로그램이 VRAM을 잡고 있을 때)."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free / 2**30 >= NEED_VRAM_GB:
            return "cuda"
        print(f"[ocr] GPU 여유 {free / 2**30:.1f}GB < {NEED_VRAM_GB}GB → CPU로 실행")
    return "cpu"


class OCR:
    def __init__(self, device="auto", det_thresh=0.35):
        from anime_tools.ocr.sfx import SfxReader
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        device = pick_device(device)
        self.device, self.det_thresh = device, det_thresh
        self.proc = RTDetrImageProcessor.from_pretrained(DET_REPO)
        self.det = RTDetrV2ForObjectDetection.from_pretrained(DET_REPO).to(device).eval()
        self.reader = SfxReader.load(device=device)

    @torch.no_grad()
    def detect(self, page: Image.Image):
        page = page.convert("RGB")
        inp = self.proc(images=page, return_tensors="pt").to(self.device)
        res = self.proc.post_process_object_detection(
            self.det(**inp), target_sizes=[page.size[::-1]], threshold=self.det_thresh)[0]
        out = []
        for s, l, b in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
            x0, y0, x1, y1 = b
            out.append((LABELS[l], (max(0, int(x0)), max(0, int(y0)), min(page.width, int(x1 + .5)), min(page.height, int(y1 + .5))), s))
        return out

    def read(self, page: Image.Image) -> tuple[list[Block], list[tuple]]:
        """(글 블록 목록, bubble 박스 목록). 글 블록은 읽기 순서(오른쪽 위 → 왼쪽 아래, 만화 기준)는 번역 단계에서 정함."""
        dets = self.detect(page)
        bubbles = [b for k, b, _ in dets if k == "bubble"]
        blocks = [Block(k, b, s) for k, b, s in dets if k != "bubble"]
        blocks = _nms(blocks)
        for blk in blocks:
            cands = [(_iou_in(blk.box, bb), bb) for bb in bubbles]
            cands = [c for c in cands if c[0] > 0.6]
            if cands:
                # 가장 작은(가장 안쪽) bubble
                blk.bubble = min(cands, key=lambda c: (c[1][2] - c[1][0]) * (c[1][3] - c[1][1]))[1]
        bgr = np.asarray(page.convert("RGB"))[:, :, ::-1].copy()
        reads = self.reader.read_boxes_scored(bgr, [b.box for b in blocks])
        for i, (blk, r) in enumerate(zip(blocks, reads)):
            blk.id = i
            if r:
                blk.text, blk.score = r
        return blocks, bubbles


def _nms(blocks, thr=0.7):
    keep = []
    for b in sorted(blocks, key=lambda b: -b.det):
        if all(_iou_in(b.box, k.box) < thr and _iou_in(k.box, b.box) < thr for k in keep):
            keep.append(b)
    return keep


if __name__ == "__main__":
    import sys
    import time

    t0 = time.time()
    ocr = OCR()
    print(f"load {time.time() - t0:.1f}s")
    for p in sys.argv[1:]:
        t0 = time.time()
        blocks, bubbles = ocr.read(Image.open(p))
        print(f"== {p}  {len(blocks)} blocks, {len(bubbles)} bubbles  {time.time() - t0:.1f}s")
        for b in blocks:
            print(f"  [{b.id}] {b.kind:11s} {b.box} det={b.det:.2f} sc={b.score:.2f} bubble={b.bubble}  {b.text!r}")
