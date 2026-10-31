"""번역: LLM(기본 Ollama Cloud gemma4:31b, 또는 OpenAI 호환 API) — 설정은 config.py / .env.

- make_bible: 작품 전체 대사(읽는 순서) + 일부 페이지 그림 → 설정집 JSON(줄거리·인물·호칭/말투·용어). 1회 호출.
- translate: 페이지 이미지 + 블록 목록 (+ 설정집, 앞뒤 페이지 대사) → 블록별 {kind, ko}. 페이지당 1회 호출.
- shorten: 말풍선에 안 들어가는 대사의 짧은 후보.

JSON 강제(format)는 백엔드마다 지원이 달라서 프롬프트로 JSON을 요구하고 응답에서 직접 파싱한다.
"""
import base64
import io
import json
import re
import time

import httpx
from PIL import Image

import config

SYSTEM = """너는 일본 만화를 한국어로 옮기는 숙련된 식자·번역가야.
페이지 그림과 OCR로 읽은 글 블록 목록(읽는 순서)을 받는다. 그림을 보고 장면·화자·말투를 파악해서 블록마다 자연스러운 한국어로 옮겨라.
작품 설정집이 주어지면 인물 이름 표기·호칭·말투·용어를 반드시 그대로 따른다. 앞뒤 페이지 대사는 흐름 파악용 참고다.

규칙:
- 커뮤니티 만화 번역체. 직역 말고 캐릭터 말투와 상황에 맞게. 존댓말/반말은 원문을 따른다.
- 말풍선 안 글로 들어갈 길이로 간결하게. 원문보다 길게 늘이지 마라.
- ♪, …, !!, ? 같은 기호와 늘인 소리(ーー, あああ)는 살려라 (예: きゃーーー!! → 꺄아아악!!, みいちゃあああん!! → 미이쨩~~!!).
- OCR 오독이 보이면 그림을 보고 바로잡아라.
- kind:
  "dialogue" = 대사·독백·속마음·손글씨 혼잣말,
  "text" = 나레이션·설명문·제목·화수·광고 문구처럼 번역해서 넣을 글,
  "sfx" = 효과음·의성어(번역 안 함),
  "sign" = 그림 속 간판·상표·작가명, 그리고 표지의 작품명 로고처럼 그림의 일부로 두는 글(번역 안 함).
           페이지 안의 제목·헤드라인·코너 제목은 디자인된 글자여도 "text"로 번역한다,
  "symbol" = 기호만 있는 블록(♪, !, ♡ 등).
  sfx·sign·symbol은 ko에 원문을 그대로 둔다.

출력은 JSON 하나: {"items": [{"id": 블록id, "kind": "...", "ko": "..."}]} — 입력 블록 전부, 같은 id로."""

BIBLE_SYSTEM = """너는 일본 만화 번역팀의 감수자야. 한 화(또는 여러 화)의 전체 대사를 페이지·읽는 순서대로 받는다.
번역가들이 페이지를 나눠 번역할 때 일관성을 지키도록 작품 설정집을 만들어라. 모르면 추측이라고 적어라.
- 대사는 OCR 결과라 비슷한 한자 오독이 섞여 있다(예: 藁→薬). 문맥상 맞는 말로 고쳐 생각하라.
- 제목 로고·화수·작가명·간판·편집부 문구도 섞여 있다. 작가명이나 간판을 등장인물로 착각하지 마라.
- 일부 페이지 그림을 함께 준다. 인물 생김새·성별·누가 누구에게 말하는지 판단에 써라.
- 용어 번역어(terms.ko)는 한국 독자가 바로 알아듣는 짧은 한국어로 정하라. 말풍선에 들어가야 하니 길이가 중요하다.
  일본어 발음 그대로의 긴 음차(예: 우시노토키마이리)는 피하고 뜻을 살린 짧은 말(예: 축시 참배)을 써라. 인물 이름은 음차.

JSON 하나로 출력:
{
 "synopsis": "이번 화 줄거리 3~5문장 (한국어)",
 "characters": [{"ja": "원문 이름/호칭", "ko": "한국어 표기", "gender": "여/남/불명", "role": "누구인지",
                 "speech": "말투 (예: 반말, 털털함, 선배에게 존댓말)"}],
 "address": [{"from": "화자 ko", "to": "상대 ko", "how": "부르는 법·존댓말 여부"}],
 "terms": [{"ja": "용어", "ko": "번역어", "note": "설명"}],
 "tone": "작품 분위기와 번역 톤 한 줄"
}"""


def _img_b64(page: Image.Image, long_side=1280):
    page = page.convert("RGB")
    s = long_side / max(page.size)
    if s < 1:
        page = page.resize((round(page.width * s), round(page.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    page.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _to_openai(msgs):
    """Ollama 형식 메시지({"images": [b64]})를 OpenAI 형식(content 배열 + image_url)으로."""
    out = []
    for m in msgs:
        if not m.get("images"):
            out.append({"role": m["role"], "content": m["content"]})
            continue
        parts = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in m["images"]]
        out.append({"role": m["role"], "content": parts + [{"type": "text", "text": m["content"]}]})
    return out


def _chat(msgs, retries=2, timeout=300, need="items", check=None):
    """메시지를 보내고 응답에서 JSON 객체를 파싱해 (obj, 걸린 시간). need 키가 없는 응답은 재시도한다.
    check(obj): 형식이 틀리면 ValueError(재시도, 끝내 틀리면 실패), 빠진 게 있으면 False(재시도, 끝내 그대로 받음)."""
    local = any(h in config.LLM_URL for h in ("localhost", "127.0.0.1", "0.0.0.0"))
    if not config.LLM_KEY and not local:
        raise RuntimeError(f"번역 API 키가 없어요 ({config.LLM_URL}) — .env의 MT_LLM_KEY를 채우세요")
    headers = {"Authorization": f"Bearer {config.LLM_KEY}"} if config.LLM_KEY else {}
    if config.LLM_API == "openai":
        url = config.LLM_URL + "/chat/completions"
        body = {"model": config.LLM_MODEL, "messages": _to_openai(msgs), "temperature": 0.3}
    else:
        url = config.LLM_URL + "/api/chat"
        body = {"model": config.LLM_MODEL, "messages": msgs, "stream": False,
                "options": {"temperature": 0.3, "num_ctx": config.LLM_CTX}}
    t0 = time.time()
    for attempt in range(retries + 1):
        try:
            r = httpx.post(url, headers=headers, timeout=timeout, json=body)
            r.raise_for_status()
            d = r.json()
            content = d["choices"][0]["message"]["content"] if config.LLM_API == "openai" else d["message"]["content"]
            obj = json.loads(re.search(r"\{.*\}", content, re.S).group(0))
            if need and not isinstance(obj.get(need), list):
                raise ValueError(f"응답에 '{need}' 목록이 없음")
            if check and not check(obj) and attempt < retries:
                raise ValueError("응답에 빠진 블록이 있음")
            return obj, time.time() - t0
        except (httpx.HTTPError, KeyError, ValueError, AttributeError):
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)


def script_text(pages):
    """[(페이지 이름, [블록 dict])] → 'p001 [0] 原文' 형태 대본."""
    lines = []
    for name, blocks in pages:
        lines.append(f"### {name}")
        lines += [f"[{b['id']}] {b['text']}" for b in blocks if b.get("text")]
    return "\n".join(lines)


def make_bible(pages, images=()):
    """pages: [(이름, 블록 목록)], images: [(이름, PIL 이미지)] 일부 페이지(인물 파악용)."""
    note = ("첨부 그림: " + ", ".join(n for n, _ in images) + "\n\n") if images else ""
    obj, dt = _chat(need="characters", msgs=[{"role": "system", "content": BIBLE_SYSTEM},
                     {"role": "user", "images": [_img_b64(im, 896) for _, im in images],
                      "content": note + "전체 대사:\n" + script_text(pages) + "\n\nJSON만 출력."}])
    return obj, dt


def translate(page: Image.Image, blocks, bible=None, prev_text="", next_text=""):
    """blocks의 .text를 번역해서 각 블록 extra['ko'], extra['kind']를 채운다. 걸린 시간 반환."""
    listing = [{"id": b.id, "where": "말풍선 안" if b.kind == "text_bubble" else "말풍선 밖",
                "box": list(b.box), "ja": b.text} for b in blocks]
    ctx = ""
    if bible:
        ctx += "작품 설정집:\n" + json.dumps(bible, ensure_ascii=False) + "\n\n"
    if prev_text:
        ctx += "앞 페이지 대사(참고):\n" + prev_text + "\n\n"
    if next_text:
        ctx += "뒤 페이지 대사(참고):\n" + next_text + "\n\n"
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "images": [_img_b64(page)],
         "content": ctx + f"이 페이지 크기 {page.width}x{page.height}. 블록 목록(읽는 순서):\n"
                    + json.dumps(listing, ensure_ascii=False, indent=1) + "\n\nJSON만 출력."},
    ]
    want = {b.id for b in blocks if b.text}
    obj, dt = _chat(msgs, check=lambda o: not _missing(want, _items(o, "ko")))
    items = _items(obj, "ko")
    missing = _missing(want, items)
    for b in blocks:   # 끝내 빠진 블록은 ko가 비어 원문 그대로 두고(plan_page의 keep), 다음 번역 때 다시 요청한다
        it = items.get(b.id, {})
        b.extra["kind"] = it["kind"] if it.get("kind") in KINDS else "dialogue"
        b.extra["ko"] = it.get("ko", "")
        b.extra["missing"] = b.id in missing
    return dt


KINDS = ("dialogue", "text", "sfx", "sign", "symbol")


def _missing(want, items):
    """번역이 와야 하는데 없거나 빈 블록 id."""
    return {i for i in want if i not in items or (items[i].get("kind") not in KINDS[2:] and not items[i]["ko"].strip())}


def _items(obj, field):
    """{"items": [...]} → {id: item}. id가 정수가 아니거나 field 형식이 틀린 항목은 버린다."""
    out = {}
    for it in obj.get("items") or []:
        if not isinstance(it, dict):
            continue
        i = it.get("id")
        if isinstance(i, str) and i.strip().isdigit():
            i = int(i)
        if not isinstance(i, int) or isinstance(i, bool) or i in out:
            continue
        v = it.get(field)
        if (field == "ko" and isinstance(v, str)) or (field == "cands" and isinstance(v, list)):
            out[i] = it
    return out


SHORTEN_SYSTEM = """너는 만화 식자가야. 아래 한국어 대사들이 말풍선에 안 들어간다.
뜻과 말투(존댓말/반말, 감정)는 그대로 두고 더 짧게 다시 써라. 문장 끝 어미의 높임(반말↔존댓말)은 절대 바꾸지 마라.
- 각 대사마다 후보 3개. 첫 후보는 살짝 줄인 것, 마지막 후보는 가장 많이 압축한 것.
- max_chars는 공백을 뺀 글자 수 목표치다. 마지막 후보는 반드시 그 안으로.
- 긴 단어는 짧은 동의어로 바꿔도 좋다(예: 상당히→꽤, 고용해→써). 맞춤법·띄어쓰기는 지켜라.
- …, !!, ? , ♪ 같은 기호는 유지.
출력은 JSON 하나: {"items": [{"id": id, "cands": ["...", "...", "..."]}]}"""


def shorten(items, bible=None):
    """items: [{id, ja, ko, max_chars}] → {id: [후보...]}, 걸린 시간."""
    ctx = ("작품 설정집(이름·말투 참고):\n" + json.dumps(bible, ensure_ascii=False) + "\n\n") if bible else ""
    obj, dt = _chat([{"role": "system", "content": SHORTEN_SYSTEM},
                     {"role": "user", "content": ctx + json.dumps(items, ensure_ascii=False, indent=1)
                      + "\n\nJSON만 출력."}])
    return {i: [c for c in it["cands"] if isinstance(c, str)] for i, it in _items(obj, "cands").items()}, dt
