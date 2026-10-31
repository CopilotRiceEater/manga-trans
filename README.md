# manga-trans

일본 만화 페이지를 넣으면 **글자 찾기(OCR) → LLM 번역 → 한국어 식자**까지 자동으로 해 주는 도구입니다. 웹 UI로 작품 단위(수백 쪽)를 올리고, 진행 상황을 보고, 결과를 PNG·AVIF·WebP·JPG로 받을 수 있습니다.

- 말풍선 안 대사는 **기계적으로 지우고 폰트로 식자**합니다. 생성 모델로 글자를 그리지 않으니 한글이 틀리지 않습니다.
- 말풍선 밖(그림 위) 대사만 인페인팅으로 지우고 외곽선 글씨로 넣습니다. 효과음·간판·로고는 그대로 둡니다.
- 사람이 손볼 필요 없이 끝까지 자동으로 처리하도록 설계했습니다(아래 "식자 엔진").

## 필요한 것

| 항목 | 내용 |
|---|---|
| GPU | NVIDIA, VRAM 4GB 이상 권장 (OCR ~3GB + LaMa ~1GB). CPU만 있어도 돌지만 느립니다 |
| 번역 LLM | 이미지 입력이 되는 모델의 API. 기본은 [Ollama Cloud](https://ollama.com)의 `gemma4:31b` (API 키 필요). OpenAI 호환 API(OpenAI, Gemini, OpenRouter, 로컬 vLLM·LM Studio 등)도 됩니다 |
| 파이썬 | [uv](https://docs.astral.sh/uv/)만 있으면 됩니다 (Python 3.13과 의존성은 uv가 설치) |

처음 실행할 때 모델을 자동으로 받습니다: 글자 검출기, 만화 OCR(PaddleOCR-VL + 만화 LoRA, 약 2.4GB), LaMa(약 200MB).

## 설치·실행

```bash
git clone https://github.com/CopilotRiceEater/manga-trans.git
cd manga-trans
cp .env.example .env        # 열어서 MT_LLM_KEY(번역 API 키) 등 입력
uv sync
uv run python mt/serve.py   # → http://127.0.0.1:8196
```

Windows는 `copy .env.example .env` 후 `run.bat`, Linux/macOS는 `./run.sh`로도 실행됩니다. (Linux에서 개발·시험했고, Windows는 아직 직접 시험하지 못했습니다. Windows용 torch는 CUDA 13.2 빌드를 받으니 최신 NVIDIA 드라이버가 필요합니다.)

웹 페이지에 이미지 여러 장이나 ZIP/CBZ를 끌어다 놓으면 바로 번역을 시작합니다. 파일 이름 순서(1, 2, 10 순)로 페이지를 정렬합니다. 한 번에 최대 3000장, 압축을 푼 크기 8GB까지 받습니다.

## 설정 (.env)

자세한 항목은 `.env.example`에 있습니다. 자주 바꾸는 것:

| 항목 | 설명 |
|---|---|
| `MT_LLM_API` / `MT_LLM_URL` / `MT_LLM_KEY` / `MT_LLM_MODEL` | 번역 백엔드. `ollama` 또는 `openai`(OpenAI 호환) |
| `MT_INPAINT` | 그림 위 글자 지우기: `lama`(기본) / `opencv`(모델 없음) / `moss`(아래 참고). LaMa는 글자 영역을 통째로 메우므로 글자가 인물 위에 걸치면 그 부분이 뭉개질 수 있습니다 |
| `MT_FONT_*` | 역할별 글꼴 경로(대사·외침·손글씨) |
| `MT_HOST` | 기본 `127.0.0.1`(이 PC에서만). 다른 기기에서 접속하려면 `0.0.0.0` — **인증이 없어** 같은 네트워크의 누구나 원본을 받고, 번역(API 비용)을 돌리고, 작품을 지울 수 있으니 믿을 수 있는 네트워크에서만 |

### 고급: MOSS로 그림 위 글자 지우기 (`MT_INPAINT=moss`)

[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) + sorryhyun의 [sanitize LoRA](https://huggingface.co/sorryhyun/MOSS-I2I-LORAs)를 ComfyUI로 돌립니다. LaMa보다 자연스럽지만 VRAM ~26GB가 필요합니다.

- ComfyUI(네이티브 MOSS-Image 지원 버전) + [ComfyUI-MOSS-Toolkit](https://github.com/sorryhyun/ComfyUI-MOSS-Toolkit) 0.10 이상 + [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler)
- `models/loras/moss_i2i_sanitize_v3.safetensors`, MOSS-Image-Base, Qwen3.5 텍스트 인코더(예: [int8](https://huggingface.co/sorryhyun/MOSS-Image-Qwen3.5-9B-int8)) — ComfyUI 안의 이름은 `MT_MOSS_*`로 지정
- `MT_COMFY_DIR`를 지정하면 필요할 때 ComfyUI를 자동으로 띄웁니다. `MT_BEFORE_GPU`에 그 직전 실행할 명령(예: 다른 GPU 프로그램 끄기)을 넣을 수 있습니다

## 어떻게 동작하나

```
페이지 ─ 검출(말풍선 안/밖 구분) ─ 만화 OCR ─ 칸 단위 읽기 순서
       └ 작품 설정집(줄거리·인물·호칭·용어, 일부 페이지 그림 참고, 인물 자동 검증)
       └ 페이지별 번역(설정집 + 앞뒤 페이지 대사 + 페이지 그림) → 블록별 {대사/설명문/효과음/간판/기호, 번역}
       └ 지우기: 말풍선 안 = 흰 바탕 마스크로 기계 지우기 / 그림 위 = 인페인팅 후 그 영역만 합성
       └ 식자 엔진
```

**식자 엔진** — 블록마다 아래 순서로 시도하고, 앞에서 성공하면 끝납니다. 마지막 단계는 항상 결과를 냅니다.

1. 작품 기본 글자 크기(원문 글자 크기에서 추정)로 어절 단위 줄바꿈 (줄 폭이 고르게 나누는 DP)
2. 여백·줄 간격 좁히기
3. 한 어절짜리 외침이면 음절 단위로 끊기
4. **말풍선 키우기** — 원래 모양대로 가로 위주로 부풀려 다시 그림(같은 칸 안, 다른 말풍선과 안 겹칠 때만, 꼬리도 옮겨 붙임)
5. LLM에 더 짧은 표현 후보 요청(말투가 바뀐 후보는 탈락)
6. 최종 수단: 하한을 넘어 줄이기

배치 후보는 실제로 그려 보고 줄끼리 맞닿으면 버립니다. 원문 글자가 크게 쓰인 대사(외침·강조)는 크게, 그림 위 손글씨는 손글씨체로 넣습니다.

### 명령줄로 쓰기

```bash
# books/<이름>/src/ 에 페이지 이미지를 넣고
uv run python mt/book.py books/<이름> ocr
uv run python mt/book.py books/<이름> bible
uv run python mt/book.py books/<이름> translate
uv run python mt/book.py books/<이름> render      # → books/<이름>/ko/
```

## 크레딧·라이선스

코드는 MIT. 사용하는 모델·글꼴은 각자의 라이선스를 따릅니다.

| 구성 | 라이선스 |
|---|---|
| [anime_tools](https://github.com/sorryhyun/anime_tools) (만화 OCR 리더 연결) — sorryhyun | MIT |
| [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) + [만화 LoRA](https://huggingface.co/sorryhyun/paddleocr-vl-1.6-manga-lora) | Apache-2.0 |
| [comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) — ogkalu | Apache-2.0 |
| [LaMa](https://github.com/advimman/lama) 가중치 ([simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) 릴리스의 TorchScript 파일을 받음, 그 코드는 쓰지 않음) | Apache-2.0 |
| anime_tools가 함께 설치하는 [SAM 3](https://github.com/facebookresearch/sam3) 코드 (이 도구는 SAM 모델을 받거나 쓰지 않음) | SAM License |
| MOSS-Image(선택) / sanitize LoRA(선택, 직접 받기) | Apache-2.0 / 제작자 표기 따름 |
| 글꼴: 나눔스퀘어라운드·나눔펜([네이버 나눔글꼴](https://hangeul.naver.com/font)), 도현(배달의민족), Gothic A1(한양정보통신) | SIL OFL 1.1 (`fonts/`, 글꼴마다 라이선스 파일) |

나눔스퀘어라운드: 이 프로그램에는 네이버에서 제공한 나눔글꼴이 적용되어 있습니다. (© NAVER Corporation, 디자인 산돌커뮤니케이션)

## 주의

개인적인 감상을 돕는 도구입니다. 저작권이 있는 작품의 번역본을 배포하는 것은 저작권 침해가 될 수 있으며, 그 책임은 사용자에게 있습니다.
