# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

DINOv3 백본에 LoRA(PEFT)와 선형 분류 헤드를 결합해 철스크랩(Iron-Scraps) 이미지를 3-클래스로 미세 분류(FGVC)하는 **실험 레포**입니다. 프로덕션 서비스 코드가 아니라 손실 함수·데이터셋 구성별 비교 실험이 목적입니다.

## 실행 전제 (모두 `.gitignore` 대상 — 클론 직후에는 없음)

1. 프로젝트 루트에 `git clone https://github.com/facebookresearch/dinov3.git` (`torch.hub.load(source="local")`가 이 경로를 참조)
2. 백본 가중치 `.pth`를 `models/dino/weights/backbone/` 에 배치
3. 데이터셋을 `data/Iron-Scraps/` 에 배치

## 명령어

```bash
uv sync                          # 의존성 동기화 (Python 3.12, torch는 cu124 인덱스 고정)

# 학습 + 테스트 + 샘플 시각화
uv run python3 -m fgvc.cli train --experiment lora_focal --model vitl16 --dataset unique_sampling_0pct

# Optuna HPO (기본 30 trials, --n-trials로 조정)
uv run python3 -m fgvc.cli hpo --experiment lora_focal --model vitl16 --dataset unique_sampling_0pct

# 실험 행렬 일괄 실행 (--dry-run이면 조합·산출물 경로만 출력하고 종료)
uv run python3 -m fgvc.cli sweep --experiments all --models vitl16,vits16 \
  --datasets unique_sampling_0pct,team_share_0pct --dry-run

uv run python3 -m fgvc.cli present --experiment lora_focal --model vitl16 --dataset unique_sampling_0pct
uv run python3 -m fgvc.cli pca --model vitl16 --dataset unique_sampling_0pct

# 데이터셋 빌드 (8:2 분할)
uv run python3 preprocess_pipeline/select_dominance_label_train_val_8_2/main.py
```

- **반드시 `-m fgvc.cli` 형태로 실행하세요.** 그래야 CWD가 `sys.path[0]`이 되어 `fgvc` 패키지가 해석됩니다. 파일 경로로 직접 실행하면 `ModuleNotFoundError`가 납니다.
- CLI 인자를 생략하면 `input()` 프롬프트로 폴백합니다. 자동화 스크립트에서는 반드시 인자를 넘기세요 (누락 시 조용히 입력 대기 상태가 됩니다).
- 설정 오류(`ValueError`/`FileNotFoundError`)는 `cli.main()`이 잡아 트레이스백 없이 `[설정 오류]` 메시지로 출력합니다.
- 테스트 프레임워크·린터·CI가 없습니다. `fgvc/engine/test.py`는 pytest가 아니라 **모델 평가(testset inference)** 코드입니다.

## 아키텍처에서 먼저 알아야 할 것

### `fgvc/config.py` — 설정과 경로의 단일 원천

`hyperparams.yaml` + 선택 키 3개(experiment / model / dataset)를 받아 `ExperimentConfig`(frozen dataclass)를 조립합니다. **산출물 경로 조립은 전부 이 클래스의 프로퍼티에서만 이뤄집니다** — 다른 모듈에서 `f"results/..."` 같은 경로 문자열을 만들지 마세요.

```
config.checkpoint_path   → {CHECKPOINT_DIR}/best_model_{experiment}_{dataset}_{model}.pth
config.threshold_path    → {CHECKPOINT_DIR}/best_val_threshold_90_{experiment}_{dataset}_{model}.txt
config.results_dir(split)→ {RESULTS_ROOT}/{experiment}/{model}/{dataset}/{split}
```

경로에 `experiment`가 들어가는 것이 핵심입니다. 과거 폴더 복제 구조에서는 CE 실험과 Focal 실험이 같은 경로에 써서 서로를 덮어썼습니다.

HPO는 `config.with_overrides(**kwargs)`로 일부 값만 바꾼 사본을 만들어 씁니다 (알 수 없는 필드명은 즉시 `ValueError`).

### 실험은 YAML의 4개 축으로 정의된다

`hyperparams.yaml`의 `experiment:` 블록이 실험을 규정합니다. 새 실험을 추가할 때 **코드를 건드릴 필요가 없습니다** — 항목 하나만 더하면 됩니다.

| 축 | 값 | 분기 위치 |
|---|---|---|
| `BACKBONE_TUNING` | `lora` / `frozen` | `fgvc/models.py: build_model()` |
| `HEAD` | `mlp`(512→256) / `linear`(1-layer) | `fgvc/models.py: build_head()` |
| `LOSS` | `cross_entropy` / `focal` | `fgvc/losses.py: build_criterion()` |
| `SELECTION_METRIC` | `pr_auc` / `mcc` | `fgvc/pipeline.py: run_train_val()` |

`SELECTION_METRIC`은 **베스트 체크포인트 선정 기준이자 HPO 목적 지표**로 함께 쓰입니다. 두 곳이 같은 값을 읽으므로 어긋날 수 없습니다(과거 `no_lora` 폴더에서 실제로 어긋나 있던 버그).

> **설계 경계**: 파라미터는 YAML, 제어 흐름은 코드. 2-stage 학습처럼 루프 구조 자체가 다른 변형은 설정 축으로 우겨넣지 말고 `fgvc/pipeline.py`에 별도 함수로 추가하세요.

### 클래스 인덱스 1 = `danger` 고정 관례

`ImageFolder`가 폴더명을 알파벳순 정렬하므로 클래스 순서는 `cut`(0) / `danger`(1) / `excluded`(2)입니다. 학습·검증·테스트 코드 전반에서 `precision_score(...)[1]` 처럼 **인덱스 1을 하드코딩**해 `danger` 클래스 지표만 추적합니다. 클래스를 추가·개명하면 이 인덱싱이 조용히 깨집니다.

### Train → Test 사이의 threshold 파일 핸드오프

오탐 억제가 중요한 도메인 요구를 반영한 설계이며, 이 레포에서 가장 비자명한 교차 모듈 계약입니다.

1. `fgvc/engine/validate.py`가 매 epoch **precision ≥ 0.90을 만족하는 최소 threshold**를 계산해 반환 (반환값 10개 튜플의 9번째)
2. `fgvc/pipeline.py: run_train_val()`이 베스트 epoch 갱신 시 `config.threshold_path`에 기록
3. `fgvc/engine/test.py: test_model()`이 그 파일을 읽어 `danger` 확률이 threshold 이상인 샘플의 예측을 **강제로 1(danger)로 덮어쓴** 뒤 지표를 재계산

따라서 test 지표는 순수 argmax 결과가 아닙니다. MCC가 낮게(0.24~0.35) 보이는 것도 이 override가 예측을 `danger` 쪽으로 밀기 때문이며 버그가 아닙니다. 파일이 없으면 경고 후 argmax로 폴백합니다.

### Attention 히트맵의 전역 캐시

`fgvc/metrics/attention.py`는 몽키패치로 어텐션 가중치를 모듈 전역(`captured_attn_weights`, `captured_attn_gradients`)에 캡처합니다. 샘플을 순회할 때 **매 반복 후 `attention.captured_* = None`으로 초기화**해야 이전 샘플의 값이 새지 않습니다 (`fgvc/engine/test.py`, `fgvc/scripts/presentation.py` 참고).

Grad-CAM은 backward가 필요하므로 `inference_mode` 안에서는 생성할 수 없습니다. `test.py`는 `gradcam_heatmap=None`을 넘겨 3단 플롯으로, `presentation.py`는 실제 Grad-CAM을 생성해 4단 플롯으로 렌더링합니다.

### 전처리 파이프라인

`preprocess_pipeline/`은 COCO `annotations.json` + 원본 4K 이미지를 `ImageFolder` 구조로 빌드합니다. 설계 의도는 `preprocess_pipeline/README.md`에 상세히 기술되어 있으며, 핵심 3가지는:

- **클러스터 단위 분할**: 동일 객체가 여러 사진에 중복 등장하므로, DINOv3 임베딩 클러스터링으로 묶은 뒤 **개별 이미지가 아닌 클러스터 폴더 레벨**에서 train/val/test를 나눠 Data Leakage를 차단
- **Cascade Consensus**: 클러스터 내 지배 라벨이 소수 라벨을 덮어써 오탐을 자동 교정 (빈도 동률이면 클러스터 전체를 `unknown`으로 탈락)
- **`unique_sampling` vs `vanilla`**: 클러스터당 대표 1장만 남긴 셋과, 분할 후 클러스터 내 전 샘플을 재수집한 셋을 동시 출력

두 가지 버전이 있습니다 — `select_dominance_label/`(8:1:1, 자체 테스트셋, 7 step)과 `select_dominance_label_train_val_8_2/`(8:2, 실제 테스트셋을 수동 이식하므로 clustering·testset_sampler 단계 없음, 6 step). 패딩 비율별(`base_0pct`, `base_25pct`, …)로 폴더가 통째 복제되어 있고 경로 상수만 다릅니다. 실행 전 소스 내 `CLS_DATA_DIR` 상수를 환경에 맞게 조정해야 합니다.

**빌드 후 필수 수동 조치**: 출력 데이터셋에서 `background_bg`(이전 태스크 디텍션 잔재)와 `unknown`(동률 탈락) 폴더를 전량 삭제한 뒤에 학습을 시작해야 합니다.

## 과거 구조에 대한 참고

`linear_head/`, `linear_head_focal_loss/`, `linear_head_no_lora/`, `linear_head_focal_loss_supercon/` 4개 폴더로 나뉘어 있던 복제 구조를 `fgvc/` 단일 패키지로 통합했습니다. supercon(Focal + SupCon 2-Stage)은 실험에서 배제되어 제거했으며, 필요하면 git 히스토리에서 복구할 수 있습니다.

`results/`와 `results_no_lora/` 아래의 기존 산출물은 구 경로 규칙(`{model}/{dataset}/`)으로 남아 있는 과거 기록입니다. 신규 산출물은 `results/{experiment}/{model}/{dataset}/`에 저장되므로 섞이지 않습니다.

## 문서 규약

README를 포함한 이 레포의 Markdown 문서는 모두 한국어로 작성되어 있습니다. 코드·CLI 명령어·API명·라이브러리명·지표명 등 기술 식별자는 원문을 유지합니다.
