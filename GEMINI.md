# GEMINI.md

This file provides guidance and instructions for Antigravity (`agy`) when working with code in this repository.

## 프로젝트 개요

DINOv3 백본에 LoRA(PEFT)와 선형 분류 헤드를 결합하여 철스크랩(Iron-Scraps) 이미지를 3-클래스(`cut`, `danger`, `excluded`)로 미세 분류(FGVC)하는 **비교 실험 레포지토리**입니다.
손실 함수 및 데이터셋 구성별 성능 비교 실험을 목표로 하며, 학습·평가 코드는 `fgvc/` 단일 패키지로 통합되어 있습니다.

---

## 실행 환경 및 전제 조건

- **Python 버전**: `>=3.12`
- **패키지 관리자**: `uv` (PyTorch는 CUDA 12.4 인덱스 `https://download.pytorch.org/whl/cu124` 고정)
- **외부 필수 자원** (모두 `.gitignore` 대상):
  1. 프로젝트 루트에 Meta DINOv3 공식 레포 클론:
     ```bash
     git clone https://github.com/facebookresearch/dinov3.git
     ```
     (`torch.hub.load(source="local")`가 루트의 `dinov3` 경로를 참조)
  2. 백본 가중치(`.pth`): `models/dino/weights/backbone/` 경로에 배치
  3. 데이터셋: `data/Iron-Scraps/` 경로에 배치

---

## 주요 명령어

```bash
# 가상환경 및 의존성 동기화
uv sync

# 학습 + 평가 + 시각화 파이프라인 (반드시 프로젝트 루트에서 실행)
uv run python3 -m fgvc.cli train --experiment lora_focal --model vitl16 --dataset unique_sampling_0pct

# Optuna 하이퍼파라미터 최적화 (기본 30 trials)
uv run python3 -m fgvc.cli hpo --experiment lora_focal --model vitl16 --dataset unique_sampling_0pct

# 실험 행렬 일괄 실행 (--dry-run은 조합/경로만 출력)
uv run python3 -m fgvc.cli sweep --experiments all --models vitl16,vits16 \
  --datasets unique_sampling_0pct,team_share_0pct --dry-run

# 데이터셋 빌드 파이프라인 (8:2 분할 기준)
uv run python3 preprocess_pipeline/select_dominance_label_train_val_8_2/main.py

# 발표용 Attention / Grad-CAM 히트맵 생성
uv run python3 -m fgvc.cli present --experiment lora_focal --model vitl16 --dataset unique_sampling_0pct

# 백본 패치 토큰 PCA -> RGB 시각화
uv run python3 -m fgvc.cli pca --model vitl16 --dataset unique_sampling_0pct
```

### 실행 시 주의사항
- **`-m fgvc.cli` 형태로 실행**: 그래야 CWD가 `sys.path[0]`이 되어 `fgvc` 패키지가 해석됩니다. 파일 경로로 직접 실행하면 `ModuleNotFoundError`가 발생합니다.
- **CLI 인자 + 프롬프트 폴백**: 인자를 생략하면 `input()` 프롬프트로 폴백합니다. 자동화 스크립트에서는 인자를 반드시 넘겨야 합니다(누락 시 입력 대기 상태가 됩니다).
- **테스트 코드**: `fgvc/engine/test.py`는 단위 테스트(pytest)가 아니라 모델 테스트셋 추론/평가(Inference) 코드입니다.

---

## 아키텍처 및 핵심 규칙

### 1. `hyperparams.yaml` (단일 설정 원천)
- 데이터셋 경로, 백본 가중치 경로, 학습 하이퍼파라미터가 통합 관리됩니다.
- **2단 키 구조**: 사용자가 프롬프트에 입력하는 키는 축약형(`vits16`, `unique_sampling_0pct`)이며, 실제 모델명 및 결과 경로에는 하위의 `NAME`(`dinov3_vits16`)이 매핑됩니다.
- `PATH` / `DATASET_DIR` 키를 스캔하여 동적으로 선택지를 생성하므로, 새 모델/데이터셋은 YAML 설정에 추가하면 자동 반영됩니다.
- `fgvc/config.py`가 YAML + 선택 키 3개를 받아 `ExperimentConfig`(frozen dataclass)로 조립합니다. **산출물 경로 조립은 이 클래스의 프로퍼티에서만** 수행하며, 다른 모듈에서 경로 문자열을 직접 만들지 않습니다.

### 2. 실험은 YAML의 4개 축으로 정의된다
`hyperparams.yaml`의 `experiment:` 블록이 실험을 규정합니다. 새 실험 추가 시 **코드 수정이 필요 없습니다**.

| 축 | 값 | 분기 위치 |
| :--- | :--- | :--- |
| `BACKBONE_TUNING` | `lora` / `frozen` | `fgvc/models.py: build_model()` |
| `HEAD` | `mlp`(512→256) / `linear`(1-layer) | `fgvc/models.py: build_head()` |
| `LOSS` | `cross_entropy` / `focal` | `fgvc/losses.py: build_criterion()` |
| `SELECTION_METRIC` | `pr_auc` / `mcc` | `fgvc/pipeline.py: run_train_val()` |

기본 제공 실험은 `lora_ce`, `lora_focal`, `frozen_ce` 3종입니다.
`SELECTION_METRIC`은 베스트 체크포인트 선정 기준이자 HPO 목적 지표로 함께 쓰이므로 두 값이 어긋날 수 없습니다.

> **설계 경계**: 파라미터는 YAML, 제어 흐름은 코드. 2-stage 학습처럼 루프 구조 자체가 다른 변형은 설정 축으로 표현하지 말고 `fgvc/pipeline.py`에 별도 함수로 추가합니다.

### 3. 클래스 인덱스 고정 관례
- `torchvision.datasets.ImageFolder`의 알파벳순 정렬에 따라 클래스 인덱스가 고정됩니다:
  - `0`: `cut`
  - `1`: `danger` (주요 탐지 대상)
  - `2`: `excluded`
- 코드 전반에서 `precision_score(...)[1]`과 같이 **인덱스 1이 `danger`로 하드코딩**되어 있으므로 클래스 추가/변경 시 주의해야 합니다.

### 4. Train -> Test 임계값(Threshold) 핸드오프 계약
오탐 억제를 위해 검증 시점의 최적 임계값을 테스트에 전달하는 설계입니다:
1. `fgvc/engine/validate.py`: 매 epoch마다 **precision >= 0.90**을 만족하는 최소 threshold를 계산.
2. `fgvc/pipeline.py: run_train_val()`: 베스트 epoch 달성 시 `config.threshold_path`에 기록.
3. `fgvc/engine/test.py: test_model()`: 해당 파일을 읽어 `danger` 클래스 예측 확률이 threshold 이상인 경우 **예측값을 강제로 1(`danger`)로 override**한 뒤 지표를 산출.
4. 파일이 없을 경우 경고 후 일반 argmax로 폴백합니다.

### 5. 전처리 파이프라인 (`preprocess_pipeline/`)
- **클러스터 기반 분할**: DINOv3 임베딩 클러스터 단위로 train/val/test를 분할하여 객체 중복으로 인한 Data Leakage 방지.
- **Cascade Consensus**: 클러스터 내 지배적인 라벨로 다수결 교정 (동률 시 `unknown` 처리).
- **데이터 빌드 후 필수 수동 삭제**: 출력 폴더에서 `background_bg`와 `unknown` 폴더는 반드시 수동 제거 후 학습을 시작해야 합니다.

### 6. 결과물 경로 규칙
- `results/{experiment}/{model_name}/{dataset_name}/{val|test}/`: 러닝 커브, Confusion Matrix, PR 커브, `metrics_*.md`, YAML 스냅샷 저장.
- 경로에 `experiment`가 포함되므로 실험 간 산출물이 서로 덮어쓰지 않습니다.
- `results_pca_visualization/`: 백본 토큰 PCA 결과 저장.
- `results/`, `results_no_lora/` 아래의 구 경로(`{model}/{dataset}/`) 산출물은 통합 이전의 과거 기록입니다.

---

## Attention 히트맵의 전역 캐시

`fgvc/metrics/attention.py`는 몽키패치로 어텐션 가중치를 모듈 전역(`captured_attn_weights`, `captured_attn_gradients`)에 캡처합니다. 샘플 순회 시 **매 반복 후 `attention.captured_* = None`으로 초기화**해야 이전 샘플 값이 새지 않습니다.

Grad-CAM은 backward가 필요하므로 `inference_mode` 안에서는 생성할 수 없습니다. `test.py`는 `gradcam_heatmap=None`(3단 플롯), `fgvc/scripts/presentation.py`는 실제 Grad-CAM(4단 플롯)을 사용합니다.

---

## 과거 구조에 대한 참고

`linear_head*` 4개 폴더로 나뉘어 있던 복제 구조를 `fgvc/` 단일 패키지로 통합했습니다. supercon(Focal + SupCon 2-Stage)은 실험에서 배제되어 제거했으며 git 히스토리에서 복구 가능합니다.

---

## 문서 및 코딩 규약

- 프로젝트 내 문서(README, 보고서 등)는 **한국어** 작성을 기본으로 합니다.
- 코드, CLI 명령어, 라이브러리/모델명, 파일 경로, 메트릭 명칭 등 기술 식별자는 원문(영어)을 유지합니다.

---

## 계획 수립 및 자율 실행 규약

- **플랜 모드(`/plan`)**:
  - 작업 착수 전 영향 범위를 분석하고, 목표·변경 대상 파일·상세 diff·검증 절차를 포함한 구현 계획서(Implementation Plan artifact)를 반드시 작성하여 사용자에게 제안한다.
  - 사용자의 명시적 승인이 있기 전에는 파일 수정이나 실행 명령을 진행하지 않는다.
- **실행 및 편집 모드 (Accept Edits / 승인 후 실행)**:
  - 계획이 승인되었거나 일반 실행 단계에서는 파일 생성·수정, 터미널 명령어 실행, 검증 작업 등을 중간에 불필요하게 되묻지 않고 계획대로 자율적·연속적으로 끝까지 완수한다.

