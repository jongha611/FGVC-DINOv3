# GEMINI.md

This file provides guidance and instructions for Antigravity (`agy`) when working with code in this repository.

## 프로젝트 개요

DINOv3 백본에 LoRA(PEFT)와 선형 분류 헤드를 결합하여 철스크랩(Iron-Scraps) 이미지를 3-클래스(`cut`, `danger`, `excluded`)로 미세 분류(FGVC)하는 **비교 실험 레포지토리**입니다.
손실 함수 및 데이터셋 구성별 성능 비교 실험을 목표로 하며, 4가지 형태의 독립된 학습 모듈(`linear_head*`)을 포함합니다.

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
uv run python3 linear_head/main.py
uv run python3 linear_head_focal_loss/main.py
uv run python3 linear_head_no_lora/main.py

# Optuna 하이퍼파라미터 최적화 (30 trials)
uv run python3 linear_head/train_utils/run_optuna.py

# 데이터셋 빌드 파이프라인 (8:2 분할 기준)
uv run python3 preprocess_pipeline/select_dominance_label_train_val_8_2/main.py

# 발표용 Attention / Grad-CAM 히트맵 생성
uv run python3 main_for_presentation.py

# 백본 패치 토큰 PCA -> RGB 시각화
uv run python3 visualize_pca.py
```

### 실행 시 주의사항
- **인터랙티브 프롬프트 (`input()`)**: `main.py`는 CLI 인자가 아닌 터미널 대화형 `input()` 프롬프트로 모델과 데이터셋을 선택합니다 (`--hyperparams` 인자만 CLI 지원).
- **모듈 임포트 경로 (`PYTHONPATH`)**: `__init__.py`가 없는 PEP 420 네임스페이스 패키지이므로, 스크립트 실행 시 `ModuleNotFoundError: linear_head`가 발생할 경우 `PYTHONPATH=. uv run python3 ...`로 실행합니다.
- **테스트 코드**: `test.py` 및 `test_utils/`는 단위 테스트(pytest)가 아니라 모델 테스트셋 추론/평가(Inference) 코드입니다.

---

## 아키텍처 및 핵심 규칙

### 1. `hyperparams.yaml` (단일 설정 원천)
- 데이터셋 경로, 백본 가중치 경로, 학습 하이퍼파라미터가 통합 관리됩니다.
- **2단 키 구조**: 사용자가 프롬프트에 입력하는 키는 축약형(`vits16`, `unique_sampling_0pct`)이며, 실제 모델명 및 결과 경로에는 하위의 `NAME`(`dinov3_vits16`)이 매핑됩니다.
- `main.py`는 `PATH` / `DATASET_DIR` 키를 스캔하여 동적으로 선택지를 생성하므로, 새 모델/데이터셋은 YAML 설정에 추가하면 자동 반영됩니다.

### 2. 3종 학습 모듈 구조
동일한 인터페이스(`main.py`, `train_test_elements.py`, `get_data_loaders.py`, `train.py`, `validate.py`, `test.py`, `train_utils/`, `test_utils/`)를 공유하며 Loss 및 헤드 구성만 차이가 납니다:

| 모듈 | Loss | 백본 | 헤드 | 베스트 체크포인트 기준 |
| :--- | :--- | :--- | :--- | :--- |
| `linear_head` | CrossEntropy | LoRA | 3-layer MLP | **PR-AUC** |
| `linear_head_focal_loss` | Focal Loss | LoRA | 3-layer MLP | **MCC** |
| `linear_head_no_lora` | CrossEntropy | 동결 (Linear Probing) | 1-layer Linear | **MCC** |

> **변경 전파 주의**: 한 모듈의 공통 로직(예: `validate_model` 반환값 등)을 수정할 경우 나머지 2개 모듈에도 동기화가 필요한지 반드시 점검해야 합니다.

`train_test_elements.py`가 핵심 오케스트레이터 역할을 하며, `get_model()`, `get_modules()`, `run_train_val_process()`, `run_testset_process()` 등이 정의되어 있습니다.

### 3. 클래스 인덱스 고정 관례
- `torchvision.datasets.ImageFolder`의 알파벳순 정렬에 따라 클래스 인덱스가 고정됩니다:
  - `0`: `cut`
  - `1`: `danger` (주요 탐지 대상)
  - `2`: `excluded`
- 코드 전반에서 `precision_score(...)[1]`과 같이 **인덱스 1이 `danger`로 하드코딩**되어 있으므로 클래스 추가/변경 시 주의해야 합니다.

### 4. Train -> Test 임계값(Threshold) 핸드오프 계약
오탐 억제를 위해 검증 시점의 최적 임계값을 테스트에 전달하는 설계입니다:
1. `validate.py`: 매 epoch마다 **precision >= 0.90**을 만족하는 최소 threshold를 계산.
2. `train_test_elements.py`: 베스트 epoch 달성 시 `{CHECKPOINT_DIR}/best_val_threshold_90_{dataset}_{model}.txt`에 기록.
3. `test.py`: 해당 파일을 읽어 `danger` 클래스 예측 확률이 threshold 이상인 경우 **예측값을 강제로 1(`danger`)로 override**한 뒤 지표를 산출.
4. 파일이 없을 경우 경고 후 일반 argmax로 폴백합니다.

### 5. 전처리 파이프라인 (`preprocess_pipeline/`)
- **클러스터 기반 분할**: DINOv3 임베딩 클러스터 단위로 train/val/test를 분할하여 객체 중복으로 인한 Data Leakage 방지.
- **Cascade Consensus**: 클러스터 내 지배적인 라벨로 다수결 교정 (동률 시 `unknown` 처리).
- **데이터 빌드 후 필수 수동 삭제**: 출력 폴더에서 `background_bg`와 `unknown` 폴더는 반드시 수동 제거 후 학습을 시작해야 합니다.

### 6. 결과물 경로 규칙
- `results/{model_name}/{dataset_name}/{val|test}/`: 러닝 커브, Confusion Matrix, PR 커브, `metrics_*.md`, YAML 스냅샷 저장.
- `results_no_lora/`: Linear Probing 대조군 결과 저장.
- `results_pca_visualization/`: 백본 토큰 PCA 결과 저장.

---

## 알려진 결함 및 주의사항

1. **`run_optuna.py` 언패킹 오류**:
   `validate_model()` 함수는 10개 값을 반환하지만, 각 모듈의 `run_optuna.py`에서 8개 값으로 언패킹하고 있어 실행 시 `ValueError`가 발생할 수 있습니다 (수정 필요).
2. **미사용 시각화 코드**:
   각 모듈의 `validate.py` 내 `visualize_results()` / `visualize_confusion_matrix()`는 사문화된 코드입니다. 실제 시각화는 `train_utils/train_metrics_utils.py`를 참조합니다.

---

## 문서 및 코딩 규약

- 프로젝트 내 문서(README, 보고서 등)는 **한국어** 작성을 기본으로 합니다.
- 코드, CLI 명령어, 라이브러리/모델명, 파일 경로, 메트릭 명칭 등 기술 식별자는 원문(영어)을 유지합니다.
