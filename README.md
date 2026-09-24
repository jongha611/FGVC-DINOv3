# FGVC DINOv3 with Linear Head (LoRA PEFT)

Meta의 **DINOv3** 백본(Backbone) 네트워크에 PEFT(LoRA) 기법과 다양한 목적의 손실 함수(Loss Function) 기반 선형 분류 헤드(Linear Head)를 결합하여, 철스크랩(Iron-Scraps) 데이터셋의 미세 분류(Fine-Grained Visual Categorization, FGVC)를 수행하는 프로젝트입니다.

---

## 📌 주요 특징
* **Backbone**: DINOv3 (`vit_s16`, `vit_b16`, `vit_l16`, `vit_h16plus`, `vit7b16` 지원)
* **Efficient Fine-Tuning**: LoRA(Low-Rank Adaptation)를 통한 매개변수 효율적 튜닝 (`qkv`, `proj` 타겟팅)
* **Data Preprocessing Pipeline**: Data Leakage 방지를 위한 클러스터링(Clustering) 및 오탐 방지를 위한 다수결(Majority Voting) 기반 자동 전처리
* **Configuration-Driven Experiments**: 손실 함수·헤드 구조·파인튜닝 전략을 `hyperparams.yaml`의 `experiment` 블록으로 정의 (코드 수정 없이 실험 추가)
* **Hyperparameter Optimization**: `Optuna`를 이용한 하이퍼파라미터 자동 최적화 지원
* **Experiment Sweep**: experiment × model × dataset 행렬을 단일 명령으로 일괄 실행

---

## ⚙️ 준비 사항 (Setup)

### 1. DINOv3 공식 레포지토리 클론
프로젝트 루트 폴더 내에 Meta의 공식 DINOv3 레포지토리를 클론해 주세요.
```bash
git clone https://github.com/facebookresearch/dinov3.git
```

### 2. 프리트레인 가중치 배치
공식 레포지토리에서 다운로드한 백본 가중치(`.pth`) 파일들을 아래 경로 규격에 맞춰 배치합니다.
> [!IMPORTANT]
> **가중치 배치 경로**: `models/dino/weights/backbone/`

---

## 📂 프로젝트 핵심 구조 (Directory Structure)
```
├── README.md
├── hyperparams.yaml                        # 학습 하이퍼파라미터 및 가중치/데이터셋 경로 통합 설정
├── preprocess_pipeline/                    # 데이터셋 빌드 및 전처리 파이프라인 (Data Leakage 및 오탐 방지)
├── fgvc/                                   # 학습·평가 단일 패키지
│   ├── config.py                           #   실험 설정 조립 및 산출물 경로의 단일 원천
│   ├── models.py / losses.py               #   백본·헤드·손실 함수 조립
│   ├── engine/                             #   train / validate / test 루프
│   ├── metrics/                            #   지표 시각화, Attention·Grad-CAM
│   ├── pipeline.py / hpo.py / sweep.py     #   파이프라인, Optuna HPO, 실험 행렬 러너
│   └── cli.py                              #   진입점
├── data/                                   # 철스크랩 원본 및 빌드된 데이터셋 (set_with_testset 등)
├── dinov3/                                 # Meta DINOv3 공식 레포지토리
├── models/
│   └── dino/weights/backbone/              # 다운로드한 DINOv3 가중치 (.pth)
└── results/                                # 성능 메트릭 요약 및 그래프, Attention Heatmap 저장소
```

*(전처리 파이프라인 폴더 내부(`preprocess_pipeline/`)에는 해당 모듈의 작동 원리를 담은 별도의 상세 `README.md`가 존재합니다.)*

---

## 🚀 실행 가이드 (Usage)
프로젝트 패키지 및 런타임 관리에 `uv`를 사용합니다.

```bash
# 1. 가상환경 및 의존성 동기화
uv sync

# 2. 데이터셋 전처리 파이프라인 가동 (필요 시)
uv run python3 preprocess_pipeline/select_dominance_label_train_val_8_2/main.py

# 3. 학습 + 테스트 + 시각화 (단일 실행)
uv run python3 -m fgvc.cli train --experiment lora_focal --model vitl16 --dataset unique_sampling_0pct

# 4. 하이퍼파라미터 최적화 (Optuna)
uv run python3 -m fgvc.cli hpo --experiment lora_focal --model vitl16 --dataset unique_sampling_0pct

# 5. 실험 행렬 일괄 실행 (--dry-run으로 계획만 먼저 확인)
uv run python3 -m fgvc.cli sweep --experiments all --models vitl16,vits16 \
  --datasets unique_sampling_0pct,team_share_0pct --dry-run
```
*(인자를 생략하면 기존처럼 터미널 프롬프트로 선택할 수 있습니다.)*

### 실험 정의
학습 설정은 `hyperparams.yaml`의 `experiment` 블록에서 4개 축으로 정의합니다.
새 실험은 이 블록에 항목을 추가하는 것만으로 만들 수 있습니다.

| experiment | BACKBONE_TUNING | HEAD | LOSS | SELECTION_METRIC |
| :--- | :--- | :--- | :--- | :--- |
| `lora_ce` | lora | mlp (512→256) | cross_entropy | pr_auc |
| `lora_focal` | lora | mlp (512→256) | focal | mcc |
| `frozen_ce` | frozen (Linear Probing) | linear (1-layer) | cross_entropy | mcc |

> `SELECTION_METRIC`은 베스트 체크포인트 선정 기준이자 HPO 목적 지표로 함께 쓰이므로
> 두 값이 어긋날 수 없습니다.

---

## 📊 학습 결과 예시 (Training Metrics)
> **Configuration**: `dinov3_vits16` on `unique_sampling_25pct` | **Best Epoch**: 101

| Metric | Value |
| :--- | :--- |
| **Train Loss** | 0.5219 |
| **Train Accuracy** | 78.36% |
| **Val Loss** | 0.6378 |
| **Val Accuracy** | 75.44% |
| **Danger Precision** | 0.7093 |
| **Danger Recall** | 0.6869 |
| **Danger F1 Score** | 0.6979 |
| **MCC** | 0.5860 |
| **Danger PR-AUC** | 0.7813 |
| **Danger F-beta (0.5)** | 0.7047 |

---

## 📈 시각화 자료 (Visualization)

산출물은 `results/{experiment}/{model}/{dataset}/{val|test}/` 아래에 저장되어
실험 간에 서로 덮어쓰지 않습니다. 생성되는 자료는 다음과 같습니다:
1. **학습 곡선 (Learning Curve)**: Loss, Accuracy, F1, PR-AUC 등 Epoch에 따른 지표 변화
2. **혼동 행렬 (Confusion Matrix)**: 모델의 예측 분포 확인
3. **어텐션 히트맵 (Attention Map)**: DINOv3의 Self-Attention 맵을 시각화하여 모델이 이미지를 어떻게 바라보는지 검증

</details>
