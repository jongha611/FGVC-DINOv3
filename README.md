# FGVC DINOv3 with Linear Head (LoRA PEFT)

Meta의 **DINOv3** 백본(Backbone) 네트워크에 PEFT(LoRA) 기법과 다양한 목적의 손실 함수(Loss Function) 기반 선형 분류 헤드(Linear Head)를 결합하여, 철스크랩(Iron-Scraps) 데이터셋의 미세 분류(Fine-Grained Visual Categorization, FGVC)를 수행하는 프로젝트입니다.

---

## 📌 주요 특징
* **Backbone**: DINOv3 (`vit_s16`, `vit_b16`, `vit_l16`, `vit_h16plus`, `vit7b16` 지원)
* **Efficient Fine-Tuning**: LoRA(Low-Rank Adaptation)를 통한 매개변수 효율적 튜닝 (`qkv`, `proj` 타겟팅)
* **Data Preprocessing Pipeline**: Data Leakage 방지를 위한 클러스터링(Clustering) 및 오탐 방지를 위한 다수결(Majority Voting) 기반 자동 전처리
* **Multiple Loss Formulations**: 클래스 불균형을 극복하기 위해 Cross-Entropy, Focal Loss 기반 파이프라인 지원
* **Hyperparameter Optimization**: `Optuna`를 이용한 하이퍼파라미터 자동 최적화 지원

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
├── linear_head/                            # 1. Standard Cross-Entropy 기반 분류기 학습 모듈
├── linear_head_focal_loss/                 # 2. 클래스 불균형 해소를 위한 Focal Loss 기반 모듈
├── linear_head_no_lora/                    # 3. 백본 동결(Linear Probing) 대조군 모듈
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

# 3. 학습 및 테스트 파이프라인 수행 (단일 학습, 원하는 모듈 선택)
uv run python3 linear_head/main.py
# uv run python3 linear_head_focal_loss/main.py
# uv run python3 linear_head_no_lora/main.py

# 4. 하이퍼파라미터 최적화 (HPO) 파이프라인 수행 (Optuna)
uv run python3 linear_head/train_utils/run_optuna.py
```
*(실행 시 프롬프트를 통해 모델 버전과 데이터셋 종류(vanilla_50pct, unique_sampling_50pct 등)를 동적으로 선택할 수 있습니다.)*

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

`results/` 디렉토리 내에 파이프라인 가동 결과로 다음 자료들이 자동 생성 및 저장됩니다:
1. **학습 곡선 (Learning Curve)**: Loss, Accuracy, F1, PR-AUC 등 Epoch에 따른 지표 변화
2. **혼동 행렬 (Confusion Matrix)**: 모델의 예측 분포 확인
3. **어텐션 히트맵 (Attention Map)**: DINOv3의 Self-Attention 맵을 시각화하여 모델이 이미지를 어떻게 바라보는지 검증

</details>
