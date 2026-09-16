# FGVC-DINO-v3: Linear Head Training Modules

본 프로젝트에는 DINOv3 백본의 임베딩을 기반으로 파인튜닝(PEFT LoRA)과 분류기(Linear Head) 학습을 수행하기 위해 세 가지 다른 목적의 학습 폴더가 구성되어 있습니다. 각 폴더는 데이터의 클래스 불균형(Class Imbalance)이나 세밀한 특징 구분(Fine-grained) 등 해결하고자 하는 문제에 따라 손실 함수(Loss Function)의 구조가 다르게 설계되어 있습니다.

---

## 📂 1. `linear_head/` (Standard Cross-Entropy)
가장 기본이 되는 학습 파이프라인으로, 표준적인 **Cross-Entropy Loss**를 사용하여 분류기(Classifier)를 학습합니다.
* **주요 특징:**
  * DINOv3 백본에 LoRA(Low-Rank Adaptation)를 적용하여 효율적으로 파인튜닝을 진행합니다.
  * 클래스 간 데이터 수가 비교적 균일하거나, 기본적인 모델의 성능(Base Performance)을 측정할 때 사용합니다.
  * HPO(Hyperparameter Optimization)를 위해 Optuna가 세팅되어 있으며(`train_utils/run_optuna.py`), 학습 완료 후 Test셋에 대한 다양한 지표(PR-AUC, F1, MCC 등)와 Attention Heatmap 시각화를 제공합니다.

---

## 📂 2. `linear_head_focal_loss/` (Focal Loss)
데이터셋 내의 **클래스 불균형(Class Imbalance)** 문제를 적극적으로 해결하기 위해 도입된 파이프라인입니다. 기존 Cross-Entropy 대신 **Focal Loss**를 적용합니다.
* **주요 특징:**
  * 소수 클래스나 분류하기 어려운(Hard) 샘플에 모델이 더 집중할 수 있도록 `gamma` 파라미터를 사용합니다.
  * `gamma = 0`으로 설정할 경우, 복잡한 로직 없이 순수한 **클래스 가중치(Class Weight)** 기반의 Weighted Cross-Entropy Loss로 완벽히 치환되어 동작합니다. (알파 $\alpha$ 값만 적용)
  * 다수 클래스에 모델이 편향되는 현상을 방지하고자 할 때 최적의 선택입니다.

---

## 📂 3. `linear_head_no_lora/` (Linear Probing 대조군)
LoRA 파인튜닝의 실제 기여도를 측정하기 위한 **대조군(Baseline)** 파이프라인입니다.
* **주요 특징:**
  * DINOv3 백본 전체를 동결(Freeze)하고, `embed_dim → num_classes` 1-layer 선형 분류기만 학습합니다.
  * 사전학습된 임베딩이 그 자체로 얼마나 선형 분리 가능한지를 측정하는 Linear Probing의 정석 구성입니다.
  * LoRA 적용 모듈과 성능을 비교해 파인튜닝으로 얻은 이득을 정량화하는 데 사용합니다.
  * 결과물은 다른 모듈과 분리된 `results_no_lora/` 트리에 저장됩니다.

---

## 🚀 공통 실행 방법
각 폴더 내부의 구조는 모듈화되어 동일한 인터페이스를 가집니다. 학습 파이프라인과 Optuna 기반 하이퍼파라미터 탐색은 다음 스크립트로 실행합니다.

```bash
# 일반 파이프라인 실행 (평가 및 시각화 포함)
uv run python3 [원하는_linear_head_폴더]/main.py

# Optuna를 이용한 HPO (최적 파라미터 탐색)
uv run python3 [원하는_linear_head_폴더]/train_utils/run_optuna.py
```
*(실행 시 터미널에서 데이터셋 버전을 묻는 프롬프트가 나타나면 `vanilla_50pct` 혹은 `unique_sampling_50pct` 등으로 입력해 주시면 됩니다.)*
