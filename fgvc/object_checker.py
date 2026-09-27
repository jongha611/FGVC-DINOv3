from config import (
    ExperimentConfig, 
    BackboneTuning, 
    HeadType, 
    LossType
)

from pathlib import Path

config = ExperimentConfig(
    experiment="test_experiment",
    model_name="test_model",
    dataset_name="test_dataset",
    model_path=Path("test_model_path"),
    dataset_dir=Path("test_dataset_dir"),
    batch_size=32,
    image_size=224,
)

print(config)