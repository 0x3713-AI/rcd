"""
Central configuration for the UAV-PDD2023 pavement-distress detection pipeline.

Everything downstream (data prep, dataset, model, train, evaluate, infer) reads
from a single `Config` instance so the stages stay in sync. Override fields on
the CLI (see pipeline.py) or by editing `CFG` below.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Data ---
    dataset_name: str = "vikhyatk/uav-pdd2023"
    train_split: str = "train"
    val_fraction: float = 0.1
    seed: int = 42

    # Fixed, human-readable class order (matches the six UAV-PDD2023 distress
    # types from the source paper: longitudinal / transverse / oblique /
    # alligator cracks, patching/repair, potholes). If the dataset's raw
    # `objects[i]["name"]` strings don't match these keys, `data/load.py`
    # falls back to deriving a sorted class map straight from the data so the
    # pipeline still runs -- but this explicit map is what you want for
    # reproducible label ids across runs and checkpoints.
    canonical_class_map: dict = field(
        default_factory=lambda: {
            "longitudinal crack": 0,
            "transverse crack": 1,
            "oblique crack": 2,
            "alligator crack": 3,
            "repair": 4,
            "pothole": 5,
        }
    )

    # --- Model / image processor ---
    # "facebook/detr-resnet-50" is the reference DEtection TRansformer.
    # Swap-in candidates if small, thin objects (hairline cracks) underperform:
    #   - "SenseTime/deformable-detr"  (deformable attention, better on small objects)
    #   - "PekingU/rtdetr_r50vd"       (RT-DETR, strong small-object accuracy + faster)
    # model.py builds via AutoModelForObjectDetection/AutoImageProcessor, so
    # any of these can be dropped in without touching other stages.
    model_checkpoint: str = "facebook/detr-resnet-50"
    image_shortest_edge: int = 480
    image_longest_edge: int = 800

    # --- Training ---
    output_dir: str = "runs/uav-pdd-detr"
    num_train_epochs: int = 50
    max_train_steps: int = -1  # If > 0, overrides num_train_epochs for quick testing
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    backbone_learning_rate: float = 1e-6  # lower LR for the pretrained CNN backbone
    weight_decay: float = 1e-4
    warmup_steps: int = 50
    max_grad_norm: float = 0.1
    logging_steps: int = 25
    save_total_limit: int = 3
    dataloader_num_workers: int = 4
    mixed_precision: bool = True  # fp16 on CUDA

    # --- Evaluation / inference ---
    eval_score_threshold: float = 0.5
    eval_batch_size: int = 4


CFG = Config()
