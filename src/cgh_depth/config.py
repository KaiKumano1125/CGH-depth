from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


def _require_section(data: dict, key: str) -> dict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid config section: {key}")
    return value


@dataclass(frozen=True)
class PathsConfig:
    data_root: str
    weight_dir: str
    result_dir: str
    log_dir: str


@dataclass(frozen=True)
class EncoderConfig:
    res: int
    pitch: float
    wavelength: float
    depth_range_m: float
    include_rgb: bool
    include_depth: bool
    include_freq_cos: bool
    include_freq_sin: bool

    @property
    def in_channels(self) -> int:
        channels = 0
        if self.include_rgb:
            channels += 3
        if self.include_depth:
            channels += 1
        if self.include_freq_cos:
            channels += 1
        if self.include_freq_sin:
            channels += 1
        return channels


@dataclass(frozen=True)
class ModelConfig:
    name: str
    out_channels: int
    base_channels: int
    use_cross_attention: bool = False


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    shuffle: bool
    num_workers: int
    checkpoint_every: int
    resume_checkpoint: str


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint: str
    test_index: str
    batch_output_subdir: str
    prediction_prefix: str


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    paths: PathsConfig
    encoder: EncoderConfig
    model: ModelConfig
    train: TrainConfig
    inference: InferenceConfig
    config_path: Path
    project_root: Path

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def data_root(self) -> Path:
        return self.resolve_path(self.paths.data_root)

    @property
    def weight_dir(self) -> Path:
        return self.resolve_path(self.paths.weight_dir)

    @property
    def result_dir(self) -> Path:
        return self.resolve_path(self.paths.result_dir)

    @property
    def log_dir(self) -> Path:
        return self.resolve_path(self.paths.log_dir)


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path).resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    if path.parent.name == "experiments":
        project_root = path.parent.parent.parent
    else:
        project_root = path.parent.parent

    paths = _require_section(raw, "paths")
    encoder = _require_section(raw, "encoder")
    model = _require_section(raw, "model")
    train = _require_section(raw, "train")
    inference = _require_section(raw, "inference")

    return ExperimentConfig(
        experiment_name=str(raw.get("experiment_name", path.stem)),
        paths=PathsConfig(
            data_root=str(paths["data_root"]),
            weight_dir=str(paths["weight_dir"]),
            result_dir=str(paths["result_dir"]),
            log_dir=str(paths["log_dir"]),
        ),
        encoder=EncoderConfig(
            res=int(encoder["res"]),
            pitch=float(encoder["pitch"]),
            wavelength=float(encoder["wavelength"]),
            depth_range_m=float(encoder["depth_range_m"]),
            include_rgb=bool(encoder["include_rgb"]),
            include_depth=bool(encoder["include_depth"]),
            include_freq_cos=bool(encoder["include_freq_cos"]),
            include_freq_sin=bool(encoder["include_freq_sin"]),
        ),
        model=ModelConfig(
            name=str(model["name"]),
            out_channels=int(model["out_channels"]),
            base_channels=int(model["base_channels"]),
            use_cross_attention=bool(model.get("use_cross_attention", False)),
        ),
        train=TrainConfig(
            batch_size=int(train["batch_size"]),
            learning_rate=float(train["learning_rate"]),
            epochs=int(train["epochs"]),
            shuffle=bool(train["shuffle"]),
            num_workers=int(train["num_workers"]),
            checkpoint_every=int(train["checkpoint_every"]),
            resume_checkpoint=str(train.get("resume_checkpoint", "")),
        ),
        inference=InferenceConfig(
            checkpoint=str(inference["checkpoint"]),
            test_index=str(inference["test_index"]),
            batch_output_subdir=str(inference["batch_output_subdir"]),
            prediction_prefix=str(inference["prediction_prefix"]),
        ),
        config_path=path,
        project_root=project_root,
    )
