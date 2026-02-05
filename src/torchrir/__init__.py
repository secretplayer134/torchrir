"""TorchRIR public API."""

from .sim import SimulationConfig, default_config, simulate_dynamic_rir, simulate_rir
from .signal import DynamicConvolver
from .infra import LoggingConfig, get_logger, setup_logging
from .viz import animate_scene_gif
from .io import build_metadata, save_metadata_json
from .viz import plot_scene_dynamic, plot_scene_static, plot_scene_and_save
from .models import MicrophoneArray, RIRResult, Room, Scene, Source
from .sim import FDTDSimulator, ISMSimulator, RIRSimulator, RayTracingSimulator
from .signal import convolve_rir, fft_convolve
from .datasets import (
    BaseDataset,
    CmuArcticDataset,
    CmuArcticSentence,
    choose_speakers,
    CollateBatch,
    collate_dataset_items,
    DatasetItem,
    LibriSpeechDataset,
    LibriSpeechSentence,
    list_cmu_arctic_speakers,
    SentenceLike,
    load_dataset_sources,
    TemplateDataset,
    TemplateSentence,
)
from .io import load_wav_mono, save_wav
from . import infra, io, models, sim, signal, viz
from . import geometry
from .utils import (
    att2t_SabineEstimation,
    att2t_sabine_estimation,
    beta_SabineEstimation,
    DeviceSpec,
    estimate_beta_from_t60,
    estimate_t60_from_beta,
    resolve_device,
    t2n,
)

__all__ = [
    "MicrophoneArray",
    "Room",
    "Source",
    "RIRResult",
    "RIRSimulator",
    "ISMSimulator",
    "RayTracingSimulator",
    "FDTDSimulator",
    "convolve_rir",
    "att2t_SabineEstimation",
    "att2t_sabine_estimation",
    "beta_SabineEstimation",
    "DeviceSpec",
    "BaseDataset",
    "CmuArcticDataset",
    "CmuArcticSentence",
    "choose_speakers",
    "CollateBatch",
    "collate_dataset_items",
    "DatasetItem",
    "LibriSpeechDataset",
    "LibriSpeechSentence",
    "DynamicConvolver",
    "estimate_beta_from_t60",
    "estimate_t60_from_beta",
    "fft_convolve",
    "get_logger",
    "list_cmu_arctic_speakers",
    "LoggingConfig",
    "animate_scene_gif",
    "build_metadata",
    "resolve_device",
    "SentenceLike",
    "load_dataset_sources",
    "load_wav_mono",
    "TemplateDataset",
    "TemplateSentence",
    "geometry",
    "infra",
    "io",
    "models",
    "sim",
    "signal",
    "viz",
    "plot_scene_dynamic",
    "plot_scene_and_save",
    "plot_scene_static",
    "save_wav",
    "save_metadata_json",
    "Scene",
    "setup_logging",
    "SimulationConfig",
    "default_config",
    "simulate_dynamic_rir",
    "simulate_rir",
    "t2n",
]
