import pytest

from torchrir.experimental import FDTDSimulator, RayTracingSimulator, TemplateDataset


def test_experimental_simulator_init_warns() -> None:
    with pytest.warns(RuntimeWarning, match="experimental"):
        RayTracingSimulator()
    with pytest.warns(RuntimeWarning, match="experimental"):
        FDTDSimulator()


def test_experimental_dataset_init_warns() -> None:
    with pytest.warns(RuntimeWarning, match="experimental"):
        TemplateDataset()
