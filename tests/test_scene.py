import pytest
import torch

from torchrir import MicrophoneArray, Room, Scene, Source


def test_scene_validate_static():
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    sources = Source.positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.positions([[2.0, 1.5, 1.0]])
    scene = Scene(room=room, sources=sources, mics=mics)
    scene.validate()


def test_scene_validate_dynamic_shapes():
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    sources = Source.positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.positions([[2.0, 1.5, 1.0]])
    src_traj = torch.tensor(
        [
            [[1.0, 1.0, 1.0]],
            [[1.5, 1.0, 1.0]],
        ]
    )
    mic_traj = torch.tensor(
        [
            [[2.0, 1.5, 1.0]],
            [[2.2, 1.5, 1.0]],
        ]
    )
    scene = Scene(room=room, sources=sources, mics=mics, src_traj=src_traj, mic_traj=mic_traj)
    scene.validate()


def test_scene_validate_mismatch_time():
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    sources = Source.positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.positions([[2.0, 1.5, 1.0]])
    src_traj = torch.tensor(
        [
            [[1.0, 1.0, 1.0]],
            [[1.5, 1.0, 1.0]],
        ]
    )
    mic_traj = torch.tensor(
        [
            [[2.0, 1.5, 1.0]],
            [[2.2, 1.5, 1.0]],
            [[2.4, 1.5, 1.0]],
        ]
    )
    scene = Scene(room=room, sources=sources, mics=mics, src_traj=src_traj, mic_traj=mic_traj)
    with pytest.raises(ValueError):
        scene.validate()
