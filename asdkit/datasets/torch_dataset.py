import glob
import logging
from typing import List, Tuple

import torch
import torchaudio
import tqdm
from torch.utils.data import Dataset

from asdkit.models.audio_feature.base import BaseAudioFeature
from asdkit.utils.common import instantiate_tgt, read_json

logger = logging.getLogger(__name__)


def parse_path_selector(selector: str) -> List[str]:
    assert isinstance(selector, str)
    if selector.endswith(".json"):  # json format
        path_list = read_json(selector)
        logger.info(f"Loaded {len(path_list)} paths from {selector}")
    elif "*" in selector:  # glob format
        path_list = sorted(glob.glob(selector))
        logger.info(f"Loaded {len(path_list)} paths from {selector}")
    elif selector.endswith(".wav"):  # single file format
        path_list = [selector]
    else:
        raise ValueError(f"Unknown path_list format: {selector}")
    return path_list  # type: ignore


def torch_mono_wav_load(path: str, audio_channel: int | str = "first") -> torch.Tensor:
    wave, sr = torchaudio.load(path)
    if sr != 16000:
        raise ValueError(f"Unexpected sampling rate: {sr}")
    if wave.ndim != 2:
        raise ValueError(f"Unexpected wave shape: {wave.shape}")

    if wave.shape[0] == 1:
        return wave[0]

    if isinstance(audio_channel, int):
        if audio_channel < 0 or audio_channel >= wave.shape[0]:
            raise ValueError(
                f"audio_channel={audio_channel} is out of range for {wave.shape[0]} channels"
            )
        return wave[audio_channel]

    if audio_channel in ["first", "left", "near"]:
        return wave[0]
    if audio_channel in ["second", "right", "far"]:
        if wave.shape[0] < 2:
            raise ValueError("second/right/far channel was requested for mono audio")
        return wave[1]
    if audio_channel == "mean":
        return wave.mean(dim=0)
    if audio_channel == "diff":
        if wave.shape[0] < 2:
            raise ValueError("diff channel was requested for mono audio")
        return wave[0] - wave[1]

    raise ValueError(f"Unknown audio_channel: {audio_channel}")


class WaveDataset(Dataset):
    def __init__(
        self,
        path_selector_list: List[str],
        audio_channel: int | str = "first",
    ):
        super().__init__()
        self.path_list = []
        self.audio_channel = audio_channel

        logger.info("Start Loading Paths")
        for selector in path_selector_list:
            self.path_list += parse_path_selector(selector)
        logger.info("Finished Loading Paths")

    def get_item(self, idx) -> dict:
        wave = torch_mono_wav_load(
            path=self.path_list[idx], audio_channel=self.audio_channel
        )
        items = {
            "wave": wave,
            "path": self.path_list[idx],
        }
        return items

    def __getitem__(self, idx):
        return self.get_item(idx)

    def __len__(self):
        return len(self.path_list)


class AudioFeatDataset(Dataset):
    def __init__(
        self,
        path_selector_list: List[str],
        audio_feat_cfg: dict,
        audio_channel: int | str = "first",
    ):
        super().__init__()
        self.audio_feat_extractor: BaseAudioFeature = instantiate_tgt(audio_feat_cfg)
        self.path_list = []
        self.audio_channel = audio_channel

        logger.info("Start Loading Paths")
        for selector in path_selector_list:
            self.path_list += parse_path_selector(selector)
        logger.info("Finished Loading Paths")

        self.audio_feat_tensor, self.path_idx_list = self.prepare_audio_feat()
        assert len(self.audio_feat_tensor) == len(self.path_idx_list)

    def prepare_audio_feat(self) -> Tuple[torch.Tensor, List[int]]:
        logger.info("Start Extracting Audio Features")
        audio_feat_list = []
        path_idx_list = []
        for i, path in enumerate(tqdm.tqdm(self.path_list)):
            wave = torch_mono_wav_load(path=path, audio_channel=self.audio_channel)[
                None
            ]  # 1, T
            x = self.audio_feat_extractor(wave)  # N x D
            audio_feat_list += [x]
            path_idx_list += [i] * len(x)
        logger.info("Finished Extracting Audio Features")
        return torch.concat(audio_feat_list, dim=0), path_idx_list

    def get_item(self, idx) -> dict:
        items = {
            "feat": self.audio_feat_tensor[idx],
            "path": self.path_list[self.path_idx_list[idx]],
        }
        return items

    def __getitem__(self, idx):
        return self.get_item(idx)

    def __len__(self):
        return len(self.audio_feat_tensor)
