import logging
from typing import Any, Dict, Optional

from torch import nn

from asdkit.models.pooling import frequency_pooling, sequence_to_time_frequency
from asdkit.models.pretrained_models.beats import restore

from .base import BaseFrozenModel

logger = logging.getLogger(__name__)


class BEATsFrozenModel(BaseFrozenModel):
    def __init__(self, model_cfg: Optional[dict] = None):
        """
        Args:
            model_cfg (Dict[str, Any]): Configuration for the model. Parameters in this dictionary are used in `self.construct_model`.
        """
        super().__init__(model_cfg=model_cfg)

    def construct_model(
        self,
        ckpt_path: str,
        update_cfg: Optional[dict] = None,
    ) -> nn.Module:
        model, _ = restore(ckpt_path=ckpt_path, update_cfg=update_cfg)
        return model

    def extract(self, batch: dict) -> Dict[str, Any]:
        x = batch["wave"]

        if self.device != x.device:
            logger.info("Move Model to the same device")
            self.device = x.device
            self.model.to(self.device)

        z = self.model.extract_features(x)[0]
        z = z.mean(1)  # (B, L, D) -> (B, D)
        return {"embed": z}


class BEATsFrequencyPoolingFrozenModel(BEATsFrozenModel):
    """Frozen BEATs frontend retaining frequency-aligned patch embeddings."""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
        pooling: str = "rdp",
        gamma: float = 4.0,
        eps: float = 1e-8,
        emit_patch_sequence: bool = False,
    ):
        if pooling not in {"mean", "rdp"}:
            raise ValueError(f"pooling must be 'mean' or 'rdp', but got {pooling!r}")
        self.pooling = pooling
        self.gamma = gamma
        self.eps = eps
        self.emit_patch_sequence = emit_patch_sequence
        super().__init__(model_cfg=model_cfg)

    def extract(self, batch: dict) -> Dict[str, Any]:
        x = batch["wave"]

        if self.device != x.device:
            logger.info("Move Model to the same device")
            self.device = x.device
            self.model.to(self.device)

        z_seq, sequence_padding_mask, grid_shape = (
            self.model.extract_features_with_grid(
                x, padding_mask=batch.get("padding_mask")
            )
        )
        z_tf = sequence_to_time_frequency(z_seq, grid_shape)
        valid_time_mask = batch.get("valid_time_mask")
        if sequence_padding_mask is not None:
            sequence_valid_mask = ~sequence_padding_mask.reshape(
                x.shape[0], grid_shape[0], grid_shape[1]
            )
            if valid_time_mask is None:
                valid_time_mask = sequence_valid_mask
            else:
                if valid_time_mask.dtype != sequence_valid_mask.dtype:
                    raise TypeError("valid_time_mask must have dtype torch.bool")
                if valid_time_mask.device != sequence_valid_mask.device:
                    raise ValueError(
                        "valid_time_mask and BEATs output must be on the same device"
                    )
                expected_2d = (x.shape[0], grid_shape[0])
                expected_3d = (x.shape[0], grid_shape[0], grid_shape[1])
                if valid_time_mask.shape == expected_2d:
                    valid_time_mask = valid_time_mask.unsqueeze(-1)
                elif valid_time_mask.shape != expected_3d:
                    raise ValueError(
                        "valid_time_mask must have shape [B, T] or [B, T, F], "
                        f"but got {tuple(valid_time_mask.shape)}"
                    )
                valid_time_mask = valid_time_mask & sequence_valid_mask
        z_freq = frequency_pooling(
            z_tf,
            mode=self.pooling,
            gamma=self.gamma,
            eps=self.eps,
            valid_time_mask=valid_time_mask,
        )
        output = {
            "embed_freq": z_freq,
            "embed": z_freq.flatten(start_dim=1),
        }
        if self.emit_patch_sequence:
            output["patch_sequence"] = z_seq
        return output
