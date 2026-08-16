from .unet import UNet

__all__ = ["UNet"]


def build_default_unet() -> UNet:
    """Single source of truth for the architecture hyperparameters,
    so every method (ddpm/ddim/score/flow) instantiates an identical net."""
    return UNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        time_emb_dim=32,
        dropout=0.0,
    )
