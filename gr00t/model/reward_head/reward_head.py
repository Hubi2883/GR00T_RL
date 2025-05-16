import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------
BACKBONE_FEATURE_KEY = "backbone_features"
REWARD_KEY           = "reward_pred"
LOSS_KEY             = "loss"

# ---------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------
@dataclass
class RewardHeadConfig(PretrainedConfig):
    """
    Hyper-parameters for the reward head.
    `hidden_dim` controls width; increase or decrease as you like.
    """
    model_type : str  = field(init=False, default="reward_head")
    hidden_dim : int  = field(default=2048,
                              metadata={"help": "Width of the first dense layer"})
    dropout    : float = field(default=0.1,
                               metadata={"help": "Dropout probability"})
    reward_dim : int  = field(default=1,
                              metadata={"help": "Output reward dimension"})
    tune_projector: bool = field(default=True,
                                 metadata={"help": "Whether to update head params"})

# ---------------------------------------------------------------------
# implementation
# ---------------------------------------------------------------------
class RewardHead(nn.Module):
    """
    Deeper + wider feed-forward projection from frozen GR00T backbone
    features to a scalar reward.
    """
    config_class = RewardHeadConfig

    def __init__(self, config: RewardHeadConfig):
        super().__init__()
        self.config = config
        backbone_dim = 1536                 # GR00T-N1 CLS/1st-token width

        d = config.hidden_dim
        self.reward_projector = nn.Sequential(
            nn.Linear(backbone_dim, d),
            nn.GELU(),
            nn.Dropout(config.dropout),

            # extra depth
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(config.dropout),

            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),

            nn.Linear(d // 2, config.reward_dim)  # -> (B, 1)
        )

        self.set_trainable_parameters(config.tune_projector)

    # -----------------------------------------------------------------
    # public helpers
    # -----------------------------------------------------------------
    def set_trainable_parameters(self, tune_projector: bool = True):
        """
        Freeze or un-freeze this head.
        """
        for p in self.parameters():
            p.requires_grad = tune_projector
        print(f"Tune reward head projector: {tune_projector}")

    # -----------------------------------------------------------------
    # forward / inference
    # -----------------------------------------------------------------
    def forward(self, backbone_outputs: BatchFeature, inputs=None) -> BatchFeature:
        """
        During training `inputs` carries `"target_reward"`.
        """
        x = backbone_outputs[BACKBONE_FEATURE_KEY]
        if x.dim() == 3:                     # (B, T, C)  ->  (B, C)
            x = x[:, 0]

        reward_pred = self.reward_projector(x)          # (B,1)
        out = {REWARD_KEY: reward_pred}

        if inputs is not None and "target_reward" in inputs:
            target = inputs["target_reward"].view_as(reward_pred)
            se = F.mse_loss(reward_pred, target, reduction="none")
            pt = torch.exp(-se)
            focal_weight = (1 - pt)**0.5
            loss = (focal_weight * se).mean()
            out[LOSS_KEY] = loss

        return BatchFeature(out)

    @torch.no_grad()
    def get_reward(self, backbone_outputs: BatchFeature) -> BatchFeature:
        """
        Inference helper – returns only reward_pred.
        """
        x = backbone_outputs[BACKBONE_FEATURE_KEY]
        if x.dim() == 3:
            x = x[:, 0]
        return BatchFeature({REWARD_KEY: self.reward_projector(x)})
