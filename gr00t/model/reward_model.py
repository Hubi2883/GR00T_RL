import torch
import tree
from dataclasses import dataclass, field
from transformers import PreTrainedModel, PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.gr00t_n1 import GR00T_N1          # backbone class
from gr00t.model.reward_head import RewardHead, RewardHeadConfig

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------
BACKBONE_FEATURE_KEY = "backbone_features"
REWARD_KEY           = "reward_pred"
LOSS_KEY             = "loss"
ERROR_MSG            = "GR00TReward: unexpected input / output shape"

# ---------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------
@dataclass
class GR00TRewardConfig(PretrainedConfig):
    """
    Combines backbone & reward-head hyper-params for saving/loading.
    """
    model_type: str = field(init=False, default="gr00t_reward")
    backbone_cfg: dict = field(default_factory=dict)
    reward_head_cfg: dict = field(default_factory=dict)
    compute_dtype: str = field(default="float32")
    pruned_heads: dict = field(default_factory=dict, metadata={"help": "Heads pruned from the model"})

# ---------------------------------------------------------------------
# model
# ---------------------------------------------------------------------
class GR00TReward(PreTrainedModel):
    """
    Wrapper that swaps the original GR00T action head for our RewardHead.
    """
    config_class = GR00TRewardConfig


    def __init__(
        self,
        backbone_model: GR00T_N1,
        reward_head_cfg: RewardHeadConfig | None = None,
        compute_dtype: str = "float32",
    ):
        # Build a combined HF config that embeds both backbone + head settings
        cfg = GR00TRewardConfig(
            backbone_cfg=backbone_model.config.to_dict(),
            reward_head_cfg=(reward_head_cfg or RewardHeadConfig()).to_dict(),
            compute_dtype=compute_dtype,
        )
        super().__init__(cfg)

        # Share the GR00T backbone
        self.backbone = backbone_model.backbone

        # -----------------------------------------------------------------
        # Whitelist only the true RewardHeadConfig fields
        # -----------------------------------------------------------------
        head_cfg_dict = dict(cfg.reward_head_cfg)
        allowed = {"hidden_dim", "dropout", "reward_dim", "tune_projector"}
        filtered = {k: head_cfg_dict[k] for k in head_cfg_dict.keys() & allowed}

        # Construct your RewardHead with exactly those args
        self.reward_head = RewardHead(RewardHeadConfig(**filtered))

        # Register dtype for casting inputs later
        self.compute_dtype = compute_dtype

        # HuggingFace hook to initialize any uninitialized params
        self.post_init()
    # -----------------------------------------------------------------
    # training / inference
    # -----------------------------------------------------------------
    def forward(self, inputs):
        backbone_in, reward_in = self.prepare_input(inputs)
        backbone_out = self.backbone(backbone_in)
        reward_out   = self.reward_head(backbone_out, reward_in)

        # sanity-check
        assert BACKBONE_FEATURE_KEY in backbone_out, ERROR_MSG
        assert (LOSS_KEY in reward_out) or (REWARD_KEY in reward_out), ERROR_MSG

        if LOSS_KEY in reward_out:
            return {"loss": reward_out[LOSS_KEY]}
        return reward_out

    def get_reward(self, inputs):
        backbone_in, reward_in = self.prepare_input(inputs)
        backbone_out = self.backbone(backbone_in)
        return self.reward_head.get_reward(backbone_out)

    # -----------------------------------------------------------------
    # utils
    # -----------------------------------------------------------------
    def prepare_input(self, inputs):
        """
        Splits dict into backbone-specific and reward-head-specific parts,
        moves to the right device, casts dtype.
        """
        backbone_in = self.backbone.prepare_input(inputs)
        reward_in   = {}
        if "target_reward" in inputs:
            reward_in["target_reward"] = inputs["target_reward"]

        device = next(self.parameters()).device
        to_d = lambda x: x.to(device=device,
                              dtype=getattr(torch, self.compute_dtype)
                              if torch.is_floating_point(x) else None)

        backbone_in = tree.map_structure(to_d, backbone_in)
        reward_in   = tree.map_structure(to_d, reward_in)
        return backbone_in, reward_in

    # expose convenient toggles
    def set_trainable_parameters(self,
                                 tune_visual=False,
                                 tune_llm=False,
                                 tune_projector=True):
        self.backbone.set_trainable_parameters(tune_visual=tune_visual,
                                               tune_llm=tune_llm)
        self.reward_head.set_trainable_parameters(tune_projector)
        print(f"Tune visual={tune_visual}, llm={tune_llm}, head={tune_projector}")

    # shortcut constructor
    @classmethod
    def from_pretrained_backbone(cls,
                                 gr00t_model: GR00T_N1,
                                 reward_head_cfg: RewardHeadConfig | None = None,
                                 **kwargs):
        return cls(gr00t_model, reward_head_cfg, **kwargs)
