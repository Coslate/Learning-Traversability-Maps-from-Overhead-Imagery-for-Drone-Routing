from __future__ import annotations

from dataclasses import dataclass

from transformers import SegformerConfig, SegformerForSemanticSegmentation


SEGFORMER_VARIANTS = {
    "segformer-b0": {
        "depths": [2, 2, 2, 2],
        "hidden_sizes": [32, 64, 160, 256],
        "decoder_hidden_size": 256,
    },
    "segformer-b1": {
        "depths": [2, 2, 2, 2],
        "hidden_sizes": [64, 128, 320, 512],
        "decoder_hidden_size": 256,
    },
    "segformer-b2": {
        "depths": [3, 4, 6, 3],
        "hidden_sizes": [64, 128, 320, 512],
        "decoder_hidden_size": 768,
    },
}

HF_PRETRAINED_NAMES = {
    "segformer-b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer-b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "segformer-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
}


@dataclass
class SegformerBuildConfig:
    variant: str = "segformer-b0"
    num_labels: int = 8
    ignore_index: int = 0
    pretrained: bool = False


def _make_config(cfg: SegformerBuildConfig) -> SegformerConfig:
    if cfg.variant not in SEGFORMER_VARIANTS:
        raise ValueError(f"Unsupported variant: {cfg.variant}")

    variant = SEGFORMER_VARIANTS[cfg.variant]
    id2label = {i: str(i) for i in range(cfg.num_labels)}
    label2id = {v: k for k, v in id2label.items()}

    return SegformerConfig(
        num_labels=cfg.num_labels,
        depths=variant["depths"],
        hidden_sizes=variant["hidden_sizes"],
        decoder_hidden_size=variant["decoder_hidden_size"],
        num_attention_heads=[1, 2, 5, 8],
        sr_ratios=[8, 4, 2, 1],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        mlp_ratios=[4, 4, 4, 4],
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        reshape_last_stage=True,
        semantic_loss_ignore_index=cfg.ignore_index,
        id2label=id2label,
        label2id=label2id,
    )


def build_segformer_model(cfg: SegformerBuildConfig) -> SegformerForSemanticSegmentation:
    """Build a SegFormer segmentation model.

    When pretrained=False, the model is initialized from scratch and does not require
    network access. When pretrained=True, Hugging Face will try to load ImageNet/ADE
    pretrained weights and remap the decoder head to the requested num_labels.
    """

    config = _make_config(cfg)
    if not cfg.pretrained:
        return SegformerForSemanticSegmentation(config)

    hf_name = HF_PRETRAINED_NAMES[cfg.variant]
    return SegformerForSemanticSegmentation.from_pretrained(
        hf_name,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
        id2label=config.id2label,
        label2id=config.label2id,
    )
