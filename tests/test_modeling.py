from __future__ import annotations

import pytest

from loveda_project.data import CLASS_NAMES, IGNORE_INDEX
from loveda_project.modeling import (
    HF_PRETRAINED_NAMES,
    SEGFORMER_VARIANTS,
    SegformerBuildConfig,
    build_segformer_model,
)
from scripts.eval_segformer import parse_args as parse_eval_args
from scripts.train_segformer import parse_args as parse_train_args


EXPECTED_VARIANT_CONFIGS = {
    "segformer-b0": {"depths": [2, 2, 2, 2], "decoder_hidden_size": 256},
    "segformer-b1": {"depths": [2, 2, 2, 2], "decoder_hidden_size": 256},
    "segformer-b2": {"depths": [3, 4, 6, 3], "decoder_hidden_size": 768},
    "segformer-b3": {"depths": [3, 4, 18, 3], "decoder_hidden_size": 768},
    "segformer-b4": {"depths": [3, 8, 27, 3], "decoder_hidden_size": 768},
    "segformer-b5": {"depths": [3, 6, 40, 3], "decoder_hidden_size": 768},
}


@pytest.mark.parametrize("variant", sorted(EXPECTED_VARIANT_CONFIGS))
def test_segformer_variants_build_eight_class_heads_without_pretrained_weights(variant: str) -> None:
    model = build_segformer_model(
        SegformerBuildConfig(
            variant=variant,
            num_labels=len(CLASS_NAMES),
            ignore_index=IGNORE_INDEX,
            pretrained=False,
        )
    )

    assert model.config.num_labels == len(CLASS_NAMES)
    assert model.config.semantic_loss_ignore_index == IGNORE_INDEX
    assert model.decode_head.classifier.out_channels == len(CLASS_NAMES)
    assert model.config.depths == EXPECTED_VARIANT_CONFIGS[variant]["depths"]
    assert model.config.decoder_hidden_size == EXPECTED_VARIANT_CONFIGS[variant]["decoder_hidden_size"]


def test_segformer_variant_registry_and_hf_names_include_b0_through_b5() -> None:
    expected_names = {
        "segformer-b0": "nvidia/segformer-b0-finetuned-ade-512-512",
        "segformer-b1": "nvidia/segformer-b1-finetuned-ade-512-512",
        "segformer-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
        "segformer-b3": "nvidia/segformer-b3-finetuned-ade-512-512",
        "segformer-b4": "nvidia/segformer-b4-finetuned-ade-512-512",
        "segformer-b5": "nvidia/segformer-b5-finetuned-ade-512-512",
    }

    assert set(SEGFORMER_VARIANTS) == set(expected_names)
    assert HF_PRETRAINED_NAMES == expected_names


@pytest.mark.parametrize("variant", ["segformer-b3", "segformer-b4", "segformer-b5"])
def test_train_and_eval_cli_accept_larger_segformer_variants(variant: str) -> None:
    train_args = parse_train_args(["--variant", variant])
    eval_args = parse_eval_args(["--checkpoint", "dummy.pth", "--variant", variant])

    assert train_args.variant == variant
    assert eval_args.variant == variant


def test_segformer_model_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="Unsupported variant"):
        build_segformer_model(
            SegformerBuildConfig(
                variant="segformer-b9",
                num_labels=len(CLASS_NAMES),
                ignore_index=IGNORE_INDEX,
                pretrained=False,
            )
        )
