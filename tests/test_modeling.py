from __future__ import annotations

import pytest

from loveda_project.data import CLASS_NAMES, IGNORE_INDEX
from loveda_project.modeling import SegformerBuildConfig, build_segformer_model


@pytest.mark.parametrize("variant", ["segformer-b0", "segformer-b1", "segformer-b2"])
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
