
# Run initialize script
PYTHONPATH=src python scripts/day1_day2_setup.py \
    --root ./data \
    --output-dir ./outputs/day1_day2 \
    --train-scenes rural urban \
    --val-scenes rural urban \
    --patch-size 512 \
    --num-vis 20 \
    --download

# segformer-b0 + pretrained + CE only + constant lr=6e-5 + 25 epochs -> lean to predict background
PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/day3_day5 \
  --variant segformer-b0 \
  --pretrained \
  --epochs 25 \
  --batch-size 4 \
  --lr 6e-5 \
  --weight-decay 1e-2 \
  --loss-name ce \
  --dice-weight 0.25 \
  --amp \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b0_pt_ce_patch512

# segformer-b0 + pretrained + CE_DICE + constant lr=6e-5 + 25 epochs -> lean to predict background
PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/day3_day5 \
  --variant segformer-b0 \
  --pretrained \
  --epochs 25 \
  --batch-size 4 \
  --lr 6e-5 \
  --weight-decay 1e-2 \
  --loss-name ce_dice \
  --dice-weight 0.50 \
  --amp \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b0_pt_cedice_patch512

PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/day3_day5 \
  --variant segformer-b0 \
  --pretrained \
  --epochs 25 \
  --batch-size 4 \
  --lr 6e-5 \
  --weight-decay 1e-2 \
  --loss-name ce_dice \
  --dice-weight 0.25 \
  --amp \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b0_pt_cedice_w0p25_patch512

PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/day3_day5_warmcos \
  --variant segformer-b0 \
  --pretrained \
  --epochs 25 \
  --batch-size 4 \
  --lr 6e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 2 \
  --min-lr 1e-6 \
  --weight-decay 1e-2 \
  --loss-name ce_dice \
  --dice-weight 0.25 \
  --amp \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b0_pt_cedice_w0p25_warmcos_patch512

PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/day3_day5_warmcos \
  --variant segformer-b0 \
  --pretrained \
  --epochs 35 \
  --batch-size 4 \
  --lr 8e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --weight-decay 1e-2 \
  --loss-name ce_dice \
  --dice-weight 0.25 \
  --amp \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b0_pt_cedice_w0p25_warmcos4_lr8en5_patch512

PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/day3_day5_warmcos \
  --variant segformer-b0 \
  --pretrained \
  --epochs 35 \
  --batch-size 4 \
  --lr 8e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --weight-decay 1e-2 \
  --loss-name ce_dice \
  --dice-weight 0.25 \
  --amp \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b0_pt_cedice_w0p25_warmcos4_lr8en5_patch512_test

PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/seg_release/pr03_b1_cedice_median_warmcos \
  --variant segformer-b1 \
  --pretrained \
  --epochs 45 \
  --batch-size 2 \
  --lr 6e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --weight-decay 1e-2 \
  --loss-name ce_dice \
  --dice-weight 0.25 \
  --class-weight-mode median \
  --class-stats outputs/class_stats/urban_rural.json \
  --amp \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b1_pt_cedice_effective_warmcos45_patch512

PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/seg_release/pr03_b1_cedice_inverse_warmcos \
  --variant segformer-b1 \
  --pretrained \
  --epochs 45 \
  --batch-size 2 \
  --lr 6e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --weight-decay 1e-2 \
  --loss-name ce_dice \
  --dice-weight 0.25 \
  --class-weight-mode inverse \
  --class-stats outputs/class_stats/urban_rural.json \
  --amp \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b1_pt_cedice_inverse_warmcos45_patch512  

PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/seg_release/pr04_b1_cedice_median_strong_aug \
  --variant segformer-b1 \
  --pretrained \
  --epochs 45 \
  --batch-size 2 \
  --lr 6e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --weight-decay 1e-2 \
  --loss-name ce_dice \
  --dice-weight 0.25 \
  --class-weight-mode median \
  --class-stats outputs/class_stats/urban_rural.json \
  --aug-preset strong \
  --amp \
  --save-every 0 \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b1_pt_cedice_median_strongaug_warmcos45_patch512

PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/seg_release/pr04_b1_cedice_inverse_strong_aug \
  --variant segformer-b1 \
  --pretrained \
  --epochs 45 \
  --batch-size 2 \
  --lr 6e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --weight-decay 1e-2 \
  --loss-name ce_dice \
  --dice-weight 0.25 \
  --class-weight-mode inverse \
  --class-stats outputs/class_stats/urban_rural.json \
  --aug-preset strong \
  --amp \
  --save-every 0 \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b1_pt_cedice_inverse_strongaug_warmcos45_patch512

PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/seg_release/pr05_b2_cedice_median_basic_aug \
  --variant segformer-b2 \
  --pretrained \
  --epochs 45 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --lr 6e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --weight-decay 1e-2 \
  --loss-name ce_dice \
  --dice-weight 0.25 \
  --class-weight-mode median \
  --class-stats outputs/class_stats/urban_rural.json \
  --aug-preset basic \
  --amp \
  --save-every 0 \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b2_pt_cedice_median_basic_warmcos45_patch512

PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/seg_release/pr05_b2_cedice_inverse_basic_aug \
  --variant segformer-b2 \
  --pretrained \
  --epochs 45 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --lr 6e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --weight-decay 1e-2 \
  --loss-name ce_dice \
  --dice-weight 0.25 \
  --class-weight-mode inverse \
  --class-stats outputs/class_stats/urban_rural.json \
  --aug-preset basic \
  --amp \
  --save-every 0 \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b2_pt_cedice_inverse_basic_warmcos45_patch512

PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/seg_release/pr05_b2_focal_median_basic_aug \
  --variant segformer-b2 \
  --pretrained \
  --epochs 45 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --lr 6e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --weight-decay 1e-2 \
  --loss-name focal \
  --dice-weight 0.25 \
  --class-weight-mode median \
  --class-stats outputs/class_stats/urban_rural.json \
  --aug-preset basic \
  --amp \
  --save-every 0 \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b2_pt_focal_median_basic_warmcos45_patch512

PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/seg_release/pr05_b2_focal_inverse_basic_aug \
  --variant segformer-b2 \
  --pretrained \
  --epochs 45 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --lr 6e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --weight-decay 1e-2 \
  --loss-name focal \
  --dice-weight 0.25 \
  --class-weight-mode inverse \
  --class-stats outputs/class_stats/urban_rural.json \
  --aug-preset basic \
  --amp \
  --save-every 0 \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b2_pt_focal_inverse_basic_warmcos45_patch512

PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/seg_release/pr05_b2_cedice_median_strong_aug \
  --variant segformer-b2 \
  --pretrained \
  --epochs 45 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --lr 6e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --weight-decay 1e-2 \
  --loss-name ce_dice \
  --dice-weight 0.25 \
  --class-weight-mode median \
  --class-stats outputs/class_stats/urban_rural.json \
  --aug-preset strong \
  --amp \
  --save-every 0 \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b2_pt_cedice_median_strong_warmcos45_patch512

PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/seg_release/ablate_b2_cedice_inverse_patch1024 \
  --variant segformer-b2 \
  --pretrained \
  --epochs 45 \
  --batch-size 2 \
  --grad-accum-steps 4 \
  --patch-size 1024 \
  --lr 3e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --weight-decay 1e-2 \
  --loss-name ce_dice \
  --dice-weight 0.25 \
  --class-weight-mode inverse \
  --class-stats outputs/class_stats/urban_rural.json \
  --aug-preset basic \
  --amp \
  --save-every 0 \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b2_pt_cedice_inverse_basic_patch1024_warmcos45

PYTHONPATH=src python scripts/train_segformer.py \
  --root ./data \
  --output-dir ./outputs/seg_release/ablate_b2_cedice_inverse_strong_patch1024 \
  --variant segformer-b2 \
  --pretrained \
  --epochs 45 \
  --batch-size 2 \
  --grad-accum-steps 4 \
  --patch-size 1024 \
  --lr 3e-5 \
  --scheduler-type warmup+cosine \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --weight-decay 1e-2 \
  --loss-name ce_dice \
  --dice-weight 0.25 \
  --class-weight-mode inverse \
  --class-stats outputs/class_stats/urban_rural.json \
  --aug-preset strong \
  --amp \
  --save-every 0 \
  --use-wandb \
  --wandb-project loveda-segformer \
  --wandb-run-name b2_pt_cedice_inverse_strong_patch1024_warmcos45


# Calculate class stats
PYTHONPATH=src python scripts/compute_class_stats.py \
  --root ./data \
  --train-scenes urban rural \
  --output outputs/class_stats/urban_rural.json

# Only get the first 4 samples from ./data/Val concatenated ['urban', 'rural'] dataset.
PYTHONPATH=src python scripts/eval_segformer.py \
  --checkpoint outputs/day3_day5_warmcos/checkpoints/best_model.pth \
  --root ./data \
  --output-dir outputs/eval_smoke \
  --max-samples 4

# Eval on the whole ./data/Val concatedated ['urban', 'rural'] dataset
PYTHONPATH=src python scripts/eval_segformer.py \
  --checkpoint outputs/day3_day5_warmcos/checkpoints/best_model.pth \
  --root ./data \
  --output-dir outputs/eval_day3_day5_warmcos

PYTHONPATH=src python scripts/eval_segformer.py \
  --checkpoint outputs/seg_release/pr03_b1_cedice_median_warmcos/checkpoints/best_model.pth \
  --root ./data \
  --output-dir outputs/eval_pr03_b1_cedice_median_warmcos

PYTHONPATH=src python scripts/eval_segformer.py \
  --checkpoint outputs/seg_release/pr05_b2_cedice_inverse_basic_aug/checkpoints/best_model.pth \
  --root ./data \
  --output-dir outputs/seg_release/pr06_b2_inverse_sliding_ms_eval \
  --variant segformer-b2 \
  --patch-size 1024 \
  --batch-size 1 \
  --tta sliding \
  --window-size 512 \
  --stride 256 \
  --scales 0.75 1.0 1.25

PYTHONPATH=src python scripts/eval_segformer.py \
  --checkpoint outputs/seg_release/pr05_b2_cedice_median_basic_aug/checkpoints/best_model.pth \
  --root ./data \
  --output-dir outputs/seg_release/pr06_b2_median_sliding_ms_eval \
  --variant segformer-b2 \
  --patch-size 1024 \
  --batch-size 1 \
  --tta sliding \
  --window-size 512 \
  --stride 256 \
  --scales 0.75 1.0 1.25

PYTHONPATH=src python scripts/eval_segformer.py \
  --checkpoint outputs/seg_release/pr05_b2_focal_median_basic_aug/checkpoints/best_model.pth \
  --root ./data \
  --output-dir outputs/seg_release/pr06_b2_focal_median_sliding_ms_eval \
  --variant segformer-b2 \
  --patch-size 1024 \
  --batch-size 1 \
  --tta sliding \
  --window-size 512 \
  --stride 256 \
  --scales 0.75 1.0 1.25

PYTHONPATH=src python scripts/eval_segformer.py \
  --checkpoint outputs/seg_release/pr04_b1_cedice_inverse_strong_aug/checkpoints/best_model.pth \
  --root ./data \
  --output-dir outputs/seg_release/pr06_b1_inverse_strong_sliding_ms_eval \
  --variant segformer-b1 \
  --patch-size 1024 \
  --batch-size 1 \
  --tta sliding \
  --window-size 512 \
  --stride 256 \
  --scales 0.75 1.0 1.25

PYTHONPATH=src python scripts/eval_segformer.py \
  --checkpoint outputs/seg_release/pr05_b2_cedice_inverse_basic_aug/checkpoints/best_model.pth \
  --root ./data \
  --output-dir outputs/seg_release/pr06_b2_inverse_sliding_s1_ms_eval \
  --variant segformer-b2 \
  --patch-size 1024 \
  --batch-size 1 \
  --tta sliding \
  --window-size 512 \
  --stride 256 \
  --scales 1.0

PYTHONPATH=src python scripts/eval_segformer.py \
  --checkpoint outputs/seg_release/pr05_b2_cedice_inverse_basic_aug/checkpoints/best_model.pth \
  --root ./data \
  --output-dir outputs/seg_release/pr06_b2_inverse_sliding_ms_eval \
  --variant segformer-b2 \
  --patch-size 1024 \
  --batch-size 1 \
  --tta sliding \
  --window-size 512 \
  --stride 128 \
  --scales 0.75 1.0 1.25

PYTHONPATH=src python scripts/eval_segformer.py \
  --checkpoint outputs/seg_release/ablate_b2_cedice_inverse_patch1024/checkpoints/best_model.pth \
  --root ./data \
  --output-dir outputs/seg_release/eval_b2_inverse_basic_patch1024_singlepass \
  --variant segformer-b2 \
  --patch-size 1024 \
  --batch-size 1 \
  --tta none

PYTHONPATH=src python scripts/eval_segformer.py \
  --checkpoint outputs/seg_release/ablate_b2_cedice_inverse_patch1024/checkpoints/best_model.pth \
  --root ./data \
  --output-dir outputs/seg_release/pr06_b2_inverse_basic_patch1024_tta_w1024_s1_eval \
  --variant segformer-b2 \
  --patch-size 1024 \
  --batch-size 1 \
  --tta sliding \
  --window-size 1024 \
  --stride 512 \
  --scales 1.0

PYTHONPATH=src python scripts/eval_segformer.py \
  --checkpoint outputs/seg_release/ablate_b2_cedice_inverse_patch1024/checkpoints/best_model.pth \
  --root ./data \
  --output-dir outputs/seg_release/pr06_b2_inverse_basic_patch1024_tta_w1024_ms_eval \
  --variant segformer-b2 \
  --patch-size 1024 \
  --batch-size 1 \
  --tta sliding \
  --window-size 1024 \
  --stride 512 \
  --scales 0.75 1.0 1.25