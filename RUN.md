
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
