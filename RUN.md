
# Run initialize script
PYTHONPATH=src python scripts/day1_day2_setup.py \
    --root ./data \
    --output-dir ./outputs/day1_day2 \
    --train-scenes rural urban \
    --val-scenes rural urban \
    --patch-size 512 \
    --num-vis 20 \
    --download
