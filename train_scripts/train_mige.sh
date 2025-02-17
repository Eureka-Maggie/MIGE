PYTHON=/usr/bin/python3 accelerate launch \
    --config_file=accelerate_config.yaml \
    train_scripts/train_mige.py \
    --config configs/config_mige.py \
    --work-dir output/train_mige \
    --report_to tensorboard