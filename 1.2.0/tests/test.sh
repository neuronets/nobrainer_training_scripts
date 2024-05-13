#!/bin/bash -l

# test training from scratch (binary and multi-class)
python scripts/train/brainy_train.py tests/configs/brainy_01a.yml
python scripts/train/brainy_train.py tests/configs/brainy_02a.yml

python scripts/train/kwyk_train.py tests/configs/kwyk_01a.yml
python scripts/train/kwyk_train.py tests/configs/kwyk_02a.yml

# test resumption (from checkpoint using decorators)
python scripts/train/brainy_train.py tests/configs/brainy_01b.yml
python scripts/train/kwyk_train.py tests/configs/kwyk_01b.yml

# test decorators
python scripts/train/decorators.py tests/configs/brainy_01a.yml
python scripts/train/decorators.py tests/configs/kwyk_01b.yml