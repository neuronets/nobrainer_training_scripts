#!/bin/bash -l

# test: brainy train and resume (binary)
rm -rf output/test_brainy_01 && python scripts/train/brainy_train.py tests/configs/brainy_01a.yml
python scripts/train/brainy_train.py tests/configs/brainy_01b.yml

# test: brainy train and resume (multi-class)
rm -rf output/test_brainy_02 && python scripts/train/brainy_train.py tests/configs/brainy_02a.yml
python scripts/train/brainy_train.py tests/configs/brainy_02b.yml

# test: kwyk train and resume (binary)
rm -rf output/test_kwyk_01 && python scripts/train/kwyk_train.py tests/configs/kwyk_01a.yml
python scripts/train/kwyk_train.py tests/configs/kwyk_01b.yml

# test: brainy train and resume (multi-class)
rm -rf output/test_kwyk_02 && python scripts/train/kwyk_train.py tests/configs/kwyk_02a.yml
python scripts/train/kwyk_train.py tests/configs/kwyk_02b.yml

# test: brainy train and resume (decorators)
rm -rf output/test_brainy_01 && python scripts/train/decorators.py brainy tests/configs/brainy_01a.yml
python scripts/train/decorators.py brainy tests/configs/brainy_01b.yml

# test: kwyk train and resume (decorators)
rm -rf output/test_kwyk_01 && python scripts/train/decorators.py kwyk tests/configs/kwyk_01a.yml
python scripts/train/decorators.py kwyk tests/configs/kwyk_01b.yml
