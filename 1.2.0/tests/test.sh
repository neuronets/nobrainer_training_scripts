#!/bin/bash -l
python brainy_train.py tests/configs/brainy_01.yml
python brainy_train.py tests/configs/brainy_02.yml
python brainy_train.py tests/configs/brainy_03.yml

python kwyk_train.py tests/configs/kwyk_01.yml
python kwyk_train.py tests/configs/kwyk_02.yml
python kwyk_train.py tests/configs/kwyk_03.yml
