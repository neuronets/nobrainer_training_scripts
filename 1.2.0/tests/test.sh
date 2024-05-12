#!/bin/bash -l
python brainy_train.py tests/configs/brainy_01a.yml
python brainy_train.py tests/configs/brainy_02a.yml
python brainy_train.py tests/configs/brainy_03a.yml

python kwyk_train.py tests/configs/kwyk_01a.yml
python kwyk_train.py tests/configs/kwyk_02a.yml
python kwyk_train.py tests/configs/kwyk_03a.yml

python brainy_train.py tests/configs/brainy_01b.yml
python brainy_train.py tests/configs/brainy_02b.yml
python brainy_train.py tests/configs/brainy_03b.yml

python kwyk_train.py tests/configs/kwyk_01b.yml
python kwyk_train.py tests/configs/kwyk_02b.yml
python kwyk_train.py tests/configs/kwyk_03b.yml
