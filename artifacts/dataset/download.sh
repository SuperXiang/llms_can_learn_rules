# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

!/usr/bin/env bash

# CLUTRR
wget "https://drive.google.com/u/2/uc?id=1SEq_e1IVCDDzsBIBhoUQ5pOVH5kxRoZF&export=download" -O clutrr.zip
unzip -po clutrr.zip data_emnlp_final/data_089907f8.zip > data_089907f8.zip
unzip -o data_089907f8.zip -d clutrr
rm clutrr.zip data_089907f8.zip

# Arithmetic
URL=https://raw.githubusercontent.com/ZhaofengWu/counterfactual-evaluation/master/arithmetic/data
for BASE in 9 10 11 16
do
  mkdir -p arithmetic/base-${BASE}
  wget ${URL}/0shot/base${BASE}.txt -O - | head -n 900 > arithmetic/base-${BASE}/2_train.txt
  wget ${URL}/0shot/base${BASE}.txt -O - | tail -n 100 > arithmetic/base-${BASE}/2_test.txt
  wget ${URL}/0shot_3digits/base${BASE}.txt -O - | tail -n 100 > arithmetic/base-${BASE}/3_test.txt
  wget ${URL}/0shot_4digits/base${BASE}.txt -O - | tail -n 100 > arithmetic/base-${BASE}/4_test.txt
done

# List Functions
URL=https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/list_functions/
mkdir list_functions
for i in {1..250}
do
  id=$(printf "%03d" ${i})
  wget ${URL}/c${id}/task.json -O list_functions/c${id}.json
done