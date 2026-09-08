#!/bin/bash

for experiment in 4 5 6; do
  for tau in expert l1_to_l2 speedup; do
    triton/.venv/bin/python triton/experiments/analyze-search.py \
      --experiment "$experiment" --platform tuolumne --tau-name $tau

    triton/.venv-matrix/bin/python triton/experiments/analyze-search.py \
      --experiment "$experiment" --platform matrix --tau-name $tau
  done
done