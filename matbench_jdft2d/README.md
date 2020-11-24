# matbench_jdft2d

## Introduction

- 636 2D structures and their exfoliation energies

Results to beat:

| Method | Exfoliation energy (meV) |
|:-------|-------------------------:|
| AM     | 38.6                     |
| RF     | 49.9                     |
| CGCNN  | 49.2                     |
| MEGNet | 55.9                     |
| Dummy  | 67.3                     |

## Notes

- Another long-tailed dataset: dummy predictor does very well
- Trying MSE instead of MAE to capture tails

## To-do

- [x] Featurize
- [x] Feature selection 
- [x] Baseline models
- [ ] Feature importance

## Results

| Method | Exfoliation energy (meV) |
|:-------|-----------------:|
| Rough MODNet baseline | 32.7 |
