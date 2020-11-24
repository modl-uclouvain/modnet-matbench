# matbench_steels

## Introduction

- Small dataset of only 312 steel compositions with associated yield strengths.

Results to beat:

| Method | Yield strength (MPa) |
|:-------|-----------------:|
| AM     | 95.2 |
| RF     | 104  |
| Dummy  | 230  |

## Notes

- Extremely variable depending on validation set; probably high degree of bias in underlying data.
- Extremely well-behaved learning curves, consistently ~50 MPa on test set but double that on validation set.
- Errors are very long-tailed
- Should really consider doing feature importance with left-out data

## To-do

- [x] Featurize
- [x] Feature selection 
- [x] Baseline models
- [ ] Feature importance

## Results

| Method | Yield strength (MPa) |
|:-------|---------------------:|
| Rough MODNet baseline   | 107.8 |
