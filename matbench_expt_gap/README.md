# matbench_expt_gap

## Introduction

- 4604 compositions and experimental band gaps

Results to beat:

| Method | Band gap (eV)
|:-------|-------|
| AM     | 0.416 | 
| RF     | 0.446 |
| Dummy  | 1.14  |

## Notes

- Extremely variable depending on validation set; probably high degree of bias in underlying data.
- Extremely well-behaved learning curves, consistently ~50 MPa on test set but double that on validation set.
- Errors are very long-tailed
- Should really consider doing feature importance with left-out data

## To-do

- [x] Featurize
- [x] Feature selection 
- [x] Baseline models
- [ ] Train best model
- [ ] Feature importance

## Results

| Method | Band gap (eV)
|:-------|-----------------:|
| Rough MODNet baseline     | 0.372
