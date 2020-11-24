# matbench_phonons

## Introduction

- Predicting the largest phonon PhDOS peak for 1265 structures from DFPT dataset.

Results to beat:

| Method | PhDOS peak (1/cm) |
|:-------|------------------:|
| AM     |             50.8  |
| MEGNet |             36.9  |
| CGCNN  |             57.8  |
| RF     |             68.0  |

## To-do

- [x] Featurize
- [x] Feature selection (extremely slow, >months...)
- [x] Baseline MODNet model (~45 1/cm)

## Notes

- Very slow to do feature selection; need to do sampling

## Results

| Method/architecture | PhDOS peak (1/cm) |
|:--------------------|------------------:|
| MODNet baseline ([64, 64, 8, 8], 175 features) |   38.7 |
