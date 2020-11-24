# matbench_elastic

## Introduction

- Multi-target prediction of Voigt-Reuss-Hill bulk modulus and shear modulus from structures.
- 10,987 structures

Results to beat:

| Method | log K (log(GPa)) | log G (log(GPa)) |
|:-------|-----------------:|-----------------:|
| Automatminer | 0.0679 | 0.0849 |
| MEGNet | 0.0712 | 0.0914 |
| CGCNN | 0.0712 | 0.0895 |

## To-do

- [x] Featurize
- [ ] Feature selection (extremely slow, >months...)

## Notes

- Very slow to do feature selection; need to do sampling

## Results

| Method/architecture | log K (log(GPa)) | log G (log(GPa)) |
|:--------------------|-----------------:|-----------------:|
| MODNet              |                  |                  |
