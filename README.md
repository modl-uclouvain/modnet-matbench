# MODNet matbench benchmarks

## Interim results

| Dataset   | Structure? |    Best result (MODNet)      | Dataset size |
|:----------|:----------:|-----------------------------:|-------------:|
| steels    |      x     | 95.2 [Automatminer] (120)    |      312     |
| JDFT2D    |            | 38.6 [Automatminer] (32.7)   |      636     |
| PhDOS     |            | 36.9 [MEGNet] (41.7)         |     1265     |
| Band gap  |      x     | 0.42 [Automatminer] (0.37)   |     4604     |
| *n*       |            | 0.30 [Automatminer] (-)      |     4764     |
| log G     |            | 0.085 [Automatminer] (-)     |   10987      |
| log K     |            | 0.085 [Automatminer] (-)     |   10987      |

## To-discuss

- Feature selection inside NCV, 
    - Will be expensive, so could also just subsample
    - Some datasets hang when doing NMI; need to split it up
- Investigate feature importance vs selected features
    - Second feature often poor
    - More important for composition-only datasets
    - Could also include new matminer sets 
    - Should benchmark at least one other feature selection method
- Training is *very* variable
    - General rules for overfitting: often validation set error > 3 * test set error
    - Relatively large spread between folds
    - Need to investigate dropout/regularisation
    - Learning rate schemes: currently halving but could do high -> short tuning
- Output scaling

