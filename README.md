34.3# MODNet matbench benchmarks

## Results

| Dataset         | Structure? |    Best result (MODNet)          | Dataset size | Leader? |
|:----------------|:----------:|---------------------------------:|-------------:|:-------:|
| steels          |     No     | 95.2 [Automatminer] (120->101)   |      312     |   No    |
| JDFT2D          |            | 38.6 meV/atom [Automatminer] (32.7)       |      636     |   **Yes**   |
| PhDOS           |            | 36.9 [MEGNet] (41.7->34.3)       |      1265    |   **Yes**   |
| Band gap        |     No     | 0.42 eV [Automatminer] (0.37->0.33) |      4604    |   **Yes**   |
| n               |            | 0.30 [Automatminer] (0.31)       |      4764    |   Maybe |
| log G           |            | 0.0849 log(GPa) [Automatminer] (0.0846)   |      10987   |   **Yes**   |
| log K           |            | 0.0679 log(GPa) [Automatminer] (0.0633)   |      10987   |   **Yes**   |
| expt is metal   |     No     | 0.92 [Automatminer] (0.96)       |      4921    |   **Yes**   |
| glass           |     No     | 0.861 [Automatminer] (0.925)     |      5680    |   **Yes**   |
|           |          |       |        |      |
| mp_e_form       |          | 0.0327 eV/atom [MEGNet] (-)     |      -    |   probably no  |
| mp_gap       |          | 0.228 eV [CGCNN] (-)     |      -    |   probably no  |
| mp_is_metal       |          | 0.977 [MEGNet] (-)     |      -    |   probably no  |
| perovskites       |          | 0.0417 [MEGNet] (-)     |      -    |   probably no  |
