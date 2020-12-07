# MODNet matbench benchmarks

## Interim results

<table>
<tr>
  <th>Dataset</th>
  <th>Structure?</th>
  <th>Leading result (MODNet)</th>
  <th>Dataset size</th>
</tr>
<tr>
  <td>steels</td>
  <td>no</td>
  <td style="backgruond-color: LightRed;">95.2 [Automatminer] (120)</td>
  <td>312</td>
</tr>
<tr>
  <td>JDFT2D</td>
  <td> </td>
  <td style="background-color: LightGreen">38.6 [Automatminer] (32.7)</td>
  <td>636</td>
</tr>
<tr>
  <td>PhDOS</td>
  <td> </td>
  <td style="background-color: LightBlue;">36.9 [MEGNet] (38.7)</td>
  <td>1265</td>
</tr>
<tr>
  <td>Expt. band gap</td>
  <td>no</td>
  <td style="background-color: LightGreen;">0.42 [Automatminer] (0.37)</td>
  <td>4604</td>
</tr>
<tr>
  <td><i>n</i></td>
  <td> </td>
  <td style="background-color: LightBlue;">0.30 [Automatminer] (0.31)</td>
  <td>4764</td>
</tr>
<tr>
  <td>log G</td>
  <td> </td>
  <td style="background-color: LightGrey;">0.085 [Automatminer] (-)</td>
  <td>10987</td>
</tr>
<tr>
  <td>log K</td>
  <td> </td>
  <td style="background-color: LightGrey;">0.068 [Automatminer] (-)</td>
  <td>10987</td>
</tr>
</table>


## To discuss

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

