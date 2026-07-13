# Face-shape classifier metrics

Trained on 4072 images, evaluated on a held-out test set of 1019 images. Identical stratified split (seed 42) for both models.

## Cross-validation on the training set (5-fold accuracy)

| model | mean | std |
|---|---|---|
| LogisticRegression | 0.464 | 0.018 |
| RandomForest | 0.555 | 0.017 |

Primary model selected by cross-validation: **RandomForest**.

## Head-to-head on the held-out test set

| model | role | test accuracy |
|---|---|---|
| scikit-learn (RandomForest) | primary | 0.526 |
| PyTorch MLP | benchmark | 0.471 |

The scikit-learn model serves in production. On a low-dimensional structured feature space of nine geometric ratios, a classical model is the right tool, and the neural net does not beat it. The MLP is kept as a benchmark, not a serving path. Confidence comes from the forest's own predict_proba. Post-hoc calibration was tried and rejected because it collapsed the tiny Diamond class to zero predictions.

### scikit-learn (RandomForest), per class

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Diamond | 0.29 | 0.13 | 0.18 | 15 |
| Heart | 0.50 | 0.47 | 0.48 | 202 |
| Oblong | 0.60 | 0.65 | 0.62 | 200 |
| Oval | 0.42 | 0.43 | 0.43 | 200 |
| Round | 0.53 | 0.50 | 0.51 | 201 |
| Square | 0.58 | 0.61 | 0.60 | 201 |

Confusion matrix (rows are true labels):

| true \ pred | Diamond | Heart | Oblong | Oval | Round | Square |
|---|---|---|---|---|---|---|
| Diamond | 2 | 2 | 6 | 1 | 3 | 1 |
| Heart | 0 | 94 | 36 | 44 | 11 | 17 |
| Oblong | 2 | 29 | 130 | 27 | 1 | 11 |
| Oval | 1 | 43 | 29 | 87 | 29 | 11 |
| Round | 2 | 14 | 4 | 33 | 101 | 47 |
| Square | 0 | 7 | 11 | 14 | 47 | 122 |

### PyTorch MLP, per class

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Diamond | 0.05 | 0.40 | 0.09 | 15 |
| Heart | 0.47 | 0.35 | 0.40 | 202 |
| Oblong | 0.64 | 0.61 | 0.63 | 200 |
| Oval | 0.41 | 0.34 | 0.37 | 200 |
| Round | 0.51 | 0.49 | 0.50 | 201 |
| Square | 0.55 | 0.57 | 0.56 | 201 |

Confusion matrix (rows are true labels):

| true \ pred | Diamond | Heart | Oblong | Oval | Round | Square |
|---|---|---|---|---|---|---|
| Diamond | 6 | 3 | 5 | 0 | 1 | 0 |
| Heart | 35 | 71 | 33 | 36 | 6 | 21 |
| Oblong | 22 | 21 | 123 | 20 | 3 | 11 |
| Oval | 27 | 38 | 21 | 67 | 31 | 16 |
| Round | 14 | 13 | 3 | 26 | 98 | 47 |
| Square | 8 | 4 | 7 | 13 | 54 | 115 |

## Notes

- Diamond is the smallest class in the source data, so its per-class scores are the weakest and it is most often confused with its nearest neighbors. This reflects the data, not a bug.
- Features are sex-neutral geometric ratios, so the female-skewed labels transfer across sexes better than a raw-pixel model would, with a mild distributional skew.
- All ratios come from a single 2D photo, so out-of-plane head pose is an inherent ceiling. Front-facing photos give the best results.
