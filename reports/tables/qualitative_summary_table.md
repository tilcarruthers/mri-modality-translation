| run                   | normalization      |         mse |      rmse |     ssim |
|:----------------------|:-------------------|------------:|----------:|---------:|
| U-Net + global        | global min-max     | 0.000372128 | 0.0192906 | 0.85592  |
| ResUNet + global      | global min-max     | 0.00057993  | 0.0240817 | 0.902884 |
| Baseline + global     | global min-max     | 0.000666983 | 0.025826  | 0.928267 |
| U-Net + percentile    | percentile min-max | 0.0015798   | 0.0397467 | 0.363207 |
| Baseline + percentile | percentile min-max | 0.00174357  | 0.0417561 | 0.801112 |
