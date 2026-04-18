| run_name                                   | label                    | normalization      |         mse |        mae |      rmse |    psnr |     ssim |
|:-------------------------------------------|:-------------------------|:-------------------|------------:|-----------:|----------:|--------:|---------:|
| baseline_encoder_decoder_global_minmax     | Baseline encoder-decoder | global min-max     | 0.000666983 | 0.00551545 | 0.025826  | 45.3978 | 0.928267 |
| unet_global_minmax                         | U-Net                    | global min-max     | 0.000372128 | 0.00724286 | 0.0192906 | 43.164  | 0.85592  |
| baseline_encoder_decoder_percentile_minmax | Baseline encoder-decoder | percentile min-max | 0.00174357  | 0.0125319  | 0.0417561 | 35.7369 | 0.801112 |
| unet_percentile_minmax                     | U-Net                    | percentile min-max | 0.0015798   | 0.025495   | 0.0397467 | 31.7867 | 0.363207 |
| resunet_global_minmax                      | ResUNet                  | global min-max     | 0.00057993  | 0.00648077 | 0.0240817 | 43.9691 | 0.902884 |
