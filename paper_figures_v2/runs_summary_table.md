# Key Training Runs Summary

| Run                      | Change                     |   Val PER |   Test PER | Notes                                                                    |
|:-------------------------|:---------------------------|----------:|-----------:|:-------------------------------------------------------------------------|
| speechBaseline4          | Baseline                   |    0.2046 |     0.2046 | Constant LR=0.02, FP32, heavy augmentation                               |
| run11_plateau_break      | OneCycleLR + AMP           |    0.2863 |     0.283  | OneCycleLR max_lr=0.004, BF16, grad_accum=2                              |
| run15_warmup_cosine_safe | Warmup→Cosine (no EMA)     |    0.269  |     0.2636 | Warmup 1200→Cosine 8800, peak LR=1.5e-3, no EMA                          |
| run16_greedy_ema_warmcos | EMA 0.999                  |    0.2174 |     0.2095 | Warmup→Cosine + EMA=0.999, peak LR=1.6e-3                                |
| run16_4                  | EMA 0.9995 + input_dropout |    0.2139 |     0.2078 | EMA=0.9995, input_dropout=0.05, peak LR=1.4e-3                           |
| run16_5                  | Long-tail + EMA 0.9995     |    0.1995 |     0.1939 | 20k batches, cosine=18500, EMA=0.9995, input_dropout=0.05, peak LR=0.001 |
