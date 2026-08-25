# MI300A byte-scale focused recheck

The only nonzero-regret case from the N=256 candidate-panel experiment was
rerun with 20 samples, 10 kernel iterations per sample, and 5 warmup
iterations. Tau, edge construction, dtype, launch geometry, and layout
membership were unchanged.

| GESUMMV FP16 layout | Frontier member | Initial median | Recheck median |
| --- | --- | ---: | ---: |
| `tile8_column_major` | no | 0.028680 ms | 0.027908 ms |
| `tile16x8_column_major` | yes | 0.029547 ms | 0.028624 ms |

The confirmed measured-panel regret is 2.566%. Both evaluators passed the
five-point correctness check; FP16 storage uses FP32 accumulation.
