# MI300A byte-scale validation

One unchanged edge construction and hardware profile are evaluated at
2-, 4-, and 8-byte elements. FP16 uses FP16 storage and FP32 accumulation.
Runtime regret is against the measured cross-dtype candidate panel, not
the complete 73-layout corpus.

| Kernel | N | Type | Analytical frontier | Panel | Regret | Speedup vs row-major |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| ATAX | 256 | FP64 | 5/73 | 16 | 0.000% | 1.414x |
| ATAX | 256 | FP32 | 12/73 | 16 | 0.000% | 1.409x |
| ATAX | 256 | FP16-storage/FP32-accumulation | 10/73 | 16 | 0.000% | 1.517x |
| GESUMMV | 256 | FP64 | 7/73 | 16 | 0.000% | 2.001x |
| GESUMMV | 256 | FP32 | 6/73 | 16 | 0.000% | 2.150x |
| GESUMMV | 256 | FP16-storage/FP32-accumulation | 4/73 | 16 | 3.023% | 2.153x |

## Analytical invariants

For each kernel/size pair, edge geometry, component names, tau, and kappa are identical across data types. The useful-byte denominator scales exactly with element width.

## Aggregate measured-panel result

- Exact panel winner: 5/6.
- Within 1%: 5/6.
- Mean regret: 0.504%.
- Maximum regret: 3.023%.
