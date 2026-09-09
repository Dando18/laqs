# Pilot J_area/L2 Spearman correlations

J_area versus all collected L2-related counters for Experiments 1–3, all three tau profiles, both devices, and all three stratifications. Each per-kernel rho is computed across distinct layout mappings using average ranks for ties. Counter observations are the saved median across three profiler-launch medians (20 steady-state dispatches per launch). The summary is the median of defined per-kernel rho values, not a mean, not a pooled correlation, and not a median across launches. Undefined values are excluded from the median; the final column gives its kernel count. Bias+ReLU is excluded as requested. Undefined values can result from constant scores or counters; the CSV records the reason. Profiles reuse the pilot measurements with different J_area weights; learned-profile results describe the pilot data used for tuning.

Aliases are shown once: on H100, l1_miss_demand_to_l2 equals tex_source_l2_read_requests, and l1_to_l2_read_traffic equals l2_read_work. On MI300A, l1_miss_demand_to_l2 equals l1_to_l2_read_requests. MI300A total requests and hit rate are derived summaries, included alongside every native L2/boundary counter. L1-only and HBM counters are outside this report.

## Counter definitions

- H100 — Read requests: lts__t_requests_srcunit_tex_op_read.sum (tex_source_l2_read_requests)
- H100 — Read sectors: lts__t_sectors_srcunit_tex_op_read.sum (l2_read_work)
- H100 — Read-sector misses: lts__t_sectors_op_read_lookup_miss.sum (l2_read_misses)
- MI300A — TCP→TCC reads: TCP_TCC_READ_REQ_sum (l1_to_l2_read_requests)
- MI300A — TCP→TCC writes: TCP_TCC_WRITE_REQ_sum (l1_to_l2_write_requests)
- MI300A — TCP→TCC total: derived: TCP_TCC_READ_REQ_sum + TCP_TCC_WRITE_REQ_sum (l1_to_l2_total_requests)
- MI300A — TCC tag requests: TCC_REQ_sum (l2_tag_requests)
- MI300A — TCC reads: TCC_READ_sum (second_level_read_requests)
- MI300A — TCC hits: TCC_HIT_sum (l2_hits)
- MI300A — TCC misses: TCC_MISS_sum (l2_misses)
- MI300A — TCC hit rate: derived: 100 × TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum) (l2_hit_rate_percent)

## H100 · expert · Experiment 1 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.375 | 0.508 | 0.493 | 0.911 | 0.824 | 0.916 | 0.666 | 6 |

| Read sectors | -0.800 | -0.764 | 0.254 | 0.925 | 0.947 | 0.959 | 0.589 | 6 |

| Read-sector misses | -0.054 | 0.248 | -0.372 | 0.585 | 0.654 | 0.909 | 0.416 | 6 |

## H100 · expert · Experiment 1 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.578 | 0.551 | 0.642 | 0.935 | 0.877 | 0.947 | 0.759 | 6 |

| Read sectors | -0.778 | -0.773 | 0.057 | 0.913 | 0.955 | 0.954 | 0.485 | 6 |

| Read-sector misses | 0.029 | 0.066 | -0.332 | 0.671 | 0.719 | 0.896 | 0.369 | 6 |

## H100 · expert · Experiment 1 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.427 | 0.452 | 0.496 | 0.856 | 0.800 | 0.901 | 0.648 | 6 |

| Read sectors | -0.871 | -0.780 | 0.397 | 0.925 | 0.942 | 0.969 | 0.661 | 6 |

| Read-sector misses | 0.114 | 0.127 | -0.308 | 0.580 | 0.778 | 0.906 | 0.354 | 6 |

## H100 · expert · Experiment 2 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.156 | 0.310 | 0.648 | 0.920 | 0.929 | 0.903 | 0.776 | 6 |

| Read sectors | -0.915 | -0.908 | 0.482 | 0.956 | 0.912 | 0.950 | 0.697 | 6 |

| Read-sector misses | -0.201 | -0.295 | -0.356 | 0.779 | 0.634 | 0.931 | 0.217 | 6 |

## H100 · expert · Experiment 2 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.291 | 0.274 | 0.663 | 0.949 | 0.923 | 0.926 | 0.793 | 6 |

| Read sectors | -0.901 | -0.879 | -0.187 | 0.951 | 0.961 | 0.940 | 0.376 | 6 |

| Read-sector misses | 0.123 | -0.271 | -0.253 | 0.791 | 0.631 | 0.922 | 0.377 | 6 |

## H100 · expert · Experiment 2 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.176 | 0.366 | 0.649 | 0.907 | 0.946 | 0.915 | 0.778 | 6 |

| Read sectors | -0.907 | -0.905 | 0.449 | 0.962 | 0.894 | 0.946 | 0.672 | 6 |

| Read-sector misses | 0.015 | -0.176 | -0.057 | 0.776 | 0.696 | 0.939 | 0.355 | 6 |

## H100 · expert · Experiment 3 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.302 | 0.063 | 0.620 | 0.914 | 0.623 | 0.853 | 0.622 | 6 |

| Read sectors | -0.881 | -0.795 | -0.467 | 0.908 | 0.807 | 0.923 | 0.170 | 6 |

| Read-sector misses | 0.091 | -0.038 | 0.483 | 0.862 | 0.582 | 0.913 | 0.533 | 6 |

## H100 · expert · Experiment 3 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.207 | 0.165 | 0.684 | 0.915 | 0.618 | 0.716 | 0.651 | 6 |

| Read sectors | -0.898 | -0.696 | -0.427 | 0.923 | 0.803 | 0.897 | 0.188 | 6 |

| Read-sector misses | 0.067 | -0.001 | 0.371 | 0.898 | 0.636 | 0.930 | 0.504 | 6 |

## H100 · expert · Experiment 3 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.280 | 0.068 | 0.557 | 0.890 | 0.579 | 0.864 | 0.568 | 6 |

| Read sectors | -0.840 | -0.829 | -0.428 | 0.892 | 0.830 | 0.922 | 0.201 | 6 |

| Read-sector misses | -0.127 | -0.109 | 0.541 | 0.858 | 0.461 | 0.901 | 0.501 | 6 |

## H100 · l1_to_l2 · Experiment 1 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.454 | 0.578 | 0.628 | 0.956 | 0.871 | 0.937 | 0.750 | 6 |

| Read sectors | -0.721 | -0.708 | 0.357 | 0.807 | 0.885 | 0.933 | 0.582 | 6 |

| Read-sector misses | -0.089 | 0.231 | -0.423 | 0.508 | 0.634 | 0.882 | 0.370 | 6 |

## H100 · l1_to_l2 · Experiment 1 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.697 | 0.671 | 0.711 | 0.968 | 0.939 | 0.959 | 0.825 | 6 |

| Read sectors | -0.687 | -0.679 | 0.221 | 0.816 | 0.890 | 0.925 | 0.518 | 6 |

| Read-sector misses | 0.015 | 0.093 | -0.394 | 0.633 | 0.660 | 0.857 | 0.363 | 6 |

## H100 · l1_to_l2 · Experiment 1 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.476 | 0.486 | 0.628 | 0.927 | 0.870 | 0.921 | 0.749 | 6 |

| Read sectors | -0.822 | -0.737 | 0.475 | 0.782 | 0.884 | 0.954 | 0.628 | 6 |

| Read-sector misses | 0.106 | 0.100 | -0.360 | 0.447 | 0.730 | 0.897 | 0.276 | 6 |

## H100 · l1_to_l2 · Experiment 2 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.299 | 0.442 | 0.771 | 0.966 | 0.959 | 0.931 | 0.851 | 6 |

| Read sectors | -0.847 | -0.836 | 0.563 | 0.910 | 0.904 | 0.925 | 0.733 | 6 |

| Read-sector misses | -0.138 | -0.247 | -0.430 | 0.728 | 0.613 | 0.912 | 0.238 | 6 |

## H100 · l1_to_l2 · Experiment 2 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.488 | 0.461 | 0.715 | 0.976 | 0.961 | 0.942 | 0.828 | 6 |

| Read sectors | -0.834 | -0.746 | -0.008 | 0.894 | 0.919 | 0.914 | 0.443 | 6 |

| Read-sector misses | 0.114 | -0.300 | -0.396 | 0.771 | 0.625 | 0.888 | 0.369 | 6 |

## H100 · l1_to_l2 · Experiment 2 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.300 | 0.488 | 0.766 | 0.960 | 0.962 | 0.935 | 0.850 | 6 |

| Read sectors | -0.846 | -0.793 | 0.465 | 0.898 | 0.891 | 0.915 | 0.678 | 6 |

| Read-sector misses | 0.061 | -0.156 | -0.052 | 0.691 | 0.707 | 0.926 | 0.376 | 6 |

## H100 · l1_to_l2 · Experiment 3 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.383 | 0.129 | 0.764 | 0.958 | 0.807 | 0.933 | 0.785 | 6 |

| Read sectors | -0.863 | -0.767 | -0.390 | 0.929 | 0.836 | 0.947 | 0.223 | 6 |

| Read-sector misses | 0.069 | -0.019 | 0.414 | 0.891 | 0.624 | 0.942 | 0.519 | 6 |

## H100 · l1_to_l2 · Experiment 3 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.293 | 0.247 | 0.799 | 0.947 | 0.829 | 0.824 | 0.811 | 6 |

| Read sectors | -0.900 | -0.681 | -0.405 | 0.935 | 0.835 | 0.929 | 0.215 | 6 |

| Read-sector misses | 0.024 | 0.035 | 0.274 | 0.928 | 0.609 | 0.963 | 0.442 | 6 |

## H100 · l1_to_l2 · Experiment 3 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.351 | 0.112 | 0.739 | 0.945 | 0.769 | 0.944 | 0.754 | 6 |

| Read sectors | -0.831 | -0.805 | -0.323 | 0.925 | 0.839 | 0.943 | 0.258 | 6 |

| Read-sector misses | -0.111 | -0.112 | 0.394 | 0.895 | 0.494 | 0.935 | 0.444 | 6 |

## H100 · speedup · Experiment 1 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | -0.019 | -0.125 | 0.851 | 0.786 | 0.761 | 0.843 | 0.774 | 6 |

| Read sectors | 0.662 | 0.703 | 0.552 | 0.786 | 0.767 | 0.964 | 0.735 | 6 |

| Read-sector misses | 0.011 | -0.263 | -0.441 | 0.247 | 0.760 | 0.932 | 0.129 | 6 |

## H100 · speedup · Experiment 1 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | -0.267 | -0.231 | 0.754 | 0.470 | 0.469 | 0.842 | 0.469 | 6 |

| Read sectors | 0.806 | 0.796 | 0.719 | 0.470 | 0.530 | 0.944 | 0.757 | 6 |

| Read-sector misses | -0.052 | -0.015 | -0.477 | 0.039 | 0.494 | 0.924 | 0.012 | 6 |

## H100 · speedup · Experiment 1 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | -0.091 | 0.017 | 0.859 | 0.826 | 0.795 | 0.845 | 0.811 | 6 |

| Read sectors | 0.718 | 0.670 | 0.806 | 0.826 | 0.799 | 0.966 | 0.803 | 6 |

| Read-sector misses | -0.065 | -0.107 | -0.389 | 0.231 | 0.794 | 0.908 | 0.083 | 6 |

## H100 · speedup · Experiment 2 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.267 | 0.125 | 0.836 | 0.729 | 0.748 | 0.829 | 0.738 | 6 |

| Read sectors | 0.805 | 0.776 | 0.827 | 0.729 | 0.758 | 0.964 | 0.791 | 6 |

| Read-sector misses | 0.232 | 0.188 | -0.517 | 0.329 | 0.740 | 0.923 | 0.280 | 6 |

## H100 · speedup · Experiment 2 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.097 | 0.113 | 0.656 | 0.420 | 0.509 | 0.789 | 0.465 | 6 |

| Read sectors | 0.854 | 0.875 | 0.458 | 0.442 | 0.574 | 0.950 | 0.714 | 6 |

| Read-sector misses | -0.097 | 0.205 | -0.515 | 0.059 | 0.532 | 0.942 | 0.132 | 6 |

## H100 · speedup · Experiment 2 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.274 | 0.132 | 0.852 | 0.760 | 0.774 | 0.822 | 0.767 | 6 |

| Read sectors | 0.772 | 0.768 | 0.796 | 0.760 | 0.775 | 0.966 | 0.774 | 6 |

| Read-sector misses | 0.123 | 0.057 | -0.096 | 0.366 | 0.769 | 0.913 | 0.244 | 6 |

## H100 · speedup · Experiment 3 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.163 | 0.102 | 0.049 | -0.042 | 0.143 | 0.542 | 0.123 | 6 |

| Read sectors | -0.084 | 0.273 | -0.425 | -0.063 | 0.078 | 0.495 | 0.008 | 6 |

| Read-sector misses | 0.137 | 0.061 | 0.746 | -0.124 | 0.227 | 0.434 | 0.182 | 6 |

## H100 · speedup · Experiment 3 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.064 | 0.039 | 0.205 | 0.126 | 0.229 | 0.459 | 0.165 | 6 |

| Read sectors | 0.173 | 0.082 | -0.264 | 0.126 | 0.041 | 0.336 | 0.104 | 6 |

| Read-sector misses | -0.024 | 0.021 | 0.474 | 0.114 | 0.229 | 0.279 | 0.172 | 6 |

## H100 · speedup · Experiment 3 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| Read requests | 0.135 | 0.107 | -0.001 | 0.043 | 0.108 | 0.518 | 0.108 | 6 |

| Read sectors | 0.038 | 0.142 | -0.426 | 0.054 | 0.076 | 0.502 | 0.065 | 6 |

| Read-sector misses | 0.016 | 0.060 | 0.803 | -0.018 | 0.113 | 0.464 | 0.087 | 6 |

## MI300A · expert · Experiment 1 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.983 | 0.692 | 0.914 | 0.971 | 0.985 | 0.993 | 0.977 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.983 | 0.692 | 0.914 | 0.971 | 0.985 | 0.993 | 0.977 | 6 |

| TCC tag requests | 0.978 | 0.692 | 0.897 | 0.957 | 0.970 | 0.981 | 0.964 | 6 |

| TCC reads | 0.978 | 0.694 | 0.881 | 0.960 | 0.972 | 0.983 | 0.966 | 6 |

| TCC hits | 0.978 | 0.692 | 0.901 | 0.960 | 0.847 | 0.983 | 0.931 | 6 |

| TCC misses | -0.069 | 0.281 | -0.336 | 0.845 | 0.930 | -0.900 | 0.106 | 6 |

| TCC hit rate | 0.977 | 0.693 | 0.901 | 0.963 | 0.291 | 0.982 | 0.932 | 6 |

## MI300A · expert · Experiment 1 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.982 | 0.612 | 0.926 | 0.980 | 0.991 | 0.988 | 0.981 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.982 | 0.612 | 0.926 | 0.980 | 0.991 | 0.988 | 0.981 | 6 |

| TCC tag requests | 0.982 | 0.613 | 0.920 | 0.984 | 0.977 | 0.980 | 0.978 | 6 |

| TCC reads | 0.978 | 0.614 | 0.906 | 0.984 | 0.983 | 0.981 | 0.980 | 6 |

| TCC hits | 0.979 | 0.613 | 0.927 | 0.985 | 0.918 | 0.982 | 0.953 | 6 |

| TCC misses | 0.029 | 0.540 | -0.354 | 0.748 | 0.863 | -0.886 | 0.284 | 6 |

| TCC hit rate | 0.978 | 0.613 | 0.927 | 0.985 | 0.543 | 0.979 | 0.953 | 6 |

## MI300A · expert · Experiment 1 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.983 | 0.771 | 0.899 | 0.948 | 0.984 | 0.993 | 0.965 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.983 | 0.771 | 0.899 | 0.948 | 0.984 | 0.993 | 0.965 | 6 |

| TCC tag requests | 0.978 | 0.772 | 0.879 | 0.919 | 0.958 | 0.979 | 0.938 | 6 |

| TCC reads | 0.981 | 0.772 | 0.902 | 0.921 | 0.964 | 0.979 | 0.943 | 6 |

| TCC hits | 0.978 | 0.771 | 0.899 | 0.928 | 0.805 | 0.981 | 0.913 | 6 |

| TCC misses | 0.030 | 0.195 | -0.431 | 0.862 | 0.939 | -0.899 | 0.113 | 6 |

| TCC hit rate | 0.977 | 0.771 | 0.898 | 0.933 | 0.161 | 0.978 | 0.915 | 6 |

## MI300A · expert · Experiment 2 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.953 | 0.726 | 0.903 | 0.966 | 0.997 | 0.992 | 0.959 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.953 | 0.726 | 0.903 | 0.966 | 0.997 | 0.992 | 0.959 | 6 |

| TCC tag requests | 0.965 | 0.730 | 0.893 | 0.953 | 0.984 | 0.977 | 0.959 | 6 |

| TCC reads | 0.960 | 0.727 | 0.895 | 0.953 | 0.985 | 0.978 | 0.956 | 6 |

| TCC hits | 0.962 | 0.730 | 0.890 | 0.959 | 0.846 | 0.979 | 0.924 | 6 |

| TCC misses | -0.357 | 0.180 | -0.145 | 0.916 | 0.955 | -0.892 | 0.017 | 6 |

| TCC hit rate | 0.960 | 0.727 | 0.889 | 0.957 | 0.084 | 0.977 | 0.923 | 6 |

## MI300A · expert · Experiment 2 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.968 | 0.761 | 0.925 | 0.976 | 0.990 | 0.988 | 0.972 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.968 | 0.761 | 0.925 | 0.976 | 0.990 | 0.988 | 0.972 | 6 |

| TCC tag requests | 0.967 | 0.765 | 0.929 | 0.973 | 0.979 | 0.980 | 0.970 | 6 |

| TCC reads | 0.963 | 0.760 | 0.913 | 0.971 | 0.983 | 0.980 | 0.967 | 6 |

| TCC hits | 0.961 | 0.765 | 0.923 | 0.975 | 0.900 | 0.981 | 0.942 | 6 |

| TCC misses | -0.122 | 0.181 | -0.308 | 0.857 | 0.896 | -0.888 | 0.030 | 6 |

| TCC hit rate | 0.959 | 0.765 | 0.921 | 0.973 | 0.403 | 0.977 | 0.940 | 6 |

## MI300A · expert · Experiment 2 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.961 | 0.752 | 0.895 | 0.957 | 1.000 | 0.994 | 0.959 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.961 | 0.752 | 0.895 | 0.957 | 1.000 | 0.994 | 0.959 | 6 |

| TCC tag requests | 0.963 | 0.763 | 0.905 | 0.923 | 0.988 | 0.980 | 0.943 | 6 |

| TCC reads | 0.955 | 0.752 | 0.886 | 0.922 | 0.986 | 0.981 | 0.938 | 6 |

| TCC hits | 0.964 | 0.761 | 0.905 | 0.933 | 0.825 | 0.982 | 0.919 | 6 |

| TCC misses | -0.388 | 0.307 | -0.202 | 0.920 | 0.973 | -0.893 | 0.053 | 6 |

| TCC hit rate | 0.964 | 0.761 | 0.902 | 0.934 | -0.048 | 0.979 | 0.918 | 6 |

## MI300A · expert · Experiment 3 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.970 | 0.983 | 0.596 | 0.940 | 0.836 | 0.977 | 0.955 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.970 | 0.983 | 0.596 | 0.940 | 0.836 | 0.977 | 0.955 | 6 |

| TCC tag requests | 0.961 | 0.982 | 0.601 | 0.918 | 0.835 | 0.970 | 0.939 | 6 |

| TCC reads | 0.959 | 0.982 | 0.597 | 0.928 | 0.829 | 0.972 | 0.943 | 6 |

| TCC hits | 0.961 | 0.982 | 0.599 | 0.920 | 0.858 | 0.972 | 0.940 | 6 |

| TCC misses | 0.221 | 0.359 | 0.351 | 0.910 | 0.831 | 0.040 | 0.355 | 6 |

| TCC hit rate | 0.961 | 0.981 | 0.600 | 0.916 | 0.149 | 0.971 | 0.938 | 6 |

## MI300A · expert · Experiment 3 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.981 | 0.964 | 0.719 | 0.942 | 0.842 | 0.988 | 0.953 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.981 | 0.964 | 0.719 | 0.942 | 0.842 | 0.988 | 0.953 | 6 |

| TCC tag requests | 0.969 | 0.960 | 0.713 | 0.931 | 0.821 | 0.968 | 0.945 | 6 |

| TCC reads | 0.968 | 0.959 | 0.715 | 0.935 | 0.832 | 0.976 | 0.947 | 6 |

| TCC hits | 0.970 | 0.959 | 0.714 | 0.925 | 0.851 | 0.976 | 0.942 | 6 |

| TCC misses | 0.337 | 0.577 | 0.424 | 0.920 | 0.818 | -0.008 | 0.501 | 6 |

| TCC hit rate | 0.970 | 0.958 | 0.714 | 0.921 | 0.157 | 0.968 | 0.939 | 6 |

## MI300A · expert · Experiment 3 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.966 | 0.972 | 0.633 | 0.930 | 0.841 | 0.978 | 0.948 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.966 | 0.972 | 0.633 | 0.930 | 0.841 | 0.978 | 0.948 | 6 |

| TCC tag requests | 0.961 | 0.972 | 0.632 | 0.920 | 0.832 | 0.973 | 0.940 | 6 |

| TCC reads | 0.962 | 0.972 | 0.636 | 0.932 | 0.821 | 0.974 | 0.947 | 6 |

| TCC hits | 0.962 | 0.972 | 0.634 | 0.915 | 0.869 | 0.972 | 0.938 | 6 |

| TCC misses | 0.260 | 0.329 | 0.356 | 0.910 | 0.823 | -0.034 | 0.343 | 6 |

| TCC hit rate | 0.962 | 0.966 | 0.635 | 0.910 | 0.240 | 0.972 | 0.936 | 6 |

## MI300A · l1_to_l2 · Experiment 1 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.988 | 0.692 | 0.935 | 0.978 | 1.000 | 0.999 | 0.983 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.988 | 0.692 | 0.935 | 0.978 | 1.000 | 0.999 | 0.983 | 6 |

| TCC tag requests | 0.983 | 0.691 | 0.935 | 0.973 | 0.982 | 0.986 | 0.978 | 6 |

| TCC reads | 0.983 | 0.691 | 0.935 | 0.974 | 0.985 | 0.987 | 0.978 | 6 |

| TCC hits | 0.984 | 0.691 | 0.935 | 0.974 | 0.855 | 0.987 | 0.955 | 6 |

| TCC misses | -0.082 | 0.286 | -0.291 | 0.868 | 0.943 | -0.906 | 0.102 | 6 |

| TCC hit rate | 0.983 | 0.691 | 0.935 | 0.973 | 0.294 | 0.986 | 0.954 | 6 |

## MI300A · l1_to_l2 · Experiment 1 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.989 | 0.596 | 0.939 | 0.988 | 1.000 | 0.998 | 0.989 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.989 | 0.596 | 0.939 | 0.988 | 1.000 | 0.998 | 0.989 | 6 |

| TCC tag requests | 0.985 | 0.596 | 0.939 | 0.986 | 0.987 | 0.982 | 0.984 | 6 |

| TCC reads | 0.985 | 0.596 | 0.939 | 0.986 | 0.991 | 0.984 | 0.985 | 6 |

| TCC hits | 0.985 | 0.596 | 0.939 | 0.986 | 0.922 | 0.985 | 0.962 | 6 |

| TCC misses | 0.026 | 0.551 | -0.379 | 0.747 | 0.877 | -0.900 | 0.288 | 6 |

| TCC hit rate | 0.985 | 0.596 | 0.939 | 0.986 | 0.545 | 0.982 | 0.961 | 6 |

## MI300A · l1_to_l2 · Experiment 1 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.988 | 0.781 | 0.933 | 0.953 | 1.000 | 0.999 | 0.971 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.988 | 0.781 | 0.933 | 0.953 | 1.000 | 0.999 | 0.971 | 6 |

| TCC tag requests | 0.982 | 0.780 | 0.933 | 0.947 | 0.978 | 0.983 | 0.962 | 6 |

| TCC reads | 0.982 | 0.780 | 0.933 | 0.947 | 0.982 | 0.985 | 0.965 | 6 |

| TCC hits | 0.983 | 0.780 | 0.933 | 0.948 | 0.823 | 0.986 | 0.941 | 6 |

| TCC misses | 0.035 | 0.198 | -0.417 | 0.905 | 0.959 | -0.906 | 0.116 | 6 |

| TCC hit rate | 0.982 | 0.780 | 0.933 | 0.947 | 0.163 | 0.983 | 0.940 | 6 |

## MI300A · l1_to_l2 · Experiment 2 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.977 | 0.727 | 0.935 | 0.988 | 1.000 | 0.999 | 0.982 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.977 | 0.727 | 0.935 | 0.988 | 1.000 | 0.999 | 0.982 | 6 |

| TCC tag requests | 0.975 | 0.727 | 0.935 | 0.974 | 0.989 | 0.983 | 0.975 | 6 |

| TCC reads | 0.975 | 0.727 | 0.935 | 0.975 | 0.990 | 0.984 | 0.975 | 6 |

| TCC hits | 0.976 | 0.727 | 0.935 | 0.977 | 0.855 | 0.985 | 0.955 | 6 |

| TCC misses | -0.381 | 0.172 | -0.160 | 0.942 | 0.960 | -0.899 | 0.006 | 6 |

| TCC hit rate | 0.975 | 0.727 | 0.935 | 0.974 | 0.087 | 0.983 | 0.954 | 6 |

## MI300A · l1_to_l2 · Experiment 2 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.981 | 0.761 | 0.939 | 0.987 | 1.000 | 0.998 | 0.984 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.981 | 0.761 | 0.939 | 0.987 | 1.000 | 0.998 | 0.984 | 6 |

| TCC tag requests | 0.979 | 0.761 | 0.939 | 0.984 | 0.988 | 0.982 | 0.981 | 6 |

| TCC reads | 0.979 | 0.761 | 0.939 | 0.984 | 0.991 | 0.984 | 0.981 | 6 |

| TCC hits | 0.979 | 0.761 | 0.939 | 0.985 | 0.903 | 0.984 | 0.959 | 6 |

| TCC misses | -0.137 | 0.197 | -0.336 | 0.881 | 0.906 | -0.906 | 0.030 | 6 |

| TCC hit rate | 0.979 | 0.761 | 0.939 | 0.984 | 0.404 | 0.982 | 0.959 | 6 |

## MI300A · l1_to_l2 · Experiment 2 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.976 | 0.751 | 0.933 | 0.986 | 1.000 | 0.999 | 0.981 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.976 | 0.751 | 0.933 | 0.986 | 1.000 | 0.999 | 0.981 | 6 |

| TCC tag requests | 0.974 | 0.750 | 0.933 | 0.965 | 0.988 | 0.983 | 0.970 | 6 |

| TCC reads | 0.974 | 0.750 | 0.933 | 0.965 | 0.987 | 0.985 | 0.970 | 6 |

| TCC hits | 0.975 | 0.751 | 0.933 | 0.970 | 0.825 | 0.986 | 0.952 | 6 |

| TCC misses | -0.409 | 0.312 | -0.234 | 0.956 | 0.974 | -0.900 | 0.039 | 6 |

| TCC hit rate | 0.974 | 0.750 | 0.933 | 0.965 | -0.048 | 0.983 | 0.949 | 6 |

## MI300A · l1_to_l2 · Experiment 3 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.991 | 0.983 | 0.791 | 0.950 | 0.849 | 1.000 | 0.967 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.991 | 0.983 | 0.791 | 0.950 | 0.849 | 1.000 | 0.967 | 6 |

| TCC tag requests | 0.984 | 0.982 | 0.790 | 0.927 | 0.846 | 0.992 | 0.955 | 6 |

| TCC reads | 0.984 | 0.982 | 0.788 | 0.932 | 0.843 | 0.994 | 0.957 | 6 |

| TCC hits | 0.985 | 0.982 | 0.790 | 0.933 | 0.874 | 0.994 | 0.958 | 6 |

| TCC misses | 0.242 | 0.366 | 0.439 | 0.919 | 0.842 | 0.064 | 0.402 | 6 |

| TCC hit rate | 0.984 | 0.979 | 0.791 | 0.928 | 0.150 | 0.992 | 0.953 | 6 |

## MI300A · l1_to_l2 · Experiment 3 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.986 | 0.965 | 0.836 | 0.953 | 0.860 | 1.000 | 0.959 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.986 | 0.965 | 0.836 | 0.953 | 0.860 | 1.000 | 0.959 | 6 |

| TCC tag requests | 0.974 | 0.963 | 0.828 | 0.936 | 0.841 | 0.980 | 0.950 | 6 |

| TCC reads | 0.974 | 0.963 | 0.828 | 0.942 | 0.851 | 0.989 | 0.953 | 6 |

| TCC hits | 0.975 | 0.963 | 0.830 | 0.936 | 0.874 | 0.987 | 0.950 | 6 |

| TCC misses | 0.348 | 0.582 | 0.420 | 0.929 | 0.839 | -0.008 | 0.501 | 6 |

| TCC hit rate | 0.975 | 0.960 | 0.831 | 0.930 | 0.157 | 0.979 | 0.945 | 6 |

## MI300A · l1_to_l2 · Experiment 3 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | 0.991 | 0.956 | 0.797 | 0.941 | 0.843 | 1.000 | 0.949 | 6 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | 0.991 | 0.956 | 0.797 | 0.941 | 0.843 | 1.000 | 0.949 | 6 |

| TCC tag requests | 0.986 | 0.956 | 0.791 | 0.921 | 0.831 | 0.993 | 0.938 | 6 |

| TCC reads | 0.985 | 0.955 | 0.789 | 0.934 | 0.823 | 0.994 | 0.945 | 6 |

| TCC hits | 0.987 | 0.956 | 0.792 | 0.924 | 0.872 | 0.994 | 0.940 | 6 |

| TCC misses | 0.304 | 0.328 | 0.399 | 0.913 | 0.822 | 0.007 | 0.363 | 6 |

| TCC hit rate | 0.987 | 0.949 | 0.792 | 0.920 | 0.251 | 0.993 | 0.934 | 6 |

## MI300A · speedup · Experiment 1 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | -0.959 | -0.681 | 0.842 | — | — | — | -0.681 | 3 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | -0.959 | -0.681 | 0.842 | — | — | — | -0.681 | 3 |

| TCC tag requests | -0.955 | -0.682 | 0.809 | — | — | — | -0.682 | 3 |

| TCC reads | -0.953 | -0.686 | 0.779 | — | — | — | -0.686 | 3 |

| TCC hits | -0.953 | -0.683 | 0.816 | — | — | — | -0.683 | 3 |

| TCC misses | 0.046 | -0.275 | -0.331 | — | — | — | -0.275 | 3 |

| TCC hit rate | -0.953 | -0.684 | 0.815 | — | — | — | -0.684 | 3 |

## MI300A · speedup · Experiment 1 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | -0.948 | -0.622 | 0.850 | — | — | — | -0.622 | 3 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | -0.948 | -0.622 | 0.850 | — | — | — | -0.622 | 3 |

| TCC tag requests | -0.952 | -0.624 | 0.838 | — | — | — | -0.624 | 3 |

| TCC reads | -0.944 | -0.626 | 0.812 | — | — | — | -0.626 | 3 |

| TCC hits | -0.946 | -0.625 | 0.853 | — | — | — | -0.625 | 3 |

| TCC misses | -0.027 | -0.512 | -0.243 | — | — | — | -0.243 | 3 |

| TCC hit rate | -0.944 | -0.623 | 0.852 | — | — | — | -0.623 | 3 |

## MI300A · speedup · Experiment 1 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | -0.965 | -0.756 | 0.821 | — | — | — | -0.756 | 3 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | -0.965 | -0.756 | 0.821 | — | — | — | -0.756 | 3 |

| TCC tag requests | -0.962 | -0.758 | 0.781 | — | — | — | -0.758 | 3 |

| TCC reads | -0.967 | -0.759 | 0.814 | — | — | — | -0.759 | 3 |

| TCC hits | -0.961 | -0.757 | 0.811 | — | — | — | -0.757 | 3 |

| TCC misses | -0.029 | -0.186 | -0.423 | — | — | — | -0.186 | 3 |

| TCC hit rate | -0.959 | -0.757 | 0.810 | — | — | — | -0.757 | 3 |

## MI300A · speedup · Experiment 2 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | -0.910 | -0.702 | 0.817 | — | — | — | -0.702 | 3 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | -0.910 | -0.702 | 0.817 | — | — | — | -0.702 | 3 |

| TCC tag requests | -0.929 | -0.709 | 0.801 | — | — | — | -0.709 | 3 |

| TCC reads | -0.921 | -0.703 | 0.808 | — | — | — | -0.703 | 3 |

| TCC hits | -0.923 | -0.708 | 0.799 | — | — | — | -0.708 | 3 |

| TCC misses | 0.337 | -0.167 | -0.118 | — | — | — | -0.118 | 3 |

| TCC hit rate | -0.920 | -0.703 | 0.798 | — | — | — | -0.703 | 3 |

## MI300A · speedup · Experiment 2 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | -0.925 | -0.735 | 0.819 | — | — | — | -0.735 | 3 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | -0.925 | -0.735 | 0.819 | — | — | — | -0.735 | 3 |

| TCC tag requests | -0.925 | -0.741 | 0.836 | — | — | — | -0.741 | 3 |

| TCC reads | -0.917 | -0.732 | 0.811 | — | — | — | -0.732 | 3 |

| TCC hits | -0.913 | -0.741 | 0.827 | — | — | — | -0.741 | 3 |

| TCC misses | 0.107 | -0.152 | -0.234 | — | — | — | -0.152 | 3 |

| TCC hit rate | -0.911 | -0.741 | 0.824 | — | — | — | -0.741 | 3 |

## MI300A · speedup · Experiment 2 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | -0.925 | -0.733 | 0.807 | — | — | — | -0.733 | 3 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | -0.925 | -0.733 | 0.807 | — | — | — | -0.733 | 3 |

| TCC tag requests | -0.929 | -0.747 | 0.827 | — | — | — | -0.747 | 3 |

| TCC reads | -0.917 | -0.730 | 0.794 | — | — | — | -0.730 | 3 |

| TCC hits | -0.931 | -0.745 | 0.832 | — | — | — | -0.745 | 3 |

| TCC misses | 0.360 | -0.288 | -0.159 | — | — | — | -0.159 | 3 |

| TCC hit rate | -0.931 | -0.745 | 0.828 | — | — | — | -0.745 | 3 |

## MI300A · speedup · Experiment 3 · all stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | -0.278 | -0.420 | 0.647 | — | — | — | -0.278 | 3 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | -0.278 | -0.420 | 0.647 | — | — | — | -0.278 | 3 |

| TCC tag requests | -0.269 | -0.401 | 0.644 | — | — | — | -0.269 | 3 |

| TCC reads | -0.260 | -0.398 | 0.642 | — | — | — | -0.260 | 3 |

| TCC hits | -0.262 | -0.405 | 0.644 | — | — | — | -0.262 | 3 |

| TCC misses | -0.072 | -0.056 | 0.338 | — | — | — | -0.056 | 3 |

| TCC hit rate | -0.263 | -0.468 | 0.646 | — | — | — | -0.263 | 3 |

## MI300A · speedup · Experiment 3 · issue stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | -0.233 | -0.252 | 0.677 | — | — | — | -0.233 | 3 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | -0.233 | -0.252 | 0.677 | — | — | — | -0.233 | 3 |

| TCC tag requests | -0.216 | -0.210 | 0.680 | — | — | — | -0.210 | 3 |

| TCC reads | -0.188 | -0.219 | 0.685 | — | — | — | -0.188 | 3 |

| TCC hits | -0.203 | -0.215 | 0.679 | — | — | — | -0.203 | 3 |

| TCC misses | -0.007 | -0.010 | 0.457 | — | — | — | -0.007 | 3 |

| TCC hit rate | -0.201 | -0.237 | 0.678 | — | — | — | -0.201 | 3 |

## MI300A · speedup · Experiment 3 · temporal stratification

Layout counts: gemv: 100, gesummv: 100, mvt: 100, embedding_bag: 100, softmax_bias: 100, stencil5: 100

| Counter | GEMV | GESUMMV | MVT | Embedding bag | Softmax+bias | Stencil5 | Median | Defined kernels |

|---|---|---|---|---|---|---|---|---|

| TCP→TCC reads | -0.183 | -0.396 | 0.659 | — | — | — | -0.183 | 3 |

| TCP→TCC writes | — | — | — | — | — | — | — | 0 |

| TCP→TCC total | -0.183 | -0.396 | 0.659 | — | — | — | -0.183 | 3 |

| TCC tag requests | -0.179 | -0.387 | 0.660 | — | — | — | -0.179 | 3 |

| TCC reads | -0.179 | -0.395 | 0.675 | — | — | — | -0.179 | 3 |

| TCC hits | -0.183 | -0.393 | 0.663 | — | — | — | -0.183 | 3 |

| TCC misses | -0.038 | -0.104 | 0.437 | — | — | — | -0.038 | 3 |

| TCC hit rate | -0.183 | -0.433 | 0.663 | — | — | — | -0.183 | 3 |
