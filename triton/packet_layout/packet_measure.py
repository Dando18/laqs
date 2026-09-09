"""Independent-process graph timing and a bounded, explicitly measured choice."""
from __future__ import annotations

import statistics

from experiment_support import process_summary


def graph_samples(launches, contexts, output_arguments, *, process_index, samples=21,
                  iterations=50, warmup=10):
    import torch
    from layout_runtime import fresh_outputs

    ordinary = launches['ordinary']
    launches = dict(launches)
    launches['identity'] = fresh_outputs(ordinary, output_arguments)
    contexts = {**contexts, 'identity': contexts['ordinary']}
    # Rotate physical output allocations across labels in independent processes.
    banks = [[launch.values[index] for index in output_arguments] for launch in launches.values()]
    for i, launch in enumerate(launches.values()):
        for argument, tensor in zip(output_arguments, banks[(i + process_index) % len(banks)]):
            launch.values[argument] = tensor
    launches['same_pointer'] = ordinary
    contexts['same_pointer'] = contexts['ordinary']
    graphs = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for label, launch in launches.items():
            with contexts[label]():
                for _ in range(warmup):
                    launch.run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(iterations):
                        launch.run()
            graphs[label] = graph
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for _ in range(warmup):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    records = {label: [] for label in graphs}
    labels = tuple(graphs)
    for index in range(samples):
        rotation = (index + process_index) % len(labels)
        order = labels[rotation:] + labels[:rotation]
        if (index + process_index) % 2:
            order = order[::-1]
        for label in order:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graphs[label].replay()
            end.record()
            end.synchronize()
            records[label].append(start.elapsed_time(end) / iterations)
    return {label: {'samples_ms': values, 'median_ms': statistics.median(values),
                    'mean_ms': statistics.fmean(values), 'min_ms': min(values)}
            for label, values in records.items()}


def summarize(records):
    labels = set.intersection(*(set(record['timings']) for record in records)) - {'ordinary', 'identity', 'same_pointer'}
    labels.add('ordinary')
    result = {}
    for label in sorted(labels):
        paired = [{'timings': {'baseline': row['timings']['ordinary'],
                               'selected': row['timings'][label],
                               'identity': row['timings']['identity']}} for row in records]
        summary = process_summary(paired)
        summary['same_pointer_max_deviation'] = max(abs(row['timings']['ordinary']['median_ms'] /
            row['timings']['same_pointer']['median_ms'] - 1) for row in records)
        result[label] = summary
    return result


def packing_cost(launch, layouts, repeats=3):
    """Measure warm packing plus allocation, without compilation or input creation."""
    if not layouts:
        return {'median_ms': 0., 'gpu_median_ms': 0., 'samples_ms': []}
    import torch
    from time import perf_counter
    from layout_runtime import replace_inputs
    packed = replace_inputs(launch, layouts)
    torch.cuda.synchronize()
    wall, gpu = [], []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        before = perf_counter()
        start.record()
        packed = replace_inputs(launch, layouts)
        end.record()
        end.synchronize()
        wall.append((perf_counter() - before) * 1000)
        gpu.append(start.elapsed_time(end))
    return {'median_ms': statistics.median(wall), 'gpu_median_ms': statistics.median(gpu),
            'samples_ms': wall, 'gpu_samples_ms': gpu,
            'method': 'warm synchronous packing plus allocation; compilation and input generation excluded'}


def conversion_result(baseline_ms, selected_ms, packing_ms):
    saving = baseline_ms - selected_ms
    return {'kernel_speedup': baseline_ms / selected_ms,
            'one_use_including_conversion_speedup': baseline_ms / (selected_ms + packing_ms),
            'break_even_reuses': int(packing_ms / saving) + 1 if saving > 0 else None}


def measured_choice(summaries, *, minimum_gain=0.01):
    """Require the tuning interval to clear gain plus a two-sided control budget."""
    eligible = ['ordinary']
    for label, value in summaries.items():
        interval = value.get('speedup_ci95')
        deviation = max(value['identity_max_deviation'], value['same_pointer_max_deviation'])
        threshold = (1 + minimum_gain) * (1 + deviation) / (1 - deviation) if deviation < 1 else float('inf')
        if label != 'ordinary' and interval and interval[0] > threshold:
            eligible.append(label)
    return max(eligible, key=lambda label: (summaries[label]['speedup'], label == 'ordinary'))
