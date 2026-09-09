"""Independent-process graph timing and a bounded, explicitly measured choice."""
from __future__ import annotations

import statistics

from experiment_support import process_summary


def sample_record(values):
    return {'samples_ms': values, 'median_ms': statistics.median(values),
            'mean_ms': statistics.fmean(values), 'min_ms': min(values)}


def balanced_order(labels, index):
    rotation = index % len(labels)
    order = labels[rotation:] + labels[:rotation]
    return order[::-1] if index % 2 else order


def allocation_samples(launches, contexts, input_arguments, output_arguments, *,
                       process_index, samples=21, iterations=50, warmup=10,
                       placements=3, on_capture=None, on_ready=None):
    """Compare every variant at every shared input placement, outside packing time.

    Prepared input bytes are copied into shared storage before each measurement.
    Same-candidate graph warmup follows the copy, so the copy itself and the
    source packing allocations are outside the measured steady-state dispatches.
    Placements are repeated measures; the independent unit remains a process.
    """
    import os
    import socket
    import torch
    from layout_runtime import fresh_outputs

    launches, contexts = dict(launches), dict(contexts)
    ordinary = launches['ordinary']
    launches['identity'] = fresh_outputs(ordinary, output_arguments)
    launches['same_pointer'] = ordinary
    for label in ('identity', 'same_pointer'):
        contexts[label] = contexts['ordinary']
    sources = {label: [launch.values[i] for i in input_arguments] for label, launch in launches.items()}
    templates = sources['ordinary']
    for values in sources.values():
        for value, template in zip(values, templates):
            if (not value.is_contiguous() or value.numel() != template.numel()
                    or value.dtype != template.dtype or value.device != template.device):
                raise ValueError('controlled input placements require equal-sized dense inputs')
    banks = [[torch.empty_like(t.reshape(-1)) for t in templates] for _ in range(placements)]
    for label, launch in launches.items():
        if label != 'identity':
            for index in output_arguments:
                launch.values[index] = ordinary.values[index]
    labels = tuple(launches)
    graphs, addresses = {}, []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for placement, bank in enumerate(banks):
            address = {}
            for label in balanced_order(labels, placement + process_index):
                launch = launches[label]
                for index, buffer, source in zip(input_arguments, bank, sources[label]):
                    buffer.copy_(source.reshape(-1))
                    launch.values[index] = buffer.view(source.shape)
                address[label] = {str(i): launch.values[i].data_ptr() for i in (*input_arguments, *output_arguments)}
                with contexts[label]():
                    for _ in range(warmup):
                        launch.run()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        for _ in range(iterations):
                            kernel = launch.run()
                graphs[placement, label] = graph
                if on_ready is not None:
                    on_ready(label, launch)
                if on_capture is not None and placement == 0:
                    on_capture(label, kernel)
            addresses.append(address)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    records = [{label: [] for label in labels} for _ in banks]
    for sample in range(samples):
        for placement in balanced_order(tuple(range(placements)), sample + process_index):
            bank = banks[placement]
            for label in balanced_order(labels, sample + placement + process_index):
                for buffer, source in zip(bank, sources[label]):
                    buffer.copy_(source.reshape(-1))
                graph = graphs[placement, label]
                for _ in range(warmup):
                    graph.replay()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                records[placement][label].append(start.elapsed_time(end) / iterations)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        'timings': {label: sample_record([v for record in records for v in record[label]]) for label in labels},
        'allocation': {
            'method': 'shared input banks; prepared bytes copied outside timing; same-candidate graph warmup',
            'placements': [{'addresses': address, 'timings': {label: sample_record(values) for label, values in record.items()}}
                           for address, record in zip(addresses, records)],
            'input_arguments': list(input_arguments), 'output_arguments': list(output_arguments),
            'device': {'host': socket.gethostname(), 'index': torch.cuda.current_device(), 'name': props.name,
                       'uuid': str(getattr(props, 'uuid', 'unavailable')),
                       'visibility': {key: os.environ.get(key) for key in
                                      ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')}},
        },
    }


def graph_samples(launches, contexts, output_arguments, *, process_index, samples=21,
                  iterations=50, warmup=10, same_pointer_labels=(), on_capture=None):
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
    for label in same_pointer_labels:
        for index in output_arguments:
            launches[label].values[index] = ordinary.values[index]
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
                        kernel = launch.run()
            graphs[label] = graph
            if on_capture is not None:
                on_capture(label, kernel)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for _ in range(warmup):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    records = {label: [] for label in graphs}
    labels = tuple(graphs)
    for index in range(samples):
        order = balanced_order(labels, index + process_index)
        for label in order:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graphs[label].replay()
            end.record()
            end.synchronize()
            records[label].append(start.elapsed_time(end) / iterations)
    return {label: sample_record(values) for label, values in records.items()}


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
        if all('allocation' in row for row in records):
            summary['placement_speedups'] = [
                [p['timings']['ordinary']['median_ms'] / p['timings'][label]['median_ms']
                 for p in row['allocation']['placements']] for row in records]
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
