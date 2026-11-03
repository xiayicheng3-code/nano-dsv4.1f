"""Read-only source audit: rejection sites and 8K/16K/32K complete-prefix yield.

No training shards are written. Sample yields are never extrapolated to capacity.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import inspect
import itertools
import json
import linecache
import math
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import prepare_posttrain_corpus as prep
import prepare_trace_corpus as traces
from nano_dsv41f.chat_protocol import EOS_TOKEN_ID
from nano_dsv41f.trace_corpus import assistant_sft_loss_mask, render_case_v41
from nano_dsv41f.reasoning_effort import assign_length_guided_reasoning_effort, reasoning_text

LIMITS = (8192, 16384, 32768)


def audited_adapter(adapter, source, row, index):
    """Observe original adapter returns/exceptions without changing its decisions."""
    filename = inspect.getsourcefile(adapter)
    events = []

    def trace(frame, event, arg):
        if frame.f_code.co_filename != filename:
            return None
        if event == 'return' and arg is None:
            events.append({'event': 'return_none', 'function': frame.f_code.co_name,
                           'line': frame.f_lineno})
        elif event == 'exception':
            events.append({'event': 'exception', 'function': frame.f_code.co_name,
                           'line': frame.f_lineno, 'exception_type': arg[0].__name__})
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        case = adapter(source, row, index)
        status = 'accepted' if case is not None else 'rejected'
    except Exception as exc:
        case, status = None, 'raised_exception'
        events.append({'event': 'uncaught_exception', 'exception_type': type(exc).__name__,
                       'function': adapter.__name__, 'line': adapter.__code__.co_firstlineno})
    finally:
        sys.settrace(previous)
    # Accepted rows may also traverse helpers returning None; do not call those rejections.
    if status == 'accepted':
        return case, status, []
    for event in events:
        n = event['line']
        event['context'] = ''.join(linecache.getline(filename, i)
                                   for i in range(max(1, n - 2), n + 1)).strip()
    return case, status, events


def row_schema(row):
    """Bounded structural evidence; never save complete source text or credentials."""
    result = {'fields': {str(k): type(v).__name__ for k, v in row.items()}}
    for key in ('target', 'correctness', 'verified', 'success', 'exit_status'):
        value = row.get(key)
        if isinstance(value, (bool, int, float, str)) or value is None:
            result[key] = {'type': type(value).__name__, 'value': str(value)[:80]}
    raw = row.get('trajectory')
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            result['trajectory_parse'] = 'invalid_json'
    result['trajectory_decoded_type'] = type(raw).__name__
    if isinstance(raw, list):
        result['trajectory_items'] = len(raw)
        result['trajectory_roles'] = dict(Counter(str(x.get('role'))[:50]
                                                for x in raw if isinstance(x, dict)))
        result['first_item_fields'] = [sorted(x) for x in raw[:3] if isinstance(x, dict)]
    return result


def measure_views(ids, case):
    views = {limit: prep.prefix_view(ids, case, limit) for limit in LIMITS}
    a = np.asarray(ids)
    ends = np.flatnonzero((a == EOS_TOKEN_ID) & assistant_sft_loss_mask(a).astype(bool)) + 1
    final_end = int(ends[-1]) if len(ends) else None
    metrics = {}
    for limit, view in views.items():
        tokens = len(view.tokens) if view is not None else 0
        metrics[limit] = {
            'eligible_records': int(view is not None),
            'raw_tokens': tokens,
            'supervised_tokens': int(view.sft_loss_mask.sum()) if view is not None else 0,
            'over_8k_records': int(tokens > 8192),
            'over_16k_records': int(tokens > 16384),
            'full_through_final_assistant_records': int(bool(tokens) and tokens == final_end),
            'partial_prefix_records': int(bool(tokens) and tokens != final_end),
        }
    return views, metrics, final_end


def sampled_rows(source, seed, max_rows, config_counts):
    configs = [f'seed_{i}' for i in range(42, 58)] if source.key == 'openresearcher' else [source.config]
    remaining = max_rows
    for offset, config in enumerate(configs):
        if max_rows and remaining <= 0:
            return
        # Cover every OpenResearcher seed in a bounded run, instead of only seed_42.
        take = math.ceil(remaining / (len(configs) - offset)) if max_rows else None
        stream = iter(traces.load_stream(replace(source, config=config), seed=seed + offset,
                                        shuffle_buffer=128))
        for row in itertools.islice(stream, take):
            config_counts[str(config)] += 1
            if max_rows:
                remaining -= 1
            yield row


def diagnose_source(source, tokenizer, *, max_rows=2000, batch_size=16, seed=1701,
                    deadline=None):
    source = replace(source, max_observation_chars=None, preserve_history=True)
    counts, configs, reasons, evidence, schemas, target_types = Counter(), Counter(), Counter(), {}, [], Counter()
    totals = {limit: Counter({k: 0 for k in ('eligible_records', 'raw_tokens', 'supervised_tokens',
              'over_8k_records', 'over_16k_records', 'full_through_final_assistant_records',
              'partial_prefix_records')}) for limit in LIMITS}
    full_lengths, seen, batch = [], set(), []
    args = SimpleNamespace(midtrain_tokens=600_000_000, sft_reasoning_tokens=30_000_000,
                           sft_agent_tokens=60_000_000, headroom=1.1, sft_long_token_fraction=.5)
    selectors = {limit: prep.StageSelection(source, args) for limit in LIMITS[1:]}
    selected_tasks = {limit: set() for limit in selectors}
    stop_reason = 'sample_finished' if max_rows else 'source_exhausted'

    def flush():
        if not batch:
            return
        encodings = tokenizer.encode_batch([reasoning_text(c) for c in batch], add_special_tokens=False)
        for case, enc in zip(batch, encodings):
            case['metadata']['_reasoning_tokens'] = len(enc.ids)
        labelled = assign_length_guided_reasoning_effort(
            batch, length_fn=lambda c: c['metadata']['_reasoning_tokens'], length_unit='tokens',
            jitter=2, seed=seed, preserve_existing=False)
        cases, rendered = [], []
        for case in labelled:
            try:
                text = render_case_v41(case)
            except Exception as exc:
                counts['render_rejected'] += 1
                reasons['render:' + type(exc).__name__] += 1
                continue
            cases.append(case)
            rendered.append(text)
        for case, text, enc in zip(cases, rendered, tokenizer.encode_batch(rendered, add_special_tokens=False)):
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                counts['duplicate_render'] += 1
                continue
            seen.add(digest)
            counts['unique_rendered_records'] += 1
            full_lengths.append(len(enc.ids))
            views, metrics, final_end = measure_views(enc.ids, case)
            for limit, metric in metrics.items():
                totals[limit].update(metric)
            counts['no_assistant_eos'] += int(final_end is None)
            counts['no_complete_32k_view'] += int(views[32768] is None)
            rescued = views[16384] is None and views[32768] is not None
            counts['rescued_at_32k_records'] += int(rescued)
            counts['rescued_at_32k_raw_tokens'] += len(views[32768].tokens) if rescued else 0
            counts['rescued_at_32k_supervised_tokens'] += int(views[32768].sft_loss_mask.sum()) if rescued else 0
            extended = (views[16384] is not None and views[32768] is not None
                        and len(views[32768].tokens) > len(views[16384].tokens))
            counts['extended_at_32k_records'] += int(extended)
            if extended:
                counts['extended_at_32k_added_supervised_tokens'] += int(
                    views[32768].sft_loss_mask.sum() - views[16384].sft_loss_mask.sum())
            task = prep.task_key(case)
            for limit, selector in selectors.items():
                if source.key == 'openresearcher' and task in selected_tasks[limit]:
                    continue
                # A separate counterfactual selection per context length; no shards written.
                choice = selector.select(task, {'midtrain': views[8192], 'sft': views[limit]})
                if choice:
                    selected_tasks[limit].add(task)
        batch.clear()

    try:
        for index, row in enumerate(sampled_rows(source, seed, max_rows, configs)):
            if deadline and time.monotonic() >= deadline:
                stop_reason = 'time_limit'
                break
            counts['rows_seen'] += 1
            if len(schemas) < 3:
                schemas.append(row_schema(row))
            if source.key == 'swe_success':
                value = row.get('target')
                label = type(value).__name__ + ':' + str(value)[:40]
                target_types[label] += 1
            case, status, events = audited_adapter(traces.ADAPTERS[source.adapter], source, row, index)
            counts['adapter_' + status] += 1
            if case is None:
                # Last return in the adapter identifies the rejection; helper/exception trail adds context.
                returns = [e for e in events if e['event'] == 'return_none']
                root = returns[-1] if returns else (events[-1] if events else {'function': 'unknown', 'line': 0})
                key = f"{root['function']}:{root['line']}"
                reasons[key] += 1
                if key not in evidence:
                    evidence[key] = {'events': events[-8:], 'schema': row_schema(row)}
            else:
                batch.append(case)
            if len(batch) >= batch_size:
                flush()
            if counts['rows_seen'] % 250 == 0:
                print(source.key, dict(counts), flush=True)
        flush()
    except Exception as exc:
        # A source access/tokenizer failure is not source exhaustion. Save other sources' reports.
        stop_reason = 'source_error'
        evidence['source_error'] = {'exception_type': type(exc).__name__,
                                    'frames': [{'file': Path(f.filename).name, 'line': f.lineno,
                                                'function': f.name}
                                               for f in traceback.extract_tb(exc.__traceback__)[-6:]],
                                    'hint': 'Inspect access, schema, or tokenization; no capacity inference.'}
    quantiles = {str(q): int(np.percentile(full_lengths, q)) for q in (50, 90, 95, 99, 100)} if full_lengths else {}
    return {'source': asdict(source), 'stop_reason': stop_reason, 'configs_rows_seen': dict(configs),
            'counts': dict(counts), 'adapter_rejection_sites': dict(reasons),
            'rejection_evidence': evidence, 'schema_examples': schemas,
            'swe_target_types': dict(target_types), 'full_rendered_token_percentiles': quantiles,
            'length_inventory_before_task_ownership': {str(k): dict(v) for k, v in totals.items()},
            'selection_simulation_on_scanned_rows': {str(k): s.audit() for k, s in selectors.items()}}


def save_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    temp = output / 'diagnosis.json.tmp'
    temp.write_text(json.dumps(report, indent=2) + '\n')
    temp.replace(output / 'diagnosis.json')
    rows = []
    for result in report['sources']:
        for limit, metrics in result['length_inventory_before_task_ownership'].items():
            rows.append({'source': result['source']['key'], 'limit': int(limit),
                         'rows_scanned': result['counts'].get('rows_seen', 0),
                         'stop_reason': result['stop_reason'], **metrics})
    fields = ['source', 'limit', 'rows_scanned', 'stop_reason', 'eligible_records', 'raw_tokens',
              'supervised_tokens', 'over_8k_records', 'over_16k_records',
              'full_through_final_assistant_records', 'partial_prefix_records']
    with (output / 'length_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Corpus diagnosis', '',
             'Observed rows only; sample totals are not full-dataset capacity estimates.',
             'Length inventory precedes task ownership and quota selection. It may include task variants.', '',
             '| Source | Scanned | Adapter rejected/errored | Usable 16K | Usable 32K | Newly usable at 32K | Extended at 32K | Status |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for result in report['sources']:
        c, inventory = result['counts'], result['length_inventory_before_task_ownership']
        lines.append(f"| {result['source']['key']} | {c.get('rows_seen',0)} | "
                     f"{c.get('adapter_rejected',0)+c.get('adapter_raised_exception',0)} | "
                     f"{inventory['16384'].get('eligible_records',0)} | {inventory['32768'].get('eligible_records',0)} | "
                     f"{c.get('rescued_at_32k_records',0)} | {c.get('extended_at_32k_records',0)} | {result['stop_reason']} |")
    lines += ['', '## Interpretation', '',
              '- Adapter failures happen before length checks; 32K cannot recover those rows.',
              '- Newly usable: no valid 16K view, but a valid 32K view. Extended: both fit a prefix, and 32K retains more history.',
              '- Pivot must retain its supplied final target; earlier assistant context remains masked.',
              '- Compare supervised tokens as well as context tokens: longer tool observations alone are not more target labels.',
              '- Selection simulations retain the existing disjoint task ownership and quota policy independently at each limit.',
              '- A 32K data benefit does not establish TPU feasibility. A separate 32K memory/throughput and position-scaling check is needed before changing training.',
              '- Read diagnosis.json for rejection code locations, caught exception types, schema evidence, and per-limit quota shortfalls.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')


def run(tokenizer, output, *, source_keys=None, max_rows=2000, batch_size=16, seed=1701,
        max_seconds=3600, provenance=None):
    if max_rows < 0 or batch_size < 1 or max_seconds <= 0:
        raise ValueError('Use max_rows >= 0, batch_size >= 1, and max_seconds > 0')
    sources = list(traces.REASONING_SOURCES + traces.AGENT_SOURCES)
    if source_keys is not None:
        by_key = {s.key: s for s in sources}
        sources = [by_key[key] for key in source_keys]
    report = {'diagnostic_format': 1, 'settings': {'max_rows_per_source': max_rows,
              'batch_size': batch_size, 'seed': seed, 'limits': LIMITS, 'max_seconds': max_seconds},
              'provenance': provenance or {}, 'sampling_note': 'Bounded streaming shuffle, not a uniform dataset sample. OpenResearcher budget is split across configs.',
              'sources': [], 'complete': False}
    deadline = time.monotonic() + max_seconds
    for source in sources:
        if time.monotonic() >= deadline:
            break
        print('Diagnosing', source.key, flush=True)
        result = diagnose_source(source, tokenizer, max_rows=max_rows, batch_size=batch_size,
                                 seed=seed, deadline=deadline)
        report['sources'].append(result)
        save_report(report, output)
        print(source.key, result['stop_reason'], result['adapter_rejection_sites'], flush=True)
    report['unscanned_sources'] = [s.key for s in sources[len(report['sources']):]]
    report['complete'] = not report['unscanned_sources'] and all(
        s['stop_reason'] in ('sample_finished', 'source_exhausted') for s in report['sources'])
    save_report(report, output)
    return report
