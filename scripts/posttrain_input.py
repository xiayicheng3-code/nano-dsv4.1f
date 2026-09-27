"""Checksummed, deterministic, bounded-memory input for 8K midtrain and 16K SFT."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np
from pretrain_checkpoint import digest_file

FORMAT = 'nano-dsv41f-posttrain-v1'


def inspect(root):
    root = Path(root)
    matches = list(root.rglob('posttrain_manifest.json'))
    if len(matches) != 1:
        raise ValueError('Attach exactly one completed posttrain corpus')
    root = matches[0].parent
    manifest = json.loads(matches[0].read_text())
    if manifest.get('format') != FORMAT or not manifest.get('complete'):
        raise ValueError('Incomplete/unsupported posttrain corpus')
    if digest_file(root / 'tokenizer.json') != manifest['identity']['tokenizer_sha256']:
        raise ValueError('Tokenizer checksum mismatch')
    for stage in ('midtrain', 'sft'):
        for split in ('train', 'validation'):
            for shards in catalog(root, manifest, stage, split).values():
                for path, info in shards:
                    if path.stat().st_size != info['bytes'] or digest_file(path) != info['sha256']:
                        raise ValueError(f'Shard checksum mismatch: {path}')
    return root, manifest, hashlib.sha256(matches[0].read_bytes()).hexdigest()


def catalog(root, manifest, stage, split='train'):
    pools = {'reasoning': [], 'agent': []}
    for source in manifest['sources']:
        key = source['source']['key']
        pools[source['pool']].extend(
            (root / stage / key / split / shard['file'], shard)
            for shard in source['views'][stage][split]['shards'])
    if stage == 'midtrain':
        shards = manifest['documents']['shards']
        nval = max(1, len(shards) // 50) if len(shards) > 1 else 0
        chosen = shards[-nval:] if split == 'validation' and nval else ([] if split == 'validation' else shards[:-nval] if nval else shards)
        pools['document'] = [(root / manifest['documents']['path'] / s['file'], s) for s in chosen]
    return pools


class Pool:
    def __init__(self, shards, *, seed, length, batch_rows=4):
        self.length, self.batch_rows = length, batch_rows
        rng = np.random.default_rng(seed)
        self.shards = list(shards)
        rng.shuffle(self.shards)
        self.orders = [rng.permutation(s['rows']) for _, s in self.shards]
        self.ends = np.cumsum([len(o) for o in self.orders])
        self.rows = int(self.ends[-1]) if len(self.ends) else 0
        self.cache = OrderedDict()

    def batch(self, cursor):
        lo = cursor * self.batch_rows
        if lo + self.batch_rows > self.rows:
            raise StopIteration
        rows = []
        for n in range(lo, lo + self.batch_rows):
            shard = int(np.searchsorted(self.ends, n, side='right'))
            index = int(self.orders[shard][n - (int(self.ends[shard - 1]) if shard else 0)])
            if shard not in self.cache:
                with np.load(self.shards[shard][0], allow_pickle=False) as data:
                    self.cache[shard] = {k: data[k] for k in ('input_ids','segment_ids','token_mask')}
                    self.cache[shard]['sft_loss_mask'] = (data['sft_loss_mask'] if 'sft_loss_mask' in data else data['token_mask'])
                if len(self.cache) > 2:
                    self.cache.popitem(last=False)
            rows.append({k: v[index] for k, v in self.cache[shard].items()})
        return {k: np.stack([row[k] for row in rows]) for k in rows[0]}


def batch_counts(batch, length, *, sft=False):
    ids, seg = batch['input_ids'], batch['segment_ids']
    mask, target = batch['token_mask'].astype(bool), batch['sft_loss_mask'].astype(bool)
    if any(v.shape != (4, length) for v in batch.values()):
        raise ValueError('Invalid batch shape')
    if ids.max() >= 32768 or np.any(ids[~mask] != 2):
        raise ValueError('Invalid token/padding IDs')
    if not np.array_equal(seg[:, ::2], seg[:, 1::2]) or np.any(target & ~mask):
        raise ValueError('Invalid segment alignment/target mask')
    valid = mask[:, :-1] & mask[:, 1:] & (seg[:, :-1] == seg[:, 1:])
    if sft:
        valid &= target[:, 1:]
    if not valid.any():
        raise ValueError('Batch has no supervised targets')
    batch['token_mask'], batch['sft_loss_mask'] = mask, target
    return {'real_tokens': int(mask.sum()), 'lm_tokens': int(valid.sum()), 'physical_tokens': int(ids.size)}


def choose_pool(consumed, weights):
    # Token-weighted fair scheduling (one batch granularity), deterministic on restore.
    return min(weights, key=lambda k: (consumed[k] / weights[k], k))


def capacities(manifest, stage):
    result = {'reasoning': 0, 'agent': 0}
    for source in manifest['sources']:
        result[source['pool']] += source['views'][stage]['train'].get('real_tokens', 0)
    if stage == 'midtrain':
        result['document'] = manifest['documents']['real_tokens'] * .97
    return result


def sft_budget(manifest):
    capacity = capacities(manifest, 'sft')
    # Leave headroom for one batch per pool and incomplete last batches.
    return max(0, int(min((capacity[k] - 4 * 16384) / w
                         for k, w in manifest['pool_mix']['sft'].items()) * .98))
