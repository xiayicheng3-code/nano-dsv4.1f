"""Deterministic assistant-only crops from the existing packed SFT corpus."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
from nano_dsv41f.portable_bundle import sha256_file


class SFTDraftData:
    def __init__(self, root, *, tokenizer_sha256, block_size, seq_len=512, anchors=8, seed=1701):
        root = Path(root)
        matches = list(root.rglob('posttrain_manifest.json'))
        if len(matches) != 1:
            raise ValueError('Set CORPUS_DIR to exactly one prepared SFT corpus')
        self.root = matches[0].parent
        self.manifest = json.loads(matches[0].read_text())
        if self.manifest.get('format') != 'nano-dsv41f-posttrain-v3' or not self.manifest.get('complete'):
            raise ValueError('A complete format-v3 SFT corpus is required')
        if self.manifest['identity']['tokenizer_sha256'] != tokenizer_sha256 or sha256_file(self.root/'tokenizer.json') != tokenizer_sha256:
            raise ValueError('Corpus tokenizer differs from the frozen model')
        if block_size < 1 or seq_len < block_size+2 or anchors < 1:
            raise ValueError('Invalid crop length, draft block size, or anchor count')
        self.identity = sha256_file(matches[0])
        self.block_size, self.seq_len, self.anchors, self.seed = block_size, seq_len, anchors, seed
        self.catalog = {'train': [], 'validation': []}
        self.cache = OrderedDict()
        for source in self.manifest['sources']:
            key = source['source']['key']
            for split in self.catalog:
                for info in source['views']['sft'][split]['shards']:
                    path = self.root / 'sft' / key / split / info['file']
                    if not path.resolve().is_relative_to(self.root.resolve()):
                        raise ValueError('Invalid corpus shard path')
                    self.catalog[split].append((path, info, source['pool']))
        if any(not entries for entries in self.catalog.values()):
            raise ValueError('Both SFT train and held-out validation shards are required')
        train_hashes = {info['sha256'] for _, info, _ in self.catalog['train']}
        if any(info['sha256'] in train_hashes for _, info, _ in self.catalog['validation']):
            raise ValueError('Train and validation contain an identical shard')

    def _read(self, path, info):
        if path not in self.cache:
            if path.stat().st_size != info['bytes'] or sha256_file(path) != info['sha256']:
                raise ValueError(f'SFT shard checksum mismatch: {path.name}')
            with np.load(path, allow_pickle=False) as data:
                required = ('input_ids', 'segment_ids', 'token_mask', 'sft_loss_mask')
                if any(k not in data for k in required):
                    raise ValueError('SFT shards must retain explicit assistant masks')
                arrays = {k: data[k] for k in required}
            shape = arrays['input_ids'].shape
            if len(shape) != 2 or shape[0] != info['rows'] or any(v.shape != shape for v in arrays.values()):
                raise ValueError('SFT shard shape mismatch')
            if arrays['input_ids'].dtype.kind not in 'iu' or arrays['segment_ids'].dtype.kind not in 'iu':
                raise ValueError('Token and segment IDs must be integers')
            if np.any(arrays['sft_loss_mask'].astype(bool) & ~arrays['token_mask'].astype(bool)):
                raise ValueError('Assistant targets cannot include padding')
            self.cache[path] = arrays
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        return self.cache[path]

    def sample(self, index, *, split='train'):
        # Index-derived RNG makes training and resume independent of validation calls.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index, int(split == 'validation')]))
        entries = self.catalog[split]
        mix = self.manifest['pool_mix']['sft']
        pools = sorted({pool for _, _, pool in entries})
        weights = np.asarray([mix[p] for p in pools], dtype=float)
        weights /= weights.sum()
        for _ in range(200):
            pool = rng.choice(pools, p=weights)
            choices = [(p, i) for p, i, k in entries if k == pool]
            sizes = np.asarray([info['rows'] for _, info in choices], dtype=float)
            path, info = choices[rng.choice(len(choices), p=sizes/sizes.sum())]
            data = self._read(path, info)
            row = int(rng.integers(info['rows']))
            ids = data['input_ids'][row]
            seg = data['segment_ids'][row]
            mask = data['token_mask'][row].astype(bool)
            targets = data['sft_loss_mask'][row].astype(bool)
            # Compare all adjacent segment IDs, not just endpoints.
            valid = mask[:-self.block_size].copy()
            for offset in range(1, self.block_size+1):
                valid &= mask[offset:len(mask)-self.block_size+offset]
                valid &= targets[offset:len(mask)-self.block_size+offset]
                valid &= seg[:-self.block_size] == seg[offset:len(mask)-self.block_size+offset]
            eligible = np.flatnonzero(valid)
            if not len(eligible):
                continue
            primary = int(rng.choice(eligible))
            lo = primary
            while lo > 0 and mask[lo-1] and seg[lo-1] == seg[primary]:
                lo -= 1
            hi = primary+self.block_size+1
            while hi < len(ids) and mask[hi] and seg[hi] == seg[primary]:
                hi += 1
            end = min(hi, primary+self.block_size+1+self.seq_len//4)
            start = max(lo, end-self.seq_len)
            eligible = eligible[(eligible >= start) & (eligible+self.block_size < end)]
            selected = np.sort(rng.choice(eligible, size=min(self.anchors, len(eligible)), replace=False)) - start
            return dict(input_ids=np.array(ids[start:end], dtype=np.int64), anchors=selected,
                source=path.relative_to(self.root).as_posix(), row=row, start=start, end=end, pool=pool)
        raise ValueError(f'No complete assistant draft blocks found in {split}; check corpus masks')
