from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .chat_protocol import TokenizerContract, nano_v41_tokenizer_contract


def train_byte_bpe_tokenizer(
    texts: Iterable[str],
    output_dir: str | Path,
    *,
    contract: TokenizerContract | None = None,
    min_frequency: int = 2,
) -> Path:
    """Train the frozen ~32K nano tokenizer with DeepSeek V4.1 protocol tokens.

    The tokenizer deliberately does not add BOS/EOS automatically: DeepSeek's prompt
    renderer already emits protocol markers. Byte-level BPE keeps arbitrary code/text
    representable without introducing an UNK token.
    """
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Tokenizer training needs the optional data dependency: "
            "pip install 'nano-dsv41f[data]'"
        ) from exc

    contract = contract or nano_v41_tokenizer_contract()
    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive")

    tokenizer = Tokenizer(BPE(unk_token=None, byte_fallback=True))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=contract.vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=list(contract.special_tokens),
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    actual_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab != contract.vocab_size:
        raise RuntimeError(
            f"trained tokenizer has vocab_size={actual_vocab}, expected {contract.vocab_size}"
        )
    for token, expected_id in contract.token_to_id.items():
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            raise RuntimeError(
                f"special token {token!r} got id={actual_id}; expected frozen id={expected_id}"
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 4096,
        "bos_token": contract.special_tokens[0],
        "eos_token": contract.special_tokens[1],
        "pad_token": contract.special_tokens[2],
        "add_bos_token": False,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "additional_special_tokens": list(contract.special_tokens[3:]),
        "nano_protocol": "deepseek-v4.1",
    }
    (output / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": contract.special_tokens[0],
                "eos_token": contract.special_tokens[1],
                "pad_token": contract.special_tokens[2],
                "additional_special_tokens": list(contract.special_tokens[3:]),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "nano_tokenizer_contract.json").write_text(
        json.dumps(contract.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return tokenizer_path


def iter_text_field(records: Iterable[dict], field: str = "text") -> Iterable[str]:
    for record in records:
        value = record.get(field)
        if isinstance(value, str) and value:
            yield value
