from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

# DeepSeek V4.1 prompt spellings. IDs are nano-specific except BOS/EOS/PAD,
# which intentionally follow the released V4.1 config contract (0/1/2).
BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
PAD_TOKEN = "<｜pad｜>"

SYSTEM_TOKEN = "<｜System｜>"
USER_TOKEN = "<｜User｜>"
ASSISTANT_TOKEN = "<｜Assistant｜>"
LATEST_REMINDER_TOKEN = "<｜latest_reminder｜>"
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"
DSML_TOKEN = "｜DSML｜"
DSML_CALLS_TOKEN = "<｜DSML｜ calls>"
DSML_INVOKE_TOKEN = "<｜DSML｜ invoke>"
DSML_PARAMETER_TOKEN = "<｜DSML｜ parameter>"
TOOL_RESULT_START_TOKEN = "<tool_result>"
TOOL_RESULT_END_TOKEN = "</tool_result>"
IMAGE_PLACEHOLDER_TOKEN = "<｜deepseek_image｜>"

ACTION_TOKEN = "<｜action｜>"
QUERY_TOKEN = "<｜query｜>"
AUTHORITY_TOKEN = "<｜authority｜>"
DOMAIN_TOKEN = "<｜domain｜>"
TITLE_TOKEN = "<｜title｜>"
READ_URL_TOKEN = "<｜read_url｜>"

# Production V4.1 uses a dedicated high-ID placeholder for DSpark. The nano
# tokenizer reserves its own stable placeholder instead of reusing BOS/PAD.
DSPARK_NOISE_TOKEN = "<｜dspark_noise｜>"

SPECIAL_TOKENS: tuple[str, ...] = (
    BOS_TOKEN,                    # 0
    EOS_TOKEN,                    # 1
    PAD_TOKEN,                    # 2
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    LATEST_REMINDER_TOKEN,
    THINK_START_TOKEN,
    THINK_END_TOKEN,
    DSML_TOKEN,
    DSML_CALLS_TOKEN,
    DSML_INVOKE_TOKEN,
    DSML_PARAMETER_TOKEN,
    TOOL_RESULT_START_TOKEN,
    TOOL_RESULT_END_TOKEN,
    IMAGE_PLACEHOLDER_TOKEN,
    ACTION_TOKEN,
    QUERY_TOKEN,
    AUTHORITY_TOKEN,
    DOMAIN_TOKEN,
    TITLE_TOKEN,
    READ_URL_TOKEN,
    DSPARK_NOISE_TOKEN,
)

BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 2
DSPARK_NOISE_TOKEN_ID = SPECIAL_TOKENS.index(DSPARK_NOISE_TOKEN)

TASK_TOKENS = {
    "action": ACTION_TOKEN,
    "query": QUERY_TOKEN,
    "authority": AUTHORITY_TOKEN,
    "domain": DOMAIN_TOKEN,
    "title": TITLE_TOKEN,
    "read_url": READ_URL_TOKEN,
}


@dataclass(frozen=True)
class TokenizerContract:
    vocab_size: int
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS

    def __post_init__(self) -> None:
        if self.vocab_size < len(self.special_tokens):
            raise ValueError(
                f"vocab_size={self.vocab_size} cannot fit "
                f"{len(self.special_tokens)} reserved special tokens"
            )
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special tokens must be unique")

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: i for i, token in enumerate(self.special_tokens)}

    @property
    def id_to_token(self) -> dict[int, str]:
        return {i: token for i, token in enumerate(self.special_tokens)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "bos_token": BOS_TOKEN,
            "bos_token_id": BOS_TOKEN_ID,
            "eos_token": EOS_TOKEN,
            "eos_token_id": EOS_TOKEN_ID,
            "pad_token": PAD_TOKEN,
            "pad_token_id": PAD_TOKEN_ID,
            "dspark_noise_token": DSPARK_NOISE_TOKEN,
            "dspark_noise_token_id": DSPARK_NOISE_TOKEN_ID,
            "special_tokens": list(self.special_tokens),
            "special_token_ids": self.token_to_id,
            "protocol": "deepseek-v4.1",
            "id_policy": (
                "BOS/EOS/PAD match released V4.1 IDs 0/1/2; all other IDs are "
                "nano-specific but frozen once training starts."
            ),
        }


def nano_v41_tokenizer_contract(vocab_size: int = 32_768) -> TokenizerContract:
    return TokenizerContract(vocab_size=vocab_size)


def validate_model_token_ids(config: Any, contract: TokenizerContract | None = None) -> tuple[str, ...]:
    """Return tokenizer/model ID mismatches without mutating either object."""
    contract = contract or nano_v41_tokenizer_contract(int(config.vocab_size))
    issues: list[str] = []
    if int(config.vocab_size) != contract.vocab_size:
        issues.append(
            f"model vocab_size={config.vocab_size} != tokenizer vocab_size={contract.vocab_size}"
        )
    if getattr(config.engram, "pad_token_id", None) != PAD_TOKEN_ID:
        issues.append(
            f"Engram pad_token_id={config.engram.pad_token_id}; expected {PAD_TOKEN_ID}"
        )
    if getattr(config.dspark, "noise_token_id", None) != DSPARK_NOISE_TOKEN_ID:
        issues.append(
            f"DSpark noise_token_id={config.dspark.noise_token_id}; "
            f"expected reserved ID {DSPARK_NOISE_TOKEN_ID}"
        )
    return tuple(issues)


def apply_tokenizer_contract(config: Any, contract: TokenizerContract | None = None) -> Any:
    """Return a copy of ``config`` whose token-ID-sensitive modules match the contract.

    Call this before initializing the first trainable checkpoint. Changing these IDs after
    Engram/DSpark have seen data would change model semantics.
    """
    contract = contract or nano_v41_tokenizer_contract(int(config.vocab_size))
    if contract.vocab_size != int(config.vocab_size):
        raise ValueError("tokenizer/model vocab sizes must match")
    return replace(
        config,
        engram=replace(config.engram, pad_token_id=PAD_TOKEN_ID),
        dspark=replace(config.dspark, noise_token_id=DSPARK_NOISE_TOKEN_ID),
    )
