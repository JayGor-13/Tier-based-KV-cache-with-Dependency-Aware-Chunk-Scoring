"""Paper-aligned prompting for GSM8K evaluations."""

from __future__ import annotations

from copy import deepcopy


DEFAULT_PROTOCOL = "default"
CHUNKKV_GSM8K_8SHOT_PROTOCOL = "chunkkv_gsm8k_8shot"
CHUNKKV_GSM8K_PROMPT_VERSION = 1

_PROTOCOL_ALIASES = {
    DEFAULT_PROTOCOL: DEFAULT_PROTOCOL,
    "legacy": DEFAULT_PROTOCOL,
    "chunkkv_gsm8k": CHUNKKV_GSM8K_8SHOT_PROTOCOL,
    "gsm8k_chunkkv": CHUNKKV_GSM8K_8SHOT_PROTOCOL,
    CHUNKKV_GSM8K_8SHOT_PROTOCOL: CHUNKKV_GSM8K_8SHOT_PROTOCOL,
}

# Appendix G, Table 30 of the ChunkKV paper. Keep wording and arithmetic style
# stable because this prompt is part of the experimental protocol.
CHUNKKV_GSM8K_EXEMPLARS: tuple[tuple[str, str], ...] = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in "
        "the grove today. After they are done, there will be 21 trees. How "
        "many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some "
        "more were planted. So there must have been 21 - 15 = 6. The answer "
        "is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how "
        "many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The "
        "answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how "
        "many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total "
        "they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The "
        "answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has "
        "12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to "
        "Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. How many computers "
        "are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is "
        "29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. How many golf balls did he have at the "
        "end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he "
        "had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. "
        "The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money "
        "does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = "
        "15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The "
        "answer is 8.",
    ),
)


def canonicalize_protocol(protocol: str | None) -> str:
    """Return a stable protocol identifier or reject an unknown protocol."""
    normalized = str(protocol or DEFAULT_PROTOCOL).strip().lower().replace("-", "_")
    try:
        return _PROTOCOL_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(set(_PROTOCOL_ALIASES.values())))
        raise ValueError(
            f"Unsupported evaluation protocol `{protocol}`. Supported: {supported}."
        ) from exc


def validate_protocol_for_adapter(protocol: str, adapter: str | None) -> None:
    """Reject protocol and dataset combinations with incompatible semantics."""
    if protocol == CHUNKKV_GSM8K_8SHOT_PROTOCOL and adapter != "gsm8k":
        raise ValueError(
            f"Protocol `{CHUNKKV_GSM8K_8SHOT_PROTOCOL}` requires adapter `gsm8k`."
        )


def build_chunkkv_gsm8k_prompt(question: str) -> str:
    """Render the eight CoT exemplars from ChunkKV Appendix G plus a query."""
    sections = [
        f"Question: {example_question}\n{example_answer}"
        for example_question, example_answer in CHUNKKV_GSM8K_EXEMPLARS
    ]
    sections.append(f"Question: {str(question).strip()}")
    return "\n".join(sections) + "\n"


def protocol_metadata(protocol: str | None) -> dict:
    """Return immutable experiment metadata for a canonical protocol."""
    canonical = canonicalize_protocol(protocol)
    if canonical == CHUNKKV_GSM8K_8SHOT_PROTOCOL:
        metadata = {
            "protocol": canonical,
            "prompt_version": CHUNKKV_GSM8K_PROMPT_VERSION,
            "task": "gsm8k",
            "dataset": "openai/gsm8k",
            "config": "main",
            "split": "test",
            "shots": len(CHUNKKV_GSM8K_EXEMPLARS),
            "prompt_source": "ChunkKV Appendix G, Table 30",
            "answer_contract": "The answer is <number>.",
            "judge": "gsm8k_final_numeric_exact_match",
            "judge_version": 1,
        }
    else:
        metadata = {
            "protocol": DEFAULT_PROTOCOL,
            "prompt_version": 1,
            "task": "dataset_adapter_default",
        }
    return deepcopy(metadata)


__all__ = [
    "CHUNKKV_GSM8K_8SHOT_PROTOCOL",
    "CHUNKKV_GSM8K_EXEMPLARS",
    "CHUNKKV_GSM8K_PROMPT_VERSION",
    "DEFAULT_PROTOCOL",
    "build_chunkkv_gsm8k_prompt",
    "canonicalize_protocol",
    "protocol_metadata",
    "validate_protocol_for_adapter",
]
