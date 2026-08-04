import pytest
import torch

from src.models.cache_utils import prepare_prompt, resolve_model_torch_dtype


class ChatTokenizerFixture:
    chat_template = "fixture-template"

    def __init__(self):
        self.template_calls = []
        self.tokenizer_calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return f"<user>{messages[0]['content']}<assistant>"

    def __call__(self, text, **kwargs):
        self.tokenizer_calls.append((text, kwargs))
        input_ids = torch.tensor([[4, 5, 6]], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class RawTokenizerFixture:
    chat_template = None

    def __init__(self):
        self.tokenizer_calls = []

    def __call__(self, text, **kwargs):
        self.tokenizer_calls.append((text, kwargs))
        return {"input_ids": torch.tensor([[7, 8]], dtype=torch.long)}


@pytest.mark.parametrize("serialization", ["auto", "chat"])
def test_prepare_prompt_applies_chat_template_with_generation_marker(serialization):
    tokenizer = ChatTokenizerFixture()

    prepared = prepare_prompt(
        tokenizer=tokenizer,
        prompt="Solve this problem.",
        max_length=128,
        serialization=serialization,
    )

    assert prepared.serialization == "chat"
    assert prepared.raw_text == "Solve this problem."
    assert prepared.rendered_text == "<user>Solve this problem.<assistant>"
    assert prepared.input_ids.tolist() == [[4, 5, 6]]
    assert tokenizer.template_calls == [
        {
            "messages": [{"role": "user", "content": "Solve this problem."}],
            "tokenize": False,
            "add_generation_prompt": True,
        }
    ]
    assert tokenizer.tokenizer_calls[0][1]["add_special_tokens"] is False
    assert tokenizer.tokenizer_calls[0][1]["truncation"] is True
    assert tokenizer.tokenizer_calls[0][1]["max_length"] == 128


def test_prepare_prompt_raw_mode_preserves_text_and_special_token_handling():
    tokenizer = RawTokenizerFixture()

    prepared = prepare_prompt(
        tokenizer=tokenizer,
        prompt="raw prompt",
        serialization="auto",
    )

    assert prepared.serialization == "raw"
    assert prepared.rendered_text == "raw prompt"
    assert tokenizer.tokenizer_calls[0][1]["add_special_tokens"] is True


def test_prepare_prompt_rejects_chat_mode_without_template():
    with pytest.raises(ValueError, match="tokenizer has no chat template"):
        prepare_prompt(
            tokenizer=RawTokenizerFixture(),
            prompt="prompt",
            serialization="chat",
        )


def test_prepare_prompt_rejects_unknown_serialization():
    with pytest.raises(ValueError, match="must be `auto`, `raw`, or `chat`"):
        prepare_prompt(
            tokenizer=RawTokenizerFixture(),
            prompt="prompt",
            serialization="conversation",
        )


def test_qwen_auto_dtype_uses_float32_on_t4_class_gpu():
    assert resolve_model_torch_dtype(
        "Qwen/Qwen2.5-1.5B-Instruct",
        "auto",
        device=torch.device("cuda"),
        cuda_capability=(7, 5),
    ) == torch.float32


def test_qwen_auto_dtype_uses_bfloat16_on_ampere_or_newer():
    assert resolve_model_torch_dtype(
        "Qwen/Qwen2-7B-Instruct",
        "auto",
        device=torch.device("cuda"),
        cuda_capability=(8, 0),
    ) == torch.bfloat16


def test_qwen_float16_is_rejected_on_t4_class_gpu():
    with pytest.raises(ValueError, match="numerically unstable"):
        resolve_model_torch_dtype(
            "Qwen/Qwen2.5-1.5B-Instruct",
            "float16",
            device=torch.device("cuda"),
            cuda_capability=(7, 5),
        )


def test_non_qwen_float16_remains_available_on_t4_class_gpu():
    assert resolve_model_torch_dtype(
        "meta-llama/Llama-3.1-8B-Instruct",
        "float16",
        device=torch.device("cuda"),
        cuda_capability=(7, 5),
    ) == torch.float16
