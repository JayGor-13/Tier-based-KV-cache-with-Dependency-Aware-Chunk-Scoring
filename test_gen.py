import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

def test_generation():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")
    
    prompt = "Hello, how are you? I am a large language"
    encoded = tokenizer(prompt, return_tensors="pt")
    
    # Full prefill
    with torch.no_grad():
        outputs = model(**encoded, use_cache=True)
        
    # Get cache
    legacy_cache = outputs.past_key_values
    
    # Evict cache (let's say we keep only first 3 and last 3 tokens)
    seq_len = encoded["input_ids"].shape[1]
    keep_indices = torch.tensor([0, 1, 2, seq_len-3, seq_len-2, seq_len-1], dtype=torch.long)
    
    evicted_cache = DynamicCache()
    for i, (k, v) in enumerate(legacy_cache):
        evicted_cache.update(k[:, :, keep_indices, :], v[:, :, keep_indices, :], layer_idx=i)
        
    # Now generate with the evicted cache
    input_ids = encoded["input_ids"][:, -1:] # Last token
    cache_len = len(keep_indices)
    attention_mask = torch.ones(1, cache_len + 1, dtype=torch.long)
    position_ids = torch.tensor([[seq_len]], dtype=torch.long)
    
    generated_tokens = []
    past_key_values = evicted_cache
    
    for _ in range(10):
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = next_token
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1, dtype=torch.long)], dim=1)
        position_ids = position_ids + 1
        past_key_values = out.past_key_values
        
    print("Generated:", tokenizer.decode(generated_tokens))

if __name__ == "__main__":
    test_generation()
