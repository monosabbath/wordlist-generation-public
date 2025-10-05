import torch
import xgrammar as xgr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from trie import PrefixAllowedTokens
from vllm import load_words
import click

@click.command()
@click.option("--words-path", default="en.txt")
@click.option("--n-words", default=5000)
def main(words_path, n_words):
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("using device:", device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    words = load_words(words_path, n_words)

    while True:
        system_msg = "You are a helpful assistant that uses only simple words."
        if words_path == "es.txt":
            system_msg = "Eres un asistente útil que utiliza sólo palabras sencillas."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": input("prompt: ")},
        ]
        texts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer(texts, return_tensors="pt").to(model.device)
        prompt_len = len(model_inputs[0])
        prefix_allowed_tokens_fn = PrefixAllowedTokens(
            words=words,
            prompt_len=prompt_len,
            tokenizer=tokenizer,
        )
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=150,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=50,
            do_sample=True,
            repetition_penalty=1.5,
            length_penalty=2.0,
        )
        result = tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True)
        print("generation:", result)

if __name__ == "__main__":
    main()
