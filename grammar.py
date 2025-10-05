import xgrammar as xgr
from transformers import AutoTokenizer, AutoConfig
from vllm import load_words, build_grammar


model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
words = load_words("en.txt", 1000)
grammar = build_grammar(words)
print(grammar)
compiled_grammar = grammar_compiler.compile_grammar(grammar)
#print(compiled_grammar.serialize_json())
grammar_matcher = xgr.GrammarMatcher(compiled_grammar)

while True:
    string = input("string: ")
    tokens = tokenizer.encode(string)
    tokens_str = [tokenizer.decode(x) for x in tokens]
    for t, ts in zip(tokens, tokens_str):
        print(t, ts, grammar_matcher.accept_token(t, debug_print=True))
    grammar_matcher.reset()


