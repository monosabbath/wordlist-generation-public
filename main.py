import tomllib

import click
from openai import OpenAI


def load_words(wordlist_path, n_words):
    with open(wordlist_path) as fin:
        words = [word.strip().lower() for word in fin]
    words = words[:n_words]
    return words


def build_grammar(words):
    grammar_lines = [
        r"root ::= sep? word (sep word)* sep?",
        r"sep ::= (ws | punc)+",
        r"ws ::= [ \n]",
        r"punc ::= [,.;:!?()'¿¡«»—" + '"' + "-]",
    ]
    word_clause = " | ".join(
        "[" + word[0].upper() + word[0].lower() + "] \"" + word[1:] + '"'
        for word in words
    )
    grammar_lines.append(f"word ::= {word_clause}")
    return "\n".join(grammar_lines)


@click.command()
@click.option("--n-words", default=5000)
@click.option("--wordlist", default="en.txt")
@click.option("--server", default="qwen-server.toml")
def main(n_words, wordlist, server):
    with open(server, "rb") as fin:
        server = tomllib.load(fin)
    client = OpenAI(base_url=server["base_url"], api_key=server["api_key"])
    words = load_words(wordlist, n_words)
    system_msg = "You are a helpful assistant that uses only simple words."
    if wordlist == "es.txt":
        system_msg = "Eres un asistente útil que utiliza sólo palabras sencillas."
    messages = [
        {"role": "system", "content": system_msg},
    ]
    grammar = build_grammar(words)
    while True:
        user_message = input("user: ")
        messages.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model="",
            messages=messages,
            temperature=server["temperature"],
            extra_body={
                "guided_grammar": grammar,
                "top_p": server["top_p"],
                "top_k": server["top_k"],
                "min_p": server["min_p"],
                "repetition_penalty": server["repetition_penalty"],
                "length_penalty": server["length_penalty"],
                "max_tokens": server["max_tokens"],
            },
        )
        assistant_message = response.choices[0].message.content
        print("assistant:", assistant_message)
        messages.append({"role": "assistant", "content": assistant_message})


if __name__ == "__main__":
    main()
