from enum import StrEnum, auto

import click
import httpx

from common import GenerateRequest

qwen_url = "https://comic-boxes-phrase-successful.trycloudflare.com"
gemma_url = "https://cigarette-charges-competitors-throws.trycloudflare.com"


class Lang(StrEnum):
    EN = auto()
    ES = auto()


class Model(StrEnum):
    QWEN = auto()
    GEMMA = auto()


@click.command()
@click.option("--model", type=click.Choice(Model, case_sensitive=False), default="qwen")
@click.option("--lang", type=click.Choice(Lang, case_sensitive=False), default="en")
@click.option("--n-words", default=5000)
@click.option("--max-new-tokens", type=int)
@click.option("--num-beams", type=int)
@click.option("--repetition-penalty", type=float)
@click.option("--length-penalty", type=float)
def main(
    model, lang, n_words, max_new_tokens, num_beams, repetition_penalty, length_penalty
):
    if model == Model.QWEN:
        url = qwen_url
    else:
        url = gemma_url
    while True:
        kwargs = dict(
            prompt=input("prompt: "),
            vocab_lang=lang,
            vocab_n_words=n_words,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )
        for k in list(kwargs):
            if kwargs[k] is None:
                del kwargs[k]
        request = GenerateRequest(**kwargs)
        r = httpx.post(
            url + "/generate",
            params={"token": "my-secret-token-structured-generation"},
            json=request.model_dump(),
            timeout=60,
        )
        print("answer:", r.json())


if __name__ == "__main__":
    main()
