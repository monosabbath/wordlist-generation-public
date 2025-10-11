from enum import StrEnum, auto

import click
import httpx

from common import GenerateRequest

url = "https://comic-boxes-phrase-successful.trycloudflare.com"


class Lang(StrEnum):
    EN = auto()
    ES = auto()


@click.command()
@click.argument("prompt")
@click.option("--lang", type=click.Choice(Lang, case_sensitive=False), default="en")
@click.option("--n-words", default=5000)
@click.option("--max-new-tokens", type=int)
@click.option("--num-beams", type=int)
@click.option("--repetition-penalty", type=float)
@click.option("--length-penalty", type=float)
def main(
    prompt, lang, n_words, max_new_tokens, num_beams, repetition_penalty, length_penalty
):
    kwargs = dict(
        prompt=prompt,
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
    )
    print(r.text)


if __name__ == "__main__":
    main()
