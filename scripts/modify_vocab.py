import sys
from pathlib import Path

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import sentencepiece_pb2

from sentencepiece import SentencePieceProcessor

@torch.no_grad()
def modify_vocab(
    *,
    vocab_file: Path,
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    out_path: Path = Path("checkpoints/lit-llama/tokenizer-modified.model"),
) -> None:
    """
    Modifies the sentencepiece vocab. Specifically, places the new tokens at the end of the sentencepiece vocabulary list.

    Protocol buffer
    https://github.com/google/sentencepiece/issues/121#issuecomment-400362011
    """


    with open(tokenizer_path, "rb") as f:
        model = sentencepiece_pb2.ModelProto()
        assert model.ParseFromString(f.read()) > 0

    vocab_size = len(model.pieces)

    with open(vocab_file, "r") as f:
        new_vocab = f.readlines()

    new_vocab = [vocab.strip() for vocab in new_vocab]

    assert len(new_vocab) <= vocab_size

    offset = vocab_size - len(new_vocab)
    for i in range(len(new_vocab)):
        model.pieces[offset + i].piece = new_vocab[i]

    with open(out_path, "wb") as f:
        f.write(model.SerializeToString())

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(modify_vocab)
    print("Done.")

