"""Download pretrained model files into the Hugging Face cache ($HF_HOME).

Run from the repo root:  python -m squadqa.download_model

Each model is pinned to a commit of its Hugging Face repo, and each file has a
SHA-256; a mismatch raises an error. Only the files transformers needs are
fetched: the PyTorch weights, config and tokenizer, not the TensorFlow weights
or the ELECTRA generator. Files already in the cache are not downloaded again.
Load with the same pin, so nothing newer is fetched later:

    from squadqa.download_model import MODELS
    AutoModel.from_pretrained(name, revision=MODELS[name]["revision"])

On the SoC cluster, run it in a CPU job rather than on the login node, which
limits each process to 1 GB of virtual memory.
"""

import hashlib
import os

from dotenv import load_dotenv

MODELS = {
    "microsoft/deberta-v3-large": {
        "revision": "64a8c8eab3e352a784c658aef62be1662607476f",
        "files": {
            "config.json": "ddec8b81d079d218ce9e54fc0af5d1d5937d6d53b5d42e70c1f251a1cebc830d",
            "tokenizer_config.json": "3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d",
            "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
            "pytorch_model.bin": "dd5b5d93e2db101aaf281df0ea1216c07ad73620ff59c5b42dccac4bf2eef5b5",
        },
    },
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    load_dotenv()
    if not os.environ.get("HF_HOME"):
        raise RuntimeError("HF_HOME is not set. Add it to .env (see .env.example).")
    # Imported after HF_HOME is set: huggingface_hub reads the cache path at import.
    from huggingface_hub import hf_hub_download

    for name, model in MODELS.items():
        for filename, sha256 in model["files"].items():
            path = hf_hub_download(name, filename, revision=model["revision"])
            got = sha256_file(path)
            if got != sha256:
                raise RuntimeError(f"SHA-256 mismatch for {name}/{filename}: expected {sha256}, got {got}")
            print(f"ok {name}/{filename} -> {path}")


if __name__ == "__main__":
    main()
