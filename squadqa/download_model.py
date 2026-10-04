"""Download pretrained model files into the Hugging Face cache ($HF_HOME).

Run from the repo root:
    python -m squadqa.download_model                         # every model below
    python -m squadqa.download_model microsoft/deberta-v3-base [--tokenizer-only]

Each model is pinned to a commit of its Hugging Face repo, and each file has a
SHA-256; a mismatch raises an error. Only the files transformers needs are
fetched: the weights, config and tokenizer, not the TensorFlow, ONNX or Rust
weights or the ELECTRA generator. Files already in the cache are not downloaded
again. Load with the same pin, so nothing newer is fetched later, and with an
explicit dtype: transformers 5 otherwise loads the stored dtype, which for
deberta-v3-base and -large is fp16 (wrong for fp32 master weights in training);
ModernBERT is stored in fp32. DeBERTa-v3 base and large share one tokenizer
(same spm.model), and so do ModernBERT base and large.

    from squadqa.download_model import MODELS
    AutoModel.from_pretrained(name, revision=MODELS[name]["revision"], dtype=torch.float32)

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
    # Development reader (plan §12.3); pinned 2026-10-04.
    "microsoft/deberta-v3-base": {
        "revision": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
        "files": {
            "config.json": "649f6a1ec33c6bdd9a6486d5c66019d461139e54957073eafe9bbc2d34c75b0b",
            "tokenizer_config.json": "3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d",
            "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
            "pytorch_model.bin": "691d48a2800b926a19e3051def466fc2cca4f59a15e42ce4a0cf7f1b380b5e33",
        },
    },
    # Separate verifier (E2): base for development, large for results; pinned 2026-10-04.
    "answerdotai/ModernBERT-base": {
        "revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "files": {
            "config.json": "1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f",
            "tokenizer.json": "9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1",
            "tokenizer_config.json": "3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2",
            "special_tokens_map.json": "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049",
            "model.safetensors": "340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9",
        },
    },
    "answerdotai/ModernBERT-large": {
        "revision": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
        "files": {
            "config.json": "55dca68059e98f0debf8eb63ab355c451c65b1f5d90fdca4754113fdaadd2cca",
            "tokenizer.json": "9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1",
            "tokenizer_config.json": "3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2",
            "special_tokens_map.json": "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049",
            "model.safetensors": "44510fec5d3a81a1877f225637b869495f18e55f6f23a09abb9be0acc030295f",
        },
    },
}
WEIGHTS = {"pytorch_model.bin", "model.safetensors"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download pinned model files into $HF_HOME.")
    parser.add_argument("models", nargs="*", help=f"default: all of {sorted(MODELS)}")
    parser.add_argument("--tokenizer-only", action="store_true", help="skip the weights (e.g. for CPU tests)")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.models) - set(MODELS))
    if unknown:
        raise KeyError(f"not pinned: {unknown}; choose from {sorted(MODELS)}")
    load_dotenv()
    if not os.environ.get("HF_HOME"):
        raise RuntimeError("HF_HOME is not set. Add it to .env (see .env.example).")
    # Imported after HF_HOME is set: huggingface_hub reads the cache path at import.
    from huggingface_hub import hf_hub_download

    for name in args.models or MODELS:
        model = MODELS[name]
        for filename, sha256 in model["files"].items():
            if args.tokenizer_only and filename in WEIGHTS:
                continue
            path = hf_hub_download(name, filename, revision=model["revision"])
            got = sha256_file(path)
            if got != sha256:
                raise RuntimeError(f"SHA-256 mismatch for {name}/{filename}: expected {sha256}, got {got}")
            print(f"ok {name}/{filename} -> {path}")


if __name__ == "__main__":
    main()
