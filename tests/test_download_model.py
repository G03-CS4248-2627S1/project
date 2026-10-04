import re

from squadqa.download_model import MODELS, WEIGHTS


def test_every_pin_is_a_commit_with_hashed_files():
    for name, model in MODELS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", model["revision"]), name
        assert "config.json" in model["files"] and WEIGHTS & set(model["files"]), name
        for filename, sha in model["files"].items():
            assert re.fullmatch(r"[0-9a-f]{64}", sha), (name, filename)


def test_development_and_result_models_are_pinned():
    assert {"microsoft/deberta-v3-large", "microsoft/deberta-v3-base",
            "answerdotai/ModernBERT-base", "answerdotai/ModernBERT-large"} <= set(MODELS)


def test_deberta_base_and_large_share_the_tokenizer():
    base, large = MODELS["microsoft/deberta-v3-base"]["files"], MODELS["microsoft/deberta-v3-large"]["files"]
    assert base["spm.model"] == large["spm.model"]
