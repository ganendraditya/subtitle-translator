"""Translation engine with MarianMT (primary) + NLLB fallback.

Language codes: ISO 639-1 (en, id, ja, zh, ko, etc.)
NLLB uses FLORES codes internally via ISO_TO_FLORES mapping.
"""

from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODELS_CACHE: dict[str, tuple[Any, Any]] = {}

ISO_TO_FLORES = {
    "en": "eng_Latn",
    "id": "ind_Latn",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "ar": "arb_Arab",
}


class TranslationEngine:
    def __init__(self, model_dir: str = "models", use_gpu: bool | None = None):
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"[Translate] Device: {self.device}")

    def detect_language(self, text: str) -> str:
        return "en"

    def _get_translator(self, source_lang: str, target_lang: str) -> tuple[Any, Any]:
        cache_key = f"{source_lang}-{target_lang}"
        if cache_key in MODELS_CACHE:
            return MODELS_CACHE[cache_key]

        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        try:
            print(f"[Translate] Loading MarianMT: {model_name} ...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model = model.to(self.device)
            MODELS_CACHE[cache_key] = (tokenizer, model)
            return tokenizer, model
        except Exception as e:
            print(f"[Translate] MarianMT failed: {e}")
            print("[Translate] Falling back to NLLB...")
            try:
                flores_src = ISO_TO_FLORES.get(source_lang, source_lang)
                flores_tgt = ISO_TO_FLORES.get(target_lang, target_lang)
                model_name = "facebook/nllb-200-distilled-600M"
                print(f"[Translate] Loading NLLB: {model_name} ...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, src_lang=flores_src
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                model = model.to(self.device)
                forced_bos_id = tokenizer.convert_tokens_to_ids(flores_tgt)
                MODELS_CACHE[cache_key] = (tokenizer, model, forced_bos_id)
                return tokenizer, model, forced_bos_id
            except Exception as nllb_e:
                print(f"[Translate] NLLB failed: {nllb_e}")
                raise RuntimeError("No translation model available.") from nllb_e

    def translate(
        self,
        texts: list[str],
        target_lang: str = "en",
        source_lang: str | None = None,
    ) -> list[dict]:
        if not texts:
            return []

        if source_lang is None:
            source_lang = self.detect_language(texts[0])
        if source_lang == target_lang:
            return [{"translation": text} for text in texts]

        tokenizer, model, *extra = self._get_translator(source_lang, target_lang)
        forced_bos_id = extra[0] if extra else None
        is_nllb = forced_bos_id is not None

        results = []
        for text in texts:
            try:
                if text and len(text) < 512:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    gen_kwargs = dict(max_length=512)
                    if is_nllb:
                        gen_kwargs["forced_bos_token_id"] = forced_bos_id
                    with torch.no_grad():
                        outputs = model.generate(**inputs, **gen_kwargs)
                    translated = tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    )
                    results.append({
                        "original": text,
                        "translation": translated,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                    })
                else:
                    results.append({"original": text, "translation": "[Skipped: too long]"})
            except Exception as e:
                print(f"[Translate] Error for '{text[:30]}...': {e}")
                results.append({"original": text, "translation": "[Error]"})
        return results
