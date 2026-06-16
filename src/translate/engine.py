"""Translation engine with MarianMT 2-hop.

Direct hop when source→target is available (e.g. opus-mt-en-id).
2-hop via English when direct pair unavailable (e.g. ja→id → ja→en→id).
"""

from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODELS_CACHE: dict[str, tuple[Any, Any]] = {}


class TranslationEngine:
    def __init__(self, use_gpu: bool | None = None):
        self.use_gpu = use_gpu
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"[Translate] Device: {self.device}")

    def _load_model(self, source_lang: str, target_lang: str) -> tuple[Any, Any]:
        cache_key = f"{source_lang}-{target_lang}"
        if cache_key in MODELS_CACHE:
            return MODELS_CACHE[cache_key]

        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        print(f"[Translate] Loading {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = model.to(self.device)
        MODELS_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model

    def _translate_batch(
        self, texts: list[str], tokenizer: Any, model: Any
    ) -> list[str]:
        results = []
        for text in texts:
            try:
                if text and len(text) < 512:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_length=512)
                    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    results.append(translated)
                else:
                    results.append(text)
            except Exception as e:
                print(f"[Translate] Error '{text[:30]}...': {e}")
                results.append(text)
        return results

    def translate(
        self,
        texts: list[str],
        target_lang: str = "en",
        source_lang: str | None = None,
    ) -> list[dict]:
        if not texts:
            return []

        if source_lang is None:
            source_lang = "en"
        if source_lang == target_lang:
            return [{"translation": t} for t in texts]

        # Try direct translation first (source → target)
        try:
            tokenizer, model = self._load_model(source_lang, target_lang)
            translations = self._translate_batch(texts, tokenizer, model)
            return [{"translation": t} for t in translations]
        except Exception:
            pass  # Direct unavailable, fall through to 2-hop

        # 2-hop: source → en → target
        try:
            tokenizer_en, model_en = self._load_model(source_lang, "en")
            en_texts = self._translate_batch(texts, tokenizer_en, model_en)
        except Exception as e:
            return [{"translation": f"[No model: {source_lang}→en→{target_lang}]"}]

        if target_lang == "en":
            return [{"translation": t} for t in en_texts]

        try:
            tokenizer_tgt, model_tgt = self._load_model("en", target_lang)
            final = self._translate_batch(en_texts, tokenizer_tgt, model_tgt)
        except Exception as e:
            return [{"translation": f"[No model: {source_lang}→en→{target_lang}]"}]

        return [{"translation": t} for t in final]
