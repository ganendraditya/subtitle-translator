from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODELS_CACHE: dict[str, tuple[Any, Any]] = {}


class TranslationEngine:
    def __init__(self, model_dir: str = "models", use_gpu: bool | None = None):
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"Using device: {self.device}")

    def detect_language(self, text: str) -> str:
        return "en"

    def _get_translator(self, source_lang: str, target_lang: str) -> tuple[Any, Any]:
        cache_key = f"{source_lang}-{target_lang}"
        if cache_key in MODELS_CACHE:
            return MODELS_CACHE[cache_key]

        # Try MarianMT first
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        try:
            print(f"Loading MarianMT: {model_name} ...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model = model.to(self.device)
            MODELS_CACHE[cache_key] = (tokenizer, model)
            return tokenizer, model
        except Exception as e:
            print(f"MarianMT failed: {e}")
            print("Falling back to NLLB...")
            try:
                model_name = "facebook/nllb-200-distilled-600M"
                print(f"Loading NLLB: {model_name} ...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, src_lang=source_lang
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                model = model.to(self.device)
                MODELS_CACHE[cache_key] = (tokenizer, model)
                return tokenizer, model
            except Exception as nllb_e:
                print(f"NLLB failed: {nllb_e}")
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

        tokenizer, model = self._get_translator(source_lang, target_lang)
        results = []
        for text in texts:
            try:
                if text and len(text) < 512:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_length=512)
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
                print(f"Translation error for '{text[:30]}...': {e}")
                results.append({"original": text, "translation": "[Translation Error]"})
        return results
