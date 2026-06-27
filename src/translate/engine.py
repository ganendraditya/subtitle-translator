"""Translation engine with MarianMT route loading and background preload."""

from __future__ import annotations

import threading
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODELS_CACHE: dict[str, tuple[Any, Any]] = {}
MODELS_LOCK = threading.RLock()

MODEL_ALIASES = {
    ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
    ("ja", "en"): "Helsinki-NLP/opus-mt-jap-en",
    ("en", "ko"): "Helsinki-NLP/opus-mt-tc-big-en-ko",
    ("ko", "en"): "Helsinki-NLP/opus-mt-tc-big-ko-en",
}


class TranslationEngine:
    def __init__(self, use_gpu: bool | None = None):
        self.use_gpu = use_gpu
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self._loading_pairs: set[str] = set()
        self._loading_lock = threading.Lock()
        print(f"[Translate] Device: {self.device}")

    def _cache_key(self, source_lang: str, target_lang: str) -> str:
        return f"{source_lang}-{target_lang}"

    def _model_name(self, source_lang: str, target_lang: str) -> str:
        alias = MODEL_ALIASES.get((source_lang, target_lang))
        if alias is not None:
            return alias
        return f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

    def _route(self, source_lang: str, target_lang: str) -> list[tuple[str, str]]:
        if source_lang == target_lang:
            return []
        if source_lang == "en" or target_lang == "en":
            return [(source_lang, target_lang)]
        return [(source_lang, "en"), ("en", target_lang)]

    def _route_loaded(self, source_lang: str, target_lang: str) -> bool:
        route = self._route(source_lang, target_lang)
        with MODELS_LOCK:
            return all(self._cache_key(src, tgt) in MODELS_CACHE for src, tgt in route)

    def _load_model(self, source_lang: str, target_lang: str) -> tuple[Any, Any]:
        cache_key = self._cache_key(source_lang, target_lang)
        with MODELS_LOCK:
            if cache_key in MODELS_CACHE:
                return MODELS_CACHE[cache_key]

        model_name = self._model_name(source_lang, target_lang)
        print(f"[Translate] Loading {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = model.to(self.device)
        model.eval()

        with MODELS_LOCK:
            if cache_key in MODELS_CACHE:
                return MODELS_CACHE[cache_key]
            MODELS_CACHE[cache_key] = (tokenizer, model)
            return tokenizer, model

    def _pair_key(self, source_lang: str, target_lang: str) -> str:
        return f"{source_lang}->{target_lang}"

    def is_pair_loading(self, source_lang: str, target_lang: str) -> bool:
        with self._loading_lock:
            return self._pair_key(source_lang, target_lang) in self._loading_pairs

    def preload_pair(self, source_lang: str, target_lang: str) -> None:
        """Load the selected route in the background so OCR frames do not block."""
        if self._route_loaded(source_lang, target_lang):
            return

        pair_key = self._pair_key(source_lang, target_lang)
        with self._loading_lock:
            if pair_key in self._loading_pairs:
                return
            self._loading_pairs.add(pair_key)

        def _worker() -> None:
            try:
                route = self._route(source_lang, target_lang)
                if route:
                    route_text = " -> ".join([route[0][0], *[dst for _, dst in route]])
                    print(f"[Translate] Preloading route: {route_text}")
                for src, tgt in route:
                    self._load_model(src, tgt)
                print(f"[Translate] Ready: {source_lang}->{target_lang}")
            except Exception as exc:
                print(f"[Translate] Preload failed {source_lang}->{target_lang}: {exc}")
            finally:
                with self._loading_lock:
                    self._loading_pairs.discard(pair_key)

        threading.Thread(target=_worker, daemon=True).start()

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
                        outputs = model.generate(
                            **inputs,
                            max_length=512,
                            num_beams=4,
                            do_sample=False,
                        )
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

        if self.is_pair_loading(source_lang, target_lang) and not self._route_loaded(source_lang, target_lang):
            raise RuntimeError(f"Model route still loading: {source_lang}->{target_lang}")

        route = self._route(source_lang, target_lang)
        if len(route) == 1:
            src, tgt = route[0]
            tokenizer, model = self._load_model(src, tgt)
            translations = self._translate_batch(texts, tokenizer, model)
            return [{"translation": t} for t in translations]

        try:
            tokenizer_en, model_en = self._load_model(source_lang, "en")
            en_texts = self._translate_batch(texts, tokenizer_en, model_en)
            tokenizer_tgt, model_tgt = self._load_model("en", target_lang)
            final = self._translate_batch(en_texts, tokenizer_tgt, model_tgt)
            return [{"translation": t} for t in final]
        except Exception:
            return [{"translation": f"[No model: {source_lang}->en->{target_lang}]"}]
