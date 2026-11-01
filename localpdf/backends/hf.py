"""Hugging Face Transformers backends (local weights, GPU or CPU, optional quantisation).

UNVERIFIED IN THIS SANDBOX: torch / transformers / sentence-transformers could not be installed where this file was
written, so it has only been syntax-checked. It is adapted from the working v3 notebook; run `pytest -m models` on
your machine to exercise it.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from ..config import HardwareProfile
from .base import Backends, Generation

_LOCK = threading.Lock()  # one model call at a time (chat and background ingestion share the device)


class HFEmbedder:
    def __init__(self, model_id: str, device: str):
        from sentence_transformers import SentenceTransformer

        self.id = model_id
        self.model = SentenceTransformer(model_id, device=device)
        self.model.max_seq_length = 384
        self.dim = self.model.get_sentence_embedding_dimension()
        # bge *-en-v1.5 models want an instruction on queries; bge-m3 / e5 conventions differ.
        self.query_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model_id and "m3" not in model_id else ""
        if "e5" in model_id:
            self.query_prefix = "query: "
        self.doc_prefix = "passage: " if "e5" in model_id else ""

    def embed(self, texts, is_query=False):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        pre = self.query_prefix if is_query else self.doc_prefix
        with _LOCK:
            v = self.model.encode([pre + t for t in texts], batch_size=16, normalize_embeddings=True,
                                  show_progress_bar=False)
        return np.ascontiguousarray(v.astype(np.float32))


class HFReranker:
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.id, self.device, self._torch = model_id, device, torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        m = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        self.model = (m.half() if device == "cuda" else m).to(device)

    def score(self, query, texts):
        torch = self._torch
        out = []
        for i in range(0, len(texts), 16):
            b = texts[i:i + 16]
            enc = self.tok([query] * len(b), b, padding=True, truncation=True, max_length=384,
                           return_tensors="pt").to(self.device)
            with torch.inference_mode(), _LOCK:
                out.append(self.model(**enc).logits.view(-1).float().cpu().numpy())
        return 1 / (1 + np.exp(-np.concatenate(out))) if out else np.zeros(0)


class HFLLM:
    def __init__(self, model_id: str, device: str, quantization: str = "none"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.id, self._torch = model_id, torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if device == "cuda":
            kw: dict = {"device_map": "auto"}
            if quantization == "4bit":
                from transformers import BitsAndBytesConfig

                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True)
            else:
                kw["dtype"] = torch.float16
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
            except TypeError:  # older transformers
                if "dtype" in kw:
                    kw["torch_dtype"] = kw.pop("dtype")
                self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        else:
            try:
                m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, low_cpu_mem_usage=True)
            except TypeError:
                m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
            if quantization == "int8":
                m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
            self.model = m.to("cpu")
        self.model.eval()

    def generate(self, messages, max_new_tokens=300, on_text: Callable[[str], None] | None = None) -> Generation:
        from transformers import TextIteratorStreamer

        torch = self._torch
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        gen = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id, do_sample=False,
                   temperature=None, top_p=None, top_k=None)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True, timeout=900)
        err: list[Exception] = []

        def run():
            try:
                with torch.inference_mode(), _LOCK:
                    self.model.generate(**inputs, streamer=streamer, **gen)
            except Exception as e:  # make sure the consumer loop always ends
                err.append(e)
                streamer.on_finalized_text("", stream_end=True)

        t0 = time.perf_counter()
        th = threading.Thread(target=run, daemon=True)
        th.start()
        acc, first = "", None
        for piece in streamer:
            if first is None and piece:
                first = time.perf_counter() - t0
            acc += piece
            if on_text:
                try:
                    on_text(acc.strip())
                except Exception:
                    pass
        th.join()
        if err:
            raise err[0]
        total = time.perf_counter() - t0
        n = len(self.tok(acc, add_special_tokens=False)["input_ids"])
        return Generation(text=acc.strip(), first_token_s=first if first is not None else total, total_s=total,
                          n_tokens=n)


def build_hf_backends(profile: HardwareProfile, device: str, **overrides) -> Backends:
    """overrides: llm_id=, embed_id=, rerank_id=, quantization= (used by the model-comparison lab)."""
    llm_id = overrides.get("llm_id", profile.llm_id)
    q = overrides.get("quantization", profile.quantization)
    return Backends(
        embedder=HFEmbedder(overrides.get("embed_id", profile.embed_id), device),
        reranker=HFReranker(overrides.get("rerank_id", profile.rerank_id), device),
        llm=HFLLM(llm_id, device, q),
        is_stand_in=False,
    )
