"""Line recognisers. Every recogniser has `name` and `read(crops) -> list[(text, confidence)]`.

* `TrOCRLineReader`     – TrOCR (ViT encoder + autoregressive text decoder), the core handwriting model.
                          Line confidence = exp(mean token log-probability) of the greedy decode.
* `TesseractLineReader` – Tesseract LSTM OCR per line (printed text; the Week-2 baseline recogniser).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DEFAULT_HTR_MODEL = "microsoft/trocr-small-handwritten"


class TrOCRLineReader:
    def __init__(self, name: str = DEFAULT_HTR_MODEL, device: str = "cpu", batch_size: Optional[int] = None,
                 cache_dir: Optional[Path] = None, max_new_tokens: int = 72):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        try:                                   # silence the "unauthenticated requests" notice of the Hub client
            from huggingface_hub.utils import logging as hub_logging
            hub_logging.set_verbosity_error()
        except Exception:  # pragma: no cover
            pass
        self.name, self.device, self.max_new_tokens = name, device, max_new_tokens
        self.proc = TrOCRProcessor.from_pretrained(name)
        self.model = VisionEncoderDecoderModel.from_pretrained(name).to(device).eval()
        self.bs = batch_size or (32 if device == "cuda" else 8)
        self.torch = torch
        # optional on-disk cache keyed by the crop pixels (speeds up re-runs; results are identical without it)
        self.cache_path = Path(cache_dir) / f"htr_{name.split('/')[-1]}.json" if cache_dir else None
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path and self.cache_path.exists() else {}
        self.last_seconds = 0.0

    @staticmethod
    def _key(img: np.ndarray) -> str:
        return hashlib.md5(img.tobytes() + str(img.shape).encode()).hexdigest()

    def read(self, crops: Sequence[np.ndarray]):
        """crops: list of grey uint8 arrays -> list of (text, confidence). `last_seconds` holds the recognition time."""
        from PIL import Image
        torch = self.torch
        keys = [self._key(c) for c in crops]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        with torch.inference_mode():
            for s in range(0, len(todo), self.bs):
                idx = todo[s:s + self.bs]
                t0 = time.time()
                ims = [Image.fromarray(crops[i]).convert("RGB") for i in idx]
                pv = self.proc(images=ims, return_tensors="pt").pixel_values.to(self.device)
                g = self.model.generate(pv, max_new_tokens=self.max_new_tokens, num_beams=1, output_scores=True,
                                        return_dict_in_generate=True)
                texts = self.proc.batch_decode(g.sequences, skip_special_tokens=True)
                ts = self.model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True).cpu().numpy()
                gen = g.sequences[:, 1:].cpu().numpy()
                pad = self.model.generation_config.pad_token_id
                dt = (time.time() - t0) / len(idx)
                for j, i in enumerate(idx):
                    valid = gen[j][: ts.shape[1]] != pad
                    lp = ts[j][valid] if valid.any() else ts[j]
                    self.cache[keys[i]] = [texts[j].strip(), float(np.exp(lp.mean())) if len(lp) else 0.0, dt]
        if todo and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
        self.last_seconds = float(sum(self.cache[k][2] for k in keys))
        out = [tuple(self.cache[k][:2]) for k in keys]
        if not self.cache_path:            # no persistent cache: do not let the in-memory one grow unboundedly
            self.cache = {}
        return out


class TesseractLineReader:
    """Tesseract per line (`--psm 7`). Needs the `tesseract` binary and `pytesseract`."""
    name = "tesseract-line"

    def __init__(self):
        import pytesseract  # noqa: F401  (fail early if missing)
        self.last_seconds = 0.0

    def read(self, crops: Sequence[np.ndarray]):
        import pytesseract
        t0 = time.time()
        out = []
        for c in crops:
            d = pytesseract.image_to_data(c, config="--psm 7", output_type=pytesseract.Output.DATAFRAME)
            d = d[(d.conf > -1) & d.text.notna()]
            d = d[d.text.astype(str).str.strip() != ""]
            out.append((" ".join(d.text.astype(str)), float(d.conf.mean() / 100) if len(d) else 0.0))
        self.last_seconds = time.time() - t0
        return out
