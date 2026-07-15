"""
m13_fine_tuned_classifier.py — Kapstone modul: BERT-asosli sentiment klassifikatori.

Shartnoma:
    class FineTunedClassifier:
        def fit(texts, labels, model_name, epochs, batch_size, lr)
        def predict(text) -> str          # 'ijobiy' | 'salbiy'
        def predict_proba(text) -> dict   # {'ijobiy': ..., 'salbiy': ...}
        def save(path) -> None
        def load(path) -> None

Online rejim (GPU): distilbert-base-multilingual-cased + Hugging Face Trainer.
Offline rejim (CPU): TF-IDF + LogisticRegression fallback.

d14_p13_bert_finetune.ipynb dan olingan.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

try:
    import torch
    HAS_TORCH = True
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    HAS_TORCH = False
    _DEVICE = "cpu"

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

_LABEL2ID = {"salbiy": 0, "ijobiy": 1}
_ID2LABEL = {0: "salbiy", 1: "ijobiy"}

_USE_BERT = HAS_TRANSFORMERS and HAS_TORCH and _DEVICE == "cuda"


def _build_baseline(texts: list[str], labels: list[str]):
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn talab qilinadi (offline rejim uchun).")
    clf = SkPipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    clf.fit(texts, labels)
    return clf


class FineTunedClassifier:
    """Hugging Face Trainer orqali nozik sozlangan BERT-class sentiment modeli.

    Corpus: risqaliyevds/uzbek-sentiment-analysis (MIT, 5000 subsample).
    Consumed by: m15 (agent tool: sentiment_classify), app.py (FastAPI).
    """

    def __init__(self) -> None:
        self._model_name: str = "distilbert-base-multilingual-cased"
        self._tokenizer: Any = None
        self._model: Any = None
        self._baseline: Any = None

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        model_name: str = "distilbert-base-multilingual-cased",
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
    ) -> None:
        """BERT modelini Trainer API orqali fine-tune qiladi.

        Args:
            texts:      O'quv matnlari.
            labels:     'ijobiy' yoki 'salbiy' belgilari ro'yxati.
            model_name: Hugging Face model identifikatori.
            epochs:     O'quv davrlari soni.
            batch_size: Har qurilmadagi batch hajmi.
            lr:         O'rganish tezligi.
        """
        if not texts:
            raise ValueError("texts bo'sh bo'lmasligi kerak.")
        if len(texts) != len(labels):
            raise ValueError(
                f"texts ({len(texts)}) va labels ({len(labels)}) uzunligi mos kelmadi."
            )
        for lbl in labels:
            if lbl not in _LABEL2ID:
                raise ValueError(f"Noto'g'ri label: {lbl!r}. 'ijobiy' yoki 'salbiy' bo'lishi kerak.")

        self._model_name = model_name

        if _USE_BERT:
            self._fit_bert(texts, labels, model_name, epochs, batch_size, lr)
        else:
            self._baseline = _build_baseline(texts, labels)

    def _fit_bert(self, texts, labels, model_name, epochs, batch_size, lr) -> None:
        from datasets import Dataset
        import numpy as np

        int_labels = [_LABEL2ID[l] for l in labels]
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).to(_DEVICE)

        def _tok(batch):
            return self._tokenizer(batch["text"], truncation=True, max_length=128)

        ds = Dataset.from_dict({"text": texts, "label": int_labels})
        ds = ds.map(_tok, batched=True)

        args = TrainingArguments(
            output_dir="./m13_out",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            warmup_steps=100,
            save_strategy="no",
            logging_steps=50,
        )
        Trainer(
            model=self._model,
            args=args,
            train_dataset=ds,
            tokenizer=self._tokenizer,
            data_collator=DataCollatorWithPadding(self._tokenizer),
        ).train()

    def predict(self, text: str) -> str:
        """'ijobiy' yoki 'salbiy' qaytaradi."""
        probs = self.predict_proba(text)
        return max(probs, key=probs.get)

    def predict_proba(self, text: str) -> dict[str, float]:
        """{'ijobiy': 0.87, 'salbiy': 0.13} formatida ehtimolliklar."""
        if not text:
            raise ValueError("Matn bo'sh bo'lmasligi kerak.")

        if self._baseline is not None:
            probs = self._baseline.predict_proba([text])[0]
            classes = list(self._baseline.classes_)
            return {cls: float(p) for cls, p in zip(classes, probs)}

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Avval fit() ni chaqiring.")

        self._model.eval()
        inputs = self._tokenizer(text[:512], return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
        return {"salbiy": float(probs[0]), "ijobiy": float(probs[1])}

    def save(self, path: str) -> None:
        """save_pretrained(path) orqali saqlaydi."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if self._model is not None and self._tokenizer is not None:
            self._model.save_pretrained(str(p))
            self._tokenizer.save_pretrained(str(p))
            (p / "_meta.pkl").write_bytes(pickle.dumps({"mode": "bert", "model_name": self._model_name}))
        elif self._baseline is not None:
            (p / "_baseline.pkl").write_bytes(pickle.dumps(self._baseline))
            (p / "_meta.pkl").write_bytes(pickle.dumps({"mode": "baseline", "model_name": self._model_name}))
        else:
            raise RuntimeError("Saqlash uchun o'qitilgan model yo'q. Avval fit() ni chaqiring.")

    def load(self, path: str) -> None:
        """from_pretrained(path) orqali yuklaydi."""
        p = Path(path)
        meta = pickle.loads((p / "_meta.pkl").read_bytes())
        self._model_name = meta["model_name"]
        if meta["mode"] == "bert" and HAS_TRANSFORMERS and HAS_TORCH:
            self._tokenizer = AutoTokenizer.from_pretrained(str(p))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(p)).to(_DEVICE)
            self._baseline = None
        else:
            baseline_path = p / "_baseline.pkl"
            if not baseline_path.exists():
                raise FileNotFoundError(f"Baseline model topilmadi: {baseline_path}")
            self._baseline = pickle.loads(baseline_path.read_bytes())
            self._model = None
            self._tokenizer = None
