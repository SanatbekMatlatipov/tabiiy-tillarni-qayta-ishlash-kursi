"""
m09_text_generator.py — Kapstone modul: LSTM-tayanchli belgi darajasidagi matn generatori.

Shartnoma:
    class TextGenerator:
        def __init__(random_state: int = 42)
        def train(text: str, epochs: int = 10, hidden_size: int = 64, lr: float = 1e-3)
        def generate(seed: str, length: int = 80, temperature: float = 0.7) -> str
        def save(path: str) -> None
        def load(path: str) -> None

Belgi darajasidagi LSTM (agar torch bo'lsa) yoki n-gram (bigrama) fallbackda ishlaydi.
"""
from __future__ import annotations
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, List

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _stable_softmax(logits, temperature: float = 1.0):
    """Barqaror softmax temperature bilan."""
    temperature = max(float(temperature), 1e-8)
    logits = np.asarray(logits, dtype=float) / temperature
    logits -= logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


if HAS_TORCH:
    class _CharLSTM(nn.Module):
        """Ichki LSTM modeli - foydalanuvchiga ko'rinmasligi kerak."""
        def __init__(self, vocab_size: int, hidden_size: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, vocab_size)

        def forward(self, x, hidden=None):
            x = self.embedding(x)
            out, hidden = self.lstm(x, hidden)
            return self.output(out), hidden


class TextGenerator:
    """LSTM (agar mavjud) yoki n-gram belgi darajasidagi matn generatori."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._model = None                    # torch modeli
        self._chars: List[str] = []
        self._char_to_idx: dict = {}
        self._idx_to_char: dict = {}
        self._hidden_size = 64
        # N-gram fallback uchun
        self._bigram_counts: dict = defaultdict(Counter)
        self._context_counts: Counter = Counter()

    def train(self, text: str, epochs: int = 10, hidden_size: int = 64, lr: float = 1e-3):
        """Matn korpusida modelni o'qitadi."""
        if not text or len(text) < 10:
            raise ValueError("Kamida 10 belgi uzunligidagi matn kerak.")

        # Belgilar lug'ati
        self._chars = sorted(set(text))
        self._char_to_idx = {c: i for i, c in enumerate(self._chars)}
        self._idx_to_char = {i: c for i, c in enumerate(self._chars)}
        self._hidden_size = hidden_size

        # N-gram (bigrama) chastotalarni to'playmiz - fallback uchun ham foydali
        for i in range(1, len(text)):
            prev, curr = text[i-1], text[i]
            self._bigram_counts[prev][curr] += 1
            self._context_counts[prev] += 1

        if not HAS_TORCH:
            # Torch yo'q - faqat n-gram
            self._model = None
            return

        # LSTM o'qitish
        torch.manual_seed(self.random_state)
        vocab_size = len(self._chars)
        self._model = _CharLSTM(vocab_size, hidden_size)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        # Ma'lumot tayyorlash: inputlar va targetlar
        inputs = [self._char_to_idx[c] for c in text[:-1]]
        targets = [self._char_to_idx[c] for c in text[1:]]

        # Batch uzunligini cheklash - juda uzun bo'lsa xotira yetmasligi mumkin
        MAX_LEN = 500
        if len(inputs) > MAX_LEN:
            inputs = inputs[:MAX_LEN]
            targets = targets[:MAX_LEN]

        input_t = torch.tensor([inputs], dtype=torch.long)
        target_t = torch.tensor([targets], dtype=torch.long)

        for epoch in range(epochs):
            optimizer.zero_grad()
            logits, _ = self._model(input_t)
            loss = loss_fn(logits.view(-1, vocab_size), target_t.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            optimizer.step()

    def generate(self, seed: str, length: int = 80, temperature: float = 0.7) -> str:
        """seed dan boshlab length ta belgi generatsiya qiladi."""
        if not self._chars:
            raise RuntimeError("Model o'qitilmagan. Avval train() ni chaqiring.")

        rng = np.random.default_rng(self.random_state)
        result = list(seed)

        if HAS_TORCH and self._model is not None:
            self._model.eval()
            # Seed dan hidden state olamiz
            seed_ids = [self._char_to_idx.get(c, 0) for c in seed] or [0]
            with torch.no_grad():
                cur_input = torch.tensor([seed_ids], dtype=torch.long)
                logits, hidden = self._model(cur_input)
                for _ in range(length):
                    # Oxirgi qadam logits
                    last_logits = logits[0, -1].numpy()
                    probs = _stable_softmax(last_logits, temperature)
                    next_id = int(rng.choice(len(probs), p=probs))
                    result.append(self._idx_to_char[next_id])
                    cur_input = torch.tensor([[next_id]], dtype=torch.long)
                    logits, hidden = self._model(cur_input, hidden)
        else:
            # N-gram fallback
            vocab = np.array(self._chars)
            for _ in range(length):
                prev = result[-1] if result else " "
                counts = self._bigram_counts.get(prev, Counter())
                if not counts:
                    counts = Counter(self._chars)
                # Faqat lug'atdagi belgilar uchun
                probs = np.array([counts.get(c, 0) + 1 for c in self._chars], dtype=float)
                probs = _stable_softmax(np.log(probs), temperature)
                next_idx = int(rng.choice(len(probs), p=probs))
                result.append(self._chars[next_idx])

        return "".join(result)

    def save(self, path: str) -> None:
        """Modelni faylga saqlaydi."""
        state = {
            "random_state": self.random_state,
            "chars": self._chars,
            "char_to_idx": self._char_to_idx,
            "idx_to_char": self._idx_to_char,
            "hidden_size": self._hidden_size,
            "bigram_counts": dict(self._bigram_counts),
            "context_counts": self._context_counts,
            "torch_state": self._model.state_dict() if (HAS_TORCH and self._model) else None,
        }
        Path(path).write_bytes(pickle.dumps(state))

    def load(self, path: str) -> None:
        """Modelni fayldan yuklaydi."""
        state = pickle.loads(Path(path).read_bytes())
        self.random_state = state["random_state"]
        self._chars = state["chars"]
        self._char_to_idx = state["char_to_idx"]
        self._idx_to_char = state["idx_to_char"]
        self._hidden_size = state["hidden_size"]
        self._bigram_counts = defaultdict(Counter, {k: Counter(v) for k, v in state["bigram_counts"].items()})
        self._context_counts = state["context_counts"]

        if HAS_TORCH and state["torch_state"] is not None:
            self._model = _CharLSTM(len(self._chars), self._hidden_size)
            self._model.load_state_dict(state["torch_state"])
        else:
            self._model = None
