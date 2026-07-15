"""
capstone/contracts.py — Barcha kapstone modullar uchun rasmiy interfeys shartnomasi.

Bu fayldagi imzolar (signatures) O'ZGARISHSIZ. Har bir amaliyot shu
imzolarga mos keladigan implementatsiya yozadi. Implementatsiya YO'Q —
faqat imzolar, type hints va docstringlar.

Modul zanjiri:
    m01 → m02, m03, m04, m05, m06, m07, m08, m09, m10, m11, m12, m14, m15
    m13 → m15, app.py
    m14 → m15
    m15 → (final agent — defense demo)
    app.py → (FastAPI deployment — M4 milestone)
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# m01 — TextPreprocessor   (2-kun amaliyoti, P1)
# ─────────────────────────────────────────────────────────────────────────────

class TextPreprocessor:
    """O'zbek matni uchun tokenizatsiya + normalizatsiya + stemming pipeline.

    Consumed by: m02, m04, m05, m06, m07, m08, m09, m11, m15.
    """

    def preprocess(self, text: str) -> list[str]:
        """Bitta matnni tokenize qilib, tozalab, stemlab qaytaradi.

        Args:
            text: Xom o'zbek matni (UTF-8, U+2019 yoki ASCII apostrof).

        Returns:
            Tozalangan tokenlar ro'yxati. Stopwordlar va tinywords (len<2)
            tashlanadi. Har token kichik harfda.

        Raises:
            ValueError: Agar text bo'sh string bo'lsa.
        """
        ...

    def preprocess_batch(self, texts: list[str]) -> list[list[str]]:
        """preprocess() ni bir ro'yxat uchun qo'llaydi."""
        ...

    def fit_stopwords(self, texts: list[str], max_df: float = 0.85) -> None:
        """Korpus-spesifik stopwordlarni chastota bo'yicha aniqlaydi."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# m02 — SentimentClassifier   (3-kun amaliyoti, P2)
# ─────────────────────────────────────────────────────────────────────────────

class SentimentClassifier:
    """TF-IDF + LogReg yoki NaiveBayes asosida ikkilik sentiment tahlili.

    Consumed by: M4 (FastAPI), Day 16 (agent tool).
    """

    def fit(self, texts: list[str], labels: list[str]) -> None:
        """Modelni o'qitadi.

        Args:
            texts:  Xom matnlar ro'yxati (preprocessing ichida qilinadi).
            labels: Sinf belgilari — 'ijobiy' yoki 'salbiy'.
        """
        ...

    def predict(self, text: str) -> str:
        """Bitta matn uchun sentiment bashorat qiladi.

        Returns:
            'ijobiy' yoki 'salbiy'.
        """
        ...

    def predict_proba(self, text: str) -> dict[str, float]:
        """Ehtimolliklar: {'ijobiy': 0.82, 'salbiy': 0.18}."""
        ...

    def save(self, path: str) -> None:
        """Modelni pickle orqali saqlaydi."""
        ...

    def load(self, path: str) -> None:
        """Saqlangan modelni yuklaydi."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# m03 — PretrainedEmbedder   (4-kun amaliyoti, P3)
# ─────────────────────────────────────────────────────────────────────────────

class PretrainedEmbedder:
    """Oldindan o'qitilgan Word2Vec/.kv embeddinglarini boshqaradi.

    Consumed by: m04 (LSH), m08 (GRU/LSTM pretrained init).
    """

    def load(self, path: str) -> None:
        """Gensim KeyedVectors formatidagi .kv faylni yuklaydi.

        Args:
            path: .kv fayl yo'li (masalan, 'cc_uz_100k.kv').
        """
        ...

    def embed(self, word: str) -> np.ndarray:
        """So'z uchun vektor qaytaradi; OOV uchun sifr-vektori.

        Returns:
            shape (vector_size,) float32 ndarray.
        """
        ...

    def most_similar(self, word: str, n: int = 5) -> list[tuple[str, float]]:
        """Eng o'xshash n ta so'zni qaytaradi: [(so'z, o'xshashlik), ...]."""
        ...

    def oov_rate(self, texts: list[list[str]]) -> float:
        """Tokenlar orasida lug'atda yo'q so'zlar ulushini qaytaradi [0,1]."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# m04 — SpellLSHRetriever   (5-kun amaliyoti, P4)
# ─────────────────────────────────────────────────────────────────────────────

class SpellLSHRetriever:
    """Imlo tuzatish (Noisy Channel) + LSH asosida hujjat qidirish.

    Consumed by: m15 (agent tool: spell_correct), Day 16 (pipeline).
    """

    def correct(self, word: str) -> str:
        """Noisy Channel modeli orqali eng ehtimoliy to'g'ri so'zni qaytaradi.

        Args:
            word: Imlosi xato bo'lishi mumkin bo'lgan so'z.

        Returns:
            Lug'atdagi eng yaqin to'g'ri variant (yoki word o'zi agar topilmasa).
        """
        ...

    def edit_distance(self, s1: str, s2: str) -> int:
        """Levenshtein masofasini DP orqali hisoblaydi."""
        ...

    def index_docs(self, texts: list[str]) -> None:
        """Hujjatlarni MinHash LSH indeksiga qo'shadi."""
        ...

    def retrieve_lsh(self, query: str, k: int = 5) -> list[str]:
        """LSH orqali eng o'xshash k ta hujjatni qaytaradi."""
        ...

    def save(self, path: str) -> None:
        """Spell-checker lug'ati va LSH indeksini saqlaydi."""
        ...

    def load(self, path: str) -> None:
        """Saqlangan artefaktlarni qayta yuklaydi."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# m05 — Autocomplete   (6-kun amaliyoti, P5)
# ─────────────────────────────────────────────────────────────────────────────

class Autocomplete:
    """N-gram til modeli asosida so'z/ibora to'ldirish.

    Consumed by: Day 16 (pipeline demo).
    """

    def train(self, texts: list[list[str]], n: int = 2) -> None:
        """N-gram modelini Laplace smoothing bilan o'qitadi."""
        ...

    def complete(self, prefix: str, k: int = 3) -> list[str]:
        """So'rovning davomini k ta variant bilan qaytaradi.

        Args:
            prefix: Foydalanuvchi yozgan boshlang'ich so'z(lar).
            k:      Nechta variant qaytarish.

        Returns:
            Ehtimollik bo'yicha tartiblangan k ta davom varianti.
        """
        ...

    def perplexity(self, text: str) -> float:
        """Berilgan matn uchun modelning perplexity ni qaytaradi."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# m05b — POSTagger   (6-kun amaliyoti, P5 — pedagogical only)
# ─────────────────────────────────────────────────────────────────────────────

class POSTagger:
    """Yashirin Markov Modeli + Viterbi orqali so'z turkumini teglash.

    Note: Bu modul yakuniy pipeline da ishlatilmaydi — pedagogik demo.
    """

    def train(self, tagged_sentences: list[list[tuple[str, str]]]) -> None:
        """HMM parametrlarini (emission, transition) hisoblaydi.

        Args:
            tagged_sentences: [[(token, tag), ...], ...] formatidagi corpus.
        """
        ...

    def tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        """Viterbi algoritmi yordamida har token uchun tag bashorat qiladi.

        Returns:
            [(token, tag), ...] — masalan [('men', 'PR'), ('uxladim', 'VB')].
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# m06 — CustomWord2Vec   (7-kun amaliyoti, P6)
# ─────────────────────────────────────────────────────────────────────────────

class CustomWord2Vec:
    """Gensim bilan o'zbek korpusida noldan o'qitilgan CBOW embeddinglar.

    Consumed by: m08 (GRU/LSTM pretrained Embedding layer), m09 (generator).
    """

    def train(
        self,
        texts: list[list[str]],
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 3,
        epochs: int = 10,
    ) -> None:
        """Word2Vec CBOW modelini o'qitadi."""
        ...

    def embed(self, word: str) -> np.ndarray:
        """So'z vektorini qaytaradi; OOV uchun sifr-vektori."""
        ...

    def most_similar(self, word: str, n: int = 5) -> list[tuple[str, float]]:
        """Eng yaqin n ta so'z va o'xshashlik darajalari."""
        ...

    def save(self, path: str) -> None:
        """Modelni .kv formatida saqlaydi."""
        ...

    def load(self, path: str) -> None:
        """Saqlangan .kv faylni yuklaydi."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# m07 — RNNClassifier   (8-kun amaliyoti, P7)
# ─────────────────────────────────────────────────────────────────────────────

class RNNClassifier:
    """PyTorch nn.RNN asosida matn tasniflash modeli.

    Consumed by: m08 (baseline comparison), Day 16 (pipeline demo).
    """

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        epochs: int = 5,
        hidden_size: int = 64,
        lr: float = 1e-3,
    ) -> None:
        """SimpleRNN modelini o'qitadi."""
        ...

    def predict(self, text: str) -> str:
        """Bitta matn uchun sinf belgilab qaytaradi."""
        ...

    def predict_proba(self, text: str) -> dict[str, float]:
        """Sinf ehtimolliklarini qaytaradi."""
        ...

    def save(self, path: str) -> None:
        """Modelni torch.save orqali saqlaydi."""
        ...

    def load(self, path: str) -> None:
        """Saqlangan modelni yuklaydi."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# m08 — GRULSTMClassifier   (9-kun amaliyoti, P8)
# ─────────────────────────────────────────────────────────────────────────────

class GRULSTMClassifier:
    """GRU yoki LSTM arxitekturasi bilan matn tasniflash va taqqoslash.

    Consumed by: m13 (baseline vs BERT comparison), Day 16 (pipeline).
    """

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        arch: str = "lstm",
        epochs: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        lr: float = 1e-3,
    ) -> None:
        """GRU yoki LSTM modelini o'qitadi.

        Args:
            arch: 'lstm' yoki 'gru'.
        """
        ...

    def predict(self, text: str) -> str:
        """Bitta matn uchun sinf belgisi."""
        ...

    def compare_report(self) -> dict[str, Any]:
        """GRU vs LSTM F1, val_loss, inference_time taqqoslov hisoboti."""
        ...

    def save(self, path: str) -> None:
        """Modelni saqlaydi."""
        ...

    def load(self, path: str) -> None:
        """Saqlangan modelni yuklaydi."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# m09 — TextGenerator   (10-kun amaliyoti, P9 — pedagogical only)
# ─────────────────────────────────────────────────────────────────────────────

class TextGenerator:
    """Char-level LSTM asosida matn generatsiya qiluvchi model.

    Note: Pedagogik demo — yakuniy pipeline da ishlatilmaydi.
    """

    def train(self, text: str, epochs: int = 20, hidden_size: int = 128) -> None:
        """Char-level LSTM modelini o'qitadi."""
        ...

    def generate(self, seed: str, length: int = 200, temperature: float = 0.7) -> str:
        """Berilgan boshlang'ichdan matn davomini generatsiya qiladi.

        Args:
            seed:        Boshlang'ich satr (masalan, "Bir bor edi").
            length:      Nechta belgi generatsiya qilish.
            temperature: 0.1 (deterministic) – 2.0 (ijodiy/tasodifiy).
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# m10 — NERTagger   (11-kun amaliyoti, P10)
# ─────────────────────────────────────────────────────────────────────────────

class NERTagger:
    """Bi-LSTM asosida IOB2 formatida nomlangan obyektlarni teglash.

    Corpus: WikiANN uz (train+dev ~200 jumlat). F1 past bo'lishi kutilgan —
    o'zbek NER uchun ochiq ma'lumotlar kamligi demo.

    Consumed by: m15 (agent tool: extract_entities), Day 16 (pipeline).
    """

    def fit(self, tagged_sentences: list[list[tuple[str, str]]]) -> None:
        """IOB2 teglangan jumlatlarda Bi-LSTM modelini o'qitadi."""
        ...

    def predict(self, text: str) -> list[tuple[str, str]]:
        """Matndagi tokenlar uchun IOB2 teglarni bashorat qiladi.

        Returns:
            [(token, tag), ...] — masalan [('Malika', 'B-PER'), ('Toshkentda', 'B-LOC')].
        """
        ...

    def entities(self, text: str) -> list[dict[str, str]]:
        """Aniqlangan obyektlarni lug'at ro'yxati sifatida qaytaradi.

        Returns:
            [{'text': 'Malika', 'label': 'PER', 'start': 0, 'end': 6}, ...]
        """
        ...

    def save(self, path: str) -> None:
        """Modelni saqlaydi."""
        ...

    def load(self, path: str) -> None:
        """Saqlangan modelni yuklaydi."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# m11 — Seq2SeqTranslator   (12-kun amaliyoti, P11 — pedagogical only)
# ─────────────────────────────────────────────────────────────────────────────

class Seq2SeqTranslator:
    """LSTM Encoder-Decoder + Bahdanau Attention asosida NMT modeli.

    Corpus: OPUS-100 uz-en 20k jumlat; BLEU demo sifatida ko'rsatiladi.
    Note: Pedagogik demo — yakuniy pipeline da ishlatilmaydi.
    """

    def train(
        self,
        src_texts: list[str],
        tgt_texts: list[str],
        epochs: int = 10,
        max_len: int = 50,
    ) -> None:
        """Encoder-Decoder modelini teacher forcing bilan o'qitadi."""
        ...

    def translate(self, text: str) -> str:
        """O'zbekchadan inglizchaga tarjima qiladi (demo sifatida)."""
        ...

    def bleu(self, references: list[list[str]], hypotheses: list[str]) -> float:
        """BLEU-4 metrikasini hisoblaydi."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# m12 — TransformerSummarizer   (13-kun amaliyoti, P12)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerSummarizer:
    """Mini Transformer Encoder-Decoder asosida matn qisqartirish.

    Corpus: Wikipedia uz lead-paragraph juftlari (CC-BY-SA).
    Consumed by: m15 (agent tool: summarize_text), Day 16 (pipeline).
    """

    def train(
        self,
        src_texts: list[str],
        tgt_texts: list[str],
        epochs: int = 10,
        d_model: int = 128,
        nhead: int = 4,
    ) -> None:
        """Transformer modelini o'qitadi."""
        ...

    def summarize(self, text: str, max_length: int = 60) -> str:
        """Berilgan matn uchun qisqa xulosa generatsiya qiladi."""
        ...

    def rouge1(self, references: list[str], hypotheses: list[str]) -> dict[str, float]:
        """ROUGE-1 precision, recall, F1 ni qaytaradi."""
        ...

    def save(self, path: str) -> None:
        """Modelni saqlaydi."""
        ...

    def load(self, path: str) -> None:
        """Saqlangan modelni yuklaydi."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# m13 — FineTunedClassifier   (14-kun amaliyoti, P13)
# ─────────────────────────────────────────────────────────────────────────────

class FineTunedClassifier:
    """Hugging Face Trainer orqali nozik sozlangan BERT-class sentiment modeli.

    Corpus: risqaliyevds/uzbek-sentiment-analysis (MIT, 5000 subsample).
    Consumed by: m15 (agent tool: sentiment_classify), app.py (FastAPI).
    """

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        model_name: str = "distilbert-base-multilingual-cased",
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
    ) -> None:
        """BERT modelini Trainer API orqali fine-tune qiladi."""
        ...

    def predict(self, text: str) -> str:
        """'ijobiy' yoki 'salbiy' qaytaradi."""
        ...

    def predict_proba(self, text: str) -> dict[str, float]:
        """{'ijobiy': 0.87, 'salbiy': 0.13} formatida ehtimolliklar."""
        ...

    def save(self, path: str) -> None:
        """save_pretrained(path) orqali saqlaydi."""
        ...

    def load(self, path: str) -> None:
        """from_pretrained(path) orqali yuklaydi."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# m14 — RAGEngine   (15-kun amaliyoti, P14)
# ─────────────────────────────────────────────────────────────────────────────

class RAGEngine:
    """FAISS + sentence-transformers + LLM API asosida RAG qidirish tizimi.

    Corpus: uz_kb (yangiliklar + lex.uz, 10 000 chunk, FAISS indeksi).
    Consumed by: m15 (agent tool: retrieve_docs).
    """

    def index(self, texts: list[str], batch_size: int = 32) -> None:
        """Hujjatlarni embedding qilib FAISS indeksiga qo'shadi."""
        ...

    def answer(self, question: str, k: int = 3) -> dict[str, Any]:
        """Savolga RAG pipeline orqali javob qaytaradi.

        Returns:
            {
              'answer':    str,        # LLM javob matni
              'sources':   list[str],  # Foydalanilgan hujjat parchalari
              'confidence': float,     # Retrieval o'xshashligi (0-1)
            }
        """
        ...

    def save_index(self, path: str) -> None:
        """FAISS indeksini .faiss faylga saqlaydi."""
        ...

    def load_index(self, path: str) -> None:
        """Saqlangan indeksni yuklaydi."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# m15 — DocumentAssistantAgent   (16-kun amaliyoti, P15)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentAssistantAgent:
    """LangChain ReAct agent — barcha kapstone modullarini asbob sifatida birlashtiradi.

    Tools:
        sentiment_classify  → FineTunedClassifier.predict()
        retrieve_docs       → RAGEngine.answer()
        summarize_text      → TransformerSummarizer.summarize()
        spell_correct       → SpellLSHRetriever.correct()
        extract_entities    → NERTagger.entities()
    """

    def run(self, user_message: str) -> str:
        """Foydalanuvchi so'rovini ReAct agentiga yuboradi va javob qaytaradi.

        Agent Thought → Action → Observation siklini kerak bo'lgancha
        takrorlaydi va yakuniy javobni o'zbek tilida qaytaradi.

        Args:
            user_message: Foydalanuvchining o'zbek tilidagi savoli.

        Returns:
            Agent tomonidan shakllangan o'zbek tilidagi javob matni.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# SentimentAPI   (M4 milestone — app.py da implementatsiya qilinadi)
# ─────────────────────────────────────────────────────────────────────────────

def create_sentiment_api() -> Any:
    """FastAPI ilovasi — FineTunedClassifier ni HTTP orqali ochadi.

    Endpoint:
        POST /predict
        Request body:  {"text": "matn matni"}
        Response body: {"sentiment": "ijobiy", "confidence": 0.87}

    Returns:
        FastAPI app obyekti.

    Note:
        Implementatsiya capstone/app.py faylida. Bu funksiya faqat
        shartnoma sifatida — import va test uchun.
    """
    ...
