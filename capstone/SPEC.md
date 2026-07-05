# Kapstone loyiha: O'zbek-tili Hujjat Yordamchisi

**Kurs:** Tabiiy tilni qayta ishlash (NLP) | VMQ 425-son  
**Muddat:** 15-iyun – 10-iyul 2026  
**Har bir tinglovchi** kurs davomida yagona, umumiy loyiha ustida ishlaydi.

---

## Yakuniy mahsulot nima?

Kurs oxirida sizda ishlaydigan **O'zbek-tili Hujjat Yordamchisi** bo'ladi —
matndagi savollarga o'zbek tilida javob bera oladigan va quyidagi
imkoniyatlarga ega tizim:

| Imkoniyat | Modul | Ishlab chiqiladi |
|---|---|---|
| Matnni tozalash va tokenizatsiya | `TextPreprocessor` | 2-kun amaliyoti |
| Hissiyot tahlili (TF-IDF asosida) | `SentimentClassifier` | 3-kun amaliyoti |
| So'z o'xshashligi va embedding | `PretrainedEmbedder` | 4-kun amaliyoti |
| Imlo tuzatish + LSH qidiruv | `SpellLSHRetriever` | 5-kun amaliyoti |
| So'z to'ldirish (autocomplete) | `Autocomplete` | 6-kun amaliyoti |
| Maxsus o'zbek embeddingleri | `CustomWord2Vec` | 7-kun amaliyoti |
| RNN asosida tasniflash | `RNNClassifier` | 8-kun amaliyoti |
| GRU/LSTM taqqoslov modeli | `GRULSTMClassifier` | 9-kun amaliyoti |
| NER: shaxs, joy, tashkilot | `NERTagger` | 11-kun amaliyoti |
| BERT asosida nozik sozlash | `FineTunedClassifier` | 14-kun amaliyoti |
| Transformer xulosa chiqarish | `TransformerSummarizer` | 13-kun amaliyoti |
| RAG qidirish va javob | `RAGEngine` | 15-kun amaliyoti |
| LangChain ReAct agent | `DocumentAssistantAgent` | 16-kun amaliyoti |
| FastAPI deployment | `SentimentAPI` | M4 milestone |

---

## Qanday ishlaydi?

```
Foydalanuvchi so'rovi (o'zbek tilida)
        │
        ▼
TextPreprocessor → SpellLSHRetriever (imlo + qidiruv)
        │
        ├──► FineTunedClassifier  → hissiyot: ijobiy/salbiy
        │
        ├──► NERTagger            → shaxslar, joylar, tashkilotlar
        │
        ├──► RAGEngine            → tegishli hujjat parchalari
        │
        └──► DocumentAssistantAgent (ReAct) → yakuniy javob
                        │
                FastAPI (SentimentAPI)
                        │
                    JSON javob
```

---

## Loyiha fayl tuzilmasi

```
capstone/
├── SPEC.md           ← shu fayl (loyiha shartnomasi)
├── contracts.py      ← barcha modullar uchun shablonlar (type hints + docstring)
├── app.py            ← FastAPI serveri (M4 milestone da yaratiladi)
└── modules/
    ├── m01_text_preprocessor.py
    ├── m02_sentiment_classifier.py
    ├── m03_pretrained_embedder.py
    ├── m04_spell_lsh_retriever.py
    ├── m05_autocomplete.py
    ├── m05b_pos_tagger.py
    ├── m06_custom_word2vec.py
    ├── m07_rnn_classifier.py
    ├── m08_gru_lstm_classifier.py
    ├── m09_text_generator.py
    ├── m10_ner_tagger.py
    ├── m11_seq2seq_translator.py
    ├── m12_transformer_summarizer.py
    ├── m13_bert_classifier.py
    ├── m14_rag_engine.py
    └── m15_langchain_agent.py
```

---

## Haftalik o'sish

**1-hafta** (15–19 iyun): Klassik pipeline
> TextPreprocessor → SentimentClassifier → PretrainedEmbedder → SpellLSHRetriever

**2-hafta** (22–26 iyun): Statistik modellar va neyron embeddinglar
> Autocomplete → CustomWord2Vec → RNNClassifier → GRULSTMClassifier

**3-hafta** (29 iyun – 3 iyul): Rekurrent va Transformer arxitekturalar
> TextGenerator → NERTagger → Seq2SeqTranslator → TransformerSummarizer

**4-hafta** (6–10 iyul): Transfer learning, RAG, agentlar, deploy
> FineTunedClassifier → SentimentAPI (M4) → RAGEngine → DocumentAssistantAgent

---

## Qoidalar

1. Har bir modul `contracts.py` dagi shablonga mos kelishi shart.
2. `preprocess()` va boshqa asosiy funksiyalar xatoni `ValueError` / `TypeError`
   orqali aniq xabar bilan qaytarishi kerak.
3. Barcha modellar `save(path)` va `load(path)` metodlariga ega bo'lishi shart
   (4-haftada FastAPI serveriga yuklash uchun).
4. Barcha modul fayllari `capstone/modules/` papkasida saqlanadi.
5. Har hafta milestone da modullar birlashtiriladi va umumiy pipeline sinovdan
   o'tkaziladi.

---

*Ushbu hujjat `course/course_map.yaml` dan avtomatik tarzda keltirib chiqarilgan.
Modul shablonlarini `capstone/contracts.py` faylida ko'ring.*
