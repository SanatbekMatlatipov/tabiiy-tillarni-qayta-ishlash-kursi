# LICENSES.md — Dataset License Register

**Policy**: Before any dataset is committed to a practice notebook, pull-time
license confirmation is required. "Dataset card says X" is not sufficient —
confirm the primary source license and document the basis for use below.

Every dataset used in the course must have a row here. Status must be one of:
- `CONFIRMED` — primary source license verified, use basis documented
- `EDUCATIONAL-USE` — license unknown/unspecified; educational/research use
  basis stated and risk accepted in writing (record approver)
- `NOT USED` — listed for reference (rejected candidates)

---

## Approved Datasets

### uz_news_mini / uz_news_full
| Field | Value |
|---|---|
| Source | kun.uz, gazeta.uz (Uzbek news portals) |
| License | Public web content; scraped for educational use |
| Use basis | Educational/research; non-commercial course material |
| Pull confirmation | Kaggle Dataset attachment — confirm at commit time |
| Status | **CONFIRMED (educational use)** |

---

### cc_uz_100k_kv — Word2Vec vectors
| Field | Value |
|---|---|
| Source | Common Crawl Uzbek (cc.uz subset) |
| License | Common Crawl Terms of Use — research/education permitted |
| Use basis | Truncated .kv file (~240 MB) distributed as Kaggle Dataset |
| Primary license URL | https://commoncrawl.org/terms-of-use |
| Pull confirmation | Verify Kaggle Dataset page at attach time |
| Status | **CONFIRMED** |

---

### uz_ner_wikiann — WikiANN Uzbek NER (C1 fallback)
| Field | Value |
|---|---|
| HF ID | `unimelb-nlp/wikiann` (uz config) |
| HF URL | https://huggingface.co/datasets/unimelb-nlp/wikiann |
| Size | train=100, dev=100, test=100 (used: train+dev combined, ~200 sentences) |
| Upstream license | CC-BY-SA 3.0 (data derived from Wikipedia + Wikidata) |
| HF card license field | Not explicitly stated on HF card |
| Use basis | Data is derived from CC-BY-SA Wikipedia/Wikidata. Educational use. |
| C1 search note | Two alternatives found and rejected: (a) `risqaliyevds/uzbek_ner` — MIT but wrong format (entity gazetteer, not IOB2 sentences); (b) `ShakhzoDavronov/ner-prepared-uzbek` — 19 609 IOB2 examples but no license declared, no source info. WikiANN selected as fallback. |
| Notebook disclosure | Notebook header and cell comment must state: "Ushbu dataset juda kichik (200 jumlat). Model F1 past (~40-60%) bo'lishi kutilmoqda — bu o'zbek NER uchun ochiq ma'lumotlar kamligini aks ettiradi." |
| Status | **EDUCATIONAL-USE** — Approver: project owner (2026-06-13) |

---

### uz_en_opus100 — OPUS-100 Uzbek-English (C2)
| Field | Value |
|---|---|
| HF ID | `Helsinki-NLP/opus-100` (uz-en config) |
| HF URL | https://huggingface.co/datasets/Helsinki-NLP/opus-100 |
| Size used | 20 000 sentence pairs (subsample of 177k train) |
| License | Not specified on HF card; source corpora are CC/public-domain mix |
| OPUS project | https://opus.nlpl.eu — data is from publicly available sources |
| Use basis | Standard NLP research corpus; demo-quality use (20k pairs, few epochs). Risk: license ambiguous but OPUS is the standard parallel corpus resource for academic NLP. |
| Notebook disclosure | Header comment must state: "Ushbu korpusning litsenziyasi noaniq. Demo maqsadida 20 000 jumlat foydalanilgan; BLEU ko'rsatkichi ishlab chiqarish darajasida emas." |
| Status | **EDUCATIONAL-USE** — Approver: project owner (2026-06-13) |

---

### uz_wiki_summ — Wikipedia Uzbek summarization pairs (C3)
| Field | Value |
|---|---|
| HF ID | `wikimedia/wikipedia` (uz subset) |
| HF URL | https://huggingface.co/datasets/wikimedia/wikipedia |
| Derivation | Lead-paragraph method applied to same uzwiki dump from Stage 0B pipeline; NOT a separate download |
| License | CC-BY-SA 3.0 (Wikipedia content) |
| Attribution | Wikipedia contributors; derived work must carry CC-BY-SA |
| Use basis | CC-BY-SA allows derivative works with attribution. Course materials cite Wikipedia as source. |
| Status | **CONFIRMED** — CC-BY-SA 3.0 ✓ |

---

### uz_sentiment_uzum — Uzbek Sentiment (Uzum Market)
| Field | Value |
|---|---|
| HF ID | `risqaliyevds/uzbek-sentiment-analysis` |
| HF URL | https://huggingface.co/datasets/risqaliyevds/uzbek-sentiment-analysis |
| License | MIT |
| Size | 352 151 rows; 5 000 subsample used in P13 |
| Labels | Rating 1–5 → binarize: {4,5}→ijobiy, {1,2}→salbiy, 3 discarded |
| Verification | VERIFIED 2026-06-13 — real Uzbek e-commerce reviews |
| Status | **CONFIRMED** — MIT ✓ |

---

### uz_kb — Knowledge Base (Uzbek news + lex.uz)
| Field | Value |
|---|---|
| Source | Uzbek news articles + lex.uz legal documents |
| License | Public web content; scraped for educational use |
| Use basis | Non-commercial course material; not redistributed |
| Pull confirmation | Kaggle Dataset attachment — confirm at commit time |
| Status | **CONFIRMED (educational use)** |

---

## Rejected / Not Used

| Dataset | Reason |
|---|---|
| `risqaliyevds/uzbek_ner` | MIT license ✓ but wrong format: entity gazetteer (JSON lists), not IOB2 labeled sentences. Cannot train sequence labeler. |
| `ShakhzoDavronov/ner-prepared-uzbek` | 19 609 IOB2 examples, newer (2025), but no license declared and no source attribution. License blocker. |
| FLORES-200 (uz) | ~1 012 test sentences only — too small for training. |

---

## Pull-Time Checklist

Before committing any dataset to a notebook cell:

- [ ] Dataset ID matches an entry in this file
- [ ] Status is `CONFIRMED` or `EDUCATIONAL-USE`
- [ ] If `EDUCATIONAL-USE`: approver name and date are recorded above
- [ ] Notebook header cell contains the required disclosure text (see above)
- [ ] Dataset is loaded via `load_dataset()` with `OFFLINE_FALLBACK` guard
- [ ] No dataset is committed as raw bytes to the repo (link only)
