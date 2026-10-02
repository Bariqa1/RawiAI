# RawiAI - Advanced Hybrid RAG for Arabic Poetry

[![CI/CD Pipeline](https://github.com/Bariqa1/RawiAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Bariqa1/RawiAI/actions/workflows/ci.yml)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Evaluation DeepEval](https://img.shields.io/badge/Evaluation-DeepEval-success.svg)](https://github.com/confident-ai/deepeval)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### Dual-Index Multi-Hop RAG & Domain-Specific NLP for Classical Arabic Poetry
#### Grounded by Asas Al-Balagha (Al-Zamakhshari) and Evaluated via DeepEval

RawiAI is a domain-specialized Retrieval-Augmented Generation (RAG) and Natural Language Processing (NLP) framework designed for classical Arabic poetry and heritage literature. It addresses the unique linguistic challenges of classical Arabic—complex metaphorical expressions, archaic vocabulary, metrical structures, and vocalization variances—by combining a primary poetry corpus with a secondary rhetorical dictionary index (*Asas Al-Balagha* by Al-Zamakhshari) via multi-hop retrieval and automated DeepEval evaluation.

---

## Architecture Overview

```
                                 [ User Query ]
                                        |
                             [ Intent Classification ]
                                        |
                 +----------------------+----------------------+
                 |                                             |
     [ Attribution / Completion ]                   [ Meaning / Metaphor ]
                 |                                             |
                 v                                             v
        [ Hop 1: Poetry RAG ]                       [ Multi-Hop Dual RAG ]
      (1.6M verses: Sadr, Ajuz)                                |
                 |                               +-------------+-------------+
                 |                               |                           |
                 |                               v                           v
                 |                     [ Hop 1: Poetry RAG ]      [ Hop 2: Asas Al-Balagha ]
                 |                     (Target Verse Retrieval)   (Literal & Metaphoric Lexicon)
                 |                               +-------------+-------------+
                 |                                             |
                 +----------------------+----------------------+
                                        |
                                        v
                          [ Grounded Prompt Assembly ]
                                        |
                                        v
                            [ LLM Answer Generation ]
```

---

## Tools and Technology Stack

The RawiAI framework is built using a pure, modular, production-ready stack with zero unnecessary C-dependencies:

| Category | Tool / Library | Role in System |
| :--- | :--- | :--- |
| **Language & Runtime** | **Python 3.9+** | Core programming runtime and execution environment |
| **Data Validation** | **Pydantic v2** | Strict typing and schemas for verses, prosody, and lexicon records |
| **RAG Evaluation** | **DeepEval (v2.9+)** | LLM-as-a-Judge and domain-aware metrics (Faithfulness, Relevancy, Refusal) |
| **Testing Framework** | **Pytest & Unittest** | Automated regression testing across 46 unit and integration tests |
| **Information Retrieval** | **PurePythonBM25** | Zero-dependency BM25 sparse lexical search engine |
| **Data Normalization** | **Custom Normalizer** | Multi-level Arabic normalizer (Search, Prosodic, and Phonetic Rhyme) |
| **Stemming Engine** | **PoetryArabicStemmer** | Light Arabic stemmer with radical root protection and clitic stripping |
| **Metrical Prosody** | **ProsodyAnalyzer** | Arud meter detection (Al-Kamil, Al-Tawil, etc.) and Rawiyy extraction |
| **Rhetorical Knowledge** | **Asas Al-Balagha** | Structured JSON corpus of Al-Zamakhshari's dictionary of metaphors |
| **LLM Inference** | **OpenAI API / Gemini** | GPT-4o-mini and Google Gemini API compatibility for grounded generation |
| **Environment Config** | **python-dotenv** | Secure local environment variable and API key management |

---

## Key Architectural Differentiators

### 1. Dual-Index Multi-Hop Retrieval
Standard RAG pipelines query a single vector database and frequently hallucinate interpretations when encountering archaic words. RawiAI decouples textual retrieval from rhetorical analysis:
- **Hop 1 (Poetry Index)**: Locates the exact verse, poet identity, historical era, meter, and hemistichs.
- **Hop 2 (Lexicon Index)**: Automatically extracts key roots from the retrieved verse and fetches their literal and figurative definitions from *Asas Al-Balagha*.

### 2. Why Asas Al-Balagha by Al-Zamakhshari?
- **Designed for Metaphor**: Unlike general dictionaries (e.g., *Lisan Al-Arab*), *Asas Al-Balagha* strictly partitions entries into literal (*Al-Haqiqah*) and figurative (*Al-Majaz*), directly matching poetic metaphors.
- **Optimal Chunk Size**: Entries range between 100 and 250 words, fitting embedding context windows without truncation or semantic degradation.
- **Alphabetical Root Ordering**: Organized by initial letter roots (A, B, T...), seamlessly integrating with modern stemming and indexing.

### 3. Domain-Specific NLP Components
- **Phonetic Normalization**: Normalizes hamza forms, strips diacritics and tatweel, while preserving crucial structural markers for meter and rhyme.
- **Hemistich Segmentation**: Automatically separates *Sadr* (first half) and *Ajuz* (second half) across multiple historical typographical separators (`#`, `...`, `\t`, `*`, `//`).
- **Poetry-Aware Light Stemmer**: Strips proclitics and enclitics while protecting radical roots (`ولد`, `بلاد`, `كتاب`, `يد`, `دم`) and mapping irregular plurals (`سيوف` -> `سيف`).
- **Arud Meter & Rawiyy Identifier**: Converts vocalized lines to binary metric patterns (`1` for mutaharrik, `0` for sakin) to identify the meter and extract the rhyming consonant (*Rawiyy*).

---

## Quality Benchmarks & DeepEval Results

Evaluated over a curated Golden Dataset covering author identification, verse completion, metaphorical explanation, thematic search, and out-of-domain refusal guardrails:

| Metric | Measured Score | Target Baseline | Assessment |
| :--- | :---: | :---: | :---: |
| **Hit@1 (Top-1 Retrieval Accuracy)** | **100.0%** | > 80% | Optimal retrieval performance |
| **Hit@3 (Top-3 Retrieval Accuracy)** | **100.0%** | > 90% | Optimal retrieval performance |
| **Hit@5 (Top-5 Retrieval Accuracy)** | **100.0%** | > 95% | Optimal retrieval performance |
| **MRR (Mean Reciprocal Rank)** | **1.00** | > 0.85 | First-rank precision |
| **Refusal Guardrail Accuracy** | **100.0%** | 100% | Zero hallucination on out-of-domain queries |
| **Multi-Hop Lexicon Recall** | **88.9%** | > 80% | High-fidelity rhetorical grounding |
| **Poetic Faithfulness** | **87.0%** | > 70% | Strictly grounded in context |
| **Answer Relevancy** | **100.0%** | > 70% | High semantic intent satisfaction |
| **Test Suite Pass Rate** | **100.0%** | 100% | 46 / 46 tests passing |

---

## Repository Structure

```
RawiAI/
├── rawiai/
│   ├── models/
│   │   ├── schemas.py              # EnrichedVerse, ProsodyInfo, DifficultWord models
│   │   └── lexicon_schema.py       # AsasEntry (Literal, Metaphor, Poetic Citations)
│   ├── nlp/
│   │   ├── normalizer.py           # Multi-stage Arabic text normalizer
│   │   ├── segmenter.py            # Automatic hemistich separator (Sadr & Ajuz)
│   │   ├── stemmer.py              # Poetry-aware stemmer with radical root protection
│   │   ├── prosody.py              # Arud meter classifier and Rawiyy extractor
│   │   └── lexicon.py              # Heritage vocabulary annotator
│   ├── rag/
│   ├── rag/
│   │   ├── lexicon_retriever.py    # Hybrid sparse BM25 and exact root retriever (3,719 roots)
│   │   └── dual_rag_orchestrator.py # Multi-hop coordinator and prompt assembler
│   ├── agents/
│   │   ├── schemas.py              # Pydantic data models for agent contracts & critique reports
│   │   ├── muse_agent.py           # The Muse: theme inference, meter & rhyme selection, rhetorical motifs
│   │   ├── poet_agent.py           # The Poet: verse composition and cooperative revision
│   │   ├── critic_agent.py         # Arud Critic: deterministic prosody verification & actionable critique
│   │   └── poetic_council.py       # Council Orchestrator: iterative actor-critic refinement loop
│   ├── evaluation/
│   │   ├── metrics.py              # Custom DeepEval metrics (Faithfulness, Relevancy, Refusal)
│   │   ├── deepeval_runner.py      # Dual runner (Deterministic Triad + Live LLM Judge)
│   │   └── retrieval_benchmark.py  # Standalone retrieval benchmark (Hit@K, MRR)
│   ├── data/
│   │   ├── asas_al_balagha.json    # Gold standard Asas Al-Balagha dictionary dataset
│   │   ├── asas_al_balagha_full.json # Complete digitized Asas Al-Balagha (3,719 roots, 2,018 metaphors)
│   │   ├── asas_al_balagha_shamela_full.txt # Verified full text (35,599 lines, Al-Zamakhshari)
│   │   └── golden_evaluation_dataset.json # Ground truth evaluation test suite
│   └── processors/
│       ├── corpus_processor.py     # Automated dataset column detection and verse parser
│       └── asas_parser.py          # Shamela/OpenITI text parser & structured JSON builder
├── tests/                          # 55 automated unit and integration tests
│   ├── test_poetic_council.py      # Multi-Agent Poetic Council generation & critique tests
│   ├── test_asas_parser.py         # Full book parser & dataset loading tests
│   ├── test_deepeval_suite.py      # DeepEval CLI-native test file
│   ├── test_evaluation.py          # Unit tests for benchmark and metric logic
│   ├── test_asas_balagha.py        # Asas Al-Balagha retrieval tests
│   ├── test_dual_rag.py            # Dual-RAG orchestrator and routing tests
│   ├── test_normalizer.py          # Normalization test cases
│   ├── test_segmenter.py           # Hemistich splitting test cases
│   ├── test_stemmer.py             # Stemming and radical protection tests
│   ├── test_prosody.py             # Arud and rhyme extraction tests
│   ├── test_lexicon.py             # Lexicon lookup and annotation tests
│   └── test_corpus_processor.py    # Data ingestion tests
├── demo_nlp.py                     # Interactive demo for NLP processing
├── demo_dual_rag.py                # Interactive demo for Dual-RAG multi-hop retrieval
├── demo_poetic_council.py          # Interactive demo for Multi-Agent poem generation & critique
├── run_evaluation.py               # Main CLI benchmark and evaluation report generator
├── conftest.py                     # Test runner configuration and import path setup
├── .env.example                    # Environment variable template
├── .gitignore                      # Git configuration protecting secrets and cache
├── advanced-arabic-poetry-rag.ipynb # Original research notebook
└── README.md                       # Project documentation
```

---

## Multi-Agent Poetic Council (مجلس الشعراء ونظم الشعر الآلي)

RawiAI features an autonomous **Multi-Agent Poetic Council** operating in an **Actor-Critic Self-Correction Loop**:

1. **The Muse Agent (عميل الإلهام وبلاغة المعجم):**
   - Infers poetic theme (فخر، حكمة، غزل، معاصرة وتقنية...).
   - Selects the ideal classical meter (`الكامل`, `الطويل`, `البسيط`) and rhyme letter (`الروي`).
   - Retrieves rare rhetorical metaphors and motifs from *Asas Al-Balagha* to seed into the composition.
2. **The Poet Agent (الشاعر الناظم):**
   - Composes symmetrical hemistich verses (*Sadr* and *Ajuz*) weaving the rhetorical motifs into the target meter.
3. **The Arud Critic Agent (الناقد العروضي ومحكم الشعر العربي):**
   - Audits every verse using deterministic prosodic analysis (`ProsodyAnalyzer`).
   - Verifies metric adherence, detects feet irregularities, checks rawiyy consonant matching, and tests syllabic symmetry.
   - If a verse is broken, sends actionable revision directives back to the Poet Agent until the poem achieves 100% balance.

```bash
# Run the interactive Multi-Agent Council demo
python3 demo_poetic_council.py
```
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Bariqa1/RawiAI.git
cd RawiAI
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies: `pydantic`, `deepeval`, `pytest`, `python-dotenv`, `openai`)*

### 4. Configure Environment Variables
Copy the environment template and insert your API credentials if running live evaluations:
```bash
cp .env.example .env
```
Edit `.env`:
```env
OPENAI_API_KEY="your-openai-api-key"
```

---

## Running Benchmarks & Tests

### Run Full Quality Benchmark (Retrieval + Generation Triad)
```bash
python3 run_evaluation.py
```

### Run with Live LLM-as-a-Judge (OpenAI GPT-4o-mini)
```bash
python3 run_evaluation.py --live
```

### Run DeepEval CLI Test Suite
```bash
deepeval test run tests/test_deepeval_suite.py
```

### Run All 69 Unit Tests via Pytest
```bash
pytest tests
```
Or via standard Python unittest:
```bash
python3 -m unittest discover tests
```

### Run Interactive Demos
- **Master Orchestrator & Guardrails Showcase (المايسترو الموحد وحواجز الأمان):**
  ```bash
  python3 demo_master_orchestrator.py
  ```
- **Multi-Agent Poetic Council Showcase (مجلس الشعراء التفاعلي):**
  ```bash
  python3 demo_poetic_council.py
  ```
- **Dual-RAG Multi-Hop Pipeline:**
  ```bash
  python3 demo_dual_rag.py
  ```
- **Domain NLP & Prosody Analysis:**
  ```bash
  python3 demo_nlp.py
  ```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
All classical poetry texts and lexicon entries are derived from public domain classical Arabic heritage works.

