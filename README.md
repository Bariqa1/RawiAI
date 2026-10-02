# RawiAI - Advanced Hybrid RAG for Arabic Poetry


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
│   │   ├── lexicon_retriever.py    # Hybrid sparse BM25 and exact root retriever
│   │   └── dual_rag_orchestrator.py # Multi-hop coordinator and prompt assembler
│   ├── evaluation/
│   │   ├── metrics.py              # Custom DeepEval metrics (Faithfulness, Relevancy, Refusal)
│   │   ├── deepeval_runner.py      # Dual runner (Deterministic Triad + Live LLM Judge)
│   │   └── retrieval_benchmark.py  # Standalone retrieval benchmark (Hit@K, MRR)
│   ├── data/
│   │   ├── asas_al_balagha.json    # Indexed Asas Al-Balagha dictionary dataset
│   │   └── golden_evaluation_dataset.json # Ground truth evaluation test suite
│   └── processors/
│       └── corpus_processor.py     # Automated dataset column detection and verse parser
├── tests/                          # 46 automated unit and integration tests
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
├── run_evaluation.py               # Main CLI benchmark and evaluation report generator
├── conftest.py                     # Test runner configuration and import path setup
├── .env.example                    # Environment variable template
├── .gitignore                      # Git configuration protecting secrets and cache
├── advanced-arabic-poetry-rag.ipynb # Original research notebook
└── README.md                       # Project documentation
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

### Run All 46 Unit Tests via Pytest
```bash
pytest tests
```
Or via standard Python unittest:
```bash
python3 -m unittest discover tests
```

### Run Interactive Demos
- Dual-RAG Multi-Hop Pipeline:
  ```bash
  python3 demo_dual_rag.py
  ```
- Domain NLP & Prosody Analysis:
  ```bash
  python3 demo_nlp.py
  ```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
All classical poetry texts and lexicon entries are derived from public domain classical Arabic heritage works.

