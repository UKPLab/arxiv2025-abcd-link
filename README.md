# ABCD-LINK: Annotation Bootstrapping for Cross-Document Fine-Grained Links

[![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](https://put-here-your-paper.com)
[![License](https://img.shields.io/badge/License-Apache--2.0-green?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.11-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

This repository contains all scripts and data necessary for reproducing the results from ABCD-Link

---

> **Abstract:** Understanding fine-grained relations between documents is crucial for many application domains. However, the study of automated assistance is limited by the lack of efficient methods to create training and evaluation datasets of cross-document links. To address this, we introduce a new domain-agnostic framework for selecting a best-performing approach and annotating cross-document links in a new domain from scratch. We first generate and validate semi-synthetic datasets of interconnected documents. This data is used to perform automatic evaluation, producing a shortlist of best-performing linking approaches. These approaches are then used in an extensive human evaluation study, yielding performance estimates on natural text pairs. We apply our framework in two distinct domains -- peer review and news -- and show that combining retrieval models with LLMs achieves 78\% link approval from human raters, more than doubling the precision of strong retrievers alone. Our framework enables systematic study of cross-document understanding across application scenarios, and the resulting novel datasets lay foundation for numerous cross-document tasks like media framing and peer review. We make the code, data, and annotation protocols openly available.

Contact person: [Serwar Basch](mailto:serwar.basch@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

## Quick Start

### Install requirements

First, ensure you have python 3.11\
Then, install the necessary requirements
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
````

For OpenAI-based inference, set your key:

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE
```

Ensure you have access to an appropriate GPU for the LLM inference step (at least 100GB of VRAM are needed for Qwen2.5)

### Reconstruct Dataset

To reconstruct the NEWS-HE dataset, please download the SPICED dataset from [Zenodo](https://zenodo.org/records/8044777) (filename: spiced.csv) and place it under ``./datasets/news_he``\
Then you can run

```bash
python scripts/reconstruct_dataset.py
````

---

## Run the Full Pipeline

To run **all steps** in sequence:

```bash
bash run.sh
```

This runs:

1. Retrieval model inference
2. Prompt generation
3. LLM inference (local + API)
4. Evaluation (ranked + classified)
5. Calculate IAA and Acceptance Rate from Human Evaluation
6. Calculate true recall rate on the subset of manually annotated data

Results are saved to:

* `./predictions/`
* `./data/prompts_json/`
* `./llm_results/`
* `./eval_outputs/`
* `./datasets/*_he`


---

## Run Individual Steps

You can also run specific stages:

```bash
bash run.sh --retrieval
bash run.sh --prompts
bash run.sh --llm
bash run.sh --eval
bash run.sh --anno
bash run.sh --gold
```

---

## Evaluation Customization

You can adjust evaluation parameters using flags passed to run.sh, for example:

```bash
bash run.sh --eval --type=classified --metric=f1
bash run.sh --eval --type=ranked --cutoffs=1 3 5 7 10 20 --metric=recall
```

Supported flags:

* --type=ranked|classified|all (default: all)
* --cutoffs= (for ranked)
* --metric=precision|recall|f1

---

## Gold Label Evaluation

To evaluate model outputs against human-annotated gold labels:

```bash
bash run.sh --gold
```

This evaluates precision, recall, and F1 on:

* `datasets/news_he/news_gold_labels.csv`
* `datasets/reviews_he/reviews_gold_labels.csv`

Results are saved to:

* `datasets/news_he/eval_gold_labels.json`
* `datasets/reviews_he/eval_gold_labels.json`

---

## Included Datasets

Datasets:

* `news_ecb`
* `news_synth`
* `reviews_synth`
* `reviews_f1000`

Each dataset contains:

* `docs.json`: documents split into sentences
* `<name>_links.json`: ground truth cross-document sentence-level links

## Metrics

For **retrievers** (`ranked`):

* Precision\@k, Recall\@k, F1\@k

For **LLMs** (`classified`):

* Precision, Recall, F1 over entire output

## Novel Datasets

* `news_he`
* `reviews_he`

Each dataset contains:

* `docs.json`: documents split into sentences
* `annotations.json`: annotations results from the human evaluation study

---

## NOTE

The `generate_prompts.py` script uses `dragon_plus` as the default source for top-ranked sentences based on our experiments. The value is hardcoded to ensure reproducibility of our results.

---

## 📁 Project Structure

```text
project-root/
│
├── datasets/
│   ├── news_ecb/
│   │   ├── docs.json
│   │   └── news_ecb_links.json
│   └── ...
│
├── data/                             # Input artifacts for prompt generation
│   ├── positive_examples.json
│   └── prompts_json/                 # All generated prompt files
│
├── predictions/                      # Retriever output path
│
├── llm_results/                      # LLM classification output path
│
├── eval_outputs/                     # Metrics and evaluation output path
│
├── retrieval/                        # Retriever scripts
│   ├── __init__.py
│   ├── scorers.py
│   ├── models.py
│   └── utils.py
│
├── prompts/                          # Prompt construction scripts
│   ├── __init__.py
│   ├── builder.py
│   └── generate_prompts.py
│
├── llm_inference/                    # LLM scripts
│   ├── __init__.py
│   ├── chat_utils.py                 # Shared prompt-building and vLLM setup
│   ├── executor.py                   # Local vLLM inference (Phi-4, Qwen)
│   ├── openai_utils.py
│   └── openai_executor.py            # GPT-4o inference
│
├── scripts/                          # Executable scripts
│   ├── run_retrievals.py             # Runs all retrieval models on all datasets
│   ├── run_llm_inference.py          # Runs all prompts through all LLMs
│   ├── annotation_results.py         # Calculates agreement and acceptance rates on annotations  
│   ├── evaluate_gold_labels.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

## Cite

Please use the following citation:

```
tbd
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.