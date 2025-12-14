# **`README.md`**

# Measuring Corruption from Text Data: Automated Quantification of Institutional Quality

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.09652-b31b1b.svg)](https://arxiv.org/abs/2512.09652)
[![Journal](https://img.shields.io/badge/Journal-Political%20Economy%20(econ.GN)-003366)](https://arxiv.org/abs/2512.09652)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/measuring_corruption_from_text_data)
[![Discipline](https://img.shields.io/badge/Discipline-Political%20Economy%20%7C%20NLP-00529B)](https://github.com/chirindaopensource/measuring_corruption_from_text_data)
[![Data Sources](https://img.shields.io/badge/Data-CGU%20Audit%20Reports-lightgrey)](https://www.gov.br/cgu/pt-br)
[![Data Sources](https://img.shields.io/badge/Data-IBGE%20(Municipal%20Covariates)-lightgrey)](https://www.ibge.gov.br/)
[![Data Sources](https://img.shields.io/badge/Data-Ferraz%20%26%20Finan%20(2011)-lightgrey)](https://www.aeaweb.org/articles?id=10.1257/aer.101.4.1274)
[![Data Sources](https://img.shields.io/badge/Data-Timmons%20%26%20Garfias%20(2015)-lightgrey)](https://www.sciencedirect.com/science/article/abs/pii/S030438781400138X)
[![Core Method](https://img.shields.io/badge/Method-Dictionary--Based%20Classification-orange)](https://github.com/chirindaopensource/measuring_corruption_from_text_data)
[![Analysis](https://img.shields.io/badge/Analysis-Principal%20Component%20Analysis%20(PCA)-red)](https://github.com/chirindaopensource/measuring_corruption_from_text_data)
[![Validation](https://img.shields.io/badge/Validation-Econometric%20Fixed%20Effects-green)](https://github.com/chirindaopensource/measuring_corruption_from_text_data)
[![Robustness](https://img.shields.io/badge/Robustness-Supervised%20Learning%20(LR%2FNB)-yellow)](https://github.com/chirindaopensource/measuring_corruption_from_text_data)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-%23339933.svg?style=flat&logo=python&logoColor=white)](https://www.nltk.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-blue?logo=python&logoColor=white)](https://www.statsmodels.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)

**Repository:** `https://github.com/chirindaopensource/measuring_corruption_from_text_data`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"Measuring Corruption from Text Data"** by:

*   **Arieda Muço** (Central European University)

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from the heuristic extraction of irregularities from unstructured audit reports and dictionary-based classification to dimensionality reduction via PCA, rigorous econometric validation against human experts, and supervised learning robustness checks.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `execute_full_research_pipeline`](#key-callable-execute_full_research_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Muço (2025). The core of this repository is the iPython Notebook `measuring_corruption_from_text_data_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings. The pipeline is designed to be a generalizable toolkit for quantifying institutional quality from unstructured administrative text, specifically focusing on the Brazilian municipal audit program (CGU).

The paper addresses the fundamental challenge of measuring corruption—a hidden phenomenon—by leveraging the "administrative exhaust" of government audits. This codebase operationalizes the paper's framework, allowing users to:
-   Rigorously validate and manage the entire experimental configuration via a single `config.yaml` file.
-   Extract granular irregularity segments from heterogeneous PDF-derived text using regex-based heuristics.
-   Classify irregularities as "severe" or "non-severe" using a domain-specific Portuguese dictionary.
-   Construct a latent **Corruption Index** via Principal Component Analysis (PCA) on text-derived features.
-   Validate the automated measure against hand-coded datasets (Ferraz & Finan, Timmons & Garfias) using fixed-effects regression models.
-   Verify robustness using supervised machine learning classifiers (Logistic Regression, Naive Bayes) and Leave-One-Out (LOO) sensitivity analysis.

## Theoretical Background

The implemented methods combine techniques from Natural Language Processing (NLP), Unsupervised Learning, and Econometrics.

**1. Text-as-Data Extraction & Classification:**
The pipeline treats audit reports as data. It handles structural shifts in reporting (introduction of summaries in later lotteries) to isolate "irregularity segments."
-   **Dictionary Method:** A deterministic rule classifies an irregularity $I_{ij}$ as severe if it contains specific n-grams (e.g., "empresa fantasma", "fraud") from a curated lexicon $\mathcal{L}$:
    $$ Severe(I_{ij}) = \mathbb{1}\{\exists \ell \in \mathcal{L} : Match(\ell, \phi(I_{ij}))\} $$
    where $\phi(\cdot)$ represents the text normalization pipeline (stemming, stopword removal).

**2. Dimensionality Reduction (PCA):**
To synthesize a single measure from correlated text features (image counts, page counts, severe irregularity counts), PCA is applied to the standardized feature matrix $Z$. The first principal component $v_1$ serves as the index:
    $$ \text{Corruption Index}_i = Z_i^\top v_1 $$
This component captures ~80% of the common variation, representing the latent "severity" dimension.

**3. Econometric Validation:**
The automated index is validated by regressing human-coded corruption counts ($HC_i$) on the index, controlling for state fixed effects ($\tau_t$) to account for auditor team heterogeneity:
    $$ HC_i = \alpha + \beta \text{Corruption Index}_i + \tau_t + \varepsilon_i $$
Strong predictive power ($R^2 > 0.70$) in high-agreement samples confirms criterion validity.

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/measuring_corruption_from_text_data/blob/main/measuring_corruption_from_text_data_ipo_two.png" alt="Corruption Measurement Process Summary" width="100%">
</div>

## Features

The provided iPython Notebook (`measuring_corruption_from_text_data_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The pipeline is decomposed into 19 distinct, modular tasks, each with its own orchestrator function.
-   **Configuration-Driven Design:** All study parameters (lottery cutoffs, regex markers, PCA thresholds) are managed in an external `config.yaml` file.
-   **Rigorous Data Validation:** A multi-stage validation process checks schema integrity, key uniqueness, and logical consistency of the corpus and validation data.
-   **Advanced NLP Pipeline:** Implements NFD normalization, accent stripping, and Porter stemming tailored for Portuguese administrative text.
-   **Robustness Verification:** Includes automated Leave-One-Out (LOO) analysis and a parallel Supervised Learning pipeline to cross-validate the dictionary-based measure.
-   **Reproducible Artifacts:** Generates structured dictionaries and serializable outputs for every intermediate result, ensuring full auditability.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Validation & Cleansing (Tasks 1-6):** Ingests raw corpus and validation data, normalizes identifiers (7-digit IBGE codes), cleanses text encoding, and validates numeric metadata.
2.  **Extraction & NLP (Tasks 7-9):** Applies heuristic parsing to extract irregularity segments, normalizes text and lexicon into a shared matching space, and classifies irregularities as severe/non-severe.
3.  **Index Construction (Task 10):** Builds the feature matrix, standardizes data, computes PCA, and generates the Corruption Index.
4.  **Econometric Validation (Tasks 11-14):** Merges the index with external datasets, constructs agreement samples, and estimates validation regressions (Tables 1, 2, and 3).
5.  **Robustness Checks (Tasks 16-19):** Performs LOO sensitivity analysis and executes a full supervised learning pipeline (training classifiers, rebuilding the index with ML predictions) to confirm result stability.

## Core Components (Notebook Structure)

The `measuring_corruption_from_text_data_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the 19 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `execute_full_research_pipeline`

The project is designed around a single, top-level user-facing interface function:

-   **`execute_full_research_pipeline`:** This master orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data flow between the main analysis, LOO robustness, and supervised learning robustness modules.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `nltk`, `pyyaml`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/measuring_corruption_from_text_data.git
    cd measuring_corruption_from_text_data
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scikit-learn statsmodels nltk pyyaml
    ```

4.  **Download NLTK Data:**
    The pipeline will attempt to download necessary NLTK data (stopwords) automatically, but you can pre-install them:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

## Input Data Structure

The pipeline requires two primary DataFrames:
1.  **`df_raw_corpus`**: The corpus of audit reports with columns: `report_id`, `municipality_id`, `report_full_text`, `report_summary_text`, `lottery`, `year`, `image_count`, `page_count`, etc.
2.  **`df_validation_raw`**: External validation data with columns: `municipality_id`, `ff_corruption_count`, `gt_corruption_count`, `cgu_severe_count`, and municipal covariates (`literacy_rate`, `gdp_per_capita`, etc.).

## Usage

The `measuring_corruption_from_text_data_draft.ipynb` notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell of the notebook, which demonstrates how to use the top-level `execute_full_research_pipeline` orchestrator:

```python
# Final cell of the notebook

# This block serves as the main entry point for the entire project.
if __name__ == '__main__':
    # 1. Load the master configuration from the YAML file.
    import yaml
    with open('config.yaml', 'r') as f:
        study_config = yaml.safe_load(f)
    
    # 2. Load raw datasets (Example using synthetic generator provided in the notebook)
    # In production, load from CSV/Parquet: pd.read_csv(...)
    df_raw_corpus = ... 
    df_validation_raw = ...
    
    # 3. Define Lexicon
    raw_lexicon_list = ["empresa fantasma", "fraud", "conluio", ...]

    # 4. Execute the entire replication study.
    results = execute_full_research_pipeline(
        df_raw_corpus=df_raw_corpus,
        df_validation_raw=df_validation_raw,
        language="Portuguese",
        raw_lexicon_list=raw_lexicon_list,
        study_configuration=study_config
    )
    
    # 5. Access results
    print(f"Main Pipeline R2 (Strict Agreement): {results['main_pipeline']['table1_results']['Strict Agreement']['R2']}")
```

## Output Structure

The pipeline returns a master dictionary containing all analytical artifacts:
-   **`main_pipeline`**: Contains `df_corpus_with_index` (the final index), `table1_results` (validation regressions), `table2_results` (CGU validation), `table3_results` (correlates), and `pca_artifacts`.
-   **`loo_analysis`**: Contains `detailed_results` (per-iteration stats) and `summary` (min/max ranges for $\beta$ and $R^2$).
-   **`supervised_robustness`**: Contains `classification_reports`, `ml_pca_index`, and `comparison_stats` (correlation between dictionary and ML indices).

## Project Structure

```
measuring_corruption_from_text_data/
│
├── measuring_corruption_from_text_data_draft.ipynb  # Main implementation notebook
├── config.yaml                                      # Master configuration file
├── requirements.txt                                 # Python package dependencies
│
├── LICENSE                                          # MIT Project License File
└── README.md                                        # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Parsing Logic:** `lottery_cutoff`, `regex_start_marker`.
-   **NLP Settings:** `stemmer_algorithm`, `dictionary_ngram_range`.
-   **PCA Settings:** `pca_input_features`, `eigenvalue_threshold`.
-   **Econometrics:** `robust_se_type`, `validation_fixed_effects`.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **LLM Integration:** Replacing the dictionary classifier with Large Language Models (e.g., BERT, GPT) to test if contextual embeddings improve severity classification.
-   **Temporal Analysis:** Extending the model to analyze trends in corruption severity over time.
-   **Cross-National Application:** Adapting the dictionary and extraction logic for audit reports from other countries (e.g., Mexico, Puerto Rico).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{muco2025measuring,
  title={Measuring Corruption from Text Data},
  author={Muço, Arieda},
  journal={arXiv preprint arXiv:2512.09652},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). Automated Quantification of Institutional Quality: An Open Source Implementation.
GitHub repository: https://github.com/chirindaopensource/measuring_corruption_from_text_data
```

## Acknowledgments

-   Credit to **Arieda Muço** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, Scikit-Learn, and Statsmodels**.

--

*This README was generated based on the structure and content of the `measuring_corruption_from_text_data_draft.ipynb` notebook and follows best practices for research software documentation.*
