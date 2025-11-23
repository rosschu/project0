# Building and Evaluating AI Agents to Improve Job Referral Requests to Strangers

## Overview

`setup/utils.py` is a preamble that imports packages and defines project directories for `code/`, `data/`, and `output/`. This is loaded at the beginning of every sript, and users should update the project directory path in `utils.py`. Python dependencies are set through `poetry` in `poetry.lock` and `pyproject.toml`.

Each module corresponds to a unique jupyter notebook (e.g. `dataprep/` contains python scripts for the `dataprep.ipynb` notebook). These notebooks replicate all results, which is described in detail below. 

## Replication Files

### Data Preparation (`dataprep/`)

This module prepares text data for analyses. `dataprep.ipynb` is the relevant notebook for replicating results. 

It contains the following scripts:

- `parse.py`: parses HTML file to export structured data on postings

- `basic.py`: filters out referral requests and defines basic features. 

- `mask_tokens.py`: Tokenize input text and add mask tokens to replace factual content

- `linguistic.py`: defines linguistic features based on domain knowledge

### Exploratory Analyses (`explore/`)

This module generates summary statistics for exploratory data analysis. `explore.ipynb` is the relevant notebook for replicating results. It contains the following script with helper functions: 

- `sumstat.py` defines functions for descriptive statistics.

### Model Training and Performance Eval (`modelfit/`)

This module fits text classification models and evaluates their performance. `modelfit.ipynb` is the relevant notebook for replicating results. It contains the following scripts:

- `encode.py`: fine-tunes transformers and encodes text into 768 dimensional vector embeddings

- `train.py`: performs train/test splits, trains classification models, and compresses them with `joblib`

- `performance.py`: evaluates classification performance for trained models, including bootstrapped confidence intervals


## Data Tables (`data/clean/`)

(note: for self-reference only. Not to be shared publicly)

`posts_html/referrals_full.html` contains the HTML file with post links from the Jobs & Referrals channel, which has been obtained from infinite scrolling.

Data files prepared for analyses are under `data/clean/`.

These tables contain data scraped from the Jobs & Referrals channel.

- `posts.csv`: the HTML file above is parsed into a dataframe with post links & basic info (from `html_parse.py`). It contains a preview of text content that is truncated from the full content. Unique at the `post_id` level.

- `post_fulltext.csv`: contains the full text content for each `post_id` in the above CSV file (scraped separately)

- `comments.csv`: contains all comments for each `post_id` (scraped separately)

These tables contain additional columns defined from the above tables:

- `posts_basic.csv`: flags for referral requests & defines target metrics

- `posts_mask.csv`: text content replaced with mask tokens for factual content

- `posts_encode_*.pkl`: contains transformer embeddings for titles + text content

- `posts_linguistic.csv`: defines linguistic features from domain knowledge (from `linguistic.ipynb`)

## Trained Models (`data/train/`)

`train.py` exports trained models and input data, which are loaded in other scripts.

- `model.joblib`: dictionary with trained classification models. Keys indicate model names.

- `data.joblib`: dictionary with train/test data. Keys indicate models, and values are tuples with (X_train, X_test, y_train, y_test, tfidf vectorizer)

- `predict_proba.csv`: CSV with one row for each referral requests. Columns contain predicted success rates for each model.
