'''
===================================================================

Set Project Directories & Import Packages

===================================================================
'''

# Project Directories
USERNAME="ross"
GDRIVE="/Users/RossChu/Library/CloudStorage/GoogleDrive-ross.hm.chu@gmail.com/My Drive"
PROJECT=f"{GDRIVE}/Research_Projects/capstone0"

CODE=f"{PROJECT}"
PROMPTS=f"{CODE}/agent"
OUTPUT=f"{PROJECT}/output"
DATA=f"{PROJECT}/data"
CLEAN=f"{DATA}/clean"
TRAIN=f"{DATA}/train"

# poetry add torch faiss-cpu transformers sentence-transformers peft datasets xgboost pandarallel nltk spacy pyspellchecker textstat pandas joblib openai peft pydantic seaborn geonamescache unidecode matplotlib ipykernel statsmodels lxml

# Set environment variables
import os, pathlib, getpass
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Basic utilities
import re,io,sys,math,time,pytz,json,random,dill,string
import geonamescache, unidecode, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Handling concurrent requests
import concurrent.futures as cf

from lxml import etree
from copy import deepcopy
from pprint import pprint
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict, Any


# Scikit modules
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

# ML modules
import xgboost as xgb
import scipy.sparse as sparse
import scipy.stats as stats

# Inference modules
import statsmodels.api as sm
import statsmodels.tools as smt


# Parallel processing
import joblib, psutil, tqdm

# NLP modules
import nltk, spacy, textstat
from spellchecker import SpellChecker
MASK_TOKENS = [
    '[TITLE]','[CONTENT]', 
    '[ROLE]','[SALARY]','[YEARS]','[LEETCODE]','[INDUSTRY]',
    '[MANAGER]','[ICLEVEL]','[URL]','[ORG]','[GPE]','[INFO]','[SKILL]'
]

MASK_TOKENS += [mtok.title() for mtok in MASK_TOKENS]
MASK_TOKENS += [mtok.lower() for mtok in MASK_TOKENS]


# Transformer modules
import sentence_transformers as stf
from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType


# LLM modules
from openai import OpenAI
from pydantic import BaseModel


# Logging and re-tries
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, after_log, before_sleep_log


# FAISS/Captum AI
import faiss
import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as captumviz
from html2image import Html2Image


#----------------------------
# Figure / Display Options
#----------------------------

# Pandas & Numpy
pd.options.display.max_rows = 110
pd.options.display.max_columns = 20
pd.options.display.max_colwidth = 15
pd.options.display.precision = 4
pd.options.display.float_format = '{:.7f}'.format
pd.options.display.width = None

np.set_printoptions(precision=6, floatmode='fixed')
np.set_printoptions(suppress=True)


# Default settings for matplotlib
fig_defaults = {
    'figure.figsize': (10,8),       # Figure size
    'figure.autolayout': True,      # Tight layout
    
    'axes.spines.left': True,       # Keep left spine (y-axis)
    'axes.spines.bottom': True,     # Keep bottom spine (x-axis)
    'axes.spines.right': False,     # Remove right spine
    'axes.spines.top': False,       # Remove top spine

    'axes.facecolor': 'white',      # White background
    'axes.grid': False,             # Remove gridlines
    'grid.linestyle': '--',         # Dashed gridlines (if enabled)

    'axes.titlepad': 20,            # Extra spacing for title
    'axes.labelpad': 20,            # Extra spacing for axis labels

    'axes.titlesize': 20,           # Title font size
    'axes.labelsize': 16,           # X and Y axis label font size
    'legend.fontsize': 16,          # Legend font size
    'xtick.labelsize': 12,          # X-axis tick label size
    'ytick.labelsize': 12,          # Y-axis tick label size
}

LEGENDFONT = 12 # legend font size for multiple subplots

# Seaborn settings
sns.set_theme(style="whitegrid", rc=fig_defaults)

sbb = sns.color_palette()[0]
sbo = sns.color_palette()[1]
sbg = sns.color_palette()[2]
sbr = sns.color_palette()[3]
sbp = sns.color_palette()[4]
sbgrey = sns.color_palette()[7]
sby = sns.color_palette()[8]
