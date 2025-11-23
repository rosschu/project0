'''
============================================================

Train Classification Models

============================================================
'''

# Packages and Directories
from setup.utils import *
from modelfit.encode import load_bert_model, encode_text

# Prepare data for model training & evaluation
def prepare_data(posts: pd.DataFrame, feats: pd.DataFrame, model: str, nums=None, categs=None, ngram=None) -> Tuple:
    '''
    Description:
        - Prepares data for model training & evaluation
        - All adjustments avoid leakage between training & test sets
    
    Inputs:
        - posts: data frame with post IDs & target metrics
        - feats: data frame with post IDs & features (should not contain extra columns)
        - model: name of model this data is being prepared for
        - nums: list of numerical features to apply standard scaler (mean 0, var 1). Specify as list of column names. Only applicable for domain knowledge features
        - categs: list of categorical features to apply one-hot encoding. Specify as list of column names. Only applicable for domain knowledge features
        - ngram: which n-grams to sue for TF-IDF vectorization
    
    Outputs:
        - Tuple of (X_train, X_test, tfidf)
        - tfidf is not None only for n-gram models
    '''

    # Validate argument types
    assert (posts['referral_request'] == 1).all(), "All posts must be job referral requests"

    # Merge features while preserving the order of post IDs
    featcols = [col for col in feats.columns.tolist() if col != 'post_id']
    df = posts.merge(feats, on='post_id', how='left')

    # Separate into train/test sets
    X_train = df[df['train']][featcols]
    X_test = df[df['test']][featcols]


    # Pre-process numerical & categorical features
    tfidf = None

    if ('domain_' in model) or ('bertmix_' in model):

        # Apply standard scalers on numerical features
        if nums is not None:
            scaler = StandardScaler()
            X_train[nums] = scaler.fit_transform(X_train[nums])
            X_test[nums] = scaler.transform(X_test[nums])

        # One-hot encoding for low-dimensional categorical features
        if categs is not None:

            # Loop over categorical columns
            for col in categs:

                # Find the most frequently occurring category in training data
                drop = X_train[col].value_counts().index[0]

                # One-hot encode categorical column
                X_train = pd.get_dummies(X_train, columns=[col])
                X_test = pd.get_dummies(X_test, columns=[col])

                # Drop the most frequently occurring category
                X_train = X_train.drop(columns=[f'{col}_{drop}'])
                X_test = X_test.drop(columns=[f'{col}_{drop}'])


    # For n-gram models, return data frames after applying TF-IDF to text
    if model == 'ngram':
    
        # Update default token patterns with mask token patterns
        mask_pattern = r'\[[A-Z]+\]'
        token_pattern = rf'(?u)\b\w\w+\b|{mask_pattern}'
        
        # Initialize TF-IDF vectorizer with updated token pattern
        # Don't lowercase and strip accents to preserve mask tokens 
        # (vocab doesn't expand by much)
        tfidf = TfidfVectorizer(
            token_pattern=token_pattern,
            ngram_range=ngram, min_df=5, max_df=0.95, 
            sublinear_tf=True, smooth_idf=True, norm='l2',
            stop_words='english', 
            lowercase=False, 
            strip_accents=None
        )

        # Fit TF-IDF vectorizer on train/test sets. Avoid leakage
        X_train_fit = tfidf.fit_transform(X_train.iloc[:,0])
        X_test_fit = tfidf.transform(X_test.iloc[:,0])
        
        # Print vocabulary info
        print(f"\nTF-IDF Vocabulary size: {len(tfidf.vocabulary_)}")
        print("\nTop 20 most frequent tokens:")
        sorted_vocab = sorted(tfidf.vocabulary_.items(), key=lambda x: x[1], reverse=True)
        for i, (token, freq) in enumerate(sorted_vocab[:20]):
            print(f"{i+1}: {token} (freq: {freq})")

        print("\nFirst 20 tokens in original order:")
        for i, token in enumerate(tfidf.vocabulary_):
            if i < 20:
                print(f"{i}: {token} (freq: {tfidf.vocabulary_[token]})")

        # Replace train/test data with their TF-IDF counterparts
        X_train = X_train_fit
        X_test = X_test_fit


    # Return train/test data and TF-IDF vectorizer (if applicable)
    return X_train, X_test, tfidf

#--------------------------------

# Search logit regularization parameter
def search_logit_param(X_train, y_train, model):

    logit = dict(penalty='l1',solver='saga') # dict(penalty='l2',solver='lbfgs')

    # Fit prediction model
    print(f"\n\n===== Training {logit['penalty'].upper()} logistic model with {model} ===== \n")

    # Search over 200 values for inverse regularization strength
    C_range = np.logspace(np.log10(0.0001), np.log10(20), 500)

    classifier = LogisticRegressionCV(
        **logit, Cs=C_range,
        cv=5, scoring='roc_auc', class_weight='balanced',
        max_iter=5000, random_state=42, n_jobs=-1, 
    )

    classifier.fit(X_train, y_train)
    
    # Print chosen value for inverse regularization strength
    score_vals = np.mean(classifier.scores_[1], axis=0)
    chosen_C = np.argmax(score_vals)
    C_value = C_range[chosen_C]
    print(f"\nC-values: {np.round(C_range, decimals=6)}")
    print(f"\nAUROC values: {np.round(score_vals, decimals=9)}")
    print(f"\nBest AUROC: {score_vals[chosen_C]:.3f}")
    print(f"\nChosen C: {C_value:.16f}")

# Fit prediction models
def fit_model(data: dict, model: str, pred_model='logit'):

    # Extract train/test data for the specified model
    X_train, _, _ = data[model]
    y_train = data['y_train']

    # Fit prediction model
    message = f"\nTraining {pred_model} model with {model}...\n"
    print(message)

    # For BERT models, convert into numpy arrays to avoid warnings
    if 'bert_sent_' in model or 'bert_lora_' in model:
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()

    # Logit model without penalty (for interpretation)
    if 'domain' in model and pred_model == 'logit_base':

        classifier = LogisticRegression(
            penalty=None, solver='lbfgs', n_jobs=-1, 
            max_iter=5000, random_state=42, class_weight='balanced',
        )

        classifier.fit(X_train, y_train)

    # Logit with specified L1 penalty
    elif pred_model == 'logit':

        # L1 penalty is more useful for analysis on selected features
        penalty = 'L1'
        solver = 'saga'

        if model == 'domain_full': C_value = 0.2766631643320149
        elif model == 'ngram': C_value = 1.0115482193209608
        elif model == 'bert_sent_base': C_value = 0.7360098119089060
        elif model == 'bert_sent_soft': C_value = 0.9632521270671345
        elif model == 'bert_sent_hard': C_value = 1.0115482193209608
        elif model == 'bertmix_sent_base': C_value = 0.5763025457164991
        elif model == 'bertmix_sent_soft': C_value = 0.6674064164258308
        elif model == 'bertmix_sent_hard': C_value = 0.7008692253340656
        else: C_value = 1
    
        classifier = LogisticRegression(
            penalty=penalty.lower(), solver=solver, n_jobs=-1, 
            max_iter=5000, random_state=42, class_weight='balanced',
            C = C_value, 
        )

        classifier.fit(X_train, y_train)

    # XGBoost classifier (significantly outperforms logistic)
    elif pred_model == 'xgb':
        classifier = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='logloss',
            n_estimators=100, # 100, 200, 300
            max_depth=4, # 3, 4, 5, 6
            min_child_weight=5, # 1, 3, 5
            learning_rate=0.01, # 0.01, 0.05, 0.10
            colsample_bytree=0.8, # 0.7, 0.8, 0.9
            reg_alpha=0.1, # 0, 0.1, 0.5, 1.0
            reg_lambda=1, # 0.1, 1, 5
            subsample=0.8, random_state=42, gamma=0, 
            use_label_encoder=False,
        )

        weights = compute_sample_weight(class_weight='balanced', y=y_train)
        classifier.fit(X_train, y_train, sample_weight=weights)
    
    # Return value error for invalid models
    else:
        raise ValueError("Invalid prediction model specified. Choose 'logit', 'logit_cv', or 'xgb'.")

    # Return model object
    return classifier

#--------------------------------

# Extract data and trained model
def get_data_model(data: dict, predmod: dict, model_name: str):

    data_name = model_name.replace('full_base','full').replace('_logit','').replace('_xgb','')

    X_train, X_test, tfidf = data[data_name]
    y_train, y_test = data['y_train'], data['y_test']
    model = predmod[model_name]
    train_test = (X_train, X_test, y_train, y_test)
    return model, train_test, tfidf

# Export predicted probabilities for training data
def export_proba(data: dict, predmod: dict):

    # Loop over models
    export = pd.DataFrame({'post_id': data['train_ids']})

    for model_name in predmod.keys():
        if 'baseline_' not in model_name:

            # Extract data and trained model
            model, train_test, _ = get_data_model(data, predmod, model_name)
            X_train, _, _, _ = train_test

            # Predicted probabilities for training data
            y_proba = model.predict_proba(X_train)[:,1]
            export[model_name] = y_proba

    # Export predicted probabilities to CSV
    return export

#--------------------------------

# Load encoder, embeddings, and logit for specified BERT model
def load_bert_logit_models(bert_type: str, post_id: pd.DataFrame | None = None):

    # Load BERT model
    if 'sent_' in bert_type:
        bert_name = f"{TRAIN}/{bert_type}/sentence-transformers-all-distilroberta-v1"
    elif 'lora_' in bert_type:
        bert_name = f"{TRAIN}/{bert_type}/distilbert-base-uncased"
    else:
        bert_name = bert_type

    bert = load_bert_model(bert_name)
    bert.name = bert_name.replace(TRAIN, '')

    # Prepare logit model for success probabilities (trained on above embeddings)
    predmod = joblib.load(f"{TRAIN}/model.joblib")
    logit = predmod[f'bert_{bert_type}_logit']


    # Return BERT model and logit model, unless post_id has been provided
    if post_id is None:
        return bert, logit


    # Load embeddings for training FAISS (generated from above model)
    else:
        embed_ft = pd.read_pickle(f"{CLEAN}/posts_encode_{bert_type}.pkl")
        embed_ft = embed_ft.merge(post_id, on='post_id', how='inner')
        enc = embed_ft.drop(columns='post_id')
        return bert, logit, enc

# Predict success probabilities given text & bert/logit models
def predict_success(bert, logit, text: str | list) -> np.ndarray:

    # Use BERT to encode text
    enc = encode_text(bert, text)

    # Predict success probability
    proba = logit.predict_proba(enc)[:,1]

    return proba

if __name__ == "__main__":

    # Load data on train/test IDs and target labels for referral requests
    target = 'class_offers'
    posts = pd.read_csv(f"{CLEAN}/posts_basic.csv")
    posts = posts[posts['referral_request']]

    data = {
        'train_ids' : posts[posts['train']]['post_id'],
        'test_ids'  : posts[posts['test']]['post_id'],
        'y_train'   : posts[posts['train']][target],
        'y_test'    : posts[posts['test']][target],
    }

    #----------------------------------------

    # Feature Model 1 (domain): Featurized Model (from domain knowledge)
    domain = pd.read_csv(f'{CLEAN}/posts_linguistic.csv')
    numcols = ['type_token_ratio', 'post_length','kincaid_grade','spell_errors']
    data['domain_full'] = prepare_data(posts, domain, model='domain_full', nums=numcols)


    # Feature Model 2 (terms): Term-Frequency Model (TF-IDF)
    mask = pd.read_csv(f"{CLEAN}/posts_mask.csv")
    toks = mask[['post_id','combined_tokens']]
    data['ngram'] = prepare_data(posts, toks, model='ngram', ngram=(1,2))


    # Feature Model 3 (bert): BERT embeddings as features
    embed_sent_base = pd.read_pickle(f"{CLEAN}/posts_encode_sent_base.pkl")
    data['bert_sent_base'] = prepare_data(posts, embed_sent_base, model='bert_sent_base')

    embed_sent_soft = pd.read_pickle(f"{CLEAN}/posts_encode_sent_soft.pkl")
    data['bert_sent_soft'] = prepare_data(posts, embed_sent_soft, model='bert_sent_soft')

    embed_sent_hard = pd.read_pickle(f"{CLEAN}/posts_encode_sent_hard.pkl")
    data['bert_sent_hard'] = prepare_data(posts, embed_sent_hard, model='bert_sent_hard')

    # Feature Model 4: BERT embeddings with additional non-text features
    nontext = pd.read_csv(f"{CLEAN}/posts_basic.csv")
    nums, categs = ['num_hashtags','firm_prop_offers'], ['day_of_week']
    nontext = nontext[['post_id'] + nums + categs]

    data['bertmix_sent_base'] = prepare_data(
        posts = posts.drop(columns=nums+categs), 
        feats = embed_sent_base.merge(nontext, on='post_id', how='left'), 
        model = 'bertmix_sent_base', nums = nums, categs = categs
    )

    data['bertmix_sent_soft'] = prepare_data(
        posts = posts.drop(columns=nums+categs), 
        feats = embed_sent_soft.merge(nontext, on='post_id', how='left'), 
        model = 'bertmix_sent_soft', nums = nums, categs = categs
    )

    data['bertmix_sent_hard'] = prepare_data(
        posts = posts.drop(columns=nums+categs), 
        feats = embed_sent_hard.merge(nontext, on='post_id', how='left'), 
        model = 'bertmix_sent_hard', nums = nums, categs = categs
    )

    #----------------------------------------

    # Baseline Model 1: Random Classifier
    predmod = dict()
    success = data['y_train'].mean()
    uniform_test   = np.random.uniform(0, 1, data['test_ids'].shape[0])
    uniform_train  = np.random.uniform(0, 1, data['train_ids'].shape[0])
    y_random_test  = (uniform_test < success).astype(int)
    y_random_train = (uniform_train < success).astype(int)
    predmod['baseline_test'] = y_random_test
    predmod['baseline_train'] = y_random_train


    # Baseline Model 2: Unpenalized domain knowledge model (for interpretation)
    predmod['domain_full_base'] = fit_model(data, 'domain_full', 'logit_base')


    # Loop through feature-classifier combinations
    model_list = [k for k in data.keys() if 'train' not in k and 'test' not in k]
    clf_list = ['logit','xgb']

    for model in model_list:

        # # Find penalty params for logistic models
        # X_train, y_train = data[model][0], data['y_train']
        # search_logit_param(X_train, y_train, model)

        # Fit logistic/XGB classifiers
        for clf in clf_list:
            predmod[f'{model}_{clf}'] = fit_model(data, model, clf)


    # Export predicted probabilities for all models
    proba = export_proba(data, predmod)
    proba.to_csv(f"{TRAIN}/predict_proba.csv", index=False)

    # Export training data and model
    joblib.dump(data, f"{TRAIN}/data.joblib" , compress=3)
    joblib.dump(predmod, f"{TRAIN}/model.joblib" , compress=3)


    # Load above results for exploration
    target = 'class_offers'
    posts = pd.read_csv(f"{CLEAN}/posts_basic.csv")
    posts = posts[posts['referral_request']]

    data = joblib.load(f'{TRAIN}/data.joblib')
    predmod = joblib.load(f'{TRAIN}/model.joblib')
    proba = pd.read_csv(f"{TRAIN}/predict_proba.csv")
