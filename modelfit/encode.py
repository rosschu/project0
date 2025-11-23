'''
============================================================

Encode text data into embeddings with SBERT

============================================================
'''

# Packages and Directories
from setup.utils import *
import dataprep.basic
import dataprep.mask_tokens

# Configure parameters for fine-tuning BERT for text classification
def tuning_parameters(posts: pd.DataFrame, bert_name: str, aggressive: bool, gpu=True):
    '''
        Example: 
            - N = 10 = # tokens
            - D = 768 = BERT embedding dimensions
            - H = 12 = # heads
            - V = D / H = 64 = dimensions per head

        Transformer Parameters:
            - q_lin: query vector = embedding that represents type of info that each token is looking for. q_lin = input (N x D) @ query weights (D x V) = (N x V)
            
            - k_lin: key vector = embedding that represents info contained in other tokens. k_lin = input (N x D) @ key weights (D x V) = (N x V)
            attention weights = dot_product(q_lin, k_lin) = higher scores if there is a closer match between query and key vectors. AS = q_lin @ k_lin' = N x N
            
            - v_lin: value vector = embedding that also represents info contained in each token. v_lin = input (N x D) @ value weights (D x V) = (N x V)
            Head = attention weights (N x N) * value vectors (N x V) = (N x V)
            output values = concat(Head 1, Head 2, ...) * out_lin
            
            - out_lin: output linear layer = weights applied to attention values to product final output (can be added to target before raising rank r to improve fit)

        How LoRA works:
            - BERT embeddings have dimension N x D (num tokens X BERT dimension)
            - For each head, this is projected into a lower dimension N x V
            - However, this requires training a large weight matrix D x V
            - LoRA adds an updating matrix with the same size D x V
            - However, the updating matrix is a product of two smaller matrices (D x S) @ (S x V) where S = 8 is a smaller lora_rank
            - Hence training (D x V) + (S x V) parameters is much more efficient than training (D x V) parameters while capturing most of the information
    '''

    # If fine-tuning from SBERT checkpoint, extract underlying transformer and tokenizer and save it to disk
    if bert_name.startswith("sentence-transformers/"):

        # Create dedicated folder
        modpath = pathlib.Path(f"{TRAIN}/sentbert_base")
        modpath.mkdir(parents=True, exist_ok=True)

        # Extract transformer and tokenizer if not already done
        if not (modpath / "config.json").exists():
            sent_bert = stf.SentenceTransformer(bert_name)
            sent_auto_model = sent_bert[0].auto_model
            sent_tokenizer = sent_bert.tokenizer

            sent_auto_model.save_pretrained(modpath)
            sent_tokenizer.save_pretrained(modpath)

        # Replace bert_name with relevant model path
        bert_name = str(modpath)


    # Set fine-tuning parameters
    topts = dict()

    # (stable behavior, but slower learning can hurt performance)
    if not aggressive:

        # rank of LoRA matrix. r=8 (fewer parameters), r=16 (more flexible)
        topts['lora_rank'] = 16

        # Learning rate (usually higher for LoRA, around 1e-4)
        topts['learning_rate'] = 1e-4

        # Number of batches before updating gradients & parameters
        # (higher = more compute time, but more stable due to higher effective batch size)
        topts['gradient_accumulation_steps'] = 2

        # Number of full passes over training set
        topts['num_train_epochs'] = 6

    # (faster & aggressive learning, but could be unstable)
    else:
        topts['lora_rank'] = 32
        topts['learning_rate'] = 1e-4
        topts['gradient_accumulation_steps'] = 2
        topts['num_train_epochs'] = 6


    # Set parameters
    textcol = 'combined_mask'
    labelcol = 'class_offers'
    device = torch.device('mps') if gpu else torch.device('cpu')

    # Only keep referral requests in train set to avoid leakage
    # Split this train set into further train/eval sets for fine-tuning BERT models
    base_df = posts[posts['train']]
    assert (base_df['train'] & base_df['referral_request']).all()

    train_df, eval_df = train_test_split(
        base_df[[textcol, labelcol]], 
        test_size=0.2,
        random_state=95,
        stratify=base_df[labelcol]
    )

    # Convert pandas data frame to Hugging Face dataset
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "eval": Dataset.from_pandas(eval_df)
    })


    # Load relevant tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_name)

    # Add special mask tokens
    new_tokens = [mask for mask in MASK_TOKENS if mask not in tokenizer.get_vocab()]
    has_new_tokens = len(new_tokens) > 0

    if has_new_tokens:
        tokenizer.add_tokens(new_tokens)


    # Tokenize input data
    tokenizer_map = lambda row: tokenizer(row[textcol], padding="max_length", truncation=True, max_length=256)

    tok_data = dataset.map(tokenizer_map, batched=True)
    tok_data = tok_data.rename_column(labelcol, "labels")
    tok_data = tok_data.remove_columns([textcol,'__index_level_0__'])
    tok_data = tok_data.with_format("torch")


    # Load the base model and apply LoRA
    kwargs = dict(num_labels=2) # device_map='auto', quantization_config=bnb_config
    base_model = AutoModelForSequenceClassification.from_pretrained(bert_name, **kwargs).to(device)
    base_model.tokenizer = tokenizer
    

    if has_new_tokens:
        base_model.resize_token_embeddings(len(tokenizer))
    

    # Target modules for LoRA
    target_modules = {name.split(".")[-1] for name, mod in base_model.named_modules() if isinstance(mod, nn.Linear)}

    if {"q_lin", "v_lin"} <= target_modules: # out_lin
        target_mods = ["q_lin", "v_lin"]
    elif {'query', 'value'} <= target_modules: # out_proj
        target_mods = ["query", "value"]
    else:
        raise ValueError(f"Target modules not found: {target_modules}")


    # Define LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=topts['lora_rank'],
        lora_alpha=32,  
        lora_dropout=0.1,  
        target_modules=target_mods
    )


    # Apply LoRA to base model
    model = get_peft_model(base_model, lora_config)
    model = model.to(device)
    print('\n\n')
    model.print_trainable_parameters()

    # Calculate class weights to account for imbalance
    rate_1 = train_df[labelcol].mean()
    rate_0 = 1 - rate_1
    weight_0 = 1 / (2*rate_0)
    weight_1 = 1 / (2*rate_1)
    avg_weight = (rate_0 * weight_0) + (rate_1 * weight_1)
    print(f"\n\nAverage weight: {avg_weight:.2f}\n\n")

    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)

    # Define WeightedTrainer for weighted cross entropy loss
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = nn.CrossEntropyLoss(weight=class_weights)(logits, labels)
            return (loss, outputs) if return_outputs else loss


    # Calculate metrics for monitoring performance. Fine-tuning will optimize on AUROC.
    def monitor_metric_calc(eval_pred):

        # HF trainers return logit scores & true labels
        logit_score, true_labels = eval_pred
        logits_t = torch.tensor(logit_score, device=device)
        labels_t = torch.tensor(true_labels, dtype=torch.long, device=device)
        y_true = labels_t.detach().cpu().numpy()

        # Convert logit scores into probabilities
        y_proba = torch.softmax(logits_t, dim=-1)[:, 1].detach().cpu().numpy()

        # Predicted labels
        y_pred = (y_proba >= 0.5).astype(int)

        # Performance metrics to monitor
        metrics = {
            'entropy': nn.CrossEntropyLoss(weight=class_weights)(logits_t, labels_t),
            'auroc': roc_auc_score(y_true, y_proba),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, zero_division=0, average='weighted'),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            'true_pos_rate': float(y_true.mean()),
            'pred_pos_rate': float(y_pred.mean()),
        }

        print('\n\n\n\n\n===== Metrics Reporting ====\n\n')

        for k,v in metrics.items():
            print(f"\n{k}: {v:.5f}\n")
        
        print('\n\n===== Metrics End ====\n\n')

        return metrics

    # Path for saving output
    outpath = f'{TRAIN}/sent' if 'sentbert_base' in bert_name else f'{TRAIN}/lora'
    outpath = f"{outpath}_hard" if aggressive else f"{outpath}_soft"

    # Define training arguments
    # Achieve better learning through compute time, not extra variance
    training_args = TrainingArguments(

        # Directories for saving output
        output_dir = outpath,

        # Optimizer
        optim="adamw_torch",

        # Num of full passes over training set. 
        # Typically between 3-6 for BERT-style models. More epochs can overfit
        num_train_epochs=topts['num_train_epochs'],

        # Batch size = number of samples for forward/backward passes. 
        # Larger batch = smoother gradients = better performance (generally)
        # Typically 8-32 on consumer GPUs. Larger batch = more memory consumption
        per_device_train_batch_size=32, 
        per_device_eval_batch_size=32*2,
    
        # GAS = Number of batches before updating model parameter (optimizer step)
        # Higher GAS = slower but more stable optimization (heavier compute, but not memory consumption)
        # Effective batch size = batch size * GAS * num devices (=1)
        # Aim for EBS = 32-64 for steady optimization.
        gradient_accumulation_steps=topts['gradient_accumulation_steps'], 
    
        # Size of weight updates in each optimizer step
        # Lower = stable but learns more slowly. Higher = faster but more unstable
        # Typically between 1e-5 and 3e-5 for full-tuning, 5e-5 and 2e-4 for LoRA tuning
        learning_rate=topts['learning_rate'], 


        # How fast learning rate ramps up from 0 to learning_rate.
        # warm up steps = warm up ratio * total steps (train size)
        warmup_ratio=0.06,
        
        
        # How learning rate evolves after initial warmup.
        # 'linear' for stable, predictable decay (better for short or noisy runs)
        # 'cosine' is the default for LoRA. Fast early decay, gentle tail
        # 'cosine_with_restarts' builds in hard restarts to escape local minima 
        # (but introduces extra variance and hurts stability)
        lr_scheduler_type='cosine', # cosine (aggresive) VS linear (more stable)


        # Magnitude of L2 regularization (default = 0.01)
        # Higher = stronger regularization = more stable but might underfit
        weight_decay=0.01, 


        # How often to evaluate performance metrics & save checkpoints
        # (optimizer steps per epoch = train size / effective batch size)
        # (in our setup, trainsize = 9k so 283 steps per epoch, so 2-3 evals per epoch)
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=100, 
        eval_steps=100, 
        save_steps=100, 


        # How many checkpoints to save. Heuristic is early stopping patience + 1
        # Load best model at the end of training
        save_total_limit=8,
        load_best_model_at_end=True,
        metric_for_best_model="auroc", 
        greater_is_better=True,

        # Only keep columns used by forward pass (default = True)
        # Set to False if loss calc requires extra inputs (e.g. sampling weights)
        remove_unused_columns=False, 
        
        # Apple GPU settings
        dataloader_pin_memory=False, # speeds up transfer, but slows down training for apple GPU
        # use_mps_device=True, # for apple GPU (deprecated)
        # bf16=True,  # mixed precision float for faster training. Doesn't work on Apple GPU
    )

    # Set up the Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tok_data["train"],
        eval_dataset=tok_data["eval"],
        processing_class=tokenizer, 
        compute_metrics=monitor_metric_calc,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=5, # number of evals for checking metric improvements
            early_stopping_threshold=0.0001, # stop if metric doesn't improve more
        )]
    )

    # Print model parameters before fine-tuning
    N_train = dataset['train'].shape[0]
    N_eval = dataset['eval'].shape[0]
    pos_rate_train = train_df[labelcol].mean()
    pos_rate_eval = eval_df[labelcol].mean()

    batch_size = training_args.per_device_train_batch_size
    grad_accum_steps = training_args.gradient_accumulation_steps
    num_devices = 1
    effec_batch_size = batch_size * grad_accum_steps * num_devices

    batches_per_epoch = math.ceil(N_train / (batch_size))
    optim_steps_per_epoch = math.ceil(batches_per_epoch / grad_accum_steps)
    total_steps = int(training_args.num_train_epochs * optim_steps_per_epoch)
    warmup_steps = math.ceil(training_args.warmup_ratio * total_steps)

    print(f"\nN_train = {N_train} / N_eval = {N_eval} / pos_rate_train = {pos_rate_train:.3f} / pos_rate_eval = {pos_rate_eval:.3f}\n")
    print(f"\nEffective Batch Size = {effec_batch_size} / steps_per_epoch = {optim_steps_per_epoch} / total_steps = {total_steps} / warmup_steps={warmup_steps} / evals_per_epoch = {optim_steps_per_epoch / training_args.eval_steps} / number of epochs = {training_args.num_train_epochs}\n")

    return trainer, tokenizer

# Execute finetuning job and save model
def run_tuning_job(posts: pd.DataFrame, bert_name: str, aggressive=False):

    # Output folder
    outpath = f"{TRAIN}/sent" if 'sentence' in bert_name else f"{TRAIN}/lora"
    outpath = f"{outpath}_hard" if aggressive else f"{outpath}_soft"
    outpath = f"{outpath}/{bert_name.replace('/', '-')}"

    # Set parameters for fine-tuning
    trainer, tokenizer = tuning_parameters(posts, bert_name, aggressive)

    # Fine-tune with stable parameters
    start_timer = time.time()
    trainer.train()
    trainer.save_model(outpath)
    tokenizer.save_pretrained(outpath)

    # Print tuning results
    modlabel = 'SentBERT' if 'sentence' in outpath else 'DistilBERT'
    modlabel = f"Aggressive {modlabel}" if aggressive else f"Stable {modlabel}"

    print(f"\n\n\n{modlabel} Training Time: {(time.time() - start_timer) / 60:.1f} mins\n\n\n")
    print(f"\n\n{modlabel} Eval Dataset Size: {trainer.eval_dataset.shape[0]}\n\n")
    print(f"\n\n{modlabel} Best Checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"\n\n{modlabel} Best Metric: {trainer.state.best_metric}")
    print(trainer.evaluate())

#--------------------------------

# Load BERT model
def load_bert_model(bert_name: str, gpu=True) -> stf.SentenceTransformer:
    '''
    Description:
        - Loads BERT model from sentence transformers

    Inputs:
        - bert_name: name of BERT model to load (either name of out-of-box model, or path to fine-tuned model)
        - gpu: whether to use GPU for encoding
    '''

    # Determine whether model is fine-tuned (which would be saved to disk)
    ftune = os.path.exists(bert_name)

    # Update tokenizer with mask tokens
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    
    new_tokens = [mask for mask in MASK_TOKENS if mask not in tokenizer.get_vocab()]
    has_new_tokens = len(new_tokens) > 0

    if has_new_tokens:
        tokenizer.add_tokens(new_tokens)

    # Load out-of-box BERT models from sentence transformers
    if not ftune:
        bert = stf.SentenceTransformer(bert_name)
        bert.tokenizer = tokenizer

        if has_new_tokens:
            bert[0].auto_model.resize_token_embeddings(len(tokenizer))

    # Otherwise, wrap fine-tuned BERT models with sentence transformers
    else:

        # Transformer module (including tokenizer)
        transf_mod = stf.models.Transformer(bert_name)
        max_token_length = 256 # (based on fine-tuning parameters from above)
        transf_mod.max_seq_length = max_token_length
        tokenizer.model_max_length = max_token_length

        if has_new_tokens:
            transf_mod.auto_model.resize_token_embeddings(len(tokenizer))

        # Pooling module (with CLS token)
        pool_mod = stf.models.Pooling(
            word_embedding_dimension=transf_mod.get_word_embedding_dimension(),
            pooling_mode_cls_token=True,
            pooling_mode_max_tokens=False,
            pooling_mode_mean_tokens=False
        )

        # Wrap transformer/pooling modules with sentence transformer
        bert = stf.SentenceTransformer(modules=[transf_mod, pool_mod])
        bert.tokenizer = tokenizer

    # Enable multiprocess encoding with GPU or CPU
    device = ["mps"] if gpu else ["cpu"] * os.cpu_count()
    opts = dict(device=device, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    bert.opts = opts

    # Return BERT model
    return bert

# Encode individual request
def encode_text(bert: stf.SentenceTransformer, text: str | list, debug=False) -> np.ndarray: 

    # Text should be either string or list
    text = [text] if type(text) == str else text
    start_enc = time.time()

    # Encode single string
    if len(text) == 1:
        enc = bert.encode(text[0], normalize_embeddings=True)
        enc = enc.reshape(1, -1)
    
    # Encode multiple strings
    else:
        enc = bert.encode(text, **bert.opts)
    
    # Normalize embeddings to norm = 1 (so that cosine similarity = dot product)
    row_norms = np.linalg.norm(enc, axis=1, keepdims=True)
    norm_enc = enc / row_norms
    end_enc = time.time()

    if debug:
        print(f"Total inference time: {end_enc - start_enc: .2f} seconds")

    return norm_enc

# Saves BERT embeddings to a data frame
def save_embed(post_id: list, enc: np.ndarray) -> pd.DataFrame:
    '''
    Description:
        - Save BERT embeddings to a data frame with post IDs

    Inputs:
        - post_id: list of post IDs
        - enc: dictionary of embeddings for each post title & text
        - textcol: text column to save embeddings for
    
    Outputs:
        - export: data frame with embeddings for corresponding text column
    '''

    enc_cols = [f"enc{i+1}" for i in range(enc.shape[1])]
    export = pd.DataFrame(enc, columns=enc_cols)
    export['post_id'] = post_id
    return export[['post_id']+enc_cols]

if __name__ == "__main__":

    # Split referral requests into train/test sets based on recency
    mask = pd.read_csv(f"{CLEAN}/posts_mask.csv")
    basic = pd.read_csv(f"{CLEAN}/posts_basic.csv")
    posts = basic.merge(mask, on='post_id', how='inner')
    posts = posts[posts['referral_request']]


    # Fine-tune BERT for text classification (17-ish mins)
    refit = True

    if refit:

        sent_bert_name = "sentence-transformers/all-distilroberta-v1"
        # run_tuning_job(posts, sent_bert_name, aggressive=False)
        run_tuning_job(posts, sent_bert_name, aggressive=True)

        distil_bert_name = "distilbert-base-uncased"
        run_tuning_job(posts, distil_bert_name, aggressive=False)
        run_tuning_job(posts, distil_bert_name, aggressive=True)


    # Prepare text inputs
    post_id = posts['post_id'].tolist()
    text = posts['combined_mask'].tolist()


    # Encode text data with fine-tuned BERT
    lora_soft_path = f"{TRAIN}/lora_soft/distilbert-base-uncased"
    lora_soft_bert = load_bert_model(lora_soft_path)
    enc_lora_soft = encode_text(lora_soft_bert, text)
    embed_lora_soft = save_embed(post_id, enc_lora_soft)
    embed_lora_soft.to_pickle(f"{CLEAN}/posts_encode_lora_soft.pkl")

    lora_hard_path = f"{TRAIN}/lora_hard/distilbert-base-uncased"
    lora_hard_bert = load_bert_model(lora_hard_path)
    enc_lora_hard = encode_text(lora_hard_bert, text)
    embed_lora_hard = save_embed(post_id, enc_lora_hard)
    embed_lora_hard.to_pickle(f"{CLEAN}/posts_encode_lora_hard.pkl")


    # Encode text data into embeddings with out-of-box SBERT
    sent_base_path = "sentence-transformers/all-distilroberta-v1"
    sent_base_bert = load_bert_model(sent_base_path)
    enc_sent_base = encode_text(sent_base_bert, text)
    embed_sent_base = save_embed(post_id, enc_sent_base)
    embed_sent_base.to_pickle(f"{CLEAN}/posts_encode_sent_base.pkl")


    # Encode text data into embeddings with fine-tuned SBERT
    sent_soft_path = f"{TRAIN}/sent_soft/"
    sent_soft_path += "sentence-transformers-all-distilroberta-v1"
    sent_soft_bert = load_bert_model(sent_soft_path)
    enc_sent_soft = encode_text(sent_soft_bert, text)
    embed_sent_soft = save_embed(post_id, enc_sent_soft)
    embed_sent_soft.to_pickle(f"{CLEAN}/posts_encode_sent_soft.pkl")

    sent_hard_path = f"{TRAIN}/sent_hard/"
    sent_hard_path += "sentence-transformers-all-distilroberta-v1"
    sent_hard_bert = load_bert_model(sent_hard_path)
    enc_sent_hard = encode_text(sent_hard_bert, text)
    embed_sent_hard = save_embed(post_id, enc_sent_hard)
    embed_sent_hard.to_pickle(f"{CLEAN}/posts_encode_sent_hard.pkl")
