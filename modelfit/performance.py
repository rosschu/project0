'''
============================================================

Evaluate Classification Performance for Trained Models

============================================================
'''

# Packages and Directories
from setup.utils import *
from modelfit.train import get_data_model

# Binned scatter plot of predicted prob bins VS pct true labels
def bin_scatter_plot(y_true, y_proba, model_name, bins=20, train=False):

    # Calculate average true label for each quantile bin of predicted prob
    quant = pd.DataFrame({'true_label': y_true, 'model_prob': y_proba})
    quant['bins'] = pd.qcut(quant['model_prob'], q=bins, labels=False, duplicates='drop')
    model_avg = quant.groupby('bins')[['model_prob','true_label']].mean()
    min_bin = (model_avg.iloc[0,0] // 0.05) * 0.05
    max_bin = (model_avg.iloc[-1,0] // 0.05) * 0.05 + 0.05
    bin_range = [round(min_bin, 2), round(max_bin, 2)]

    # Plot binned scatter plot
    plt.close('all')
    plt.figure(figsize=(10, 6))
    plt.plot(model_avg['model_prob'], model_avg['true_label'], marker='o', label='Model')
    plt.plot(bin_range, bin_range, linestyle='--', label='Perfect Calibration')
    plt.xlabel('Average Predicted Probability (by Quantile Bin)')
    plt.ylabel('Percentage of True Labels')
    plt.title('Predicted Prob Bins (X) vs. True Success Rate (Y)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    folder = f"{OUTPUT}/binscatter/train{int(train)}"
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)    
    plt.savefig(f"{folder}/{model_name}.png")

# Evaluate majority classifier on test set
def performance_metrics(y_true, y_pred, y_proba, noreport=False):
    
    # Essential performance metrics
    result = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
    }

    if y_proba is not None:
        result['roc_auc'] = roc_auc_score(y_true, y_proba)

    for key in result.keys():
        result[key] = round(result[key], 3)

    # Confusion matrix (optional)
    if not noreport:
        result['classrep'] = classification_report(y_true, y_pred)

    return result

# Comparison of model performance with random baseline
def compare_performance(metrics: dict, model_name: str, conf=None, train=False):

    # Start log file
    folder = f"{OUTPUT}/evaluation/train{int(train)}"
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)    
    sys.stdout = open(f"{folder}/{model_name}.txt", 'w')

    print(f'====== {model_name} Model ======\n')
    print(f':::: Performance Metrics ::::\n')
    print(metrics)

    if conf:
        print(f':::: Bootstrapped Confidence Intervals ::::\n')
        print(conf)

    # Close log file
    sys.stdout.close()
    sys.stdout = sys.__stdout__

# Bootstrapped confidence intervals (parallelized)
def confidence_interval(y_test, y_pred, y_proba, n_bootstrap=1000):

    # Single bootstrap iteration
    def bstrap_iter(seed):
        opts = {'replace': True, 'random_state': seed, 'n_samples': len(y_test)}
        if y_proba is None:
            y_samp, y_pred_samp = resample(y_test, y_pred, **opts)
            y_proba_samp = None
        else:
            y_samp, y_pred_samp, y_proba_samp = resample(y_test, y_pred, y_proba, **opts)
        return performance_metrics(y_samp, y_pred_samp, y_proba_samp, noreport=True)

    # Parallelize & collect bootstrapped metrics
    bs_metrics = joblib.Parallel(n_jobs=-1)(joblib.delayed(bstrap_iter)(seed) for seed in range(n_bootstrap))
    bs_format = defaultdict(list)
    measures = bs_metrics[0].keys()

    for i in range(n_bootstrap):
        for ms in measures:
            bs_format[ms].append(bs_metrics[i][ms])

    # Confidence interval = 2.5th and 97.5th pctile of bstrap dist
    conf = defaultdict(dict)
    for ms in measures:
        pctile = np.percentile(bs_format[ms], [2.5, 97.5])
        conf[ms] = tuple(np.round(pctile, 3))
    
    return conf

# Evaluate specified model on test set
def evaluate_model(data: dict, predmod: dict, model_name: str, bstrap=True, train=False):

    # Obtain model predictions on test set
    binsc = False
    y_proba = None

    if 'baseline' in model_name:
        keyname = 'baseline_train' if train else 'baseline_test'
        y_pred = predmod[keyname]
        y_test = data['y_test']
        y_train = data['y_train']
        y_vals = y_train if train else y_test
    else:

        # Extract model & data
        model, train_test, _ = get_data_model(data, predmod, model_name)
        X_train, X_test, y_train, y_test = train_test
        y_vals = y_train if train else y_test
        X_vals = X_train if train else X_test

        # For Gen AI models, extract predictions from data frame
        y_pred = model.predict(X_vals)
        y_proba = model.predict_proba(X_vals)[:,1]
        binsc = True

    # Create bin scatter plot
    if binsc:
        bin_scatter_plot(y_vals, y_proba, model_name, train=train)

    # Compare performance metrics for random baseline VS model
    metrics = performance_metrics(y_vals, y_pred, y_proba)
    metrics = dict(sorted(metrics.items()))

    # Bootstrapped confidence intervals
    conf = None if not bstrap else confidence_interval(y_vals, y_pred, y_proba)

    # Display and return results
    compare_performance(metrics, model_name, conf, train=train)
    del metrics['classrep']
    return metrics, conf

if __name__ == "__main__":

    # Load trained model and data
    data = joblib.load(f'{TRAIN}/data.joblib')
    predmod = joblib.load(f'{TRAIN}/model.joblib')

    # Evaluate model performance
    model_list = ['baseline'] + [k for k in predmod.keys() if 'baseline' not in k]
    for model_name in model_list:
        metrics, conf = evaluate_model(data, predmod, model_name, bstrap=True, train=False) # test set
        metrics, conf = evaluate_model(data, predmod, model_name, bstrap=True, train=True) # train set (to assess overfitting)
    
