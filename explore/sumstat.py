'''
============================================================

Basic Summary Statistics for Referral Requests Data

============================================================
'''

# Packages and Directories
from setup.utils import *

# Histograms for target metrics to be predicted
def distribution_metrics(df: pd.DataFrame, metric, color):

    # Define upper bounds for plotting purposes
    df['num_views'] = df['num_views'].apply(lambda x: 500 if x > 500 else x)
    df['num_likes'] = df['num_likes'].apply(lambda x: 6 if x > 6 else x)
    df['num_comments'] = df['num_comments'].apply(lambda x: 20 if x > 20 else x)

    # Tabulation of values
    # print(f"{metric}\n{df[metric].value_counts(normalize=True).sort_index()}\n")

    # Horizontal Axis Title
    xti = metric.replace('_', " ").replace('num','number of').title()

    # Plot the distribution for each metric
    # Histogram for views, and bar plot of unique values otherwise
    plt.close('all')
    plt.figure(figsize=(10, 6))

    if metric == 'num_views':
        sns.histplot(df[metric], bins=20, color=color, stat='percent')
    else:
        value_counts = df[metric].value_counts(normalize=True) * 100
        plt.bar(value_counts.index, value_counts.values, color=color)
        
    # Label & export figure
    plt.title("Distribution for "+ xti)
    plt.xlabel(xti)
    plt.ylabel('Percent (%)')
    plt.tight_layout() 
    folder = f"{OUTPUT}/sumstat"
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)    
    plt.savefig(f"{folder}/{metric}.png")

# Time series of referral requests
def requests_time_series(df: pd.DataFrame):

    # Filter to referral requests & extract week of date
    df = df[df['referral_request']]
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'] - df['date'].dt.weekday * pd.Timedelta(days=1)
    df = df[df['week'] < df['week'].max()]
    df = df[df['week'] > df['week'].min()]

    # Group by date and count number of referral requests
    df = df.groupby('week')['post_id'].count().reset_index()
    df = df.rename(columns={'post_id':'Number of Referral Requests'})

    # Plot time series of referral requests
    colname = 'Number of Referral Requests'
    color = sns.color_palette()[0]
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='week', y=colname, data=df, color=color)
    plt.scatter(df['week'], df[colname], color=color)
    plt.title('Time Series of Referral Requests')
    plt.xlabel('Date')
    plt.ylabel('Number of Referral Requests')
    plt.tight_layout()
    folder = f"{OUTPUT}/sumstat"
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)    
    plt.savefig(f"{folder}/time_series_requests.png")

# Distribution of Referral Offers & Requests Across Firms
def gini_curve(df: pd.DataFrame, label: str):

    # Remove "ex-" prefixes to firm names
    df['user_firm'] = df['user_firm'].str.replace('ex-','')

    # Exclude outlier firms that take up > 10% of all activity 
    # (one firm + unknown firm for referral requests)
    df['pct_activity'] = df.groupby('user_firm')['post_id'].transform('count')
    df['pct_activity'] = df['pct_activity'] / df.shape[0]
    df = df[df['pct_activity'] <= 0.10]
    df = df[df['user_firm'] != 'New']
    # print(df[df['pct_activity'] > 0.10]['user_firm'].value_counts())

    # Lorenz curve for activity by firm
    gini = df['user_firm'].value_counts(normalize=True).sort_values().reset_index()
    gini.columns = ['firm', 'pct']
    gini['cum_pct'] = gini['pct'].cumsum()

    # Equality curve for activity by firm
    bin_size = 1 / gini.shape[0]
    gini['cum_firms'] = (gini.index + 1) / gini.shape[0]
    gini['equality'] = gini['cum_firms']
    gini = gini.loc[:,['firm','cum_firms','cum_pct','equality']]
    gini['between'] = (gini['equality'] - gini['cum_pct']) * bin_size
    gini['total'] = gini['equality'] * bin_size
    
    # Calculate Gini coefficient
    assert np.isclose(gini['total'].sum(), 0.5, atol=0.001)
    assert gini['between'].sum() < gini['total'].sum()
    coef = gini['between'].sum() / gini['total'].sum()

    # Plot Gini curve
    plot_title = label.replace('_', " ").title()
    plt.close('all')
    plt.figure(figsize=(10, 6))
    plt.plot(gini['cum_firms'], gini['cum_pct'], label='Lorenz Curve')
    plt.plot([0,1], [0, 1], linestyle='--', label='Perfect Equality')
    plt.title(f'Distribution of {plot_title} Across Firms (Gini Coef: {coef:.2f})')
    plt.xlabel('Firms (ranked from least to most common)')
    plt.ylabel('Cumulative Percent of Activity')
    plt.legend()
    plt.tight_layout()
    folder = f"{OUTPUT}/sumstat"
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)    
    plt.savefig(f"{folder}/gini_{label}.png")

if __name__ == "__main__":

    # Load data on referral requests and offers
    posts = pd.read_csv(f"{CLEAN}/posts.csv")
    cmts = pd.read_csv(f"{CLEAN}/comments.csv")

    # Distribution of target metrics
    distribution_metrics(posts, 'num_views', sns.color_palette()[0])

    # Time series of referral requests
    requests_time_series(posts)

    # Distribution of user activity across firms
    requests = posts[posts['referral_request']].copy()
    offers = cmts[cmts['referral_offer']].copy()
    gini_curve(requests, label='requests')
    gini_curve(offers, label='offers')

