'''
===================================================================

Filtering Referral Requests and Defining Basic Features

===================================================================
'''

# Packages & Directories
from setup.utils import *
import dataprep.parse
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=psutil.cpu_count())

# Flag comments corresponding to referral offers
def flag_offers(cmts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Input: data frame of comments
    Output: (tuple)
        1) data frame with comment ID and referral offer flag
        2) aggregated number of referral offers by post ID
    '''

    dm = cmts[cmts['text'].str.contains(r'\bdm\b', case=False, na=False)][cmts.columns]
    dm['patterns'] = ''

    # Patterns to include as referral offers
    regex_include = {
        'dm*': r'^\s*dm\s*$',
        '*dm.!': r'\bdm\b\s*[.!]?$',
        '*dm,.!-*': r'\bdm\b\s*[-,.!]+\s+',
        'dm?': r'^\s*\bdm\b\?\s*$',
        'dm :)': r'\bdm\b\s*[:;]-?[)D]',
        'dm bro': r'\s*dm\s+bro(?:[\s\W]|$)',
        'dm pls': r'\s*dm\s+pls(?:[\s\W]|$)',
        'dm resume': r'\s*dm\s+resume(?:[\s\W]|$)',
        'dm cv': r'\s*dm\s+cv(?:[\s\W]|$)',
        'dm job': r'\s*dm\s+job(?:[\s\W]|$)',
        'dm role': r'\s*dm\s+role(?:[\s\W]|$)',
        'dm job link': r'\s*dm\s+job\s+link(?:[\s\W]|$)',
        'dm me': r'\s*dm\s+me(?:[\s\W]|$)',
        'dm ne': r'\s*dm\s+ne(?:[\s\W]|$)',
        'dm and': r'\bdm\s+and\b\s+',
        'dm here': r'\bdm\s+here\b\s*', 
        'dm if': r'\bdm\s+if\b\s+', 
        'dm - if': r'\bdm\s+-\s+if\b\s+', 
        'dm only if': r'\bdm\s+only\s+if\b\s+', 
        'dm and share': r'\bdm\s+and\s+share\b\s+', 
        'dm with': r'\bdm\s+with\s+\b', 
        'dm your': r'\bdm\s+your\s+\b', 
        'dm so that i': r'\bdm\s+so\s+that\s+i\b', 
        'dm for [char]': r'\bdm\s+for\s+.+',
        'dm in for [company]': r'\bdm\s+in\s+for\s+\w+\b',
        'dm for [company]': r'\bdm\s+for\s+\w+\b',
        'dm fo [company]': r'\bdm\s+fo\s+\w+\b',
        'dm or [company]': r'\bdm\s+or\s+\w+\b',
        'dm 4 [company]': r'\bdm\s+4\s+\w+\b',
        'dm me for [company]': r'\bdm\s+me\s+for\s+\w+\b', 
        'dm me or [company]': r'\bdm\s+me\s+or\s+\w+\b', 
        'dm me 4 [company]': r'\bdm\s+me\s+4\s+\w+\b', 
        'dm adobe': r'\bdm\s+adobe\b', 
        'dm canva': r'\bdm\s+canva\b', 
        'feel free to dm': r'\bfeel\s+free\s+to\s+dm\b',
        'send dm': r'^send\s+dm\b',
        'send me dm': r'^send\s+me\s+dm\b',
        'send me a dm': r'^send\s+me\s+a\s+dm\b',
        'shoot me dm': r'^shoot\s+me\s+dm\b',
        'shoot me a dm': r'^shoot\s+me\s+a\s+dm\b',
        'you can dm': r'^you\s+can\s+dm\b',
        'please dm *': r'^please\s+dm\s*',
        'pls dm *': r'^pls\s+dm\s*',
        '* please dm': r'\bplease\s+dm\b$',
        'dm please': r'^dm\s+please\s*',
        'happy to refer': r'\bhappy\s+to\s+refer\b\s*',
        'happy to help': r'\bhappy\s+to\s+help\b\s*',
        'I would refer': r'\bi\s+would\s+refer\b\s*',
        'I could refer': r'\bi\s+would\s+refer\b\s*',
        'I can refer': r'\bi\s+can\s+refer\b\s*',
        'I can help': r'\bi\s+can\s+help\b\s*',
        'I will refer': r'\bi\s+will\s+refer\b\s*',
        'Ill refer': r'\bill\s+refer\b\s*',
        'sure dm': r'\bsure[\s,.!?]*dm\b',
        'SAP referral': r'\bDM and anyone else for a referral\b',
        'DM (anyone else can also dm)': r'DM\s*\(\s*anyone\s*else\s*can\s*also\s*dm\s*\)',
        'DM (open for all)': r'DM\s*\(\s*open\s*for\s*all\s*\)',
    }

    # Patterns to exclude from referral offers
    regex_exclude = {
        'may I dm': r'\bmay\s+i\s+dm\b', 
        'I can dm': r'\bi\s+can\s+dm\b', 
        'can I dm': r'\bcan\s+i\s+dm\b', 
        'can I ask': r'\bcan\s+i\s+ask\b', 
        'can I please dm': r'\bcan\s+i\s+please\s+dm\b',
        'can I pls dm': r'\bcan\s+i\s+pls\s+dm\b',
        'could I please dm': r'\bcould\s+i\s+please\s+dm\b',
        'can you please dm': r'\bcan\s+you\s+please\s+dm\b',
        'sent dm': r'\bsent\s+dm\b',
        'dm you': r'\bdm\s+you\b',
        'how to dm': r'\bhow\s+to\s+dm\b',
        'how do we dm': r'\bhow\s+do\s+we\s+dm\b',
        'can not dm': r'\bcan\s+not\s+dm\b',
        'mind if I dm': r'\bmind\s+if\s+i\s+dm\b',
        'dm-d': r'\s*\bdm\'d\b\s*',
        'dm--d': r'\s*\bdm’d\b\s*',
        'dm-ed': r'\s*\bdm\'ed\b\s*',
        'dm--ed': r'\s*\bdm’ed\b\s*',
        'dm ed': r'\s*\bdm\s+ed\b\s*',
        'I am interested': r'\bi\s+am\s+interested\b', 
    }

    # Categorize all "DM" comments into patterns
    for regex_dict in [regex_include, regex_exclude]:
        for label, pattern in regex_dict.items():
            dm.loc[dm['text'].str.contains(pattern, case=False, na=False), 'patterns'] = label

    dm.loc[dm['patterns'] == '', 'patterns'] = '(no match)'
    assert (dm['patterns'] == '').sum() == 0

    # Mark whether comment belongs to referral offer or not
    dm['referral_offer'] = None
    dm.loc[ dm['patterns'].isin(regex_include.keys()) , 'referral_offer'] = True
    dm.loc[ dm['patterns'].isin(regex_exclude.keys()) , 'referral_offer'] = False
    dm.loc[ dm['patterns'] == '(no match)' , 'referral_offer'] = False
    assert dm['referral_offer'].isna().sum() == 0

    # Merge back with original comments data
    cmts = cmts[['url','post_id','comment_id','text']]
    cmts = cmts.merge(dm[['comment_id','referral_offer']], on='comment_id', how='left')
    cmts.loc[cmts['referral_offer'].isna(), 'referral_offer'] = False
    cmts['referral_offer'] = cmts['referral_offer'].astype(int)
    assert dm['comment_id'].nunique() == dm.shape[0]
    assert cmts['comment_id'].nunique() == cmts.shape[0]
    assert cmts['referral_offer'].isna().sum() == 0

    # Return offer flag back to boolean
    cmts['referral_offer'] = cmts['referral_offer'].astype(bool)

    # Return filtered comments data frames
    return cmts[['post_id','comment_id','referral_offer']]

# Flag posts that correspond to referral requests
def flag_requests(posts: pd.DataFrame) -> pd.DataFrame:
    '''
    Input: data frame of posts
    Output: data frame with post ID, filter flags, and text columns
    '''

    # Concise referral-related pattern to capture broad mentions of referrals
    referral_pattern = r'\b(referral|referrals|refer|dm me|help|seeking|looking for|need|request)\b|#referral|#needajob'

    # Refined request patterns, focusing only on request-related language
    request_patterns = [
        # Direct Needs or Requests
        r'\bneed a\b', r'\blooking for a\b', r'\bseeking a\b', r'\brequesting a\b',
        r'\bneed\b', r'\bneeded\b', r'\blooking for\b', r'\bseeking\b', r'\brequesting\b',
        r'\brequest\b', r'\brequested\b',

        # Asking for Assistance - "Can," "Could," "Would" Phrasing
        r'\bcan anyone\b', r'\bcan someone\b', r'\bcould anyone\b', r'\bcould someone\b',
        r'\bwould anyone\b', r'\bis anyone able to\b', r'\bcan anyone help\b',
        r'\bcan anyone help me\b', r'\bcan anyone help me out\b', r'\bcould anyone help\b',
        r'\bcould someone help\b', r'\bcould anyone assist\b', r'\bwould anyone be able to\b',
        r'\bcan i get\b', r'\bcan you provide me\b',

        # Direct Requests for Referral-Related Actions
        r'\brefer me\b', r'\brecommend me\b', r'\bconnect me\b', r'\bconnect me for\b',
        r'\brefer me to\b', r'\bhelp with a referral\b', r'\bhelp me get\b', r'\bhelp me out\b',
        r'\bset me up with\b', r'\bset me up with a referral\b', r'\bhook me up with\b', r'\bhook me up with a\b',

        # Specific Requests for Submission or Referrals
        r'\bwilling to submit me\b', r'\bwilling to give a referral\b', r'\bwilling to refer\b',
        r'\bkindly help me to get\b', r'\bsubmit me for a referral\b', r'\bwould really appreciate a referral\b',
        r'\bappreciate any referrals\b', r'\bappreciate a referral\b', r'\banyone here willing to submit\b',

        # Indirect Requests - Connections, Assistance, General Help
        r'\banyone have connections at\b', r'\banyone with connections at\b',
        r'\bdoes anyone know\b', r'\bdoes anyone work for\b', r'\banyone know someone at\b',
        r'\banyone familiar with\b', r'\banyone able to assist\b', r'\banyone who could help\b', 
        r'\banyone able to help\b', r'\banyone able to provide\b',

        # Phrases for Openness to Referral Assistance
        r'\banyone open to\b', r'\banyone available for\b', r'\banyone open to a referral\b', r'\banyone available for a referral\b',

        # Phrases for Willingness or Ability to Give a Referral
        r'\bis someone willing\b', r'\banyone willing to give\b', r'\banyone willing to give out\b',
        r'\banyone able to give\b', r'\banyone able to give out\b', r'\bif anyone would be willing\b',

        # Expressions of Help or Interest - "Would Appreciate," "Any Chance," etc.
        r'\bwould appreciate\b', r'\bwould greatly appreciate\b', r'\bwould be greatly appreciated\b',
        r'\bwould love\b', r'\bany chance of\b', r'\bis there anyone\b',
        r'\bi need\b', r'\bi am interested in\b',

        # Polite Requests - "Please" Phrasing
        r'\bplease dm\b', r'\bplease refer\b', r'\bplease connect\b', r'\bplease help\b',
        r'\bplease assist\b', r'\bplease recommend\b', r'\bhelp me with\b', r'\bneed assistance\b'
        r'\bcan i please\b',r'\breferral please\b',
    ]

    # Function to identify if a post contains general referral-related content
    def contains_referral(text, title):
        combined_text = f"{title} {text}"
        return bool(re.search(referral_pattern, combined_text, re.IGNORECASE))

    # Function to classify explicit referral requests
    def is_request(text, title):
        combined_text = f"{title} {text}"
        return any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in request_patterns)

    # Apply regex patterns to identify referral posts & request posts
    posts['is_referral'] = posts.parallel_apply(lambda row: contains_referral(row['text'], row['title']), axis=1)
    posts['is_request'] = posts.parallel_apply(lambda row: is_request(row['text'], row['title']), axis=1)

    # Referral requests if both are true
    posts['referral_request'] = posts['is_referral'] & posts['is_request']
    assert posts.shape[0] == posts['post_id'].nunique()
    return posts[['post_id','referral_request','is_referral','is_request']]

# Define binary / multi-class labels for each target metric
def class_labels(posts: pd.DataFrame) -> pd.DataFrame:

    # Loop over target metrics
    target_metrics = ['views','likes','comments','offers']

    for col in target_metrics:

        colname = f'class_{col}'

        # Five quantile bins for number of views
        if col == 'views':
            posts[colname] = pd.qcut(posts[f'num_{col}'], q=5, labels=False)

        # Zero VS Positive for other metrics
        else:
            posts[colname] = posts[f'num_{col}'] > 0

        # Recast as integer type
        posts[colname] = posts[colname].astype(int)

    # Return data frame
    return posts

# Perform train/test sample splits
def mark_train_test(posts: pd.DataFrame) -> pd.DataFrame:

    # Obtain the 80th percentile of dates among referral requests
    posts['date'] = pd.to_datetime(posts['date'])
    reqs = posts[posts['referral_request']].copy()
    pct80 = reqs['date'].quantile(0.80)

    # Split into train/test sets based on recency
    posts['train'] = (posts['date'] <= pct80) & posts['referral_request']
    posts['test'] = (posts['date'] > pct80) & posts['referral_request']

    return posts

# Non-text features for target prediction
def nontext_features(posts: pd.DataFrame) -> pd.DataFrame:

    # Month & day-of-week
    posts['date'] = pd.to_datetime(posts['date'])
    posts['day_of_week'] = posts['date'].dt.day_name()

    # Hashtag counts
    count_hash = lambda x: len(re.findall(r'#\w+', str(x)))
    posts['num_hashtags'] = posts['hashtags'].parallel_apply(count_hash)

    # Firm names without "ex-" prefix
    posts['user_firm_recoded'] = posts['user_firm'].parallel_apply(lambda x: str(x).replace('ex-',''))

    # Target encoding for user firm (proportion with positive label)
    # Calculate means only with requests in train data to avoid leakage
    reqs = deepcopy(posts[posts['referral_request'] & posts['train']])
    reqs['firm_count'] = reqs.groupby('user_firm')['post_id'].transform('count')
    reqs.loc[ reqs['firm_count'] < 10, 'user_firm_recoded' ] = 'SMALL_FIRM'

    for col in ['likes','comments','offers']:
        oldcol, newcol = f'class_{col}', f'firm_prop_{col}'
        props = reqs.groupby('user_firm_recoded')[oldcol].mean()
        small_firm_props = props['SMALL_FIRM']
        props = props.reset_index().rename(columns={oldcol: newcol})
        posts = posts.merge(props, on='user_firm_recoded', how='left')
        posts.loc[ posts[newcol].isnull(), newcol] = small_firm_props
        assert posts[newcol].isnull().sum() == 0

    return posts

# Sort & reorder columns prior to exporting
def format_columns(posts: pd.DataFrame) -> pd.DataFrame:
    '''
    Format columns before exporting
    '''

    # Column types
    target_metrics = ['offers','comments','likes','views']
    metric_cols = [f'class_{metric}' for metric in target_metrics]
    metric_cols = ['num_offers'] + metric_cols
    ref_cols = ['is_referral','is_request','referral_request']
    split_cols = ['train','test']
    nontext_cols = ['day_of_week','num_hashtags'] + [col for col in posts.columns if 'firm_prop' in col]

    # Order & sort columns
    posts = posts[['post_id'] + metric_cols + ref_cols + split_cols + nontext_cols]
    posts = posts.sort_values(['is_referral', 'is_request'], ascending=False)
    return posts

# Unit tests
def unit_tests(posts):

    # No missing values except text column
    exceptions = ['text','hashtags'] + [col for col in posts.columns if 'firm_prop' in col]
    colnames = [col for col in posts.columns if col not in exceptions]
    assert posts[colnames].isnull().sum().sum() == 0

    # Valid values for target metrics
    target_metrics = ['views','likes','comments','offers']

    for col in target_metrics:

        assert posts[f'num_{col}'].min() >= 0

        if col == 'views':
            assert posts[f'class_{col}'].isin([0,1,2,3,4]).all()
        else:
            assert posts[f'class_{col}'].isin([0,1]).all()
            assert ((posts[f'num_{col}'] > 0) == (posts[f'class_{col}'] == 1)).all()

    # Referral flags are boolean
    for col in ['is_referral','is_request','referral_request']:
        assert posts[col].dtype == bool

    # Unit tests on train/test splits
    assert posts['train'].sum() > posts['test'].sum()
    assert posts['train'].sum() + posts['test'].sum() == posts['referral_request'].sum()
    assert ((~posts['train']) & (~posts['test'])).sum() == (~posts['referral_request']).sum()

# Add filter flags to posts and comments data frames
if __name__ == "__main__":

    # Add referral offer flags to comments & aggregate counts by post ID
    cmts = pd.read_csv(f"{CLEAN}/comments.csv", encoding='utf-8')
    cmts = cmts.drop(columns='referral_offer')
    offs = flag_offers(cmts)
    cmts = cmts.merge(offs, on=['post_id','comment_id'], how='inner')
    assert cmts['referral_offer'].dtype == bool
    cmts.to_csv(f"{CLEAN}/comments.csv", index=False)

    # Load data frame on posts
    posts = pd.read_csv(f"{CLEAN}/posts.csv", encoding='utf-8')
    dropcols = set(posts.columns) - {'post_id'}

    # Add referral offer flags to post IDs
    reqs = flag_requests(posts.copy())
    posts = posts.merge(reqs, on='post_id', how='inner')

    # Aggregate number of referral offers by post ID
    cmts = pd.read_csv(f"{CLEAN}/comments.csv", encoding='utf-8')
    agg = cmts.groupby('post_id')['referral_offer'].sum().reset_index()
    agg = agg.rename(columns={'referral_offer':'num_offers'})

    posts = posts.merge(agg,  on='post_id', how='left')
    posts.loc[ posts['num_offers'].isnull() , 'num_offers'] = 0
    posts['num_offers'] = posts['num_offers'].astype(int)

    # Add binary / multi-class labels for each target metric
    posts = class_labels(posts)

    # Split referral requests into train & test
    posts = mark_train_test(posts)

    # Non-text features for target prediction
    posts = nontext_features(posts)

    # Unit tests
    unit_tests(posts)

    # Format columns before exporting
    posts = posts.drop(columns = dropcols)
    posts = format_columns(posts)

    # Export data frame after unit tests
    posts.to_csv(f"{CLEAN}/posts_basic.csv", index=False)
