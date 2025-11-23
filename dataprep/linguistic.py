'''
===================================================================

Defining Linguistic Features from Domain Knowledge

===================================================================
'''

# Packages & Directories
from setup.utils import *
import dataprep.parse
import dataprep.mask_tokens
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=psutil.cpu_count())

# Semantic attributes
def semantic_features(df, textcol='combined_mask'):
    """
    Extract semantic features for an entire dataframe using parallel processing with joblib.
    Returns a new dataframe with the original columns and extracted features.
    """

    spell = SpellChecker()

    # Function to extract semantic features for a single row
    def extract_features(row):
        """
        Extract semantic features for a single row in the dataframe.
        """
        features = {}

        # Feature: Type-Token Ratio (Unique Words / Total Words)
        total_words = len(str(row[textcol]).split())
        unique_words = len(set(str(row[textcol]).split()))
        features['type_token_ratio'] = unique_words / total_words if total_words > 0 else 0

        # Feature: Post Length (Word Count)
        features['post_length'] = len(str(row[textcol]).split())

        # Kincaid readability scores
        features['kincaid_grade'] = textstat.flesch_kincaid_grade(row[textcol])

        # Spelling error count with pyspellchecker
        text = re.sub(r'[^\w\s]', ' ', str(row[textcol]).lower())
        text = re.sub(r'\b(iclevel|leetcode|url|org|gpe|yrs|tc|dm|pip|pm|swe|ds|sde|de|us|eu|rif|aws|gcp)\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip().lower().split()
        features['spelling_error_count'] = len(spell.unknown(text))

        return features

    # Apply feature extraction function
    df[['type_token_ratio','post_length','kincaid_grade','spell_errors']] = df.parallel_apply(lambda row: pd.Series(extract_features(row)), axis=1)

    return df

# Attributes from specific knowledge on teamblind
def teamblind_features(df, textcol='combined_mask'):

    # Mentions layoff
    layoff_regex = r'\b(layoff(s)?|(laid|lay|laying)(\s+(me|us)\s+)?off|let(\s+(me|us)\s+)?go|(terminate|eliminate)(d)?(\s+(me|us)\s+)?)\b'
    df['layoff'] = df[textcol].str.contains(layoff_regex, case=False, na=False)

    # Talk about pip
    pip_regex = r"\b(pip|pipped|piped|pipd|pip'd|pre-pip|pre pip|rif|riffed|rifed|pre-rif|pre rif|rif/pip|pip/rif|in focus)\b"
    df['pip'] = df[textcol].str.contains(pip_regex, case=False, na=False)

    # Company Name Mentioned (Dynamic pattern to capture company names)
    df['company_name'] = df[textcol].str.contains('[ORG]', case=False, na=False)

    # Job Title Mentioned (Dynamic pattern to capture job titles)
    df['job_title'] = df[textcol].str.contains('[ROLE]', case=False, na=False)

    # Year of Experience Mentioned
    df['year_of_experience'] = df[textcol].str.contains('[YEARS]', na=False, case=False)

    # Total Compensation Mentioned
    df['total_compensation'] = df[textcol].str.contains('[SALARY]', na=False)

    # Reason for Seeking Career (other than lay-off or pip) Mentioned
    reason_regex = r"(?i)\b(?:seeking\s+new\s+opportunities|looking\s+for\s+growth|career\s+transition|transitioning\s+to|seeking\s+better\s+work-life\s+balance|laid\s+off)\b"
    df['reason_for_seeking_career'] = df[textcol].str.contains(reason_regex, na=False)

    # Previous Experience Mentioned
    previous_experience_regex = r"(?i)\b(designed|built|developed|automated|implemented|created|optimized|worked on|participated in|led|contributed to|teams at companies like|in projects for|for clients like|impactful projects|significant projects|AWS SDE[1-3]|over \d+\s+years? of experience\s+across\s+[a-zA-Z\s,]+|experience in\s+[a-zA-Z\s,]+|\b[a-zA-Z\s]+(consultant|analyst|associate|specialist|engineer|manager|director)\s+at\s+[a-zA-Z\s]+)\b"
    df['previous_experience'] = df[textcol].str.contains(previous_experience_regex, na=False)

    # Skillset Mentioned
    skillset_regex = r"(?i)\b(python|java|sql|excel|machine learning|project management|data analysis|cloud computing|HFSS|Matlab|MATLAB|ANSYS|Simulink|CAD|R|azure|pyspark|etl processes|dashboards|data solutions)\b"
    df['skillset'] = df[textcol].str.contains(skillset_regex, na=False)

    # Student Status Mentioned
    student_status_regex = r"(?i)\b(student|undergrad|graduate|master's|PhD|internship|will be a (college|university) graduate|graduating in \w+ \d{4}|I'm graduating in \w+|new grad|recent graduate)\b"
    df['student_status'] = df[textcol].str.contains(student_status_regex, na=False)

    return df

# Attributes from prior HR management literature
def domain_features(df, textcol='combined_mask'):
    # Define the regular expressions for each feature

    # Urgency Patterns (time-sensitive words or urgency signals)
    urgency_regex = r'(?i)\burgently\b|\basap\b|\btime[-\s]?sensitive\b|\bimmediately\b|\bimpress\b|\bpressing\b|\bdeadline\b|\bnow\b|\bquickly\b|\bspeedily\b|\brush\s+off\b|\bcrucial\b|\bimportant\b|\bwithin\s+the\s+next\b'
    df['urgency'] = df[textcol].str.contains(urgency_regex, na=False)

    # Gratitude Patterns (thankfulness, appreciation)
    gratitude_regex = r'(?i)\bthank\b|\bthanks\b|\bappreciate\b|\bgrateful\b|\bmuch\s+appreciated\b|\bthankful\b|\bgreatly\s+appreciated\b'
    df['gratitude'] = df[textcol].str.contains(gratitude_regex, na=False)

    # Politeness Patterns (polite phrases and courtesies)
    politeness_regex = r'(?i)\bplease\b|\bmay\s?I\b|\bcould\s?you\b|\bkindly\b|\bexcuse\s?me\b|\bif\s?you\s?please\b|\bI\s+would\s?appreciate\b'
    df['politeness'] = df[textcol].str.contains(politeness_regex, na=False)

    # Familiarity Patterns
    familiarity_regex = r'(?i)\b(?:Hey|Hi\s+everyone|Anyone\s+know|Can\s+anyone\s+help|Hey\s+guys|Hello\s+there)\b'
    df['familiarity'] = df[textcol].str.contains(familiarity_regex, na=False)

    # Being Desperate Patterns
    being_desperate_regex = r'(?i)\bplease\b|\bgot\s+stuck\b|\bdesperate\b|\bstruggling\b|\bhelp\s+me\b|\bneed\s+advice\b|\blost\b|\bcan\'t\s+find\b|\bany\s+help\s+would\s+be\s+great\b'
    df['being_desperate'] = df[textcol].str.contains(being_desperate_regex, na=False)

    # Inclusive/Exclusive Patterns
    inclusive_exclusive_regex = r'(?i)\b(?:we|I|they|us|our|their)\b'
    df['inclusive_exclusive'] = df[textcol].str.contains(inclusive_exclusive_regex, na=False)

    # Being Content Patterns
    being_content_regex = r'(?i)\bhappy\s+with\b|\bsatisfied\s+with\b|\blove\s+my\s+job\b|\benjoy\s+working\b'
    df['being_content'] = df[textcol].str.contains(being_content_regex, na=False)

    # Interview Readiness Patterns (keywords related to interview prep or readiness)
    interview_readiness_regex = r'(?i)\bleetcode\b|\binterview\s?questions\b|\bpreparation\b|\bmock\s?interview\b|\btechnical\s+interview\b'
    df['interview_readiness'] = df[textcol].str.contains(interview_readiness_regex, na=False)

    # Evidentiality Patterns
    evidentiality_regex = r'(?i)\b(?:have\s+strong\s+background|have\s+experience|am\s+skilled\s+in|bring\s+years\s+of|worked\s+on|familiar\s+with)\b'
    df['evidentiality'] = df[textcol].str.contains(evidentiality_regex, na=False)

    # Reciprocity Patterns
    reciprocity_regex = r'(?i)\bhappy\s+to\s+help\b|\boffer\s+help\s+in\s+return\b|\bassist\s+you\s+in\s+the\s+future\b|\bhelp\s+others\s+in\s+return\b'
    df['reciprocity'] = df[textcol].str.contains(reciprocity_regex, na=False)

    # High-Status Patterns
    high_status_regex = r'(?i)\b(?:been\s+in\s+the\s+industry\s+for\s+\d+\s+years|received\s+awards|promoted\s+to|led\s+teams|recognized\s+as\s+an\s+expert)\b'
    df['high_status'] = df[textcol].str.contains(high_status_regex, na=False)

    # Gain/Loss Framing Patterns
    gain_loss_framing_regex = r'(?i)\b(?:mean\s+a\s+lot|gain\s+so\s+much|change\s+my\s+life|impact\s+me|losing\s+hope)\b'
    df['gain_loss_framing'] = df[textcol].str.contains(gain_loss_framing_regex, na=False)

    return df

# Produce domain knowledge features
if __name__ == '__main__':

    # Import posts data file
    posts = pd.read_csv(f"{CLEAN}/posts.csv", encoding='utf-8')
    mask = pd.read_csv(f"{CLEAN}/posts_mask.csv", encoding='utf-8')
    posts = posts.merge(mask, on='post_id', how='inner')
    dropcols = [col for col in posts.columns if col != 'post_id']

    # Semantic attributes
    feats = semantic_features(posts.copy())

    # Attributes from specific knowledge on teamblind
    feats = teamblind_features(feats)

    # Attributes from prior HR management literature
    feats = domain_features(feats)

    # Check tabulations
    numcols = ['post_length', 'type_token_ratio','kincaid_grade','spell_errors']

    for col in feats.columns:
        if col in numcols:
            print(feats[col].describe().sort_index())
            print('\n',"-" * 20,'\n')
        elif col != 'post_id' and col not in dropcols:
            print(f"Value counts for column '{col}':")
            print(feats[col].value_counts(normalize=True))
            print('\n',"-" * 20,'\n')

    # Mark reg-ex columns below 5% true rate
    for col in feats.columns:
        if col != 'post_id' and col not in numcols and col not in dropcols:
            true_ratio = feats[col].value_counts(normalize=True)[True]
            if true_ratio < 0.05:
                feats.rename(columns={col: f"below5_{col}"}, inplace=True)

    # Remove redundant columns before exporting
    feats = feats.drop(columns=dropcols)
    below5_cols = [col for col in feats.columns if 'below5_' in col]
    other_cols = [col for col in feats.columns if col not in below5_cols]
    feats = feats[other_cols + below5_cols]
    feats.to_csv(f"{CLEAN}/posts_linguistic.csv", index=False)
