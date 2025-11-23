'''
===================================================================

Add Mask Tokens to Replace Company Names & Location Names

===================================================================
'''

# Packages & Directories
from setup.utils import *
import dataprep.parse

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=psutil.cpu_count())
SPACY_NLP = spacy.load('en_core_web_sm')

# Prepare list of company names, location names, and job titles to be masekd
def prep_company_list(cmts: pd.DataFrame) -> list:

    # List of company names without 'ex-' prefix
    firm_names = cmts[cmts['user_firm'].notnull()]['user_firm']
    firm_names = firm_names.str.replace('ex-', '').str.lower()
    firm_list = firm_names.unique().tolist()
    nospace_list = firm_names.str.replace(' ', '').unique().tolist()

    # Known company names
    extra_list = ['xoogle', 'xoogler', 'aws', 'zillow', 'cedar', 'incomm', 'incomm payments', 'jane street', 'visa', 'ms', 'msft', 'goog', 'jpmc','anthropic', 'disney','figma','snap chat','snapchat','duo lingo', 'duolingo','mindtickle', 'slalom', 'esri', 'reality labs', 'caret','c^ret','pseg', 'cvs','twitter','twitterx','twitter/x','twitter / x','citrix', 'cloud fare', 'citi bank', 'oci', 'swiggy', 'playstation', 'vast','andurilindustries', 'relativity space', 'zip recruiter','expedia', 'scale ai', 'square', 'chase', 'jp','jpm','jpmc', 'johnson', 'tia', 'venmo','azure','gcp','slack','nvidia','nvdia','morgan','amzn','meraki','fidelity', 'visa referral', 'ixl','dbt', 'oracle cloud', 'open ai', 'openai', 'delta airlines', 'delta', 'united airlines', 'american air','american airlines','aa','citi','vantara','sanofi', 'booz allen','lockheed martin','samara','monzo', 'door dash', 'amazonian', 'amazonians', 'garena', 'okx','oraclecloud', 'oracle cloud', 'akamai', 'alibaba', 'alation', 'amakunacapital', 'appdynamics', 'baidu', 'barclays', 'blizzard', 'booking', 'cameo', 'codenation', 'deshaw', 'didi', 'dunzo', 'epic', 'evernote', 'garena', 'hrt', 'hulu', 'janestreet', 'jpmorgan', 'patreon', 'pocketgems', 'ponyai', 'poshmark', 'postmates', 'quip', 'sapient', 'splice', 'triplebyte', 'yandynga', 'zappos', 'zulily', 'facebook', 'brgoogle', 'cashapp', 'cash app', 'mongodb','symbotic', 'youtube', 'discover', 'rtx', 'pnw','zapier','millenium','millennium','wework','amramp','cockroachlabs','pluto','adobe','surescripts','dentsu', 'tik tok', 'hims & hers health', 'fiserve', 'mollie', 'catawiki', 'kainos', 'linked in', 'redbull', 'red bull', 'aha!', 'renesas', 'proctor and gamble', 'proctor & gamble', 'P&G', 'P & G', 'PG&E', 'PG & E', 'whatsapp', 'Merkle', 'Veradigm', 'Captech','Bose','Hubell','Pinnaclegroup','Pinnaclegroupinc','Donux','Koent','Kong','Mightyhive']

    # General terms that overlap with company names
    exclude = {'new', 'here', 'january', 'visa', 'match', 'tableu', 'tableau','outreach', 'chief', 'mongodb'}

    # Sort company names by token length
    company_names = set(firm_list + nospace_list + extra_list) - exclude
    sorted_companies = sorted(list(company_names), key=lambda x: len(x.split()), reverse=True)
    return sorted_companies

def prep_location_list() -> list:

    # Load names of countries and cities from geonamescach
    gc = geonamescache.GeonamesCache()
    countries = gc.get_countries()
    cities = gc.get_cities()
    country_names = [country['name'] for country in countries.values()]
    city_names = [city['name'] for city in cities.values()]
    location_names = country_names + city_names

    # Custom list of location names observed from Blind
    custom_locations = [
        "canada", "india", "finland", "england", "europe","ireland", "scotland", "switzerland", "germany", "france", "italy", "spain", "china",
        "japan", "korea", "south korea", "north korea", "russia", "australia",
        "new zealand", "mexico", "brazil", "argentina", "egypt", "nigeria",
        "south africa", "turkey", "saudi arabia", "qatar", "singapore", "malaysia",
        "thailand", "indonesia", "philippines", "vietnam", "london", "paris",
        "tokyo", "new york", "los angeles", "chicago", "beijing", "shanghai",
        "hong kong", "dubai", "mumbai", "delhi", "bangalore", "sydney", "melbourne",
        "united states", "great britain", "united kingdom", "barcelona", "remote europe",
        "remote us", 'west coast', 'east coast', 'eastern europe', 'western europe', 'bay area', 'midwest', 'mid west', 'south', 'seattle', 'mission bay', 'manhattan', 'brooklyn', 'bayarea'
    ]

    custom_locations += ['al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy'] + ['eu',"usa", "uk", "uae", "nyc",'sf' ,'/us', '-us', 'the us', 'in us','u.s','u.s.','the u.s.', 'the u.s' 'across us' 'sfo', 'lax', 'lga', 'nyc', 'sea','(us)','oc','us/','/us','us based', 'outside us','/can','/can','can.']

    custom_locations += location_names

    # Remove unicode characters & convert to lower case
    custom_locations = [unidecode.unidecode(loc).lower() for loc in custom_locations]

    # Add hyphenated versions of custom locations
    hyphens = [loc.replace(' ', '-') for loc in custom_locations if ' ' in loc]
    spaces = [loc.replace(' ', '') for loc in custom_locations if ' ' in loc]
    custom_locations += hyphens + spaces

    # Remove duplicates
    custom_locations = set(custom_locations)
    custom_locations = list(custom_locations)

    # Sort by token length & string length
    custom_locations = sorted(custom_locations, key=lambda name: (len(name.split()), len(name)), reverse=True)
    assert len(custom_locations) == len(set(custom_locations))


    return custom_locations

def prep_role_list() -> list:
    pm_roles = ['product','program','project','account','sales','sale','customer'] + ['PM','TPM','PGM','PJM','PMT','PMS','UX','UXD']
    pm_roles = ['PM','TPM','PGM','PJM','PMT','PMS','UX','UXD'] + [role + ' manager' for role in pm_roles] + [role + ' management' for role in pm_roles] + [role + ' designer' for role in pm_roles] + [role + ' design' for role in pm_roles]
    pm_roles += ['technical ' + role for role in pm_roles] + ['tech ' + role for role in pm_roles]

    eng_roles = ['backend','frontend','python','software','software development','software developer','software dev','software engineering','softwareengineering','software engineer','software eng','machine learning', 'artificial intelligence','full stack', 'full-stack', 'business', 'business intelligence', 'business intel', 'biz', 'biz intelligence', 'biz intel', 'research', 'data', 'engineer', 'engineering', 'solutions', 'developer', 'techical support', 'support', 'mechanical','mechanic','thermal','mechanics', 'document','cloud','testing','software testing','quality','quality assurance', 'mlops', 'ml ops','analytics'] + ['SWE', 'SDE', 'DE', 'ML', 'MLE', 'AI', 'ML/AI', 'AI/ML', 'BI', 'BIE','STE','SDET','QA']
    eng_roles += [eng + ' eng' for eng in eng_roles if 'eng' not in eng] + [eng + ' engineer' for eng in eng_roles if 'engineer' not in eng] + [eng + ' engineering' for eng in eng_roles if 'engineering' not in eng] + [eng + ' dev' for eng in eng_roles if 'dev' not in eng] + [eng + ' development' for eng in eng_roles if 'development' not in eng] + [eng + ' developer' for eng in eng_roles if 'developer' not in eng] + [eng + ' devops' for eng in eng_roles if 'devops' not in eng] + [eng + ' role' for eng in eng_roles if 'role' not in eng] + [eng + ' roles' for eng in eng_roles if 'roles' not in eng] + [eng + '/' for eng in eng_roles] + ['/' + eng for eng in eng_roles] + [eng + '-' for eng in eng_roles] + ['(' + eng + ')' for eng in eng_roles]
    eng_roles = list(set(eng_roles) - set(['ML', 'AI', 'ML/AI', 'AI/ML','support','document','business','research','data','solutions','cloud','testing','quality','python'])) + ['sde1','sde2','sde3','sde4','sde5','sde6','sde7','sde8','sde9','sde10'] + ['swe1','swe2','swe3','swe4','swe5','swe6','swe7','swe8','swe9','swe10']

    sci_roles = ['applied', 'research', 'research data', 'algorithm', 'algorithms', 'algorithms data', 'data', 'business', 'scientist', 'science', 'software', 'marketing', 'marketing data', 'product marketing', 'product marketing data'] + ['DS','DE','DA','RS','AI','ML','AI/ML','ML/AI']
    sci_roles = [sci + ' science' for sci in sci_roles if 'science' not in sci] + [sci + ' scientist' for sci in sci_roles if 'scientist' not in sci] + [sci + ' research' for sci in sci_roles if 'research' not in sci] + [sci + ' researcher' for sci in sci_roles if 'researcher' not in sci]

    anal_roles = ['data', 'business data', 'biz data', 'research', 'research data', 'HR', 'Human Resource', 'Human Resources', 'marketing']
    anal_roles += [anal + ' analyst' for anal in anal_roles] + [anal + ' analytics' for anal in anal_roles] + [anal + ' analysis' for anal in anal_roles] + ['market analyst']

    other_roles = ['consultant','trader', 'HR', 'recruiter', 'recruitment']

    roles = pm_roles + eng_roles + sci_roles + anal_roles + other_roles
    roles += [r + ',' for r in roles]
    roles = sorted(roles, key=lambda x: len(x.split()), reverse=True)

    return roles

# Prepare stopwords for tokenization
def prep_stopwords(stop_type: str) -> set:

    assert stop_type in ['nltk','spacy','both']

    # Load stop words
    spacy_stops = SPACY_NLP.Defaults.stop_words
    nltk_stops = set(nltk.corpus.stopwords.words('english'))

    # Define stopwords
    if stop_type == 'nltk':
        stopwords = nltk_stops
    elif stop_type == 'spacy':
        stopwords = spacy_stops
    elif stop_type == 'both':
        stopwords = set().union(spacy_stops, nltk_stops)

    # Remove exceptions
    stopwords = stopwords - {'rather','nothing','anyone','first','mostly','please','during','someone','serious','whoever','only','because','except','anything','again','since','whoever','something','whenever','everyone','nobody','across','upon','within','while','really','several','give','get','many','whatever','where','down','perhaps','anywhere','toward','became','somewhere', 'somehow','nowhere','should','wherever','everywhere','elsewhere','seemed','even','everything','least','towards','use','full','about','enough','sometime','after','made','along','haven\'t','throughout','might','how','never','although','beforehand','become','anyway', 'off'}

    return stopwords

# Mask company names, job titles, and other factual info
def mask_companies(text, sorted_companies: list, sorted_roles: list) -> pd.Series:

    # Convert null strings into empty strings
    if not isinstance(text, str):
        text = str(text) if not pd.isna(text) else ""

    # Lower-case everything except for the first word at the beginning of each sentence
    # (to avoid NER parsers confusing job titles for org entities)
    sentences = []
    current = []
    recent = ''
    old_text = deepcopy(str(text))
    tokens = re.split(r'([\s])', old_text.lower())
    is_special = lambda tok: '[TITLE]' in tok.upper() or '[CONTENT]' in tok.upper()

    for i,tok in enumerate(tokens):

        # Add space tokens as is
        if tok == '' or tok.isspace():
            current.append(tok)
        else:
            # Convert special tokens into uppercase
            if is_special(tok):
                current.append(tok.upper())
            
            # Capitalize tokens after special tokens
            elif is_special(recent):
                current.append(tok.capitalize())

            # Add other tokens as is
            else:
                current.append(tok)

            # Update most recent token
            recent = tok

        # Append sentence if punctuation is found
        if tok.endswith(('.', '!', '?', '\n')) or is_special(tok):
            sent = ''.join(current)
            sentences.append(sent)
            current = []

    # List of sentences in text
    if current is not None:
        sentences.append(''.join(current))
    
    # Capitalize first letter of each sentence
    new_sent = []

    for sent in sentences:

        if is_special(sent):
            new_sent.append(sent.upper())
        else:
            content = sent.strip()
            newcontent = content.capitalize()
            new_sent.append(sent.replace(content, newcontent).strip(' '))

    text = ' '.join(new_sent).replace('\n ', '\n')

    # Replace job titles
    role_patterns = [r'\b' + re.escape(role) + r'\b' for role in sorted_roles if role.lower() != 'python']
    pattern = '|'.join(role_patterns)
    role_regex = re.compile(pattern, flags=re.IGNORECASE)
    text = role_regex.sub('[ROLE]', text)


    # Total comp with numbers before or after
    adj_levels = ['total', 'current', 'previous', 'curr', 'prev', 'tot', 'was']
    comp_levels = ['compensation', 'comp', 'pay', 'salary', 'package', 'TC']
    adj_comp_levels = [f"{adj} {comp}"for adj in adj_levels for comp in comp_levels] + comp_levels
    adj_comp_levels += [txt + ',' for txt in adj_comp_levels]
    
    for comp in adj_comp_levels:
        # Numbers before the phrase (e.g., "150k salary", "150k+ salary", "150+k salary")
        pattern = re.compile(f'\\b((?:\\$|dollars?|£|pounds?|€|euros?|₹|rupees?)?\\d+\\+?[kKmM]?\\+?)\\s*{re.escape(comp)}\\b', flags=re.IGNORECASE)
        text = pattern.sub('[SALARY]', text)
        
        # Numbers after the phrase (e.g., "salary: $150k", "salary: $150k+", "salary: $150+k")
        pattern = re.compile(f'\\b{re.escape(comp)}\\s*[:=]?\\s*((?:\\$|dollars?|£|pounds?|€|euros?|₹|rupees?)?\\d+\\+?[kKmM]?\\+?)\\b', flags=re.IGNORECASE)
        text = pattern.sub('[SALARY]', text)


    # Years of experience with numbers before or after
    yoe_levels = ['yrs of experience', 'years of experience', 'yrs of', 'years of', 'yoe', 'years', 'yrs', 'year']
    yoe_levels += [txt + ',' for txt in yoe_levels]
    
    for yoe in yoe_levels:
        # Numbers before the phrase (e.g., "5 years of experience", "5+ years of experience")
        pattern = re.compile(f'\\b(\\d+)\\+?\\s*{re.escape(yoe)}\\b', flags=re.IGNORECASE)
        text = pattern.sub('[YEARS]', text)
        
        # Numbers after the phrase (e.g., "yoe 5", "yoe 5+")
        pattern = re.compile(f'\\b{re.escape(yoe)}\\s*[:=]?\\s*(\\d+)\\+?\\b', flags=re.IGNORECASE)
        text = pattern.sub('[YEARS]', text)


    # Leetcode
    lc_levels = ['leetcode','lc']
    lc_levels += [txt + ',' for txt in lc_levels]
    
    for lc in lc_levels:

        pattern = re.compile(f'\\b(\\d+)\\+?[kK]?\\+?\\s*{re.escape(lc)}\\b', flags=re.IGNORECASE)
        text = pattern.sub('[LEETCODE]', text)

        pattern = re.compile(f'\\b{re.escape(lc)}\\s*[:=]?\\s*(\\d+)\\+?[kK]?\\+?\\b', flags=re.IGNORECASE)
        text = pattern.sub('[LEETCODE]', text)


    # Standardize job levels to avoid revealing company identity
    m_levels = ['Senior Manager','Manager','EM','PEM','SDM','MGR','EDM']
    ic_levels = ['Entry','Junior','Senior','Staff','Senior/Staff','Principal','JR','SR']
    ic_levels = [level + ' level\s?' for level in ic_levels] + [level + '-level\s?' for level in ic_levels]
    ic_levels += ['Level\s?','Senior/Staff\s?','Grade\s?','Band\s?']
    ic_levels += ['new grad','post grad','tech lead','new-grad','tech-lead','intern','interns','internship','internships','undergrad','undergrads','grads', 'grad','undergraduates','graduates','undergraduate','ICT','ICL','IC','E','L','I','T',]

    m_levels += [txt + ',' for txt in m_levels]
    ic_levels += [txt + ',' for txt in ic_levels]

    for i,fmt in enumerate(['\s*\d+\.\d', '\s*\d+', '\s*(?:i|ii|iii|iv|v|vi|vii|viii)']):

        # Managers
        for lv in m_levels:
            pattern = re.compile(f'\\b{lv + fmt}\\b', flags=re.IGNORECASE)
            text = pattern.sub('[MANAGER]', text)

            if i == 2:
                pattern = re.compile(f'\\b{lv}\\b', flags=re.IGNORECASE)
                text = pattern.sub('[MANAGER]', text)

        # Individual Contributors
        for lv in ic_levels:
            pattern = re.compile(f'\\b{lv + fmt}\\b', flags=re.IGNORECASE)
            text = pattern.sub('[ICLEVEL]', text)

            if i == 2:
                pattern = re.compile(f'\\b(?:junior|senior|staff|jr|sr)\\s*{lv}\\b', flags=re.IGNORECASE)
                text = pattern.sub('[ICLEVEL]', text)

                if len(lv) > 2:
                    pattern = re.compile(f'\\b{lv}\\b', flags=re.IGNORECASE)
                    text = pattern.sub('[ICLEVEL]', text)

    pattern = re.compile(f'\\b(mid|junior|senior|staff|principal|jr|sr)\\b', flags=re.IGNORECASE)
    text = pattern.sub('[ICLEVEL]', text)


    # Industry
    industry_list = ['supply chain', 'big tech', 'management consulting', 'motor sport', 'finance tech', 'social media', 'social networking', 'mechanical engineering', 'tech','finance', 'consulting', 'healthcare', 'retail', 'media', 'transportation', 'logistics',  'rideshare', 'saas', 'ecommerce', 'security', 'cybersecurity', 'crypto', 'motorsport', 'fintech', 'marketplace', 'ads', 'advertisement', 'manufacturing', 'manufacture','sustainability']
    industry_list += [txt + ',' for txt in industry_list]
    industry_patterns = [r'\b' + re.escape(indus) + r'\b' for indus in industry_list]
    pattern = '|'.join(industry_patterns)
    industry_regex = re.compile(pattern, flags=re.IGNORECASE)
    text = industry_regex.sub('[INDUSTRY]', text)


    # Remove URLs, hashtags, 'ex-' prefix for company names
    text = re.sub(r'http\S+', '[URL]', text, flags=re.IGNORECASE)
    text = re.sub(r'www.\S+', '[URL]', text, flags=re.IGNORECASE)
    text = re.sub(r'#(\w+)', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'@(\w+)', r'\1', text, flags=re.IGNORECASE)
    text = text.replace('ex-', '').replace('Ex-', '').replace('ex ', '').replace('Ex ', '')


    # Grammar for I
    text = text.replace('i\'m', 'I\'m').replace(' i ', ' I ')
    text = text.replace(' ii', ' II').replace('ii ', 'II ')


    # Blind-specific terms
    for special in ['i', 'tc','yoe','col']:
        text = re.sub(r'\b'+special+r'\b', f'{special.upper()}', text, flags=re.IGNORECASE)


    # Job titles & numbers that should NOT be mistaken for org names
    except_list = ['partner','product','sde','tc','yoe','swe','pm','phd','kindest','’m','lookout','wedding']
    except_list += [f'{i}k' for i in range(10)]
    pattern = '|'.join(except_list)
    except_regex = re.compile(pattern, flags=re.IGNORECASE)


    # Use NER in Spacy to replace organization names with [ORG_NAME]
    replacements = []
    emojis = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001F9FF]'

    for ent in reversed(SPACY_NLP(text.lower()).ents):
        if (ent.label_ == 'ORG' 
            and not bool(except_regex.search(ent.text))
            and not bool(re.search(emojis, ent.text))
        ):
            replacements.append((ent.start_char, ent.end_char))

    for start, end in replacements:
        text = text[:start] + '[ORG]' + text[end:]


    # Mask company names with curated list
    company_patterns = [r'\b' + re.escape(company) + r'\b' for company in sorted_companies]
    pattern = '|'.join(company_patterns)
    company_regex = re.compile(pattern, flags=re.IGNORECASE)
    text = company_regex.sub('[ORG]', text)

    # Replace redundant brackets
    text = re.sub(r'\[+', '[', text)
    text = re.sub(r'\]+', ']', text)
    # print(f"{old_text}\n\n{text}")

    return text

# Mask locations
def mask_locations(text: str, location_names: list, sorted_companies: list) -> str:

    # Exceptions to not mask
    exceptions = {'bar','hook','hit','in','id','or','hi','me','can','hm','of','along','officer','python','nice','most','university','same','opportunity','than','come','god','much', 'management', 'refer', 'swe','ds', 'normal', 'pace', 'tc','nightfall', 'persona', 'minware','Ad Tech', 'columbia', 'january','february', 'march','april','may','june','july','august','september','october','november','december','deal','cv','enterprise', 'advance', 'finance', 'mentor', 'best', 'co','goes', 'de', 'date', 'node', 'sake','pop','save', 'buy', 'berkeley', 'peer','lend', 'golden', 'mission','surprise'}
    exceptions = exceptions.union(sorted_companies)
    exceptions = {x.lower() for x in exceptions}

    # Replace GPE entities with [GPE] tokens while avoiding duplicates
    replaced_entities = set()

    for ent in SPACY_NLP(text.lower()).ents:
        word = ent.text.lower()
        if (ent.label_ == "GPE" 
            and word not in replaced_entities 
            and word not in exceptions
            and '0k' not in word
        ):
            text = re.sub(r'\b' + re.escape(ent.text) + r'\b', "[GPE]", text, flags=re.IGNORECASE)
            replaced_entities.add(ent.text)

    # Replace custom matched locations with [GPE] tokens, excluding already replaced ones
    exclude = set(list(exceptions) + list(replaced_entities))
    pattern = '|'.join(
        r'\b' + re.escape(loc) + r'\b' 
        for loc in location_names 
        if loc.lower() not in exclude
    )
    text = re.sub(pattern, '[GPE]', text, flags=re.IGNORECASE)

    # Replace redundant brackets
    text = re.sub(r'\[+', '[', text)
    text = re.sub(r'\]+', ']', text)

    # Add spacing for special tokens
    for tok in MASK_TOKENS:
        text = text.replace(tok, f' {tok} ').replace('  ', ' ').replace(' ,', ',')
    
    text = text.replace(" [TITLE]", "[TITLE]").replace("\n\n [CONTENT] \n\n", "\n\n[CONTENT]\n\n")

    return text

# Text processing for n-gram models
def process_tokens(text: str, stopwords) -> pd.Series:

    # Convert to lower cases & remove space-like characters
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    # Remove stopwords before & after punctuation removal
    remove_stop = lambda text: [word for word in text.split() if word not in stopwords]
    text = ' '.join(remove_stop(text))
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\[\]]', ' ', text)
    text = ' '.join(remove_stop(text))
    text = re.sub(r'\s+', ' ', text)

    # Lemmatize into root words
    roots = []

    for token in text.split():

        if re.match(r'\[.*\]', token):
            roots.append(token.upper())
        else:
            roots.append(SPACY_NLP(token)[0].lemma_)

    text = ' '.join(roots)

    # Datum --> data
    text = re.sub(r'\bdatum\b', 'data', text, flags=re.IGNORECASE)

    return text

# Comparing text columns before & after masking
def debug_print(posts, textcol='combined', task='mask', compute=None, num=20, sample='head'):
    """
    Debugging function to compare text columns before & after masking function

    Inputs:
        - posts: data frame of posts
        - textcol: column name of text to be masked
        - task: task to debug (masking company names or extracting tokens)
        - compute: masking function to be applied (for debugging)
        - num: number of samples to print
        - sample: head, tail, or random
    """

    # Check argument type
    assert textcol in ['combined','title','text']
    assert sample in ['head','tail','random']
    assert task in ['mask','tokenize']

    # Set text columns
    if task == 'mask':
        source, target = textcol, f"{textcol}_mask"
    elif task == 'tokenize':
        source, target = f'{textcol}_mask', f'{textcol}_tokens'

    # Take samples
    if sample == 'head':
        sampdf = posts.head(num)
    elif sample == 'tail':
        sampdf = posts.tail(num)
    elif sample == 'random':
        sampdf = posts.sample(num)
    
    # Apply masking function on sample
    if compute is not None:
        sampdf[target] = sampdf[source].parallel_apply(compute)

    # Print text before VS after masking function
    for i in range(num):
        print("\n=============== Original Text ===============\n")
        print(sampdf.iloc[i,:][source])
        print("\n\n--------------- Processed Text ---------------\n")
        print(sampdf.iloc[i,:][target])
        print("\n\n\n\n")

# Add filter flags to posts and comments data frames
if __name__ == "__main__":

    # Load data on posts and comments
    cmts = pd.read_csv(f"{CLEAN}/comments.csv", encoding='utf-8')
    posts = pd.read_csv(f"{CLEAN}/posts.csv", encoding='utf-8')
    dropcols = [col for col in posts.columns if col != 'post_id']

    # Prepare list of company names & location names to be masked
    company_names = prep_company_list(cmts)
    location_names = prep_location_list()
    role_names = prep_role_list()

    # Mask company names
    stopwords = prep_stopwords(stop_type = 'nltk')
    mask_comp = lambda text: mask_companies(text, company_names, role_names)
    mask_loc = lambda text: mask_locations(text, location_names, company_names)
    make_tokens = lambda text: process_tokens(text, stopwords)

    # Replace company & location names with mask tokens
    posts['combined_mask'] = posts['combined'].parallel_apply(mask_comp)
    posts['combined_mask'] = posts['combined_mask'].parallel_apply(mask_loc)

    # Tokenize title & text for n-gram models
    posts['combined_tokens'] = posts['combined_mask'].parallel_apply(make_tokens)

    # Export as CSV
    posts = posts.drop(columns=dropcols)
    posts.to_csv(f"{CLEAN}/posts_mask.csv", index=False)

    # (inspecting output)
    posts_test = pd.read_csv(f"{CLEAN}/posts_mask.csv")
    posts_orig = pd.read_csv(f"{CLEAN}/posts.csv", encoding='utf-8')
    posts_basic = pd.read_csv(f"{CLEAN}/posts_basic.csv")
    posts_test = posts_test.merge(posts_orig, on='post_id', how='inner')
    posts_test = posts_test.merge(posts_basic, on='post_id', how='inner')
    posts_test = posts_test[posts_test['referral_request']]
    debug_print(posts_test, task='mask', compute=None, sample='random')
