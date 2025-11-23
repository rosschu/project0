'''
============================================================

Parsing HTML file on Job Referrals Channel

============================================================
'''

# Packages & Directories
from setup.utils import *
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=psutil.cpu_count())

# Parse HTML file to extract posts
def html_to_csv(filepath: str) -> pd.DataFrame:
    '''
    Description:
        - Parse HTML file to extract basic info on posts

    Inputs:
        - filepath: location of HTML file containing posts
    '''

    # Import HTML file as tree
    with open(filepath, 'r', encoding='utf-8') as file:
        html_content = file.read()
        # tree = html.fromstring(html_content)
        tree = etree.fromstring(html_content, etree.HTMLParser())

    # Links, Title, Content
    baselink = "https://www.teamblind.com"
    xp_postid = "//article/div/a/@href"
    xp_title = "//h2[contains(@class, 'pointer-events-none')]"
    xp_content = "//p[contains(@class, 'pointer-events-none')]"
    post_id = [el for el in tree.xpath(xp_postid)]
    post_id = [link.replace(baselink,'').replace("/post/",'') for link in post_id]
    title = [el.text for el in tree.xpath(xp_title)]
    text = [el.text for el in tree.xpath(xp_content)]

    # User name, firm, posting date
    xp_poster = "//div[contains(@class,'h-full items-center text-xs')]"
    poster = tree.xpath(xp_poster)
    user_name = [el.xpath(".//text()")[1] for el in poster]
    user_firm = [el.xpath(".//text()")[0] for el in poster]
    date      = [el.xpath(".//text()")[2] for el in poster]

    # Likes, Comments, Views
    xp_sharing = "//button[contains(@aria-label,'Like this') and not(@disabled)]/parent::div"
    shares = {s: [] for s in ['likes','comments','views']}
    share_els = tree.xpath(xp_sharing)
    assert len(share_els) == len(post_id)

    for el in share_els:

        # If nothing can be extracted, assume zero
        recorder = lambda x: int(x[0]) if len(x) > 0 else 0

        # Extract likes, comments, views
        xp_likes = "button[contains(@aria-label,'Like this') and not(@disabled)]/@data-count" 
        shares['likes'].append(recorder(el.xpath(xp_likes)))

        xp_comments = xp_likes.replace("Like this", "Comment on this")
        shares['comments'].append(recorder(el.xpath(xp_comments)))

        xp_views = xp_likes.replace("Like this", "Views")
        shares['views'].append(recorder(el.xpath(xp_views)))

    # Scrape date
    scrape_date = datetime(
        2024, 11, 17, 12, 0, 0, # 2024-10-22
        tzinfo=pytz.timezone('America/Los_Angeles')
    )
    scrape_date = scrape_date.strftime("%Y-%m-%d")

    # CSV File
    data = {
        "post_id": post_id,
        "user_name": user_name,
        "user_firm": user_firm,
        "date": date,
        "scrape_date": [scrape_date for i in range(len(post_id))],
        "num_likes": shares['likes'],
        "num_comments": shares['comments'],
        "num_views": shares['views'],
        "title": title,
        "post_preview": text,
    }

    return pd.DataFrame(data)

# Prepare data with full text content + hashtags
def fulltext_hashtags(posts: pd.DataFrame, filepath: str) -> pd.DataFrame:
    '''
    Description:
        - Prepare data with full text content + hashtags

    Inputs:
        - posts: dataframe of posts parsed from HTML file
        - filepath: location for CSV file containing full text
    '''

    # Helper routine to separate hash tags from text content    
    def extract_content(txt: str):

        # Split text into tokens (including new lines)
        txt = txt.strip(" \n\t.,!?-:;")
        txt = txt.replace('\n',' \n ').replace('  ', ' ')
        txt = txt.replace('#', ' #').replace('  ', ' ')
        txt = txt.split(' ')

        # Clean up extra punctuation around hashtags
        for i,token in enumerate(txt):
            if token.startswith('#'):
                txt[i] = token.strip(" \n\t.,!?-:;")

        # Find the last token that doesn't start with a hashtag
        idx = None
        for i in reversed(range(len(txt))):

            # Check whether the token contains a hashtag
            punc = " \n\t.,!?-:;"
            token = txt[i].strip(punc)
            if not token.startswith('#'):

                # Confirm that this token isn't surrounded by hashtags
                prev = txt[i-1].strip(punc) if i > 0 else ''
                next = txt[i+1].strip(punc) if i < len(txt)-1 else ''

                # IF so, return the index that marks the beginning of hashtags
                if not (prev.startswith('#') and next.startswith('#')):
                    idx = i
                    break
        
        # If non-hashtag tokens do not exist, set index to -1
        idx = -1 if idx is None else idx
        
        # If index is out of range, it will join an empty list
        content = ' '.join(txt[:idx+1]).replace(' \n', '\n').replace('\n ', '\n').replace('#','')
        hashtags = ' '.join(txt[idx+1:]).replace('\n','').replace('  ', ' ').strip()

        return pd.Series((content, hashtags))

    # Import data on full text content
    # Fill in missing text with previews, and separate text from hashtags
    df = pd.read_csv(filepath, encoding='utf-8')
    df.loc[ df['full_text'].isnull(), 'full_text' ] = posts['post_preview']
    df[['text', 'hashtags']] = df['full_text'].parallel_apply(extract_content)

    # Add full text + hashtag columns to posts
    # Combine title and text into a single column
    posts = posts.merge(df, on='post_id', how='inner')

    posts.loc[ posts['text'].isnull(), 'text' ] = ''
    posts['combined'] = "[TITLE]\n\n" + posts['title'] + "\n\n\n[CONTENT]\n\n" + posts['text']
    assert posts['title'].isnull().sum() == 0
    assert posts['text'].isnull().sum() == 0
    assert posts['combined'].isnull().sum() == 0

    return posts.drop(columns=['full_text'])

# Formatting to dates
def format_dates(posts: pd.DataFrame) -> pd.DataFrame:
    '''
    Description:
        - Convert date column into datetime format by replacing relative dates (e.g. Yesterday, 2d, 2m) with their actual YYYY-MM-DD strings
    
    Inputs: 
        - posts: dataframe of posts containing date column
    '''

    # Helper function to convert relative dates to YYYY-MM-DD strings
    def extract_dates(date_str: str):

        reference_date = datetime(2024, 11, 17)

        # If already a Timestamp, return as is
        if isinstance(date_str, pd.Timestamp):  
            return date_str

        elif pd.isna(date_str):
            return None

        # Handle both spellings of "yesterday"
        elif "Yesterday" in date_str or "Yesteray" in date_str:
            return reference_date - timedelta(days=1)
        
        # "x d" format indicates days ago from October 22
        elif "d" in date_str:
            days_ago = int(date_str.replace("d", "").strip())
            return reference_date - timedelta(days=days_ago)
        
        # "x h" format indicates hours ago, setting them to October 22
        elif "h" in date_str:
            return reference_date
        
        # "x m" (minutes ago) is recent, setting as October 22
        elif "m" in date_str:
            return reference_date
        
        # Directly parse dates in "Mon DD" format (e.g., "Oct 14")
        else:
            try:
                return pd.to_datetime(date_str+', 2024', format='%b %d, %Y')
            except ValueError:
                return None

    # Convert 'date' column to YYYY-MM-DD format
    posts['date'] = posts['date'].parallel_apply(extract_dates)

    return posts

# Unit tests
def unit_tests(posts):

    # No missing values
    assert posts.isnull().sum().sum() == 0

    # String columns that must be non-empty
    strcols = {col for col in posts.columns if 'num_' not in col}
    strcols -= {'text','hashtags'}
    for col in strcols:
        assert (posts[col] == '').sum() == 0

    # Post date precedes scrape date
    assert (posts['date'] <= posts['scrape_date']).all()

if __name__ == "__main__":

    # Parse HTML file into a data frame
    html_file = f"{DATA}/posts_html/referrals_full.html"
    posts_html = html_to_csv(html_file)

    # Merge in full text content + hashtags
    fulltext_file = f"{CLEAN}/posts_fulltext.csv"
    posts = fulltext_hashtags(posts_html, fulltext_file)

    # Format dates
    posts = format_dates(posts)

    # Run unit tests
    unit_tests(posts)

    # Export scraped posts to CSV
    lastcols = ['post_preview','hashtags','title','text','combined']
    firstcols = [col for col in posts.columns if col not in lastcols]
    posts = posts[firstcols + lastcols]
    posts.to_csv(f"{CLEAN}/posts.csv", index=False)

