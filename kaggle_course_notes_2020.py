"""LEARN on kaggle.

The following are notes and code snippets from the following Kaggle courses.

I prefer reading these notes in raw text / script format, as opposed to a
nicely formatted Jupyter notebook, because I can fit more text on a
vertical screen with no graphs or output. The layout is not the cleanest, as
it is for personal use / reference.  I created on Spyder IDE on home
computer, uploading to kaggle and github merely for backup / documentation of
my progress in studies.

COURSES
x - Data Cleaning
x - Intro SQL
x - Advanced SQL
x - Pandas (Data Wrangling)

- Feature Engineering
- Intro Machine Learning
- Intermediate Machine Learning
- Machine Learning Explainability

Lecture notes on linear models
https://online.stat.psu.edu/stat501/lesson/9
"""
# %% DATA CLEANING
# - missing values
# - scaling and normalization
# - Parsing dates
# - Character encoding
# - Inconsistent data entry / Fuzzy matching

# Try replacing all the NaN's in the sf_permits data with the one that comes
# directly after it and then replacing any remaining NaN's with 0.
# Set the result to a new DataFrame sf_permits_with_na_imputed.

# isnull.sum().sum() sums each column na's; product of shape is # cells in df
percent_missing = sf_permits.isnull().sum().sum() \
    / np.product(sf_permits.shape)*100
sf_permits_with_na_imputed = sf_permits.fillna(method='bfill',
                                               axis=0).fillna(0)

def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = \
        mis_val_table.rename(columns={0: 'Missing Values',
                                      1: '% of Total Values'})
    nonzeros = mis_val_table_ren_columns.iloc[:, 1] != 0
    mis_val_table_ren_columns = mis_val_table_ren_columns[nonzeros] \
        .sort_values('% of Total Values', ascending=False) \
        .round(1)
    ncols = str(df.shape[1])
    mis_cols = str(mis_val_table_ren_columns.shape[0])
    print("Your selected dataframe has " + ncols + " columns.\n")
    print("There are " + mis_cols + " columns that have missing values.")
    return mis_val_table_ren_columns

# %% DATA CLEANING - scaling and normalization
# In scaling, you're changing the range of your data, while
# you want to scale data when you're using methods based on measures of
# how far apart data points are, like SVMs or k-nearest neighbors (KNN)
# With these algorithms, a change of "1" in any numeric feature is given the
# same importance.  A gradient descent to find optimal coefficients in a
# linear model can converge more
# quickly for a given learning rate when the data for the features are scaled
# similarly

import pandas as pd
import numpy as np
from scipy import stats  # for Box-Cox Transformation
from mlxtend.preprocessing import minmax_scaling
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(0)

# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size=1000)

# min-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])
fig, ax = plt.subplots(1, 2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")

# distplot combines the matplotlib hist function (with automatic
# calculation of a good default bin size) with the seaborn kdeplot() and
# rugplot() functions. It can also fit scipy.stats distributions and
# plot the estimated PDF over the data. because data is scaled to 0-1 on x
# axis, and distribution needs to still sum to 1, the height of histogram bars
# is multiplied by around 8, which is max value of original data
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")

# NORMALIZATION
# -------------
# In Normalization, you're changing the shape of the distribution of your data.
# to a normal distribution.
# Scaling just changes the range of your data.

# In general, you'll normalize your data if you're going to be using a
# machine learning or statistics technique that assumes your data is normally
# distributed. Some examples of these include linear discriminant analysis
# (LDA) and Gaussian naive Bayes. (Pro tip: any method with "Gaussian" in
# the name probably assumes normality.)

# Normalize the exponential data with Box-Cox Transformation method
# At the core of the Box Cox transformation is an exponent, lambda (λ),
# which varies from -5 to 5. All values of λ are considered and the
# optimal value for your data is selected (based on stats);
# The “optimal value” is the one which results in the best approximation of a
# normal distribution curve.  stats.boxcox() does the transformations and
# statistical tests to return the optimal normalized data
# The BoxCox tries raising the data x to various powers and also takes the log
# x^-5, x^-4..x^-1, log(x), x^1, ...x^5

# Test only works for positive data. However, Box and Cox did propose a second
# formula that can be used for negative y-values
normalized_data = stats.boxcox(original_data)

fig, ax = plt.subplots(1, 2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
# normalized_data[0] is data, [1] is the optimal lambda
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")

# %% DATA CLEANING - Parsing dates

landslides['date_parsed'] = pd.to_datetime(landslides['date'],
                                           format="%m/%d/%y")
# don't always use infer_datetime_format = True because
# 1) pandas won't always been able to figure out the correct date format
# 2) it's much slower than specifying the exact format of the date
landslides['date_parsed'] = pd.to_datetime(landslides['Date'],
                                           infer_datetime_format=True)
day_of_month_landslides = landslides['date_parsed'].dt.day
# plot the day of the month; should be almost uniform for all #s 1-31
sns.distplot(day_of_month_landslides, kde=False, bins=31)

# %% DATA CLEANING - Character encoding
# UTF-8 is the standard text encoding. All Python code is in
# UTF-8 and, ideally, all your data should be as well.
# It's when things aren't in UTF-8 that you run into trouble.

# 2 main data types of data for text in Python 3
# 1) string (UTF-8 by default in Python 3)
# 2) bytes data type, which is a sequence of integers.
# encode - turns string into bytes
# decode - turns bytes into string

import pandas as pd
import numpy as np
import chardet  # special library for guessing encoding
np.random.seed(0)

mystring = "This is the euro symbol: €"
mybytes = mystring.encode("utf-8", errors="replace")
print('mybytes is ', type(mybytes))
print('Decode mybytes back into mystring, specifying original encoding:\n',
      mybytes.decode("utf-8"))

# encode it to a different encoding, replacing characters that raise errors
mybytes_error = mystring.encode("ascii", errors="replace")
# try decode back to utf-8
print('Because we encoded mystring with incorrect type, decoding it will be corrupt')
print('Corrupted:', mybytes_error.decode("ascii"))
print("vs")
print('Original:', mystring)

# We've lost the original underlying byte string! It's been
# replaced with the underlying byte string for the unknown character

# It's far better to convert all our text to UTF-8 as soon as we can and
# keep it in that encoding. The best time to convert non UTF-8 input into
# UTF-8 is when you read in files
# UnicodeDecodeError - error received when tried to decode UTF-8 bytes as if they were
# ASCII, which tells us that this file isn't actually UTF-8.

# use the chardet module to try and automatically guess the right encoding is
# look at subsample of data
# If you have a file that's in UTF-8 but has just a couple of weird-looking
# characters in it, you can try out the ftfy module and see if it helps.
with open("../input/kickstarter-projects/ks-projects-201801.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
print(result)
# read in the file with the encoding detected by chardet
kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv",
                               encoding='Windows-1252')

# %% DATA CLEANING - Inconsistent data entry / Fuzzy matching

import pandas as pd
import numpy as np
import fuzzywuzzy
from fuzzywuzzy import process
import chardet

thefile = "../input/pakistan-intellectual-capital/pakistan_intellectual_capital.csv"
professors = pd.read_csv(thefile)
np.random.seed(0)

countries = professors['Country'].unique()
countries.sort()
countries

# lowercase, strip whitespace before/after
professors['Country'] = professors['Country'].str.lower()
professors['Country'] = professors['Country'].str.strip()

# Fuzzy matching: The process of automatically finding text strings that are
# very similar to the target string. In general, a string is considered closer
# to another one the fewer characters you'd need to change if you were
# transforming one string into another. So "apple" and "snapple" are two
# changes away from each other (add "s" and "n") while "in" and "on" and
# one change away (rplace "i" with "o"). You won't always be able to rely
# on fuzzy matching 100%, but it will usually end up saving you at least a
# little time.

# Fuzzywuzzy returns a ratio given two strings. The closer the ratio is to 100,
# the smaller the edit distance between the two strings.
# Top 10 closest matches to "south korea" from countries
matches = fuzzywuzzy.process.extract("south korea", countries, limit=10,
                                     scorer=fuzzywuzzy.fuzz.token_sort_ratio)
matches


# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the string
def replace_matches_in_column(df, column, string_to_match, min_ratio=47):
    strings = df[column].unique()
    # top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings,
                                         limit=10,
                                         scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    close_matches = [matches[0] for matches in matches
                     if matches[1] >= min_ratio]
    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)
    # replace all rows with close matches with the input matches
    df.loc[rows_with_matches, column] = string_to_match
    print("All done!")

# %% INTRO TO and ADVANCED SQL
# ----------------- ---------------------------------------------------
# - setup of BigQuery
# - analytic functions
# - nested and repeated data
# - efficient queries
# - example queries
# https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

# %% INTRO TO and ADVANCED SQL - setup of BigQuery

from google.cloud import bigquery

client = bigquery.Client()
# Construct a reference to the "hacker_news" dataset
# projects have datasets, datasets have tables
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")

# These tables being very large, we run a query on the table to get a smaller
# subset that we can then put in a dataframe to explore

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)
# List all the tables in the "hacker_news" dataset
tables = list(client.list_tables(dataset))

# Print names of all tables in the dataset (there are four!)
for table in tables:
    print(table.table_id)

# Return a list of names of tables
list_of_tables = []

for table in list(client.list_tables(dataset)):
    list_of_tables.append(table.table_id)

# Construct a reference to the "full" table
table_ref = dataset_ref.table("full")

# API request - fetch the table
table = client.get_table(table_ref)
# Print information on all the columns in the "full" table in the "hacker_news"
# dataset table.schema

# SchemaField:
# SchemaField('by', 'string', 'NULLABLE', "The username of the item's author.",())
# 	the field (or column) is called by,
# 	the data in this field is strings,
# 	NULL values are allowed, and
# 	it contains the usernames corresponding to each item's author.
# Preview the first five lines of the "full" table

client.list_rows(table, max_results=5).to_dataframe()
# Preview the first five entries in the "by" column of the "full" table
client.list_rows(table, selected_fields=table.schema[:1],
                 max_results=5).to_dataframe()

# ----------------- ---------------------------------------------------
# NOTE - argument we pass to FROM is in backticks

table_ref = dataset_ref.table("global_air_quality")
table = client.get_table(table_ref)
client.list_rows(table, max_results=5).to_dataframe()

query = """
        SELECT city
        FROM `bigquery-public-data.openaq.global_air_quality`
            # `< project >.< dataset >.< table >`
        WHERE country = 'US'
        """
client = bigquery.Client()
query_job = client.query(query)
us_cities = query_job.to_dataframe()
us_cities.city.value_counts().head()

query = """
        SELECT score, title
        FROM `bigquery-public-data.hacker_news.full`
        WHERE type = "job"
        """

# Create a QueryJobConfig object to estimate size of query without running it
dry_run_config = bigquery.QueryJobConfig(dry_run=True)
# API request - dry run query to estimate costs
dry_run_query_job = client.query(query, job_config=dry_run_config)
print("Bytes processed: {}".format(dry_run_query_job.total_bytes_processed))

# Only run the query if it's less than 1 GB
ONE_GB = 1000*1000*1000
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_GB)

# Set up the query (will only run if it's less than 1 GB)
safe_query_job = client.query(query, job_config=safe_config)
# API request - try to run the query, and return a pandas DataFrame
job_post_scores = safe_query_job.to_dataframe()
job_post_scores.score.mean()

# %% INTRO TO and ADVANCED SQL - Analytic functions

"""Analytic functions - see Exercises in Example Queries in cell below
All analytic functions have an OVER clause, which defines the sets of rows
used in each calculation. The OVER clause has three (optional) parts:

1) The PARTITION BY clause divides the rows of the table into different groups.
    In the query above, we divide by id so that the calculations are separated
    by runner.
2) The ORDER BY clause defines an ordering within each partition. In the
    sample query, ordering by the date column ensures that earlier training
    sessions appear first.
3)	The final clause (ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) is known as
    a window frame clause. It identifies the set of rows used in each
    calculation. We can refer to this group of rows as a window. (Actually,
    analytic functions are sometimes referred to as analytic window functions
    or simply window functions!)

Examples
- ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
- ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING
- ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING - all rows in
    partition.

Three types of analytic functions
---------------------------------
The example above uses only one of many analytic functions. BigQuery supports
a wide variety of analytic functions, and we'll explore a few here. For a
complete listing, you can take a look at the documentation.

1) Analytic aggregate functions
   AVG() is an aggregate function
 The OVER clause is what ensures that it's treated as an analytic (aggregate)
 function. Aggregate functions take all of the values within the window as
 input and return a single value.
    - MIN() (or MAX()) - Returns the minimum (or maximum) of input values
    - AVG() (or SUM()) - Returns the average (or sum) of input values
    - COUNT() - Returns the number of rows in the input

2) Analytic navigation functions
    Navigation functions assign a value based on the value in a (usually)
    different row than the current row.
    - FIRST_VALUE() (or LAST_VALUE()) - Returns the first (or last) value in
        the input
    - LEAD() (and LAG()) - Returns the value on a subsequent (or preceding) row

3) Analytic numbering functions
    Numbering functions assign integer values to each row based on the ordering.
    - ROW_NUMBER() - Returns the order in which rows appear in the input
    (starting with 1)
    - RANK() - All rows with the same value in the ordering column receive
    the same rank value, where the next row receives a rank value which
    increments by the number of rows with the previous rank value.
"""


# %% ADVANCED SQL - Nested tables and repeated data
"""Nested columns have type STRUCT (or type RECORD)

 To query a column with nested data, we need to identify each field in the
 context of the column that contains it:
    Toy.Name refers to the "Name" field in the "Toy" column, and
    Toy.Type refers to the "Type" field in the "Toy" column.
 UNNEST() to pull data out of repeated data column, because we need additional
 rows for the repeated values to occupy; when NOT repeated, then we simply
 add additional columns with column.field as the source
 - By storing the information in the "device" and "totals" columns as STRUCTs
 (as opposed to separate tables), we avoid expensive JOINs. This increases
 performance and keeps us from having to worry about JOIN keys
 (and which tables have the exact data we need).
"""

# committer column has multiple values for name, so does add additional rows
# for committer.name, but only one additional column

max_commits_query = """
                    SELECT
                        committer.name AS committer_name,
                        COUNT(*) AS num_commits
                    FROM `bigquery-public-data.github_repos.sample_commits`
                    WHERE EXTRACT(YEAR from committer.date) = 2016
                    GROUP BY committer_name
                    ORDER BY num_commits DESC
                    """

pop_lang_query = """
                SELECT
                     l.name AS language_name,
                     COUNT(*) AS num_repos
                FROM `bigquery-public-data.github_repos.languages`,
                     UNNEST(language) AS l
                GROUP BY language_name
                ORDER BY num_repos DESC
                """

# adds extra rows for each l.name, but adds two extra columns, one for l.name
# and one for l.bytes
all_langs_query = """
                 SELECT l.name AS name, l.bytes AS bytes
                 FROM `bigquery-public-data.github_repos.languages`,
                     UNNEST(language) AS l
                 WHERE repo_name = 'polyrabbit/polyglot'
                 ORDER BY bytes DESC
                 """

# %% ADVANCED SQL - Efficient Queries

show_amount_of_data_scanned()  # shows the amount of data the query uses.
show_time_to_run()  # prints how long it takes for the query to execute.
"""Notes
 We will use two functions to compare the efficiency of different queries:
 # these 3 can have 100x impact on time to run or data scanned
 # 1. Only select the columns you want (don't select *)
 # 2. Read less data. (avoid selecting big text fields you
 #   don't need or calculate across fields unnecessarily')
 # 3. Avoid N:N JOINs
 # https://www.oreilly.com/library/view/google-bigquery-the/9781492044451/
 query_to_optimize = 3
 # Why 3: Because data is sent for each costume at each second, this is the
 #   query that is likely to involve the most data (by far). And it will be
 #   run on a recurring basis. So writing this well could pay off on a
 #   recurring basis.
 # Why not 1: This is the second most valuable query to optimize. It will be
 #   run on a recurring basis, and it involves merges, which is commonly a
 #   place where you can make your queries more efficient
 # Why not 2: This sounds like it will be run only one time. So, it probably
 #   doesn’t matter if it takes a few seconds extra or costs a few cents more
 #   to run that one time. Also, it doesn’t involve JOINs. While the data has
 #   text fields (the reviews), that is the data you need. So, you can’t leave
 #   these out of your select query to save computation.

"""

# %% INTRO TO and ADVANCED SQL - EXAMPLE QUERIES

query = """
        SELECT COUNT(consecutive_number) AS num_accidents,
               EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        GROUP BY day_of_week
        ORDER BY num_accidents DESC
        """

# ---------------------------------- ----------------- -----------------
# Intro SQL, AS and WITH, Exercise #3
rides_per_year_query = """
            SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year,
            COUNT(unique_key) AS num_trips
            FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
            GROUP BY year
        """

# Intro SQL, AS and WITH, Exercise #5
speeds_query = """
               WITH RelevantRides AS
               (
                   SELECT
                       trip_start_timestamp,
                       trip_miles,
                       trip_seconds
                   FROM  `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                   WHERE trip_miles>0
                   AND trip_seconds>0
                   AND trip_start_timestamp > '2017-01-01'
                   AND trip_start_timestamp < '2017-07-01'
               )
               SELECT
                    EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day,
                    COUNT(1) AS num_trips,
                    3600 * SUM(trip_miles) / SUM(trip_seconds) AS avg_mph

               FROM RelevantRides
               GROUP BY hour_of_day
               ORDER BY hour_of_day
               """
# ---------------------------------- ----------------- -----------------

query = """
        SELECT *
        FROM `bigquery-public-data.pet_records.pets`
        WHERE Name LIKE '%ipl%'
        """


def expert_finder(topic, client):
    """Return a DF with the user IDs of who's written answers on a topic.

    Inputs:
        topic: A string with the topic of interest
        client: A Client object that specifies the connection to the Stack
        Overflow dataset

    Outputs:
        results: A DataFrame with columns for user_id and number_of_answers.
        Follows similar logic to bigquery_experts_results shown above.
    """
    my_query = """
               SELECT a.owner_user_id AS user_id, COUNT(1) AS number_of_answers
               FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
               INNER JOIN `bigquery-public-data.stackoverflow.posts_answers`
                   AS a
                   ON q.id = a.parent_Id
               WHERE q.tags like '%{topic}%'
               GROUP BY a.owner_user_id
               """

    # Set up the query (a real service would have good error handling for
    # queries that scan too much data)
    safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
    my_query_job = client.query(my_query, job_config=safe_config)

    # API request - run the query, and return a pandas DataFrame
    results = my_query_job.to_dataframe()

    return results


# ---------------------------------- ----------------- -----------------
# Union concatenates vertically, UNION DISTINCT excludes duplicates
all_users_query = """
                SELECT q.owner_user_id
                FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
                WHERE q.creation_date >= '2019-01-01'
                    AND q.creation_date < '2019-01-02'
                UNION DISTINCT
                SELECT a.owner_user_id
                FROM `bigquery-public-data.stackoverflow.posts_answers` AS a
                WHERE a.creation_date >= '2019-01-01'
                    AND a.creation_date < '2019-01-02'
                """

# ---------------------------------- ----------------- -----------------
# COMPARE time to run of big vs. small query - Efficient Query Lesson

star_query = "SELECT * FROM `bigquery-public-data.github_repos.contents`"
# show_amount_of_data_scanned(star_query)

big_join_query = """
                 SELECT repo,
                     COUNT(DISTINCT c.committer.name) as num_committers,
                     COUNT(DISTINCT f.id) AS num_files
                 FROM `bigquery-public-data.github_repos.commits` AS c,
                     UNNEST(c.repo_name) AS repo
                 INNER JOIN `bigquery-public-data.github_repos.files` AS f
                     ON f.repo_name = repo
                 WHERE f.repo_name IN ('tensorflow/tensorflow',
                                       'facebook/react',
                                       'twbs/bootstrap',
                                       'apple/swift',
                                       'Microsoft/vscode',
                                       'torvalds/linux')
                 GROUP BY repo
                 ORDER BY repo
                 """
# show_time_to_run(big_join_query)
small_join_query = """
                   WITH commits AS
                   (SELECT
                        COUNT(DISTINCT committer.name) AS num_committers, repo
                   FROM `bigquery-public-data.github_repos.commits`,
                       UNNEST(repo_name) as repo
                   WHERE repo IN ('tensorflow/tensorflow',
                                       'facebook/react',
                                       'twbs/bootstrap',
                                       'apple/swift',
                                       'Microsoft/vscode',
                                       'torvalds/linux')
                   GROUP BY repo
                   ),
                   files AS
                   (SELECT
                       COUNT(DISTINCT id) AS num_files, repo_name as repo
                   FROM `bigquery-public-data.github_repos.files`
                   WHERE repo_name IN ('tensorflow/tensorflow',
                                       'facebook/react',
                                       'twbs/bootstrap',
                                       'apple/swift',
                                       'Microsoft/vscode',
                                       'torvalds/linux')
                   GROUP BY repo
                   )
                   SELECT commits.repo, commits.num_committers, files.num_files
                   FROM commits
                   INNER JOIN files
                       ON commits.repo = files.repo
                   ORDER BY repo
                   """
# show_time_to_run(small_join_query)

# ---------------------------------- ----------------- -----------------
# ADVANCED SQL, Analytics Functions, Exercise 1
# Note, there is no PARTITION BY here
avg_num_trips_query = """
                      WITH trips_by_day AS
                      (
                      SELECT DATE(trip_start_timestamp) AS trip_date,
                          COUNT(*) as num_trips
                      FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                      WHERE trip_start_timestamp >= '2016-01-01'
                          AND trip_start_timestamp < '2018-01-01'
                      GROUP BY trip_date
                      ORDER BY trip_date
                      )
                      SELECT
                          trip_date,
                          AVG(num_trips)
                              OVER (
                               ORDER BY trip_date
                               ROWS BETWEEN 15 PRECEDING AND 15 FOLLOWING
                               ) AS avg_num_trips
                      FROM trips_by_day
                      """

# ADVANCED SQL, Analytics Functions, Exercise 2 of 3
trip_number_query = """
                    SELECT
                        pickup_community_area,
                        trip_start_timestamp,
                        trip_end_timestamp,
                        RANK()
                            OVER(
                                PARTITION BY pickup_community_area
                                ORDER BY trip_start_timestamp
                                ) AS trip_number
                    FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                    WHERE DATE(trip_start_timestamp) = '2017-05-01'
                    """

# ADVANCED SQL, Analytics Functions, Exercise 3 of 3
break_time_query = """
                   SELECT
                       taxi_id,
                       trip_start_timestamp,
                       trip_end_timestamp,
                       TIMESTAMP_DIFF(
                           trip_start_timestamp,
                           LAG(trip_end_timestamp, 1)
                               OVER (
                                    PARTITION BY taxi_id
                                    ORDER BY trip_start_timestamp),
                           MINUTE) as prev_break
                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                   WHERE DATE(trip_start_timestamp) = '2017-05-01'
                   """

# %% PANDAS - indexing, selection, assigning
"""
Topics
- indexing, selection, assigning
- SUMMARY FUNCTIONS
- GROUPBY, SORT, AGG

- data-types-and-missing-values
- renaming-and-combining
-------------------
Choosing between loc and iloc

iloc uses the Python stdlib indexing scheme, where the first element of the
range is included and the last one excluded.
0:10 will select entries 0,...,9

loc, indexes inclusively. So 0:10 will select entries 0,...,10.

reviews.iloc[0]  # ROW 0
reviews.iloc[[0, 1, 2], 0] # rows 0,1,2, column 0
reviews.iloc[-5:]  # last 5 rows
reviews.loc[0, 'country']  # row 0, column 'country'

# all rows, selected columns
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]

# boolean to select certain rows
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
reviews.loc[reviews.country.isin(['Italy', 'France'])]

# certain rows
sample_reviews = reviews.loc[[1, 2, 3, 5, 8]]

# certain rows, certain columns
df = reviews.loc[0:99, ['country', 'variety']]

# all columns, certain rows
top_oceania_wines = \
    reviews.loc[reviews.country.isin(['Australia', 'New Zealand']) \
                & (reviews.points>=95)]
"""

# %% PANDAS - SUMMARY FUNCTIONS
""" MAP AND APPLY() ARE TWO MAP FUNCTIONS

The function you pass to map() should expect a single value
from the Series (a point value, in the above example),
and return a transformed version of that value.

Note that map() and apply() return new, transformed Series and DataFrames,
respectively. They don't modify the original data they're called on.

review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)

def remean_points(row):
    row.points = row.points - review_points_mean
    return row
# function is applied to each row of reviews dataframe, but the function
# addresses .points column specifically, so rest of row is unchanged from
# original

reviews.apply(remean_points, axis='columns')

def remean_points2(row):
    # this will create error, because onecolumn has Strings not integers
    row = row - 2
    return row

If we had called reviews.apply() with axis='index', then instead of
passing a function to transform each row, we would need to give a function to
transform each column.

Faster way with pandas
review_points_mean = reviews.points.mean()
reviews.points = reviews.points - review_points_mean

These operators are faster than map() or apply() because they uses speed
ups built into pandas. All of the standard Python operators (>, <, ==, and
so on) work in this manner.

EXERCISES
---------

bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']

ntrop = reviews.description.map(lambda x: 'tropical' in x).sum()
nfruit = reviews.description.map(lambda x: 'fruity' in x).sum()
descriptor_counts = pd.Series([ntrop, nfruit], index=['tropical', 'fruity'])
descriptor_counts
"""

# %% PANDAS - GROUPBY, SORT, AGG
"""
    We can replicate what value_counts() does by doing the following:

reviews.groupby('points').points.count()

# first row in title for each winery
reviews.groupby('winery').apply(lambda df: df.title.iloc[0])

# number of wines, min and max price
reviews.groupby(['country']).price.agg([len, min, max])

    # put mosthighly rated wineby country/province
reviews.groupby(['country', 'province']).apply(lambda df: \
                                               df.loc[df.points.idxmax()])

    # number of countries reviewed by country and province
    # results in multiindex index; can be undone with reset_index()
countries_reviewed = \
    reviews.groupby(['country', 'province']).description.agg([len])

countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len')
countries_reviewed.sort_index()
countries_reviewed.sort_values(by=['country', 'len'])

EXERCISES:
----------
 - What is the best wine I can buy for a given amount of money? Create a Series
 whose index is wine prices and whose values is the maximum number of points a
 wine costing that much was given in a review. Sort the values by price,
 ascending (so that 4.0 dollars is at the top and 3300.0 dollars is at the
           bottom).
best_rating_per_price = reviews.groupby('price').points.apply(lambda s: \
                          s[s.idxmax()]).sort_index(ascending=True)

- What are the minimum and maximum prices for each `variety` of wine? Create a
 `DataFrame` whose index is the `variety` category from the dataset and whose
 values are the `min` and `max` values thereof.
price_extremes = reviews.groupby('variety').price.agg([min, max])

- What are the most expensive wine varieties? Make variable sorted_varieties
 containing a copy of the dataframe from the previous question where varieties
 are sorted in descending order based on minimum price, then on maximum price
 (to break ties).
sorted_varieties = price_extremes.copy().sort_values(by=['min', 'max'],
                                                     ascending=False)

5. Create a Series whose index is reviewers and whose values is the average
    review score given out by that reviewer. Hint: you will need the
    taster_name and points columns.
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()

- What combination of countries and varieties are most common?
    Create a Series whose index is a MultiIndexof {country, variety} pairs.
    For example, a pinot noir produced in the US should map to
    {"US", "Pinot Noir"}.
    Sort the values in the Series in descending order based on wine count.

country_variety_counts = \
    reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)


"""

# %% PANDAS - data-types-and-missing-values
"""
    # columns consisting entirely of strings do not get their own type;
    # they are instead given the object type.

reviews.price.dtype
reviews.dtypes  # id's type of each column

EXERCISE 2
reviews.points.astype('float64')  # recasts type

reviews.region_2.fillna("Unknown")
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")
point_strings = reviews.points.astype('str')

EXERCISE 3
n_missing_prices = reviews.price.isnull().sum()

EXERCISE 4
# don't need a groupby because value_counts does this aggregation; length of
# result is not length of reviews.region_1, but is the # of different regions
reviews_per_region = \
    reviews.region_1.fillna('Unknown').value_counts().sort_values(ascending=False)

"""

# %% PANDAS - renaming-and-combining
"""
reviews.rename(columns={'points': 'score'})
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})

    Both the row index and the column index can have their own name attribute.
    The complimentary rename_axis() method may be used to change these names.
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')

The lsuffix and rsuffix parameters are necessary here because the data
 has the same column names in both British and Canadian datasets. If this
 wasn't true (because, say, we'd renamed them beforehand) we wouldn't need
    them.

    . In order of increasing complexity, these are concat(), join(),
    and merge(). Most of what merge() can do can also be done more simply
    with join(), so we will omit it and focus on the first two functions here.

The simplest combining method is concat(). Given a list of elements, this
function will smush those elements together along an axis.

This is useful when we have data in different DataFrame or Series objects
but having the same fields (columns). One example: the YouTube Videos dataset,
 which splits the data up based on country of origin (e.g. Canada and the UK,
                                                      in this example).
 If we want to study multiple countries simultaneously, we can use concat() t
 o smush them together:
pd.concat([canadian_youtube, british_youtube])

left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])
left.join(right, lsuffix='_CAN', rsuffix='_UK')

EXERCISES
renamed = reviews.copy().rename(columns={'region_1': 'region',
                                         'region_2': 'locale'})

reindexed = reviews.rename_axis('wines', axis='rows')
powerlifting_combined = powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))
"""

# %% FEATURING ENGINEERING
"""
Outline
- MAKING FEATURES READY FOR MODELING with LabelEncoder() or pd.factorize(),
    OneHotEncoder() or pd.get_dummies()
- Baseline model
- Categorical Encodings
- Feature Generation
    > timeseries and transforms
- Feature Selection
"""

# %% FEATURING ENGINEERING - MAKING FEATURES READY FOR MODELING
"""
https://medium.com/@vaibhavshukla182/want-to-know-the-diff-among-pd-factorize-a8591eb3347d

encode labels into categorical variables
                    result will have 1 dimension
                    good for column with 2 categories/factors
                    with more than 2, scaling could be an issue
from sklearn import preprocessing
df = pd.DataFrame(['A', 'B', 'B', 'C'], columns=['Col'])
df['Fact'] = pd.factorize(df['Col'])[0]
le = preprocessing.LabelEncoder()
df['Lab'] = le.fit_transform(df['Col'])
print(df)
#   Col  Fact  Lab
# 0   A     0    0
# 1   B     1    1
# 2   B     1    1
# 3   C     2    2

encode categorical variable into dummy/indicator (binary) variables

good for >2 categories, since then each category gets its own column where
1=yes, and 0 = no
can result in feature explosion, as 4 categories results in 4 columns
OneHotEncoder can only be used with categorical integers while get_dummies can
be used with other type of variables.

df = pd.DataFrame(['A', 'B', 'B', 'C'], columns=['Col'])
df = pd.get_dummies(df)

print(df)
#    Col_A  Col_B  Col_C
# 0    1.0    0.0    0.0
# 1    0.0    1.0    0.0
# 2    0.0    1.0    0.0
# 3    0.0    0.0    1.0

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
df = pd.DataFrame(['A', 'B', 'B', 'C'], columns=['Col'])
# We need to transform first character into integer in order to use the
OneHotEncoder
le = preprocessing.LabelEncoder()
df['Col'] = le.fit_transform(df['Col'])
enc = OneHotEncoder()
df = pd.DataFrame(enc.fit_transform(df).toarray())

print(df)
#      0    1    2
# 0  1.0  0.0  0.0
# 1  0.0  1.0  0.0
# 2  0.0  1.0  0.0
# 3  0.0  0.0  1.0
"""

# %% FEATURING ENGINEERING - Baseline model
"""
Steps to creating baseline model
1. load data
2. prepare target column
3. convert timestamps
4. prep categorical variables

5. create training, validation, and test splits
6. train a model
7. make predictions and evaluate model

Separating timestamp into separate columns
Exercise:
clicks['day'] = clicks['click_time'].dt.day.astype('uint8')
# Fill in the rest
clicks['hour'] = clicks['click_time'].dt.hour.astype('uint8')
clicks['minute'] = clicks['click_time'].dt.minute.astype('uint8')
clicks['second'] = clicks['click_time'].dt.second.astype('uint8')

from sklearn import preprocessing

cat_features = ['ip', 'app', 'device', 'os', 'channel']

# Create new columns in clicks using preprocessing.LabelEncoder()
for feature in cat_features:
    le = preprocessing.LabelEncoder()
    clicks[feature + '_labels'] = le.fit_transform(clicks[feature])

3. Does onehotcoding make sense?
[len(clicks[x].unique()) for x in cat_features]
[260238, 389, 1918, 232, 181]

Solution: NO. The ip column has 260238 unique values, which means itll create
extremely sparse matrix with that many columns (NOTE - answer says 58,000,
but I think data has been updated with more differentvalues, hence the 260K
is the number returned from the data when I ran it). This many columns will
make your model run very slow, so in general you want to avoid one-hot
encoding features with many levels. LightGBM models work with label encoded
features, so you don't actually need to one-hot encode the categorical
features.

4. Timeseries considerations: Solution: Since our model is meant to
predict events in the future, we must also validate the model on events
in the future. If the data is mixed up between the training and test sets,
then future data will leak in to the model and our validation results will
overestimate the performance on new data.
The first 80% of the rows are the train set, the next 10% are the validation
set, and the last 10% are the test set.

5.
---
feature_cols = ['day', 'hour', 'minute', 'second',
                'ip_labels', 'app_labels', 'device_labels',
                'os_labels', 'channel_labels']
---
valid_fraction = 0.1
clicks_srt = clicks.sort_values('click_time')
valid_rows = int(len(clicks_srt) * valid_fraction)
train = clicks_srt[:-valid_rows * 2]
# valid size == test size, last two sections of the data
valid = clicks_srt[-valid_rows * 2:-valid_rows]
test = clicks_srt[-valid_rows:]

---
import lightgbm as lgb

dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])
dtest = lgb.Dataset(test[feature_cols], label=test['is_attributed'])

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10)

---
from sklearn import metrics

ypred = bst.predict(test[feature_cols])
score = metrics.roc_auc_score(test['is_attributed'], ypred)
print(f"Test score: {score}")

Test score: 0.9726727334566094

------------------
Reference Code
------------------
"""
# Intro to Machine Learning - EXAMPLE CODE - Baseline Model
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ks = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv',
                 parse_dates=['deadline', 'launched'])

# Drop live projects
ks = ks.query('state != "live"')

# Add outcome column, "successful" == 1, others are 0
ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))

# Timestamp features; adds four new columns with separate timestamp values
ks = ks.assign(hour=ks.launched.dt.hour,
               day=ks.launched.dt.day,
               month=ks.launched.dt.month,
               year=ks.launched.dt.year)

cat_features = ['category', 'currency', 'country']
encoder = LabelEncoder()
encoded = ks[cat_features].apply(encoder.fit_transform)

data_cols = ['goal', 'hour', 'day', 'month', 'year', 'outcome']
data = ks[data_cols].join(encoded)
# so data should have 6 + 3 columns, or total of 9; encoded has same index as
# ks, so the join happens automatically

# %% FEATURING ENGINEERING - Categorical encodings
"""
    basic encodings - one-hot, labelencoding
    more advanced encodings - count encoding, target encoding,
            CatBoost encoding.

        a. count encoding - Count encoding replaces each categorical value
        with the number of times it appears in the dataset. For example,
        if the value "GB" occured 10 times in the country feature,
        then each "GB" would be replaced with the number 10.
        the categorical-encodings package to get this encoding.
        The encoder itself is available as CountEncoder.

import category_encoders as ce
cat_features = ['category', 'currency', 'country']

count_enc = ce.CountEncoder()

# Transform the features, rename the columns with _count suffix, and join data
# NOTE - this encoded uses all data, so there is data leakage. Ideally, would
# fit using training data only, and only then transform the features in the
# validation and test datasets
count_encoded = count_enc.fit_transform(ks[cat_features])

# the original features are included in data; we need to manually remove them
# if we don't want them in dataframe
data = data.join(count_encoded.add_suffix("_count"))
train, valid, test = get_data_splits(data)
train_model(train, valid)
-----  ----------  ----------  -----

        b. target encoding replaces a categorical value with the average
        value of the target (label) for that value of the feature.
        For example, given the country value "CA", you'd calculate the average
        outcome for all the rows with country == 'CA', around 0.28.
        ex: in 100 occurrences of CA,if 28 targets are 1 and 72 are 0,
        28/100 = 0.28, so CA is not a strong indicator that target/label
        should be 1.
        On the other hand, if all CA's had target of 1, then 1.0 or 100% would
        be the encoding.

        This is often blended with the target probability over the entire
        dataset to reduce the variance of values with few occurences.

        This technique uses the targets to create new features.
        So including the validation or test data in the target encodings
        would be a form of target leakage. To avoid leakage, you should learn
        the target encodings from the training dataset only and apply
        it to the other datasets.

        category_encoders.TargetEncoder

target_enc = ce.TargetEncoder(cols=cat_features)
target_enc.fit(train[cat_features], train['outcome'])

train_TE = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))
valid_TE = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))
# we are training on original features and their encoded versions
train_model(train_TE, valid_TE)
-----  -----   -----  ----------  ----------  -----

        c. CatBoost encoding
        This is similar to target encoding in that it's based on the target
        probablity for a given value. However with CatBoost, for each row,
        the target probability is calculated only from the rows before it.
        This avoids leakage of information from future events.

EXAMPLE
target_enc = ce.CatBoostEncoder(cols=cat_features)
target_enc.fit(train[cat_features], train['outcome'])
train_CBE = train.join(target_enc.transform(train[cat_features]).add_suffix('_cb'))
valid_CBE = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_cb'))
train_model(train_CBE, valid_CBE)

----- ---------- ---------- ----------
EXERCISES:
----- ---------- ---------- ----------
Baseline model
Validation AUC score: 0.9622743228943659

-----  ----------  ----------  ----------  -----
1. Solution: You should calculate the encodings from the training set only.
If you include data from the validation and test sets into the encodings,
you'll overestimate the model's performance. You should in general be
vigilant to avoid leakage, that is, including any information from the
validation and test sets into the model. For a review on this topic, see
our lesson on data leakage
-----  ----------  ----------  ----------  -----
2.
# Create the count encoder
count_enc = ce.CountEncoder(cols=cat_features)

# Learn encoding from the training set; i.e,. no counting occurrences in
# validation set, so if one category has a big count in training data,
# it is likely that other categories will have smaller count and could
# be 'grouped together' ie., any category with count of 1 would look like
# all other features with count of 1 in the count_encoded column

count_enc.fit(train[cat_features])

# note that we separate the fit and transform into two steps this time,so that
# the fit is done on train data only (avoid leakage); transform applied to the
# data as needed; fit and transforming in one step wouldn't allow for encoding
# to be done to valid data

# Apply encoding to the train and validation sets as new columns
# Make sure to add `_count` as a suffix to the new columns
# join simply adds new columns; does not get rid of old columns

train_encoded = train.join(count_enc.transform(train[cat_features]).add_suffix("_count"))
valid_encoded = valid.join(count_enc.transform(valid[cat_features]).add_suffix("_count"))

print(train_encoded.head())
print(valid_encoded.head())

# Train the model on the encoded datasets
# This can take around 30 seconds to complete
# WE PASS FULL encoded training and valid sets, so they have count and original
# columns being considered
_ = train_model(train_encoded, valid_encoded)

-----  ----------  ----------  -----
Validation AUC score: 0.9653051135205329

3. Solution: Rare values tend to have similar counts (with values like 1 or 2),
so you can classify rare values together at prediction time.
Common values with large counts are unlikely to have the same exact
count as other values. So, the common/important values get their own grouping
and therefore have higher predictive impact on model.

-----  ----------  ----------  -----
4) Target encoding

Here you'll try some supervised encodings that use the labels (the targets)
to transform categorical features. The first one is target encoding.
    Create the target encoder from the category_encoders library.
    Then, learn the encodings from the training dataset, apply the encodings
    to all the datasets, and retrain the model.

ANSWER:

target_enc = ce.TargetEncoder(cols=cat_features)
target_enc.fit(train[cat_features], train['is_attributed'])
train_encoded = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))
valid_encoded = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))

_ = train_model(train_encoded, valid_encoded)

Validation AUC score: 0.9540530347873288 worse because IP data isn't very
predictive and adds noise to the model

using cat_features2 = ['app', 'device', 'os', 'channel'] # no IP
Validation AUC score: 0.9627457957514338, above baseline but below count
encoding

-----  ----------  ----------  -----
Solution: Target encoding attempts to measure the population mean of the
target for each level in a categorical feature. This means when there is less
data per level, the estimated mean will be further away from the "true"
mean, there will be more variance. There is little data per IP address so
it's likely that the estimates are much noisier than for the other features.
The model will rely heavily on this feature since it is extremely predictive.
This causes it to make fewer splits on other features, and those features are
fit on just the errors left over accounting for IP address.
So, the model will perform very poorly when seeing new IP addresses that
weren't in the training data (which is likely most new data). Going
forward, we'll leave out the IP feature when trying different encodings.

6) CatBoost Encoding
# Remove IP from the encoded features, as explained above
cat_features = ['app', 'device', 'os', 'channel']
cb_enc = ce.CatBoostEncoder(cols=cat_features, random_state=7)
# Learn encoding from the training set ONLY, otherwise there will be leakage
cb_enc.fit(train[cat_features], train['is_attributed'])
train_encoded = train.join(cb_enc.transform(train[cat_features]).add_suffix('_cb'))
valid_encoded = valid.join(cb_enc.transform(valid[cat_features]).add_suffix('_cb'))

"""
# %% FEATURING ENGINEERING - Feature generation
"""
Interactions

One of the easiest ways to create new features is by combining categorical
variables. For example, if one record has the country "CA" and category
"Music", you can create a new value "CA_Music". This is a new categorical
feature that can provide information about correlations between categorical
variables. This type of feature is typically called an interaction.

In general, you would build interaction features from all pairs of
categorical features. You can make interactions from three or more features
as well, but you'll tend to get diminishing returns.

Pandas lets us simply add string columns together like normal Python strings.
Then, we can label encode the interaction feature and add it to our data.

interactions = ks['category'] + "_" + ks['country']
label_enc = LabelEncoder()
data_interaction = baseline_data.assign(category_country=label_enc.fit_transform(interactions))
data_interaction.head()

So, label_enc turns the combined strings into their own set of numbers or
categories, as expected. (note - scaling considerations of the interactions

    goal    hourday month year 	outcome category currency 	country category_country
0 	1000.0 	12 	11 	8 	2015 	0 		108 	5 			9 		1900
1 	30000.0 4 	2 	9 	2017 	0 		93 		13 			22 		1630
2 	45000.0 0 	12 	1 	2013 	0 		93 		13 			22 		1630
3 	5000.0 	3 	17 	3 	2012 	0 		90 		13 			22 		1595
4 	19500.0 8 	4 	7 	2015 	0 		55 		13 			22	 	979

Windows / Time differences/ windows (see section below)

Transform
A handy method for performing operations within groups is to use
.groupby then .transform.
The .transform method takes a function then passes a series or dataframe to
that function for each group. This returns a dataframe with the same
indices as the original dataframe. In our case, we'll perform a groupby
on "category" and use transform to calculate the time differences for
each category.

Transforming numerical features (normalization)
The distribution of the values in "goal" shows that most projects have goals
less than 5000 USD. However, there is a long tail of goals going up to
$100,000 (RIGHT SKEW). Some models (Neural networksn) work
better when the features are normally distributed, so it might help to
transform the goal values. Common choices for this are the square root and
natural logarithm. BoxCox transform tries different powers of the variable
and the log, in range -5,5, and returns transformed data that is the most
nromal among all the choices.
These transformations can also help constrain outliers.

NOTE - The log transformation won't help tree-based models, which are scale
    invariant

*** sqrt, or log fixes right skew / outliers
plt.hist(np.sqrt(ks.goal), range=(0, 400), bins=50);
plt.title('Sqrt(Goal)');

plt.hist(np.log(ks.goal), range=(0, 25), bins=50);
plt.title('Log(Goal)');

Other transformations include squares and other powers,
exponentials, etc. These might help the model discriminate,
like the kernel trick for SVMs.
Again, it takes a bit of experimentation to see what works.
One method is to create a bunch of new features and later choose the best
ones with feature selection algorithms (boxcox essentially does this, trying
                                        a number of different powers)
EXERCISES
1. Create interaction features

import itertools

cat_features = ['ip', 'app', 'device', 'os', 'channel']
interactions = pd.DataFrame(index=clicks.index)

# Iterate through each pair of features, combine them into interaction features
inter_features =[]
for i,j in itertools.combinations(cat_features, 2):
    new_col = i + '_' + j
    inter_features.append(new_col)  # used in debugging
    new_values = clicks[i].map(str) + '_' + clicks[j].map(str)
    encoder = preprocessing.LabelEncoder()
    interactions[new_col] = encoder.fit_transform(new_values)

NOTE
From output below, we can see that the interaction features are even more
diverse in terms of unique values than the original categorical features.
Thus, though problem does not address how to resolve, this could cause a
scaling problem, if we chose a non-tree based model where scaling of features
would be important. For tree models, scaling isn't an issue.

clicks.columns
[len(clicks[x].unique()) for x in cat_features]
inter_features
[len(interactions[x].unique()) for x in inter_features]

> Index(['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time',
       'is_attributed', 'day', 'hour', 'minute', 'second'],
      dtype='object')
> [260238, 389, 1918, 232, 181]

> Index(['ip_app', 'ip_device', 'ip_os', 'ip_channel', 'app_device', 'app_os',
> [878418, 335047, 885666, 1274426, 5209, 5918, 884, 2734, 4941, 9369]
       'app_channel', 'device_os', 'device_channel', 'os_channel'],
      dtype='object')

2. Number of events in the past six hours, not counting current event
def count_past_events(series):
    series2 = pd.Series(series.index, index=series.values)
    return series2.rolling('6h').count()-1 # exclude current event from count

3. Features should NOT be based on future information, because that would cause
data leakage; should make predictive model using same information that one
would have when trying to make prediction

EX 4;
def time_diff(series):
    # Returns a series with the time since the last timestamp in
    # seconds; each entry is difference between entry and previous entry; first value in returned series should be
    # entry is NaN; on dataset provided on site, there were few ips with
    # multiple time stamps, so the output is mostly NaN (since there is no
    second timestamp to take the difference)

    # for some reason,this causes an error if I use dt.total_seconds()
    # and try to pass time_diff to the statement below when I tried creating
    # my own dummy
    # timedeltas = clicks.groupby('ip')['click_time'].transform(time_diff)
    return series.diff().dt.total_seconds()


EX 5
    def previous_attributions(series):
        # Subtracting raw values so I don't count the current event
        sums = series.expanding(min_periods=2).sum() - series
        return sums
OR
    def previous_attributions(series):
        # Subtracting raw values so I don't count the current event
        x = series.cumsum() - series
        x[0]=np.NaN
        return x

for i in len(x.clicks):
    x.clicks[i]=datetime.strptime(x.clicks[i], '%m/%d/%Y %H:%M')

EX 6.
The features themselves will work for either model. However, numerical
inputs to neural networks need to be standardized first. That is, the
features need to be scaled such that they have 0 mean and a standard
deviation of 1. This can be done using sklearn.preprocessing.StandardScaler.
"""

# %% FEATURING ENGINEERING - Feature Generation, timeseries / transforms
import pandas as pd
from datetime import datetime

data = pd.read_csv('test.csv')

print(data, '\n\ndata type in clicks column BEFORE', type(data.clicks.iloc[0]))

for i in range(len(data)):
    data.loc[i, 'clicks'] = datetime.strptime(data.loc[i, 'clicks'],
                                              '%m/%d/%Y %H:%M')
print(data, '\n\ndata type in clicks column AFTER', type(data.clicks.iloc[0]))
print('\ndata type of clicks column', type(data.clicks))

# %% time series with windows
# Implement a function count_past_events that takes a Series of click times
# (timestamps) and returns another Series with the number of events in the
# last six hours. Tip: The rolling method is useful for this.

# First, create a Series with a timestamp index; note, the index will have
# the same name as in data, i.e., 'clicks', while the values of the series
# will not have a name above it
mytseries = pd.Series(data.index, index=data.clicks,
                      name="count_time").sort_index()
# sort_index() does not do it in place,i.e., does not chanage passed data or
# index
print(mytseries.head())

# %%
# exclude current hour
count_36_hours = mytseries.rolling('36h').count() - 1
print(count_36_hours)
print(count_36_hours.values)  # the number of events in last 36 hours
print(count_36_hours.index)   # the time stamps
print(count_36_hours.shape)   # (21,)
# %%
# count_36_hours windows is indexed by timestamps, as is mytseries, so
# replacing the time stamps with the values of mytseries, we get the original
# index back which can be used to find the categorical variables

count_36_hours.index = mytseries.values
print(count_36_hours)
# %%
x = count_36_hours.sort_index()
#  this does same as sorting by the index
y = count_36_hours.reindex(data.index)
# x and y are equivalent


def count_past_events(series):
    """count_past_events."""
    series2 = pd.Series(series.index, index=series.values)
    return series2.rolling('6h').count()-1

# %% timediff transform by group


def time_diff(series):
    """Return a series with the time since the last timestamp.

    in seconds; each entry is difference between entry and previous entry;
    first value in returned series should be #nan.
    """
    return series.diff()


def time_diff2(series):
    return series.diff().dt.total_seconds()


x = data.groupby('cat')['clicks'].transform(time_diff)
print(x)
print(x.dt.total_seconds())

y = data.groupby('cat')['clicks'].transform(time_diff2)
print(y)

# %% FEATURING ENGINEERING - Feature Selection
"""
2 problems with too many features
1.  overfitting
2. it takes longer it will take to train your model and optimize
hyperparameters

simplest and fastest methods of feature selection are based on
univariate statistical tests. For each feature, measure how strongly the
target depends on the feature using a statistical test like χ2 or ANOVA.

sklearn.feature_selection.SelectKBest returns K best features given some
scoring function.
 -. e.g., For classification, the module provides 3:
1. χ2,
2. ANOVA F-value - measures the linear dependency between the feature
variable and the target. This means the score might underestimate the relation
between a feature and the target if the relationship is nonlinear.

3. mutual information score - is nonparametric and so can capture
nonlinear relationships.

With SelectKBest, we define the number of features to keep, based on the score
from the scoring function.
Using .fit_transform(features, target) we get back an array with only the
selected features.

f_classif - Compute the ANOVA F-value for the provided sample.

from sklearn.feature_selection import SelectKBest, f_classif

feature_cols = baseline_data.columns.drop('outcome')
selector = SelectKBest(f_classif, k=5)  # Keep 5 features
X_new = selector.fit_transform(baseline_data[feature_cols],
                               baseline_data['outcome'])
X_new

# BUT WE SHOULD LIMIT TO JUST TRAINING DATA TO AVOID LEAKAGE FROM VALIDATION
# AND TEST DATA

feature_cols = baseline_data.columns.drop('outcome')
train, valid, _ = get_data_splits(baseline_data)
selector = SelectKBest(f_classif, k=5)
X_new = selector.fit_transform(train[feature_cols], train['outcome'])
X_new

to figure out which columns in the dataset were kept with SelectKBest,
we can use .inverse_transform to get back an array with the
shape of the original data.

# Get back the features we've kept, zero out all other features
selected_features = pd.DataFrame(selector.inverse_transform(X_new),
                                 index=train.index,
                                 columns=feature_cols)
selected_features.head()

This returns a DataFrame with the same index and columns as the training set,
 but all the dropped columns are filled with zeros. We can find the selected
 columns by choosing features where the variance is non-zero.

selected_columns = selected_features.columns[selected_features.var() != 0]

# Get the valid dataset with the selected features.
valid[selected_columns].head()

---------- ---------- ----------
L1 regularization- we can make our selection using all of the features by
including them in a linear model with L1 regularization
---------- ---------- ----------
l1 penalizes the absolute magnitude of the coefficients, as compared to L2
(Ridge) regression which penalizes the square of the coefficients.

As the strength of regularization is increased, features which are less
important for predicting the target are set to 0. This allows us to perform
feature selection by adjusting the regularization parameter. We choose the
parameter by finding the best performance on a hold-out set, or decide
ahead of time how many features to keep.


For regression problems you can use
sklearn.linear_model.Lasso, or
sklearn.linear_model.LogisticRegression for classification.

These can be used along with sklearn.feature_selection.SelectFromModel
to select the non-zero coefficients. Otherwise, the code is similar to
the univariate tests.

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

train, valid, _ = get_data_splits(baseline_data)

X, y = train[train.columns.drop("outcome")], train['outcome']

# Set the regularization parameter C=1
logistic = LogisticRegression(C=1, penalty="l1", solver='liblinear',
                              random_state=7).fit(X, y)
model = SelectFromModel(logistic, prefit=True)

X_new = model.transform(X)
X_new

# Get back the kept features as a DataFrame with dropped columns as all 0s
selected_features = pd.DataFrame(model.inverse_transform(X_new),
                                 index=X.index,
                                 columns=X.columns)

# Dropped columns have values of all 0s, keep other columns
selected_columns = selected_features.columns[selected_features.var() != 0]

In general, feature selection with L1 regularization is more powerful
THAN univariate tests, but it can also be VERY SLOW when you have a lot
of data and a lot of features.

Univariate tests will be much faster on large datasets, but also will likely
perform worse.

---------- ---------- ----------
EXERCISES
---------- ---------- ----------
2.
from sklearn.feature_selection import SelectKBest, f_classif
feature_cols = clicks.columns.drop(['click_time', 'attributed_time', 'is_attributed'])
train, valid, test = get_data_splits(clicks)

selector = SelectKBest(f_classif, k=40)  # keep 40 features
X_new = selector.fit_transform(train[feature_cols], train['is_attributed'])
selected_features = pd.DataFrame(selector.inverse_transform(X_new), index=train.index,
                                 columns=feature_cols)
dropped_columns = selected_features.columns[selected_features.var()==0]
_ = train_model(train.drop(dropped_columns, axis=1),
                valid.drop(dropped_columns, axis=1))

3. Solution: To find the best value of K, you can fit multiple models with
increasing values of K, then choose the smallest K with validation score
 above some threshold or some other criteria. A good way to do this is loop
 over values of K and record the validation scores for each iteration.

4) Use L1 regularization for feature selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

def select_features_l1(X, y):
    #Return selected features using logistic regression with an L1 penalty.
    logistic = LogisticRegression(C=0.1, penalty="l1", solver='liblinear',
                              random_state=7).fit(X, y)
    model = SelectFromModel(logistic, prefit=True)
    X_new = model.transform(X)
    # Get back the kept features as a DataFrame with dropped columns as all 0s
    selected_features = pd.DataFrame(model.inverse_transform(X_new), index=X.index,
                                     columns=X.columns)
    # Dropped columns have values of all 0s, keep other columns
    selected_columns = selected_features.columns[selected_features.var() != 0]

    return selected_columns

5)Solution: Instead of using logisticregression or l1 to choose features,
    You could use something like RandomForestClassifier or
    ExtraTreesClassifier to find feature importances.
    SelectFromModel can use the feature importances to find the best features.

6) Solution: To select a certain number of features with L1 regularization,
   you need to find the regularization parameter that leaves the desired
   number of features. To do this you can iterate over models with different
   regularization parameters from low to high and choose the one that leaves
   K features. Note that for the scikit-learn models C is the inverse of the
   regularization strength. i.e., C=0.1 is more potent than c=1


----------- ---------------------------------
Reference code that might be useful
----------- ---------------------------------
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
import lightgbm as lgb

import os

clicks = pd.read_parquet('../input/feature-engineering-data/baseline_data.pqt')
data_files = ['count_encodings.pqt',
              'catboost_encodings.pqt',
              'interactions.pqt',
              'past_6hr_events.pqt',
              'downloads.pqt',
              'time_deltas.pqt',
              'svd_encodings.pqt']
data_root = '../input/feature-engineering-data'
for file in data_files:
    features = pd.read_parquet(os.path.join(data_root, file))
    clicks = clicks.join(features)

def get_data_splits(dataframe, valid_fraction=0.1):

    dataframe = dataframe.sort_values('click_time')
    valid_rows = int(len(dataframe) * valid_fraction)
    train = dataframe[:-valid_rows * 2]
    # valid size == test size, last two sections of the data
    valid = dataframe[-valid_rows * 2:-valid_rows]
    test = dataframe[-valid_rows:]

    return train, valid, test

def train_model(train, valid, test=None, feature_cols=None):
    if feature_cols is None:
        feature_cols = train.columns.drop(['click_time', 'attributed_time',
                                           'is_attributed'])
    dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])

    param = {'num_leaves': 64, 'objective': 'binary',
             'metric': 'auc', 'seed': 7}
    num_round = 1000
    print("Training model!")
    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid],
                    early_stopping_rounds=20, verbose_eval=False)

    valid_pred = bst.predict(valid[feature_cols])
    valid_score = metrics.roc_auc_score(valid['is_attributed'], valid_pred)
    print(f"Validation AUC score: {valid_score}")

    if test is not None:
        test_pred = bst.predict(test[feature_cols])
        test_score = metrics.roc_auc_score(test['is_attributed'], test_pred)
        return bst, valid_score, test_score
    else:
        return bst, valid_score

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

def select_features_l1(X, y):
    # Return selected features using logistic regression with an L1 penalty.
    logistic = LogisticRegression(C=0.1, penalty="l1", solver='liblinear',
                              random_state=7).fit(X, y)
    model = SelectFromModel(logistic, prefit=True)
    X_new = model.transform(X)
    # Get back the kept features as a DataFrame with dropped columns as all 0s
    selected_features = pd.DataFrame(model.inverse_transform(X_new), index=X.index,
                                     columns=X.columns)
    # Dropped columns have values of all 0s, keep other columns
    selected_columns = selected_features.columns[selected_features.var() != 0]

    return selected_columns

# Check your answer

"""


# %% INTRO TO MACHINE LEARNING -
"""
Outline
- Underfitting / Overfitting
- Random Forests
- Intro to AutoML

References
- AutoML guide. Note, AutoML Tables is a paid service.
In the exercise that follows this tutorial, we'll show you how to claim
$300 of free credits that you can use to train your own models
https://cloud.google.com/automl-tables/docs/beginners-guide

- Notebooks of kagglemodels
https://www.kaggle.com/vbmokin/data-science-for-tabular-data-advanced-techniques

7 Steps of Machine Learning
https://towardsdatascience.com/the-7-steps-of-machine-learning-2877d7e5548e

1. Gather data
2. Prepare the data - Deal with missing values and categorical data.
(Feature engineering is covered in a separate course.)
3. Select a model
4. Train the model - Fit decision trees and random forests to patterns in
training data.
5. Evaluate the model - Use a validation set to assess how well a trained
model performs on unseen data.
6. Tune parameters - Tune parameters to get better performance from XGBoost
models.
7. Get predictions - Generate predictions with a trained model and submit your
results to a Kaggle competition.

"""
# %%  INTRO TO MACHINE LEARNING - Under/overfitting
"""
overfitting can bre controlled by varying max # of leaf nodes (more means
deeper tree, more likely to overfit, as there are fewer datapoints in each
leaf node)

# function we can call in a loop to test MAE score for various values
# of max # leaves

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,
                                  random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# You know the best tree size. If you were going to deploy this model in
practice, you would make it even more accurate by using all of the data and
keeping that tree size. That is, you don't need to hold out the validation
data now that you've made all your modeling decisions.

scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for
          leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)

final_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=0)

"""
# %% INTRO TO MACHINE LEARNING - Random Forests
"""
Better predictive accuracy than a single decision tree and it works well
 with default parameters. If you keep modeling, you can learn more models
 with even better performance, but many of those are sensitive to getting
 the right parameters.

XGBoost model, provides better performance when tuned well with the right
parameters (but which requires some skill to get the right model parameters).

EXERCISES
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

1. Use RF

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_mae = mean_absolute_error(rf_model.predict(val_X), val_y)

"""
# %% INTRO TO MACHINE LEARNING - AutoML
"""
7 steps of Machine Learning
1. Gather data
2. Prepare the data - Deal with missing values and categorical data.
(Feature engineering is covered in a separate course.)
3. Select a model
4. Train the model - Fit decision trees and random forests to patterns in
training data.
5. Evaluate the model - Use a validation set to assess how well a trained
model performs on unseen data.
6. Tune parameters - Tune parameters to get better performance from XGBoost
models.
7. Get predictions - Generate predictions with a trained model and submit your
results to a Kaggle competition.

Google Cloud AutoML Tables automates the machine learning process,steps 2-7


"""
# %% INTERMEDIATE MACHINE LEARNING -
"""
Outline
-------
- Missing Values
- Categorical Variables
- Pipelines
- Cross-Validation
- XGBoost
- Data Leakage

"""

# %%  INTERMEDIATE MACHINE LEARNING - Missing Values
"""
3 Approaches
------------
1) Drop Columns with Missing Values¶
2) Imputation with mean
    If columns with missing values are mostly full, makes sense to do
    imputation to keep rest of data.
3) blend -  impute missing values, as before, but for each column with
    missing entries in the original dataset, we add a new column that shows
    the location of the imputed entries.

    In some cases, this will meaningfully improve results.
    In other cases, it doesn't help at all.

-------------
1. Drop columns with missing
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

-------------
2.SimpleImputer to replace missing values with the mean value along each
column.

Although it's simple, filling in the mean value generally performs quite
 well (but this varies by dataset).

 While statisticians have experimented
 with more complex ways to determine imputed values (such as regression
imputation, for instance), the complex strategies typically give no
additional benefit once you plug the results into sophisticated ML models.

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
-------------
3. blended approach
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus,
                    y_train, y_valid))

------------------
Reference Code
------------------

Missing Values
--------------
import pandas as pd
from sklearn.model_selection import train_test_split

X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8,
                                                      test_size=0.2,
                                                      random_state=0)
missing_val_count_by_column = (X_train.isnull().sum())
#print(missing_val_count_by_column)
print(missing_val_count_by_column[missing_val_count_by_column > 0])

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches; for example, comparing different
# imputation strategies for variables (drop, median imputation, average,
# blended, etc. or different encoding types on categorical variables -
# drop, labelencoding or onehot encoding )
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

drop_cols = missing_val_count_by_column[missing_val_count_by_column > 0].index
reduced_X_train = X_train.drop(drop_cols, axis=1)
reduced_X_valid = X_valid.drop(drop_cols, axis=1)

# alternate strategy of iputation - median
# myimputer = SimpleImputer(strategy='median')

# preprocess test data, make predictions, save to file
final_X_test = pd.DataFrame(myimputer.transform(X_test))
preds_test = model.predict(final_X_test)
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

Categorical variable encodings
------------------------------

# All categorical columns
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if
                   set(X_train[col]) == set(X_valid[col])]

# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))

print('Categorical columns that will be label encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

from sklearn.preprocessing import LabelEncoder

label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

label_encoder = LabelEncoder()
for col in good_label_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])

# TESTING CARDINALITY
# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])

# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)

from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data and low cardinality
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# -- submission
from sklearn.impute import SimpleImputer

myimputer = SimpleImputer(strategy='most_frequent') # use  “most_frequent” to fill in nan or missing for both numerical and categoricals
imputed_X_train = pd.DataFrame(myimputer.fit_transform(X_train))
imputed_X_test = pd.DataFrame(myimputer.transform(X_test))


imputed_X_train.columns = X_train.columns
imputed_X_test.columns = X_test.columns
imputed_X_test.index = X_test.index

# OH encode imputed categorical columns with low_cardinality
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
_ = pd.DataFrame(OH_encoder.fit_transform(imputed_X_train[low_cardinality_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(imputed_X_test[low_cardinality_cols]))

# One-hot encoding removed index; put it back
OH_cols_test.index = imputed_X_test.index

# Remove categorical columns (will replace with one-hot encoding of low cardinality categorical columns)
num_X_test = imputed_X_test.drop(object_cols, axis=1)

# Add one-hot encoded columns of low cardinality categorical to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

# train data on X_test_clean_num_lowcat
# train model on train data
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(OH_X_train, y_train)
# predict using test
preds_test = model.predict(OH_X_test)

# df with index of test, prediction
output = pd.DataFrame({'Id': OH_X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

EXERCISES
---------
Why imputation gave worse results:
Solution: Given that there are so few missing values in the dataset, we'd
expect imputation to perform better than dropping columns entirely.
However, we see that dropping columns performs slightly better! While this
 can probably partially be attributed to noise in the dataset, another
 potential explanation is that the imputation method is not a great match
 to this dataset. That is, maybe instead of filling in the mean value, it
 makes more sense to set every missing value to a value of 0, to fill in
 the most frequently encountered value, or to use some other method. For
 instance, consider the GarageYrBlt column (which indicates the year that
 the garage was built). It's likely that in some cases, a missing value
 could indicate a house that does not have a garage. Does it make more sense
 to fill in the median value along each column in this case? Or could we get
 better results by filling in the minimum value along each column? It's not
 quite clear what's best in this case, but perhaps we can rule out some
 options immediately - for instance, setting missing values in this column
 to 0 is likely to yield horrible results!


"""
# %%  INTERMEDIATE MACHINE LEARNING -  Categorical variables
"""3 approaches to categorical variables
1.	Drop them – because they need to be preprocessed by ML models; they are
not numbers that can be entered directly

2.	Label encoding
a.	Can be ordinal #s, which do well in tree models and Random forest,
because no scaling is needed
b.	Be sure to check unique values in both train and valid data sets,
because there may be different values in validation, which could cause
error in one hot or label encoding
c.	The simplest approach, however, is to drop the problematic categorical
columns.

3.	One hot encoding
a.	Don’t use if there are more than 15 different values that variable can
take
b.	Works well when no ordering of values, i.e., each value is equal weight
c.	We set handle_unknown = 'ignore' to avoid errors when the validation data
contains classes that aren't represented in the training data,
d.	setting sparse=False ensures that the encoded columns are returned as a
numpy array (instead of a sparse matrix).

"""
# %%  INTERMEDIATE MACHINE LEARNING - Pipelines
"""
Pipelines are a simple way to keep your data preprocessing and modeling code
organized. Specifically, a pipeline bundles preprocessing and modeling steps
 so you can use the whole bundle as if it were a single step.

Many data scientists hack together models without pipelines, but pipelines
 have some important benefits. Those include:

    Cleaner Code: Accounting for data at each step of preprocessing can get
    messy. With a pipeline, you won't need to manually keep track of your
    training and validation data at each step.

    Fewer Bugs: There are fewer opportunities to misapply a step or forget
    a preprocessing step.

    Easier to Productionize: It can be surprisingly hard to transition a
    model from a prototype to something deployable at scale. We won't go
    into the many related concerns here, but pipelines can help.

    More Options for Model Validation: You will see an example in the
    next tutorial, which covers cross-validation.

Step 1: Define Preprocessing Steps¶
Step 2: Define the Model
Step 3: Create and Evaluate the Pipeline
--------------------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(
    steps=[('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[('num', numerical_transformer, numerical_cols),
                  ('cat', categorical_transformer, categorical_cols)])

model = RandomForestRegressor(n_estimators=100, random_state=0)

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])
my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_valid)
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
"""


# %%  INTERMEDIATE MACHINE LEARNING -  Cross Validation

"""
Cross valid takes longer, but can yield more accurate models and cleans up
code (don't need to keep track of validation dataset)
- For small datasets, where extra computational burden isn't a big deal,
    run cross-validation.
- For larger datasets, a single validation set is sufficient.

scikit learn has many different scoring criteria for crossvalidation
https://scikit-learn.org/stable/modules/model_evaluation.html


Reference Code from EXERCISES
------------------
import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('../input/train.csv', index_col='Id')
test_data = pd.read_csv('../input/test.csv', index_col='Id')
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)

# use numeric columns for simplicity in this example
numeric_cols = [cname for cname in train_data.columns if
                train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])


def get_score(n_estimators):
    # Return the average MAE over 3 CV folds of random forest model.

    #     Keyword argument:
    # n_estimators -- the number of trees in the forest

    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                  ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
                                 ])
    # specify negative MAE because Scikit-learn has a convention where all
    # metrics are defined so a high number is better. Using negatives here
    # allows them to be consistent with that convention, though negative MAE is
    # almost unheard of elsewhere.
    scores = -1 * cross_val_score(my_pipeline, X, y, cv=3,
                                  scoring='neg_mean_absolute_error')

    return scores.mean()

trees = [50*(i+1) for i in range(8)]
results = {tree:get_score(tree) for tree in trees}

"""

# %%  INTERMEDIATE MACHINE LEARNING -  data leakage
"""
target leakage - when predictors include data that will not be available
at the time you make predictions. It is important to think about target
leakage in terms of the timing or chronological order that data
becomes available, not merely whether a feature helps make good predictions.

train-test contamination -  if the validation data affects the preprocessing
behavior. For example, imagine you run preprocessing (like fitting an
imputer for missing values) before calling train_test_split().
The end result? Your model may get good validation scores, giving you great
confidence in it, but perform poorly when you deploy it to make decisions.

EXERCISES
1.  Nike has hired you as a data science consultant to help them save money
on shoe materials. Your first assignment is to review a model one of their
employees built to predict how many shoelaces they'll need each month. The
features going into the machine learning model include:

    The current month (January, February, etc)
    Advertising expenditures in the previous month
    Various macroeconomic features (like the unemployment rate) as of the
    beginning of the current month
    The amount of leather they ended up using in the current month

The results show the model is almost perfectly accurate if you include the
feature about how much leather they used. But it is only moderately accurate
if you leave that feature out. You realize this is because the amount of
 leather they use is a perfect indicator of how many shoes they produce,
 which in turn tells you how many shoelaces they need.

Do you think the leather used feature constitutes a source of data leakage?
If your answer is "it depends," what does it depend on?
A. This is tricky, and it depends on details of how data is collected
 (which is common when thinking about leakage). Would you at the beginning
 of the month decide how much leather will be used that month? If so, this
 is ok. But if that is determined during the month, you would not have access
 to it when you make the prediction. If you have a guess at the beginning of
 the month, and it is subsequently changed during the month, the actual amount
 used during the month cannot be used as a feature, because it causes
 leakage.

--- --- ---------------------
Q2. You have a new idea. You could use the amount of leather Nike ordered
(rather than the amount they actually used) leading up to a given month as
a predictor in your shoelace model.

Does this change your answer about whether there is a leakage problem? If
 you answer "it depends," what does it depend on?

2A. This could be fine, but it depends on whether they order shoelaces first
or leather first. If they order shoelaces first, you won't know how much
 leather they've ordered when you predict their shoelace needs. If they
 order leather first, then you'll have that number available when you place
 your shoelace order, and you should be ok.

--- --- ---------------------
Q3. Your friend, who is also a data scientist, says he has built a model
that will let you turn your bonus into millions of dollars. Specifically,
his model predicts the price of a new cryptocurrency (like Bitcoin, but a
                                                      newer one) one day
ahead of the moment of prediction. His plan is to purchase the cryptocurrency
 whenever the model says the price of the currency (in dollars) is about to
 go up.

The most important features in his model are:

    Current price of the currency
    Amount of the currency sold in the last 24 hours
    Change in the currency price in the last 24 hours
    Change in the currency price in the last 1 hour
    Number of new tweets in the last 24 hours that mention the currency

The value of the cryptocurrency in dollars has fluctuated up and down by over
100 in last year, but model's average error is less than 1.

He says this is proof his model is accurate, and you should invest with
him, buying the currency whenever the model says it is about to go up.

Is he right? If there is a problem with his model, what is it?

Solution: There isno source of leakage here.  These features should be
available at the moment you want to makea prediction, and htey're unlikely to
be changed in the trainingdata after the prediction target is determined.
But, the way he describes accuracy could be misleading if you aren't careful.
If the price moves gradually, today's price will be an accurate predictor of
tomorrow's price, but it may not tell you whether it's a good time to invest.
For instance, if it is 100 today, a model prediction of a 100 tomorrow may
seem accurate, even if it can't tell you whether the price is going up or
down from the current price. A better prediction target would be the change
in price over the next day. If you can consistently predict whether the price
is about to go up or down (and by how much), you may have a winning
investment opportunity.
--- --- ---------------------

Q4. An agency that provides healthcare wants to predict which patients
 from a rare surgery are at risk of infection, so it can alert the nurses
 to be especially careful when following up with those patients.

You want to build a model. Each row in the modeling dataset will be a
 single patient who received the surgery, and the prediction target will
 be whether they got an infection.

Some surgeons may do the procedure in a manner that raises or lowers the
risk of infection. But how can you best incorporate the surgeon information
into the model?

You have a clever idea.

    Take all surgeries by each surgeon and calculate the infection rate
    among those surgeons.
    For each patient in the data, find out who the surgeon was and plug in
    that surgeon's average infection rate as a feature.

Does this pose any target leakage issues? Does it pose any train-test
contamination issues?

A4.
Solution: This poses a risk of both
TARGET LEAKAGE and TRAIN-TEST CONTAMINATION
(though you may be able to avoid both if you are careful).

You have TARGET LEAKAGE if a given patient's outcome contributes to the
infection rate for his surgeon, which is then plugged back into the
prediction model for whether that patient becomes infected. You can avoid
 target leakage if you calculate the surgeon's infection rate by using only
 the surgeries before the patient we are predicting for. Calculating this
 for each surgery in your training data may be a little tricky.

You also have a TRAIN-TEST CONTAMINATION problem if you calculate this
using all surgeries a surgeon performed, including those from the test-set.
 The result would be that your model could look very accurate on the test
 set, even if it wouldn't generalize well to new patients after the model
 is deployed. This would happen because the surgeon-risk feature accounts
 for data in the test set. Test sets exist to estimate how the model will
 do when seeing new data. So this contamination defeats the purpose of the
 test set.
--- --- ---------------------

Q5. You will build a model to predict housing prices. The model will be
deployed on an ongoing basis, to predict the price of a new house when a
description is added to a website. Here are four features that could be
used as predictors.

    a Size of the house (in square meters)
    b Average sales price of homes in the same neighborhood
    c Latitude and longitude of the house
    d Whether the house has a basement

You have historic data to train and validate the model.

Which of the features is most likely to be a source of leakage?

A5.
2 is the source of target leakage. Here is an analysis for each feature:

    The size of a house is unlikely to be changed after it is sold
    (though technically it's possible). But typically this will be
     available when we need to make a prediction, and the data won't
     be modified after the home is sold. So it is pretty safe.

    We don't know the rules for when this is updated. If the field is
    updated in the raw data after a home was sold, and the home's sale
    is used to calculate the average, this constitutes a case of target
    leakage. At an extreme, if only one home is sold in the neighborhood,
    and it is the home we are trying to predict, then the average will be
    exactly equal to the value we are trying to predict. In general, for
    neighborhoods with few sales, the model will perform very well on the
    training data. But when you apply the model, the home you are predicting
    won't have been sold yet, so this feature won't work the same as it did
    in the training data.

    These don't change, and will be available at the time we want to make a
    prediction. So there's no risk of target leakage here.

    This also doesn't change, and it is available at the time we want to
    make a prediction. So there's no risk of target leakage here.

"""

# %% MACHINE LEARNING EXPLAINABILITY -
"""
Outline
- Use Cases for Model Insights
- Permutation Importance
- Partial dependence Plots
- SHAP Values
- Advanced Uses of SHAP Values (aggregating SHAP values, i.e, over more than
                                a single prediction)
"""

# %% MACHINE LEARNING EXPLAINABILITY - Use Cases
"""
- help in debugging and human decision-making/intuition
- inform feature engineering and future data collection
- build trust with and explain algorithms to team members that are not
  data scientists

"""
# %% MACHINE LEARNING EXPLAINABILITY - Permutation Importance

"""
Compared to most other approaches to measuring feature importance,
permutation importance is:
    fast to calculate,
    widely used and understood, and
    consistent with properties we would want a feature importance measure to
    have.
- PI is calculated after a model has been fitted
- Shuffle the values in a single column, make predictions using the
resulting dataset. Use these predictions and the true target values to
calculate how much the loss function suffered from shuffling. That
performance deterioration measures the importance of the variable you
just shuffled.

PI  = loss function(shuffled) - loss function(unshuffled)
because loss will be bigger when shuffled (usually), a bigger PI means the loss
increased a lot

- Return the data to the original order (undoing the shuffle from step 2).
Now repeat step 2 with the next column in the dataset, until you have
calculated the importance of each column.

- SOMEx multiple shufflings are done, because somex low importance variables
provide more accurate predictions AFTER shuffling

CON
- PI doesn't give a full understanding of variable's impact
- can be negative for low importance variables, because shuffle yields better
  results
example:
    If a feature has medium permutation importance, that could mean it has
    a large effect for a few predictions, but no effect in general, or
    a medium effect for all predictions.

# The width of the effects range is not a reasonable approximation to
# permutation importance because it can be determined by just a few outliers.
However if all dots on the graph
# are widely spread from each other, that is a reasonable indication
# that permutation importance is high.

# Because the range of effects is so
# sensitive to outliers, permutation importance is a better measure of
# what's generally important to the model.

output is simple chart

weight   feature
------   --------
0.15     goal scored
0.05     distance covered
etc.      ...

"""
import numpy as np
import pandas as pd
import eli5
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from eli5.sklearn import PermutationImportance

data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from "Yes"/"No" to 1/0
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=100,
                                  random_state=0).fit(train_X, train_y)
perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names=val_X.columns.tolist())

# EXERCISES
# difference in latitude pickup and dropoff is more important probably because
# more taxi customers take long distancerides up and down manhattan; long
# distances on manhattan are not as common

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv',
                   nrows=50000)

# Remove data with extreme outlier coordinates or negative fares
data = data.query('pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' +
                  'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +
                  'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +
                  'dropoff_longitude > -74 and dropoff_longitude < -73.9 and '
                  + 'fare_amount > 0'
                  )
y = data.fare_amount

base_features = ['pickup_longitude',
                 'pickup_latitude',
                 'dropoff_longitude',
                 'dropoff_latitude',
                 'passenger_count']

X = data[base_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
first_model = RandomForestRegressor(n_estimators=50, random_state=1).fit(train_X, train_y)

# %% MACHINE LEARNING EXPLAINABILITY - Partial dependence Plots

"""
partial dependence plots show HOW a feature affects predictions.
- Controlling for all other house features, what impact do longitude and
latitude have on home prices? To restate this, how would similarly sized houses
be priced in different areas?

- Are predicted health differences between two groups due to differences in
their diets, or due to some other factor?

can be interpreted similarly as coefficients in linear or log regression
models, although partial dependence plots can capture even more complex
patterns

How?
- calculated AFTER a model has been fit
- start with single row of data, and use model to predict outcome on vertical
as chosen variable along x axis is increased, but all other variables held
constant
- repeat for other rows, increasing our variable but holding others constant;
then average outcome to plot on vertical axis
- interaction between features may cause plot for a single row to be atypical
The y axis is interpreted as change in the prediction from what it would
be predicted at the baseline or leftmost value.

2D partial dependence plots
- contour plots show interaction of features; what combination of two features
leads to in prediction
- for regression,such as taxi fare, contour will be value of fare
- for logistic regression, values of probability of prediction will be output;
will be a series of levels across the grid of x,y coords
"""

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model,
                            dataset=val_X,
                            model_features=feature_names,
                            feature='Goal Scored')
# plot it
pdp.pdp_plot(pdp_goals, 'Goal Scored')
plt.show()
# -------
# Build Random Forest model
rf_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

pdp_dist = pdp.pdp_isolate(model=rf_model,
                           dataset=val_X,
                           model_features=feature_names,
                           feature=feature_to_plot)

pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()

# Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot
features_to_plot = ['Goal Scored', 'Distance Covered (Kms)']
inter1  =  pdp.pdp_interact(model=tree_model, dataset=val_X,
                            model_features=feature_names,
                            features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()

# EXERCISES
# 1.
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

feat_name = 'pickup_longitude'
pdp_dist = pdp.pdp_isolate(model=first_model, dataset=val_X,
                           model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

# 2.
features_to_plot = ['pickup_longitude', 'dropoff_longitude']
inter1  =  pdp.pdp_interact(model=first_model, dataset=val_X,
                            model_features=base_features,
                            features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1,
                      feature_names=features_to_plot, plot_type='contour')
plt.show()

# 4.  When absolute distance travalled is controlled for with new variable,
# partial dependence on pickup_longitude has much less impact on fare
# amount. So we can have seemingly important variable become uninmportant
# once real variable is found/formulated.  This is where intuition needs
# to help, because the training of a model can fit outcome to input
# even though the input is not directly related to output

# 5. Consider a scenario where you have only 2 predictive features, which
# we will call feat_A and feat_B. Both features have minimum values of -1
# and maximum values of 1. The partial dependence plot for feat_A increases
# steeply over its whole range, whereas the partial dependence plot for
#  feature B increases at a slower rate (less steeply) over its whole range.

# Does this guarantee that feat_A will have a higher permutation
# importance than feat_B. Why or why not?

# Solution: No. This doesn't guarantee feat_a is more important.
# For example, feat_a could have a big effect in the cases where it varies,
# but could have a single value 99% of the time. In that case, permuting
# feat_a wouldn't matter much, since most values would be unchanged.

# 6. This code creates a x1, x2 which are random in range
# [-2,2], and y variable that is negative slope
# from for X1, x2 in -2 to -1, positive -1 to 1, and, negative
# again for 1-2, and y not defined otherwise;
# the boolean expressions (x1<-1) and (X1>1) acts to make this a
# discontinuous function

X1 = 4 * rand(n_samples) - 2
X2 = 4 * rand(n_samples) - 2
y = -2 * X1 * (X1<-1) + X1 - 2 * X1 * (X1>1) - X2

""" EX 7
Create a dataset with 2 features and a target, such that the pdp of the
 first feature is flat, but its permutation importance is high. We will use a
 RandomForest for the model.
"""
import eli5
from eli5.sklearn import PermutationImportance

n_samples = 20000

X1 = 4 * rand(n_samples) - 2
X2 = 4 * rand(n_samples) - 2
y = (X1>-1)*(X1<1)*X2
# this way, X1 controls y; y follows X2 in range
# [-1,1] but has 0 slope against X1. y=0 outside the range
# logistical regressions are built this way

# create dataframe because pdp_isolate expects a dataFrame as an argument
my_df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
predictors_df = my_df.drop(['y'], axis=1)

my_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(predictors_df, my_df.y)


pdp_dist = pdp.pdp_isolate(model=my_model, dataset=my_df, model_features=['X1', 'X2'], feature='X1')
pdp.pdp_plot(pdp_dist, 'X1')
plt.show()

perm = PermutationImportance(my_model).fit(predictors_df, my_df.y)

# Check your answer
q_7.check()

# show the weights for the permutation importance you just calculated
eli5.show_weights(perm, feature_names = ['X1', 'X2'])

# %% MACHINE LEARNING EXPLAINABILITY - SHAP Values
"""
SHAP Values (SHapley Additive exPlanations) break down a prediction to show
the impact of each feature.
SHAP values interpret the impact of having a certain value for a given
feature in comparison to the prediction we'd make if that feature took some
 baseline value.
- it's easier to give a concrete, numeric answer if we restate this as:
  How much was a prediction driven by the fact that the team scored 3 goals,
  instead of some baseline number of goals.

- A model says a bank shouldn't loan someone money, and the bank is
legally required to explain the basis for each loan rejection
- A healthcare provider wants to identify what factors are driving each
patient's risk of some disease so they can directly address those risk
factors with targeted health interventions

https://towardsdatascience.com/one-feature-attribution-method-to-supposedly-rule-them-all-shapley-values-f3e04534983d
"""
sum(SHAP values for all features) = pred_for_team - pred_for_baseline_values

row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]
# use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)


my_model.predict_proba(data_for_prediction_array)
# array([[0.29, 0.71]])
# The team is 70% likely to have a player win the award.
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)

"""
The shap_values object above is a list with two arrays. The first array
is the SHAP values for a negative outcome (don't win the award), and the
second array is the list of SHAP values for the positive outcome
(wins the award). We typically think about predictions in terms
of the prediction of a positive outcome, so we'll pull out SHAP values for
positive outcomes (pulling out shap_values[1]).

Other fnctions:

    shap.DeepExplainer works with Deep Learning models.
    shap.KernelExplainer works with all models, though it is slower than
    other Explainers and it offers an approximation rather than exact Shap
    values.

"""
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1],
                data_for_prediction)
# use Kernel SHAP to explain test set predictions
k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap.force_plot(k_explainer.expected_value[1], k_shap_values[1],
                data_for_prediction)

# EXERCISES
# --------------------
"""1. You have built a simple model, but the doctors say they don't know
how to evaluate a model, and they'd like you to show them some evidence the
model is doing something in line with their medical intuition. Create any
 graphics or tables that will show them a quick overview of what the model
 is doing?

They are very busy. So they want you to condense your model overview into
 just 1 or 2 graphics, rather than a long string of graphics.

We'll start after the point where you've built a basic model.
Just run the following cell to build the model called my_model.
"""
# --------------------
#  Reference code
# --------------------

# Calculate and show permutation importance:

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())

#Calculate and show partial dependence plot:

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X,
                            model_features=feature_names,
                            feature='Goal Scored')
# plot it
pdp.pdp_plot(pdp_goals, 'Goal Scored')
plt.show()

# Calculate and show Shap Values for One Prediction:

import shap  # package used to calculate Shap values

data_for_prediction = val_X.iloc[0,:]
# use 1 row of data here. Could use multiple rows if desired

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0],
                data_for_prediction)

# %% MACHINE LEARNING EXPLAINABILITY - Advanced usage of SHAP Values
"""
1. summary plots - give birds eye view of variable's impact, feature on
   vertical, horizontal is SHAP value (vertical on PDP), color is high/low
   feature value (horizontal axis on PDP plot)
2. dependence contribution plots - show variation of output with value of
    feature, but also importance; each dot is a row of data; spread indicates
    there is interaction with other features; color can indicate the second
    variable (feature 1 along x axis, feature 2 is color, y axis is SHAP value
          for feature 1 on prediction of best player award, for example)
+ SHAP means has positive impact on target variable outcome vs. baseline value
of feature
- SHAP means there is negative impact vs. baseline value of feature
"""

import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(val_X)

# Make plot. Index of [1] is prediction of True; for classification problems,
# there is an array ofSHAP values for each possible outcome

# NOTE - SHAP values/plots can take a while, XGBoost has optimizations, though.
shap.summary_plot(shap_values[1], val_X)

# SHAP Dependence Contribution Plots¶

import shap  # package used to calculate Shap values

explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(X)
shap.dependence_plot('Ball Possession %', shap_values[1], X,
                     interaction_index="Goal Scored")

"""
EXERCISES
4.
The jumbling suggests that sometimes increasing that feature leads to
 higher predictions, and other times it leads to a lower prediction.
 Said another way, both high and low values of the feature can have both
 positive and negative effects on the prediction. The most likely
 explanation for this "jumbling" of effects is that the variable
 (in this case num_lab_procedures) has an INTERACTION EFFECT with other
 variables. For example, there may be some diagnoses for which it is good
 to have many lab procedures, and other diagnoses where suggests
 increased risk. We don't yet know what other feature is interacting
 with num_lab_procedures though we could investigate that with SHAP
 contribution dependence plots.
"""
