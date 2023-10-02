from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


current_path = os.path.dirname(os.path.realpath(__file__))
root_path = f"{current_path}/../../.."
filename = "DF_UNsession_rawtxt_per_country_from1990.pkl"


def build_dataframe() -> pd.DataFrame:
    # df = pd.read_pickle(f"{root_path}_DF_UNsession_rawtxt_per_country_from1990.pkl")
    path = f"{root_path}/data/{filename}"
    df = pd.read_pickle(path)
    # use iso mapping to create country column
    iso_to_country_name_map = build_iso_to_country_name_map()
    df["country"] = df["country"].map(iso_to_country_name_map)
    return df

def create_target_value_matrix(country_year_pairs, keywords, stemmer=PorterStemmer(), tokenizer=RegexpTokenizer(r'\w+')) -> np.array:
    countries = [pair[0] for pair in country_year_pairs]
    years = [pair[1] for pair in country_year_pairs] 
    df = create_prepoccesed_df(countries=countries, years=years, keywords=keywords, stemmer=stemmer, tokenizer=tokenizer)
    result = []
    for (country, year) in country_year_pairs:
        row = df[(df['country'] == country) & (df['year'] == year)]
        if len(row) != 1:
            print("this should not happen")
        result.append(row["year_score"])
    return np.array(result)

# def create_target_value_matrix(countries, years, keywords, stemmer=PorterStemmer(), tokenizer=RegexpTokenizer(r'\w+')) -> np.array:
#     df = create_prepoccesed_df(countries=countries, years=years, keywords=keywords, stemmer=stemmer, tokenizer=tokenizer)
#     result = []
#     for country in countries:
#         for year in years:
#             row = df[(df['country'] == country) & (df['year'] == year)]
#             if len(row) != 1:
#                 print("this should not happen")
#             result.append(row["year_score"])
#     return np.array(result)


def create_prepoccesed_df(countries, years, keywords, stemmer=PorterStemmer(), tokenizer=RegexpTokenizer(r'\w+')):
    print("prepoccesing df")
    # df = pd.read_pickle("../../data/DF_UNsession_rawtxt_per_country_from1990.pkl")
    df = build_dataframe()
    ## filter df for countries and years
    df = df[df["country"].isin(countries)]
    df = df[df["year"].isin(years)]
    df = create_stemmed_df(df, stemmer, tokenizer)
    # df = add_stemmed_df_year(df, stemmer, tokenizer)
    df = add_keyword_counts_column(df, keywords, stemmer, tokenizer)
    df = add_year_score_column(df, stemmer, tokenizer)
    return df
         
def create_stemmed_df(df, stemmer=PorterStemmer(), tokenizer=RegexpTokenizer(r'\w+')):
    df_tokenized_per_country = pd.DataFrame()
    for row in df.iterrows():
        country = row[1]["country"]
        year = row[1]["year"]
        txt = row[1]["txt"]

        txt_tokenized = stem_tokenizer(txt, stemmer, tokenizer)
        
        # Create a new row DataFrame with year and country and the tokenized text
        new_row = pd.DataFrame({"country":[country], "year": [year], "txt_stemmed": [txt_tokenized]})
        
        # Concatenate the new row DataFrame to df_tokenized_per_year
        df_tokenized_per_country = pd.concat([df_tokenized_per_country, new_row], ignore_index=True)
    return df_tokenized_per_country         
         
def add_year_score_column(df, stemmer=PorterStemmer(), tokenizer=RegexpTokenizer(r'\w+')):
    print("adding year score column to df")
    for year in df["year"].unique():
        df_year = df.loc[df["year"]==year]
        max_count_year = df_year["keyword_counts"].max()
        min_count_year = df_year["keyword_counts"].min()
        # for each country, add normalized score columm to df for that year
        df.loc[df["year"]==year, "year_score"] = df_year["keyword_counts"].apply(lambda x: normalize_score_min_max(x, min_count_year, max_count_year))
    return df
              

def add_keyword_counts_column(df, keywords, stemmer=PorterStemmer(), tokenizer=RegexpTokenizer(r'\w+')):
    '''
    Returns a dataframe with a column containing the stemmed text of the original dataframe.

            Parameters:
                    df (pd.DataFrame): The dataframe containing the disaster data
                    stemmer (PorterStemmer): The stemmer to use
                    tokenizer (RegexpTokenizer): The tokenizer to use

            Returns:
                    df (pd.DataFrame): The dataframe containing the disaster data with an extra column containing the stemmed text
    '''
    keywords = [stem_tokenizer(keyword, stemmer=stemmer, tokenizer=tokenizer) for keyword in keywords]

    print("adding keyword counts column to df by matching keywords in stemmed stemmed text")
    df['keyword_counts'] = df['txt_stemmed'].apply(lambda x: count_keywords_in_text(keywords, x))
    return df


def build_tokenized_text(df, country, year, tokenizer=RegexpTokenizer(r'\w+'), stemmer=PorterStemmer()) -> str:
    sub_df = df[(df['Country'] == country) & (df['Year'] == year)]
    df_of_year = df.loc[df["year"]==year]
    corpus_of_year = ' '.join(sub_df.txt)
    corpus_tokanized = stem_tokenizer(corpus_of_year, stemmer, tokenizer)
    

def stem_tokenizer(txt, stemmer, tokenizer):
    txt = tokenizer.tokenize(txt.lower())
    txt = [stemmer.stem(word) for word in txt]
    txt = ' '.join(txt)
    return txt

def count_keywords_in_text(keywords, text):
    count = 0
    for keyword in keywords:
        count  += text.count(keyword)
    return count


def normalize_score_min_max(score, min_count, max_count):
    if max_count == min_count:
        denom = 0
    else :
        denom = max_count - min_count
    return (score - min_count) / denom

def build_iso_to_country_name_map():
    path = f"{root_path}/data/CO2emissions.csv"
    # create df
    df = pd.read_csv(path)# Replace 'your_file.csv' with the actual path to your CSV file

    # create dictionary with iso codes as keys and countries as values using Country and ISO 3166-1 alpha-3 columns
    d = dict(zip(df['ISO 3166-1 alpha-3'], df["Country"]))
    return d
