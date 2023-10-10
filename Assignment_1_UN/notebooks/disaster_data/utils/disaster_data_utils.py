import pandas as pd
import os
import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = f"{current_path}/../../.."

def build_clean_dataframe() -> pd.DataFrame:
    df = build_dataframe()
    df = remove_irrelevant_disasters(df)
    df = remove_irrelevant_columns(df)
    # only drop rows with missing values after deleting all irelevant columns because there are a lot of columns with missing values in general.
    df = df.dropna()

    return df

def build_feature_vector_v1(df, country, year) -> np.array:
    '''
    Returns a feature vector for the given country in the given year using the information present in the provided dataframe.

            Parameters:
                    df (pd.DataFrame): The dataframe containing the disaster data
                    country (string): The country to build the feature vector for
                    year (int): The year to build the feature vector for

            Returns:
                    vector (np.array): a feature vector for the given country in the given year
    '''
    row = df[(df['Country'] == country) & (df['Year'] == year)]
    num_disasters = len(row)
    num_deaths = row['Total Deaths'].sum()
    num_deaths_per_disaster = num_deaths / num_disasters if num_disasters > 0 else 0
    num_deaths_at_biggest_disaster = row['Total Deaths'].max()
    vector = np.array([num_disasters, num_deaths, num_deaths_per_disaster, num_deaths_at_biggest_disaster])
    return vector

# X{array-like, sparse matrix} of shape (n_samples, n_features)
def create_feature_train_matrix(country_year_pairs, feature_vector_builder=build_feature_vector_v1) -> np.array:
    """
    Create a feature matrix for a given DataFrame and list of years.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing data for different countries and years.
    - years (list): A list of years for which feature vectors should be created.
    - feature_vector_builder (callable): A function used to build feature vectors for each country and year.
                                         Default is build_feature_vector_v1.

    Returns:
    - np.array: A 2D numpy array representing the feature matrix, where each row corresponds to a country-year pair.

    Example:
    ```
    from utils.disaster_data_utils import *
    import numpy as np
    df = build_dataframe()
    df = build_clean_dataframe(df)
    feature_matrix = create_feature_matrix(df, np.arange(2000, 2005))
    ```
    """
    print(f"Creating feature matrix for {len(country_year_pairs)} country-year pairs)")
    df = build_clean_dataframe()
    result = []
    for (country, year) in country_year_pairs:
        # row = df[(df['Country'] == country) & (df['Year'] == year)]
        # # if len (row) != 0:
        # #     last_row = row
        feature_vector = feature_vector_builder(df, country, year)
        result.append(feature_vector)
    result = np.array(result)
    assert(result.shape[0] == len(country_year_pairs), "The number of rows in the feature matrix should be equal to the number of country-year pairs.")
    return result
# def create_feature_train_matrix(countries, years, feature_vector_builder=build_feature_vector_v1) -> np.array:
#     """
#     Create a feature matrix for a given DataFrame and list of years.

#     Parameters:
#     - df (pd.DataFrame): The input DataFrame containing data for different countries and years.
#     - years (list): A list of years for which feature vectors should be created.
#     - feature_vector_builder (callable): A function used to build feature vectors for each country and year.
#                                          Default is build_feature_vector_v1.

#     Returns:
#     - np.array: A 2D numpy array representing the feature matrix, where each row corresponds to a country-year pair.

#     Example:
#     ```
#     from utils.disaster_data_utils import *
#     import numpy as np
#     df = build_dataframe()
#     df = build_clean_dataframe(df)
#     feature_matrix = create_feature_matrix(df, np.arange(2000, 2005))
#     ```
#     """
#     df = build_clean_dataframe()
#     result = []
#     for country in countries:
#         for year in years:
#             row = df[(df['Country'] == country) & (df['Year'] == year)]
#             if len (row) != 0:
#                 last_row = row
#                 feature_vector = feature_vector_builder(df, country, year)
#                 result.append(feature_vector)
            
#     return np.array(result)


def build_dataframe() -> pd.DataFrame:
    path = f"{root_path}/data/disasters.xlsx"
    df = pd.read_excel(path, skiprows=6)
    return df
  
def add_column_causes_climate_change_sentinent(df) -> pd.DataFrame:
    new_column_name = "causes climate change sentiment"
    types_to_drop = ["Earthquake", "Transport accident", "Miscellaneous accident", "Industrial accident", "Epidemic", "Volcanic activity", "Complex Disasters", "Impact", "Animal accident"]
    df[new_column_name] = df["Disaster Type"].apply(lambda x: False if x in types_to_drop else True)
    return df

def remove_irrelevant_disasters(df) -> pd.DataFrame:
    df = add_column_causes_climate_change_sentinent(df)
    df = df[df["causes climate change sentiment"] == True]
    return df
 
def remove_irrelevant_columns(df) -> pd.DataFrame:
    columns_of_interest = ["Country", "Total Deaths", "Year", "Disaster Type"]
    df = df[columns_of_interest]
    return df
    
    
      