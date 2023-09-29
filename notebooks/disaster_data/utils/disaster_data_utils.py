import pandas as pd
import os

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = f"{current_path}/../../.."

def build_clean_dataframe() -> pd.DataFrame:
    df = build_dataframe()
    df = remove_irrelevant_disasters(df)
    df = remove_irrelevant_columns(df)
    # only drop rows with missing values after deleting all irelevant columns because there are a lot of columns with missing values in general.
    df = df.dropna()

    return df

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
    
    
      