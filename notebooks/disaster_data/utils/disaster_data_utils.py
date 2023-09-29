import pandas as pd

root_path = "../.."

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
 
def build_clean_dataframe(df) -> pd.DataFrame:
  df = remove_irrelevant_disasters(df)
  return df
  