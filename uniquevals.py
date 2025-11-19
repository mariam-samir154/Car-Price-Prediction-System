import pandas as pd

def get_unique_values(df, features):
    unique_values = {}
    for feature in features:
        if feature in df.columns:
            unique_values[feature] = df[feature].dropna().unique().tolist()
        else:
            unique_values[feature] = []
    return unique_values

print("new runnnn")
def get_model_options(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'Model' not in df.columns:
            raise KeyError("'Model' column not found in the dataset.")
        model_options = sorted(df['Model'].dropna().unique().tolist())
        return model_options
    except Exception as e:
        return []
