import pandas as pd


def preprocess(df, year):
    start_year = pd.Timestamp(f'{year}-01-01')
    df_params = df[['PRIORITY', 'CALLTYPE_CODE']]
    df_params['OFFENSE_DAY'] = (pd.to_datetime(df['OFFENSE_DATE']) + pd.to_timedelta(df['OFFENSE_TIME']) -
                                start_year).dt.total_seconds()/(60 * 60 * 24)
    df_params['RESULT'] = df['FINAL_DISPO_CODE'].apply(
        lambda code: 1 if (code == 'A' or code == 'B' or code == 'C') else 0)
    return df_params


def standardize(df, cols):
    df_new = df.copy()
    for col in cols:
        df_new[col] = (df_new[col] - df_new[col].mean()) / df_new[col].std()
    return df_new


def normalize(df, cols):
    df_new = df.copy()
    for col in cols:
        df_new[col] = (df_new[col] - df_new[col].min()) / \
            (df_new[col].max() - df_new[col].min())
    return df_new


def pp_standardize_normalize(df, year, cols=['PRIORITY', 'OFFENSE_DAY']):
    preprocessed = preprocess(df, year)
    standardized = standardize(preprocessed, cols)
    normalized = normalize(preprocessed, cols)
    return (preprocessed, standardized, normalized)
