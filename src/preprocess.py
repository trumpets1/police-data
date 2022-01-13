import pandas as pd

def pp(df, year):
    start_year = pd.Timestamp(f'{year}-01-01')
    df_params = df[['PRIORITY', 'CALLTYPE_CODE']]
    df_params['OFFENSE_DAY'] = (pd.to_datetime(df['OFFENSE_DATE']) + pd.to_timedelta(df['OFFENSE_TIME']) -
                                start_year).dt.total_seconds()/(60 * 60 * 24)
    df_params['RESULT'] = df['FINAL_DISPO_CODE'].apply(
        lambda code: 1 if (code == 'A' or code == 'B' or code == 'C') else 0)
    return df_params