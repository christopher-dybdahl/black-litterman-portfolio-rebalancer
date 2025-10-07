import eikon as ek
import numpy as np
import pandas as pd


def eikonimport(start_date, end_date):
    ek.set_app_key('SECRET_KEY')

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    frames_data_1 = []
    frames_data_2 = []
    frames_bm = []
    frames_index = []

    upper = end_date
    lower = upper
    while start_date < lower:
        lower -= pd.DateOffset(years=3)
        formatted_lower = lower.strftime('%Y-%m-%d')
        formatted_upper = (upper - pd.DateOffset(day=1)).strftime('%Y-%m-%d')

        df_1_chunk = ek.get_timeseries(['.RUA', 'ACWX.O', 'AGG'], ['Close'], start_date=formatted_lower,
                                       end_date=formatted_upper,
                                       interval='daily')
        frames_data_1.append(df_1_chunk)

        df_2_chunk = ek.get_timeseries(['.SPCBMICUSRE', '.MERH0A0', '.BCOM'], ['Close'],
                                       start_date=formatted_lower,
                                       end_date=formatted_upper,
                                       interval='daily')
        df_2_chunk = df_2_chunk.merge(df_1_chunk, right_index=True, left_index=True, how='left')
        df_2_chunk = df_2_chunk[['.RUA', 'ACWX.O', '.SPCBMICUSRE', 'AGG', '.MERH0A0', '.BCOM']]
        frames_data_2.append(df_2_chunk)

        df_bm_chunk = ek.get_timeseries(['.SP500'], ['Close'], start_date=formatted_lower, end_date=formatted_upper,
                                        interval='daily')
        frames_bm.append(df_bm_chunk)

        df_index_chunk = ek.get_timeseries(['.SPVXSPID', 'EUR=', '.MRILT'], ['Close'], start_date=formatted_lower,
                                           end_date=formatted_upper,
                                           interval='daily')
        frames_index.append(df_index_chunk)

        upper = lower

    frames_data_1.reverse()
    frames_data_2.reverse()
    frames_bm.reverse()
    frames_index.reverse()

    df_1 = pd.concat(frames_data_1)
    df_2 = pd.concat(frames_data_2)
    df_bm = pd.concat(frames_bm)
    df_index = pd.concat(frames_index)

    df_1 = df_1.loc[start_date:end_date]
    df_2 = df_2.loc[start_date:end_date]
    df_bm = df_bm.loc[start_date:end_date]
    df_index = df_index.loc[start_date:end_date]

    df_1.to_csv('df_1.csv')
    df_2.to_csv('df_2.csv')
    df_bm.to_csv('df_bm.csv')
    df_index.to_csv('df_index.csv')

if __name__ == '__main__':
    start = '2006-01-01'
    end = '2023-11-30'
    eikonimport(start, end)
