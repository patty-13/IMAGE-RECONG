import yfinance as yf
import pandas as pd
def getTestData():
    dd = yf.download("NQ=F", interval="5m", period="1d")
    dd = pd.DataFrame(dd)
    dd.columns = dd.columns.get_level_values(0)
    filtered_df = dd[["Open", "High", "Close", "Low"]]
    ddd = pd.DataFrame(filtered_df, index=pd.to_datetime(filtered_df.index))
    ddd = ddd.between_time("06:00", "10:09")
    ddd['ochl_avg'] = (ddd['Open'] + ddd['High'] + ddd['Low'] + ddd['Close']) / 4
    ddd = ddd.reset_index()
    return ddd['ochl_avg']

# pp = getTestData()
# print(len(pp))

# def preprocess(df: pd.DataFrame) -> pd.DataFrame:
#     df.rename(columns={'<DATE>': 'date',
#                        '<TIME>': 'time',
#                        '<OPEN>': 'open',
#                        '<HIGH>': 'high',
#                        '<LOW>': 'low',
#                        '<CLOSE>': 'close',
#                        '<TICKVOL>': 'tickvol',
#                        '<VOL>': 'volume',
#                        '<SPREAD>': 'spread'},
#               inplace=True)
#     return df
# df = pd.read_csv('D:\pycharm_projects\CLEARAIMODELS\IMAGERECOGNIZER\@ENQ_M1.csv', sep='\t').pipe(preprocess)
# print(df)
# complete_chart = df[df['date'] == '2024.08.15'].reset_index(drop=True)
# print(complete_chart)