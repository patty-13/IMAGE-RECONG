import faiss
import pandas as pd

from __lib__ import *

class DataPrepSequence:
  def __init__(self):
    self.sequences = {}
    self.window_sizes = [30, 50, 70, 90, 110, 150, 180]

  def preprocess(self, df:pd.DataFrame)-> pd.DataFrame:
    df.rename(columns={'<DATE>': 'date',
                       '<TIME>': 'time',
                       '<OPEN>': 'open',
                       '<HIGH>': 'high',
                       '<LOW>': 'low',
                       '<CLOSE>': 'close',
                       '<TICKVOL>': 'tickvol',
                       '<VOL>': 'volume',
                       '<SPREAD>': 'spread'},
              inplace=True)
    return df

  def splitIntoSequences(self, data):
      for window in self.window_sizes:
          self.sequences[window] = pd.DataFrame(columns=['date', 'start_time', 'end_time', 'ochl_Avg'])
      total_step = sum((len(data) - window) for window in self.window_sizes)
      with tqdm(total=total_step, desc="Processing") as pbar:
          for window in self.window_sizes:
              for i in range(0, len(data) - window):
                  st_index, ed_index = i, i + window
                  temp_data = data.iloc[st_index:ed_index]
                  if temp_data['date'].nunique() > 1:
                      continue

                  DATE = temp_data['date'].iloc[0]
                  START_TIME = temp_data['time'].iloc[0]
                  END_TIME = temp_data['time'].iloc[-1]
                  OCHLV_AVG = temp_data['ochl_avg'].tolist()
                  OCHLV_AVG_NORM = self.normalize_sequence(temp_data['ochl_avg'])

                  new_row = {
                      'ochl_Avg': OCHLV_AVG,
                      'date': DATE,
                      'start_time': START_TIME,
                      'end_time': END_TIME,
                      'ochlv_avg_norm': OCHLV_AVG_NORM
                  }

                  self.sequences[window] = pd.concat([self.sequences[window], pd.DataFrame([new_row])],
                                                     ignore_index=True)
                  pbar.update(1)
      return self.sequences

  def normalize_sequence(self, seq):
    seq = np.array(seq)
    seq_reshaped = seq.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled_seq = scaler.fit_transform(seq_reshaped)
    return scaled_seq.flatten().tolist()

  def create_and_save_faiss_index(self, data, index_file, metadata_file, use_ip=False):
      vectors = np.array([np.array(seq) for seq in data['ochlv_avg_norm']], dtype=np.float32)
      metric = faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2

      if use_ip:
          norms = np.linalg.norm(vectors, axis=1, keepdims=True)
          vectors = vectors / norms

      dimension = vectors.shape[1]
      index = faiss.IndexFlat(dimension, metric)
      index.add(vectors)

      faiss.write_index(index, index_file)
      data[['date', 'start_time', 'end_time', 'ochl_Avg','ochlv_avg_norm']].to_csv(metadata_file, index=False)

      print(f"FAISS index saved to {index_file}")
      print(f"Metadata saved to {metadata_file}")


# driver code#
if __name__ == '__main__':
    obj = DataPrepSequence()
    df_5MIN_path = "D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/@ENQ_M5.csv"
    df_10MIN_path = "D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/@ENQ_M10.csv"
    file_path = "D:\pycharm_projects\CLEARAIMODELS\IMAGERECOGNIZER\STORE_DATA"
    # Load and preprocess data
    df1 = pd.read_csv(df_5MIN_path, sep='\t')
    df1 = obj.preprocess(df1)
    df2 = pd.read_csv(df_10MIN_path, sep='\t')
    df2 = obj.preprocess(df2)

    df1 = df1.iloc[13:].reset_index(drop=True)
    df2 = df2.iloc[7:].reset_index(drop=True)

    df1['ochl_avg'] = (df1['open'] + df1['high'] + df1['low'] + df1['close']) / 4
    df2['ochl_avg'] = (df2['open'] + df2['high'] + df2['low'] + df2['close']) / 4

    df_3M_5MIN = df1[(df1['date'] >= '2024.07.01') & (df1['date'] <= '2024.10.30')].reset_index(drop=True)

    # Generate sequences
    df_5M_SEQ = obj.splitIntoSequences(df_3M_5MIN)

    # Save sequences and create FAISS index for each window size
    for window, df_seq in df_5M_SEQ.items():
        sequence_file = f"df_5M_{window}_seq.csv"
        index_file = f"faiss_index_5M_{window}.index"
        metadata_file = f"metadata_5M_{window}.csv"

        seq_file_path = os.path.join(file_path, sequence_file)
        index_file_path = os.path.join(file_path, index_file)
        metadata_file_path = os.path.join(file_path, metadata_file)

        df_seq.to_csv(seq_file_path, index=False)

        print(f"Saved sequences to {sequence_file}")

        # Create FAISS index
        obj.create_and_save_faiss_index(df_seq, index_file_path, metadata_file_path, True)



