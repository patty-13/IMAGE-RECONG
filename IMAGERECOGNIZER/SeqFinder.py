import argparse

import faiss
from fastdtw import fastdtw as dtw

from __lib__ import *

class sequenceFinder:
    def __init__(self, testSeq):
        self.testSeq = testSeq
        self.ninjaSeqLen = 0
        self.window = [30, 50, 70, 90, 110, 150, 180]

    def seqCheck(self):

        closest_len = min(self.window, key=lambda x: abs(len(self.testSeq) - x))
        if closest_len == len(self.testSeq):
            return self.testSeq, len(self.testSeq)
        return self.testSeq + [0.0] * (closest_len - len(self.testSeq)), len(self.testSeq + [0.0] * (closest_len - len(self.testSeq)))

    def normalize_seq(self, seq):
        """
        Normalizes the sequence using MinMaxScaler.
        """
        seq = np.array(seq)
        seq_reshaped = seq.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_seq = scaler.fit_transform(seq_reshaped)
        return scaled_seq.flatten()

    # matchedStoredSeq
    def seqMatches(self, nmatches ,use_DTW=False):

        # checking ninja seq length
        self.testSeq, self.ninjaSeqLen  = self.seqCheck()
        print(f"TEST STEQ : {self.testSeq}")
        print(f"len : {self.ninjaSeqLen}")
        print(f"nmatches: {nmatches}")

        # normalizing seq got from ninja
        norm_test_seq = self.normalize_seq(self.testSeq)

        # setting up the path from where the sequences will be mapped.
        if self.ninjaSeqLen >= self.window[6]:
            return "LENGTH TOO BIG AND NO SEQ LENGTH THIS BIG IN DATABASE"
        else:
            stored_seq_file = f'D:\pycharm_projects\CLEARAIMODELS\IMAGERECOGNIZER\STORE_DATA\df_5M_{self.ninjaSeqLen}_seq.csv'
            faiss_index_file = f'D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/STORE_DATA/faiss_index_5M_{self.ninjaSeqLen}.index'
            meta_data_file = f'D:\pycharm_projects\CLEARAIMODELS\IMAGERECOGNIZER\STORE_DATA\metadata_5M_{self.ninjaSeqLen}.csv'

        if use_DTW:
            results = []

            stored_seq_df = pd.read_csv(stored_seq_file)

            for idx, row in tqdm(stored_seq_df.iterrows(), total=len(stored_seq_df)):
                stored_seq = np.array(eval(row['ochlv_avg_norm']))
                distance, _= dtw(norm_test_seq, stored_seq)
                results.append({
                    'date': row['date'],
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'ochl_avg': row['ochl_Avg'],
                    'dtw_distance': distance
                })

            results.sort(key=lambda x: x['dtw_distance'])
            return results[:nmatches]
        else:

            index  = faiss.read_index(faiss_index_file)
            metadata = pd.read_csv(meta_data_file)

            test_vector = norm_test_seq.reshape(1, -1).astype(np.float32)

            # Search for the closest matches
            distances, indices = index.search(test_vector, 100)

            results = []
            for idx in indices[0]:
                if idx == -1:
                    continue  # Skip invalid indices
                row = metadata.iloc[idx]
                stored_seq = np.array(eval(row['ochlv_avg_norm']))
                dtw_dist, _ = dtw(test_vector.flatten(), stored_seq)
                results.append({
                    'date': row['date'],
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'ochl_avg': row['ochl_Avg'],
                    'dtw_distance': dtw_dist
                })

            results.sort(key=lambda x: x['dtw_distance'])
            return results[:nmatches]








