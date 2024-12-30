import argparse

import pandas as pd
from fastdtw import fastdtw as dtw
from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
import logging
import requests
import numpy as np
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import faiss
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import sys
#from pathlib import Path


# SEQUENCE FINDER
class sequenceFinder:
    def __init__(self, testSeq, timeFrame, startTime):
        self.testSeq = testSeq

        self.timeFrame = timeFrame
        self.startTime = startTime
        self.SeqLen = 0
        self.splitVal = 0
        self.window = list(range(31, 961))
        self.w1 = 0.3
        self.w2 = 0.7
        self.ST = None
        self.ET = None

        # if getattr(sys, 'frozen', False):  # Running as a PyInstaller executable
        #     self.base_dir = os.path.dirname(sys._MEIPASS)
        # else:  # Running as a Python script
        #     self.base_dir = os.path.dirname(__file__).parent.resolve()
        #
        # self.data_dir = self.base_dir / "STORE_DATA"
        # self.data_file = self.base_dir / "DATA_FILE"
        # self.plot_dir = self.base_dir / "PLOT_IMAGES"
        # self.img_data = self.base_dir / "IMG_DATA"
        #
        # # Create directories if they do not exist
        # self.data_dir.mkdir(parents=True, exist_ok=True)
        # self.data_file.mkdir(parents=True, exist_ok=True)
        # self.plot_dir.mkdir(parents=True, exist_ok=True)
        # self.img_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def weightSeq(self, normSeq):
        if self.timeFrame == 1:
            self.splitVal = int(self.startTime * 60)
        elif self.timeFrame == 5:
            self.splitVal = int(self.startTime * 60 // 5)
        elif self.timeFrame == 10:
            self.splitVal = int(self.startTime * 60 // 10)

        print(f"split value : {self.splitVal}")
        pre_seq = normSeq[:self.splitVal]
        post_seq = normSeq[self.splitVal:]
        NFS_pre = pd.Series(pre_seq) * self.w1
        NFS_post = pd.Series(post_seq) * self.w2
        norm_weighted_full_seq = pd.concat([NFS_pre, NFS_post]).reset_index(drop=True)
        return norm_weighted_full_seq

    def pad_series(self, series, target_length):
        current_length = len(series)
        if current_length >= target_length:
            return series
        pad_value = np.mean(series)
        padding = pd.Series([pad_value] * (target_length - current_length), index=range(current_length, target_length))
        return pd.concat([series, padding])

    def seqCheck(self):
        closest_len = min(self.window, key=lambda x: abs(len(self.testSeq) - x))
        print(f"closest_len:{closest_len}")
        if closest_len == len(self.testSeq):
            return self.testSeq, len(self.testSeq)
        else:
            padded_seq = self.pad_series(self.testSeq, closest_len)
            return padded_seq, len(padded_seq)

    def normalize_seq(self, seq):
        seq = np.array(seq)
        seq_reshaped = seq.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_seq = scaler.fit_transform(seq_reshaped)
        return scaled_seq.flatten()

    def seqMatches(self, nmatches, use_DTW=False):
        # checking seq length
        self.testSeq, self.SeqLen = self.seqCheck()

        # normalzing seq
        norm_test_seq = self.normalize_seq(self.testSeq)
        norm_test_seq = self.weightSeq(norm_test_seq)
        print(f"norm_test_seq 2 : {norm_test_seq}")

        if isinstance(norm_test_seq, pd.Series):
            norm_test_seq = norm_test_seq.to_numpy()
        elif isinstance(norm_test_seq, list):
            norm_test_seq = np.array(norm_test_seq)

        # setting up the path from where the sequences will be mapped.
        if self.SeqLen >= self.window[-1]:
            return "LENGTH TOO BIG AND NO SEQ LENGTH THIS BIG IN DATABASE"
        else:
            try:

                # stored_seq_file = os.path.join( self.data_dir, f'NQ_{self.timeFrame}M/df_{self.timeFrame}M_{self.startTime}_{self.SeqLen}_seq.csv')
                # faiss_index_file = os.path.join(self.data_dir, f'NQ_{self.timeFrame}M/faiss_index_{self.timeFrame}M_{self.startTime}_{self.SeqLen}.index')
                # meta_data_file = os.path.join(self.data_dir, f'NQ_{self.timeFrame}M/metadata_{self.timeFrame}M_{self.startTime}_{self.SeqLen}.csv')

                stored_seq_file =  f'D:/STORE_DATA/NQ_{self.timeFrame}M/df_{self.timeFrame}M_{self.startTime}_{self.SeqLen}_seq.csv'
                faiss_index_file = f'D:/STORE_DATA/NQ_{self.timeFrame}M/faiss_index_{self.timeFrame}M_{self.startTime}_{self.SeqLen}.index'
                meta_data_file = f'D:/STORE_DATA/NQ_{self.timeFrame}M/metadata_{self.timeFrame}M_{self.startTime}_{self.SeqLen}.csv'
                print(stored_seq_file)
                print(faiss_index_file)
                print(meta_data_file)
            except Exception as e:
                print(f"Error : {e}")

        if use_DTW:
            results = []
            stored_seq_file = pd.read_csv(stored_seq_file)

            for idx, row in stored_seq_file.iterrows():
                stored_seq = np.array(eval[row['ochlv_avg_norm']])
                distance, _ = dtw(norm_test_seq, stored_seq)
                results.append({
                    'date': row['date'],
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'ochl_avg': row['ochl_Avg'],
                    'dtw_distance': distance,
                    'timeFrame': self.timeFrame,
                })
            results.sort(key=lambda x: x['dtw_distance'])
            return results[:nmatches]
        else:
            index = faiss.read_index(faiss_index_file)
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
                    'dtw_distance': dtw_dist,
                    'timeFrame': self.timeFrame,
                })

            results.sort(key=lambda x: x['dtw_distance'])
            return results[:nmatches]

    def store_plots(self, data, matchNumber):
        row_data = []
        plot_dir = r'D:\pycharm_projects\CLEARAIMODELS\IMAGERECOGNIZER\PLOT_IMAGE_1'
        # dataFile = os.path.join(self.data_file, f'@ENQ_M{self.timeFrame}.csv')

        # complete_chart = pd.read_csv(dataFile, sep='\t').pipe(self.preprocess)
        complete_chart= pd.read_csv('D:/pycharm_projects/CLEARAIMODELS/IMAGERECOGNIZER/@ENQ_M1.csv', sep='\t').pipe(self.preprocess)
        for i, entry in enumerate(data[:matchNumber]):  # Top 5 results

            # defining paths
            match_plot_path = os.path.join(plot_dir, f'Matching_Plot_{i + 1}.png')
            complete_plot_path = os.path.join(plot_dir, f'Complete_Plot_{i + 1}.png')

            ## debugger statement
            self.ST = entry['start_time']
            self.ET = entry['end_time']
            ochl_avg_list = eval(entry['ochl_avg'])
            print(f"DATE: {entry['date']}")
            print(f"ST : {entry['start_time']}")
            print(f"ET : {entry['end_time']}")
            print(f"DTW : {entry['dtw_distance']}")

            if self.timeFrame == 1:
                self.splitVal = int(self.startTime * 60)
            elif self.timeFrame == 5:
                self.splitVal = int(self.startTime * 60 // 5)
            elif self.timeFrame == 10:
                self.splitVal = int(self.startTime * 60 // 10)

            x_values = range(self.splitVal, len(ochl_avg_list))
            y_values = ochl_avg_list[self.splitVal:]

            #saving subplots
            self.createSubPlots(x_values, y_values, match_plot_path)
            # saving complete chart
            print(f"COMPLTE_CHART : {complete_chart}")
            sub_chart = complete_chart[complete_chart['date'] == entry['date']].reset_index(drop=True)
            print(f"SUB CHART : {sub_chart}")
            self.createCompletePlot(sub_chart, complete_plot_path)


            # Information column
            info = (
                f"Date: {entry['date']}\n"
                f"Start Time: {entry['start_time']}\n"
                f"End Time: {entry['end_time']}\n"
                f"DTW Distance: {entry['dtw_distance']:.6f}"
            )

            # Add row
            row_data.append(
                [
                    f"{complete_plot_path}",
                    f"{match_plot_path}",
                    info
                ]
            )
        # row_data_df = pd.DataFrame(
        #     row_data,
        #     columns=["Complete Plot Path", "Match Plot Path", "Info"]
        # )
        # row_data_df.to_csv(os.path.join(self.plot_dir, 'image_data.csv'), index=False)

    def createSubPlots(self, x_data, y_data, plot_path):
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.plot(x_data, y_data, label='ochl_avg')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.axis('off')
        plt.savefig(plot_path)

    def createCompletePlot(self, completeChart, plot_path):
        fig = go.Figure(data=[go.Candlestick(x=completeChart['time'],
                                             open=completeChart['open'],
                                             high=completeChart['high'],
                                             low=completeChart['low'],
                                             close=completeChart['close'])])

        fig.update_layout(
            shapes=[
                dict(
                    type="rect",
                    x0=self.ST,  # Set start time
                    x1=self.ET,  # Set end time
                    y0=0,  # Y-axis start (you can adjust this depending on your data)
                    y1=1,  # Y-axis end (usually the range for the candlestick data)
                    xref="x",  # Use x-axis scale
                    yref="paper",  # Y-axis is defined from 0 to 1 (height of the chart)
                    fillcolor="rgba(0, 100, 255, 0.3)",  # Set color of the bounding box
                    line=dict(color="rgba(0, 100, 255, 0.5)", width=2)  # Set border of the bounding box
                )
            ],
            xaxis_rangeslider_visible=False,  # Disable the range slider

            width=800,  # Set the width of the chart
            height=600  # Set the height of the chart
        )
        fig.write_image(plot_path)


# FLASK CONNECTION
class ConnectionBuilder:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.startTime = "00:00:00"
        self.endTime = "00:00:00"
        self.TF = None
        self.testdata = None
        self.seqData = None
        self.routeSetup()
        self.nmatches = 5
        self.data = pd.DataFrame()

    def routeSetup(self):
        @self.app.route('/getData', methods=['POST', 'GET'])
        def getDataFromFrontEnd():
            try:
                # getting data from frontend
                self.testData = request.json
                self.data = pd.DataFrame(self.testData)
                self.data['ochl_avg'] = (self.data['open'] + self.data['high'] + self.data['close'] + self.data['low']) / 4

                self.seqData = self.data['ochl_avg']
                print(self.seqData)

                self.TF = self.data['timeFrame'][0]
                print(f"TIME FRAME : {self.TF}")
                self.startTime = 5  # self.data['startTime']
                # preprocessing the data
                seqMatcher = sequenceFinder(self.seqData, self.TF, self.startTime)
                self.seqData = seqMatcher.seqMatches(self.nmatches, False)
                print(f"SEQ DATA: {self.seqData}")
                seqMatcher.store_plots(self.seqData, self.nmatches)
                return jsonify("got data"), 200




            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def run(self, PORT):
        try:
            print(f"Flask Start: http://127.0.0.1:{PORT}")
            self.app.run(host='127.0.0.1', port=PORT, threaded=False, debug=True)
        except Exception as e:
            print(f"Error: Failed to start Flask server :{e}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--Port', type=int, required=True, help='Port Number for Flask Server')
    # args = parser.parse_known_args()
    server = ConnectionBuilder()
    server.run(59044) #args.Port
