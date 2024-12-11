from flask import Flask, request, jsonify
from flask_restful import  Api, Resource, reqparse
import logging
import requests
from flask_cors import CORS


import threading
class ConnectionBuilder:
    def __init__(self,port=5000):
        self.app1 = Flask(__name__)
        # self.api = Api(self.app1)
        self.start_time = "0.0"
        self.end_time = "0.0"
        self.ninjaData = None
        self.port = port
        self.ninja_port = 52428
        self.time_request =  { "StartTime": self.start_time, "EndTime": self.end_time }
        CORS(self.app1)
        self.routeSetup()

    def format_time(self, minutes):
        """Convert minutes to HH:MM format."""
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours:02d}:{minutes:02d}"
    def routeSetup(self):
        @self.app1.route('/sendTimeToNinja', methods=['POST', 'GET'])
        def sendDataToNinja():
            try:
                # seq_data = requests.post(f"http://127.0.0.1:{self.ninja_port}/receiveTime", json=time_request)
                # seq_data.raise_for_status()
                return jsonify(self.time_request)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app1.route('/receiveData', methods=['POST','GET'])
        def getDataFromNinja():
            try:
                self.ninjaData = request.json
                print(f"ninjaData: {self.ninjaData}")
                return jsonify({"message": "Data received from UI"}), 200

            except Exception as e:
                return Print(f"Error :{e}")

        #receving data from front end
        @self.app1.route('/uiSendData', methods=['POST', 'GET'])
        def getData():
            try:
                    data = request.json
                    self.start_time = data.get('start_time')
                    self.end_time = data.get('end_time')
                    self.start_time = self.format_time(self.start_time)
                    self.end_time = self.format_time(self.end_time)
                    print(f"Data from front end: {self.start_time}, {self.end_time}")

                    # Sending and receiving data from Ninja (POST request to Ninja)
                    self.time_request = {"StartTime": self.start_time, "EndTime": self.end_time}

                    #getDataFromNinja(time_request)  # Get response from Ninja

                    return jsonify({"message": "Data received from UI"}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500


        @self.app1.route('/sendDatatoUI', methods=['POST','GET'])
        def sendData():
            try:
                return jsonify(self.ninjaData), 200
            except Exception as e:
                return jsonify({'error':str(e)}), 500

    def run(self):
        try:
            print(f"Starting Flask applicaiton on http://127.0.0.1:{self.ninja_port}")
            self.app1.run(host='127.0.0.1', port =self.ninja_port, threaded =False, debug=True)
        except Exception as e:
            print(f"Failed to start Flask server: {e}")
#
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--ninja_port', type=int, required=True, help='Port Number for the Flask Server')
    # args = parser.parse_args()
    server = ConnectionBuilder() #ninja_port=args.ninja_port
    server.run()