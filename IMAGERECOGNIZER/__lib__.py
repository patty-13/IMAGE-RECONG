# Basic Python libraries
import os
import json
import math
import random
from pathlib import Path
from datetime import time
import warnings
import itertools
import ast
import pickle
import io
import faiss
import gc
from flask import Flask, request, jsonify
from flask_restful import  Api, Resource, reqparse
import logging
import requests
import threading
# import eventlet
# from eventlet import wsgi
from multiprocessing import Process
import multiprocessing
from subprocess import Popen


# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.graph_objects as go
import plotly.io as pio
from pyts.image import GramianAngularField
from skimage.transform import resize
from io import BytesIO
from datetime import datetime

# Machine learning and preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, RFE
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Pytorch and utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torchvision import models, transforms

# Progress bar utilities
from tqdm import tqdm

# Joblib for model saving/loading
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')


# UI creation libraries
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtQml import QQmlApplicationEngine
from PyQt6.QtQuick import QQuickWindow
from PyQt6.QtQuick import QQuickView


# from kivy.app import App
# from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.button import Button
# from kivy.uix.label import Label
# from kivy.uix.spinner import Spinner
# from kivy.uix.image import Image
# from kivy.uix.floatlayout import FloatLayout
# from kivy.uix.gridlayout import GridLayout
# from kivy.uix.pagelayout import PageLayout
#
#
#
# from kivymd.app import MDApp
# from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.gridlayout import GridLayout
# from kivymd.uix.button import MDRaisedButton
# from kivymd.uix.label import MDLabel
# from kivymd.uix.slider import MDSlider
# from kivymd.uix.textfield import MDTextField
# from kivy.metrics import dp
# from kivymd.uix.datatables import MDDataTable
# from kivymd.uix.screen import MDScreen

# import faiss
import argparse