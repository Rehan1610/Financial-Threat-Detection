import os
from flask import Flask, render_template, jsonify
import joblib
import pandas as pd
import random

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'xgboost_fraud_model.joblib')
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'live_test_data.csv')

print("Loading AI Model...")
if not os.path.exists(MODEL_PATH):
    print(f"\n[ERROR] Model file not found at: {MODEL_PATH}")
    exit()

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
