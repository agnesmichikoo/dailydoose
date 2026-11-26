"""Shared constants and data loading for the student stress app."""
import pandas as pd

DATA_PATH = "student_lifestyle_dataset.csv"
TARGET = "Stress_Level"
CLASSES = ["Low", "Moderate", "High"]
FEATURES_A = [
    "Study_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "Social_Hours_Per_Day",
]
FEATURES_B = ["Physical_Activity_Hours_Per_Day", "Social_Hours_Per_Day"]
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Student_ID", "GPA", "Extracurricular_Hours_Per_Day"])
    return df
