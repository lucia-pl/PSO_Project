"""
@file        save_data.py
@author      Lucía Parreño Legorburo
@brief       Methods to save in CSV and JSON files the results obtained in the trials, 
                they update or create (if it does not exists) the files to use them later for visualization
"""

import os

import csv
import json

from datetime import datetime
from experiments.gridsearch import extract_best
from src.data.format import to_python_type

FUNCTIONS = ["sphere", "rastrigin", "rosenbrock", "ackley"]


def data_csv(pso_results):
    """
    Used to save in a csv file the best results of the PSO for each function. 
    CSV was chosen as an easy file type to use with libraries like pandas, not as a way to easy read for human eye.

    Declares the route where the file will be saved and the named that will be used to save them, if the file already exists
    the new information will be added else it will be created from scrath adding the headers and the current values.

    Args:
        pso_results (list[dict])
    """

    file_path = os.path.join("src","io", "resultsCSV.csv")
    file_exists = os.path.isfile(file_path)

    # As a safety measure the directory will be checked and created if it doesn't exist
    os.makedirs("src/io", exist_ok=True)

    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Function",
                "Best_Fitness",
                "Best_Position",
                "w",
                "c1",
                "c2"
            ])

        for func in FUNCTIONS:
            best = extract_best(pso_results, func)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            entry = [
                timestamp,
                func,
                to_python_type(best["best_fitness"]),
                to_python_type(best["w"]),
                to_python_type(best["c1"]),
                to_python_type(best["c2"])
            ]

            writer.writerow(entry)


def data_json(pso_results):
    """
    Used to save in a json file the best results of the PSO for each function. 
    JSON was chosen as a way for easy reading for the human eye considering its separation between entries.

    Declares the route where the file will be saved and the named that will be used to save them, if the file already exists
    the new information will be added else it will be created from scrath adding the headers and the current values.

    Args:
        pso_results (list[dict])
    """

    file_path = os.path.join("src","io", "resultsJSON.json")

    # As a safety measure the directory will be checked and created if it doesn't exist
    os.makedirs("src/io", exist_ok=True)

    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []


    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for func in FUNCTIONS:
        best = extract_best(pso_results, func)

        entry = {
            "timestamp": timestamp,
            "function": func,
            "best_fitness": to_python_type(best["best_fitness"]),
            "w": to_python_type(best["w"]),
            "c1": to_python_type(best["c1"]),
            "c2": to_python_type(best["c2"])
        }

        data.append(entry)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
