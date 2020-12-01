# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import random

# our code
import utils

from autoregression import LinearModelAutoregressive
from sklearn.metrics import mean_squared_error

def load_dataset(filename):
    return pd.read_csv(os.path.join('..','data',filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "2":
        df = load_dataset("phase2_training_data-modified.csv")
        
        # Hyperparameters
        days = 115
        valid_days = 6
        days_back = 0
        best_lambda = 0.10
        # country_ids = set(df['country_id'].values)
        # country_ids.remove(np.nan)
        # country_ids.remove('CA')
        # print(country_ids)
        country_ids = ['US', 'CN', 'AU', 'FR', 'BR', 'MX', 'NO', 'JP', 'SK', 'IT', 'DE', "DK", 'ES', 'RU', 'IN', 'PK', 'SZ', 'SE', 'ID', 'UK', 'AT', 'SA']
        K = 2
        selected_ids = []
        best_err = 0

        ### Get the average validation score for just using Canadian Data
        for days_back in range(0, 40, 2):
            days_tot = days + days_back
            data_ca = df[df['country_id'] == 'CA'].tail(days_tot).values[:,2:6]
            D = data_ca[0:days-valid_days]
            ### Normalize the data
            means = np.mean(D[:days-valid_days], axis=0)
            stddev = np.std(D[:days-valid_days], axis=0, dtype=np.float64)
            stddev[stddev == 0] = 1
            D_norm = (D - means)/stddev
            ### Get Validation Set/Error
            y_validation = data_ca[days-valid_days + 1:days ]
            model = LinearModelAutoregressive(K=K)
            model.fit(D_norm, best_lambda)
            y_pred = model.predict(start=days-valid_days + 1, end=days)*stddev + means
            best_err = best_err + np.sqrt(mean_squared_error(y_validation[:, 1], y_pred[:, 1]))
        best_err = best_err/20
        print("Just Canada", best_err)

        ### Search and Score
        for i in range(len(country_ids)):
            possible_ids = country_ids
            chosen_id = ""
            for iden in possible_ids:
                err = 0
                for days_back in range(0, 40, 2):
                    days_tot = days + days_back
                    data_ca = df[df['country_id'] == 'CA'].tail(days_tot).values[:,2:6]
                    D = data_ca[0:days-valid_days]
                    ### Add all selected countries
                    for ii in selected_ids:
                        data = df[df['country_id'] == ii].tail(days_tot).values[:,3:6]
                        D = np.concatenate((D, data[0:days-valid_days]), axis=1)
                    ### Add new potential country
                    data = df[df['country_id'] == iden].tail(days_tot).values[:,3:6]
                    D = np.concatenate((D, data[0:days-valid_days]), axis=1)
                    ### Normalize the data
                    means = np.mean(D[:days-valid_days], axis=0)
                    stddev = np.std(D[:days-valid_days], axis=0, dtype=np.float64)
                    stddev[stddev == 0] = 1
                    D_norm = (D - means)/stddev
                    ### Get Validation Set/Error
                    y_validation = data_ca[days-valid_days + 1:days ]
                    model = LinearModelAutoregressive(K=K)
                    model.fit(D_norm, best_lambda)
                    y_pred = model.predict(start=days-valid_days + 1, end=days)*stddev + means
                    err = err + np.sqrt(mean_squared_error(y_validation[:, 1], y_pred[:, 1]))
                err = err/20
                print(selected_ids, iden, err)
                if err < best_err:
                    best_err = err
                    chosen_id = iden
            if chosen_id != "":
                ### Add best country if it helps the score
                selected_ids.append(chosen_id)
                possible_ids.remove(chosen_id)
                print(selected_ids, best_err)
            else:
                break
                # print(y_pred[:, 1])
                # print(y_validation[:, 1])

        days_back = 0
        # selected_ids = ['NO', 'ES', 'CN', 'SK']
        data_ca = df[df['country_id'] == 'CA'].tail(days).values[:,2:6]
        D = data_ca[0:days-valid_days-days_back]
        for ii in selected_ids:
            data = df[df['country_id'] == ii].tail(days).values[:,2:6]
            D = np.concatenate((D, data[0:days-valid_days-days_back]), axis=1)

        means = np.mean(D[:days-valid_days-days_back], axis=0)
        stddev = np.std(D[:days-valid_days-days_back], axis=0, dtype=np.float64)
        stddev[stddev == 0] = 1
        D_norm = (D - means)/stddev
        y_validation = data_ca[days-valid_days-days_back + 1:days - days_back]
        model = LinearModelAutoregressive(K=K)
        model.fit(D_norm, best_lambda)
        y_pred = model.predict(start=days-valid_days - days_back + 1, end=days-days_back)*stddev + means
        err = np.sqrt(mean_squared_error(y_validation[:, 1], y_pred[:, 1]))

        print("Chosen:", selected_ids, err)
        print('Validation:', y_validation[:, 1])
        print('Prediction:', y_pred[:, 1])
        fname = os.path.join("..", "figs", "validation.png")
        plt.plot(np.arange(0, 5), y_pred[:, 1])
        plt.plot(np.arange(0, 5), y_validation[:, 1])
        plt.title("Error")
        plt.legend(["Predicted", "Actual"])
        plt.xlabel("Date")
        plt.ylabel("# Deaths")
        plt.savefig(fname)
    else:
        print("Unknown question: %s" % question)