import numpy as np
from os.path import join as pjoin, dirname

import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


class Descriptive:
    def __init__(self, mouse_name):
        exp_data = pd.read_csv('exp_data.csv')
        self.mouse_data = exp_data[exp_data['mouse_name'] == mouse_name]

    def train(self, mouse_data):
        # training data
        train_X = []
        train_y = []

        for date in mouse_data['date'].unique():
            trial_data = mouse_data[mouse_data['date'] == date]
            trialresponseside = np.array(self.remove_nans(trial_data['trialresponseside']))
            trialreward = np.array(self.remove_nans(trial_data['trialreward']))

            trialresponseside_right = trialresponseside
            trialresponseside[trialresponseside_right == -1] = 0

            right_prob = np.convolve(trialresponseside_right, np.ones(21) / 21, 'same')

            for i in range(15, len(trialreward)):
                rewards_temp = []
                choice_temp = []
                for j in range(15):
                    if trialreward[i - j] == 1:
                        rewards_temp.append(1)
                    else:
                        rewards_temp.append(-1)
                    if trialresponseside[i-j] == 1:
                        choice_temp.append(1)
                    else:
                        choice_temp.append(-1)
                train_X.append(rewards_temp + choice_temp + [1])

            train_y = np.concatenate([train_y, right_prob[15:]])

        model = LinearRegression()
        model.fit(train_X, train_y)
        weights = np.array(model.coef_).flatten()
        return weights

    def plot_weights(self, weights):
        choices_coef = weights[:15]
        reward_coef = weights[15:30]
        plt.plot(range(15), reward_coef)
        plt.show()

    def remove_nans(self, np_array):
        return np_array[~np.isnan(np_array)]


fig, ax = plt.subplots(1, 1, figsize=(5,5))
weights_sum = [0] * 10
for mouse in ['AKED01', 'AKED02', 'AKED04', 'AKED05', 'AKED06']:
    desc = Descriptive(mouse)
    weights = desc.train(desc.mouse_data)
    ax.plot(range(10), weights[15:25], c='gray')
    weights_sum = [sum(x) for x in zip(weights_sum, weights[15:30])]

ax.plot(range(10), np.array(weights_sum)/5, c='black')
fig.show()