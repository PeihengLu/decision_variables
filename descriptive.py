import numpy as np
from os.path import join as pjoin, dirname

import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


class Descriptive:
    """load the .mat file and reform it into the Dataframe we wanted
    :parameter data_file: the path pointing towards the .mat file """

    def __init__(self, data_file):
        data_dict = loadmat(pjoin('alltaskinfo', data_file))
        trialresponseside = np.array(data_dict['trialresponseside'])
        trialreward = np.array(data_dict['trialreward'])
        # remove nan trials
        self.trialresponseside = self.remove_nans(trialresponseside)
        self.trialreward = self.remove_nans(trialreward)
        # only save rightward response
        self.trialresponseside_right = self.trialresponseside
        self.trialresponseside_right[self.trialresponseside_right == -1] = 0

    def train(self):
        right_prob = np.convolve(self.trialresponseside_right, np.ones(21)/21, 'same')
        right_prob = pd.DataFrame(right_prob).apply(lambda x: np.log(x/(1-x)))

        # training data
        train = []

        for i in range(15, len(self.trialreward)):
            rewards_temp = []
            choice_temp = []
            for j in range(15):
                if self.trialreward[i - j] == 1:
                    rewards_temp.append(1)
                else:
                    rewards_temp.append(-1)
                if self.trialresponseside[i-j] == 1:
                    choice_temp.append(1)
                else:
                    choice_temp.append(-1)
            train.append(rewards_temp + choice_temp + [1])
        X = np.array(train)
        y = np.array(right_prob[15:])
        assert X.shape[1] == 31, 'training set initialization error'

        model = LogisticRegression()
        model.fit(X, y)
        self.weights = np.array(model.coef_).flatten()

    def plot_weights(self):
        choices_coef = self.weights[:15]
        reward_coef = self.weights[15:30]
        fig, axes = plt.subplots(1, 2)
        plt.tight_layout()
        axes[0].plot(range(15), choices_coef)
        axes[1].plot(range(15), reward_coef)
        plt.show()

    def remove_nans(self, np_array):
        return np_array[~np.isnan(np_array)]

    def log_odds(self, x):
        return np.log(x / (1-x))
