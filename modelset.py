# Set of predictive models for medical data - modelset.py
# Author: BK
# Date: 2023/02/20

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# Create a wrapper class for all models


class ModelSet:

    def __init__(self) -> None:

        # Load models from pickle file
        try:
            self.chd_mdl = pickle.load(open('xgb_chd_model.pkl', 'rb'))
            self.diab_mdl = pickle.load(
                open('xgb_diab_model2_(augumented).pkl', 'rb'))
            self.obes_mdl = pickle.load(open('xgb_obes_model.pkl', 'rb'))
            self.scaler = pickle.load(open('scaler.pkl', 'rb'))

        except:
            print('Error: Models not found')
            raise Exception('Models not found in the current directory')

    def predict(self, X):
        '''
        Predict the output of the model
        Input:
            X: Input data

        Output:
            output: dictionary with likelihoods of the diseases

        Features in X vector should be in the following order:
        Sex
        Age 
        Weight_p 
        Height_p 
        Bmi 
        Smoke_years 
        Smoke_amount_day 
        Chol_all 
        Ldl 
        Hdl 
        Sugar1 
        Sugar2 
        Sugar3 
        Systolic_pressure 
        Diastolic_pressure
        '''

        # Check the correctness of the input data
        # X should be a numpy array
        if not isinstance(X, np.ndarray):
            # Convert to numpy array
            X = np.array(X)

        # Make sure the input data is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Now check the shape/len of the input data - it should have 15 features
        if X.shape[1] != 15:
            raise Exception('Input data should have 15 features')

        # Scale the input data
        X_scaled = self.scaler.transform(X[:, 1:])
        X_scaled = np.hstack((X[:, 0].reshape(-1, 1), X_scaled))

        # Predict the output
        chd_pred = self.chd_mdl.predict(X_scaled)
        diab_pred = self.diab_mdl.predict(X_scaled)
        obes_pred = self.obes_mdl.predict(X_scaled)

        # Create a dictionary with the output
        output = {
            'chd': chd_pred,
            'diabetes': diab_pred,
            'obesity': obes_pred
        }

        return output
