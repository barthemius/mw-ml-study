import numpy as np
import pandas as pd


def preprocess_pipe(data):

    # Create a copy of data object
    data = data.copy()

    data.columns = data.columns.str.capitalize()
    data.columns = data.columns.str.replace(
        'Diastolic_presurre', 'Diastolic_pressure')

    # Map data to English names for consistency

    data['Sex'] = data['Sex'].map({"Kobieta": "F", "Mężczyzna": "M"})
    data['Smoke'] = data['Smoke'].map({"Tak": 1, "Nie": 0})

    # Remove Nans in Smoke_years and Smoke_amount_day columns

    data['Smoke_years'] = data['Smoke_years'].fillna(0)
    data['Smoke_amount_day'] = data['Smoke_amount_day'].fillna(0)

    data["Sex"] = data["Sex"].apply(lambda x: 1 if x == "M" else 0)

    return data


def feature_extraction(data):

    # X1 - without Smoke column - just years and amount per day
    X1 = data[
        [
            'Sex',
            'Age',
            'Weight_p',
            'Height_p',
            'Bmi',
            # 'Smoke',
            'Smoke_years',
            'Smoke_amount_day',
            'Chol_all',
            'Ldl',
            'Hdl',
            'Sugar1',
            'Sugar2',
            'Sugar3',
            'Systolic_pressure',
            'Diastolic_pressure'
        ]
    ]

    X1 = X1.to_numpy()

    y_obes = data['Likelihood_of_obesity'].to_numpy()
    y_diab = data['Likelihood_of_diabetes'].to_numpy()
    y_chd = data['Likelihood_of_coronary_heart_disease'].to_numpy()

    return X1, y_obes, y_diab, y_chd
