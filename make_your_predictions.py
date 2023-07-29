import pandas as pd
import dill
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

import dill
def calc_BMI(X):
    return X[:, [0]] / (X[:, [1]] / 100) ** 2

def BMI_cat(X):
    df = calc_BMI(X)
    df = pd.DataFrame(df, columns=['BMI'])
    df['BMI_cat'] = pd.cut(df['BMI'],
                           bins=[0, 18.5, 25, 30, 35, 40, np.inf],
                           labels=['Underweight', 'Normal', 'Overweight', 'Obesity_Class_I',
                                   'Obesity_Class_II', 'Obesity_Class_III'])
    return df[['BMI_cat']]

def calc_BMI_pl():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(calc_BMI, feature_names_out=lambda _, __: ['BMI']),
        MinMaxScaler()
    )

def calc_BMI_cat_pl():
    return make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        FunctionTransformer(BMI_cat, feature_names_out=lambda _, __: ['BMI_cat']),
        OneHotEncoder(handle_unknown='ignore')
    )


def workload_calc(X):
    return X[:, [0]] * X[:, [1]]

def calc_work_load_pl():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(workload_calc, feature_names_out=lambda _, __: ['workload']),
        MinMaxScaler()
    )

cat_pl_rem = make_pipeline(
    OneHotEncoder(handle_unknown='ignore')
)

log_pl = make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(np.log, inverse_func=np.exp, feature_names_out='one-to-one'),
    MinMaxScaler()
)

def_num_pl = make_pipeline(
    SimpleImputer(strategy='median'),
    MinMaxScaler()
)


preprocessor = ColumnTransformer([
    ('bmi_calculator', calc_BMI_pl(), ['Weight', 'Height']),
    ('bmi_cat', calc_BMI_cat_pl(), ['Weight', 'Height']),
    ('cat_enc_rem', cat_pl_rem, ['Gender']),
    ('workload_calc', calc_work_load_pl(), ['Heart_Rate', 'Duration']),
    ('log', log_pl, ['Age']),
    ('num', def_num_pl, ['Weight', 'Height', 'Duration', 'Heart_Rate', 'Body_Temp'])
])


def get_input():

    user_input = {}
    gender = input('Enter you gender: ')
    age = float(input('Enter your age: '))
    height = float(input('Enter you height: '))
    weight = float(input('Enter you weight: '))
    duration = float(input('Enter the workout duration in minutes: '))
    heart_rate = float(input('Enter your heart rate during training: '))
    body_temp = float(input('Enter your body temperature during training: '))

    user_input['Gender'] = gender
    user_input['Age'] = age
    user_input['Height'] = height
    user_input['Weight'] = weight
    user_input['Duration'] = duration
    user_input['Heart_Rate'] = heart_rate
    user_input['Body_Temp'] = body_temp
    user_input_df = pd.DataFrame([user_input])

    return user_input_df


if __name__ == '__main__':
    
    with open('/home/elneklawy/Desktop/New Folder/final_model.pkl', 'rb') as f:
        loaded_model = dill.load(f)
    input = get_input()
    print(input)
    print(f"The estimated number of calories burnt after the training is: {loaded_model.predict(input)[0].round(2)}.")