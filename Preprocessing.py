import pandas as pd
from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample, shuffle
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split

# def map_waiting_interval_to_days(x):
#     if x == 0:
#         return 'Less than 15 days'
#     elif 0 < x <= 2:
#         return 'Between 1 day and 2 days'
#     elif 2 < x <= 7:
#         return 'Between 3 days and 7 days'
#     elif 7 < x <= 31:
#         return 'Between 7 days and 31 days'
#     else:
#         return 'More than 1 month'
# def map_age(x):
#     x = int(x)
#     if x < 12:
#         return 'Child'
#     elif 12 <= x < 18:
#         return 'Teenager'
#     elif 20 <= x < 25:
#         return 'Young Adult'
#     elif 25 <= x < 60:
#         return 'Adult'
#     else:
#         return 'Senior'
# def process_date_columns(data):

#     d = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
        
#         # data['mapped_AppointmentDay'] = data['AppointmentDay'].map(lambda x: datetime.strptime(x,"%Y-%m-%dT%H:%M:%SZ"))
#         # data['mapped_ScheduledDay'] = data['ScheduledDay'].map(lambda x: datetime.strptime(x,"%Y-%m-%dT%H:%M:%SZ"))

#         # Convert date columns to datetime if not already
#     if data['AppointmentDay'].dtype == 'object':
#         data['mapped_AppointmentDay'] = pd.to_datetime(data['AppointmentDay'], format="%Y-%m-%dT%H:%M:%S")
#     else:
#         data['mapped_AppointmentDay'] = data['AppointmentDay']

#     if data['ScheduledDay'].dtype == 'object':
#         data['mapped_ScheduledDay'] = pd.to_datetime(data['ScheduledDay'], format="%Y-%m-%dT%H:%M:%S")
#     else:
#         data['mapped_ScheduledDay'] = data['ScheduledDay']

#     data['mapped_AppointmentDay'] = data['mapped_AppointmentDay'] + pd.Timedelta('1d') - pd.Timedelta('1s')
    
    
#     data['waiting_interval'] = abs(data['mapped_ScheduledDay'] - data['mapped_AppointmentDay'])
#     data['waiting_interval_days'] = data['waiting_interval'].map(lambda x: x.days)
#     data['waiting_interval_days'] = data['waiting_interval_days'].map(lambda x: map_waiting_interval_to_days(x))
#     data['ScheduledDay_month'] = data['mapped_ScheduledDay'].map(lambda x: x.month)
#     data['ScheduledDay_day'] = data['mapped_ScheduledDay'].map(lambda x: x.day)
#     data['ScheduledDay_weekday'] = data['mapped_ScheduledDay'].map(lambda x: x.weekday())
#     data['ScheduledDay_weekday'] = data['ScheduledDay_weekday'].replace(d)
        
#     data['AppointmentDay_month'] = data['mapped_AppointmentDay'].map(lambda x: x.month)
#     data['AppointmentDay_day'] = data['mapped_AppointmentDay'].map(lambda x: x.day)
#     data['AppointmentDay_weekday'] = data['mapped_AppointmentDay'].map(lambda x: x.weekday())
#     data['AppointmentDay_weekday'] = data['AppointmentDay_weekday'].replace(d)

    
#     if 'No-show' in data.columns:
#         data['No-show'] = data['No-show'].replace({'Yes':1, 'No':0})
#     else:
#         pass

#     data['mapped_Age'] = data['Age'].map(lambda x: map_age(x))
#     data['Gender'] = data['Gender'].replace({'F':0, 'M':1})

#     # Check and convert disease columns to int if necessary
#     for col in ['Alcoholism', 'Handcap', 'Diabetes', 'Hipertension']:
#         if data[col].dtype != int:
#             data[col] = data[col].astype(int)

#     data['haveDisease'] = data.Alcoholism | data.Handcap | data.Diabetes | data.Hipertension
#     # data = data.drop(columns=[
#     #     'waiting_interval', 'AppointmentDay', 'ScheduledDay', 'PatientId', 'Age', 
#     #     'mapped_ScheduledDay', 'mapped_AppointmentDay', 'AppointmentID', 
#     #     'Alcoholism', 'Handcap', 'Diabetes', 'Hipertension', 'ScheduledDay_month', 'AppointmentDay_month'
#     # ])
    
#     return data

# def final_processing(data):
#     data = data.drop(columns=[
#         'waiting_interval', 'AppointmentDay', 'ScheduledDay', 'PatientId', 'Age', 
#         'mapped_ScheduledDay', 'mapped_AppointmentDay', 'AppointmentID', 
#         'Alcoholism', 'Handcap', 'Diabetes', 'Hipertension', 'ScheduledDay_month', 'AppointmentDay_month'
#     ])
#     # data['No-show'] = data['No-show'].replace({'Yes': 1, 'No': 0})
#     return data

class AppointmentNoShowPredictor:
    def __init__(self, model=None, original_columns=None):
        self.model = model
        self.original_columns = original_columns

    def map_waiting_interval_to_days(self, x):
        '''
        Receives an integer representing the number of days until an appointment and
        returns the category it is in.
        '''
        if x ==0 :
            return 'Less than 15 days'
        elif x > 0 and x <= 2:
            return 'Between 1 day and 2 days'
        elif x > 2 and x <= 7:
            return 'Between 3 days and 7 days'
        elif x > 7 and x <= 31:
            return 'Between 7 days and 31 days'
        else:
            return 'More than 1 month'

    def map_age(self, x):
        '''
        Receives an integer and returns the age category that this age is in.
        '''
        x = int(x)
        
        if x < 12:
            return 'Child'
        elif x > 12 and x < 18:
            return 'Teenager'
        elif x>=20 and x<25:
            return 'Young Adult'
        elif x>=25 and x<60:
            return 'Adult'
        else:
            return 'Senior'

    def process_data(self, data):

        '''
        Receives the dataset, clean data and engineer new features. 
        Return cleaned dataset with features that will be used for training model.
        '''
        d = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
        
        # data['mapped_AppointmentDay'] = data['AppointmentDay'].map(lambda x: datetime.strptime(x,"%Y-%m-%dT%H:%M:%SZ"))
        # data['mapped_ScheduledDay'] = data['ScheduledDay'].map(lambda x: datetime.strptime(x,"%Y-%m-%dT%H:%M:%SZ"))

        # Convert date columns to datetime if not already
        if data['AppointmentDay'].dtype == 'object':
            data['mapped_AppointmentDay'] = pd.to_datetime(data['AppointmentDay'], format="%Y-%m-%dT%H:%M:%S")
        else:
            data['mapped_AppointmentDay'] = data['AppointmentDay']
    
        if data['ScheduledDay'].dtype == 'object':
            data['mapped_ScheduledDay'] = pd.to_datetime(data['ScheduledDay'], format="%Y-%m-%dT%H:%M:%S")
        else:
            data['mapped_ScheduledDay'] = data['ScheduledDay']

        data['mapped_AppointmentDay'] = data['mapped_AppointmentDay'] + pd.Timedelta('1d') - pd.Timedelta('1s')
        data['waiting_interval'] = abs(data['mapped_ScheduledDay'] - data['mapped_AppointmentDay'])
        data['waiting_interval_days'] = data['waiting_interval'].map(lambda x: x.days)
        data['waiting_interval_days'] = data['waiting_interval_days'].map(lambda x: self.map_waiting_interval_to_days(x))
        
        data['ScheduledDay_month'] = data['mapped_ScheduledDay'].map(lambda x: x.month)
        data['ScheduledDay_day'] = data['mapped_ScheduledDay'].map(lambda x: x.day)
        data['ScheduledDay_weekday'] = data['mapped_ScheduledDay'].map(lambda x: x.weekday())
        data['ScheduledDay_weekday'] = data['ScheduledDay_weekday'].replace(d)
        
        data['AppointmentDay_month'] = data['mapped_AppointmentDay'].map(lambda x: x.month)
        data['AppointmentDay_day'] = data['mapped_AppointmentDay'].map(lambda x: x.day)
        data['AppointmentDay_weekday'] = data['mapped_AppointmentDay'].map(lambda x: x.weekday())
        data['AppointmentDay_weekday'] = data['AppointmentDay_weekday'].replace(d)

        
        if 'No-show' in data.columns:
            data['No-show'] = data['No-show'].replace({'Yes':1, 'No':0})
        else:
            pass

        data['mapped_Age'] = data['Age'].map(lambda x: self.map_age(x))
        data['Gender'] = data['Gender'].replace({'F':0, 'M':1})

        # Check and convert disease columns to int if necessary
        for col in ['Alcoholism', 'Handcap', 'Diabetes', 'Hipertension']:
            if data[col].dtype != int:
                data[col] = data[col].astype(int)

        data['haveDisease'] = data.Alcoholism | data.Handcap | data.Diabetes | data.Hipertension
    
        data = data.drop(columns=['waiting_interval', 'AppointmentDay', 'ScheduledDay',
                                 'PatientId','Age', 'mapped_ScheduledDay',
                                 'mapped_AppointmentDay', 'AppointmentID', 
                                  'Alcoholism','Handcap','Diabetes','Hipertension', 'ScheduledDay_month', 'AppointmentDay_month'])
    
        data = pd.get_dummies(data)

        # Ensure columns match the original columns used for training
        data = data.reindex(columns=self.original_columns, fill_value=0)
        return data
        
    def fit(self,data):

        data = self.process_data(data)

        X = data.drop(columns=['No-show'])  
        y = data['No-show']  

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # Store the original columns for consistent encoding
        self.original_columns = X.columns
        
        # Handle imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
  


        # Check the distribution of the target after SMOTE
        print(f"Original dataset shape: {y_train.value_counts()}")
        print(f"Resampled dataset shape: {y_resampled.value_counts()}")


        
        # Initialize and train the model
        self.model = XGBClassifier(n_jobs=-1)
        self.model.fit(X_resampled, y_resampled)

    def predict_csv(self, data):
        processed_input = self.process_data(data)
        
        # Make a prediction
        return self.model.predict(processed_input)

