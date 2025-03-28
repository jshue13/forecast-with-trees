from datetime import timedelta, datetime
from pandas.tseries.offsets import QuarterBegin
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# Function to reformat the time series so it can be used for the Random Forest Model
def reformat_ts(series, time_step_window):
    temp_list = []
    for index, value in series.items():
        time_frame = series[index:index+(time_step_window+1)*QuarterBegin()].tolist()
        if len(time_frame) < time_step_window+1:
            pass
        else: 
            temp_list.append(time_frame)
    return pd.DataFrame(temp_list)


class ForestForecast():
    
    def __init__(self, 
                 train,
                 time_step_window: int, 
                 time_steps_to_forecast: int
                ):
        
        # Parameters
        self.train = train
        self.time_steps_to_forecast = time_steps_to_forecast

        # Get the last number of time steps as inputs for forecasting
        self.X_input = train[-time_step_window:].to_list()
        
        # Reformat the time series
        self.df_reformatted = reformat_ts(train, time_step_window)
        
        # Separate the dependent and independent variables
        self.X = self.df_reformatted.iloc[:,:time_step_window]
        self.y = self.df_reformatted.iloc[:,time_step_window]
        
    def fit(self):
        # Initialize the Random Forest Model
        self.rfm = RandomForestRegressor(random_state=13)
        self.rfm.fit(self.X, self.y)
        return self
    
    def predict(self):
        self.preds = []
        new_X_test = self.X_input
        for i in range(self.time_steps_to_forecast):
            prediction = self.rfm.predict([new_X_test])
            self.preds.append(prediction[0])
            new_X_test = new_X_test[1:]
            new_X_test.append(prediction[0])
        return self.preds