from typing import Union, Literal
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
import statsmodels.api as sm
from .utils import transform_ts, estimate_seasonal_length

class ForestForecast(BaseEstimator, RegressorMixin):
    """
    A time series forecasting model using Random Forest Regression.
    
    Parameters
    ----------
    approach : Literal["value", "linear_detrend", "resid", "linear_detrend_resid"], default="linear_detrend"
        There are four ways to represent the values of the explanatory 
        and dependent variables that will be used to train the forest. 

        - value: uses the the raw values. When this option is chosen, 
        the forecasted values will be contained within the range of the 
        dependent variables, so do not use this option if your data has 
        trends where you expect the values to continue to increase or 
        decrease when forecasting the future.

        - linear_detrend: default option of the tool. This option performs a 
        first-order (linear) trend removal on the entire time series, and 
        these detrended values are used as the explanatory and dependent variables. 
        Using this option allows the forecasts to follow this trend into the 
        future so that the forecasted values can be estimated outside of the 
        range of the dependent variables.

        - resid: creates an ordinary least-squares (OLS) regression model 
        to predict the dependent variable based on the explanatory variables 
        within each time window. The residual of this regression model 
        (the difference between the OLS prediction and the raw value 
        of the dependent variable) is used to represent the dependent 
        variable when training the forest. 

        - linear_detrend_resid: performs a first-order (linear) trend removal 
        on the entire time series at a location. It then builds an OLS regression 
        model to predict the detrended dependent variable based on the detrended 
        explanatory variables within each time window. The residual of this 
        regression model is used to represent the dependent variable when 
        training the forest. 

    time_step_window : Union[int, Literal["auto"]], default="auto"
        The number of previous time steps to use as features.
        Otherwise, it uses spectral density to estimate the optimal season length.

    regressor_params : dict, default=None
        Parameters to pass to the RandomForestRegressor

    random_state : int, default=None
        Controls the randomness of the estimator
        
    Attributes
    ----------
    model_ : RandomForestRegressor
        The fitted random forest model
        
    X_transformed_ : pd.DataFrame
        The transformed time series data
    """
    
    def __init__(
        self,
        approach: Literal["value", "linear_detrend", "resid", "linear_detrend_resid"] = "linear_detrend",
        time_step_window: Union[int, Literal["auto"]] = "auto", 
        regressor_params: Union[dict, None] = None,
        random_state: int = 13
    ):
        self.approach = approach
        self.max_window_ratio = 1/3
        self.time_step_window = time_step_window
        self.regressor_params = regressor_params
        self.random_state = random_state
    
    def fit(self, X, y=None):
        """
        Fit the forecasting model.
        
        Parameters
        ----------
        X : Union[pd.Series, List, np.ndarray], optional
            The time series data. Can be a pandas Series, a list, or a numpy array.
        y : None
            Ignored. This parameter exists only for compatibility with sklearn pipeline.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Store time series for prediction later
        if self.approach == 'value' or self.approach == 'resid':
            self.time_series_ = X
        elif self.approach == 'linear_detrend' or self.approach == 'linear_detrend_resid':
            self.time_series_ = self._linear_detrend(X)
        
        # Estimate the time step window or use the integer specified
        if self.time_step_window == 'auto':
            self.time_step_window = estimate_seasonal_length(self.time_series_)

        # Calculate max allowed window size
        max_steps = int(np.floor(len(self.time_series_) * self.max_window_ratio))
        
        # Make sure the time step window is an integer
        if isinstance(self.time_step_window, int):
            if self.time_step_window < 1:
                raise ValueError("The time step window must be greater than 1")
            elif self.time_step_window > max_steps:
                raise ValueError("The time step window must be less than ")
        else:
            raise ValueError("The time step window must be an integer")
        
        # Transform the time series
        self.X_transformed_ = transform_ts(
            self.time_series_,
            self.time_step_window,
            self.max_window_ratio
        )

        # Get feature column count
        self.feature_count_ = self.time_step_window

        # Separate the dependent and independent variables
        X_train = self.X_transformed_.iloc[:, :self.feature_count_]
        y_train = self.X_transformed_.iloc[:, self.feature_count_]

        # If the approach uses residuals, update independent variable with the residuals
        if self.approach == 'resid' or self.approach == 'linear_detrend_resid':
            y_train = self._calculate_ols_resid(X_train, y_train)

        # Set up regressor parameters
        if self.regressor_params is None:
            self.regressor_params = {}
            
        # Add random_state if specified
        if self.random_state is not None:
            self.regressor_params['random_state'] = self.random_state

        # Initialize and fit the Random Forest Model
        self.model_ = RandomForestRegressor(**self.regressor_params)
        self.model_.fit(X_train, y_train)
        
        return self
    
    def predict(self, time_steps_to_forecast: int = 1):
        """
        Forecast future values.
        
        Parameters
        ----------
        time_steps_to_forecast : int, default=1
            The number of future time steps to forecast
            
        Returns
        -------
        array-like
            Predicted values for the requested forecast horizon
        """
        # Check if the model has been fitted
        check_is_fitted(self, ['model_', 'time_series_'])
        
        input_series = self.time_series_

        # Convert input to numpy array for consistent handling
        if isinstance(input_series, pd.Series):
            values = input_series.values
        elif isinstance(input_series, list):
            values = np.array(input_series)
        elif isinstance(input_series, np.ndarray):
            values = input_series
        else:
            raise TypeError("Input series must be a pandas Series, list, or numpy array")
        
        # Get the last number of time steps as inputs for forecasting
        self.X_input = values[-self.time_step_window:]

        preds = []
        ols_preds = []
        new_X_input = self.X_input.copy() # the initial set of independent variables
        ols_input = self.X_input.copy()
        CONSTANT = 1
        for _ in range(time_steps_to_forecast):
            prediction = self.model_.predict([new_X_input])
            preds.append(prediction[0])
            # use the fitted linear regression model predict the value
            if self.approach == 'resid' or self.approach == 'linear_detrend_resid':
                ols_prediction = self.ols_model.predict(np.hstack(([CONSTANT], ols_input)))
                ols_preds.append(ols_prediction[0])
                ols_input = ols_input[1:]
                ols_input = np.append(ols_input, ols_prediction[0])
            # move forward in the time series by one time step
            new_X_input = new_X_input[1:]
            # add the prediction as part of the new set of independent variables
            new_X_input = np.append(new_X_input, prediction[0])
        
        if self.approach == 'value':
            preds = np.array(preds)

        if self.approach == 'resid' or self.approach == 'linear_detrend_resid':
            # the preds represent the predicted residuals using the random forest model
            # add back the predictions from the linear regression to get the final value
            preds = np.array(preds) + np.array(ols_preds)

        if self.approach == 'linear_detrend' or self.approach == 'linear_detrend_resid':
            preds = self._linear_retrend(preds)

        return preds
    
    def _linear_detrend(self, X) -> np.ndarray:
        time = np.arange(len(X))
        coeffs = np.polyfit(time, X, deg=1)
        trend_line = np.polyval(coeffs, time)
        self.linear_detrend_slope = coeffs[0]
        self.linear_detrend_intercept = coeffs[1]
        detrended = X - trend_line
        return detrended
    
    def _linear_retrend(self, preds) -> np.ndarray:
        time = np.arange(len(preds))
        retrended = preds + (self.linear_detrend_slope * time + self.linear_detrend_intercept)
        return retrended
    
    def _calculate_ols_resid(self, X, y) -> np.ndarray:
        """
        Creates an ordinary least-squares (OLS) regression model to predict the 
        dependent variable based on the explanatory variables within each time window.
        The residual of this regression model (the difference between the OLS prediction 
        and the raw value of the dependent variable) is used to represent 
        the dependent variable when training the forest.
        """
        X = sm.add_constant(X)
        self.ols_model = sm.OLS(y, X).fit()
        ols_preds = self.ols_model.predict(X)
        resid = y - ols_preds
        return resid