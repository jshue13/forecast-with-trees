from typing import Union, List
import pandas as pd
import numpy as np
from scipy.signal import periodogram
import matplotlib.pyplot as plt

def transform_ts(
    series: Union[pd.Series, List, np.ndarray], 
    time_step_window: int = 1,
    max_window_ratio: float = 1/3
) -> pd.DataFrame:
    '''
    Transforms a single univariate time series into a dataframe with 
    multiple sets of explanatory and dependent variables.

    Parameters
    ----------
        series: Union[pd.Series, List, np.ndarray]
            The time series data. Can be a pandas Series, a list, or a numpy array.
        
        time_step_window: int, default=1
            The number of steps within each time window. These time steps are used 
            as explanatory variables, and the next time step after the time window 
            is the dependent variable. 
            
            For example, if there are 20 time steps, and the time window is 4 time 
            steps, there are 16 sets of explanatory and dependent variables used 
            to train the forest. The first set has time steps 1, 2, 3, and 4 as 
            explanatory variables and time step 5 as the dependent variable. The 
            second set has time steps 2, 3, 4, and 5 as explanatory variables and 
            time step 6 as the dependent variable. The time window can be as small 
            as 1 (so that there is only a single time step within each time window) 
            and cannot exceed one-third of the number of time steps.

        max_window_ratio: float, default=1/3
            The maximum window size as a ratio of the total series length. Default is 1/3.
    
    Returns
    -------
    pd.DataFrame
        The transformed data in a pandas DataFrame with time_step_window columns representing
        the input features and one column for the target value.

    Raises
    ------
    ValueError
        If the time series is empty or if time_step_window is less than 1.
    '''
    # Convert input to numpy array for consistent handling
    if isinstance(series, pd.Series):
        values = series.values
    elif isinstance(series, list):
        values = np.array(series)
    elif isinstance(series, np.ndarray):
        values = series
    else:
        raise TypeError("Input series must be a pandas Series, list, or numpy array")
    
    # Validate input
    if len(values) == 0:
        raise ValueError("Input time series cannot be empty")
    
    if time_step_window < 1:
        raise ValueError("time_step_window must be at least 1")
    
    # Calculate max allowed window size
    max_steps = int(np.floor(len(values) * max_window_ratio))

    # Adjust window size if needed
    window = min(max(1, time_step_window), max_steps)
    total_window = window + 1  # +1 for the target variable

    # Create sliding windows
    data = []
    for i in range(len(values) - window):
        row = values[i:i+total_window]
        data.append(row)
        
    return pd.DataFrame(data)

def get_spectral_density(series: Union[pd.Series, List, np.ndarray]):
    frequencies, power_spectrum = periodogram(series, return_onesided=False)
    periods = 1 / frequencies
    return frequencies, power_spectrum, periods

def estimate_seasonal_length(series: Union[pd.Series, List, np.ndarray]) -> int:
    _, power_spectrum, periods = get_spectral_density(series)
    max_signal_idx = np.argmax(power_spectrum)
    seasonal_length = int(np.floor(periods[max_signal_idx]))
    return seasonal_length

def plot_periodgram(periods, power_spectrum):
    fig, ax = plt.subplots(figsize=(20, 3))
    ax.step(periods, power_spectrum)
    ax.set_title('Periodogram')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter('{x:,.0f}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.show()