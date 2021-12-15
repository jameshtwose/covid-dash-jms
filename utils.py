import pandas as pd
import numpy as np
from sklearn import decomposition


def summary_window_FUN(x: pd.DataFrame, window_size: int = 7, user_func=decomposition.PCA, kwargs: dict = {}):
    window_range = np.arange(0, len(x)-window_size, window_size)

    cp_df = pd.DataFrame()
    for window_begin in window_range:
        current_cp_df = pd.DataFrame(user_func(n_components=None, **kwargs)
                                     .fit_transform(x.iloc[window_begin: window_begin + window_size])).iloc[:, 0]
        cp_df = pd.concat([cp_df, current_cp_df])

    return cp_df.rename(columns={0: f"windowed_{user_func.__name__}"}).reset_index(drop=True)