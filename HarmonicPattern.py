import pandas as pd
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt

class harmonic:

    def __init__(self, path, date_start=None, date_end=None):
        self.path = path
        self.date_start = date_start
        self.date_end = date_end

    def read_data(self):
        if self.date_start is None and self.date_end is None:
            return pd.read_csv(self.path)
        elif self.date_start is None or self.date_end is None:
            raise Exception('date start or date end is null')
        elif self.date_start > self.date_end:
            raise Exception('date start greater than date start')
        else:
            df = pd.read_csv('weekly_MSFT.csv')
            return df[(df['timestamp'] >= str(self.date_start)) & (df['timestamp'] <= str(self.date_end))]


    def harmonic_pattern(self):
        data = self.read_data()
        data.timestamp=pd.to_datetime(data.timestamp,format="%Y-%m-%d")
        price=data.close.iloc[0:len(data)-1]
        for i in range(100,len(data)):
            max_idx=list(argrelextrema(price.values[:i],np.greater,order=5)[0])
            min_idx=list(argrelextrema(price.values[:i],np.less,order=5)[0])
            idx = max_idx+min_idx+[len(price.values[:i])-1]
            idx.sort()
            current_idx=idx[-5:]
            current_pat=price.values[current_idx]

            plt.plot(price.values[:i])
            plt.plot(current_idx,current_pat,c='r')
            plt.show()


h=harmonic('weekly_MSFT.csv')
h.harmonic_pattern()