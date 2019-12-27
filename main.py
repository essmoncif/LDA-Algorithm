from datetime import datetime
import pandas as pd
import numpy as np
from genericLDA import genericLDA as gld
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

"""

class |  count | probabilty | statistic | open | close | high | low | volume | 
------|--------|------------|-----------|------|-------|------|-----|--------|
gain>0|  n1=   |   P(c1)=   |   Means   |      |       |      |     |        |
      |        |            |  vectors  |      |       |      |     |        |
      |        |            |-----------|------|-------|------|-----|--------|
      |        |            | covariance|                                    |
      |        |            |   matrix  |                                    |
------|--------|------------|-----------|------|-------|------|-----|--------|
gain<0|  n2=   |   P(c2)=   |   Means   |      |       |      |     |        |
      |        |            |  vectors  |      |       |      |     |        |
      |        |            |-----------|------|-------|------|-----|--------|
      |        |            | covariance|                                    |
      |        |            |   matrix  |                                    |
-----------------------------------------------------------------------------|
"""
lda=gld('weekly_MSFT.csv', '2017-01-06','2018-12-28')
print(lda.read_data()['open'])

"""
Now we create our class to start prediction 
so let's call separation function 
"""
negative_gain , positive_gain = lda.separation()

covmatrix_negative=lda.covariance_matrix(open=negative_gain['open'],close=negative_gain['close'],volume=negative_gain['volume'])
covmatrix_positive=lda.covariance_matrix(open=positive_gain['open'],close=positive_gain['close'],volume=positive_gain['volume'])
pooled_matrix=lda.pooled_covariance_matrix(covmatrix_negative,covmatrix_positive)
coeff=lda.coefficients_linear_model(pooled_matrix)
print(lda.mahalanobis(coeff))
print(lda.predict(coeff, open=15 , close=1))

def draw_candles(data):
    fig = go.Figure(data=[go.Candlestick(x=data['timestamp'],open=data['open'],close=data['close'],high=data['high'],low=data['low'])])
    fig.show()

def draw_scatter3D(data1,data2,all_data,coeff):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for color,marker,data in [('r','o',data1),('b','*',data2)]:
        open=data['open']
        close=data['close']
        volume=data['volume']
        ax.scatter(open,close,volume,c=color,marker=marker)

    x=all_data['open']*coeff[0]['open']
    y=all_data['close']*coeff[0]['close']
    z=all_data['volume']*coeff[0]['volume']

    ax.plot(x,y,z)

    ax.set_xlabel('open')
    ax.set_ylabel('close')
    ax.set_zlabel('volume')
    plt.show()


def draw_scatter2D(data1, data2):
    for color, marker, data in [('r', 'o', data1), ('b', '*', data2)]:
        open = data['open']
        close = data['close']
        plt.scatter(open, close,  c=color, marker=marker)

    plt.xlabel('open')
    plt.ylabel('close')
    plt.show()

draw_candles(lda.read_data())
draw_candles(gld('weekly_MSFT.csv' , '2018-12-28', '2019-11-29').read_data())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = lda.read_data()['open'] * coeff[0]['open']
y = lda.read_data()['close'] * coeff[0]['close']
z = lda.read_data()['volume'] * coeff[0]['volume']
ax.plot(x, y, z)


ax.set_xlabel('open')
ax.set_ylabel('close')
ax.set_zlabel('volume')
plt.show()

#draw_candles(gld('weekly_MSFT.csv').read_data())
draw_scatter3D(negative_gain,positive_gain,lda.read_data(),coeff)
draw_scatter2D(negative_gain,positive_gain)
