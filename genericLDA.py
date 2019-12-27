import numpy as np
import pandas as pd
import math


"""
How LDA creates, new axis is created according 
to two criteria 
1 - Maximize distance between means 
2 - Minimize the variation (which LDA calls `scatter`
    and is represent by s² ) within each category 

This is how we consider those two criteria simultaneously 
we have a ratio of the difference between the two means 
the squared over the sum of the scatter, the numerator is 
squared because we don't know which one could be larger 



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


TODO ::

"""


class genericLDA:

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

    def separation(self):
        """
        the separation data to two class
        the positive gain and the negative one
        the separation returns array which
        index 0 is array of the data with negative gain
        index 1 --------------------- positive gain
        """
        all_data = self.read_data()
        return all_data[all_data['open'] - all_data['close'] > 0] , all_data[all_data['open'] - all_data['close'] < 0]

    def count_class(self):
        """
        This function returns the count
        of the two classes in array format
        which
        index 0 count of negative gain
        index 1 -------- positive gain
        index 2 count of all the data which have been selected
        """
        ng , pg = self.separation()
        return len(ng) , len(pg) , len(self.read_data())

    def probability(self):
        """
        This function returns probability of
        each class in array format
        """
        totale=self.count_class()[2]
        return (self.count_class()[0]/totale) , (self.count_class()[1]/totale)

    def means_vector(self):
        """
        an ordinary algebra,
        the mean of a set of observations is
        computed by adding all the observations
        and dividing them by the number of observations
        """
        ng_count , pg_count, rd = self.count_class()
        ng_data , pg_data =self.separation()

        return ({'open' : sum(ng_data.open)/ng_count ,
                 'high': sum(ng_data.high)/ng_count ,
                 'low' :sum(ng_data.low)/ng_count ,
                 'close':sum(ng_data.close)/ng_count ,
                 'volume': sum(ng_data.volume)/ng_count }) ,\
                ({'open' :sum(pg_data.open)/pg_count ,
                  'high' : sum(pg_data.high)/pg_count ,
                  'low' : sum(pg_data.low)/pg_count ,
                  'close' : sum(pg_data.close)/pg_count ,
                  'volume' : sum(pg_data.volume)/pg_count })

    def covariance_matrix(self , **kwargs):
        """
        :'( now we are inside of the mathematics troubles
        what we need at this point ?
        so calculate covariance matrix for
        each class ( negative and positive)
        it's can be something like this
        --- for negative gain class
        open/open    open/high    open/low    open/close    open/volume
        high/open    high/high    high/low    high/close    open/volume
         low/open     low/high     low/low     low/close     low/volume
        close/open   close/high   close/low   close/close   close/volume
        volume/open  volume/high  volume/low  volume/close  volume/volume

        and the same thing for positive gain class

            /\    PROBLEM IS HOW TO MAKE THIS FUNCTION
          / ! \   MORE GENERIC FOR OUR SPICIFY VECTORS
        /______\    LET'S TRY WITH **kwargs
        """
        assert(kwargs!=None),'Error must have last one arg'
        init_matrix = list()
        co = list()
        for key , values in kwargs.items():
            init_matrix.append(list(values))
            co.append(str(key))

        init_matrix = np.array(init_matrix).transpose()
        data=pd.DataFrame(init_matrix,columns=co)

        return data.cov()



    def pooled_covariance_matrix(self , negative_covmatrix , positive_covmatrix ):

        negative_count, positive_count, rd = self.count_class()

        return (1/(negative_count + positive_count))*(negative_count*negative_covmatrix) + (positive_count*positive_covmatrix)




    def coefficients_linear_model(self, pooled_covmatrix):
        C_1 = pd.DataFrame(np.linalg.pinv(pooled_covmatrix) , columns=pooled_covmatrix.columns)
        #print('¢_1\n', C_1 )
        U1 , U2 = self.means_vector()
        index=list(pooled_covmatrix.columns)
        U1=pd.DataFrame(dict(U1),index=[0]).loc[:, index] - pd.DataFrame(dict(U2),index=[0]).loc[:, index]

        return pd.DataFrame((C_1.transpose() * U1.transpose())[0])


    def mahalanobis(self , beta ):
        U1, U2 = self.means_vector()
        index = list(beta.index)
        U1 = pd.DataFrame(dict(U1), index=[0]).loc[:, index] - pd.DataFrame(dict(U2), index=[0]).loc[:, index]

        U1 = U1.values[0]

        delta = 0
        for u in range(beta.__len__()):
            delta += U1[u]*beta[0][u]
        print('DELTA')
        return math.fabs(delta)

    def predict(self, beta , **kwargs):

        """
        **kwargs is being predict
        to which class it does belong to.
        It must to be as the same type as
        the previous approved one in `covariance_matrix`
        function
        """
        index = list(beta.index)
       # assert(list(set(index) & set(kwargs.keys())) == index), "Error in key name"
        data=pd.DataFrame(kwargs.values() , index=kwargs.keys())
        U1, U2 = self.means_vector()
        U1 = pd.DataFrame(dict(U1), index=[0]).loc[:,index] + pd.DataFrame(dict(U2),index=[0]).loc[:,index]
        U1 = pd.DataFrame(U1.values[0]/2, index=index)
        data-=U1
        #print(beta,'\n',data)
        p1,p2 = self.probability()

        if float((beta*data).sum()) > float(-math.log(p1/p2)) :
            return "negative gain"
        else:
            return "positive gain"




##_____MAIN

"""
lda =  genericLDA('weekly_MSFT.csv', '2011-04-08' ,'2019-03-08' )
negative_gain , positive_gain = lda.separation()

data =lda.covariance_matrix(open = negative_gain['open'] , close=negative_gain['close']  )
data1 = lda.covariance_matrix(open = positive_gain['open'] , close=positive_gain['close'] )
print(data)
print('###############################')
print(data1)
print('###############################')
data2 = lda.pooled_covariance_matrix(data, data1)
print(data2)
print('###############################')
data3=lda.coefficients_linear_model(data2)
print(data3)
print('###############################')
print(lda.mahalanobis(data3))
print('###############################')
#113.020  110.51
data4=lda.predict(data3, open=113.020 , close=110.51)
print(data4)
"""