import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from dwave.system import DWaveSampler, EmbeddingComposite

class QuantumPortfolioOptimization:
    def __init__(self,numberOfStocks,expectedReturns,precision,lagrange=[1,1,1],numReads=10,\
                chainStrength=1,reCalcCov=False,startDate='2013-01-01',endDate='2018-01-01'):
        self.lagrange = lagrange
        self.noStocks = numberOfStocks
        self.ER = expectedReturns
        self.numReads = numReads
        self.chainStrength = chainStrength
        self.Again = reCalcCov
        self.startDate = startDate
        self.endDate = endDate
        self.getData()
        self.getQubo(precision)
        self.dWaveExecute()
        self.Results(precision)

    def getData(self):
        if self.Again==False:
            Cov = pd.read_csv("data/covariance.csv",header=0)
            Index = Cov["Unnamed: 0"]
            Cov = Cov.drop(["Unnamed: 0"],axis=1)
            Cov = Cov.set_index(Index)
            self.Cov = Cov
            self.Means = pd.read_csv("data/meanReturns.csv",header=0).to_numpy()[:,1].tolist()
        else:
            self.reCalc()

    def reCalc(self):
        data = pd.read_csv("data/dailyClosingPrices.csv",header=0)
        Index = data["Date"]
        data = data.drop(["Date"],axis=1)
        data = data.set_index(Index)[self.startDate:self.endDate]
        returns = data.pct_change()
        self.Cov = returns.cov()*252
        self.Means = returns.mean(axis=0)*252
    
    def getQubo(self,prc):
        Qubo = defaultdict(int)
        QuboLength = prc*self.noStocks
        for j in range(QuboLength):
            for i in range(j+1):
                if i==j:
                    Qubo[(i,j)] = self.Cov.iloc[i//prc,j//prc]\
                        *self.lagrange[0]/pow(4,i%prc+1)
                else:
                    Qubo[(i,j)] = 2*self.Cov.iloc[i//prc,j//prc]\
                        *self.lagrange[0]/pow(2,i%prc+j%prc+2)

        for j in range(QuboLength):
            for i in range(j+1):
                if i==j:
                    Qubo[(i,j)] = ((self.Means[i//prc]**2)/pow(4,i%prc+1)\
                        -2*self.ER*self.noStocks*self.Means[i//prc]\
                        /pow(2,i%prc+1))*self.lagrange[1]
                else:
                    Qubo[(i,j)] = 2*self.Means[i//prc]*self.Means[j//prc]\
                        /pow(4,i%prc+j%prc+2)*self.lagrange[1]

        for j in range(QuboLength):
            for i in range(j+1):
                if i==j:
                    Qubo[(i,j)] += (1/(pow(4,i%prc+1))-2/pow(2,i%prc+1))*self.lagrange[2]
                else:
                    Qubo[(i,j)] += 2*self.lagrange[2]/(pow(2,i%prc+j%prc+2))

        self.Qubo = Qubo
    
    def dWaveExecute(self):
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample_qubo(self.Qubo, num_reads=self.numReads,\
            chain_strength=self.chainStrength)
        print("Time Spent in Quantum Computer: ",\
            response.info["timing"]["qpu_access_time"]/1000,"Milli Seconds")
        self.response = response

    def Results(self,prc):
        rank = 0
        for best, energy in self.response.data(['sample', 'energy']):
            rank+=1
            print("\n############# Rank {} #############".format(rank))
            actual_return = 0
            weights = [0 for i in range(self.noStocks)]
            for i in best.keys():
                actual_return += best[i]*self.Means[i//prc]/pow(2,i%prc+1)
                weights[i//prc] += best[i]/pow(2,i%prc+1)

            s = sum(weights)
            weights = np.array(weights)/s

            if rank==1:
                self.weights = weights
                
            volatility = 0
            for i in range(self.noStocks):
                for j in range(self.noStocks):
                    volatility += self.Cov.iloc[i,j]*weights[i]*weights[j]

            print("Weights: ",weights.tolist())
            print("Expected Return: ",self.ER)
            print("Actual Return: ",actual_return/s)
            print("Volatility: ",np.sqrt(volatility))