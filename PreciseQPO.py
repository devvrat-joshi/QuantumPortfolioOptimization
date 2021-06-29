import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from dwave.system import DWaveSampler, EmbeddingComposite

class QuantumPortfolioOptimization:
    def __init__(self,numberOfStocks,expectedReturns,precision,lagrange=[1,1,1],numReads=10,\
                chainStrength=1,reCalcCov=False,startDate='2013-01-01',endDate='2018-01-01',\
                backTest=False,amount=1e+6,periodOriginal=[2018,2021],rebalance="monthly"):
        self.lagrange = lagrange
        self.noStocks = numberOfStocks
        self.ER = expectedReturns
        self.numReads = numReads
        self.chainStrength = chainStrength
        self.Again = reCalcCov
        self.startDate = startDate
        self.endDate = endDate
        self.backTest = backTest
        self.amount = amount
        self.period = periodOriginal
        self.execute(precision)

    def execute(self,prc):
        if self.backTest==False:
            self.getData()
            self.getQubo(prc)
            self.dWaveExecute()
            self.Results(prc)
        else:
            self.doBackTest(prc)

    def getData(self):
        if self.Again==False and self.backTest==False:
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
        if self.backTest==True:
            self.prices = data[:self.endDate].iloc[-1:]
    
    def getQubo(self,prc):
        self.Qubo = defaultdict(int)
        self.QuboLength = prc*self.noStocks
        self.riskObjective(prc)
        self.returnsObjective(prc)
        self.totalStocksObjective(prc)

    def riskObjective(self,prc):
        for j in range(self.QuboLength):
            for i in range(j+1):
                if i==j:
                    self.Qubo[(i,j)] = self.Cov.iloc[i//prc,j//prc]\
                        *self.lagrange[0]/pow(4,i%prc+1)
                else:
                    self.Qubo[(i,j)] = 2*self.Cov.iloc[i//prc,j//prc]\
                        *self.lagrange[0]/pow(2,i%prc+j%prc+2)

    def returnsObjective(self,prc):
        for j in range(self.QuboLength):
            for i in range(j+1):
                if i==j:
                    self.Qubo[(i,j)] = ((self.Means[i//prc]**2)/pow(4,i%prc+1)\
                        -2*self.ER*self.noStocks*self.Means[i//prc]\
                        /pow(2,i%prc+1))*self.lagrange[1]
                else:
                    self.Qubo[(i,j)] = 2*self.Means[i//prc]*self.Means[j//prc]\
                        /pow(4,i%prc+j%prc+2)*self.lagrange[1]

    def totalStocksObjective(self,prc):
        for j in range(self.QuboLength):
            for i in range(j+1):
                if i==j:
                    self.Qubo[(i,j)] += (1/(pow(4,i%prc+1))-2/pow(2,i%prc+1))*self.lagrange[2]
                else:
                    self.Qubo[(i,j)] += 2*self.lagrange[2]/(pow(2,i%prc+j%prc+2))
    
    def dWaveExecute(self):
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample_qubo(self.Qubo, num_reads=self.numReads,\
            chain_strength=self.chainStrength)
        print("Time Spent in Quantum Computer: ",\
            response.info["timing"]["qpu_access_time"]/1000,"Milli Seconds")
        self.response = response

    def doBackTest(self,prc):
        months = ['-01-01','-02-01','-03-01','-04-01','-05-01','-06-01',\
                '-07-01','-08-01','-09-01','-10-01','-11-01','-12-01',]
        self.normal = np.array([1/self.noStocks]*self.noStocks)
        self.normalAmount = self.amount
        self.getData()
        self.executeTest(prc)
        for year in range(self.period[0],self.period[1]):
            for month in months:
                self.endDate = str(year)+month
                self.amount = self.leftOut
                self.normalAmount = self.normalLeftOut
                self.getData()
                for i in range(self.noStocks):
                    self.amount += self.shares[i]*self.prices.iloc[0,i]
                    self.normalAmount += self.normalShares[i]*self.prices.iloc[0,i]
                self.executeTest(prc)
                print("Amount at the end of "+self.endDate+" : ",self.amount)
                print("Normal Amount at the end of "+self.endDate+" : ",self.normalAmount)

    def executeTest(self,prc):
        self.getQubo(prc)
        self.dWaveExecute()
        self.Results(prc)
        self.shares = self.weights*self.amount
        self.leftOut = 0
        self.normalShares = self.normal*self.normalAmount
        self.normalLeftOut = 0
        for i in range(self.noStocks):
            self.normalLeftOut += self.normalShares[i]%self.prices.iloc[0,i]
            self.normalShares[i] //= self.prices.iloc[0,i]
            self.leftOut += self.shares[i]%self.prices.iloc[0,i]
            self.shares[i] //= self.prices.iloc[0,i]

    def Results(self,prc):
        rank = 0
        for best, energy in self.response.data(['sample', 'energy']):
            rank+=1
            if self.backTest==False:
                print("\n############# Rank {} #############".format(rank))
            actual_return = 0
            weights = [0 for i in range(self.noStocks)]
            for i in best.keys():
                actual_return += best[i]*self.Means[i//prc]/pow(2,i%prc+1)
                weights[i//prc] += best[i]/pow(2,i%prc+1)

            s = sum(weights)
            weights = np.array(weights)/s

            volatility = 0
            for i in range(self.noStocks):
                for j in range(self.noStocks):
                    volatility += self.Cov.iloc[i,j]*weights[i]*weights[j]

            if rank==1:
                self.weights = weights
                print("Weights: ",weights.tolist())
                print("Expected Return: ",self.ER)
                print("Actual Return: ",actual_return/s)
                print("Volatility: ",np.sqrt(volatility))

            if self.backTest==False and rank!=1:
                print("Weights: ",weights.tolist())
                print("Expected Return: ",self.ER)
                print("Actual Return: ",actual_return/s)
                print("Volatility: ",np.sqrt(volatility))