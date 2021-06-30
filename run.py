from PreciseQPO import QuantumPortfolioOptimization

QuantumPortfolioOptimization(
    numberOfStocks = 5,
    expectedReturns = 0.3,
    precision = 4,
    lagrange = [10,1,0.2],
    reCalcCov = False,
    startDate = '2013-01-01',
    endDate = '2018-01-01',
    backTest = True,
    periodOriginal = [2018,2021]
)