from PreciseQPO import QuantumPortfolioOptimization

QuantumPortfolioOptimization(
    numberOfStocks = 10,
    expectedReturns = 0.25,
    precision = 4,
    lagrange = [3,1,0.2],
    reCalcCov = False,
    startDate = '2013-01-01',
    endDate = '2019-01-01',
    numReads = 10,
    chainStrength = 1,
    backTest = True,
    amount = 1000000,
    periodOriginal = [2019,2020],
    rebalance = "monthly"
)

