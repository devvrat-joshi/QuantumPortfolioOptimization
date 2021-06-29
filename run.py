from PreciseQPO import QuantumPortfolioOptimization

QuantumPortfolioOptimization(
    numberOfStocks = 5,
    expectedReturns = 0.3,
    precision = 4,
    lagrange = [5,3,1],
    reCalcCov = True,
    startDate = '2013-01-01',
    endDate = '2018-01-01'
)