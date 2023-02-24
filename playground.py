from products import payerSwap
from scipy import optimize
from dynamics import *
import seaborn as sns
import matplotlib.pyplot as plt

print(' Start')
for i in range(10):
    
    print(f'{i}', flush=True)


# testDynamic = Vasicek(initial=0.05, mean=0.02, reversion=-0.5, volatility=0.02)
# fixSchedule = np.arange(0,10+1)
# floSchedule = np.arange(0,20+1)/2
# time, floatRate = testDynamic.create_path(1/12, 10, 1999)
# K=0.03
# [payerSwap(
#     time=x, 
#     fixedSchedule=fixSchedule, 
#     floatingSchedule=floSchedule, 
#     fixedRate=K, 
#     floatingRate=floatRate, 
#     floatingTimeStamp=time,
#     model=testDynamic
# )
# for x in time
# ]