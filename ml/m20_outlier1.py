import numpy as np
aaa = np.array([1,2,-1000,4,5,6,7,8,90,100,500,12])

def outlier(data_out):
    q1, q2,q3 = np.percentile(data_out,[25,50,75])
    print("1 사분위 :",q1)
    print("q2 :",q2)
    print("3 사분위 :",q3)
    
    iqr = q3-q1
    lower_bound  = q1 - (iqr*1.5)
    upper_bound  = q3 + (iqr*1.5)
    
    return np.where((data_out>upper_bound)|(data_out<lower_bound))
    
print(aaa[outlier(aaa)])  #[-1000   500]

#시각화
#boxplot

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()