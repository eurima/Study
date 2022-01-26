import numpy as np
import pandas as pd
aaa = np.array([[1,2,-20,4,5,6,7,8,30,100,500,12,13],
               [100,200,3,400,500,600,7,800,900,190,1001,1002,99]])

aaa = np.transpose(aaa) #(13,2)
df = pd.DataFrame(aaa)
print(df)

def outlier_df(data_out):
    data_out = data_out.to_numpy()
    data_out = np.transpose(data_out)
    q1, q2,q3 = np.percentile(data_out,[25,50,75])
    print("1 사분위 :",q1)
    print("q2 :",q2)
    print("3 사분위 :",q3)
    
    iqr = q3-q1
    lower_bound  = q1 - (iqr*1.5)
    upper_bound  = q3 + (iqr*1.5)
    
    return np.where((data_out>upper_bound)|(data_out<lower_bound))
 
for column_name, item in df.iteritems():
    print(outlier_df(item)) 

