import numpy as np
import pandas as pd
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer

#data = open("C:\\Users\18406\OneDrive\Desktop\TS\Group Project", "r") #for .txt, to be changed
data = pd.read_csv ('D:\\TS\\Group Project\\testing.csv')

start = time.time()

def truncated_svd_impute(data, k, tol=1e-6, max_iter=200):

    imputer = SimpleImputer(strategy='mean')
    data_no_nan = imputer.fit_transform(data)

    for _ in range(max_iter):

        svd = TruncatedSVD(n_components=k)
        reduced_data = svd.fit_transform(data_no_nan)
        data_reconstructed = svd.inverse_transform(reduced_data)

        diff = np.sqrt(np.nanmean((data_no_nan - data_reconstructed) ** 2))
        
        data_no_nan = np.where(np.isnan(data), data_reconstructed, data)
        
        if diff < tol:

            break

    return data_no_nan

k = 5

imputed_data = truncated_svd_impute(data, k)
imputed_data_int = imputed_data.astype(np.int32)

np.savetxt("myfile.csv", imputed_data_int, delimiter=",", fmt="%d")

end = time.time()

print("The time of execution of above program is :", (end-start) * 10**3, "ms")

#print(imputed_data)
#f = open("myfile.csv", "w")
#f.write(imputed_data)
#f.close()