import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer

data = open("C:\\Users\18406\OneDrive\Desktop\TS\Group Project", "r") #for .txt, to be changed

def truncated_svd_impute(data, k, tol=1e-6, max_iter=100):

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

print(imputed_data)