from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA

postal_codes = pd.read_pickle('pickles/postal_codes_features')


feats = postal_codes.drop('postal_code', 1)

scaler = MinMaxScaler()
postal_codes_scaled = scaler.fit_transform(feats)

pca = PCA(n_components=0.90)
X = pca.fit_transform(postal_codes_scaled)
print("Nombre de composantes retenues avant polynomisation :", pca.n_components_)


polynomial_features = PolynomialFeatures(degree=2)
X = polynomial_features.fit_transform(X)


X = pca.fit_transform(postal_codes_scaled)

print("Nombre de composantes retenues après polynomisation :", pca.n_components_)


pickle.dump(X, open("pickles/postal_code_poly_feats", 'wb'))
