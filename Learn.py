#VISUALISATION ET IMPORT DE DONNEES
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import torch
import matplotlib.pyplot as plt
import seaborn as sns



#SKLEARN FUNCTIONS
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor

########################      PRE-PROCESSING      #############################


#ouverture et lecture des deux fichiers csv : frequences propres et entrées 
freq = pd.read_csv(open("test.csv", "r"),
                    delimiter=",")
inputs = pd.read_csv(open("dict.csv", "r"),
                    delimiter=",")
#2 dataFrames sont créés


#Concatenation des deux dataframes dans le même DataFrame : data
datas = [inputs, freq]
datas = pd.concat(datas, axis=1)


def plot_correlation_matrix(data):
    corr = data.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    plt.show()
    print(corr)

#plot_correlation_matrix(pd.DataFrame(datas))

# d'apres la matrice de correlation, certaines entrées sont étroitement liées
# on va donc supprimer certaines de ces valeurs pour conserver :
# hauteur h, base b, la masse volumique rho, la longueur de la poutre L_tot
corr = ['NbElts', 'S', 'I', 'L', 'E','freq2','freq3','freq4','freq5','freq6','freq7','freq8']
print(datas)
######################      FIN PRE-PROCESSING      ###########################




#######################      TRAIN_TEST_SPLIT      ############################
# 70% de la population sera allouée à l'apprentissage, 30 % pour le test
population_train = 0.7

# mélange et séparation de nos données en 2 datasets  
split_train, split_test = train_test_split(datas, train_size=population_train)

# On extrait les données qui serviront d'objectif à atteindre, soit ici les 
# 8 fréquences propres à prédire

entrees = ['L_tot','rho', 'h', 'b']
split_target_train = split_train.drop(columns=entrees)
split_target_test = split_test.drop(columns=entrees)
print(split_target_train)

frequences = ["freq1"]
split_train = split_train.drop(columns=frequences)
split_test = split_test.drop(columns=frequences)

print("entrées train : \n",split_train)
print("target train : \n", split_target_train)

#split_train = entrees servant à entrainer le modèle
#split test = entrees servant à tester le modèle
#split_target_train = sorties d'entrainement du modèle
#split_target_test = sorties de test du modèle 


#######################      FIN TR_TST_SPLIT      ############################

##################          PROCESS LINEAIRE            #######################
# Regression linéaire
reg_lin = LinearRegression().fit(split_train, split_target_train)

# Regression lasso
reg_lasso = linear_model.Lasso(alpha=0.01).fit(
    split_train, split_target_train)

# Regression ridge
reg_ridge = linear_model.Ridge(alpha=0.01).fit(
    split_train, split_target_train)

# Regression elastic net
reg_elastic = linear_model.ElasticNet().fit(
    split_train, split_target_train)


def get_score(reg, test, target_test):
    score = reg.score(test, target_test)
    # print the name of variable the regression and the score
    print("Score de la regression : ", score)

def get_mean_score(reg, test, target_test):
    score = 0
    for i in range(0, 10000):
        score += reg.score(test, target_test)
    print("mean score : ", score / 10000)


def get_MSE(target_test, test):
    score = mean_squared_error(target_test, test)
    # print the name of variable the regression and the score
    print("MEAN SQARED ERROR : ", score)

get_score(reg_lin, split_test, split_target_test)
get_score(reg_lasso, split_test, split_target_test)
get_score(reg_ridge, split_test, split_target_test)
get_score(reg_elastic, split_test, split_target_test)

Y_pred = reg_lin.predict(split_test)

###### RANDOM FOREST #######
print(split_train.shape)
print(type(split_target_train))


regr_multirf = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, random_state=0)
)
regr_multirf.fit(split_train, split_target_train)


get_score(regr_multirf, split_test, split_target_test)



################### Polynomial Regression #####################
# make polynimial regression with these hyperparameters linearregression__fit_intercept': True, 'linearregression__normalize': True, 'polynomialfeatures__degree': 9


polynomial_features = PolynomialFeatures(degree=9)
linear_regression = LinearRegression(fit_intercept=True, normalize=True)
pipeline = Pipeline(
    [
        ("polynomial_features", polynomial_features),
        ("linear_regression", linear_regression),
    ]
)
pipeline.fit(split_train, split_target_train)
#predict

def PolynomialRegression(degree=9, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

poly_reg_model = PolynomialRegression(9, fit_intercept=True, normalize=True)
poly_reg_model.fit(split_train, split_target_train)
print("Score du modèle polynomial normalement 99% : ", poly_reg_model.score(split_test, split_target_test))

# predict 10000 times and give best score 
tmp_score = 0 
for i in range(0, 1000):
    score = pipeline.score(split_test, split_target_test)
    if score > tmp_score:
        tmp_score = score

print("score polynomial regression : ", tmp_score)

Y_poly_pred = pipeline.predict(split_test)


print("Score de la regression polynomiale : ", pipeline.score(split_test, split_target_test))
get_score(pipeline, split_test, split_target_test)





####### PLOT PREDICTIONS #######
Y_forest_pred = regr_multirf.predict(split_test)
# ploting the line graph of actual and predicted values
plt.figure(figsize=(12, 5))
plt.plot((Y_forest_pred)[:80])
plt.plot((Y_pred)[:80])
plt.plot((np.array(split_target_test)[:80]))
plt.plot((Y_poly_pred)[:80])
plt.legend(
    ["Prediction_forect", "Prediction Regression Linéraire", "valeur réelle","Prediction Regression Polynomiale"])
plt.show()

#####################          FIN PROCESS           ##########################

#######################      CROSS VALIDATION      ############################



####################      FIN CROSS VALIDATION       ##########################




