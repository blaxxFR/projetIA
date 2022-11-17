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

def get_score(reg, test, target_test):
    score = reg.score(test, target_test)
    # print the name of variable the regression and the score
    print("Score de la regression : ", score)

#ouverture et lecture des deux fichiers csv : frequences propres et entrées
def gen_model_rect():
    freq = pd.read_csv(open("./gen_data/test_Rectangle.csv", "r"),
                        delimiter=",")
    inputs = pd.read_csv(open("./gen_data/dict_Rectangle.csv", "r"),
                        delimiter=",")
    #2 dataFrames sont créés


    #Concatenation des deux dataframes dans le même DataFrame : data
    datas = [inputs, freq]
    datas = pd.concat(datas, axis=1)

    #plot_correlation_matrix(pd.DataFrame(datas))

    # d'apres la matrice de correlation, certaines entrées sont étroitement liées
    # on va donc supprimer certaines de ces valeurs pour conserver :
    # hauteur h, base b, la masse volumique rho, la longueur de la poutre L_tot
    to_drop = ['NbElts', 'S', 'I', 'L', 'E', 'Mat']
    datas = datas.drop(columns=to_drop)
    print(datas)
    ######################      FIN PRE-PROCESSING      ###########################




    #######################      TRAIN_TEST_SPLIT      ############################
    # 70% de la population sera allouée à l'apprentissage, 30 % pour le test
    population_train = 0.7

    # mélange et séparation de nos données en 2 datasets
    split_train, split_test = train_test_split(datas, train_size=population_train)

    # On extrait les données qui serviront d'objectif à atteindre, soit ici les
    # 8 fréquences propres à prédire

    entrees = ['L_tot','rho', 'h', 'b', 'freq1']
    split_target_train = split_train.drop(columns=entrees)
    split_target_test = split_test.drop(columns=entrees)
    print(split_target_train)

    frequences = ["freq2",'freq3','freq4','freq5','freq6','freq7','freq8']
    split_train = split_train.drop(columns=frequences)
    split_test = split_test.drop(columns=frequences)



    #######################      FIN TR_TST_SPLIT      ############################


    ##################          PROCESS LINEAIRE            #######################
    # Regression linéaire
    reg_lin = LinearRegression().fit(split_train, split_target_train)
    get_score(reg_lin, split_test, split_target_test)

    return reg_lin


def gen_model_cercle():
    freq = pd.read_csv(open("./gen_data/test_Cercle.csv", "r"),
                        delimiter=",")
    inputs = pd.read_csv(open("./gen_data/dict_Cercle.csv", "r"),
                        delimiter=",")
    #2 dataFrames sont créés


    #Concatenation des deux dataframes dans le même DataFrame : data
    datas = [inputs, freq]
    datas = pd.concat(datas, axis=1)


    # d'apres la matrice de correlation, certaines entrées sont étroitement liées
    # on va donc supprimer certaines de ces valeurs pour conserver :
    # hauteur h, base b, la masse volumique rho, la longueur de la poutre L_tot
    to_drop = ['NbElts', 'S', 'I', 'L', 'E', 'Mat']
    datas = datas.drop(columns=to_drop)
    print(datas)
    ######################      FIN PRE-PROCESSING      ###########################




    #######################      TRAIN_TEST_SPLIT      ############################
    # 70% de la population sera allouée à l'apprentissage, 30 % pour le test
    population_train = 0.7

    # mélange et séparation de nos données en 2 datasets
    split_train, split_test = train_test_split(datas, train_size=population_train)

    # On extrait les données qui serviront d'objectif à atteindre, soit ici les
    # 8 fréquences propres à prédire

    entrees = ['L_tot','rho', 'r', 'freq1']
    split_target_train = split_train.drop(columns=entrees)
    split_target_test = split_test.drop(columns=entrees)
    print(split_target_train)

    frequences = ["freq2",'freq3','freq4','freq5','freq6','freq7','freq8']
    split_train = split_train.drop(columns=frequences)
    split_test = split_test.drop(columns=frequences)



    #######################      FIN TR_TST_SPLIT      ############################


    ##################          PROCESS LINEAIRE            #######################
    # Regression linéaire
    reg_lin = LinearRegression().fit(split_train, split_target_train)
    get_score(reg_lin, split_test, split_target_test)

    return reg_lin


def gen_model_Cercle_creux():
    freq = pd.read_csv(open("./gen_data/test_Cercle_Creux.csv", "r"),
                       delimiter=",")
    inputs = pd.read_csv(open("./gen_data/dict_Cercle_Creux.csv", "r"),
                         delimiter=",")
    # 2 dataFrames sont créés

    # Concatenation des deux dataframes dans le même DataFrame : data
    datas = [inputs, freq]
    datas = pd.concat(datas, axis=1)

    # d'apres la matrice de correlation, certaines entrées sont étroitement liées
    # on va donc supprimer certaines de ces valeurs pour conserver :
    # hauteur h, base b, la masse volumique rho, la longueur de la poutre L_tot
    to_drop = ['NbElts', 'S', 'I', 'L', 'E', 'Mat']
    datas = datas.drop(columns=to_drop)
    print(datas)
    ######################      FIN PRE-PROCESSING      ###########################

    #######################      TRAIN_TEST_SPLIT      ############################
    # 70% de la population sera allouée à l'apprentissage, 30 % pour le test
    population_train = 0.7

    # mélange et séparation de nos données en 2 datasets
    split_train, split_test = train_test_split(datas, train_size=population_train)

    # On extrait les données qui serviront d'objectif à atteindre, soit ici les
    # 8 fréquences propres à prédire

    entrees = ['L_tot', 'rho','r_ext','r_int','freq1']
    split_target_train = split_train.drop(columns=entrees)
    split_target_test = split_test.drop(columns=entrees)
    print(split_target_train)

    frequences = ["freq2", 'freq3', 'freq4', 'freq5', 'freq6', 'freq7', 'freq8']
    split_train = split_train.drop(columns=frequences)
    split_test = split_test.drop(columns=frequences)

    #######################      FIN TR_TST_SPLIT      ############################

    ##################          PROCESS LINEAIRE            #######################
    # Regression linéaire
    reg_lin = LinearRegression().fit(split_train, split_target_train)
    get_score(reg_lin, split_test, split_target_test)

    return reg_lin

def gen_model_Rectangle_creux():
    freq = pd.read_csv(open("./gen_data/test_Rectangle_Creux.csv", "r"),
                       delimiter=",")
    inputs = pd.read_csv(open("./gen_data/dict_Rectangle_Creux.csv", "r"),
                         delimiter=",")
    # 2 dataFrames sont créés

    # Concatenation des deux dataframes dans le même DataFrame : data
    datas = [inputs, freq]
    datas = pd.concat(datas, axis=1)

    # d'apres la matrice de correlation, certaines entrées sont étroitement liées
    # on va donc supprimer certaines de ces valeurs pour conserver :
    # hauteur h, base b, la masse volumique rho, la longueur de la poutre L_tot
    to_drop = ['NbElts', 'S', 'I', 'L', 'E', 'Mat']
    datas = datas.drop(columns=to_drop)
    print(datas)
    ######################      FIN PRE-PROCESSING      ###########################

    #######################      TRAIN_TEST_SPLIT      ############################
    # 70% de la population sera allouée à l'apprentissage, 30 % pour le test
    population_train = 0.7

    # mélange et séparation de nos données en 2 datasets
    split_train, split_test = train_test_split(datas, train_size=population_train)

    # On extrait les données qui serviront d'objectif à atteindre, soit ici les
    # 8 fréquences propres à prédire

    entrees = ['L_tot', 'rho','h_ext','b_ext','h_int','b_int','freq1']
    split_target_train = split_train.drop(columns=entrees)
    split_target_test = split_test.drop(columns=entrees)
    print(split_target_train)

    frequences = ["freq2", 'freq3', 'freq4', 'freq5', 'freq6', 'freq7', 'freq8']
    split_train = split_train.drop(columns=frequences)
    split_test = split_test.drop(columns=frequences)

    #######################      FIN TR_TST_SPLIT      ############################

    ##################          PROCESS LINEAIRE            #######################
    # Regression linéaire
    reg_lin = LinearRegression().fit(split_train, split_target_train)
    get_score(reg_lin, split_test, split_target_test)

    return reg_lin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))
def gen_model_treillis_rect():
    freq = pd.read_csv(open("test_treillis_10elts.csv", "r"),
                        delimiter=",")
    inputs = pd.read_csv(open("dict_treillis_10elts.csv", "r"),
                        delimiter=",")
    #2 dataFrames sont créés


    #Concatenation des deux dataframes dans le même DataFrame : data
    datas = [inputs, freq]
    datas = pd.concat(datas, axis=1)

    #plot_correlation_matrix(pd.DataFrame(datas))

    # d'apres la matrice de correlation, certaines entrées sont étroitement liées
    # on va donc supprimer certaines de ces valeurs pour conserver :
    # hauteur h, base b, la masse volumique rho, la longueur de la poutre L_tot
    to_drop = ['NbElts', 'S', 'I', 'L', 'E', 'Mat','nu']
    datas = datas.drop(columns=to_drop)
    print(datas)

    datas = datas.sample(10000)
    ######################      FIN PRE-PROCESSING      ###########################




    #######################      TRAIN_TEST_SPLIT      ############################
    # 70% de la population sera allouée à l'apprentissage, 30 % pour le test
    population_train = 0.7

    # mélange et séparation de nos données en 2 datasets
    split_train, split_test = train_test_split(datas, train_size=population_train)

    # On extrait les données qui serviront d'objectif à atteindre, soit ici les
    # 8 fréquences propres à prédire

    entrees = ['L_tot','rho', 'h', 'b', 'freq1']
    split_target_train = split_train.drop(columns=entrees)
    split_target_test = split_test.drop(columns=entrees)
    print(split_target_train)

    frequences = ["freq2",'freq3','freq4','freq5','freq6','freq7','freq8']
    split_train = split_train.drop(columns=frequences)
    split_test = split_test.drop(columns=frequences)



    #######################      FIN TR_TST_SPLIT      ############################


    ##################          PROCESS LINEAIRE            #######################
    # Regression linéaire
    reg_lin = LinearRegression().fit(split_train, split_target_train)
    get_score(reg_lin, split_test, split_target_test)
    reg_poly = PolynomialRegression(3)
    reg_poly.fit(split_train, split_target_train)
    print(reg_poly.score(split_test, split_target_test))

    '''
    # plot prediction
    Y_pred_treillis_rect = reg_poly.predict(split_test)
    plt.figure(figsize=(12, 5))
    plt.plot((Y_pred_treillis_rect)[:120])
    plt.plot((np.array(split_target_test)[:120]))

    plt.legend(
        ['Y_grid_search', 'split_target_test'])
    plt.show()

    rf_reg_treillis = RandomForestRegressor(n_estimators=150, max_depth=13, random_state=0)
    rf_reg_treillis.fit(split_train, split_target_train)
    print("treillis : ", rf_reg_treillis.score(split_test, split_target_test))
    '''
    return reg_poly


reg_lin_rect = gen_model_rect()
reg_lin_cercle = gen_model_cercle()
reg_lin_rect_creux = gen_model_Rectangle_creux()
reg_lin_cercle_creux = gen_model_Cercle_creux()
reg_poly_treillis_rect = gen_model_treillis_rect()

import pickle

with open('model_rectangle_other_freq.pkl', 'wb') as f:
    pickle.dump(reg_lin_rect, f)
with open('model_cercle_other_freq.pkl', 'wb') as f:
    pickle.dump(reg_lin_cercle, f)
with open('model_rectangle_creux_other_freq.pkl', 'wb') as f:
    pickle.dump(reg_lin_rect_creux, f)
with open('model_cercle_creux_other_freq.pkl', 'wb') as f:
    pickle.dump(reg_lin_cercle_creux, f)
with open('model_rectangle_treillis_other_freq.pkl', 'wb') as f:
    pickle.dump(reg_poly_treillis_rect, f)