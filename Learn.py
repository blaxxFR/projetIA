#VISUALISATION ET IMPORT DE DONNEES
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt



#SKLEARN FUNCTIONS
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate


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
######################      FIN PRE-PROCESSING      ###########################




#######################      TRAIN_TEST_SPLIT      ############################
# 70% de la population sera allouée à l'apprentissage, 30 % pour le test
population_train = 0.7

# mélange et séparation de nos données en 2 datasets  
split_train, split_test = train_test_split(datas, train_size=population_train)

# On extrait les données qui serviront d'objectif à atteindre, soit ici les 
# 8 fréquences propres à prédire
entrees = ['NbElts', 'L_tot', 'rho', 'h',
          'b', 'S', 'I', 'L', 'E']
split_target_train = split_train.drop(columns=entrees)
split_target_test = split_test.drop(columns=entrees)


frequences = ["freq1", "freq2", "freq3", "freq4", 
              "freq5", "freq6", "freq7", "freq8"]
split_train = split_train.drop(columns=frequences)
split_test = split_test.drop(columns=frequences)


#split_train = entrees servant à entrainer le modèle
#split test = entrees servant à tester le modèle
#split_target_train = sorties d'entrainement du modèle
#split_target_test = sorties de test du modèle 


#######################      FIN TR_TST_SPLIT      ############################

# PROCESS AVEC TRAIN TEST SPLIT #

#######################      CROSS VALIDATION      ############################



####################      FIN CROSS VALIDATION       ##########################




