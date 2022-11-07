import torch
import numpy as np
import scipy.linalg as LI

from torch.utils.data import Dataset
from random import *
import csv
import pandas as pd

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#si le GPU est assez puissant, on fait tourner le GPU
#sinon on fait tourner le CPU

def create_inputs(nbElements):
   Mat = ['Acier de construction', 'Acier inoxydable', 'Aluminium', 'Cuivre', 'Titane', 'Verre', 'Beton']
   EMat = [210E9, 203E9, 69E9, 124E9, 114E9, 69E9, 30E9]
   rhoMat =[7850, 7800, 2700, 8900, 4510, 2500, 2400]
   iMat = randint(0,6)
   
   Length = np.random.uniform(low=0.1, high=1)
   Base_ext = np.random.uniform(low=Length/100, high=Length/10)
   Height_ext = np.random.uniform(low=Length/100, high=Length/10)
   Base_int = np.random.uniform(low=Base_ext / 100, high=Base_ext / 1.1)
   Height_int = np.random.uniform(low=Height_ext / 100, high=Height_ext / 1.1)
   
   d = dict();
   d['NbElts'] = nbElements
   d['L_tot'] = Length
   d['rho'] = rhoMat[iMat]
   d['h_ext'] = Height_ext
   d['b_ext'] = Base_ext
   d['h_int'] = Height_int
   d['b_int'] = Base_int
   d['S'] = d.get('b_ext')*d.get('h_ext') - d.get('b_int')*d.get('h_int')
   d['I'] = ((d.get('b_ext')*pow(d.get('h_ext'),3))/12) - (d.get('b_int')*pow(d.get('h_int'),3))/12
   d['L'] = d.get('L_tot')/d.get('NbElts')
   d['E'] = EMat[iMat]
   d['Mat'] = Mat[iMat]
   return d


def createdata(nbElements,nbFreq): #crée le jeu de données
    
    
    nbcapteur = nbFreq 

    nbre_deg_noeud = 2 #DDL par noeuds

    nb_noeuds = nbElements + 1 
    
    Params = create_inputs(nbElements)
    
    L_tot = Params.get('L_tot')
    rho = Params.get('rho')
    S = Params.get('S')
    I = Params.get('I')
    L = Params.get('L')
    E = Params.get('E')
    Mat = Params.get('Mat')
    
    #numpy matrix
    Ke2 = (I/pow(L,3))*np.matrix([[12  ,6*L,    -12,   6*L],
                             [6*L,  4*pow(L,2),  -6*L,  2*pow(L,2)],
                             [-12,  -6*L,   12,    -6*L],
                             [6*L,  2*pow(L,2),  -6*L,  4*pow(L,2)]]) # matrice de rigidité d'un élement sans E

    Me = ((rho*S*L)/420)*np.matrix([[156,   22*L,   54,     -13*L],
                                    [22*L,  4*pow(L,2),  13*L,   -3*pow(L,2)],
                      [54,    13*L,   156,    -22*L],
                     [ -13*L, -3*pow(L,2), -22*L,  4*pow(L,2)]]) # matrice de masse d'un element
                  
    sizeKe_totale = nbre_deg_noeud*nb_noeuds; #nb ddl total pour dimension Ke
    K_rigidite =[] # Matrice de rigidité : empty
    M_masse =[] # Matrice de masse : empty
    for i in range(sizeKe_totale): # dimension de K et M
        K_rigidite.append(np.zeros(sizeKe_totale)) # rempli la matrice de 0 
        M_masse.append(np.zeros(sizeKe_totale)) # rempli la matrice de 0
    
    K_rigidite = np.matrix(K_rigidite) # casté en np matrix
    M_masse = np.matrix(M_masse) # casté en np matrix

    for j in range(nbElements): # 10 éléments => 11 points => matrice 22 * 22 
        Ke = E*Ke2 # multiplie la matrice de rigidité d'un élement par chaque module d'young modifié
        #Ke = matrice 4 * 4 
        for k in range(4): #on tourne selon nos 2 dimensions
            for n in range(4):  
                K_rigidite[n+(j)*nbre_deg_noeud,k+(j)*nbre_deg_noeud] = K_rigidite[n+(j)*nbre_deg_noeud,k+(j)*nbre_deg_noeud] + Ke[n,k]
                M_masse[n+(j)*nbre_deg_noeud,k+(j)*nbre_deg_noeud] = M_masse[n+(j)*nbre_deg_noeud,k+(j)*nbre_deg_noeud] + Me[n,k]
                #rentre les Ke et Me dans la matrice globale

    #print(K_rigidite.shape)
    K_rigidite = K_rigidite[2:,2:]  #enleve les 2 premières colonnes et 2 premieres lignes 
                                    #enlève u1 et du1/dx encastrée sur le 1er point
    #print(K_rigidite.shape)
    M_masse = M_masse[2:,2:]        #enleve les 2 premières colonnes et 2 premieres lignes 
                                    #enlève u1 et du1/dx encastrée sur le 1er point
    
    Val_prop,_ = LI.eig(K_rigidite,M_masse) #eigen values


    Val_prop = Val_prop.real #extrait les valeurs propres
    Val_prop_Range= sorted(Val_prop) #triées

    Val_prop_Range =  np.sqrt( np.square(Val_prop_Range)) # valeur absolue ?
    Puls_prop = np.sqrt(Val_prop_Range) # pulsations propres
    freqHz = Puls_prop/(2*np.pi) #conversion en freq propres
    #print("freq Hz : ", freqHz)

    freq = freqHz[0:nbcapteur] # nbcapteur == nbFreq
    #print("freq : ", freq)
    return Params, np.float32(np.round(freq,4)) #Sortie fréquences en float avec 4 décimales

class myDataset(Dataset):
    def __init__(self, NBsamples):  #idf = nb defauts
        self.samples=NBsamples #sample : attribut de la classe
        NbElts = 10
        NbFreq = 8

        self.y_data = []
        self.x_data = []
        for i in range(NBsamples):
            inputs, outputs = createdata(NbElts, NbFreq)  # crée les fréquences en prenant en compte les défauts
            # passe la sortie réelle : niveau de défaut des élements associé à ces fréquences en tenseur
            self.y_data.append(torch.tensor(outputs, dtype=torch.float32))
            self.x_data.append(inputs)
        #self.i_df=i_df #nb de défaut choisi : attribut attribut de la classe
        
    def __len__(self):
        return self.samples


from torch.utils.data import DataLoader, Dataset 
targets_tensor=torch.tensor([])
print("---------start------")
sample_nb=2500
test_dataset = myDataset(sample_nb)    #choisi le dataset en fonction des données enoncées


python_list_from_pytorch_tensor = ["freq1", "freq2", "freq3", "freq4", "freq5", "freq6", "freq7", "freq8"]
dicttab = test_dataset.x_data

for i in range(len(test_dataset.y_data)):
    python_list_from_pytorch_tensor = np.vstack((python_list_from_pytorch_tensor, test_dataset.y_data[i].tolist()))


with open('test_Rectangle_Creux.csv', 'w') as f:
    writer = csv.writer(f)
    header = ''
    for i in python_list_from_pytorch_tensor:
        data = ''
        for j in i:
            data = data + str(j) + ','
        data= data[:-1]
        f.write(data)
        f.write('\n')        
    

with open('dict_Rectangle_Creux.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    header = ''
    for key,_ in dicttab[0].items():
        header = header + str(key) + ','
    header=header[:-1]
    csv_file.write(header)
    csv_file.write('\n')
    for i in dicttab:
        data = ''
        for _, value in i.items():
            data = data + str(value) + ','
        data= data[:-1]
        csv_file.write(data)
        csv_file.write('\n')



print("---------end------")
