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
   Base = np.random.uniform(low=Length/100, high=Length/10)
   Height = np.random.uniform(low=Length/100, high=Length/10)
   
   d1 = dict();
   d1['NbElts_1'] = nbElements
   d1['L_tot_1'] = Length
   d1['rho_1'] = rhoMat[iMat]
   d1['h_1'] = Height
   d1['b_1'] = Base
   d1['S_1'] = d1.get('b_1')*d1.get('h_1')
   d1['I_1'] = (d1.get('b_1')*pow(d1.get('h_1'),3))/12
   d1['L_1'] = d1.get('L_tot_1')/d1.get('NbElts_1')
   d1['E_1'] = EMat[iMat]
   d1['Mat_1'] = Mat[iMat]
   d1['nu_1'] = 0.3

   Mat = ['Acier de construction', 'Acier inoxydable', 'Aluminium', 'Cuivre', 'Titane', 'Verre', 'Beton']
   EMat = [210E9, 203E9, 69E9, 124E9, 114E9, 69E9, 30E9]
   rhoMat = [7850, 7800, 2700, 8900, 4510, 2500, 2400]
   iMat = randint(0, 6)

   Length = np.random.uniform(low=0.1, high=1)
   Base = np.random.uniform(low=Length / 100, high=Length / 10)
   Height = np.random.uniform(low=Length / 100, high=Length / 10)

   d2 = dict();
   d2['NbElts_2'] = nbElements
   d2['L_tot_2'] = Length
   d2['rho_2'] = rhoMat[iMat]
   d2['h_2'] = Height
   d2['b_2'] = Base
   d2['S_2'] = d2.get('b_2') * d2.get('h_2')
   d2['I_2'] = (d2.get('b_2') * pow(d2.get('h_2'), 3)) / 12
   d2['L_2'] = d2.get('L_tot_2') / d2.get('NbElts_2')
   d2['E_2'] = EMat[iMat]
   d2['Mat_2'] = Mat[iMat]
   d2['nu_2'] = 0.3


   return d1, d2


def createdata(nbElements,nbFreq): #crée le jeu de données
    
    
    nbcapteur = nbFreq 

    nbre_deg_noeud = 6 #DDL par noeuds

    
    Params1, Params2 = create_inputs(nbElements)
    # PROCESS POUTRE 1
    L_tot = Params1.get('L_tot_1')
    rho = Params1.get('rho_1')
    h = Params1.get('h_1')
    b = Params1.get('b_1')
    S = Params1.get('S_1')
    I = Params1.get('I_1')
    L = Params1.get('L_1')
    E = Params1.get('E_1')
    Mat = Params1.get('Mat_1')
    nu = Params1.get('nu_1')
    G=E/(2*(1+nu))  

    Iy=(b*(h**3))/12
    Iz=(h*(b**3))/12
    Jpol=Iy+Iz
    
   # convert this matlab code to python
# http://www.mathworks.com/help/matlab/ref/fft.html
    import time
    from matplotlib import pyplot as plt
    import numpy as np


    # convert this matlab code to python
    start = time.time()


 

    nb_elem=10

    Ke_loc_b=(E*S/L)*np.matrix([[1, -1], [1 ,-1]])
 


    Ke_loc_fxy=(E*Iz/L**3)*np.matrix([[12 ,  6*L  ,   -12  ,  6*L],
                        [6*L, 4*(L**2), -6*L , 2*(L)**2],
                        [-12 , -6*L  ,   12   , -6*L],
                        [6*L , 2*(L**2) , -6*L , 4*(L)**2]])

    Ke_loc_fzx=(E*Iy/L**3)*np.matrix([[12 ,  6*L  ,   -12  ,  6*L],
                            [6*L, 4*(L**2), -6*L , 2*(L)**2],
                                [-12 , -6*L  ,   12   , -6*L],
                                [6*L , 2*(L**2) , -6*L , 4*(L)**2]])


    Ke_loc_t=(G*Jpol/L)*np.matrix([[1, -1], [1 ,-1]])

    Me_loc_b=(rho*S*L/6)*np.matrix([[2, 1], [1 ,2]])

    Me_loc_fxy=(rho*S*L/420)*np.matrix([[156 ,  22*L  ,   54  , -13*L],
                                [22*L, 4*(L**2), 13*L , -3*(L)**2],
                                [54 , 13*L  ,   156   , -22*L],
                                [-13*L , -3*(L**2) , -22*L , 4*(L)**2]])

    Me_loc_fzx=(rho*S*L/420)*np.matrix([[156 ,  22*L  ,   54  , -13*L],
                                [22*L, 4*(L**2), 13*L , -3*(L)**2],
                                [54 , 13*L  ,   156   , -22*L],
                                [-13*L , -3*(L**2) , -22*L , 4*(L)**2]])


                        
    Me_loc_t=(rho*Jpol*L/6)*np.matrix([[2, 1], [1 ,2]])




    Ke_loc = np.zeros((12,12))



    """"

    """
    Ke_loc[0,0] = Ke_loc_b[0,0]
    Ke_loc[0,6] = Ke_loc_b[0,1]
    Ke_loc[6,0] = Ke_loc_b[1,0]
    Ke_loc[6,6] = Ke_loc_b[1,1]

    Ke_loc[1,1] = Ke_loc_fxy[0,0]
    Ke_loc[1,5] = Ke_loc_fxy[0,1]
    Ke_loc[1,7] = Ke_loc_fxy[0,2]
    Ke_loc[1,11] = Ke_loc_fxy[0,3]

    Ke_loc[5,1] = Ke_loc_fxy[1,0]
    Ke_loc[5,5] = Ke_loc_fxy[1,1]
    Ke_loc[5,7] = Ke_loc_fxy[1,2]
    Ke_loc[5,11] = Ke_loc_fxy[1,3]

    Ke_loc[7,1] = Ke_loc_fxy[2,0]
    Ke_loc[7,5] = Ke_loc_fxy[2,1]
    Ke_loc[7,7] = Ke_loc_fxy[2,2]
    Ke_loc[7,11] = Ke_loc_fxy[2,3]

    Ke_loc[11,1] = Ke_loc_fxy[3,0]
    Ke_loc[11,5] = Ke_loc_fxy[3,1]
    Ke_loc[11,7] = Ke_loc_fxy[3,2]
    Ke_loc[11,11] = Ke_loc_fxy[3,3]

    Ke_loc[2,2] = Ke_loc_fzx[0,0]
    Ke_loc[2,4] = Ke_loc_fzx[0,1]
    Ke_loc[2,8] = Ke_loc_fzx[0,2]
    Ke_loc[2,10] = Ke_loc_fzx[0,3]

    Ke_loc[4,2] = Ke_loc_fzx[1,0]
    Ke_loc[4,4] = Ke_loc_fzx[1,1]
    Ke_loc[4,8] = Ke_loc_fzx[1,2]
    Ke_loc[4,10] = Ke_loc_fzx[1,3]

    Ke_loc[8,2] = Ke_loc_fzx[2,0]
    Ke_loc[8,4] = Ke_loc_fzx[2,1]
    Ke_loc[8,8] = Ke_loc_fzx[2,2]
    Ke_loc[8,10] = Ke_loc_fzx[2,3]

    Ke_loc[10,2] = Ke_loc_fzx[3,0]
    Ke_loc[10,4] = Ke_loc_fzx[3,1]
    Ke_loc[10,8] = Ke_loc_fzx[3,2]
    Ke_loc[10,10] = Ke_loc_fzx[3,3]

    Ke_loc[3,3] = Ke_loc_t[0,0]
    Ke_loc[3,9] = Ke_loc_t[0,1]
    Ke_loc[9,3] = Ke_loc_t[1,0]
    Ke_loc[9,9] = Ke_loc_t[1,1]

    theta = 0
    rot = np.matrix([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0],[0,0,1]])
    rot = np.kron(np.eye(4), rot)
    Ke_loc = rot.T*Ke_loc*rot

    Me_loc = np.zeros((12,12))
    Me_loc[0,0] = Me_loc_b[0,0]
    Me_loc[0,6] = Me_loc_b[0,1]
    Me_loc[6,0] = Me_loc_b[1,0]
    Me_loc[6,6] = Me_loc_b[1,1]

    Me_loc[1,1] = Me_loc_fxy[0,0]
    Me_loc[1,5] = Me_loc_fxy[0,1]
    Me_loc[1,7] = Me_loc_fxy[0,2]
    Me_loc[1,11] = Me_loc_fxy[0,3]

    Me_loc[5,1] = Me_loc_fxy[1,0]
    Me_loc[5,5] = Me_loc_fxy[1,1]
    Me_loc[5,7] = Me_loc_fxy[1,2]
    Me_loc[5,11] = Me_loc_fxy[1,3]

    Me_loc[7,1] = Me_loc_fxy[2,0]
    Me_loc[7,5] = Me_loc_fxy[2,1]
    Me_loc[7,7] = Me_loc_fxy[2,2]
    Me_loc[7,11] = Me_loc_fxy[2,3]

    Me_loc[11,1] = Me_loc_fxy[3,0]
    Me_loc[11,5] = Me_loc_fxy[3,1]
    Me_loc[11,7] = Me_loc_fxy[3,2]
    Me_loc[11,11] = Me_loc_fxy[3,3]

    Me_loc[2,2] = Me_loc_fzx[0,0]
    Me_loc[2,4] = Me_loc_fzx[0,1]
    Me_loc[2,8] = Me_loc_fzx[0,2]
    Me_loc[2,10] = Me_loc_fzx[0,3]

    Me_loc[4,2] = Me_loc_fzx[1,0]
    Me_loc[4,4] = Me_loc_fzx[1,1]
    Me_loc[4,8] = Me_loc_fzx[1,2]
    Me_loc[4,10] = Me_loc_fzx[1,3]

    Me_loc[8,2] = Me_loc_fzx[2,0]
    Me_loc[8,4] = Me_loc_fzx[2,1]
    Me_loc[8,8] = Me_loc_fzx[2,2]
    Me_loc[8,10] = Me_loc_fzx[2,3]

    Me_loc[10,2] = Me_loc_fzx[3,0]
    Me_loc[10,4] = Me_loc_fzx[3,1]
    Me_loc[10,8] = Me_loc_fzx[3,2]
    Me_loc[10,10] = Me_loc_fzx[3,3]

    Me_loc[3,3] = Me_loc_t[0,0]
    Me_loc[3,9] = Me_loc_t[0,1]
    Me_loc[9,3] = Me_loc_t[1,0]
    Me_loc[9,9] = Me_loc_t[1,1]

    theta = 0
    rot = np.matrix([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0],[0,0,1]])
    rot = np.kron(np.eye(4), rot)
    Me_loc = rot.T*Me_loc*rot


    nodes = np.zeros((nb_elem+1,3))
    for i in range(nb_elem+1):
        nodes[i,0] = (-i)*L
        nodes[i,1] = 0
        nodes[i,2] = 0

    elements = np.zeros((nb_elem,2))
    for i in range(nb_elem):
        elements[i,0] = i+1
        elements[i,1] = i+2

    #  sparse matrix assembly
    np.zeros(((nb_elem+1)*6, (nb_elem+1)*6))
    K = np.zeros(((nb_elem+1)*6, (nb_elem+1)*6))
    M = np.zeros(((nb_elem+1)*6, (nb_elem+1)*6))



  
    nume = np.zeros((1,12))

    for i in range(nb_elem):
        nume = [6*elements[i,0]-6, 6*elements[i,0]-5, 6*elements[i,0]-4, 6*elements[i,0]-3, 6*elements[i,0]-2, 6*elements[i,0]-1, 6*elements[i,1]-6, 6*elements[i,1]-5, 6*elements[i,1]-4, 6*elements[i,1]-3, 6*elements[i,1]-2, 6*elements[i,1]-1]
        #convert every element of nume to int
        nume = [int(i) for i in nume]
        K[np.ix_(nume,nume)] = K[np.ix_(nume,nume)] + Ke_loc
        M[np.ix_(nume,nume)] = M[np.ix_(nume,nume)] + Me_loc

    K1 = K
    M1 = M
    nodes1 = nodes
    elements1 = elements
    Nb_elem1 = nb_elem
    L1 = L

    # PROCESS POUTRE 2
    L_tot = Params2.get('L_tot_2')
    rho = Params2.get('rho_2')
    h = Params2.get('h_2')
    b = Params2.get('b_2')
    S = Params2.get('S_2')
    I = Params2.get('I_2')
    L = Params2.get('L_2')
    E = Params2.get('E_2')
    Mat = Params2.get('Mat_2')
    nu = Params2.get('nu_2')
    G = E / (2 * (1 + nu))

    Iy = (b * (h ** 3)) / 12
    Iz = (h * (b ** 3)) / 12
    Jpol = Iy + Iz


    Ke_loc_b=(E*S/L)*np.matrix([[1, -1], [1 ,-1]])
  

    Ke_loc_fxy=(E*Iz/L**3)*np.matrix([[12 ,  6*L  ,   -12  ,  6*L],
                        [6*L, 4*(L**2), -6*L , 2*(L)**2],
                        [-12 , -6*L  ,   12   , -6*L],
                        [6*L , 2*(L**2) , -6*L , 4*(L)**2]])

    Ke_loc_fzx=(E*Iy/L**3)*np.matrix([[12 ,  6*L  ,   -12  ,  6*L],
                            [6*L, 4*(L**2), -6*L , 2*(L)**2],
                                [-12 , -6*L  ,   12   , -6*L],
                                [6*L , 2*(L**2) , -6*L , 4*(L)**2]])


    Ke_loc_t=(G*Jpol/L)*np.matrix([[1, -1], [1 ,-1]])

    Me_loc_b=(rho*S*L/6)*np.matrix([[2, 1], [1 ,2]])

    Me_loc_fxy=(rho*S*L/420)*np.matrix([[156 ,  22*L  ,   54  , -13*L],
                                [22*L, 4*(L**2), 13*L , -3*(L)**2],
                                [54 , 13*L  ,   156   , -22*L],
                                [-13*L , -3*(L**2) , -22*L , 4*(L)**2]])

    Me_loc_fzx=(rho*S*L/420)*np.matrix([[156 ,  22*L  ,   54  , -13*L],
                                [22*L, 4*(L**2), 13*L , -3*(L)**2],
                                [54 , 13*L  ,   156   , -22*L],
                                [-13*L , -3*(L**2) , -22*L , 4*(L)**2]])


                        
    Me_loc_t=(rho*Jpol*L/6)*np.matrix([[2, 1], [1 ,2]])




    Ke_loc = np.zeros((12,12))



    """"

    """
    Ke_loc[0,0] = Ke_loc_b[0,0]
    Ke_loc[0,6] = Ke_loc_b[0,1]
    Ke_loc[6,0] = Ke_loc_b[1,0]
    Ke_loc[6,6] = Ke_loc_b[1,1]

    Ke_loc[1,1] = Ke_loc_fxy[0,0]
    Ke_loc[1,5] = Ke_loc_fxy[0,1]
    Ke_loc[1,7] = Ke_loc_fxy[0,2]
    Ke_loc[1,11] = Ke_loc_fxy[0,3]

    Ke_loc[5,1] = Ke_loc_fxy[1,0]
    Ke_loc[5,5] = Ke_loc_fxy[1,1]
    Ke_loc[5,7] = Ke_loc_fxy[1,2]
    Ke_loc[5,11] = Ke_loc_fxy[1,3]

    Ke_loc[7,1] = Ke_loc_fxy[2,0]
    Ke_loc[7,5] = Ke_loc_fxy[2,1]
    Ke_loc[7,7] = Ke_loc_fxy[2,2]
    Ke_loc[7,11] = Ke_loc_fxy[2,3]

    Ke_loc[11,1] = Ke_loc_fxy[3,0]
    Ke_loc[11,5] = Ke_loc_fxy[3,1]
    Ke_loc[11,7] = Ke_loc_fxy[3,2]
    Ke_loc[11,11] = Ke_loc_fxy[3,3]

    Ke_loc[2,2] = Ke_loc_fzx[0,0]
    Ke_loc[2,4] = Ke_loc_fzx[0,1]
    Ke_loc[2,8] = Ke_loc_fzx[0,2]
    Ke_loc[2,10] = Ke_loc_fzx[0,3]

    Ke_loc[4,2] = Ke_loc_fzx[1,0]
    Ke_loc[4,4] = Ke_loc_fzx[1,1]
    Ke_loc[4,8] = Ke_loc_fzx[1,2]
    Ke_loc[4,10] = Ke_loc_fzx[1,3]

    Ke_loc[8,2] = Ke_loc_fzx[2,0]
    Ke_loc[8,4] = Ke_loc_fzx[2,1]
    Ke_loc[8,8] = Ke_loc_fzx[2,2]
    Ke_loc[8,10] = Ke_loc_fzx[2,3]

    Ke_loc[10,2] = Ke_loc_fzx[3,0]
    Ke_loc[10,4] = Ke_loc_fzx[3,1]
    Ke_loc[10,8] = Ke_loc_fzx[3,2]
    Ke_loc[10,10] = Ke_loc_fzx[3,3]

    Ke_loc[3,3] = Ke_loc_t[0,0]
    Ke_loc[3,9] = Ke_loc_t[0,1]
    Ke_loc[9,3] = Ke_loc_t[1,0]
    Ke_loc[9,9] = Ke_loc_t[1,1]

    theta = np.pi/2
    rot = np.matrix([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0],[0,0,1]])
    rot = np.kron(np.eye(4), rot)
    Ke_loc = rot.T*Ke_loc*rot

    Me_loc = np.zeros((12,12))
    Me_loc[0,0] = Me_loc_b[0,0]
    Me_loc[0,6] = Me_loc_b[0,1]
    Me_loc[6,0] = Me_loc_b[1,0]
    Me_loc[6,6] = Me_loc_b[1,1]

    Me_loc[1,1] = Me_loc_fxy[0,0]
    Me_loc[1,5] = Me_loc_fxy[0,1]
    Me_loc[1,7] = Me_loc_fxy[0,2]
    Me_loc[1,11] = Me_loc_fxy[0,3]

    Me_loc[5,1] = Me_loc_fxy[1,0]
    Me_loc[5,5] = Me_loc_fxy[1,1]
    Me_loc[5,7] = Me_loc_fxy[1,2]
    Me_loc[5,11] = Me_loc_fxy[1,3]

    Me_loc[7,1] = Me_loc_fxy[2,0]
    Me_loc[7,5] = Me_loc_fxy[2,1]
    Me_loc[7,7] = Me_loc_fxy[2,2]
    Me_loc[7,11] = Me_loc_fxy[2,3]

    Me_loc[11,1] = Me_loc_fxy[3,0]
    Me_loc[11,5] = Me_loc_fxy[3,1]
    Me_loc[11,7] = Me_loc_fxy[3,2]
    Me_loc[11,11] = Me_loc_fxy[3,3]

    Me_loc[2,2] = Me_loc_fzx[0,0]
    Me_loc[2,4] = Me_loc_fzx[0,1]
    Me_loc[2,8] = Me_loc_fzx[0,2]
    Me_loc[2,10] = Me_loc_fzx[0,3]

    Me_loc[4,2] = Me_loc_fzx[1,0]
    Me_loc[4,4] = Me_loc_fzx[1,1]
    Me_loc[4,8] = Me_loc_fzx[1,2]
    Me_loc[4,10] = Me_loc_fzx[1,3]

    Me_loc[8,2] = Me_loc_fzx[2,0]
    Me_loc[8,4] = Me_loc_fzx[2,1]
    Me_loc[8,8] = Me_loc_fzx[2,2]
    Me_loc[8,10] = Me_loc_fzx[2,3]

    Me_loc[10,2] = Me_loc_fzx[3,0]
    Me_loc[10,4] = Me_loc_fzx[3,1]
    Me_loc[10,8] = Me_loc_fzx[3,2]
    Me_loc[10,10] = Me_loc_fzx[3,3]

    Me_loc[3,3] = Me_loc_t[0,0]
    Me_loc[3,9] = Me_loc_t[0,1]
    Me_loc[9,3] = Me_loc_t[1,0]
    Me_loc[9,9] = Me_loc_t[1,1]

    theta = np.pi/2
    rot = np.matrix([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0],[0,0,1]])
    rot = np.kron(np.eye(4), rot)
    Me_loc = rot.T*Me_loc*rot


    nodes = np.zeros((nb_elem+1,3))
    for i in range(nb_elem+1):
        nodes[i,0] = 0
        nodes[i,1] = i*L
        nodes[i,2] = 0

 
        
    elements = np.zeros((nb_elem,2))
    for i in range(nb_elem):
        elements[i,0] = i+1
        elements[i,1] = i+2

    #  sparse matrix assembly
    np.zeros(((nb_elem+1)*6, (nb_elem+1)*6))
    K = np.zeros(((nb_elem+1)*6, (nb_elem+1)*6))
    M = np.zeros(((nb_elem+1)*6, (nb_elem+1)*6))


    nume = np.zeros((1,12))

    for i in range(nb_elem):
        nume = [6*elements[i,0]-6, 6*elements[i,0]-5, 6*elements[i,0]-4, 6*elements[i,0]-3, 6*elements[i,0]-2, 6*elements[i,0]-1, 6*elements[i,1]-6, 6*elements[i,1]-5, 6*elements[i,1]-4, 6*elements[i,1]-3, 6*elements[i,1]-2, 6*elements[i,1]-1]
        #convert every element of nume to int
        nume = [int(i) for i in nume]
        K[np.ix_(nume,nume)] = K[np.ix_(nume,nume)] + Ke_loc
        M[np.ix_(nume,nume)] = M[np.ix_(nume,nume)] + Me_loc


    K2 = K
    M2 = M
    # write M2 and K2 to file
    np.savetxt('K2.txt', K2, delimiter=',')
    np.savetxt('M2.txt', M2, delimiter=',')

    nodes2 = nodes
    nb_elem2 = nb_elem
    L2 = L

    # plot3D nodes 1 and 2
    """fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(nb_elem+1):
        ax.scatter(nodes1[i,0], nodes1[i,1], nodes1[i,2], c='r', marker='o')
        ax.scatter(nodes2[i,0], nodes2[i,1], nodes2[i,2], c='b', marker='o')
    plt.show()"""

    Mn1 = Nb_elem1+1
    Mn2 = Mn1+nb_elem2

    Kgl1 = np.zeros((Mn2*6, Mn2*6))
    Kgl2 = np.zeros((Mn2*6, Mn2*6))

    Mgl1 = np.zeros((Mn2*6, Mn2*6))
    Mgl2 = np.zeros((Mn2*6, Mn2*6))

    num1 = np.zeros((1,6*Mn1))
    num2 = np.zeros((1,6*Mn1))
  
    # 6 first element of num2 are 1,2,3,4,5,6 then continue to increment from 66 etc
    for i in range(len(num1[0])):
        num1[0,i] = i+1

 
    for i in range(6):  
        num2[0,i] = i+1



    for i in range(6,len(num2[0])):
        num2[0,i] = i+60+1

    """for i in range(nb_elem):
        nume = [6*elements[i,0]-6, 6*elements[i,0]-5, 6*elements[i,0]-4, 6*elements[i,0]-3, 6*elements[i,0]-2, 6*elements[i,0]-1, 6*elements[i,1]-6, 6*elements[i,1]-5, 6*elements[i,1]-4, 6*elements[i,1]-3, 6*elements[i,1]-2, 6*elements[i,1]-1]
        #convert every element of nume to int
        nume = [int(i) for i in nume]
        K[np.ix_(nume,nume)] = K[np.ix_(nume,nume)] + Ke_loc
        M[np.ix_(nume,nume)] = M[np.ix_(nume,nume)] + Me_loc"""

    num1 = num1.tolist()
    num2 = num2.tolist()
    
    # make num1 and num2 1D
    num1 = [item for sublist in num1 for item in sublist]
    num2 = [item for sublist in num2 for item in sublist]

    # convert to int
    num1 = [int(i) for i in num1]
    num2 = [int(i) for i in num2]

    # witdraw 1 from num1
    num1 = [i-1 for i in num1]
    num2 = [i-1 for i in num2]


    Kgl1[np.ix_(num1,num1)] = K1[np.ix_(num1,num1)]
    Kgl2[np.ix_(num2,num2)] = K2[np.ix_(num1,num1)]

    Mgl1[np.ix_(num1,num1)] = M1[np.ix_(num1,num1)]
    Mgl2[np.ix_(num2,num2)] = M2[np.ix_(num1,num1)]

    Kgl = Kgl1+Kgl2
    Mgl = Mgl1+Mgl2

    # spy mgl

    # delte 6 first rows and columns
    Kgl = np.delete(Kgl, np.s_[0:6], axis=0)
    Kgl = np.delete(Kgl, np.s_[0:6], axis=1)
    # delete 6 last rows and columns
    Kgl = np.delete(Kgl, np.s_[-6:], axis=0)
    Kgl = np.delete(Kgl, np.s_[-6:], axis=1)

    Mgl = np.delete(Mgl, np.s_[0:6], axis=0)
    Mgl = np.delete(Mgl, np.s_[0:6], axis=1)

    Mgl = np.delete(Mgl, np.s_[-6:], axis=0)
    Mgl = np.delete(Mgl, np.s_[-6:], axis=1)

    np.savetxt('Kgl.txt', Kgl, delimiter=',')
    np.savetxt('Mgl.txt', Mgl, delimiter=',')
    Mgl_inv = np.linalg.inv(Mgl)
    np.savetxt('Mgl_inv.txt', Mgl_inv, delimiter=',')

    # caculate natural frequencies
    eigval, eigvec = np.linalg.eig(np.dot(Mgl_inv, Kgl))
    eigval = np.sqrt(eigval)
    eigval = np.sort(eigval)
    eigval = eigval/2/np.pi
    # delete all value under 1 
    eigval = eigval[eigval>4]
    # delete all value over 10000
    eigval = eigval[0:nbcapteur]

    #concatenate dict Parameters Params1 and Params2
    Params = {**Params1, **Params2}

    

    return Params, np.float32(np.round(eigval, decimals=4))


class myDataset(Dataset):
    def __init__(self, NBsamples):  #idf = nb defauts
        self.samples=NBsamples #sample : attribut de la classe

        NbFreq = 8
        self.y_data = []
        self.x_data = []
        for i in range(NBsamples):
            NbElts = randint(10, 50)
            if i % 100 == 0:
                print(i)
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
sample_nb=25000
test_dataset = myDataset(sample_nb)    #choisi le dataset en fonction des données enoncées


python_list_from_pytorch_tensor = ["freq1", "freq2", "freq3", "freq4", "freq5", "freq6", "freq7", "freq8"]
dicttab = test_dataset.x_data

for i in range(len(test_dataset.y_data)):
    python_list_from_pytorch_tensor = np.vstack((python_list_from_pytorch_tensor, test_dataset.y_data[i].tolist()))


with open('test_treillis.csv', 'w') as f:
    writer = csv.writer(f)
    header = ''
    for i in python_list_from_pytorch_tensor:
        data = ''
        for j in i:
            data = data + str(j) + ','
        data= data[:-1]
        f.write(data)
        f.write('\n')        
    

with open('dict_treillis.csv', 'w') as csv_file:
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
