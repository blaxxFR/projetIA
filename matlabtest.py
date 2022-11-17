# convert this matlab code to python
# http://www.mathworks.com/help/matlab/ref/fft.html
from matplotlib import pyplot as plt
import numpy as np


# convert this matlab code to python



L=1  
N=10  
Le=L/N 
E=210e9 
nu=0.3 
G=E/(2*(1+nu))  
rho=7800
b=0.0249  
h=0.0053 
S=b*h
Iy=(b*(h**3))/12
Iz=(h*(b**3))/12
Jpol=Iy+Iz

nb_elem=10

Ke_loc_b=(E*S/Le)*np.matrix([[1, -1], [1 ,-1]])
# print inv of Ke_loc_b

# print Ke_loc_b


Ke_loc_fxy=(E*Iz/Le**3)*np.matrix([[12 ,  6*Le  ,   -12  ,  6*Le],
                       [6*Le, 4*(Le**2), -6*Le , 2*(Le)**2],
                       [-12 , -6*Le  ,   12   , -6*Le],
                       [6*Le , 2*(Le**2) , -6*Le , 4*(Le)**2]])

Ke_loc_fzx=(E*Iy/Le**3)*np.matrix([[12 ,  6*Le  ,   -12  ,  6*Le],
                          [6*Le, 4*(Le**2), -6*Le , 2*(Le)**2],
                            [-12 , -6*Le  ,   12   , -6*Le],
                            [6*Le , 2*(Le**2) , -6*Le , 4*(Le)**2]])


Ke_loc_t=(G*Jpol/Le)*np.matrix([[1, -1], [1 ,-1]])

Me_loc_b=(rho*S*Le/6)*np.matrix([[2, 1], [1 ,2]])

Me_loc_fxy=(rho*Iz*Le/420)*np.matrix([[156 ,  22*Le  ,   54  , -13*Le],
                            [22*Le, 4*(Le**2), 13*Le , -3*(Le)**2],
                            [54 , 13*Le  ,   156   , -22*Le],
                            [-13*Le , -3*(Le**2) , -22*Le , 4*(Le)**2]])

Me_loc_fzx=(rho*Iy*Le/420)*np.matrix([[156 ,  22*Le  ,   54  , -13*Le],
                            [22*Le, 4*(Le**2), 13*Le , -3*(Le)**2],
                            [54 , 13*Le  ,   156   , -22*Le],
                            [-13*Le , -3*(Le**2) , -22*Le , 4*(Le)**2]])


                       
Me_loc_t=(rho*Jpol*Le/6)*np.matrix([[2, 1], [1 ,2]])




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
    nodes[i,0] = (-i)*Le
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



print(np.shape(K))
nume = np.zeros((1,12))

for i in range(nb_elem):
    nume = [6*elements[i,0]-6, 6*elements[i,0]-5, 6*elements[i,0]-4, 6*elements[i,0]-3, 6*elements[i,0]-2, 6*elements[i,0]-1, 6*elements[i,1]-6, 6*elements[i,1]-5, 6*elements[i,1]-4, 6*elements[i,1]-3, 6*elements[i,1]-2, 6*elements[i,1]-1]
    #convert every element of nume to int
    nume = [int(i) for i in nume]
    K[np.ix_(nume,nume)] = K[np.ix_(nume,nume)] + Ke_loc
    M[np.ix_(nume,nume)] = M[np.ix_(nume,nume)] + Me_loc
    










# Assemble global stiffness matrix





























"""

"""



