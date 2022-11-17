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

Me_loc_fxy=(rho*S*Le/420)*np.matrix([[156 ,  22*Le  ,   54  , -13*Le],
                            [22*Le, 4*(Le**2), 13*Le , -3*(Le)**2],
                            [54 , 13*Le  ,   156   , -22*Le],
                            [-13*Le , -3*(Le**2) , -22*Le , 4*(Le)**2]])

Me_loc_fzx=(rho*S*Le/420)*np.matrix([[156 ,  22*Le  ,   54  , -13*Le],
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

K1 = K
M1 = M
nodes1 = nodes
elements1 = elements
Nb_elem1 = nb_elem
L1 = L

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

Me_loc_fxy=(rho*S*Le/420)*np.matrix([[156 ,  22*Le  ,   54  , -13*Le],
                            [22*Le, 4*(Le**2), 13*Le , -3*(Le)**2],
                            [54 , 13*Le  ,   156   , -22*Le],
                            [-13*Le , -3*(Le**2) , -22*Le , 4*(Le)**2]])

Me_loc_fzx=(rho*S*Le/420)*np.matrix([[156 ,  22*Le  ,   54  , -13*Le],
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
    nodes[i,1] = i*Le
    nodes[i,2] = 0

print(nodes)
    
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
print(np.shape(num2))
# 6 first element of num2 are 1,2,3,4,5,6 then continue to increment from 66 etc
for i in range(len(num1[0])):
    num1[0,i] = i+1

print(num1)

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
print(np.shape(num2))
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
eigval = eigval[eigval>1]
print(eigval)

# plot natural frequencies
plt.plot(eigval)
plt.show()


























    

# Assemble global stiffness matrix















