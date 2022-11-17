# convert this matlab code to python
# http://www.mathworks.com/help/matlab/ref/fft.html
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



Ke_loc_b=(E*S/Le)*np.matrix([[1, -1], [1 ,-1]])


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

Ke_loc[[0,6],[0,6]] = Ke_loc_b
Ke_loc[[1,5,7,11],[1,5,7,11]] = Ke_loc_fxy
Ke_loc[[2,4,8,10],[2,4,8,10]] = Ke_loc_fzx
Ke_loc[[3,9],[3,9]] = Ke_loc_t

#numpy spy
import matplotlib.pyplot as plt
plt.spy(Ke_loc)
plt.show()



