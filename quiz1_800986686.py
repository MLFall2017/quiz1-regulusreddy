# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 09:53:14 2017

@author: lalit
"""

#Quiz 09/22/2017

import numpy as np
from numpy import genfromtxt
from numpy import linalg as lg
from matplotlib import pyplot as plt



#importing the dataset
data_1 = genfromtxt("C:\Users\lalit\Desktop\dataset_1.csv", delimiter=",")
data_1

#deleting the column names
data = np.delete(data_1,(0),axis=0)
data

#converting numpy array to numpy matrix

#1. question 1
#reading each column into x, y,and z
x= data[0:1000,0]
x

y=data[0:1000,1]
y

z=data[0:1000,2]
z

#variance of x, y and z
np.var(x)
np.var(y)
np.var(z)

#covariance between x and y & between y and z
np.cov(x,y)
np.cov(y,z)


#converting numpy array to matrix
data = np.asmatrix(data)
data
type(data)
data.shape

#transpose of matrix data
data_transpose = data.transpose()
data_transpose.shape

#calculating mean center
mean_vec=np.mean(data,axis=0)
mean_vec.shape

#calculating covariance matrix
covariance_matrix = np.divide(((data-mean_vec).T.dot(data-mean_vec).T),999)
covariance_matrix
print('Covariance Matrix \n%s'%covariance_matrix)



#performing eigen value decomposition
eigen_vals, eigen_vecs = lg.eig(np.cov(data.T))
print('Eigen Vectors \n %s' %eigen_vecs)
print('Eigen Values \n %s' %eigen_vals)

# create a list of tuple (eigenvalue, eigenvector) 
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eigen_pairs:
    print(i[0])


#explained variance    
total = sum(eigen_vals)
var_exp = [(i / total)*100 for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)    
    

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(7, 7))

    plt.bar(range(3), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
   
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    
    

#projection matrix
matrix_w = np.hstack((eigen_pairs[0][1].reshape(3,1),
                      eigen_pairs[1][1].reshape(3,1)))

print(matrix_w)    



#projecting to new space
Y = data.dot(matrix_w)
Y.shape

plt.plot(Y[0:1000,0], Y[0:1000,1], 'o', markersize=2, color='blue', alpha=0.5)
plt.xlim([-4,8])
plt.ylim([-4,4])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.title('Transformed samples')
plt.show()





#question 3.2
#verifying 3.1

a = np.matrix('0 -1; 2 3')
print(a)

#performing eigen value decomposition
eigen_vals_a, eigen_vecs_a = lg.eig(a)
print('Eigen Vectors \n %s' %eigen_vecs)
print('Eigen Values \n %s' %eigen_vals)







