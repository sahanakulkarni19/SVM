from sklearn import datasets  ## importing datasets from sklearn

digits = datasets.load_digits()  ### loading data from scikit_learn library

print(digits.DESCR)  ## getting information about the data

digits.images[0]

array([[0., 0., 5., 13., 9., 1., 0., 0.],
       [0., 0., 13., 15., 10., 15., 5., 0.],
       [0., 3., 15., 2., 0., 11., 8., 0.],
       [0., 4., 12., 0., 0., 8., 8., 0.],
       [0., 5., 8., 0., 0., 9., 8., 0.],
       [0., 4., 11., 0., 1., 12., 7., 0.],
       [0., 2., 14., 5., 10., 12., 0., 0.],
       [0., 0., 6., 13., 10., 0., 0., 0.]])

import matplotlib.pyplot as plt
% matplotlib
inline
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')  ## visualizing image of '0'

digits.target  ## The numerical values represented by images, i.e. the targets

array([0, 1, 2, ..., 8, 9, 8])

digits.target.size  ## total images

from sklearn import svm

svc = svm.SVC(gamma=0.001, C=100.)

plt.subplot(321)
plt.imshow(digits.images[1750], cmap=plt.cm.gray_r,
           interpolation='nearest')
plt.subplot(322)
plt.imshow(digits.images[1751], cmap=plt.cm.gray_r,
           interpolation='nearest')

svc.fit(digits.data[:1750], digits.target[:1750])  ## fitting on training set
SVC(C=100.0, gamma=0.001)

vc.predict(digits.data[1751:1796])  ## making prediction on validation set