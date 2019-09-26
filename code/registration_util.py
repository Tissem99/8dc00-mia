"""
Utility functions for registration.
"""

import numpy as np
from cpselect.cpselect import cpselect


def test_object(centered=True):
    # Generate an F-like test object.
    # Input:
    # centered - set the object centroid to the origin
    # Output:
    # X - coordinates of the test object

    X = np.array([[4, 4, 4.5, 4.5, 6, 6, 4.5, 4.5, 7, 7, 4], [10, 4, 4, 7, 7, 7.5, 7.5, 9.5, 9.5, 10, 10]])

    if centered:
        X[0, :] = X[0, :] - np.mean(X[0, :])
        X[1, :] = X[1, :] - np.mean(X[1, :])

    return X


def c2h(X):
    # Convert cartesian to homogeneous coordinates.
    # Input:
    # X - cartesian coordinates
    # Output:
    # Xh - homogeneous coordinates

    n = np.ones([1,X.shape[1]])
    Xh = np.concatenate((X,n))

    return Xh


def t2h(T, t):
    # Convert a 2D transformation matrix to homogeneous form.
    # Input:
    # T - 2D transformation matrix
    # Xt - 2D translation vector
    # Output:
    # Th - homogeneous transformation matrix

    #------------------------------------------------------------------#
    # TODO: Implement conversion of a transformation matrix and a translation vector to homogeneous transformation matrix.
    # remove the 'pass' once implemented
    t = np.asarray(t)
    Tx = t.reshape(-1,1)
    Substep = np.concatenate((T, Tx), axis = 1)
    Th = np.concatenate((Substep, np.array([[0,0,1]])))
    
    return Th
    #------------------------------------------------------------------#


def plot_object(ax, X):
    # Plot 2D object.

    # Input:
    # X - coordinates of the shape

    ax.plot(X[0,:], X[1,:], linewidth=2)


def my_cpselect(I_path, Im_path):
    # Wrapper around cpselect that returns the point coordinates
    # in the expected format (coordinates in rows).
    # Input:
    # I - fixed image
    # Im - moving image
    # Output:
    # X - control points in the fixed image
    # Xm - control points in the moving image

    #------------------------------------------------------------------#
    #path_1 = '../data/image_data/1_1_t1.tif'
    #path_2 = '../data/image_data/1_1_t2.tif'
    #controlpointlist = cpselect(path_1, path_2)
    
    
    controlpointlist = cpselect(I_path, Im_path)

    #number of points

    numberOfPoints = len(controlpointlist)
    X = np.zeros(shape= (2,numberOfPoints))
    Xm = np.zeros(shape= (2,numberOfPoints))

    for i in range(numberOfPoints):
        valuesOfPoint = [*controlpointlist[i].values()]
        X[0][i] = valuesOfPoint[1]
        X[1][i] = valuesOfPoint[2]
        Xm[0][i] = valuesOfPoint[3]
        Xm[1][i] = valuesOfPoint[4]
    #------------------------------------------------------------------#

    return X, Xm
