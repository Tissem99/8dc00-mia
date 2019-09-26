"""
Registration project code.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_adapted_functions as reg_adapt
from IPython.display import display, clear_output


import registration_util as util
from cpselect.cpselect import cpselect


import math

def point_based_registration(I_path,Im_path, X, Xm):

    #read/load the images I and Im

    imageI = plt.imread(I_path);
    imageIm = plt.imread(Im_path);    

    #convert to homogenous coordinates using c2h

    X_h = util.c2h(X)
    Xm_h= util.c2h(Xm)    

    #compute affine transformation and make a homogenous transformation matrix
    Th = reg.ls_affine(X_h,Xm_h)
    
    #transfrom the moving image using the transformation matrix
    It, Xt = reg.image_transform(imageIm, Th)

    #plotting the results
    fig = plt.figure(figsize = (20,30))
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(imageI) #plot first image (fixed or T1) image
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(imageIm) #plot second image (moving or T2) image
    ax3 = fig.add_subplot(133)
    im3 = ax3.imshow(It) #plot (T1 moving or T2) transformed image

    ax1.set_title('T1 (fixed)')
    ax2.set_title('T1 moving or T2')
    ax3.set_title('Transformed (T1 moving or T2) image')

    return  Th

def Evaluate_point_based_registration(Th, X_target, Xm_target):

    X_target = util.c2h(X_target)
    Xmh_target = util.c2h(Xm_target)

    #transform the selected points
    Xm_target_transformed = Th.dot(Xmh_target)

    #average distance between points (= target registration error)
    Error_avgdist = calculateAvg_Distance(X_target, Xm_target_transformed)

    return Error_avgdist



def calculateAvg_Distance(points, points_t):  

    totaldistance = 0
    numberOfPoints = points.shape[1]

    for i in range(numberOfPoints):
        dist = math.sqrt((points_t[0,i] - points[0,i])**2 + (points_t[1,i] - points[1,i])**2)  
        totaldistance = totaldistance+dist

    average_distance = totaldistance/numberOfPoints        

    return average_distance


def intensity_based_registration_rigid_Corr_adapted(I, Im):

    #ADAPTED:

    #Added 1)'fun2' with the original reg.rigid_corr(I, Im, x), because

    #three outputs (C, Im_t, Th) are needed for the visualization. So you 

    #use the adapted function. 'fun' is the adapted rigid_corr. 

    

    #2) Changed "x += g*mu" to "x =np.add(x, g*mu)", because of shape/dimension error

    #3) Flattened the x-array when used as input for fun2, because of shape/dimension error

    

    # read the fixed and moving images

    # change these in order to read different images



    # initial values for the parameters

    # we start with the identity transformation

    # most likely you will not have to change these

    x = np.array([0., 0., 0.])



    # NOTE: for affine registration you have to initialize

    # more parameters and the scaling parameters should be

    # initialized to 1 instead of 0



    # the similarity function

    # this line of code in essence creates a version of rigid_corr()

    # in which the first two input parameters (fixed and moving image)

    # are fixed and the only remaining parameter is the vector x with the

    # parameters of the transformation

    fun = lambda x: reg_adapt.rigid_corr_adapted(I, Im, x)

    fun2 = lambda x: reg.rigid_corr(I, Im, x)

    # the learning rate

    mu = 0.001



    # number of iterations

    num_iter = 200



    iterations = np.arange(1, num_iter+1)

    similarity = np.full((num_iter, 1), np.nan)



    fig = plt.figure(figsize=(14,6))



    # fixed and moving image, and parameters

    ax1 = fig.add_subplot(121)



    # fixed image

    im1 = ax1.imshow(I)

    # moving image

    im2 = ax1.imshow(I, alpha=0.7)

    # parameters

    txt = ax1.text(0.3, 0.95,

        np.array2string(x, precision=5, floatmode='fixed'),

        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},

        transform=ax1.transAxes)



    # 'learning' curve

    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))



    learning_curve, = ax2.plot(iterations, similarity, lw=2)

    ax2.set_xlabel('Iteration')

    ax2.set_ylabel('Similarity')

    ax2.grid()



    # perform 'num_iter' gradient ascent updates

    for k in np.arange(num_iter):



        # gradient ascent

        g = reg.ngradient(fun, x)

        

        x =np.add(x, g*mu)



        # for visualization of the result

        S, Im_t, _ = fun2(x.flatten())



        clear_output(wait = True)



        # update moving image and parameters

        im2.set_data(Im_t)

        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))



        # update 'learning' curve

        similarity[k] = S

        learning_curve.set_ydata(similarity)



        display(fig)



def intensity_based_registration_affine__Corr_adapted():

    #ADAPTED:

    #Added 1)'fun2' with the original reg.affine_corr(I, Im, x), because

    #three outputs (C, Im_t, Th) are needed for the visualization. So you 

    #use the adapted function. 'fun' is the adapted rigid_corr. 

    
    

    #2) Changed "x += g*mu" to "x =np.add(x, g*mu)", because of shape/dimension error

    #3) Flattened the x-array when used as input for fun2, because of shape/dimension error

    

    # read the fixed and moving images

    # change these in order to read different images
    I = plt.imread('../data/image_data/1_1_t1.tif')
    Im = plt.imread('../data/image_data/1_1_t1_d.tif')


    # initial values for the parameters

    # we start with the identity transformation

    # most likely you will not have to change these

    x = np.array([0., 1., 1., 0., 0., 0., 0.])

    

    # NOTE: for affine registration you have to initialize

    # more parameters and the scaling parameters should be

    # initialized to 1 instead of 0



    # the similarity function

    # this line of code in essence creates a version of rigid_corr()

    # in which the first two input parameters (fixed and moving image)

    # are fixed and the only remaining parameter is the vector x with the

    # parameters of the transformation

    fun = lambda x: reg_adapt.affine_corr_adapted(I, Im, x)

    fun2 = lambda x: reg.affine_corr(I, Im, x)

    # the learning rate

    mu = 0.001



    # number of iterations

    num_iter = 200



    iterations = np.arange(1, num_iter+1)

    similarity = np.full((num_iter, 1), np.nan)



    fig = plt.figure(figsize=(14,6))



    # fixed and moving image, and parameters

    ax1 = fig.add_subplot(121)



    # fixed image

    im1 = ax1.imshow(I)

    # moving image

    im2 = ax1.imshow(I, alpha=0.7)

    # parameters

    txt = ax1.text(0.3, 0.95,

        np.array2string(x, precision=5, floatmode='fixed'),

        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},

        transform=ax1.transAxes)



    # 'learning' curve

    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))



    learning_curve, = ax2.plot(iterations, similarity, lw=2)

    ax2.set_xlabel('Iteration')

    ax2.set_ylabel('Similarity')

    ax2.grid()



    # perform 'num_iter' gradient ascent updates

    for k in np.arange(num_iter):

        # gradient ascent

        g = reg.ngradient(fun, x)

        x =np.add(x, g*mu)

        # for visualization of the result

        S, Im_t, _ = fun2(x.flatten())
        clear_output(wait = True)



        # update moving image and parameters

        im2.set_data(Im_t)

        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))



        # update 'learning' curve

        similarity[k] = S

        learning_curve.set_ydata(similarity)



        display(fig)



def intensity_based_registration_affine_MI_adapted(I, Im):

    #ADAPTED:

    #Added 1)'fun2' with the original reg.affine_corr(I, Im, x), because

    #three outputs (C, Im_t, Th) are needed for the visualization. So you 

    #use the adapted function. 'fun' is the adapted rigid_corr. 

    

    #2) Changed "x += g*mu" to "x =np.add(x, g*mu)", because of shape/dimension error

    #3) Flattened the x-array when used as input for fun2, because of shape/dimension error

    

    # read the fixed and moving images

    # change these in order to read different images



    # initial values for the parameters

    # we start with the identity transformation

    # most likely you will not have to change these

    x = np.array([0., 1., 1., 0., 0., 0., 0.])

    

    # NOTE: for affine registration you have to initialize

    # more parameters and the scaling parameters should be

    # initialized to 1 instead of 0



    # the similarity function

    # this line of code in essence creates a version of rigid_corr()

    # in which the first two input parameters (fixed and moving image)

    # are fixed and the only remaining parameter is the vector x with the

    # parameters of the transformation

    fun = lambda x: reg_adapt.affine_corr_adapted(I, Im, x)

    fun2 = lambda x: reg.affine_corr(I, Im, x)

    # the learning rate

    mu = 0.001



    # number of iterations

    num_iter = 200



    iterations = np.arange(1, num_iter+1)

    similarity = np.full((num_iter, 1), np.nan)



    fig = plt.figure(figsize=(14,6))



    # fixed and moving image, and parameters

    ax1 = fig.add_subplot(121)



    # fixed image

    im1 = ax1.imshow(I)

    # moving image

    im2 = ax1.imshow(I, alpha=0.7)

    # parameters

    txt = ax1.text(0.3, 0.95,

        np.array2string(x, precision=5, floatmode='fixed'),

        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},

        transform=ax1.transAxes)



    # 'learning' curve

    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))



    learning_curve, = ax2.plot(iterations, similarity, lw=2)

    ax2.set_xlabel('Iteration')

    ax2.set_ylabel('Similarity')

    ax2.grid()



    # perform 'num_iter' gradient ascent updates

    for k in np.arange(num_iter):



        # gradient ascent

        g = reg.ngradient(fun, x)

        

        x =np.add(x, g*mu)



        # for visualization of the result

        S, Im_t, _ = fun2(x.flatten())



        clear_output(wait = True)



        # update moving image and parameters

        im2.set_data(Im_t)

        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))



        # update 'learning' curve

        similarity[k] = S

        learning_curve.set_ydata(similarity)



        display(fig)

def intensity_based_registration_demo():

    # read the fixed and moving images
    # change these in order to read different images
    I = plt.imread('../data/image_data/1_1_t1.tif')
    Im = plt.imread('../data/image_data/1_1_t1_d.tif')

    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
    x = np.array([0., 0., 0.])

    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation
    fun = lambda x: reg.rigid_corr(I, Im, x)

    # the learning rate
    mu = 0.001

    # number of iterations
    num_iter = 200

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    fig = plt.figure(figsize=(14,6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt = ax1.text(0.3, 0.95,
        np.array2string(x, precision=5, floatmode='fixed'),
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):

        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g*mu

        # for visualization of the result
        S, Im_t, _ = fun(x)

        clear_output(wait = True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)

        display(fig)
