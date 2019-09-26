#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Registration project

This script consists of a adapted functions of registration.py. This was needed to use the demo in registration_project.

Now every function returns only the similarity measure. 



The functions: 

    * rigid_corr_adapted

    * affine_corr_adapted

    * affine_mi_adapted

"""

import numpy as np

import registration as reg

import registration_util as util



#Adapted functions from registration



def rigid_corr_adapted(I, Im, x):

    #ADAPTED: In order to determine the gradient of the parameters, you only 

    #need the correlation value. So, the outputs: Th and Im_t are removed. 

    #Nothing else is changed.

    

    #Attention this function will be used for ngradient in the function:

    #intensity_based_registration_rigid_adapted. 

   



    # Computes normalized cross-correlation between a fixed and

    # a moving image transformed with a rigid transformation.

    # Input:

    # I - fixed image

    # Im - moving image

    # x - parameters of the rigid transform: the first element

    #     is the rotation angle and the remaining two elements

    #     are the translation

    # Output:

    # C - normalized cross-correlation between I and T(Im)

    # Im_t - transformed moving image T(Im)



    SCALING = 100



    # the first element is the rotation angle

    T = reg.rotate(x[0])



    # the remaining two element are the translation

    #

    # the gradient ascent/descent method work best when all parameters

    # of the function have approximately the same range of values

    # this is  not the case for the parameters of rigid registration

    # where the transformation matrix usually takes  much smaller

    # values compared to the translation vector this is why we pass a

    # scaled down version of the translation vector to this function

    # and then scale it up when computing the transformation matrix

    Th = util.t2h(T, x[1:]*SCALING)



    # transform the moving image

    Im_t, Xt = reg.image_transform(Im, Th)



    # compute the similarity between the fixed and transformed

    # moving image

    C = reg.correlation(I, Im_t)



    return C



def affine_corr_adapted(I, Im, x):

    #ADAPTED: In order to determine the gradient of the parameters, you only 

    #need the correlation value. So, the outputs: Th and Im_t are removed. 

    #Nothing else is changed.

    

    #Attention this function will be used for ngradient in the function:

    #intensity_based_registration_rigid_adapted. 

   

    

    # Computes normalized cross-corrleation between a fixed and

    # a moving image transformed with an affine transformation.

    # Input:

    # I - fixed image

    # Im - moving image

    # x - parameters of the rigid transform: the first element

    #     is the roation angle, the second and third are the

    #     scaling parameters, the fourth and fifth are the

    #     shearing parameters and the remaining two elements

    #     are the translation

    # Output:

    # C - normalized cross-corrleation between I and T(Im)

    # Im_t - transformed moving image T(Im)



    NUM_BINS = 64

    SCALING = 100

    

    #------------------------------------------------------------------#

    # TODO: Implement the missing functionality   

    

    T_rotate = reg.rotate(x[0]) #make rotation matrix (2x2 matrix)

    T_scaled = reg.scale(x[1],x[2]) #make scale matrix (2x2 matrix)

    T_shear = reg.shear(x[3],x[4]) # make shear matrix (2x2 matrix)

    t = np.array(x[5:])*SCALING #scale translation vector

    

    T_total = T_shear.dot((T_scaled).dot(T_rotate)) #multiply the matrices to get the transformation matrix (2x2)

    Th = util.t2h(T_total, t) #convert to homogeneous transformation matrix (3x3 matrix)

    

    Im_t, Xt = reg.image_transform(Im, Th) #apply transformation to moving image

    C = reg.correlation(Im, Im_t) #determine the correlation between the moving and transformed moving image

    #------------------------------------------------------------------#



    return C



def affine_mi_adapted(I, Im, x):

    # Computes mutual information between a fixed and

    # a moving image transformed with an affine transformation.

    # Input:

    # I - fixed image

    # Im - moving image

    # x - parameters of the rigid transform: the first element

    #     is the rotation angle, the second and third are the

    #     scaling parameters, the fourth and fifth are the

    #     shearing parameters and the remaining two elements

    #     are the translation

    # Output:

    # MI - mutual information between I and T(Im)

    # Im_t - transformed moving image T(Im)



    NUM_BINS = 64

    SCALING = 100

    

    #------------------------------------------------------------------#

    # TODO: Implement the missing functionality

    T_rotate = reg.rotate(x[0])

    T_scaled = reg.scale(x[1],x[2])

    T_shear = reg.shear(x[3],x[4])

    t = np.array(x[5:])*SCALING

    T_total = T_shear.dot((T_scaled).dot(T_rotate))

    Th = util.t2h(T_total, t)

    

    Im_t, Xt = reg.image_transform(Im, Th)

    p = reg.joint_histogram(Im, Im_t)

    MI = reg.mutual_information(p)

    #------------------------------------------------------------------#



    return MI

