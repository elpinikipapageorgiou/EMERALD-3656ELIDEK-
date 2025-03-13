# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:50:16 2021

@author: John
"""

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import pandas as pd
# tf.config.list_physical_devices('GPU')


sys.path.insert(1, 'C:\\Users\\User\\DSS EXPERIMENTS\\EME_SPN_Factory_CT_LIDC2PET - Official\\')

import spn_data_loader
import spn_main_functions
import spn_metrics
import spn_model_evaluation_plots

import spn_clinical_functions
import spn_clinical_models
import spn_ml_model_evaluation_plots


# import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# print(os.getenv('‘TF_GPU_ALLOCATOR’'))


#%% PARAMETER ASSIGNEMENT




OPTIONS = {'MODEL'        : 'lvgg', #lvgg vgg19-final-spn xception simple_cnn
           'EPOCHS'       : 200,
           'BATCH_SIZE'   : 64,
           'IN_SHAPE'     : (80,80,3),
           'TUNE'         : 'auto', # aut
           'CLASSES'      : 2,
           'N_SPLIT'      : 10,
           'AUGMENTATION' : True,
           'VERBOSE'      : True,
           'CLASS_NAMES'  : ['Benign','Malignant'],
           'EARLY_STOP'   : True,
           'PLOTS'        : True,
            }


# att_vgg19  lvgg    inception    vgg19-base    ffvgg19    att_ffvgg19  efficient  vgg19-final-spn xception
# ioapi_vit ioapi_swimtr ioapi_perceiver ioapi_involutional ioapi_convmixer simple_cnn
#  ioapi_big_transfer ioapi_eanet ioapi_fnet ioapi_gmlp ioapi_mlpmixer

a = ['ioapi_vit','ioapi_perceiver','ioapi_fnet','ioapi_gmlp','ioapi_mlpmixer', 
             'ioapi_involutional','ioapi_convmixer','ioapi_big_transfer']

#%%


'''

LOAD IMAGE DATA


'''

path_lidc = 'C:\\Users\\User\\EMERALD DATA\\Processed Datasets\\LIDC_NEW\\'
data_lidc, labels_lidc = spn_data_loader.load_lidc(path_lidc, in_shape=OPTIONS['IN_SHAPE'], verbose=True)


path_petct = 'C:\\Users\\User\\EMERALD DATA\\Processed Datasets\\NSCLC_SMALLSAMPLE_FOR_PROCESS\\2D_SEP23 (the main)\\'
data_ct_doctor,data_ct_follow,data_ct_biopsy,data_pet_doctor,data_pet_follow,data_pet_biopsy,labels_doctor,labels_follow,labels_biopsy,INFO_DOCTOR,INFO_FOLLOW,INFO_BIOPSY = spn_data_loader.load_spn(path_petct, in_shape=OPTIONS['IN_SHAPE'], verbose=False)

image = data_ct_doctor[0]
from matplotlib import pyplot as plt
plt.imshow(image, interpolation='nearest')
plt.show()

image = data_lidc[0]
from matplotlib import pyplot as plt
plt.imshow(image, interpolation='nearest')
plt.show()


#%%

#################################################################################
#################################################################################
#                       10-FOLD CROSS VALIDATIONS
#################################################################################
#################################################################################



'''

10-fold cross validation on LIDC


'''

data = data_lidc
labels = labels_lidc
INFO_TRAIN = pd.concat([INFO_DOCTOR,INFO_FOLLOW],axis=0)

history_ct, model_ct, PREDICTIONS_TOTAL_ct, PREDICTIONS_BINARY_TOTAL_ct, LABELS_TOTAL, fold_metrics_ct, metrics_dict_ct = spn_main_functions.train_kfold(data,labels,OPTIONS)

spn_clinical_functions.print_metrics (metrics_dict_ct)


#%%

#################################################################################
#################################################################################
#                       MODEL FIT
#################################################################################
#################################################################################

'''

FIT on ct

'''

data = data_lidc
data = np.concatenate((data_ct_biopsy, data_lidc,data_ct_doctor), axis=0)

labels = labels_lidc
labels = np.concatenate((labels_biopsy, labels_lidc,labels_doctor), axis=0)
INFO_TRAIN = pd.concat([INFO_DOCTOR,INFO_FOLLOW],axis=0)

model_fit_ct, predictions_fit_ct, predictions_binary_fit_ct = spn_main_functions.train_fit(data, labels, OPTIONS)

#from tensorflow.keras.models import save_model, load_model

#model_fit_ct.save('ct_fit_82perc_biopsy.h5')

'''

TEST ON CT OVER EXTERNAL DATA

'''

test_ct =  data_ct_follow
#test_ct =  np.concatenate((data_ct_follow, data_ct_biopsy, data_ct_doctor), axis=0)

test_labels = labels_follow
#test_labels = np.concatenate((labels_follow, labels_biopsy, labels_doctor), axis=0)

test_predictions_ct = model_fit_ct.predict(test_ct)

test_predictions_ct_binary = np.argmax(test_predictions_ct, axis=-1)


ct_biopsy_test_metrics = spn_clinical_functions.metrics(test_predictions_ct_binary,test_labels[:,1],test_predictions_ct[:,1])
spn_clinical_functions.print_metrics (ct_biopsy_test_metrics)


# PLOT THE ROC
import numpy as np
all_predictions_proba = np.array(test_predictions_ct)
spn_model_evaluation_plots.plot_roc_scikit(test_predictions_ct,test_labels)




#%%

#################################################################################
#################################################################################
#                       EVALUATION ON EXTERNAL
#################################################################################
#################################################################################


'''

PREDICT THE EXTERNAL DATA

'''
from tensorflow.keras.models import save_model, load_model
# We need a function which received the data and the models. It firstly gets the
# predicitons on pet, ct and then integrates them to the data so that
# the ML classifier gives its overall prediction
# it returns every prediction


# here, do the metrics and plots

test_ct =  np.concatenate((data_ct_follow, data_ct_biopsy), axis=0)
test_pet = np.concatenate((data_pet_follow, data_pet_biopsy), axis=0)
test_labels = np.concatenate((labels_follow, labels_biopsy), axis=0)
test_clinical = pd.concat([INFO_FOLLOW,INFO_BIOPSY],axis=0)

model_fit_ct = load_model('ct_fit_82perc_biopsy.h5')
model_fit_pet = load_model('pet_fit_82perc_biopsy.h5')

test_predictions_ct,test_predictions_pet,ml_data_test,labels_test,test_prediction_overall,test_prediction_overall_proba = spn_clinical_functions.predict_external(test_ct,test_pet,test_labels,test_clinical,model_fit_ct,model_fit_pet,trained_classifier,selected_features)

test_predictions_ct_binary = np.argmax(test_predictions_ct, axis=-1)
test_predictions_pet_binary = np.argmax(test_predictions_pet, axis=-1)

ct_biopsy_test_metrics = spn_clinical_functions.metrics(test_predictions_ct_binary,labels_test[:,1],test_predictions_ct[:,1])
spn_clinical_functions.print_metrics (ct_biopsy_test_metrics)

pet_biopsy_test_metrics = spn_clinical_functions.metrics(test_predictions_pet_binary,labels_test[:,1],test_predictions_pet[:,1])
spn_clinical_functions.print_metrics (pet_biopsy_test_metrics)


ultimate_metrics_biopsy = spn_clinical_functions.metrics(test_prediction_overall,labels_test[:,1],test_prediction_overall_proba[:,1])
spn_clinical_functions.print_metrics (ultimate_metrics_biopsy)



# PLOT THE ROC
import numpy as np
all_predictions_proba = np.array(test_prediction_overall_proba)
spn_model_evaluation_plots.plot_roc_scikit(test_prediction_overall_proba,labels_test)


# PLOT KS
spn_model_evaluation_plots.plot_ks_statistic(labels_test[:,1],test_prediction_overall_proba)


# COHENS
kappa_score, observed_agreement, expected_agreement = spn_model_evaluation_plots.calculate_cohens_kappa_matrices(labels_test[:,1], test_prediction_overall)





#%%


# '''

# GRAD-CAM PLUS PLUS


# '''

# import mr_gradcamplusplus

items_no = [i for i in range (len(test_ct[30:100]))]
#items_no = [17,18, 100, 500, 600, 601, 602, 603, 604, 152]
base_path = 'C:\\Users\\apost\\Desktop\\EME_SPN_Factory (multi-modal) DEC23 OFFICIAL\\'

test_predictions_ct_bin = np.argmax(test_predictions_ct, axis=-1)


import spn_gradcamplusplus
# # GradCAM++
# spn_gradcamplusplus.gradcamplusplus (items_no=items_no,predictions_all=test_predictions_ct_bin,labels=test_labels,data=test_ct,model3=model_fit_ct,verbose = False,show=False, save = True, base_path=base_path)


# # Score CAM
spn_gradcamplusplus.scorecam (items_no=items_no,predictions_all=test_predictions_ct_bin,labels=test_labels,data=test_ct,model3=model_fit_ct,verbose = False,show=False, save = True, base_path=base_path)

# # GradCAM
# spn_gradcamplusplus.gradcam (items_no=items_no,predictions_all=test_predictions_ct_bin,labels=test_labels,data=test_ct,model3=model_fit_ct,verbose = False,show=False, save = True, base_path=base_path)


# # Saliency
# spn_gradcamplusplus.saliency (items_no=items_no,predictions_all=test_predictions_ct_bin,labels=test_labels,data=test_ct,model3=model_fit_ct,verbose = False,show=False, save = True, base_path=base_path)

# # Smooth Grad
# spn_gradcamplusplus.smoothgrad (items_no=items_no,predictions_all=test_predictions_ct_bin,labels=test_labels,data=test_ct,model3=model_fit_ct,verbose = False,show=False, save = True, base_path=base_path)



#%%

'''
LIME
'''

# import mr_lime_func

# # LIME COMMANDS
# #items_no = [17,18, 100, 500, 600, 601, 602, 603, 604, 152]
# items_no = [i for i in range (len(data[:30]))]
# base_path = 'C:\\Users\\User\\DSS EXPERIMENTS\\MRI Classification - Explainability\\XAI\\GLIOMA\\'

# mr_lime_func.the_lime (items_no,predictions_all,labels,data,1,model3,verbose = False,show=False, save = True, base_path=base_path)









