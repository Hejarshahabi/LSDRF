#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[16]:


#import the following libraries
import numpy as np
import os
import gdal
import pyrsgis as gis
from pyrsgis import raster
gdal.UseExceptions()
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

###############################################################################################
os.getcwd()
#changing the current working directory to the one that includes your data 
os.chdir("F:\jp2")
#labeled data (grounf truth) 
inventory= "label.tif"
#satellite iamges (e.g Senitnel-2A)
img = "train.tif"
#for landslide detection slope is an important factor, so it is used in this case
slope = "slope.tif"
#the satellite images and slope layer which we want to apply our trained model on
img_prediction = "prediction.tif"
slope_prediction  = "slope-prediction.tif"
#################################################################################################
#pre processing the data as the input for classification
def preprocessing():
 #reading data, ds includes satellite iamges data structure that is necessary for exporting the ruslt in TIF format 
    ds, inv = raster.read (inventory)
    ds, slp = raster.read (slope)
    ds, s2A = raster.read (img)
#reshaping data
    inv = np.reshape (inv, (inv.shape [0] *inv.shape [1], 1)). astype (int)
    slp = (np.reshape (slp, inv.shape)). astype (int)
#reshape n dimensional satellite data to 2d array. each column represents an image band or feature
    for i in range (s2A.shape [0]):
        s2 = np.zeros ((s2A.shape [1]* s2A.shape [2], s2A.shape [0]))
        s2 [:, i] = (np.reshape (s2A [i,:,:], (s2A.shape [1] * s2A.shape [2]))). astype (int)
        stack = np.concatenate ((s2, slp), axis = 1)
    print (f'the shape of labeled data: {inv.shape} ')
    #print (f'the shape of image features: {s2A.shape} ')
    #print (f'the shape of slope layer: {slp.shape} ')
    print (f'image features and slope layer are stacked together with the shape of: {stack.shape}')
    return (stack, inv)

####################################################################################################
#preparing satellite images that we want to classify

def forclassification():
    ds, slp = raster.read("slope-prediction.tif")
    ds, img = raster.read("prediction.tif")
    slp = np.reshape(slp, (slp.shape[0]*slp.shape[1], 1)).astype(int)
    for i in range(img.shape[0]):
        img1= np.zeros((slp.shape[0], img.shape[0]), dtype=int)
        img1[:,i]= (np.reshape(img[i,:,:], (slp.shape[0]))).astype(int)
        
    stack= np.concatenate((img1,slp), axis=1)
    return(stack)

#################################################################################################
#model training

def training():
 #spliting train and test data 
    x,y= preprocessing()
    print(x.shape,y.shape)
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size= 0.3, random_state=50)
    print(xtrain.shape, xtest.shape)
 
 #implementing random forest model
    forest = RandomForestClassifier(n_estimators= 400, max_leaf_nodes=20)
    forest.fit(xtrain, ytrain)
    fpred= forest.predict(xtest)
 
 #model performance evaluation
    print("random forest: ")
    print(classification_report(ytest, fpred))

    return (forest)
    
training()

##################################################################################################
#classifiying the image 

ds, img2 = raster.read("prediction.tif")
def prediction():
 #recalling data
    data = forclassification()
 #recalling trained model
    model = training()
    a= model.predict(data)
    print(a.shape)
 #reshaping the prediction or classification result to original image size
    clmap = np.reshape(a,(img2.shape[1], img2.shape[2]))
 #converting array into TIF map
    output = "landslide_detection_map.tif"
    raster.export(clmap, ds, filename=output, dtype="int")
    return(clmap, clmap.shape)
prediction()



# #pre processing data as input for classification
# preprocessing def ():
#  #reading data
#     ds, inv = raster.read (inventory)
#     ds, slp = raster.read (slope)
#     ds, s2A = raster.read (img)
# # reshape the data
#     inv = np.reshape (inv, (inv.shape [0] *inv.shape [1], 1)). astype (int)
#     slp = (np.reshape (slp, inv.shape)). astype (int)
# # reshape the satellite images in n dimensions in a 2D array, each column represents an iamge band
#     for i in the range (s2A.shape [0]):
#         s2 = np.zeros ((s2A.shape [1]* s2A.shape [2], s2A.shape [0]))
#         s2 [:, i] = (np.reshape (s2A [i,:,:], (s2A.shape [1] * s2A.shape [2]))). astype (int)
#         stack = np.concatenate ((s2, slp), axis = 1)
#     print (f 'the form of the labeled data: {inv.shape}')
#     print (f 'the shape of the image characteristics: {s2A.shape}')
#     print (f 'the shape of the slope layer: {slp.shape}')
#     print (f 'image features and slope layer are stacked with the form of: {stack.shape}')
#     return (stack, inv)

# #pre processing data as input for classification
# preprocessing def ():
#  #reading data
#     ds, inv = raster.read (inventory)
#     ds, slp = raster.read (slope)
#     ds, s2A = raster.read (img)
# # reshape the data
#     inv = np.reshape (inv, (inv.shape [0] *inv.shape [1], 1)). astype (int)
#     slp = (np.reshape (slp, inv.shape)). astype (int)
# # reshape the satellite images in n dimensions in a 2D array, each column represents an iamge band
#     for i in the range (s2A.shape [0]):
#         s2 = np.zeros ((s2A.shape [1]* s2A.shape [2], s2A.shape [0]))
#         s2 [:, i] = (np.reshape (s2A [i,:,:], (s2A.shape [1] * s2A.shape [2]))). astype (int)
#         stack = np.concatenate ((s2, slp), axis = 1)
#     print (f 'the form of the labeled data: {inv.shape}')
#     print (f 'the shape of the image characteristics: {s2A.shape}')
#     print (f 'the shape of the slope layer: {slp.shape}')
#     print (f 'image features and slope layer are stacked with the form of: {stack.shape}')
#     return (stack, inv)

# #pre processing data as input for classification
# preprocessing def ():
#  #reading data
#     ds, inv = raster.read (inventory)
#     ds, slp = raster.read (slope)
#     ds, s2A = raster.read (img)
# # reshape the data
#     inv = np.reshape (inv, (inv.shape [0] *inv.shape [1], 1)). astype (int)
#     slp = (np.reshape (slp, inv.shape)). astype (int)
# # reshape the satellite images in n dimensions in a 2D array, each column represents an iamge band
#     for i in the range (s2A.shape [0]):
#         s2 = np.zeros ((s2A.shape [1]* s2A.shape [2], s2A.shape [0]))
#         s2 [:, i] = (np.reshape (s2A [i,:,:], (s2A.shape [1] * s2A.shape [2]))). astype (int)
#         stack = np.concatenate ((s2, slp), axis = 1)
#     print (f 'the form of the labeled data: {inv.shape}')
#     print (f 'the shape of the image characteristics: {s2A.shape}')
#     print (f 'the shape of the slope layer: {slp.shape}')
#     print (f 'image features and slope layer are stacked with the form of: {stack.shape}')
#     return (stack, inv)

# #pre traitement des données comme entrée pour la classification
# prétraitement def ():
#  #reading data
#     ds, inv = raster.read (inventaire)
#     ds, slp = raster.read (pente)
#     ds, s2A = raster.read (img)
# # remodeler les données
#     inv = np.reshape (inv, (inv.shape [0] *inv.shape [1], 1)). astype (entier)
#     slp = (np.reshape (slp, inv.shape)). astype (entier)
# # refaçonner les images satellite en n dimensions dans un tableau 2D, chaque colonne représente une bande iamge
#     pour i dans la plage (s2A.shape [0]):
#         s2 = np.zeros ((s2A.shape [1]* s2A.shape [2], s2A.shape [0]))
#         s2 [:, i] = (np.reshape (s2A [i,:,:], (s2A.shape [1] * s2A.shape [2]))). astype (entier)
#         stack = np.concatenate ((s2, slp), axis = 1)
#     print (f 'la forme des données étiquetées: {inv.shape}')
#     print (f 'la forme des caractéristiques de l'image: {s2A.shape}')
#     print (f 'la forme de la couche de pente: {slp.shape}')
#     print (f 'les caractéristiques d'image et la couche de pente sont empilées avec la forme de: {stack.shape}')
#     retour (pile, inv)

# #pre traitement des données comme entrée pour la classification
# prétraitement def ():
#  #reading data
#     ds, inv = raster.read (inventaire)
#     ds, slp = raster.read (pente)
#     ds, s2A = raster.read (img)
# # remodeler les données
#     inv = np.reshape (inv, (inv.shape [0] *inv.shape [1], 1)). astype (entier)
#     slp = (np.reshape (slp, inv.shape)). astype (entier)
# # refaçonner les images satellite en n dimensions dans un tableau 2D, chaque colonne représente une bande iamge
#     pour i dans la plage (s2A.shape [0]):
#         s2 = np.zeros ((s2A.shape [1]* s2A.shape [2], s2A.shape [0]))
#         s2 [:, i] = (np.reshape (s2A [i,:,:], (s2A.shape [1] * s2A.shape [2]))). astype (entier)
#         stack = np.concatenate ((s2, slp), axis = 1)
#     print (f 'la forme des données étiquetées: {inv.shape}')
#     print (f 'la forme des caractéristiques de l'image: {s2A.shape}')
#     print (f 'la forme de la couche de pente: {slp.shape}')
#     print (f 'les caractéristiques d'image et la couche de pente sont empilées avec la forme de: {stack.shape}')
#     retour (pile, inv)

# #pre traitement des données comme entrée pour la classification
# prétraitement def ():
#  #reading data
#     ds, inv = raster.read (inventaire)
#     ds, slp = raster.read (pente)
#     ds, s2A = raster.read (img)
# # remodeler les données
#     inv = np.reshape (inv, (inv.shape [0] *inv.shape [1], 1)). astype (entier)
#     slp = (np.reshape (slp, inv.shape)). astype (entier)
# # refaçonner les images satellite en n dimensions dans un tableau 2D, chaque colonne représente une bande iamge
#     pour i dans la plage (s2A.shape [0]):
#         s2 = np.zeros ((s2A.shape [1]* s2A.shape [2], s2A.shape [0]))
#         s2 [:, i] = (np.reshape (s2A [i,:,:], (s2A.shape [1] * s2A.shape [2]))). astype (entier)
#         stack = np.concatenate ((s2, slp), axis = 1)
#     print (f 'la forme des données étiquetées: {inv.shape}')
#     print (f 'la forme des caractéristiques de l'image: {s2A.shape}')
#     print (f 'la forme de la couche de pente: {slp.shape}')
#     print (f 'les caractéristiques d'image et la couche de pente sont empilées avec la forme de: {stack.shape}')
#     retour (pile, inv)

# #pre traitement des données comme entrée pour la classification
# prétraitement def ():
#  #reading data
#     ds, inv = raster.read (inventaire)
#     ds, slp = raster.read (pente)
#     ds, s2A = raster.read (img)
# # remodeler les données
#     inv = np.reshape (inv, (inv.shape [0] *inv.shape [1], 1)). astype (entier)
#     slp = (np.reshape (slp, inv.shape)). astype (entier)
# # refaçonner les images satellite en n dimensions dans un tableau 2D, chaque colonne représente une bande iamge
#     pour i dans la plage (s2A.shape [0]):
#         s2 = np.zeros ((s2A.shape [1]* s2A.shape [2], s2A.shape [0]))
#         s2 [:, i] = (np.reshape (s2A [i,:,:], (s2A.shape [1] * s2A.shape [2]))). astype (entier)
#         stack = np.concatenate ((s2, slp), axis = 1)
#     print (f 'la forme des données étiquetées: {inv.shape}')
#     print (f 'la forme des caractéristiques de l'image: {s2A.shape}')
#     print (f 'la forme de la couche de pente: {slp.shape}')
#     print (f 'les caractéristiques d'image et la couche de pente sont empilées avec la forme de: {stack.shape}')
#     retour (pile, inv)
