import matplotlib
from matplotlib import pyplot as plt 
import numpy as np 
import tensorflow as tf 
import os 
import cv2 
import imghdr
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision , Recall, BinaryAccuracy 
from tensorflow.keras.models import load_model
#***LOADING IN DATA
#remove corrupted images 
# dataDirectory = 'data' #directory of the data

# imageExtensions = ['jpeg', 'jpg', 'png', 'bmp'] # valid image extensions 


# #Removes all sketchy images 
# for imageClass in os.listdir(dataDirectory): # for every folder in the 'data' folder (cats, dogs )
#     for image in os.listdir(os.path.join(dataDirectory, imageClass)): #for every image in that specific folder 
#         imagePath = os.path.join(dataDirectory, imageClass, image )
#         try: 
#             img = cv2.imread(imagePath) #opens an image with opencv
#             extension = imghdr.what(imagePath)
#             if extension not in imageExtensions: # if the end is not in our list of valid extensions 
#                 print('Image doest not have a valid extesion : {}'.format(imagePath))
#                 os.remove(imagePath) #removes invalid image 
#         except Exception as e : 
#             print("Issue with image {}".format(imagePath)) #cv2 can't open image so its probably corrupted 
#             #os.remove(imagePath)
        
    
    
# #load our data using the tensorflow dataset api 

# #It allows us to build data pipelines: This makes it wayyyy faster than just loading all the images onto memory
# #this builds an image dataset for us on the fly 
# #including labels, classes, and prepocessing :D
# data = tf.keras.utils.image_dataset_from_directory('data') #default - batch : 32 , size : 256, 256  !!!you can look at the documentation and completely customisze how you want the data 


# # #Great now that we have the data as a pipeline we can use to generate images on the fly for us
# # dataIterator = data.as_numpy_iterator() # allowing us to access the data 
# # batch = dataIterator.next() # batch is a multidimensional array ; batch[0] is the images for the batch and batch[1] are the labels for those images 


# #We don't really know if dogs is label 1 or label 0 and same with cats so we can make a quick subplot to figure this out 

# # figure, ax = plt.subplots(ncols=4 , figsize=(20,20))    
# # for idx, img in enumerate(batch[0][:4]):
# #     ax[idx].imshow(img.astype(int))
# #     ax[idx].title.set_text(batch[1][idx])

# # plt.show()

# #Great now we know cause of the code above that dogs are 1 and cats are 0 
# #Class 1 = DOGS
# #Class 0 = CATS


# #***PREPROCESSING***

# #Okay ; We finished loading our data now we have to preprocess
# #Preprocessing includes :
# #scaling/normalizing the data (we want the image numbers to be between 0 and 1 )
# #splitting up the data into training and testing sets to ensure that we don't over fit 


# #scale data 
# #divide the values in the batch by 255 (the highest number in the batch )
# #Use the data pipeline capability to do this more efficiently
# #There are a lot of functions that can be used to transform our data (look up in the tensorflow datasets api ) (zip, skip , and fromTensor are common ones )
# scaledData = data.map(lambda x,y: (x/255.0 , y)) # as we load in our data we divide all the number in the images by 255

# scaledIterator  = scaledData.as_numpy_iterator()

# #Just to test that the scaling worked 
# # batch = scaledIterator.next(); 

# # figure, ax = plt.subplots(ncols=4 , figsize=(20,20))    
# # for idx, img in enumerate(batch[0][:4]):
# #     ax[idx].imshow(img)
# #     ax[idx].title.set_text(batch[1][idx])

# # plt.show()


# # Now split into testing and training sets 
# #In out data we have 7 batches of 32 
# trainSize = int(len(scaledData)*.7) # 70% of the data ; data model is trained on 
# valSize = int (len(scaledData ) *.2) +1 # 20% of the data ; data that is used when testing acc of the model during training (adding one to test and val cause it rounds down) 
# testSize = int (len(scaledData)*.1)+1 #10% of the data  ; data used to test the model after training 

# #use the take and skip functions to take the data that you need for each partition (train, val , test)  

# trainData = scaledData.take(trainSize)
# valData = scaledData.skip(trainSize).take(valSize)
# testData = scaledData.skip(trainSize+valSize).take(testSize)







# #**** BUILD THE MODEL ****

# model = Sequential() # add layers sequentially 
# #first param is the number of filters , second is the size of each filter (16x16 pixels), third is the stride the filters take (1 pixel) , fourth is the activation,  
# #The conv layers extracts important information 
# # the last param is the dimensions of the image/input (256 pixels x 256 pixels x 3 channels (rgb))
# model.add(Conv2D(16, (3,3),1, activation = 'relu', input_shape=(256,256,3))) 
# #the pooling basically halves the data (halves the inputs )
# model.add(MaxPooling2D())   #gonna take the maximum value and use that to condense that information down 

# model.add(Conv2D(32, (3,3) , 1, activation= 'relu'))
# model.add(MaxPooling2D())

# model.add(Conv2D(16, (3,3), 1, activation='relu'))
# model.add(MaxPooling2D())

# #flattens the output so the dense layer can receive it 
# model.add(Flatten())    

# #16 x 16 =256 when its flattened 
# model.add(Dense(256,activation='relu'))
# model.add(Dense(1, activation='sigmoid')) #output layer : 1 is dogs 0 is cats 







# #compile the model 
# #optimzer (adam), loss type (binary since there are only two choics )
# model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics = ['accuracy'])

# logDirectory = 'logs'
# #logs our model at specific times 
# tensorBoardCallback = tf.keras.callbacks.TensorBoard(log_dir = logDirectory)


# #train model 
# hist= model.fit(trainData, epochs = 20 , validation_data = valData, callbacks = [tensorBoardCallback])



# #plot performance 
# lossFigure = plt.figure()
# plt.plot(hist.history['loss'], color='red', label = 'loss')
# plt.plot(hist.history['val_loss'], color='blue', label = 'valLoss')
# lossFigure.suptitle('Loss', fontsize=20 )
# plt.legend(loc="upper left")
# plt.show()  

# accFigure = plt.figure()
# plt.plot(hist.history['accuracy'], color = 'red', label = 'accuracy')
# plt.plot(hist.history['val_accuracy'], color = 'orange', label = 'val_accuracy')    
# accFigure.suptitle('Accuracy', fontsize =20 )
# plt.legend(loc='upper left')
# plt.show()

# #if validation loss is straying away from actual loss : overfitment or regularization is needed 
# #if loss isn't going down at all may need better data, a more sophisticated network, or both 



# #****EVALUATE PERFORMANCE***

# # test precision, recall 
# precision = Precision()
# recall = Recall()
# binAccuracy = BinaryAccuracy()

# # use the test dataset to check these metrics 
# #X is the images (the input) , y is the labels (desired output)
# #yhat is our model's predictions 
# for batch in testData.as_numpy_iterator(): 
#     X , y =batch ; 
#     yhat = model.predict(X)
#     precision.update_state(y, yhat)
#     recall.update_state(y, yhat)
#     binAccuracy.update_state(y, yhat)


# print("Precision: {}".format(precision.result().numpy()), "Recall: {}".format(recall.result().numpy()), "Accuracy: {}".format(binAccuracy.result().numpy()))



#Test it with new images 
testImage = cv2.cvtColor(cv2.imread('dogTestImage2.jpg'),cv2.COLOR_BGR2RGB) 
resizedImage =tf.image.resize(testImage,(256,256)) # resize image to the size of our input 
plt.imshow(resizedImage.numpy().astype(int))
plt.show()

model = load_model(os.path.join('models', 'dogVsCatmodelV1.h5'))
#since our model only takes batches of images as inputs we have to make a batch for this one image to properly get a result 
testPrediction = model.predict(np.expand_dims(resizedImage/255, 0 )) #divides the values of the pixels by 255 to normalize the image first 
confidence = abs(testPrediction-.5)*2
print("TestPrediction Value : {}".format(str(round(testPrediction.max()*100, 2))))
if(testPrediction > .5 ): 
    print("I am {}% confident that this image is a dog ".format(str(round(confidence.max()*100))))
else: 
    print("I am {}% confident that this image is a cat ".format(str(round(confidence.max()*100))))
    
    
    
    

#***** SAVE THE MODEL ******
#model.save(os.path.join('models', 'dogVsCatmodelV1.h5'))#serialize the model (save it to the disk )
    











        
        




