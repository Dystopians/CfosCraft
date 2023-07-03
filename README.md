# Advanced Image Segmentation Techniques for Neural Activity Detection via C-fos Immediate Early Gene Expression
## 0. Preface 
This project is a subsidiary code of the Advanced Image Segmentation Techniques for Neural Activity Detection via C-fos Immediate Early Gene Expression project. Due to the extremely low coupling of the modules, I have split the functions into different files and integrated them into easy-to-use ipynb files, mainly containing the following six:

1.Mask_Image_Ploting.ipynb is responsible for processing the manually annotated Labelme Json files and generating high-quality mask images in the corresponding directory.

2.Autoencoder_and_clustering.ipynb is responsible for K-means clustering of the dataset, and then saved in different directories.

3.Local_automatic_mask_generator_example.ipynb SAM intervenes in the generation pipeline of the mask image, not able to run directly, this item is ported to Google Colab in the project, and reads the web files to work.

4./SAM_CRAFT/UNET.ipynb The initialization, training, prediction and iteration of the UNET network architecture are done in this link. This is the backbone part of our work.

5./CroppingCraft/CROPPING.ipynb (cropping.py) is responsible for the overlapping and sliding window cropping of the dataset.

6./CroppingCraft/COMBI.ipynb (combi.py) is responsible for the sliding window reduction of the dataset and the central cropping and stitching of the images to be processed.

## 1.Environment Configuration
Python version = 3.9.16

TensorFlow version = 2.10.0 

Numpy version = 1.24.3 

GPU/CPU = GTX1650 

Labelme (python = 3.7) 

## 2. Dataset and Pre-trained Model Preparation
Download our dataset and pre-trained model (\SAM_CRAFT\RESULT) here.

A simple merge of our code with the sibling directory of the dataset will give us the complete project.
If you want to customize the dataset and generate it with SAM, you can refer to our Local_automatic_mask_generator_example.ipynb

## 2.Build the Network and Train
Just run the code in UNET normally
You can change the parameters(trainset, testset, batch_size, epoch)in this training code:
```
results = model.fit(X_train,Y_train, batch_size = 8, epochs = 15)
```
You can load the model in thiscode:
```
model = tf.keras.models.load_model('SAMPLE.h5')
```
You can predict the TEST_SET in thiscode:
```
prediction_train = model.predict(TEST,verbose = 1)  
prediction_train_t = (prediction_train<0.5).astype(np.uint8)
```
You can evaluate the result in thiscode:
```
miou = compute_miou(prediction_train_t, TEST_MASK)  
conF1 = compute_conF1(prediction_train_t, TEST_MASK)  
print("mIOU:", miou)  
print("F1:", conF1)
```
## 3.Iterative training
The second half of the UNET code involves iterative processing of the manually labeled test set, which can be run normally to save the new training set, and then just continue training and evaluation as above. The number of iterations depends entirely on the user's preference.

## 4.Applications on high resolution images
If your dataset comes from a Cropping.py crop, then you can deliver its output to Combi.py for recombination after making predictions. You will end up with a complete predicted result without costing more performance.
