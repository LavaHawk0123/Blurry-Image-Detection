Creating an ML pipeline to help classify images as blurred or clear

### <h1>Directory Structure:</h1>

    .
    ├── model.h5                   # model file stored using model.save()
    ├── train.py                   # file to train and test of the model
    ├── test.py                    # file to test the model `model.h5` 
    └── requirments.txt            # Contains all the packages that needs to be installed for the pipeline to run(Useful for new/edge computing systems)
    
    .
    ├── data                       # Folder containing all the data
        ├── Blurred                # Class `Blurred` containing all the blurred images 
        ├── Undistorted            # Class `Undistorted` containing all the clear images
        └── gaussian_blur_test.png # File to test the model(different from training/test example)



### Note : Make sure the directory structure is maintained to ensure the code runs.
<h1> To run the code</h1>

### Open the terminal and run
```
python3 train.py
```

### Then run:
```
python3 test.py
```
### Note : both train.py and test.py will ask you to enter the path of the class images or the image to test on. Enter the path directly to the folder of the images. The train dataset path should contain 2 classes representing the 2 folders

<h1> Output </h1>

### The model predicts 2 outputs. Each represents the probability that the image belongs to class 1(blurred) or class 2(clear). It outputs both probabilities in the final layer. For model structure, refer below : 
```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_4 (Conv2D)           (None, 198, 198, 32)      896       
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 99, 99, 32)       0         
 2D)                                                             
                                                                 
 dropout_5 (Dropout)         (None, 99, 99, 32)        0         
                                                                 
 conv2d_5 (Conv2D)           (None, 97, 97, 64)        18496     
                                                                 
 max_pooling2d_5 (MaxPooling  (None, 48, 48, 64)       0         
 2D)                                                             
                                                                 
 dropout_6 (Dropout)         (None, 48, 48, 64)        0         
                                                                 
 conv2d_6 (Conv2D)           (None, 46, 46, 128)       73856     
                                                                 
 max_pooling2d_6 (MaxPooling  (None, 23, 23, 128)      0         
 2D)                                                             
                                                                 
 dropout_7 (Dropout)         (None, 23, 23, 128)       0         
                                                                 
 conv2d_7 (Conv2D)           (None, 21, 21, 256)       295168    
                                                                 
 max_pooling2d_7 (MaxPooling  (None, 10, 10, 256)      0         
 2D)                                                             
                                                                 
 dropout_8 (Dropout)         (None, 10, 10, 256)       0         
                                                                 
 flatten_1 (Flatten)         (None, 25600)             0         
                                                                 
 dense_2 (Dense)             (None, 250)               6400250   
                                                                 
 dropout_9 (Dropout)         (None, 250)               0         
                                                                 
 dense_3 (Dense)             (None, 2)                 502       
                                                                 
=================================================================
Total params: 6,789,168
Trainable params: 6,789,168
Non-trainable params: 0
_________________________________________________________________

```
