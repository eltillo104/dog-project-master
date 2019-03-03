
# Artificial Intelligence Nanodegree

## Convolutional Neural Networks

## Project: Write an Algorithm for a Dog Identification App 

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this IPython notebook.



---
### Why We're Here 

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 6](#step6): Write your Algorithm
* [Step 7](#step7): Test Your Algorithm

---
<a id='step0'></a>
## Step 0: Import Datasets

### Import Dog Dataset

In the code cell below, we import a dataset of dog images.  We populate a few variables through the use of the `load_files` function from the scikit-learn library:
- `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
- `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels 
- `dog_names` - list of string-valued dog breed names for translating labels


```python
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
```

    Using TensorFlow backend.
    

    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.
    

### Import Human Dataset

In the code cell below, we import a dataset of human images, where the file paths are stored in the numpy array `human_files`.


```python
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))
```

    There are 13234 total human images.
    

---
<a id='step1'></a>
## Step 1: Detect Humans

We use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.

In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[110])
print(human_files[110])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,123,0),5)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    lfw\John_Howard\John_Howard_0016.jpg
    Number of faces detected: 1
    


![png](output_5_1.png)


Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.


```python
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### (IMPLEMENTATION) Assess the Human Face Detector

__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?  
- What percentage of the first 100 images in `dog_files` have a detected human face? 

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

__Answer:__ 


```python
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
per = 0
# Do NOT modify the code above this line.
def human_faces(array):
    total = 0
    perc = 0
    for x in range (len(array)):
        result = face_detector(array[x])
        if(result==True):
            perc = perc+1
        total = total+1
    return((perc/total)*100)
per = human_faces(human_files_short)
print('There are %s total human faces detected in human images.' % "{0:.0f}%".format(per))
per = human_faces(dog_files_short)
print('There are %s total human faces detected in dog images.' % "{0:.0f}%".format(per))
## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
```

    There are 100% total human faces detected in human images.
    There are 12% total human faces detected in dog images.
    

__Question 2:__ This algorithmic choice necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unneccessarily frustrated users!). In your opinion, is this a reasonable expectation to pose on the user? If not, can you think of a way to detect humans in images that does not necessitate an image with a clearly presented face?

__Answer:__ The solution is to train a Deep Convolutional Neural Network. But instead of training the network to recognize pictures objects, we are going to train it to generate 128 measurements for each face.
Geitgey, Alan. "Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning" Medium. 24 Jul. 2016.

We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on each of the datasets.


```python
## (Optional) TODO: Report the performance of another  
## face detection algorithm on the LFW dataset
### Feel free to use as many code cells as needed.
```

---
<a id='step2'></a>
## Step 2: Detect Dogs

In this section, we use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.


```python
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
```

### Pre-process the Data

When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

$$
(\text{nb_samples}, \text{rows}, \text{columns}, \text{channels}),
$$

where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.  

The `path_to_tensor` function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.  The function first loads the image and resizes it to a square image that is $224 \times 224$ pixels.  Next, the image is converted to an array, which is then resized to a 4D tensor.  In this case, since we are working with color images, each image has three channels.  Likewise, since we are processing a single image (or sample), the returned tensor will always have shape

$$
(1, 224, 224, 3).
$$

The `paths_to_tensor` function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape 

$$
(\text{nb_samples}, 224, 224, 3).
$$

Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths.  It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!


```python
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

### Making Predictions with ResNet-50

Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing.  First, the RGB image is converted to BGR by reordering the channels.  All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as $[103.939, 116.779, 123.68]$ and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image.  This is implemented in the imported function `preprocess_input`.  If you're curious, you can check the code for `preprocess_input` [here](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py).

Now that we have a way to format our image for supplying to ResNet-50, we are now ready to use the model to extract the predictions.  This is accomplished with the `predict` method, which returns an array whose $i$-th entry is the model's predicted probability that the image belongs to the $i$-th ImageNet category.  This is implemented in the `ResNet50_predict_labels` function below.

By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). 


```python
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```

### Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the `ResNet50_predict_labels` function above returns a value between 151 and 268 (inclusive).

We use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
```

### (IMPLEMENTATION) Assess the Dog Detector

__Question 3:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

__Answer:__ 


```python
percent = 0
def dog_det(array):
    total = 0
    perc = 0
    for x in range (len(array)):
        result = dog_detector(array[x])
        #print(result)
        if(result==True):
            perc = perc+1
        total = total+1
    return((perc/total)*100)
percent = dog_det(human_files_short)
print('There are %s total dogs detected in human images.' % "{0:.0f}%".format(percent))
percent = dog_det(dog_files_short)
print('There are %s total dogs detected in dog images.' % "{0:.0f}%".format(percent))
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
```

    There are 1% total dogs detected in human images.
    There are 100% total dogs detected in dog images.
    

---
<a id='step3'></a>
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 1%.  In Step 5 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

Be careful with adding too many trainable layers!  More parameters means longer training, which means you are more likely to need a GPU to accelerate the training process.  Thankfully, Keras provides a handy estimate of the time that each epoch is likely to take; you can extrapolate this estimate to figure out how long it will take for your algorithm to train. 

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.  

Brittany | Welsh Springer Spaniel
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun! 

### Pre-process the Data

We rescale the images by dividing every pixel in every image by 255.


```python
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

    100%|██████████████████████████████████████████████████████████████████████████████| 6680/6680 [01:42<00:00, 64.93it/s]
    100%|████████████████████████████████████████████████████████████████████████████████| 835/835 [00:11<00:00, 71.45it/s]
    100%|████████████████████████████████████████████████████████████████████████████████| 836/836 [00:22<00:00, 37.23it/s]
    

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        model.summary()

We have imported some Python modules to get you started, but feel free to import as many modules as you need.  If you end up getting stuck, here's a hint that specifies a model that trains relatively fast on CPU and attains >1% test accuracy in 5 epochs:

![Sample CNN](images/sample_cnn.png)
           
__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task.

__Answer:__ I think this architecture will work good because it has 3 sets of convolutional with max pooling layers. The first layer detects lines and blobs, the second layer detects circles, stripes and rectangles, the third layer detects grids, honeycombs and faces. I could add more convolutional and max pooling layers but I will have more parameters and the training will be slower. I understand that if I add more layers it will detect more complex features, but I feel this will work for the required greater than 1%.


```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=2, padding='valid', activation='relu', 
                        input_shape=(224, 224, 3))) # I implemented a convolutional layer with 16 filters and padding to valid
                                                    # to obtain a layer of filters with size 223x223 with a depth of 16
model.add(MaxPooling2D(pool_size=2)) # I implemented a max pooling layer of size 2 to reduce the pixels in the filters by half
model.add(Conv2D(filters=32, kernel_size=2, padding='valid', activation='relu')) # I implemented a convolutional layer with 32 filters
                                    # and padding to valid to obtain a layer of filters with size 110x110 with a depth of 32            
model.add(MaxPooling2D(pool_size=2)) # I implemented another max pooling layer to reduce the pixels in the filters again by half
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')) # I implemented a convolutional layer with 64 filters and padding to same
                                                    # to obtain a layer of filters with size 55x55 with a depth of 64
model.add(GlobalAveragePooling2D()) # I implemented a global average pooling layer to obtain the average for each filter to convert it to a vector
model.add(Dropout(0.3)) # I implemented a Dropout layer to avoid overfitting
model.add(Dense(133, activation='softmax')) # I implemented a final output layer of 133 different classifications for dog breeds using 
                                            # softmax to obtain the probabilities of each dog breed
### TODO: Define your architecture.

model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 223, 223, 16)      208       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 111, 111, 16)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 110, 110, 32)      2080      
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 55, 55, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 55, 55, 64)        8256      
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 64)                0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 133)               8645      
    =================================================================
    Total params: 19,189
    Trainable params: 19,189
    Non-trainable params: 0
    _________________________________________________________________
    

### Compile the Model


```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 5

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    
    Epoch 5/5
    4080/6680 [=================>............] - ETA: 6:21 - loss: 4.8591 - acc: 0.100 - ETA: 6:12 - loss: 4.8233 - acc: 0.050 - ETA: 6:25 - loss: 4.7927 - acc: 0.050 - ETA: 6:16 - loss: 4.8065 - acc: 0.037 - ETA: 6:18 - loss: 4.8034 - acc: 0.030 - ETA: 6:11 - loss: 4.8037 - acc: 0.025 - ETA: 6:13 - loss: 4.7969 - acc: 0.021 - ETA: 6:05 - loss: 4.8147 - acc: 0.018 - ETA: 6:02 - loss: 4.8172 - acc: 0.016 - ETA: 5:59 - loss: 4.8099 - acc: 0.020 - ETA: 5:54 - loss: 4.8068 - acc: 0.018 - ETA: 5:49 - loss: 4.7958 - acc: 0.020 - ETA: 5:46 - loss: 4.7957 - acc: 0.023 - ETA: 5:42 - loss: 4.7887 - acc: 0.025 - ETA: 5:42 - loss: 4.7891 - acc: 0.023 - ETA: 5:38 - loss: 4.7906 - acc: 0.025 - ETA: 5:35 - loss: 4.7889 - acc: 0.023 - ETA: 5:33 - loss: 4.7838 - acc: 0.022 - ETA: 5:30 - loss: 4.7876 - acc: 0.021 - ETA: 5:27 - loss: 4.7873 - acc: 0.020 - ETA: 5:25 - loss: 4.7908 - acc: 0.021 - ETA: 5:23 - loss: 4.7889 - acc: 0.020 - ETA: 5:23 - loss: 4.7840 - acc: 0.019 - ETA: 5:20 - loss: 4.7794 - acc: 0.018 - ETA: 5:18 - loss: 4.7756 - acc: 0.024 - ETA: 5:16 - loss: 4.7846 - acc: 0.023 - ETA: 5:16 - loss: 4.7820 - acc: 0.024 - ETA: 5:15 - loss: 4.7758 - acc: 0.023 - ETA: 5:13 - loss: 4.7757 - acc: 0.025 - ETA: 5:12 - loss: 4.7791 - acc: 0.026 - ETA: 5:10 - loss: 4.7786 - acc: 0.025 - ETA: 5:08 - loss: 4.7791 - acc: 0.026 - ETA: 5:08 - loss: 4.7817 - acc: 0.025 - ETA: 5:06 - loss: 4.7776 - acc: 0.025 - ETA: 5:04 - loss: 4.7762 - acc: 0.025 - ETA: 5:04 - loss: 4.7765 - acc: 0.025 - ETA: 5:03 - loss: 4.7708 - acc: 0.024 - ETA: 5:03 - loss: 4.7681 - acc: 0.023 - ETA: 5:02 - loss: 4.7650 - acc: 0.024 - ETA: 5:01 - loss: 4.7616 - acc: 0.023 - ETA: 4:59 - loss: 4.7586 - acc: 0.023 - ETA: 4:58 - loss: 4.7514 - acc: 0.023 - ETA: 4:56 - loss: 4.7498 - acc: 0.023 - ETA: 4:55 - loss: 4.7503 - acc: 0.022 - ETA: 4:53 - loss: 4.7543 - acc: 0.022 - ETA: 4:52 - loss: 4.7565 - acc: 0.021 - ETA: 4:51 - loss: 4.7547 - acc: 0.022 - ETA: 4:49 - loss: 4.7514 - acc: 0.022 - ETA: 4:48 - loss: 4.7507 - acc: 0.023 - ETA: 4:47 - loss: 4.7540 - acc: 0.023 - ETA: 4:46 - loss: 4.7520 - acc: 0.023 - ETA: 4:45 - loss: 4.7533 - acc: 0.023 - ETA: 4:43 - loss: 4.7515 - acc: 0.023 - ETA: 4:42 - loss: 4.7565 - acc: 0.023 - ETA: 4:41 - loss: 4.7572 - acc: 0.023 - ETA: 4:40 - loss: 4.7591 - acc: 0.023 - ETA: 4:39 - loss: 4.7573 - acc: 0.023 - ETA: 4:38 - loss: 4.7524 - acc: 0.024 - ETA: 4:37 - loss: 4.7494 - acc: 0.024 - ETA: 4:36 - loss: 4.7529 - acc: 0.024 - ETA: 4:34 - loss: 4.7534 - acc: 0.023 - ETA: 4:37 - loss: 4.7500 - acc: 0.024 - ETA: 4:35 - loss: 4.7519 - acc: 0.023 - ETA: 4:34 - loss: 4.7548 - acc: 0.024 - ETA: 4:33 - loss: 4.7534 - acc: 0.023 - ETA: 4:32 - loss: 4.7542 - acc: 0.023 - ETA: 4:30 - loss: 4.7541 - acc: 0.023 - ETA: 4:30 - loss: 4.7517 - acc: 0.022 - ETA: 4:29 - loss: 4.7517 - acc: 0.022 - ETA: 4:28 - loss: 4.7498 - acc: 0.024 - ETA: 4:27 - loss: 4.7486 - acc: 0.023 - ETA: 4:25 - loss: 4.7488 - acc: 0.023 - ETA: 4:24 - loss: 4.7506 - acc: 0.024 - ETA: 4:24 - loss: 4.7503 - acc: 0.023 - ETA: 4:23 - loss: 4.7517 - acc: 0.023 - ETA: 4:23 - loss: 4.7540 - acc: 0.023 - ETA: 4:22 - loss: 4.7548 - acc: 0.024 - ETA: 4:21 - loss: 4.7546 - acc: 0.025 - ETA: 4:21 - loss: 4.7551 - acc: 0.024 - ETA: 4:20 - loss: 4.7571 - acc: 0.024 - ETA: 4:19 - loss: 4.7578 - acc: 0.024 - ETA: 4:18 - loss: 4.7574 - acc: 0.023 - ETA: 4:18 - loss: 4.7584 - acc: 0.024 - ETA: 4:17 - loss: 4.7572 - acc: 0.023 - ETA: 4:16 - loss: 4.7569 - acc: 0.024 - ETA: 4:15 - loss: 4.7574 - acc: 0.023 - ETA: 4:14 - loss: 4.7569 - acc: 0.024 - ETA: 4:13 - loss: 4.7571 - acc: 0.025 - ETA: 4:12 - loss: 4.7561 - acc: 0.025 - ETA: 4:12 - loss: 4.7566 - acc: 0.025 - ETA: 4:11 - loss: 4.7569 - acc: 0.025 - ETA: 4:10 - loss: 4.7588 - acc: 0.025 - ETA: 4:09 - loss: 4.7594 - acc: 0.025 - ETA: 4:09 - loss: 4.7593 - acc: 0.025 - ETA: 4:08 - loss: 4.7599 - acc: 0.025 - ETA: 4:07 - loss: 4.7601 - acc: 0.025 - ETA: 4:06 - loss: 4.7595 - acc: 0.024 - ETA: 4:05 - loss: 4.7600 - acc: 0.024 - ETA: 4:04 - loss: 4.7604 - acc: 0.024 - ETA: 4:03 - loss: 4.7616 - acc: 0.024 - ETA: 4:02 - loss: 4.7620 - acc: 0.024 - ETA: 4:01 - loss: 4.7634 - acc: 0.024 - ETA: 4:01 - loss: 4.7617 - acc: 0.024 - ETA: 4:00 - loss: 4.7606 - acc: 0.025 - ETA: 3:59 - loss: 4.7627 - acc: 0.024 - ETA: 3:58 - loss: 4.7623 - acc: 0.024 - ETA: 3:59 - loss: 4.7622 - acc: 0.024 - ETA: 3:58 - loss: 4.7637 - acc: 0.025 - ETA: 3:57 - loss: 4.7642 - acc: 0.025 - ETA: 3:56 - loss: 4.7657 - acc: 0.025 - ETA: 3:55 - loss: 4.7668 - acc: 0.025 - ETA: 3:54 - loss: 4.7668 - acc: 0.025 - ETA: 3:53 - loss: 4.7669 - acc: 0.024 - ETA: 3:52 - loss: 4.7669 - acc: 0.024 - ETA: 3:51 - loss: 4.7669 - acc: 0.024 - ETA: 3:50 - loss: 4.7665 - acc: 0.024 - ETA: 3:49 - loss: 4.7673 - acc: 0.024 - ETA: 3:48 - loss: 4.7682 - acc: 0.024 - ETA: 3:47 - loss: 4.7683 - acc: 0.023 - ETA: 3:46 - loss: 4.7681 - acc: 0.023 - ETA: 3:45 - loss: 4.7682 - acc: 0.023 - ETA: 3:44 - loss: 4.7692 - acc: 0.023 - ETA: 3:43 - loss: 4.7689 - acc: 0.023 - ETA: 3:43 - loss: 4.7692 - acc: 0.023 - ETA: 3:42 - loss: 4.7694 - acc: 0.023 - ETA: 3:41 - loss: 4.7707 - acc: 0.023 - ETA: 3:40 - loss: 4.7714 - acc: 0.023 - ETA: 3:39 - loss: 4.7710 - acc: 0.023 - ETA: 3:38 - loss: 4.7708 - acc: 0.023 - ETA: 3:37 - loss: 4.7703 - acc: 0.023 - ETA: 3:36 - loss: 4.7710 - acc: 0.023 - ETA: 3:35 - loss: 4.7724 - acc: 0.023 - ETA: 3:34 - loss: 4.7730 - acc: 0.023 - ETA: 3:33 - loss: 4.7726 - acc: 0.023 - ETA: 3:32 - loss: 4.7726 - acc: 0.023 - ETA: 3:31 - loss: 4.7724 - acc: 0.023 - ETA: 3:30 - loss: 4.7715 - acc: 0.023 - ETA: 3:29 - loss: 4.7709 - acc: 0.023 - ETA: 3:28 - loss: 4.7707 - acc: 0.023 - ETA: 3:26 - loss: 4.7717 - acc: 0.023 - ETA: 3:25 - loss: 4.7725 - acc: 0.023 - ETA: 3:24 - loss: 4.7722 - acc: 0.022 - ETA: 3:23 - loss: 4.7724 - acc: 0.022 - ETA: 3:22 - loss: 4.7727 - acc: 0.022 - ETA: 3:21 - loss: 4.7717 - acc: 0.022 - ETA: 3:20 - loss: 4.7723 - acc: 0.022 - ETA: 3:19 - loss: 4.7721 - acc: 0.022 - ETA: 3:18 - loss: 4.7710 - acc: 0.022 - ETA: 3:16 - loss: 4.7715 - acc: 0.021 - ETA: 3:15 - loss: 4.7726 - acc: 0.021 - ETA: 3:14 - loss: 4.7719 - acc: 0.021 - ETA: 3:13 - loss: 4.7722 - acc: 0.022 - ETA: 3:12 - loss: 4.7728 - acc: 0.022 - ETA: 3:11 - loss: 4.7724 - acc: 0.022 - ETA: 3:10 - loss: 4.7725 - acc: 0.021 - ETA: 3:11 - loss: 4.7733 - acc: 0.021 - ETA: 3:10 - loss: 4.7734 - acc: 0.021 - ETA: 3:08 - loss: 4.7733 - acc: 0.021 - ETA: 3:07 - loss: 4.7737 - acc: 0.021 - ETA: 3:06 - loss: 4.7727 - acc: 0.021 - ETA: 3:05 - loss: 4.7733 - acc: 0.021 - ETA: 3:04 - loss: 4.7738 - acc: 0.021 - ETA: 3:03 - loss: 4.7738 - acc: 0.021 - ETA: 3:01 - loss: 4.7736 - acc: 0.021 - ETA: 3:00 - loss: 4.7742 - acc: 0.021 - ETA: 2:59 - loss: 4.7739 - acc: 0.021 - ETA: 2:58 - loss: 4.7734 - acc: 0.021 - ETA: 2:57 - loss: 4.7734 - acc: 0.021 - ETA: 2:56 - loss: 4.7733 - acc: 0.021 - ETA: 2:54 - loss: 4.7738 - acc: 0.021 - ETA: 2:53 - loss: 4.7738 - acc: 0.021 - ETA: 2:52 - loss: 4.7741 - acc: 0.020 - ETA: 2:51 - loss: 4.7736 - acc: 0.020 - ETA: 2:50 - loss: 4.7735 - acc: 0.020 - ETA: 2:49 - loss: 4.7728 - acc: 0.020 - ETA: 2:49 - loss: 4.7736 - acc: 0.020 - ETA: 2:48 - loss: 4.7741 - acc: 0.020 - ETA: 2:47 - loss: 4.7741 - acc: 0.021 - ETA: 2:46 - loss: 4.7758 - acc: 0.020 - ETA: 2:45 - loss: 4.7767 - acc: 0.021 - ETA: 2:44 - loss: 4.7768 - acc: 0.021 - ETA: 2:43 - loss: 4.7778 - acc: 0.020 - ETA: 2:41 - loss: 4.7779 - acc: 0.021 - ETA: 2:40 - loss: 4.7781 - acc: 0.020 - ETA: 2:39 - loss: 4.7777 - acc: 0.021 - ETA: 2:39 - loss: 4.7776 - acc: 0.021 - ETA: 2:38 - loss: 4.7779 - acc: 0.020 - ETA: 2:36 - loss: 4.7777 - acc: 0.021 - ETA: 2:35 - loss: 4.7776 - acc: 0.020 - ETA: 2:34 - loss: 4.7779 - acc: 0.021 - ETA: 2:33 - loss: 4.7783 - acc: 0.021 - ETA: 2:32 - loss: 4.7775 - acc: 0.021 - ETA: 2:31 - loss: 4.7779 - acc: 0.021 - ETA: 2:30 - loss: 4.7782 - acc: 0.021 - ETA: 2:29 - loss: 4.7780 - acc: 0.021 - ETA: 2:28 - loss: 4.7777 - acc: 0.021 - ETA: 2:27 - loss: 4.7778 - acc: 0.021 - ETA: 2:26 - loss: 4.7777 - acc: 0.021 - ETA: 2:25 - loss: 4.7775 - acc: 0.021 - ETA: 2:24 - loss: 4.7772 - acc: 0.021 - ETA: 2:23 - loss: 4.7764 - acc: 0.021 - ETA: 2:22 - loss: 4.7756 - acc: 0.021 - ETA: 2:21 - loss: 4.7749 - acc: 0.021 - ETA: 2:20 - loss: 4.7744 - acc: 0.02186680/6680 [==============================] - ETA: 2:19 - loss: 4.7744 - acc: 0.021 - ETA: 2:18 - loss: 4.7745 - acc: 0.021 - ETA: 2:17 - loss: 4.7749 - acc: 0.021 - ETA: 2:15 - loss: 4.7752 - acc: 0.021 - ETA: 2:14 - loss: 4.7755 - acc: 0.021 - ETA: 2:13 - loss: 4.7753 - acc: 0.021 - ETA: 2:12 - loss: 4.7754 - acc: 0.021 - ETA: 2:11 - loss: 4.7761 - acc: 0.021 - ETA: 2:10 - loss: 4.7759 - acc: 0.021 - ETA: 2:09 - loss: 4.7767 - acc: 0.021 - ETA: 2:08 - loss: 4.7771 - acc: 0.021 - ETA: 2:07 - loss: 4.7767 - acc: 0.021 - ETA: 2:06 - loss: 4.7775 - acc: 0.021 - ETA: 2:06 - loss: 4.7776 - acc: 0.021 - ETA: 2:05 - loss: 4.7775 - acc: 0.021 - ETA: 2:04 - loss: 4.7776 - acc: 0.020 - ETA: 2:02 - loss: 4.7780 - acc: 0.020 - ETA: 2:01 - loss: 4.7778 - acc: 0.020 - ETA: 2:00 - loss: 4.7775 - acc: 0.020 - ETA: 1:59 - loss: 4.7781 - acc: 0.020 - ETA: 1:58 - loss: 4.7777 - acc: 0.021 - ETA: 1:57 - loss: 4.7774 - acc: 0.021 - ETA: 1:56 - loss: 4.7774 - acc: 0.021 - ETA: 1:55 - loss: 4.7777 - acc: 0.021 - ETA: 1:54 - loss: 4.7782 - acc: 0.021 - ETA: 1:53 - loss: 4.7775 - acc: 0.020 - ETA: 1:52 - loss: 4.7770 - acc: 0.020 - ETA: 1:51 - loss: 4.7771 - acc: 0.020 - ETA: 1:49 - loss: 4.7767 - acc: 0.020 - ETA: 1:48 - loss: 4.7767 - acc: 0.020 - ETA: 1:47 - loss: 4.7767 - acc: 0.020 - ETA: 1:46 - loss: 4.7764 - acc: 0.020 - ETA: 1:45 - loss: 4.7758 - acc: 0.020 - ETA: 1:44 - loss: 4.7747 - acc: 0.020 - ETA: 1:43 - loss: 4.7752 - acc: 0.020 - ETA: 1:42 - loss: 4.7753 - acc: 0.020 - ETA: 1:41 - loss: 4.7754 - acc: 0.020 - ETA: 1:39 - loss: 4.7756 - acc: 0.020 - ETA: 1:38 - loss: 4.7752 - acc: 0.020 - ETA: 1:37 - loss: 4.7747 - acc: 0.020 - ETA: 1:36 - loss: 4.7748 - acc: 0.020 - ETA: 1:35 - loss: 4.7749 - acc: 0.020 - ETA: 1:34 - loss: 4.7744 - acc: 0.020 - ETA: 1:33 - loss: 4.7740 - acc: 0.020 - ETA: 1:32 - loss: 4.7731 - acc: 0.020 - ETA: 1:31 - loss: 4.7739 - acc: 0.020 - ETA: 1:29 - loss: 4.7739 - acc: 0.020 - ETA: 1:28 - loss: 4.7748 - acc: 0.020 - ETA: 1:27 - loss: 4.7747 - acc: 0.020 - ETA: 1:26 - loss: 4.7746 - acc: 0.020 - ETA: 1:25 - loss: 4.7743 - acc: 0.020 - ETA: 1:24 - loss: 4.7739 - acc: 0.020 - ETA: 1:23 - loss: 4.7739 - acc: 0.020 - ETA: 1:22 - loss: 4.7743 - acc: 0.020 - ETA: 1:21 - loss: 4.7746 - acc: 0.020 - ETA: 1:19 - loss: 4.7751 - acc: 0.020 - ETA: 1:18 - loss: 4.7752 - acc: 0.020 - ETA: 1:17 - loss: 4.7756 - acc: 0.020 - ETA: 1:16 - loss: 4.7758 - acc: 0.020 - ETA: 1:15 - loss: 4.7754 - acc: 0.020 - ETA: 1:14 - loss: 4.7752 - acc: 0.020 - ETA: 1:13 - loss: 4.7752 - acc: 0.020 - ETA: 1:12 - loss: 4.7751 - acc: 0.020 - ETA: 1:11 - loss: 4.7748 - acc: 0.020 - ETA: 1:10 - loss: 4.7749 - acc: 0.020 - ETA: 1:08 - loss: 4.7751 - acc: 0.020 - ETA: 1:07 - loss: 4.7755 - acc: 0.020 - ETA: 1:06 - loss: 4.7758 - acc: 0.020 - ETA: 1:05 - loss: 4.7758 - acc: 0.020 - ETA: 1:04 - loss: 4.7756 - acc: 0.020 - ETA: 1:03 - loss: 4.7758 - acc: 0.020 - ETA: 1:02 - loss: 4.7753 - acc: 0.019 - ETA: 1:01 - loss: 4.7758 - acc: 0.019 - ETA: 1:00 - loss: 4.7755 - acc: 0.020 - ETA: 59s - loss: 4.7755 - acc: 0.020 - ETA: 58s - loss: 4.7758 - acc: 0.02 - ETA: 57s - loss: 4.7761 - acc: 0.01 - ETA: 56s - loss: 4.7760 - acc: 0.02 - ETA: 54s - loss: 4.7759 - acc: 0.02 - ETA: 53s - loss: 4.7756 - acc: 0.02 - ETA: 52s - loss: 4.7756 - acc: 0.02 - ETA: 51s - loss: 4.7756 - acc: 0.02 - ETA: 50s - loss: 4.7756 - acc: 0.02 - ETA: 49s - loss: 4.7748 - acc: 0.02 - ETA: 48s - loss: 4.7742 - acc: 0.02 - ETA: 47s - loss: 4.7745 - acc: 0.02 - ETA: 46s - loss: 4.7742 - acc: 0.02 - ETA: 45s - loss: 4.7743 - acc: 0.02 - ETA: 44s - loss: 4.7740 - acc: 0.02 - ETA: 43s - loss: 4.7748 - acc: 0.02 - ETA: 42s - loss: 4.7752 - acc: 0.02 - ETA: 41s - loss: 4.7753 - acc: 0.02 - ETA: 40s - loss: 4.7749 - acc: 0.02 - ETA: 39s - loss: 4.7745 - acc: 0.02 - ETA: 37s - loss: 4.7747 - acc: 0.02 - ETA: 36s - loss: 4.7744 - acc: 0.02 - ETA: 35s - loss: 4.7749 - acc: 0.02 - ETA: 34s - loss: 4.7752 - acc: 0.02 - ETA: 33s - loss: 4.7755 - acc: 0.02 - ETA: 32s - loss: 4.7757 - acc: 0.02 - ETA: 31s - loss: 4.7757 - acc: 0.02 - ETA: 30s - loss: 4.7759 - acc: 0.02 - ETA: 29s - loss: 4.7755 - acc: 0.02 - ETA: 28s - loss: 4.7754 - acc: 0.02 - ETA: 27s - loss: 4.7753 - acc: 0.02 - ETA: 26s - loss: 4.7753 - acc: 0.02 - ETA: 25s - loss: 4.7748 - acc: 0.01 - ETA: 23s - loss: 4.7740 - acc: 0.02 - ETA: 22s - loss: 4.7738 - acc: 0.02 - ETA: 21s - loss: 4.7738 - acc: 0.02 - ETA: 20s - loss: 4.7736 - acc: 0.02 - ETA: 19s - loss: 4.7734 - acc: 0.02 - ETA: 18s - loss: 4.7734 - acc: 0.02 - ETA: 17s - loss: 4.7733 - acc: 0.02 - ETA: 16s - loss: 4.7728 - acc: 0.01 - ETA: 15s - loss: 4.7723 - acc: 0.02 - ETA: 14s - loss: 4.7717 - acc: 0.02 - ETA: 13s - loss: 4.7715 - acc: 0.02 - ETA: 12s - loss: 4.7716 - acc: 0.02 - ETA: 10s - loss: 4.7718 - acc: 0.02 - ETA: 9s - loss: 4.7719 - acc: 0.0202 - ETA: 8s - loss: 4.7716 - acc: 0.020 - ETA: 7s - loss: 4.7716 - acc: 0.020 - ETA: 6s - loss: 4.7718 - acc: 0.020 - ETA: 5s - loss: 4.7718 - acc: 0.020 - ETA: 4s - loss: 4.7718 - acc: 0.020 - ETA: 3s - loss: 4.7722 - acc: 0.019 - ETA: 2s - loss: 4.7719 - acc: 0.019 - ETA: 1s - loss: 4.7720 - acc: 0.019 - 383s 57ms/step - loss: 4.7724 - acc: 0.0198 - val_loss: 4.7847 - val_acc: 0.0323
    
    Epoch 00005: val_loss improved from 4.80009 to 4.78471, saving model to saved_models/weights.best.from_scratch.hdf5
    




    <keras.callbacks.History at 0x221da0146a0>



### Load the Model with the Best Validation Loss


```python
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```

### Test the Model

Try out your model on the test dataset of dog images.  Ensure that your test accuracy is greater than 1%.


```python
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 2.5120%
    

---
<a id='step4'></a>
## Step 4: Use a CNN to Classify Dog Breeds

To reduce training time without sacrificing accuracy, we show you how to train a CNN using transfer learning.  In the following step, you will get a chance to use transfer learning to train your own CNN.

### Obtain Bottleneck Features


```python
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
```

### Model Architecture

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model.  We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.


```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_2 ( (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 133)               68229     
    =================================================================
    Total params: 68,229
    Trainable params: 68,229
    Non-trainable params: 0
    _________________________________________________________________
    

### Compile the Model


```python
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### Train the Model


```python
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    
    Epoch 20/20
    6680/6680 [==============================] - ETA: 5s - loss: 10.4768 - acc: 0.35 - ETA: 3s - loss: 5.4113 - acc: 0.6643 - ETA: 3s - loss: 5.5564 - acc: 0.646 - ETA: 3s - loss: 5.5420 - acc: 0.650 - ETA: 3s - loss: 5.4468 - acc: 0.657 - ETA: 3s - loss: 5.4606 - acc: 0.656 - ETA: 3s - loss: 5.6191 - acc: 0.646 - ETA: 2s - loss: 5.6589 - acc: 0.644 - ETA: 2s - loss: 5.6120 - acc: 0.646 - ETA: 2s - loss: 5.6162 - acc: 0.645 - ETA: 2s - loss: 5.5807 - acc: 0.647 - ETA: 2s - loss: 5.5515 - acc: 0.649 - ETA: 2s - loss: 5.6026 - acc: 0.646 - ETA: 2s - loss: 5.5174 - acc: 0.651 - ETA: 2s - loss: 5.4518 - acc: 0.655 - ETA: 2s - loss: 5.4643 - acc: 0.655 - ETA: 2s - loss: 5.4499 - acc: 0.656 - ETA: 2s - loss: 5.4476 - acc: 0.656 - ETA: 2s - loss: 5.4438 - acc: 0.656 - ETA: 2s - loss: 5.5010 - acc: 0.652 - ETA: 2s - loss: 5.5434 - acc: 0.649 - ETA: 2s - loss: 5.5030 - acc: 0.652 - ETA: 2s - loss: 5.4853 - acc: 0.653 - ETA: 2s - loss: 5.5167 - acc: 0.651 - ETA: 2s - loss: 5.4898 - acc: 0.653 - ETA: 1s - loss: 5.4755 - acc: 0.653 - ETA: 1s - loss: 5.4978 - acc: 0.652 - ETA: 1s - loss: 5.4905 - acc: 0.652 - ETA: 1s - loss: 5.5008 - acc: 0.652 - ETA: 1s - loss: 5.4815 - acc: 0.653 - ETA: 1s - loss: 5.4592 - acc: 0.655 - ETA: 1s - loss: 5.4502 - acc: 0.655 - ETA: 1s - loss: 5.4351 - acc: 0.656 - ETA: 1s - loss: 5.4629 - acc: 0.654 - ETA: 1s - loss: 5.4775 - acc: 0.653 - ETA: 1s - loss: 5.4976 - acc: 0.652 - ETA: 1s - loss: 5.4905 - acc: 0.653 - ETA: 1s - loss: 5.5470 - acc: 0.649 - ETA: 1s - loss: 5.5351 - acc: 0.650 - ETA: 1s - loss: 5.5449 - acc: 0.649 - ETA: 1s - loss: 5.5243 - acc: 0.650 - ETA: 1s - loss: 5.5310 - acc: 0.650 - ETA: 0s - loss: 5.5371 - acc: 0.650 - ETA: 0s - loss: 5.5331 - acc: 0.650 - ETA: 0s - loss: 5.5045 - acc: 0.651 - ETA: 0s - loss: 5.5064 - acc: 0.651 - ETA: 0s - loss: 5.5042 - acc: 0.651 - ETA: 0s - loss: 5.5219 - acc: 0.650 - ETA: 0s - loss: 5.5187 - acc: 0.651 - ETA: 0s - loss: 5.5408 - acc: 0.649 - ETA: 0s - loss: 5.5458 - acc: 0.649 - ETA: 0s - loss: 5.5429 - acc: 0.649 - ETA: 0s - loss: 5.5535 - acc: 0.649 - ETA: 0s - loss: 5.5369 - acc: 0.650 - ETA: 0s - loss: 5.5580 - acc: 0.648 - ETA: 0s - loss: 5.5575 - acc: 0.648 - ETA: 0s - loss: 5.5623 - acc: 0.648 - ETA: 0s - loss: 5.5711 - acc: 0.648 - 4s 589us/step - loss: 5.5735 - acc: 0.6478 - val_loss: 6.5738 - val_acc: 0.5078
    
    Epoch 00020: val_loss improved from 6.65655 to 6.57378, saving model to saved_models/weights.best.VGG16.hdf5
    




    <keras.callbacks.History at 0x221de161390>



### Load the Model with the Best Validation Loss


```python
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
```

### Test the Model

Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.  We print the test accuracy below.


```python
# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 50.7177%
    

### Predict Dog Breed with the Model


```python
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step5'></a>
## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

In Step 4, we used transfer learning to create a CNN using VGG-16 bottleneck features.  In this section, you must use the bottleneck features from a different pre-trained model.  To make things easier for you, we have pre-computed the features for all of the networks that are currently available in Keras:
- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

The files are encoded as such:

    Dog{network}Data.npz
    
where `{network}`, in the above filename, can be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.  Pick one of the above architectures, download the corresponding bottleneck features, and store the downloaded file in the `bottleneck_features/` folder in the repository.

### (IMPLEMENTATION) Obtain Bottleneck Features

In the code block below, extract the bottleneck features corresponding to the train, test, and validation sets by running the following:

    bottleneck_features = np.load('bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']


```python
### TODO: Obtain bottleneck features from another pre-trained CNN.
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import random
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline 
from keras.preprocessing import image                  
from tqdm import tqdm
from PIL import ImageFile 
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
bottleneck_features = np.load('bottleneck_features/DogInceptionV3Data.npz')
train_Inception = bottleneck_features['train']
valid_Inception = bottleneck_features['valid']
test_Inception = bottleneck_features['test']
```

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        <your model's name>.summary()
   
__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__ I think the current architecture is suitable for the problem because it would take a long time to use my own model to be trained. I take advantage of the pretrained model since it will be faster to train the model. I also have a better accuracy than with the model I implemented. These are models that are useful for classification tasks.




```python
### TODO: Define your architecture.
Inception_model = Sequential() #I'm using the InceptionV3 model since it performed really good in Large Visual Recognition Challenge by the Google Brain Team.
Inception_model.add(GlobalAveragePooling2D(input_shape=train_Inception.shape[1:])) #I apply a global average pooling layer to convert
                                                                                #matrices into vectors
Inception_model.add(Dense(133, activation='softmax')) # I implemented a fully connected layer for our 133 dog breed categories

Inception_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_3 ( (None, 2048)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 133)               272517    
    =================================================================
    Total params: 272,517
    Trainable params: 272,517
    Non-trainable params: 0
    _________________________________________________________________
    

### (IMPLEMENTATION) Compile the Model


```python
### TODO: Compile the model.
Inception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.  

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
### TODO: Train the model.
checkpointer2 = ModelCheckpoint(filepath='saved_models/weights.best.Inception.hdf5', 
                               verbose=1, save_best_only=True)

Inception_model.fit(train_Inception, train_targets, 
          validation_data=(valid_Inception, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer2], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    
    Epoch 2/20
    6680/6680 [==============================] - ETA: 5s - loss: 0.7281 - acc: 0.850 - ETA: 6s - loss: 0.6865 - acc: 0.812 - ETA: 5s - loss: 0.5072 - acc: 0.856 - ETA: 5s - loss: 0.4260 - acc: 0.877 - ETA: 6s - loss: 0.3969 - acc: 0.882 - ETA: 6s - loss: 0.4262 - acc: 0.876 - ETA: 5s - loss: 0.4173 - acc: 0.869 - ETA: 5s - loss: 0.4226 - acc: 0.864 - ETA: 5s - loss: 0.4186 - acc: 0.871 - ETA: 5s - loss: 0.4351 - acc: 0.873 - ETA: 5s - loss: 0.4494 - acc: 0.873 - ETA: 5s - loss: 0.4685 - acc: 0.867 - ETA: 5s - loss: 0.4726 - acc: 0.861 - ETA: 5s - loss: 0.4768 - acc: 0.857 - ETA: 5s - loss: 0.4867 - acc: 0.851 - ETA: 5s - loss: 0.5028 - acc: 0.848 - ETA: 5s - loss: 0.5034 - acc: 0.850 - ETA: 5s - loss: 0.4942 - acc: 0.851 - ETA: 4s - loss: 0.4864 - acc: 0.852 - ETA: 4s - loss: 0.4808 - acc: 0.853 - ETA: 4s - loss: 0.4816 - acc: 0.850 - ETA: 4s - loss: 0.4812 - acc: 0.850 - ETA: 4s - loss: 0.4796 - acc: 0.849 - ETA: 4s - loss: 0.4738 - acc: 0.850 - ETA: 4s - loss: 0.4681 - acc: 0.853 - ETA: 4s - loss: 0.4694 - acc: 0.851 - ETA: 4s - loss: 0.4618 - acc: 0.853 - ETA: 4s - loss: 0.4623 - acc: 0.854 - ETA: 4s - loss: 0.4576 - acc: 0.854 - ETA: 4s - loss: 0.4588 - acc: 0.853 - ETA: 4s - loss: 0.4622 - acc: 0.853 - ETA: 4s - loss: 0.4633 - acc: 0.852 - ETA: 4s - loss: 0.4560 - acc: 0.855 - ETA: 4s - loss: 0.4570 - acc: 0.855 - ETA: 4s - loss: 0.4578 - acc: 0.853 - ETA: 4s - loss: 0.4554 - acc: 0.854 - ETA: 4s - loss: 0.4611 - acc: 0.853 - ETA: 4s - loss: 0.4568 - acc: 0.854 - ETA: 4s - loss: 0.4564 - acc: 0.854 - ETA: 4s - loss: 0.4657 - acc: 0.853 - ETA: 4s - loss: 0.4611 - acc: 0.853 - ETA: 4s - loss: 0.4655 - acc: 0.852 - ETA: 4s - loss: 0.4679 - acc: 0.851 - ETA: 4s - loss: 0.4620 - acc: 0.853 - ETA: 3s - loss: 0.4581 - acc: 0.853 - ETA: 3s - loss: 0.4645 - acc: 0.852 - ETA: 3s - loss: 0.4722 - acc: 0.850 - ETA: 3s - loss: 0.4700 - acc: 0.851 - ETA: 3s - loss: 0.4668 - acc: 0.852 - ETA: 3s - loss: 0.4674 - acc: 0.851 - ETA: 3s - loss: 0.4685 - acc: 0.851 - ETA: 3s - loss: 0.4738 - acc: 0.850 - ETA: 3s - loss: 0.4734 - acc: 0.851 - ETA: 3s - loss: 0.4800 - acc: 0.849 - ETA: 3s - loss: 0.4796 - acc: 0.848 - ETA: 3s - loss: 0.4794 - acc: 0.848 - ETA: 3s - loss: 0.4774 - acc: 0.849 - ETA: 3s - loss: 0.4767 - acc: 0.849 - ETA: 2s - loss: 0.4842 - acc: 0.847 - ETA: 2s - loss: 0.4862 - acc: 0.846 - ETA: 2s - loss: 0.4840 - acc: 0.847 - ETA: 2s - loss: 0.4852 - acc: 0.847 - ETA: 2s - loss: 0.4860 - acc: 0.846 - ETA: 2s - loss: 0.4851 - acc: 0.846 - ETA: 2s - loss: 0.4821 - acc: 0.847 - ETA: 2s - loss: 0.4860 - acc: 0.846 - ETA: 2s - loss: 0.4822 - acc: 0.847 - ETA: 2s - loss: 0.4794 - acc: 0.849 - ETA: 2s - loss: 0.4799 - acc: 0.848 - ETA: 2s - loss: 0.4788 - acc: 0.849 - ETA: 2s - loss: 0.4766 - acc: 0.850 - ETA: 2s - loss: 0.4734 - acc: 0.850 - ETA: 2s - loss: 0.4729 - acc: 0.850 - ETA: 2s - loss: 0.4738 - acc: 0.850 - ETA: 1s - loss: 0.4742 - acc: 0.850 - ETA: 1s - loss: 0.4733 - acc: 0.851 - ETA: 1s - loss: 0.4757 - acc: 0.851 - ETA: 1s - loss: 0.4749 - acc: 0.851 - ETA: 1s - loss: 0.4810 - acc: 0.850 - ETA: 1s - loss: 0.4804 - acc: 0.850 - ETA: 1s - loss: 0.4796 - acc: 0.850 - ETA: 1s - loss: 0.4802 - acc: 0.850 - ETA: 1s - loss: 0.4788 - acc: 0.850 - ETA: 1s - loss: 0.4789 - acc: 0.850 - ETA: 1s - loss: 0.4788 - acc: 0.850 - ETA: 1s - loss: 0.4795 - acc: 0.850 - ETA: 1s - loss: 0.4806 - acc: 0.850 - ETA: 1s - loss: 0.4837 - acc: 0.850 - ETA: 1s - loss: 0.4843 - acc: 0.850 - ETA: 0s - loss: 0.4842 - acc: 0.850 - ETA: 0s - loss: 0.4847 - acc: 0.849 - ETA: 0s - loss: 0.4865 - acc: 0.849 - ETA: 0s - loss: 0.4891 - acc: 0.849 - ETA: 0s - loss: 0.4862 - acc: 0.850 - ETA: 0s - loss: 0.4870 - acc: 0.849 - ETA: 0s - loss: 0.4861 - acc: 0.849 - ETA: 0s - loss: 0.4855 - acc: 0.849 - ETA: 0s - loss: 0.4863 - acc: 0.848 - ETA: 0s - loss: 0.4857 - acc: 0.849 - ETA: 0s - loss: 0.4847 - acc: 0.849 - ETA: 0s - loss: 0.4841 - acc: 0.849 - ETA: 0s - loss: 0.4830 - acc: 0.849 - ETA: 0s - loss: 0.4825 - acc: 0.850 - ETA: 0s - loss: 0.4797 - acc: 0.850 - ETA: 0s - loss: 0.4812 - acc: 0.849 - 7s 1ms/step - loss: 0.4803 - acc: 0.8500 - val_loss: 0.6414 - val_acc: 0.8467
    
    Epoch 00002: val_loss improved from 0.67614 to 0.64140, saving model to saved_models/weights.best.Inception.hdf5
    




    <keras.callbacks.History at 0x221de61beb8>



### (IMPLEMENTATION) Load the Model with the Best Validation Loss


```python
### TODO: Load the model weights with the best validation loss.
Inception_model.load_weights('saved_models/weights.best.Inception.hdf5')
```

### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%.


```python
### TODO: Calculate classification accuracy on the test dataset.
# get index of predicted dog breed for each image in test set
Inception_predictions = [np.argmax(Inception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Inception]

# report test accuracy
test_accuracy = 100*np.sum(np.array(Inception_predictions)==np.argmax(test_targets, axis=1))/len(Inception_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 78.9474%
    

### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan_hound`, etc) that is predicted by your model.  

Similar to the analogous function in Step 5, your function should have three steps:
1. Extract the bottleneck features corresponding to the chosen CNN model.
2. Supply the bottleneck features as input to the model to return the predicted vector.  Note that the argmax of this prediction vector gives the index of the predicted dog breed.
3. Use the `dog_names` array defined in Step 0 of this notebook to return the corresponding breed.

The functions to extract the bottleneck features can be found in `extract_bottleneck_features.py`, and they have been imported in an earlier code cell.  To obtain the bottleneck features corresponding to your chosen CNN architecture, you need to use the function

    extract_{network}
    
where `{network}`, in the above filename, should be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.


```python
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
from extract_bottleneck_features import *

def dog_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Inception_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step6'></a>
## Step 6: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are __required__ to use your CNN from Step 5 to predict dog breed.  

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_output.png)


### (IMPLEMENTATION) Write your Algorithm


```python
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.
personal_files = np.array(glob("personal/*"))
def predictor(imagefile):
    breed = dog_predict_breed(imagefile)
    result = dog_detector(imagefile)
    result2 = face_detector(imagefile)
    if result:
        print("You're a dog")
        doggy = cv2.imread(imagefile)
        plt.imshow(cv2.cvtColor(doggy,cv2.COLOR_BGR2RGB))
        plt.show()
        print("You look like a:")
        print(breed)
    elif result2:
        print("You're a human")
        doggy = cv2.imread(imagefile)
        plt.imshow(cv2.cvtColor(doggy,cv2.COLOR_BGR2RGB))
        plt.show()
        print("You look like a:")
        print(breed)
    else:
        print("Error")
```

---
<a id='step7'></a>
## Step 7: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that __you__ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images!

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ 


```python
x=0
for x in range(0,6):
    predictor(personal_files[x])
```

    You're a human
    


![png](output_65_1.png)


    You look like a:
    Chinese_crested
    You're a human
    


![png](output_65_3.png)


    You look like a:
    Chinese_crested
    You're a dog
    


![png](output_65_5.png)


    You look like a:
    Portuguese_water_dog
    You're a dog
    


![png](output_65_7.png)


    You look like a:
    Bichon_frise
    You're a dog
    


![png](output_65_9.png)


    You look like a:
    Collie
    You're a human
    


![png](output_65_11.png)


    You look like a:
    Chinese_crested
    

The output is better than I expected. The program got 2 out of 3 dogs right, it also detected humans and a funny resemblance.
I can make my algorithm better by:
    Using a different optimizer, Increasing the number of epochs, Using a different pre-trained model
    
