
<h1>Flower Image Classifier</h1>
In this project, I have created deep learning network to classify flowers per the labels provided. This was a submission project for a AI course provided by Udacity. The project is divided into two parts 1) We select an Neural Network architecture and train it for a centrain number of times using GPU and then saves the model for further use. 2) We load the saved model and provide it with an image to for testing the classification model. The project also utilizes transfer learning to import already trained classifiers from the PyTorch package while modifying the classifier attribute of each package.

## Deep Learning
• ```Challenge```: Udacity Data Scientist Nanodegree project for deep learning module titled as 'Image Classifier with Deep Learning' attempts to train an image classifier to recognize different species of flowers. We can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice we had to train this classifier, then export it for use in our application. We had used a dataset (http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories.

• ```Solution```: Used torchvision to load the data. The dataset is split into three parts, training, validation, and testing. For the training, applied transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. Also need to load in a mapping from category label to category name. Wrote inference for classification after training and testing the model. Then processed a PIL image for use in a PyTorch model. 

• ```Result```: Using the following software and Python libraries: Torch, PIL, Matplotlib.pyplot, Numpy, Seaborn, Torchvision. Thus, achieved an accuracy of 80% on test dataset as a result of above approaches. Performed a sanity check since it's always good to check that there aren't obvious bugs despite achieving a good test accuracy. Plotted the probabilities for the top 5 classes as a bar graph, along with the input image.

### Software and Libraries
This project uses the following software and Python libraries: <br>
NumPy, pandas, Sklearn / scikit-learn, Matplotlib (for data visualization), Seaborn (for data visualization)



<h2>Files Included</h2>

These are the files included as part of the project and what each contains:
<ol>
        <li><b>Image Classifier Project.ipynb:</b> This is the Jupyter notebook where I conducted all my activities, including a little more than what is included in the predict.py and train.py files.</li>

<li><b>Image Classifier Project.html:</b> Same as the file above, except in HTML form.</li>
<li><b>train.py:</b> This file accepts inputs from the command line prompt and takes the work from the Jupyter notebook for the following activities:</li>
    <ul>
            <li>Creating the Datasets</li>
            <li>Creating the Architecture</li>
            <li>Training the model</li>
            <li>Saving the Model</li>
    </ul>
<li><b>predict.py:</b> This file accepts inputs from the command line prompt and takes the work from the Jupyter notebook for the following activities</li>
    <ul>
        <li>Loading the Model</li>
        <li>Class Prediction</li>
    </ul>
</ol>




<h2>Instructions for running the classifier</h2>

<ol>
    <li>Train a new network on a data set with train.py</li>
            <ul>
                    <li>Basic usage: python train.py data_directory</li>
                    <li>Prints out training loss, validation loss, and validation accuracy as the network trains</li>
                    <li>Other params for different purpose:
                    <ul>
                        <li>Set directory to save checkpoints      :<b>"python train.py data_dir --save_dir save_directory"</b></li>
                        <li>Choose architecture                    :<b>"python train.py data_dir --arch "vgg13" "</b></li>
                        <li>Set hyperparameters                    :<b>"python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20"</b></li>
                        <li>Use GPU for training                   :<b>"python train.py data_dir --gpu"</b></li>
                    </ul>
            </ul>
    <li>Predict flower name from an image with <b>predict.py</b> along with the probability of that name. That is, you'll pass in a single image <b>/path/to/image</b> and return     the flower name   and class probability.</li>
        <ul>
            <li>Basic usage: python predict.py /path/to/image checkpoint</li>
            <li>Options:<li>
                <ul>
                    <li>Return top KK most likely classes          :<b>"predict.py input checkpoint --top_k 3"</b></li>
                    <li>Use a mapping of categories to real names  :<b>"python predict.py input checkpoint --category_names cat_to_name.json"</b></li>
                    <li>Use GPU for inference                      :<b>"python predict.py input checkpoint --gpu"</b></li>
                </ul>
        </ul>
</ol>
