
<h1>Flower Image Classifier</h1>
In this project, we'll be creating a deep learning network to classify flowers per the labels provided. This project was established by Udacity and performed within Udacity's GPU enabled workspace, so unfortunately the source files for this project are not included. The project also utilizes transfer learning to import already trained classifiers from the PyTorch package while modifying the classifier attribute of each package.


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

<h2>Instructions for running the classifier</h2>
<ol>
<li>Train a new network on a data set with train.py</li>
<ul>
<li>Basic usage: python train.py data_directory</li>
<li>Prints out training loss, validation loss, and validation accuracy as the network trains</li>
<li>Options:
 <ol>
<li>
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory</li>
<li>Choose architecture: python train.py data_dir --arch "vgg13"</li>
<li>Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20</li>
<li>Use GPU for training: python train.py data_dir --gpu</li>
 <li> 
</ul>
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
