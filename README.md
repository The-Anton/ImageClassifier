
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
</ol>




<h2>Instructions for running the classifier</h2>

<ol>
    <li>Train a new network on a data set with train.py</li>
            <ul>
                    <li>Basic usage: python train.py data_directory</li>
                    <li>Prints out training loss, validation loss, and validation accuracy as the network trains</li>
                    <li>Other params for different purpose:
                    <ul>
                        <li>Set directory to save checkpoints:&emsp;&emsp;&emsp;&emsp;<b>"python train.py data_dir --save_dir save_directory"</b></li>
                        <li>Choose architecture:&emsp;&emsp;&emsp;&emsp;&emsp;<b>"python train.py data_dir --arch "vgg13" "</b></li>
                        <li>Set hyperparameters:&emsp;&emsp;&emsp;&emsp;&emsp;<b>"python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20"</b></li>
                        <li>Use GPU for training:&emsp;&emsp;&emsp;&emsp;&emsp;<b>"python train.py data_dir --gpu"</b></li>
                    </ul>
            </ul>
    <li>Predict flower name from an image with <b>predict.py</b> along with the probability of that name. That is, you'll pass in a single image <b>/path/to/image</b> and return     the flower name   and class probability.</li>
        <ul>
            <li>Basic usage: python predict.py /path/to/image checkpoint</li>
            <li>Options:<li>
                <ul>
                    <li>Return top KK most likely classes:&nbsp;               <b>"predict.py input checkpoint --top_k 3"</b></li>
                    <li>Use a mapping of categories to real names:&nbsp;       <b>"python predict.py input checkpoint --category_names cat_to_name.json"</b></li>
                    <li>Use GPU for inference:&nbsp;                           <b>"python predict.py input checkpoint --gpu"</b></li>
                </ul>
        </ul>
</ol>
