# Upsampling eye movement signal using Convolutional Neural Networks

The code for the "Upsampling eye movement signal using Convolutional Neural Networks" paper.
At first use requirements.txt to prepare the environment:

$ pip install -r requirements.txt

Then train the models (it is optional, the models are already trained in h5 files):

$ python train.py - creates models (h5 files) based on datasets.

And you can test:

$ python test.py - tests all models, calculates errors and prints the results in the form of LaTEX tables.

