This folder contains the source codes required to run the thesis associated
with it.

This folder includes Python scripts and Jupyter notebooks that were used 
for the research.

baseline_model_support.py: Python script for supporting (by extracting visual 
                       features) the baseline model.

baseline_model_deep.py: Python script for defining, training, and evaluating 
                      the baseline deep learning model.

my_model.py: Python script for defining, training, and evaluating the enhanced 
            model described in the thesis.

language_model.py: Python script for the embeddings from language model 
			 used in the research.

combined_dataset_model.py: Python script for the combined model that uses 
				   both datasets (earth and pics).

data_scraping.ipynb: Jupyter notebook detailing the data collection process 
				from the respective sources.

performance_of_18_models.py: Python script for comparing and analyzing the 
				performance of the 18 models used in this research.

featureExtract56.mat: MATLAB file used for feature extraction, particularly 
				color entropy. Additional .mat files support this.


Raw data: earth.csv and pics.csv.

Processed data: earth_final_model.csv and pics_final_model.csv, 
these are the datasets after preprocessing and feature extraction stages.


Requirements:
Python IDE
jupyter notebook

This study used the following setup:
Hardware
Operating system - Windows 11 Home 64-bit
Processor - Intel i7-9750H
RAM - 16 GB
GPU - NVIDIA GeForce RTX 2070

Software
IDE - Visual Studio, Code - editor - Visual studio code
Language - Python 3.9.2
Tensorflow gpu - 2.9.0
CUDA - 11.8, CuDnn - 8.6


Note

Due to data privacy and protection considerations, some of the data files
 used in the research are not included like the images used. If you're 
interested you can email me at - sericsheon@gmail.com and i send them to you.