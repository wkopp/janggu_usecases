# janggu_usecases
Examples for deep learning in genomics using Janggu

## Download the datasets
In order to download the required datasets, enter the 00_preparation folder.
It contains jupyter notebooks that specify and control the data download. 
Furthermore, it sets up the regions of interest for the model training and evaluation.

## JunD prediction

Run the jupyter notebook 'predicting_jund_binding.ipynb' in order to reproduce the results.
Since model fitting is rather time-consuming, you have to outcomment the respective cells in the notebook.
Alternatively, if you have access to a cluster system, you may use the model fitting scripts individually outside of the notebook.

## DeepSEA and DanQ experiments

To train and evaluate the DeepSEA and DanQ comparison, enter the '02_deepsea_danq_prediction' folder and launch the
jupyter notebook 'deepsea_danq_experiments.ipynb'.
For training the respective models, remove the option '-evaluate'.


## CAGE-tag prediction

To reproduce the CAGE-tag prediction use case, enter '03_cage_prediction' and launch the 'predicting_cage_tags.ipynb' notebook.
In order to run the cross-validation analysis, outcomment the respective command line invocations of the script 'cage_prediction.py'.
The script may also be invoked outside of the notebook.
