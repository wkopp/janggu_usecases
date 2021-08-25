# janggu_usecases
Examples for deep learning in genomics using Janggu

## Requirements

```
jupyter
bedtools
pybedtools
samtools
dash
janggu
R
rpy2
tzlocal
r-ggplot2
r-ggrepel
r-dplyr
statsmodels
pandas
numpy
```

These can be installed via conda and pip.

The respective cells in the notebook for installing requirements may be outcommented.

## Download the datasets
In order to download the required datasets, enter the 00_preparation folder.
It contains jupyter notebooks that specify and control the data download. 
Furthermore, it sets up the regions of interest for the model training and evaluation.

## Note

Some of the steps in the notebooks may be outcommented or deactivated, 
including the invocation of time-consuming training steps,
so that during evaluation, they are not re-run. You may either activate them within the notebook
or invoke the scripts on the command line if you wish to train the models from scratch.
It may also be necessary to adapt the use of `CUDA_VISIBLE_DEVICES` (see tensorflow docs). The GPU device is selected via the `-dev` option in use case 2.
These were chosen for our specific setup with 8 GPUs. For example, if you only have access to one GPU specify
`CUDA_VISIBLE_DEVICES=0` before running the scripts.

## JunD prediction

Run the jupyter notebook 'predicting_jund_binding.ipynb' in order to reproduce the results.
You can control on which gpu the models are trained by specifying the environment variable `CUDA_VISIBLE_DEVICES` (see tensorflow documentation).

## DeepSEA and DanQ experiments

To train and evaluate the DeepSEA and DanQ comparison, enter the '02_deepsea_danq_prediction' folder and launch the
jupyter notebook 'deepsea_danq_experiments.ipynb'.
To activate model training, set the parameter `train_models = True`.
Otherwise, the notebook merely evaluates the results.
You may need to adapt `-dev` to select a specfic GPU.


## CAGE-tag prediction

To reproduce the CAGE-tag prediction use case, enter '03_cage_prediction' and launch the 'predicting_cage_tags.ipynb' notebook.
In order to run the cross-validation analysis, outcomment the respective command line invocations of the script 'cage_prediction.py'.
You can control on which gpu the models are trained by specifying the environment variable `CUDA_VISIBLE_DEVICES` (see tensorflow documentation).

## For Google collab:
!pip install tensorflow==2.2 keras==2.4.3
#if you get Error No Module:'keras.engine.topology' use the below command
from tensorflow.keras.layers import InputSpec,Layer
#if you want to run conda commands.use the below command
##################################################################################
! wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
! chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh
! bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')
##################################################################################


