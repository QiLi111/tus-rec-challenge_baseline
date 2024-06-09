# tus-rec-challenge_baseline
This is the official baseline code for TUS-REC Challenge - MICCAI2024

Refer to [Prevost et al. 2018](https://doi.org/10.1016/j.media.2018.06.003) and [Li et al. 2023](https://doi.org/10.1109/TBME.2023.3325551) for more information about the algorithm.

> [!IMPORTANT]  
> The code currently includes only the training process; the testing code will be coming soon in 2~3 weeks.

> [!NOTE]  
>When testing, the algorithm is expected to take the entire scan as input and output two different sets of transformation-representing displacement vectors as results, a set of displacement vectors on individual pixels and a set of displacement vectors on provided landmarks. There is no requirement on how the algorithm is designed internally, for example, whether it is learning-based method; frame-, sequence- or scan-based processing; or, rigid-, affine- or nonrigid transformation assumptions. 

## Install conda environment
``` bash
conda create -n freehand-US python=3.9.13
conda activate freehand-US
pip install -r requirements.txt
conda install pytorch3d --no-deps -c pytorch3d
``` 
## Code usage
This repository provides a framework for freehand US pose regression, including usage of various types of predictions and labels (see [transformation.py](https://github.com/QiLi111/tus-rec-challenge_baseline/blob/main/utils/transform.py)). Please note that the networks used here are small and simplified for demonstration purposes.

For instance, the network can predict the transformation between two US frames as 6 DOF "parameter", and if the label type is "point", the loss is calculated as the point distance (by transforming "parameter" to "point" using functions in [transformation.py](https://github.com/QiLi111/tus-rec-challenge_baseline/blob/main/utils/transform.py)). The command below illustrates an example of training a pose regression model. 

### Code Structure
```
├── train.py # Training script 
├── test.py # Testing script 
├── data_process.py # An example of data load and scan plot
├── utils # Utility scripts 
│ ├── transform.py # Transformation functions
│ ├── loader.py # Data loader
│ ├── loss.py # Loss function and metrics
│ ├── network.py # Networks
│ ├── funs.py # Functions used during training
│ ├── data_process_functions.py # Functions used during data processing
│ ├── rigid_transform_3D.py # Adapted transformation function
├── options # Model hyperparameter 
│ ├── base_options.py # Hyperparameters for dataroot and GPU
│ └── train_options.py # Hyperparameters for training and testing
├── requirements # packages for environment installation
```

### Train
* Case 1: Input two adjacent frames, output the transformation parameter (6 DOF) between them
    ``` bash
    python3 train.py --NUM_SAMPLES 2 --SAMPLE_RANGE 2 --NUM_PRED 1 --PRED_TYPE parameter --LABEL_TYPE point --DATA_PATH Path/To/Dataset --FILENAME_CALIB Path/To/Calibration_matrix
    ``` 
* Case 2: Input an US sequence with ten frames, output multiple transformation parameters (6 DOF) among them
    ``` bash
    python3 train.py --NUM_SAMPLES 10 --SAMPLE_RANGE 10 --NUM_PRED 9 --PRED_TYPE parameter --LABEL_TYPE point --DATA_PATH Path/To/Dataset --FILENAME_CALIB Path/To/Calibration_matrix
    ```

### Test (coming soon)
* Case 1:
    ``` bash
    python3 test.py --NUM_SAMPLES 2 --SAMPLE_RANGE 2 --NUM_PRED 1 --PRED_TYPE parameter --LABEL_TYPE point --DATA_PATH Path/To/Dataset --FILENAME_CALIB Path/To/Calibration_matrix
    ``` 
* Case 2:
    ``` bash
    python3 test.py --NUM_SAMPLES 10 --SAMPLE_RANGE 10 --NUM_PRED 9 --PRED_TYPE parameter --LABEL_TYPE point --DATA_PATH Path/To/Dataset --FILENAME_CALIB Path/To/Calibration_matrix
    ``` 
