# tus-rec-challenge_baseline
This is the official baseline code for TUS-REC Challenge - MICCAI2024

Refer to [Prevost et al. 2018](https://doi.org/10.1016/j.media.2018.06.003) and [Li et al. 2023](https://doi.org/10.1109/TBME.2023.3325551) for more information about the algorithm.

## Install conda environment
``` bash
conda create -n freehand-US python=3.9.13
conda activate freehand-US
pip install -r requirements.txt
conda install pytorch3d --no-deps -c pytorch3d
``` 
## Instruction
This repository provides a framework for freehand US pose regression, including usage of various types of predictions and labels (see [transformation.py](https://github.com/QiLi111/tus-rec-challenge_baseline/blob/main/utils/transform.py)). Please note that the networks used here are small and simplified for demonstration purposes.

For instance, the network can predict the transformation between two US frames as 6 DOF "parameter", and if the label type is "point", the loss is calculated as the point distance (by transforming "parameter" to "point" using functions in [transformation.py](https://github.com/QiLi111/tus-rec-challenge_baseline/blob/96e62989b5b5b04296294cb8d5dff1ca6878266c/utils/transform.py#L267)). The steps below illustrate an example of training a pose regression model and generate 4 kinds of displacements. 

### Steps to run the code
#### 1. Clone the repository.
```
git clone https://github.com/QiLi111/tus-rec-challenge_baseline.git
```

<!-- #### 2. Download data (You may need to install curl)
Note: the mentioned TOKEN is included in the secret link you received via email, e.g., https://zenodo.org/records/11178509?token=<YOUR-TOKEN-HERE/>


* Navigate to the "data" directory
    ``` bash
    cd tus-rec-challenge_baseline/data
    ```

* Download Train data (Part 1)
    ``` bash
    curl --cookie-jar zenodo-cookies_1.txt "https://zenodo.org/record/11178509?token=<YOUR-TOKEN-HERE>"
    ``` 
    ``` bash
    curl --cookie zenodo-cookies_1.txt "https://zenodo.org/api/records/11178509"
    ```
    ``` bash
    curl --cookie zenodo-cookies_1.txt "https://zenodo.org/api/records/11178509/files/train_part1.zip/content" > train_part1.zip
    ```
    ``` bash
    curl --cookie zenodo-cookies_1.txt "https://zenodo.org/api/records/11180795/files/calib_matrix.csv/content"
    ``` 

* Download Train data (Part 2)

    ``` bash
    curl --cookie-jar zenodo-cookies_2.txt "https://zenodo.org/record/11180795?token=<YOUR-TOKEN-HERE>"
    ```
    ``` bash
    curl --cookie zenodo-cookies_2.txt "https://zenodo.org/api/records/11180795"
    ```
    ``` bash
    curl --cookie zenodo-cookies_2.txt "https://zenodo.org/api/records/11180795/files/train_part2.zip/content" > train_part2.zip
    ``` 

* Download Train data (Part 3)

    ``` bash
    curl "https://zenodo.org/records/11355500/files/landmark.zip?download=1" > landmark.zip
    ```  -->
#### 2. Navigate to the root directory.
```
cd tus-rec-challenge_baseline
```

#### 3. Create directories.
```
mkdir data
cd data
mkdir frames_transfs
mkdir landmarks
```

#### 4. Download data from [Train Dataset (Part 1,](https://zenodo.org/doi/10.5281/zenodo.11178508) [Part 2,](https://zenodo.org/doi/10.5281/zenodo.11180794) [Part 3)](https://zenodo.org/doi/10.5281/zenodo.11355499).
   Put `train_part1.zip`, `train_part2.zip`, `landmark.zip`, and `calib_matrix.csv` into `./data` directory.

#### 5. Unzip.
Unzip `train_part1.zip` and `train_part2.zip` into `./data/frames_transfs` directory, and unzip `landmark.zip` into `./data/landmarks` directory.

```
unzip \train_part\*.zip -d frames_transfs
unzip \landmark.zip -d landmarks
```

#### 6. Make sure the folder structure is the same as follows.
```bash
├── data/ # Contains used data set 
│ ├── frames_transfs/ # Unzipped from `train_part1.zip` and `train_part2.zip`, including 50 folders
│  ├── 000/ # Contains 24 scans in subject 000
│    ├── RH_Per_L_DtP.h5 # Scan info including frames and transformations 
│    ├── ...
│  ├── ...
│  ├── 049/ # Contains 24 scans in subject 049
│ ├── landmarks/ # Unzipped from `landmark.zip`
│  ├── landmark_000.h5 # Landmark coordinates in subject 000
│  ├── ...
│  ├── landmark_049.h5 # Landmark coordinates in subject 049
├── train.py # Training script 
├── generate_DDF.py # Generate 4 kinds of DDF 
├── plot.py # An example of load data and plot scan
├── utils/ # Utility functions 
│ ├── transform.py # Transformation functions
│ ├── loader.py # Data loader
│ ├── loss.py # Loss function and metrics
│ ├── network.py # Networks
│ ├── funs.py # Functions used during training
│ ├── data_process_functions.py # Functions used during data processing
│ ├── rigid_transform_3D.py # Adapted transformation function
│ ├── evaluation.py # functions used during DDF generation
├── options/ # Model hyperparameter 
│ ├── base_options.py # Hyperparameters for dataroot and GPU
│ └── train_options.py # Hyperparameters for training and testing
├── requirements.txt # packages for environment installation
```

#### 7. Train model.
``` bash
python3 train.py
```
#### 8. Generate DDF.
``` bash
python3 generate_DDF.py
```




