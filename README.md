# Trackerless 3D Freehand Ultrasound Reconstruction Challenge 2024 (TUS-REC2024)
<!-- ## About -->

[Website](https://github-pages.ucl.ac.uk/tus-rec-challenge/TUS-REC2024/) |
[Train Dataset (Part 1,](https://zenodo.org/doi/10.5281/zenodo.11178508) 
[Part 2,](https://zenodo.org/doi/10.5281/zenodo.11180794)
[Part 3)](https://zenodo.org/doi/10.5281/zenodo.11355499) |
[Training Code Usage Instruction](#training-code) |
[Validation Dataset](https://zenodo.org/doi/10.5281/zenodo.12979481) |
[Submission Requirement and Example Docker](/submission/README.md) |
[Data Usage Policy](#data-usage-policy)

<div align=center>
  <a target="_blank"><img style="padding: 10px;" src="img/logo.png" width=200px></a>
</div >

**Welcome to the Trackerless 3D Freehand Ultrasound Reconstruction Challenge 2024 (TUS-REC2024)！**   

The TUS-REC2024 Challenge is a part of the 27th International Conference on Medical Image Computing and Computer Assisted Intervention ([MICCAI 2024](https://conferences.miccai.org/2024)), held in conjunction with the 5th ASMUS workshop, from October 6th to 10th 2024 in Marrakesh, Morocco. The challenge is supported by the MICCAI Special Interest Group in Medical Ultrasound ([SIGMUS](https://miccai.org/index.php/special-interest-groups/sig/)) and will be presented at its international workshop [ASMUS 2024](https://miccai-ultrasound.github.io/#/asmus24). 

## Background
Reconstructing 2D Ultrasound (US) images into a 3D volume enables 3D representations of anatomy to be generated which are beneficial to a wide range of downstream tasks such as quantitative biometric measurement, multimodal registration, 3D visualisation and interventional guidance. Although substantive progress has been made recently through non-deep-learning- and deep-learning-based approaches, this application is still challenging due to 1) inherent accumulated error - frame-to-frame transformation error will be accumulated through time when reconstructing long sequence of US frames, and 2) a lack of publicly-accessible data with synchronised spatial location, often obtained from tracking devices, for benchmarking the performance and for training learning-based methods. The TUS-REC challenge aims to provide a benchmark for freehand US reconstruction with publicly available in vivo US data from forearms of one hundred volunteers, using multiple predefined scanning protocols, targeted at improving the reconstruction performance in this challenging task. The outcome of the challenge includes 1) open-sourcing the first largest tracked US datasets with accurate positional information; 2) establishing one of the first benchmarks for 3D US reconstruction, suitable for modern learning-based data-driven approaches.

## Task Description
The aim of this task is to reconstruct 2D US images into a 3D volume. The algorithm is expected to take the entire scan as input and output two different sets of transformation-representing displacement vectors as results, a set of displacement vectors on individual pixels and a set of displacement vectors on provided landmarks. There is no requirement on how the algorithm is designed internally, for example, whether it is learning-based method; frame-, sequence- or scan-based processing; or, rigid-, affine- or nonrigid transformation assumptions. 

Participant teams are expected to make use of the sequential data and potentially make knowledge transfer from other domains such as computer vision and computer-assisted intervention. The participant teams are expected to take US scan as input and output two sets of pixel displacement vectors, indicating the transformation to reference frame, i.e., first frame in this task. The evaluation process will take the generated displacement vectors from their dockerized models, and produce the final accuracy score to represent the reconstruction performance, at local and global levels, representing different clinical application of the reconstruction methods.

For details information, please see [Assessment Criteria](https://github-pages.ucl.ac.uk/tus-rec-challenge/TUS-REC2024/assessment.html), and also [generate_DDF.py](https://github.com/QiLi111/tus-rec-challenge_baseline/blob/main/generate_DDF.py) for an example of generating required 4 kinds of displacement vectors.

## Dataset

Acquisition devices and config: The 2D US images were acquired using an Ultrasonix machine (BK, Europe) with a curvilinear probe (4DC7-3/40). The associated position information of each frame was recorded by an optical tracker (NDI Polaris Vicra, Northern Digital Inc., Canada). The acquired US frames were recorded at 20 fps, with an image size of 480×640, without speckle reduction. The frequency was set at 6MHz with a dynamic range of 83 dB, an overall gain of 48% and a depth of 9 cm. 

Scanning protocol: Both left and right forearms of volunteers were scanned. For each forearm, the US probe moves in three different trajectories (straight line shape, "C" shape, and "S" shape), in a distal-to-proximal direction followed by a proximal-to-distal direction, with the US plane perpendicular of and parallel to the scanning direction, as illustrated in the figure below. The train dataset contains 1200 scans in total, 24 scans associated with each subject.

![image info](img/scan_traj.png)

### Dataset structure: 

* The training data contains 50 folders (one subject per folder), each with 24 scans. Each .h5 file corresponds to one scan, storing image and transformation of each frame within this scan. Key-value pairs in each .h5 file are explained below.

    * `frames`  - All frames in the scan; with a shape of [N,H,W], where N refers to the number of frames in the scan, H and W denote the height and width of a frame. 

    * `tforms` - All transformations in the scan; with a shape of [N,4,4], where N is the number of frames in the scan, and the transformation matrix denotes the transformation from tracker tool space to camera space. 

    * Notations in the name of each .h5 file: “RH”: right arm; “LH”: “left arm”; “Per”: perpendicular; “Par”: parallel; “L”: straight line shape; “C”: C shape; “S”: S shape; “DtP”: distal-to-proximal direction; “PtD”: proximal-to-distal direction; For example, “RH_Per_L_DtP.h5” denotes a scan on the right forearm, with ultrasound probe perpendicular of the forearm sweeping along straight line, in distal-to-proximal direction.

* Calibration matrix: The calibration matrix was obtained using a pinhead-based method. The `scaling_from_pixel_to_mm` and `spatial_calibration_from_image_coordinate_system_to_tracking_tool_coordinate_system` are provided in the “calib_matrix.csv”, where `scaling_from_pixel_to_mm` is the scale between image coordinate system (in pixel) and image coordinate system (in mm), and `spatial_calibration_from_image_coordinate_system_to_tracking_tool_coordinate_system` is the rigid transformation between image coordinate system (in mm) to tracking tool coordinate system. Please refer to an example that this calibration matrix is read and used in the baseline code [here](https://github.com/QiLi111/tus-rec-challenge_baseline/blob/0b7eb6c9c7df941f9425dd0d558ab285329b4ae6/train.py#L65).

## Training Code

### Instruction
This repository provides an example framework for freehand US pose regression, including usage of various types of predictions and labels (see [transformation.py](https://github.com/QiLi111/tus-rec-challenge_baseline/blob/main/utils/transform.py)). Please note that the networks used here are small and simplified for demonstration purposes.

For instance, the network can predict the transformation between two US frames as 6 DOF "parameter". If the label type is "point", the loss is calculated as the point distance (by transforming "parameter" to "point" using function [parameter_to_point](https://github.com/QiLi111/tus-rec-challenge_baseline/blob/a7a235044d9fa0275470024df25db099c7e65522/utils/transform.py#L267)). The steps below illustrate an example of training a pose regression model and generate 4 kinds of displacements. 

<!-- We use the transformation from image coordinate system (in mm) to image coordinate system (in mm), for example described in function [to_transform_t2t](https://github.com/QiLi111/tus-rec-challenge_baseline/blob/2cdc92c003af8d985a50f27ea97900ba35da5c98/utils/transform.py#L93).  -->

<!-- The model trained with labels defined above is independent of the rigid part in calibration matrix, and only dependent of the scaling. That is to say, the trained model is independent of the relative position between the tracker tool and the probe, and only dependent of the configuration of the probe.  -->

<!-- For more information about the algorithms, refer to [Prevost et al. 2018](https://doi.org/10.1016/j.media.2018.06.003) and [Li et al. 2023](https://doi.org/10.1109/TBME.2023.3325551). -->

### Steps to run the code
#### 1. Clone the repository.
```
git clone https://github.com/QiLi111/tus-rec-challenge_baseline.git
```

#### 2. Navigate to the root directory.
```
cd tus-rec-challenge_baseline
```

#### 3. Install conda environment

``` bash
conda create -n freehand-US python=3.9.13
conda activate freehand-US
pip install -r requirements.txt
conda install pytorch3d --no-deps -c pytorch3d
```

#### 4. Create directories.
```
mkdir -p data/frames_transfs
mkdir -p data/landmarks
```

#### 5. Download data from [Train Dataset (Part 1,](https://zenodo.org/doi/10.5281/zenodo.11178508) [Part 2,](https://zenodo.org/doi/10.5281/zenodo.11180794) [Part 3)](https://zenodo.org/doi/10.5281/zenodo.11355499).
   Put `train_part1.zip`, `train_part2.zip`, `landmark.zip`, and `calib_matrix.csv` into `./data` directory.

#### 6. Unzip.
Unzip `train_part1.zip` and `train_part2.zip` into `./data/frames_transfs` directory, and unzip `landmark.zip` into `./data/landmarks` directory.

```
unzip data/train_part\*.zip -d data/frames_transfs
unzip data/landmark.zip -d data/landmarks
```

#### 7. Make sure the data folder structure is the same as follows.
```bash
├── data/ # Contains data set 
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
```

#### 8. Train model. 
``` bash
python3 train.py
```
#### 9. Generate DDF.
``` bash
python3 generate_DDF.py
```

## Data Usage Policy
The training and validation data provided may be utilized within the research scope of this challenge and in subsequent research-related publications. However, commercial use of the training and validation data is prohibited. In cases where the intended use is ambiguous, participants accessing the data are requested to abstain from further distribution or use outside the scope of this challenge. Please refer to [this Section](https://github-pages.ucl.ac.uk/tus-rec-challenge/TUS-REC2024/policies.html) for detailed data usage policy.

After we publish the summary paper of the challenge, if you use our dataset in your publication, please cite the summary paper (reference will be provided once published) and some of the follwing articles: 
* Qi Li, Ziyi Shen, Qianye Yang, Dean C. Barratt, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Nonrigid Reconstruction of Freehand Ultrasound without a Tracker." In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 689-699. Cham: Springer Nature Switzerland, 2024. doi: [10.1007/978-3-031-72083-3_64](https://doi.org/10.1007/978-3-031-72083-3_64).
* Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Long-term Dependency for 3D Reconstruction of Freehand Ultrasound Without External Tracker." IEEE Transactions on Biomedical Engineering, vol. 71, no. 3, pp. 1033-1042, 2024. doi: [10.1109/TBME.2023.3325551](https://ieeexplore.ieee.org/abstract/document/10288201).
* Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Trackerless freehand ultrasound with sequence modelling and auxiliary transformation over past and future frames." In 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI), pp. 1-5. IEEE, 2023. doi: [10.1109/ISBI53787.2023.10230773](https://doi.org/10.1109/ISBI53787.2023.10230773).
* Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Privileged Anatomical and Protocol Discrimination in Trackerless 3D Ultrasound Reconstruction." In International Workshop on Advances in Simplifying Medical Ultrasound, pp. 142-151. Cham: Springer Nature Switzerland, 2023. doi: [https://doi.org/10.1007/978-3-031-44521-7_14](https://doi.org/10.1007/978-3-031-44521-7_14).