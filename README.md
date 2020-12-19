# Defect Detection and Elliptical Object Localization with DenseNet-169 on subset of DAGM 2007 Defect Dataset

## Task:

Detection of anomalies / disorders of images with convolutional neural networks and detection with an ellipse bounding
boxes. The different and difficult part from normal detection methods is the creation of an elliptical bounding box.
Because it has a parametric and high level function, it is more difficult for the model to predict the elliptical region
than box detection algorithms. In addition to the transfer learning method, both dense and convolutional detection
blocks are placed at the end of the feature extracting blocks. For this reason, in addition to the normal model, the
ellipse shape is estimated by using the regional densities and center point parameters. Instead of the region proposal
network model, which is a more complex structure, direct object localization has been tried and tested whether it can be
detected with a normal feature extractor. Since data augmentation with the elliptic bounding box cannot be performed
with the algorithms used in traditional object detection methods, there is not any data augmentation algorithm is used
on shape. However, some spatial (brightness / contrast / gamma etc.) augmentation algorithms were used to improve
feature extractor performance and reduce model over-fitting.

---

## Project Information:

- Dataset: [**DAGM 20007**](https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html) _(Class 1, Class 2 and Class 3 used)_
    - In the dataset, 150 normal and 150 abnormal images were selected for each class. 
      There are 3 classes in total. 20% of the images were randomly selected for testing.
    - 900 images in total, 720 for training, 180 for testing.
    - All data distributions are equal.


- Used model: **DenseNet-169**
    - Although different state-of-the-art models such as ResNet50 and EfficientNet-B1 have been tried, the best model
      has been DenseNet-169 in terms of parameter number and model success.
    - Transfer learning with custom top structure.


- Environment:
    - `Python 3.8.5` used on `Ubuntu 20.04.01 LTS`.
    - Conda (`Miniconda v4.9.2`) package manager used
    - Required packages for conda and pip are given in
      `environment.yml` and `requirements.txt` respectively.
      

---

## Project Hierarchy

Project folder structure is shown below:

    defect-detection
    ---| train.py
    ---| inference.py
    ---| environment.yml        [Conda environment configurations]
    ---| requirements.txt       [Pip dependencies]
    ---| model.png              [Output of inference
    ---| README.md              [Documentation]
    ---\ models
       ---| __init__.py
       ---| defect_model.py
    ---\ utils
       ---| __init__.py
       ---| losses.py
    ---\ data
       ---| __init__.py
       ---| csv_preprocess.py
       ---| dataset.py
       ---| test_raw_bbox.py
       ---| Class1_def.txt
       ---| Class2_def.txt
       ---| Class3_def.txt
       ---\ csv
          ---| train.csv        [Output of csv_preprocess.py]
          ---| test.csv         [Output of csv_preprocess.py]
       ---\ raw
          ---\ Class1_def
             ---| 001_1.png     [Image_no]_[Class_id].png
             ---| ...
          ---\ Class1_norm
             ---| 001_1.png     [Image_no]_[Class_id].png
             ---| ...
          ---| ...
       ---| test_data
          ---| class1_def_1.png
          ---| class1_def_2.png
          ---| class1_norm_1.png
          ---| class1_norm_2.png
          ---| ...
       ---\ logs
          ---| result.png       [Output of inference]
          ---\ [date-time]      [Output of training]
             ---| checkpoint
             ---| ...
             ---\ tensorboard

---

## Usage

### How to train with CLI

* Train model 100 epoch from scratch. Set learning rate to 0.01


        $ python3 train.py --save-checkpoint --epoch 100 --learning-rate 1e-2


* Use checkpoint to continue training

    
   
        $ python3 train.py --load-model logs/[checkpoint-dir]/checkpoint.ckpt --save-checkpoint --epoch 30


### How to inference with CLI

* Inference on test images


        $python3 train.py --inference --load-model logs/[checkpoint-dir]/checkpoint.ckpt --test-image data/test_data/class1_def_2.png --save-result


### Other parameters

- `--raw-data` : Path to raw data, default is `data/raw`
- `--txt-data` : Path to txt data, default is `data`
- `--csv-data` : Path to csv data, default is `data/csv`
- `--log-dir` : Path to log directory, default is `logs`
- `--load-model` : Path to checkpoint file to load model, trains from scratch if it's not specified
- `--learning-rate` : Set learning rate, default is `1e-3`
- `--epoch` : Set epoch for training, default is `20`
- `--batch-size` : Set batch size for data pipeline, default is `64`
- `--save-checkpoint` : Saves checkpoint to `--log` directory with date-time
- `--inference` : Activates inference mode, `--load-model` param is required for that
- `--real-time` : Not added yet
- `--test-image` : Path to images on disk, multiple paths allowed
- `--save-result` : Saves inference result with bounding box added image to `--logs` directory


---

Contact for permissions, [E-mail](mailto:omerf.sarioglu@gmail.com), [LinkedIn](https://www.linkedin.com/in/omerfsarioglu/)

Readme will be updated...
