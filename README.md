# ML_CRAFT: Character Region Awareness for Text Detection

This github repository is a Pytorch reimplementation of CRAFT model for text detection in natural images.
Paper: [CRAFT](https://arxiv.org/abs/1904.01941)
Supplementary: [Video](https://www.youtube.com/watch?v=HI8MzpY8KMI&feature=youtu.be)
Authors: **[Youngmin Baek](mailto:youngmin.baek@navercorp.com), Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee.** from Clova AI Research, NAVER Corp.
About me: [Loi Nguyen Van](https://github.com/lloydnguyen96/) from DSLab, HUST.

## Description

This network consists of **backbone (VGG16D + Unet) + prediction branch** using two heatmaps for text localization.

### To-do list

* add more data augmentation methods
* split arbitrary dataset into training and validation set in a reproducible way
* use string to index a dataset example
* Multi-process Data Loading
* reproducible code
* train and validate model concurrently
* use multiple datasets concurrently (modify WeightedRandomSampler)
* use weakly-supervised and fully-supervised training mode concurrently
* train model for other languages: Chinese, ... (model is currently used for Latin characters)
* data preparation uses watershed algorithm vs postprocessing uses thresholding, why?
* check init functions
* check dataset mode switching
* check training, testing
* modify Resize in order to be suitable for batching multiple images
* visualize training process
* train model with weakly-supervised learning
* complete demo code (prediction.py)
* test model on several benchmark datasets

## Updates

**24 Oct 2020**: Release date

## Getting Started

### Installing

* Download SynthText dataset (~41 GB) for training: [SynthText](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)
* Clone this repository:
```
git clone https://github.com/lloydnguyen96/CRAFT
```

### Dependencies

* Linux environment (tested on Ubuntu 20.04.1 LTS)
* Conda environment (tested on Anaconda 4.8)
* CUDA 10.1
* Python 3.8.5
* Pytorch 1.6.0
* Torchvision 0.7.0
```
conda env create -f requirements.yaml
conda activate ML_CRAFT
pip install -r requirements.txt
```

### Executing program

* Change data_root parameter of SynthTextDataset class in main function in training.py module with your own root path to SynthText dataset directory (line 195)
* Change indices parameter of Subset class in main function in training.py module with your own list of indices (each index corresponds to one image in SynthText dataset) for training a subset of SynthText dataset (line 202)
* Change some configurations in project_config.py for your own customization (like batch size, image size, ...)
* Change number of epochs in main function in training.py module (line 227)
* For training:
```
python training.py
```
* For demo:
```
python prediction.py
```

## Help

If you have any questions, please add an issue

## Authors

* [Loi Nguyen Van](https://github.com/lloydnguyen96)

## Acknowledgments

This implementation is partially based on below implementations for model architecture:
* [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch)
