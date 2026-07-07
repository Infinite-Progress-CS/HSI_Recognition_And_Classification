# CACL

This repository provides the code for the methods and experiments presented in our paper 'Class-Aware Consistency Learning for Open-Set Semi-Supervised Hyperspectral Image Classification'.

## Environment

Experiment on GTX3090, CUDA Version:12.0

Create conda environment

```
conda create -n cacl python=3.8.18
```

Install the required Python environment using `requirement.txt`:

```
pip install -r requirement.txt
```

## Dataset Splitting

Run the following command to split the dataset:

```
python trainTestSplit.py
```

## Training

Train the model with the following command:

```
python train.py --cuda 0 --dataset Indian --patchsize 13
```

* `--cuda 0`: This parameter is used to specify the **GPU ID** for running model training.
* `--dataset Indian`: This parameter is used to specify the **dataset** used for training. Available datasets include `Indian`, `paviaU`, and `salinas`.
* `--patchsize 13`: This parameter is used to specify the **patch size** of the input data.

For setting more parameters, please refer to the argparse configuration in the `train.py` file.
