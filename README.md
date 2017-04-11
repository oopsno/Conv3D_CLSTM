# Conv3D CLSTM

> [English][en] | [简体中文][hans]

## Prerequisites

+ TensorFlow 1.0.0+
+ Keras 2.0.2+

See [requirements.txt][req].

## Setup

### Linux / macOS

```shell
# Clone this repo
git clone --recursive https://github.com/oopsno/Conv3D_CLSTM.git
cd Conv3D_CLSTM

# [optional] Create a virtual environment and activate it
virtualenv Conv3D_CLSTM
source bin/activate

# install requirements
pip install -r requirements.txt
```

### Windows

#### The hard way

> May get stuck while installing scipy.
> If so, try Anaconda.

```cmd
:: Clone this repo
git clone --recursive https://github.com/oopsno/Conv3D_CLSTM.git
cd Conv3D_CLSTM

:: [optional] Create a virtual environment and activate it
virtualenv Conv3D_CLSTM
cd Script
activate
cd ..

:: install requirements
pip install -r requirements.txt
```

#### Using The Anaconda Distribution

```cmd
:: Clone this repo
git clone --recursive https://github.com/oopsno/Conv3D_CLSTM.git
cd Conv3D_CLSTM

:: [optional] Create a virtual environment and activate it
conda create -n Conv3D_CLSTM
activate Conv3D_CLSTM

:: install requirements
pip install -r requirements.txt
```

## Training

Say, if you wanna train an RGB-based model on IsoGR

1. modify `configurations/isogr_rgb.yaml`
2. `python training_isogr.py configurations/isogr_rgb.yaml`

[en]:   https://github.com/oopsno/Conv3D_CLSTM/blob/keras/README.md
[req]:  https://github.com/oopsno/Conv3D_CLSTM/blob/keras/requirements.txt
[hans]: https://github.com/oopsno/Conv3D_CLSTM/blob/keras/README-zh-cmn-Hans.md
