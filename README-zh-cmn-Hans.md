# Conv3D CLSTM

> [English][en] | [简体中文][hans]

## 依赖

+ TensorFlow 1.0.0+
+ Keras 2.0.2+

详见 [requirements.txt][req].

## 配置

### Linux / macOS

克隆本仓库
```shell
git clone https://github.com/oopsno/Conv3D_CLSTM.git
cd Conv3D_CLSTM
```

[可选] 使用 `virtualenv` 创建并激活一个虚拟环境
```shell
virtualenv Conv3D_CLSTM
source bin/activate
```

安装依赖项
```shell
pip install -r requirements.txt
```

### Windows

#### 默认发行版

> 如果在安装 Scipy 时遭遇问题，请尝试使用 Anaconda 发行版。

克隆本仓库
```cmd
git clone https://github.com/oopsno/Conv3D_CLSTM.git
cd Conv3D_CLSTM
```

[可选] 使用 `virtualenv` 创建并激活一个虚拟环境
```cmd
virtualenv Conv3D_CLSTM
cd Script
activate
cd ..
```

安装依赖项
```cmd
pip install -r requirements.txt
```

#### Anaconda 发行版

克隆本仓库
```cmd
git clone https://github.com/oopsno/Conv3D_CLSTM.git
cd Conv4D_CLSTM
```

[可选] 使用 `conda` 创建并激活一个虚拟环境
```cmd
conda create -n Conv3D-CLSTM
activate Conv3D-CLSTM
```

安装依赖项
```
pip install -r requirements.txt
```

## 训练

以在 IsoGR 数据集上使用 RGB 数据训练为例

1. 修改 `configurations/isogr_rgb.yaml`
2. `python training_isogr.py configurations/isogr_rgb.yaml`

[en]:   https://github.com/oopsno/Conv3D_CLSTM/blob/keras/README.md
[req]:  https://github.com/oopsno/Conv3D_CLSTM/blob/keras/requirements.txt
[hans]: https://github.com/oopsno/Conv3D_CLSTM/blob/keras/README-zh-cmn-hans.md
