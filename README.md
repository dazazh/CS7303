# ClearGrasp: 3D Shape Estimation of Transparent Objects for Manipulation



## Installation

This code is tested with Ubuntu 16.04, Python3.6 and [Pytorch](https://pytorch.org/get-started/locally/) 1.3, and CUDA 9.0.  

学校服务器的配置能用，python3.8实测也能用

### System Dependencies

需要sudo权限，联系师兄在服务器上装一下，具体方法如下

```bash
sudo apt-get install libhdf5-10 libhdf5-serial-dev libhdf5-dev libhdf5-cpp-11
sudo apt install libopenexr-dev zlib1g-dev openexr
sudo apt install xorg-dev  # display widows
sudo apt install libglfw3-dev
```




## Setup

1. Clone the repository. A small sample dataset of 3 real and 3 synthetic images is included.

   ```bash
   git clone git@github.com:Shreeyak/cleargrasp.git
   ```

2. Install pip dependencies by running in terminal:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the data:  
   a) 数据集：zzy上传到服务器，存储路径：

   <img src="/data/readme_images/data_path.png" style="zoom:50%;" />

   b)  [Model Checkpoints](https://storage.googleapis.com/cleargrasp/cleargrasp-checkpoints.zip) (0.9GB) - (包含了masks、boundary和surface normal原模型，前两个我们可以直接用) checkpoints位置在三个模型文件夹下的config目录中设置

   <img src="/data/readme_images/model.png" style="zoom:50%;" />

4. Compile depth2depth (global optimization):

   `depth2depth` is a C++ global optimization module used for depth completion, adapted from the [DeepCompletion](http://deepcompletion.cs.princeton.edu/) project. It resides in the `api/depth2depth/` directory.

   - To compile the depth2depth binary, you will first need to identify the path to libhdf5. Run the following command in terminal:

     ```bash
     find /usr -iname "*hdf5.h*"
     ```

     Note the location of `hdf5/serial`. It will look similar to: `/usr/include/hdf5/serial/hdf5.h`.

   - Edit BOTH lines 28-29 of the makefile at `api/depth2depth/gaps/apps/depth2depth/Makefile` to add the path you just found as shown below:

     ```bash
     USER_LIBS=-L/usr/include/hdf5/serial/ -lhdf5_serial
     USER_CFLAGS=-DRN_USE_CSPARSE "/usr/include/hdf5/serial/"
     ```

   - Compile the binary:

     ```bash
     cd api/depth2depth/gaps
     export CPATH="/usr/include/hdf5/serial/"  # Ensure this path is same as read from output of `find /usr -iname "*hdf5.h*"`
     
     make
     ```

     This should create an executable, `api/depth2depth/gaps/bin/x86_64/depth2depth`. The config files will need the path to this executable to run our depth estimation pipeline.

   - Check the executable, by passing in the provided sample files:

     ```bash
     cd api/depth2depth/gaps
     bash depth2depth.sh
     ```

     This will generate `gaps/sample_files/output-depth.png`, which should match the `expected-output-depth.png` sample file. It will also generate RGB visualizations of all the intermediate files.

## To run the code:



### 1. ClearGrasp Quick Demo - Evaluation of Depth Completion of Transparent Objects

原项目自带的evaluation模块，是对整个深度预测的evaluation，而surface_normal的eval在`/pytorch_networks/surface_normals/eval.py`，需要等Train结果出来再调。

We provide a script to run our full pipeline on a dataset and calculate accuracy metrics (**RMSE, MAE, etc**). Resides in the directory `eval_depth_completion/`.  **对应论文的测试指标，看描述他有直接的实现**

- Install dependencies and follow [Setup](#setup) to download our model checkpoints and compile `depth2depth`.

- Create a local copy of the config file:

  新建配置文件，原来那个是个样本文件，每次需要新的配置的时候复制原来的样本修改即可

  ```bash
  cd eval_depth_completion/
  cp config/config.yaml.sample config/config.yaml
  ```

- Edit the `config/config.yaml` file to set `pathWeightsFile` parameters to the paths of the respective model checkpoints. To run evaluation on the different datasets, set the path(s) to their director(ies) within the `files` parameter.

  设置checkpoint路径

- Run ClearGrasp on the sample dataset:

  ```bash
  python eval_depth_completion.py -c config/config.yaml
  ```

  

### 2. Training Code

The folder `pytorch_networks/` contains the code used to train the
surface normals, occlusion boundary and semantic segmentation models.

- Go the to respective folder (eg: `pytorch_networks/surface_normals`) and create a local copy of the config file:

  ```bash
  cp config/config.yaml.sample config/config.yaml
  ```

- Edit the `config.yaml` file to fill in the paths to the dataset, select hyperparameter values, etc. All the parameters are explained in comments within the config file.

- Start training: 

  ```bash
  python train.py -c config/config.yaml
  ```

- Eval script can be run by: 

  ```bash
  python eval.py -c config/config.yaml
  ```



### 3. 可能会报的错误

- 路径问题，修改`surface_normal/dataloader.py`为个人路径
- 显存/显卡问题，修改device为空闲设备`cuda:0,1,2...`
- 训练validation阶段报错，联系我
- 可能会有一个numpy版本的错，问下gpt改一下，没记错的话是`bool->bool_`