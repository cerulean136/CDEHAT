<p align="center">
    <img src="assets/logo.png" width="500">
</p>


## CDEHAT: Conditional Diffusion-Assisted Enhanced Hybrid Attention Transformer for Remote Sensing Imagery Super-Resolution

[Paper]() | [Project Page]()


## <a name="update"></a>:dna:Network
 <p align="center">
    <img src="assets/CDEHAT.png"  width="900">
</p>


## :book:Table Of Contents

- [Visual Results Display](#visual_results)
- [Installation](#installation)
- [Quick Start](#quick_start)
- [Inference](#inference)
- [Train](#train)


## <a name="update"></a>:new:Update
Our paper was accepted on **March 3, 2026**.

The publicly available **code** and newly created **dataset CA-2022-S2-NAIP** are being compiled and uploaded... 

expected to be completed within two business days: **March 3 and March 4, 2026**

**Finally, we thank the editors and reviewers for their work on this paper.**

<!--
- **2026.03.03**: The paper is accepted.
- **2025.12.02**: This repo is released.
-->


## <a name="installation"></a>:gear:Installation
```shell
# clone this repo
git clone https://github.com/cerulean136/CDEHAT.git
cd CDEHAT

# create an environment with python >= 3.9
conda create -n cdehat python=3.9
conda activate cdehat
pip install -r requirements.txt

# enter SR directory
cd SR
```


## <a name="quick_start"></a>:flight_departure:Quick Start

Download [temp.pth](https://github.com/cerulean136/CDEHAT/releases/download/demo_v1.0.0/temp.pth) to `weights/`, then run the following command to begin the model inference demonstration.
```shell
python -m super_resolution.test -opt super_resolution/options/test/test_Real_CDEHAT_GAN_SRx4_trained_on_AID.yml
```
Alternatively, you can run our script `begin_test_in_run_window.py` to quickly perform model inference.


## 🎁 Dataset
Please download the following remote sensing benchmarks:
| Data Type | [AID](https://captain-whu.github.io/AID/) | [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) | [DIOR](https://www.sciencedirect.com/science/article/pii/S0924271619302825) | [UC Merced Land Use](https://vision.ucmerced.edu/datasets/) | [WHU-RS19](https://captain-whu.github.io/BED4RS/) | [CA-2022-S2-NAIP]()|
| :----: | :-----: | :----: | :----: | :----: |:----:|:----:|
|Training | [Download](https://captain-whu.github.io/AID/) | None | None | None | None | [Download]() |
|Testing | [Download](https://captain-whu.github.io/AID/) | [Download](https://captain-whu.github.io/DOTA/dataset.html) | [Download](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC) | [Download](https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset) | [Download](https://www.kaggle.com/datasets/sunray2333/whurs191) | [Download]() | 


## <a name="inference"></a>:crossed_swords:Inference

Refer to `./super_resolution/options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  

Then run the following codes (taking `temp.pth` as an example):

```shell
cd SR
python -m super_resolution.test -opt -opt super_resolution/options/test/test_Real_CDEHAT_GAN_SRx4_trained_on_AID.yml
```

Alternatively, you can run our script `begin_test_in_run_window.py` to quickly perform model inference.

The testing results will be saved in the `./results` folder.  

Please note that the test configuration file parameters `dataroot_gt` and `dataroot_lq` are used individually. When using `dataroot_gt`, the ground truth (GT) images for the test set are automatically generated into lq images by the data processing flow during the test. When using `dataroot_lq`, the test images are the lq images for the test set.



## <a name="train"></a>:stars:Train

### Stage 1

First, we train a EncoderHR, which will be used for guid content and degradation during the training of stage 2.

<a name="gen_file_list"></a>
1. Generate file list of training set and validation set, a file list looks like:

    ```txt
    /train/
        category 1/image_1
        category 1/image_2
        ...
        category 2/image_1
        ...
    /val/
        category 1/image_3
        category 1/image_4
        ...
        category 2/image_2
        ...
    ...
    ```

    You can write a simple python script or directly use shell command to produce file lists. Here is an example:
    
    ```shell
    # collect all iamge files in img_dir
    find [img_dir] -type f > files.list
    # shuffle collected files
    shuf files.list > files_shuf.list
    # pick train_size files in the front as training set
    head -n [train_size] files_shuf.list > files_shuf_train.list
    # pick remaining files as validation set
    tail -n +[train_size + 1] files_shuf.list > files_shuf_val.list
    ```

2. Fill in the [training configuration file](SR/super_resolution/options/train/train_CDEHAT_MSE_SRx4_trained_on_AID.yml) with appropriate values.

3. Start training!

    ```shell
    python -m super_resolution.train -opt super_resolution/options/train/train_CDEHAT_MSE_SRx4_trained_on_AID.yml
    ```

### Stage 2









You can find our **CDEHAT model architecture and training framework** in the **CDEHAT/SR** folder; you can also compare the **local attribution map* of different models in **CDEHAT/LAM**. (The README document for this project is still being further refined and improved.)

Our **CA-2022-S2-NAIP** is currently being uploaded, and the process is expected to be completed by March 5, 2026.



Please download the weights we **released** and place them in your local folder **CDEHAT/SR/weights**. Then, navigate to the configuration file **CDEHAT/SR/super_resolution/options/test/test_Real_CDEHAT_GAN_SRx4_trained_on_AID.yml** and ensure the weight path is correct: **pretrain_network_g: .\weights\temp.pth**. Afterward, you can run **CDEHAT/SR/begin_test_in_run_window.py** to obtain the SR results for the demo image **CDEHAT/SR/demo/demo.jpg**. You can also reconstruct **your own remote sensing data** by changing the data folder path in the configuration file.


## <a name="visual_results"></a>:eyes:Visual Results Display

### Visual on AID
 ![image](/assets/DOTA_image.png)

### Visual on CA-2022-S2-NAIP
![image](/assets/CA-2022-S2-NAIP_image.png)


## Acknowledgement

This project is based on [HAT](https://github.com/XPixelGroup/HAT), [DiffIR](https://github.com/Zj-BinXia/DiffIR) and [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks for their awesome work.


## Contact

If you have any questions, please feel free to contact with me at yangliao@zjut.edu.cn.


## Citation

Please cite us if our work is useful for your research.

```
@ARTICLE{,
  author={},
  journal={}, 
  title={}, 
  year={},
  volume={},
  number={},
  pages={},
  doi={}
}
```
