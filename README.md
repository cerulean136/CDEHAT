<p align="center">
    <img src="assets/logo.png" width="500">
</p>

## CDEHAT: Conditional Diffusion-Assisted Enhanced Hybrid Attention Transformer for Remote Sensing Imagery Super-Resolution

[Paper]() | [Project Page]()

## <a name="update"></a>:dna:Network
<p align="center">
    <img src="assets/CDEHAT.png"  width="700">
</p>

## :book:Table Of Contents

- [Visual Results Display](#visual_results)
- [Installation](#installation)
- [Quick Start](#quick_start)
- [Inference](#inference)
- [Train](#train)


## <a name="visual_results"></a>:eyes:Visual Results Display

## <a name="update"></a>:new:Update
Our paper was accepted on **March 3, 2026**.

The publicly available **code** and newly created **dataset CA-2022-S2-NAIP** are being compiled and uploaded... 

expected to be completed within two business days: **March 3 and March 4, 2026**

**Finally, we thank the editors and reviewers for their work on this paper.**

- **2026.03.03**: The paper is accepted.
- **2025.12.02**: This repo is released.

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

Download demo_v1.0.0 [temp.pth](https://github.com/cerulean136/CDEHAT/releases/download/demo_v1.0.0/temp.pth) to `weights/`, then run the following command to begin the model inference demonstration.
```shell
python -m super_resolution.test -opt super_resolution/options/test/test_Real_CDEHAT_GAN_SRx4_trained_on_AID.yml
```
Alternatively, you can run our script `begin_test_in_run_window.py` to quickly perform model inference.

## <a name="train"></a>:stars:Train

### Stage 1

First, we train a SwinIR, which will be used for degradation removal during the training of stage 2.

<a name="gen_file_list"></a>
1. Generate file list of training set and validation set, a file list looks like:

    ```txt
    /path/to/image_1
    /path/to/image_2
    /path/to/image_3
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

2. Fill in the [training configuration file](configs/train/train_stage1.yaml) with appropriate values.

3. Start training!

    ```shell
    accelerate launch train_stage1.py --config configs/train/train_stage1.yaml
    ```

### Stage 2

1. Download pretrained [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) to provide generative capabilities. :bulb:: If you have ran the [inference script](inference.py), the SD v2.1 checkpoint can be found in [weights](weights).

    ```shell
    wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt --no-check-certificate
    ```

2. Generate file list as mentioned [above](#gen_file_list). Currently, the training script of stage 2 doesn't support validation set, so you only need to create training file list.

3. Fill in the [training configuration file](configs/train/train_stage2.yaml) with appropriate values.

4. Start training!

    ```shell
    accelerate launch train_stage2.py --config configs/train/train_stage2.yaml
    ```







You can find our **CDEHAT model architecture and training framework** in the **CDEHAT/SR** folder; you can also compare the **local attribution map* of different models in **CDEHAT/LAM**. (The README document for this project is still being further refined and improved.)

Our **CA-2022-S2-NAIP** is currently being uploaded, and the process is expected to be completed by March 5, 2026.



Please download the weights we **released** and place them in your local folder **CDEHAT/SR/weights**. Then, navigate to the configuration file **CDEHAT/SR/super_resolution/options/test/test_Real_CDEHAT_GAN_SRx4_trained_on_AID.yml** and ensure the weight path is correct: **pretrain_network_g: .\weights\temp.pth**. Afterward, you can run **CDEHAT/SR/begin_test_in_run_window.py** to obtain the SR results for the demo image **CDEHAT/SR/demo/demo.jpg**. You can also reconstruct **your own remote sensing data** by changing the data folder path in the configuration file.

## Citation

Please cite us if our work is useful for your research.

```
@article{
}
```
