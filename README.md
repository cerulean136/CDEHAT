<p align="center">
    <img src="assets/logo.png" width="500">
</p>

## CDEHAT: Conditional Diffusion-Assisted Enhanced Hybrid Attention Transformer for Remote Sensing Imagery Super-Resolution

[Paper]() | [Project Page]()

## :book:Table Of Contents

- [Visual Results Display](#visual_results)
- [Installation](#installation)
- [Quick Start (gradio demo)](#quick_start)
- [Inference](#inference)
- [Train](#train)


## <a name="visual_results"></a>:eyes:Visual Results Display

## <a name="update"></a>:new:Update

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

Download [demo_v1.0.0.pth](https://github.com/cerulean136/CDEHAT/releases/download/demo_v1.0.0/temp.pth) to `weights/`, then run the following command to begin the reasoning demonstration.






Our paper was accepted on **March 3, 2026**.

The publicly available **code** and newly created **dataset CA-2022-S2-NAIP** are being compiled and uploaded... 

expected to be completed within two business days: **March 3 and March 4, 2026**

**Finally, we thank the editors and reviewers for their work on this paper.**

<p align="center">
    <img src="assets/CDEHAT.png" style="border-radius: 15px">
</p>



You can find our **CDEHAT model architecture and training framework** in the **CDEHAT/SR** folder; you can also compare the **local attribution map* of different models in **CDEHAT/LAM**. (The README document for this project is still being further refined and improved.)

Our **CA-2022-S2-NAIP** is currently being uploaded, and the process is expected to be completed by March 5, 2026.



Please download the weights we **released** and place them in your local folder **CDEHAT/SR/weights**. Then, navigate to the configuration file **CDEHAT/SR/super_resolution/options/test/test_Real_CDEHAT_GAN_SRx4_trained_on_AID.yml** and ensure the weight path is correct: **pretrain_network_g: .\weights\temp.pth**. Afterward, you can run **CDEHAT/SR/begin_test_in_run_window.py** to obtain the SR results for the demo image **CDEHAT/SR/demo/demo.jpg**. You can also reconstruct **your own remote sensing data** by changing the data folder path in the configuration file.
