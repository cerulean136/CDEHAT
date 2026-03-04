from utils.operations import execute_command

# TODO: Use module paths instead of file system paths to support relative imports.

# CDEHAT
train_CDEHAT_MSE_SRx4_trained_on_AID = \
    "python -m super_resolution.train -opt super_resolution/options/train/train_CDEHAT_MSE_SRx4_trained_on_AID.yml"

# CDEHAT-PD
train_CDEHAT_GAN_SRx4_trained_on_AID = \
    "python -m super_resolution.train -opt super_resolution/options/train/train_CDEHAT_GAN_SRx4_trained_on_AID.yml --debug"

# CDEHAT
train_CDEHAT_MSE_SRx4_trained_on_CA2022S2NAIP = \
    "python -m super_resolution.train -opt super_resolution/options/train/train_CDEHAT_MSE_SRx4_trained_on_CA2022S2NAIP.yml"

# Real CDEHAT
train_Real_CDEHAT_MSE_SRx4_trained_on_AID = \
    "python -m super_resolution.train -opt super_resolution/options/train/train_Real_CDEHAT_MSE_SRx4_trained_on_AID.yml"

# Real CDEHAT-PD
train_Real_CDEHAT_GAN_SRx4_trained_on_AID = \
    "python -m super_resolution.train -opt super_resolution/options/train/train_Real_CDEHAT_GAN_SRx4_trained_on_AID.yml"

if __name__ == "__main__":
    # Execute the command in a new process, print real-time output.
    execute_command(train_Real_CDEHAT_MSE_SRx4_trained_on_AID)
