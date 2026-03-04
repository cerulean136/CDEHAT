from utils.operations import execute_command

# TODO: Use module paths instead of file system paths to support relative imports.

# CDEHAT
test_CDEHAT_MSE_SRx4_trained_on_AID = \
    "python -m super_resolution.test -opt super_resolution/options/test/test_CDEHAT_MSE_SRx4_trained_on_AID.yml"

# CDEHAT-PD
test_CDEHAT_GAN_SRx4_trained_on_AID = \
    "python -m super_resolution.test -opt super_resolution/options/test/test_CDEHAT_GAN_SRx4_trained_on_AID.yml"

# CDEHAT
test_CDEHAT_MSE_SRx4_trained_on_CA2022S2NAIP = \
    "python -m super_resolution.test -opt super_resolution/options/test/test_CDEHAT_MSE_SRx4_trained_on_CA2022S2NAIP.yml"

# Real CDEHAT
test_Real_CDEHAT_MSE_SRx4_trained_on_AID = \
    "python -m super_resolution.test -opt super_resolution/options/test/test_Real_CDEHAT_MSE_SRx4_trained_on_AID.yml"

# Real CDEHAT-PD
test_Real_CDEHAT_GAN_SRx4_trained_on_AID = \
    "python -m super_resolution.test -opt super_resolution/options/test/test_Real_CDEHAT_GAN_SRx4_trained_on_AID.yml"

# Demo
test_demo = \
    "python -m super_resolution.test -opt super_resolution/options/test/demo.yml"

if __name__ == "__main__":
    execute_command(test_demo)

