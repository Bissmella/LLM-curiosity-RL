# Curiosity based RL training for LLM agents



## Setup
### Create your conda environment

    conda create -n viper python==3.9.0
    conda activate viper

### Install Alfworld

    cd alfworld/TextWorld
    ```
    conda install cython
    conda install numpy
    pip install --no-build-isolation -e .[full]
    ```

    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

    cd ..

    pip install -e .[full]
### Install lamorel
    cd lamorel/lamorel
    pip install -e .
    pip install wandb gym peft bitsandbytes pyvirtualdisplay

### Download Alfworld episodes data and generate game files
    export ALFWORLD_DATA=<storage_path>
    alfworld-download
    alfworld-generate
    change data path in alfworld configs to your custom path

