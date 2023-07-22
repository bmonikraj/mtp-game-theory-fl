# Local-FedGT

Installation of Tensorflow GPU on Windows WSL - https://www.tensorflow.org/install/pip#windows-wsl2_1

Create soft link in current directory: `ln -s /mnt/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.6/nvvm/ nvvm`

Conda activate env: `conda activate <env>`

Installation:

- `sudo apt-get install libxcb-xinerama0`
- `conda install -c conda-forge cudatoolkit=11.8.0`
- `conda install -c nvidia cuda-nvcc`
- `pip install tensorflow==2.12.*`
- `pip install pandas`
- `conda install -c anaconda seaborn`
- `conda install -c conda-forge matplotlib`
- `pip install opencv-python-headless`
- `conda install -c anaconda scipy`


Run program: `python local-fedgt.py <number of clients> <game theory test data split> <game theory utility threshold>`
