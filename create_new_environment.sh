conda update conda --yes
conda create --name trackpyenv python==3.10  jupyter spyder conda pims numpy cython matplotlib pandas anaconda --yes
conda activate trackpyenv 
conda update conda --yes
conda install -c conda-forge trackpy ffmpeg av==9.2.0 tifffile  --yes
pip install moviepy numba seaborn scipy tiffile lmfit
pip3 install opencv-python 
cd Users
jupyter notebook
