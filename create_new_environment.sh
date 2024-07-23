conda create --name my_env jupyter spyder conda pims numpy cython matplotlib pandas anaconda lmfit
conda activate my_env
conda update conda --yes
conda install -c conda forge trackpy ffmpeg tiffile av --yes
pip install pims moviepy numba seaborn scipy
pip3 install opencv-python 
cd Users
jupyter notebook