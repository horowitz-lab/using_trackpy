conda create --name my_env python==3.10 av==9.2.0 jupyter spyder conda pims numpy cython matplotlib pandas anaconda lmfit
conda activate my_env
conda update conda --yes
conda install -c conda-forge trackpy ffmpeg --yes
pip install moviepy numba seaborn scipy tiffile 
pip3 install opencv-python 
cd Users
jupyter notebook
