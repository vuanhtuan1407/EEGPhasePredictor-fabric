# create and activate conda env
conda create -n EEGPP python=3.10
conda activate EGGPP

# install libs
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c conda-forge lightning==2.4.0 -y
conda install -c conda-forge torchinfo==1.8.0 -y
conda install -c conda-forge pyyaml==6.0.2 -y
conda install -c conda-forge scikit-learn==1.5.2 -y
conda install -c conda-forge seaborn=0.13.2 -y
conda install -c conda-forge pandas==2.2.2 -y
conda install -c conda-forge scipy==1.14.1 -y
conda install -c conda-forge numpy==2.1.1 -y
conda install -c conda-forge pywavelets==1.7.0 -y
conda install -c conda-forge dropbox
pip install joblib==1.4.2
pip install ptwt==0.1.9
pip install python-dotenv==1.0.1