export http_proxy=http://10.223.133.20:52107
export https_proxy=http://10.223.133.20:52107

pip install visdom
export PYTHONPATH=/root/.cache/wheels/:/usr/local/lib/python2.7:$PYTHONPATH

python train.py