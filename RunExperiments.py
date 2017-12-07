# from experiments.lenet import lenet5-ard

# exec ('experiments/lenet/lenet5-ard.py')

import os

dct_environ_var = {
    "THEANO_FLAGS" : "floatX=float32,device=cuda,lib.cnmem=1",
    "PATH" : "/usr/local/cuda-8.0/bin", #:$PATH
    "LD_LIBRARY_PATH": "/usr/local/cuda/lib64" # :$LD_LIBRARY_PATH
}

for k, v in dct_environ_var.items():
    if k in os.environ.keys():
        if not v in os.environ[k]:
            os.environ[k] += ':{0}'.format(v)
    else:
        os.environ[k] = v

# os.environ["THEANO_FLAGS"] = "floatX=float32,device=cuda,lib.cnmem=1"#"mode=FAST_RUN,device=gpu,floatX=float32"
# # os.environ["THEANO_FLAGS"] = "floatX=float32,device=cpu,lib.cnmem=1"#"mode=FAST_RUN,device=gpu,floatX=float32"
#
# export PATH="/usr/local/cuda-8.0/bin:$PATH"
# export THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32
# export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
#
# os.environ['CPLUS_INCLUDE_PATH'] = '/usr/local/cuda/include'

# print (os.environ)
# os.exit(0)

# THEANO_FLAGS='floatX=float32,device=cuda,lib.cnmem=1'
# python -c '' \
import theano
print(theano.config.device)
print(theano.gpuarray.dnn.dnn_present())

# import experiments.vgglike.vgglike_ard#.py

print(os.listdir('experiments/vgglike'))

# __import__('experiments/vgglike/vgglike-wot')
# import sys
# sys.exe
# os.execf execfile(

exec(open('experiments/vgglike/vgglike-wot.py').read())