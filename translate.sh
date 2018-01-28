#!/bin/sh

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=tools/moses

# theano device, in case you do not want to compute on gpu, change it to cpu
device=gpu

#model prefix
prefix=model_es_en/model.npz

#输入需要翻译的文件，和输出文件
dev=test/test.bpe.es
ref=test/test.tok.en


# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python nematus/translate.py \
     -m $prefix.best_bleu \
     -i $dev \
     -o $dev.output.dev \
     -k 12 -n -p 1



./postprocess-dev.sh < $dev.output.dev > $dev.output.postprocessed.dev













