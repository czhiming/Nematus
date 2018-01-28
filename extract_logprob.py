#-*- coding:utf8 -*-
'''
Created on Sep 22, 2017

@author: czm
'''
from nematus.nmt import prepare_data,build_model,init_params
from nematus.theano_util import init_theano_params,load_params
from nematus.util import load_config
from nematus.data_iterator import TextIterator
import theano
import sys
import numpy
import cPickle as pkl

    
def extract_logprob(
        model='model/model.npz.best_bleu',
        train=['test/train.bpe.en','test/train.bpe.es'],
        test=['test/test.bpe.en','test/test.bpe.es'],
        batch_size=10
        ):
    """
    @function:获得对数似然特征
    """
    options = load_config(model)
    
    params = init_params(options)
    params = load_params(model, params)
    
    tparams = init_theano_params(params)
    
    trng,use_noise,x,x_mask,y,y_mask,\
        opt_ret, cost, ctx, tt, _ = build_model(tparams,options)
    
    #加载数据
    train = TextIterator(train[0], train[1],
                        options['dictionaries'][0], options['dictionaries'][1],
                        n_words_source=options['n_words_src'], n_words_target=options['n_words_tgt'],
                        batch_size=batch_size,
                        maxlen=1000, #设置尽可能长的长度
                        sort_by_length=False) #设为 False
    
    test = TextIterator(test[0], test[1],
                        options['dictionaries'][0], options['dictionaries'][1],
                        n_words_source=options['n_words_src'], n_words_target=options['n_words_tgt'],
                        batch_size=batch_size,
                        maxlen=1000, #设置尽可能长的长度
                        sort_by_length=False) #设为 False
    
    f_cost = theano.function([x,x_mask,y,y_mask],cost,name='f_cost')

    #################### train #######################
    n_samples = 0
    for x, y in train:
            # 准备数据用于训练
            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=1000,
                                                n_words_src=options['n_words_src'],
                                                n_words=options['n_words_tgt'])
            logprob = f_cost(x,x_mask,y,y_mask)
            with open('features/train.es-en.logprob','a+') as fp:
                fp.writelines('\n'.join(map(lambda x:str(x), list(logprob)))+'\n')
                     
            n_samples += y.shape[1]
            print 'processed:',n_samples,'samples ...'
    
    ################### test ########################
    n_samples = 0
    for x, y in test:
            # 准备数据用于训练
            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=1000,
                                                n_words_src=options['n_words_src'],
                                                n_words=options['n_words_tgt'])
            logprob = f_cost(x,x_mask,y,y_mask)
            with open('features/test.es-en.logprob','a+') as fp:
                fp.writelines('\n'.join(map(lambda x:str(x), list(logprob)))+'\n')
                     
            n_samples += y.shape[1]
            print 'processed:',n_samples,'samples ...'

if __name__ == '__main__':
    extract_logprob(
        model='model_es_en/model.npz.best_bleu',
        train=['test/train.bpe.es','test/train.bpe.en'],
        test=['test/test.bpe.es','test/test.bpe.en'],
        batch_size=10
        )
    
    
    
    
    
    
    
    
    