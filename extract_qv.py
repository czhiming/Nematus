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

    
def extract_qv(
        model='model/model.npz.best_bleu',
        train=['test/train.bpe.en','test/train.bpe.es'],
        test=['test/test.bpe.en','test/test.bpe.es'],
        batch_size=10
        ):
    """
    @function:获得质量向量(quality vector)
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
    
    f_tt = theano.function([x,x_mask,y,y_mask],tt,name='f_tt')

    #################### train #######################
    n_samples = 0
    for x, y in train:
            # 准备数据用于训练
            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=1000,
                                                n_words_src=options['n_words_src'],
                                                n_words=options['n_words_tgt'])
            tt_ = f_tt(x,x_mask,y,y_mask)
            Wt = tparams['ff_logit_W'].get_value()
            
            for j in range(y.shape[1]):
                
                qv_ = []
                for i in range(y.shape[0]):
                    if y_mask[i][j] == 1:
                        index = y[i][j]
                        qv = Wt[:,index].T*tt_[i,j,:]
                        qv_.append(list(qv))
                qv_ = numpy.array(qv_)
                qv_ = list(map(lambda x:str(x),qv_.mean(axis=0)))
                
                with open('features/train.nmt.qv','a+') as fp:
                    fp.writelines('\t'.join(qv_)+'\n')
                    
            n_samples += y.shape[1]
            print 'processed:',n_samples,'samples ...'
    
    ################### test ########################
    n_samples = 0
    for x, y in test: #*****
            # 准备数据用于训练
            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=1000,
                                                n_words_src=options['n_words_src'],
                                                n_words=options['n_words_tgt'])
            tt_ = f_tt(x,x_mask,y,y_mask)
            Wt = tparams['ff_logit_W'].get_value()
            
            for j in range(y.shape[1]):
                
                qv_ = []
                for i in range(y.shape[0]):
                    if y_mask[i][j] == 1:
                        index = y[i][j]
                        qv = Wt[:,index].T*tt_[i,j,:]
                        qv_.append(list(qv))
                qv_ = numpy.array(qv_)
                qv_ = list(map(lambda x:str(x),qv_.mean(axis=0)))
                
                with open('features/test.nmt.qv','a+') as fp: #*****
                    fp.writelines('\t'.join(qv_)+'\n')
                    
            n_samples += y.shape[1]
            print 'processed:',n_samples,'samples ...'


if __name__ == '__main__':
    extract_qv(
        model='model_en_es/model.npz.best_bleu',
        train=['test/train.bpe.en','test/train.bpe.es'],
        test=['test/test.bpe.en','test/test.bpe.es'],
        batch_size=10
        )
    
    
    
    
    
    
    
    
    