#coding:utf-8
'''
Created on Dec 12, 2017

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

def get_emb(x, Wemb):
    emb = []
    for x_row in x:
        emb_row = []
        for idx in x_row:
            emb_row.append(list(Wemb[idx]))
        emb_row = numpy.array(emb_row)
        emb_row = numpy.sum(emb_row, axis=0) # 此处修改其他操作 mean,max,min,tf-idf
        emb.append(list(emb_row))
    return numpy.array(emb)

def word_embedding(
        model='model/model.npz.best_bleu',
        train=['test/train.bpe.en','test/train.bpe.es'],
        dev=['test/dev.bpe.en','test/dev.bpe.es'],
        test=['test/test.bpe.en','test/test.bpe.es'],
        batch_size=10
        ):
    """
    @function:获得词向量
    """
    options = load_config(model) # 加载设置的超参数
    
    params = init_params(options)
    params = load_params(model, params) # 加载模型参数
    
    #加载数据
    train = TextIterator(train[0], train[1],
                        options['dictionaries'][0], options['dictionaries'][1],
                        n_words_source=options['n_words_src'], n_words_target=options['n_words_tgt'],
                        batch_size=batch_size,
                        maxlen=1000, #设置尽可能长的长度
                        sort_by_length=False) #设为 False
    
    dev = TextIterator(dev[0], dev[1],
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
    
    #################### train #######################
    Wemb = params['Wemb']
    Wemb_dec = params['Wemb_dec']
    
    n_samples = 0
    for x, y in train:
        x_emb = get_emb(x, Wemb)
        y_emb = get_emb(y, Wemb_dec)
        
        with open('features/emb/train.es-en.es.emb','a+') as fp:
            for x_row in x_emb: 
                fp.writelines('\t'.join(map(lambda x:str(x), x_row))+'\n')
        with open('features/emb/train.es-en.en.emb','a+') as fp:
            for y_row in y_emb: 
                fp.writelines('\t'.join(map(lambda x:str(x), y_row))+'\n')
                
        n_samples += len(x)
        print 'processed:',n_samples,'samples ...'
    
    ################### test ########################
    Wemb = params['Wemb']
    Wemb_dec = params['Wemb_dec']
    
    n_samples = 0
    for x, y in test:
        x_emb = get_emb(x, Wemb)
        y_emb = get_emb(y, Wemb_dec)
        
        with open('features/emb/test.es-en.es.emb','a+') as fp:
            for x_row in x_emb: 
                fp.writelines('\t'.join(map(lambda x:str(x), x_row))+'\n')
        with open('features/emb/test.es-en.en.emb','a+') as fp:
            for y_row in y_emb: 
                fp.writelines('\t'.join(map(lambda x:str(x), y_row))+'\n')
                
        n_samples += len(x)
        print 'processed:',n_samples,'samples ...'
    
    ################### dev ########################
    Wemb = params['Wemb']
    Wemb_dec = params['Wemb_dec']
    
    n_samples = 0
    for x, y in dev:
        x_emb = get_emb(x, Wemb)
        y_emb = get_emb(y, Wemb_dec)
        
        with open('features/emb/dev.es-en.es.emb','a+') as fp:
            for x_row in x_emb: 
                fp.writelines('\t'.join(map(lambda x:str(x), x_row))+'\n')
        with open('features/emb/dev.es-en.en.emb','a+') as fp:
            for y_row in y_emb: 
                fp.writelines('\t'.join(map(lambda x:str(x), y_row))+'\n')
                
        n_samples += len(x)
        print 'processed:',n_samples,'samples ...'


if __name__ == '__main__':
    word_embedding(
        model='model_es_en/model.npz.best_bleu',
        train=['test/train.bpe.es','test/train.bpe.en'],
        dev=['test/dev.bpe.es','test/dev.bpe.en'],
        test=['test/test.bpe.es','test/test.bpe.en']
        )
    



