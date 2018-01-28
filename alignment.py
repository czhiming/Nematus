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
import matplotlib.pyplot as plt
import codecs

def get_data(source, target, alignment):
    with codecs.open(source,'r',encoding='utf8') as fp:
        src = fp.readlines()
    with codecs.open(target,'r',encoding='utf8') as fp:
        trg = fp.readlines()
    align = []
    with open(alignment) as fp:
        align_data = []
        for lines in fp:
            lines = lines.strip()
            if lines != "":
                align_data.append(map(lambda x:float(x), lines.split('\t')))
            else:
                align.append(align_data)
                align_data = []
    
    for i in range(len(src)):
        align_matrix = numpy.array(align[i])
        src_sentence = src[i].strip().split()
        trg_sentence = trg[i].strip().split()
        
        show_matrix(align_matrix, src_sentence, trg_sentence)
        

def show_matrix(align_matrix, source, target):
    """
    @function:画出词对齐矩阵
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['xtick.direction'] = 'out'  
    plt.rcParams['ytick.direction'] = 'out'
    
    source = source + [u'</s>']
    target = target + [u'</s>']
    print 'source:',source
    print 'target:',target
    
    fig, ax = plt.subplots()
    width = 10
    
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    
    ax.xaxis.set_ticks_position('top')
    #ax.spines['top'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    #ax.spines['left'].set_position(('data',0))
    
    align_shape = align_matrix.shape
    indx = numpy.arange(align_shape[1])
    indy = numpy.arange(align_shape[0])
    
    scale_ = 10 # 图像大小
    out_matrix = numpy.ones([scale_*align_shape[0],scale_*align_shape[1]])
    for j in range(align_shape[0]):
        for k in range(align_shape[1]):
            out_matrix[j*scale_:(j+1)*scale_,k*scale_:(k+1)*scale_] *= align_matrix[j,k]
    #ax.pcolor(out_matrix)  
    ax.imshow(out_matrix, plt.cm.gray)
    ax.set_xticks(indx*width+5)
    ax.set_xticklabels(source, fontdict={'size':10, 'rotation':90})
    ax.set_yticks(indy*width+5)
    ax.set_yticklabels(target, fontdict={'size':10})
    plt.show()

def save_vocabulary(filename):
    vocab = {}
    with open(filename) as fp:
        for lines in fp:
            words = lines.strip().split()
            for word in words:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
    with open(filename+'.pkl', 'w') as fp:
        pkl.dump(vocab,fp)

def make_vocabulary(source, target):
    
    save_vocabulary(source)
    save_vocabulary(target)

def get_alignscore(filename):
    """
    @function: 获得对齐质量得分
    """
    print 'Process file name:%s' % filename
    with open(filename+'.vec', 'w') as fp:
        sum = 0
        for lines in open(filename):
            lines = lines.strip()
            if lines != "":
                data = lines.split('\t')
                data = map(lambda x:float(x), data)
                max = numpy.max(data)
                sum += max
            else:
                fp.writelines(str(sum)+'\n')
                sum = 0
    
def alignment(
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
    
    f_align = theano.function([x,x_mask,y,y_mask],opt_ret,name='f_cost')

    #################### train #######################
    """
    n_samples = 0
    for x, y in train:
            # 准备数据用于训练
            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=1000,
                                                n_words_src=options['n_words_src'],
                                                n_words=options['n_words_tgt'])
            align = f_align(x,x_mask,y,y_mask)['dec_alphas'] # (y, batch_size, x)
            align = align * y_mask[:,:,None] # 注意此处技巧
            align_shp = align.shape
            for j in range(align_shp[1]):
                row_ = int(numpy.sum(y_mask[:,j]))
                col_ = int(numpy.sum(x_mask[:,j]))
                align_data = align[:row_,j,:col_] # 词对齐矩阵
                
                with open('features/alignment/train.en-es.word.align','a+') as fp:
                    for data in align_data:
                        fp.writelines('\t'.join(map(lambda x:str(x), data))+'\n')
                    fp.writelines('\n')
                     
            n_samples += y.shape[1]
            print 'processed:',n_samples,'samples ...'
    """
    ################### test ########################
    n_samples = 0
    for x, y in test:
            # 准备数据用于训练
            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=1000,
                                                n_words_src=options['n_words_src'],
                                                n_words=options['n_words_tgt'])
            align = f_align(x,x_mask,y,y_mask)['dec_alphas'] # (y, batch_size, x)
            align = align * y_mask[:,:,None] # 注意此处技巧
            align_shp = align.shape
            for j in range(align_shp[1]):
                row_ = int(numpy.sum(y_mask[:,j]))
                col_ = int(numpy.sum(x_mask[:,j]))
                align_data = align[:row_,j,:col_] # 词对齐矩阵
                
                with open('features/alignment/test.en-es.word.align','a+') as fp:
                    for data in align_data:
                        fp.writelines('\t'.join(map(lambda x:str(x), data))+'\n')
                    fp.writelines('\n')
                     
            n_samples += y.shape[1]
            print 'processed:',n_samples,'samples ...'

if __name__ == '__main__':
    
#     alignment(
#         model='model_en_es/model.npz.best_bleu',
#         train=['test/train.bpe.en','test/train.bpe.es'],
#         test=['test/test.bpe.en','test/test.bpe.es'],
#         batch_size=10
#         )

#     获取数据画出词对齐矩阵
    get_data('test/train.bpe.en','test/train.bpe.es','features/alignment/train.en-es.word.align')
    
#     get_alignscore('features/alignment/train.en-es.word.align')
#     get_alignscore('features/alignment/test.en-es.word.align')
    
    
    
    
    
    
    
    
    
    
    