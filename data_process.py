import pandas as pd
import pickle
import numpy as np

def read_file(path,train_test="train"):
    data=pd.read_excel(path,header=None)
    sentences,labels=[],[]
    if train_test=="train":
        for seq,tag in zip(data[0].values,data[1].values):
            try:
                assert type(seq)==str
            except:
                seq=str(seq)
            sentences.append(seq)
            labels.append(tag)
        return sentences,labels
    else:
        for seq in data[0].values:
            try:
                assert type(seq)==str
            except:
                print(seq)
                seq=str(seq)
            sentences.append(seq)
        return sentences

def get_parameter(sentences,labels,embedding_dim,pa_path):
    tag_set=set()
    tag2id={}
    word2id={}
    all_words=[]
    for each_sentence,each_label in zip(sentences,labels):
        tag_set.add(each_label)
        try:
            sentence_list=each_sentence.strip().split()
        except:
            print(each_sentence)
        for each_word in sentence_list:
            all_words.append(each_word)
    tag_list=list(tag_set)
    for tag in tag_list:
        tag2id[tag]=len(tag2id)
    import collections
    import operator
    counter_words=collections.Counter(all_words)
    sorted_list=sorted(counter_words.items(),key=operator.itemgetter(1),reverse=True)
    for word,freq in sorted_list:
        word2id[word]=len(word2id)
    word2id['UNK']=len(word2id)
    print("length of word2id is ",len(word2id))
    embedding_matrix=np.random.uniform(-1.0,1.0,size=(len(word2id),embedding_dim))
    parameter_=(word2id,tag2id,embedding_matrix)
    with open(pa_path,'wb') as f:
        pickle.dump(parameter_,f) 

def sentence_to_id(sentences,word2id):
    sentence_list=[]
    for each_sentence in sentences:
        each_sentence_list=each_sentence.strip().split()
        sentence_list.append(each_sentence_list)
    sentence_ids=[]
    for each_sentence in sentence_list:
        seq_id=[]
        for word in each_sentence:
            word_id=word2id.get(word,word2id['UNK'])
            seq_id.append(word_id)
        sentence_ids.append(seq_id)
    return sentence_ids

def pad_sentence_ids(sentence_ids,max_seq_len):
    pad_seq_ids=[]
    for each_seq in sentence_ids:
        length=len(each_seq)
        if length>=max_seq_len:
            pad_seq_ids.append(each_seq[:max_seq_len])
        else:
            pad_seq_ids.append(each_seq[:max_seq_len]+[0]*(max_seq_len-length))
    assert np.array(pad_seq_ids).shape==(len(sentence_ids),max_seq_len)
    return pad_seq_ids

def tag_ids(labels,tag2id):
    train_label=np.zeros(shape=(len(labels),len(tag2id)))
    for i in range(len(labels)):
        currect_tag=labels[i]
        assert currect_tag in tag2id
        tag_id=tag2id[currect_tag]
        train_label[i][tag_id]=1
    return train_label


    
    