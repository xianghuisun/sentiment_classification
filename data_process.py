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

def get_padded_char(sentences_list,char2id,max_seq_len,max_word_len):
    char_embedding=np.zeros(shape=(len(sentences_list),max_seq_len,max_word_len),dtype=np.int32)
    for i in range(len(sentences_list)):
        sentence=sentences_list[i]
        sentence_length=len(sentence)
        for j in range(min(sentence_length,max_seq_len)):
            word=sentence[j]
            word_length=len(word)
            for k in range(min(word_length,max_word_len)):
                char=word[k]
                char_id=char2id.get(char,char2id['UNK'])
                char_embedding[i][j][k]=char_id
            char_embedding[i][j][min(word_length,max_word_len):]=0
        char_embedding[i][min(max_seq_len,sentence_length):][:]=0
    return char_embedding#返回的就是一个三维矩阵，第一维是每一个句子，第二维是每一个单词，第三维是每一个字符，将它送入神经网络提取特征。这就是train_chars

def get_parameter(sentences,labels,embedding_dim,char_embedding_dim,pa_path):
    tag_set=set()
    tag2id={}
    word2id={}
    all_words=[]
    char_set=set()
    char2id={}
    char2id['UNK']=len(char2id)
    sentences_list=[]
    for each_sentence,each_label in zip(sentences,labels):
        tag_set.add(each_label)
        try:
            sentence_list=each_sentence.strip().split()
        except:
            print(each_sentence)
        sentences_list.append(sentence_list)
        for each_word in sentence_list:
            all_words.append(each_word)
            for each_char in each_word:
                char_set.add(each_char)
    
    for char in list(char_set):
        char2id[char]=len(char2id)
    print("the length of char2id is ",len(char2id))
    
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
    char_embedding_matrix=np.random.uniform(-1.0,1.0,size=(len(char2id),char_embedding_dim))
    parameter_=(word2id,tag2id,embedding_matrix,char2id,char_embedding_matrix)
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
    return pad_seq_ids   #这就是train_seqs

def tag_ids(labels,tag2id):
    train_label=np.zeros(shape=(len(labels),len(tag2id)))
    for i in range(len(labels)):
        currect_tag=labels[i]
        assert currect_tag in tag2id
        tag_id=tag2id[currect_tag]
        train_label[i][tag_id]=1
    return train_label      #这就是train_labels


    
    