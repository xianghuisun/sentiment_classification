from data_process import *
import tensorflow as tf

def batch_generate(seqs,tags,actual_length,batch_size):
    assert len(seqs)==len(tags)
    shuffled=np.random.permutation(len(seqs))
    seqs=np.array(seqs)[shuffled]
    tags=np.array(tags)[shuffled]
    actual_length=np.array(actual_length)[shuffled]
    num_batches=len(seqs)//batch_size
    start=0
    for i in range(num_batches):
        yield seqs[start:start+batch_size],tags[start:start+batch_size],actual_length[start:start+batch_size]
        start+=batch_size
     
class Config:
    def __init__(self,embedding_size):
        self.embedding_size=embedding_size
        self.hidden_dim=128
        self.embedding_dim=100
        self.batch_size=100
        self.max_seq_len=100
     
class S_A_model:
    def __init__(self,tag2id,config):
        self.num_tags=len(tag2id)
        self.tag2id=tag2id
        self.batch_size=config.batch_size
        self.embedding_dim=config.embedding_dim
        self.embedding_size=config.embedding_size
        self.hidden_dim=config.hidden_dim
        self.max_seq_len=config.max_seq_len
    
    def add_placeholder(self):
        self.word_ids=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_seq_len])
        self.label_ids=tf.placeholder(dtype=tf.int32,shape=[self.batch_size])
        self.seq_length=tf.placeholder(dtype=tf.int32,shape=[self.batch_size])
        
    def embedding_layer(self,embedding_matrix):
        embedding_matrix=tf.constant(embedding_matrix,dtype=tf.float32)
        