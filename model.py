from data_process import *
import tensorflow as tf

def batch_generate(seqs,tags,actual_length,batch_size):
    assert len(seqs)==len(tags)
    assert type(seqs)==list and type(tags)==np.ndarray
    shuffled=np.random.permutation(len(seqs))
    seqs=np.array(seqs)[shuffled]
    tags=tags[shuffled]
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
    def __init__(self,tag2id,config,embedding_matrix):
        self.num_tags=len(tag2id)
        self.tag2id=tag2id
        self.batch_size=config.batch_size
        self.embedding_dim=config.embedding_dim
        self.embedding_size=config.embedding_size
        self.hidden_dim=config.hidden_dim
        self.max_seq_len=config.max_seq_len
        self.embedding_matrix=embedding_matrix
        self.num_layers=3
    
    def add_placeholder(self):
        self.word_ids=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_seq_len])
        self.label_ids=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.num_tags])
        self.seq_length=tf.placeholder(dtype=tf.int32,shape=[self.batch_size])
        
    def embedding_layer(self):
        embedding_matrix=tf.constant(self.embedding_matrix,dtype=tf.float32)
        self.embeddings=tf.nn.embedding_lookup(params=embedding_matrix,ids=self.word_ids)
        assert self.embeddings.shape==(self.batch_size,self.max_seq_len,self.embedding_dim)
        
    def lstm_layer(self):
        cell=tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
        cells=tf.contrib.rnn.MultiRNNCell([cell,tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim),tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)])
        initial_state=cells.zero_state(self.batch_size,dtype=tf.float32)
        outputs,states=tf.nn.dynamic_rnn(cells,self.embeddings,initial_state=initial_state)
        assert outputs.shape==(self.batch_size,self.max_seq_len,self.hidden_dim)
        assert len(states)==self.num_layers and states[0][0].shape==states[2][1].shape==(self.batch_size,self.hidden_dim)
        outputs=tf.transpose(outputs,perm=[1,0,2])
        assert outputs.shape==(self.max_seq_len,self.batch_size,self.hidden_dim)
        self.lstm_out=outputs[-1]
        assert self.lstm_out.shape==(self.batch_size,self.hidden_dim)
    
    def project_layer(self):
        weights=tf.Variable(initial_value=tf.random_normal(shape=[self.hidden_dim,self.num_tags],dtype=tf.float32))
        biases=tf.Variable(initial_value=tf.random_normal(shape=[self.num_tags],dtype=tf.float32))
        self.logits=tf.matmul(self.lstm_out,weights)+biases
        assert self.logits.shape==(self.batch_size,self.num_tags)
    
    def loss_layer(self):
        assert self.logits.shape==self.label_ids.shape
        losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.label_ids)
        self.loss=tf.reduce_mean(losses)
        self.train_op=tf.train.AdamOptimizer(0.01).minimize(self.loss)
    
    def build_graph(self):
        self.add_placeholder()
        self.embedding_layer()
        self.lstm_layer()
        self.project_layer()
        self.loss_layer()
        print("Graph has been built !")
    
    def train(self,train_seq,actual_length,train_label):
        num_batches=len(train_seq)//self.batch_size
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(20):
                batches=batch_generate(train_seq,train_label,actual_length,self.batch_size)
                total_loss=0.0
                for step,(batch_x,batch_y,batch_length) in enumerate(batches):
                    assert batch_x.shape==self.word_ids.shape
                    assert batch_y.shape==self.label_ids.shape
                    assert self.seq_length.shape==batch_length.shape
                    feed_dict={self.word_ids:batch_x,self.label_ids:batch_y,self.seq_length:batch_length}
                    loss_val,_=sess.run([self.loss,self.train_op],feed_dict=feed_dict)
                    total_loss+=loss_val
                print("loss value is ",total_loss/num_batches)

if __name__ == "__main__":
    file_path='/home/xhsun/Documents/assignment/sentiment_classification/train.xlsx'
    sentences,labels=read_file(file_path)
    parameter_path='/home/xhsun/Documents/assignment/parameter.pkl'
    #get_parameter(sentences,labels,embedding_dim=100,pa_path=parameter_path)
    with open(parameter_path,'rb') as f:
        word2id,tag2id,embedding_matrix=pickle.load(f)
    sentence_ids=sentence_to_id(sentences,word2id)
    config=Config(embedding_size=len(word2id))
    pad_seq_ids,actual_length=pad_sentence_ids(sentence_ids,max_seq_len=100)
    train_label=tag_ids(labels,tag2id)
    model=S_A_model(tag2id,config,embedding_matrix)
    model.build_graph()
    model.train(train_seq=pad_seq_ids,actual_length=actual_length,train_label=train_label)
    
            
        
        