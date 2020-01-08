from data_process import *
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
gpu_config=tf.ConfigProto()
gpu_config.gpu_options.allow_growth=True

def batch_generate(batch_size,paded_seqs,tags=None,chars=None,train_test="train"):
    shuffled=np.random.permutation(len(paded_seqs))
    start=0
    paded_seqs=np.array(paded_seqs)[shuffled]
    if train_test=="train" and type(chars)==np.ndarray:
        tags=tags[shuffled]
        chars=chars[shuffled]
        for i in range(0,len(paded_seqs),batch_size):
            yield paded_seqs[start:start+batch_size],tags[start:start+batch_size],chars[start:start+batch_size]
            start+=batch_size
    elif train_test=="train" and chars==None:
        tags=tags[shuffled]
        for i in range(0,len(paded_seqs),batch_size):
            yield paded_seqs[start:start+batch_size],tags[start:start+batch_size]
            start+=batch_size
    elif train_test=="test" and type(chars)==np.ndarray:
        chars=chars[shuffled]
        for i in range(0,len(paded_seqs),batch_size):
            yield paded_seqs[start:start+batch_size],chars[start:start+batch_size]
            start+=batch_size
    else:
        assert train_test=="test" and chars==None
        for i in range(0,len(paded_seqs),batch_size):
            yield paded_seqs[start:start+batch_size]
            start+=batch_size

class Config:
    def __init__(self,embedding_size):
        self.embedding_size=embedding_size
        self.hidden_dim=128
        self.char_hidden_dim=100
        self.embedding_dim=100
        self.max_seq_len=70
        self.model_save_path='/home/sun_xh/sentiment_analysis/log/bi_model.ckpt'
        #self.model_save_path='/home/xhsun/Documents/assignment/log/bi_model.ckpt'
        self.max_word_len=5
    
class BiLSTM_model:
    def __init__(self,tag2id,config,embedding_matrix,char_embedding_matrix,batch_size):
        self.num_tags=len(tag2id)
        self.tag2id=tag2id
        self.batch_size=batch_size
        self.embedding_dim=config.embedding_dim
        self.embedding_size=config.embedding_size
        self.hidden_dim=config.hidden_dim
        self.char_hidden_dim=config.char_hidden_dim
        self.max_seq_len=config.max_seq_len
        self.embedding_matrix=embedding_matrix
        self.char_embedding_matrix=char_embedding_matrix
        self.char_embedding_dim=char_embedding_matrix.shape[-1]
        self.model_save_path=config.model_save_path
        self.max_word_len=config.max_word_len
        tf.reset_default_graph()
    def add_placeholder(self):
        self.word_ids=tf.placeholder(dtype=tf.int32,shape=[None,self.max_seq_len])#将它作用在词嵌入张量上，得到(batch_size,max_seq_len,embedding_dim)
        self.label_ids=tf.placeholder(dtype=tf.int32,shape=[None,self.num_tags])
        self.char_ids=tf.placeholder(dtype=tf.int32,shape=[None,self.max_seq_len,self.max_word_len])#将它作用在字符嵌入张量上，得到(batch_size,max_seq_len,max_word_len,char_embedding_dim)
    
    def embedding_layer(self):
        embedding_matrix=tf.constant(self.embedding_matrix,dtype=tf.float32)
        char_embedding_matrix=tf.constant(self.char_embedding_matrix,dtype=tf.float32)
        self.embeddings=tf.nn.embedding_lookup(params=embedding_matrix,ids=self.word_ids)
        self.char_embeddings=tf.nn.embedding_lookup(params=char_embedding_matrix,ids=self.char_ids)
    
    def char_bilstm_layer(self):
        char_cell_fw=tf.contrib.rnn.BasicLSTMCell(num_units=self.char_hidden_dim)
        char_cell_bw=tf.contrib.rnn.BasicLSTMCell(num_units=self.char_hidden_dim)
        char_lstm_inputs=tf.reshape(tensor=self.char_embeddings,shape=[-1,self.max_seq_len,self.max_word_len*self.char_embedding_dim])
        outputs,states=tf.nn.bidirectional_dynamic_rnn(char_cell_fw,char_cell_bw,inputs=char_lstm_inputs,dtype=tf.float32)
        self.char_lstm_outputs=tf.concat(values=[outputs[0],outputs[1]],axis=-1)
        #batch_size,max_seq_len,char_hidden_dim*2
    
    def bilstm_layer(self):
        bilstm_input=tf.concat(values=[self.char_lstm_outputs,self.embeddings],axis=-1)
        assert bilstm_input.shape[-1]==self.char_hidden_dim*2+self.embedding_dim
        cell_fw=tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
        cell_bw=tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
        outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs=bilstm_input,dtype=tf.float32)
        self.bilstm_outputs=tf.concat(values=[outputs[0],outputs[1]],axis=-1)
        self.bilstm_outputs=tf.transpose(self.bilstm_outputs,perm=[1,0,2])[-1]

    
    def project_layer(self):
        weights=tf.Variable(initial_value=tf.random_normal(shape=[self.hidden_dim*2,self.num_tags],dtype=tf.float32))
        biases=tf.Variable(initial_value=tf.random_normal(shape=[self.num_tags],dtype=tf.float32))
        self.logits=tf.matmul(self.bilstm_outputs,weights)+biases
        self.predict_=tf.cast(tf.argmax(self.logits,axis=-1),dtype=tf.int32)
        

    def loss_layer(self):
        losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.label_ids)
        self.loss=tf.reduce_mean(losses)
        self.train_op=tf.train.AdamOptimizer(0.01).minimize(self.loss)
    
    def build_graph(self):
        self.add_placeholder()
        self.embedding_layer()
        self.char_bilstm_layer()
        self.bilstm_layer()
        self.project_layer()
        self.loss_layer()
        print("The graph has been built!")
    
    def train(self,train_seq,train_tags,train_chars):
        assert len(train_seq)==len(train_tags)==len(train_chars)
        saver=tf.train.Saver()
        num_batches=len(train_seq)//self.batch_size
        with tf.Session(config=gpu_config) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(30):
                batches=batch_generate(batch_size=self.batch_size,paded_seqs=train_seq,tags=train_tags,chars=train_chars)
                total_loss=0.0
                for step,(batch_seq,batch_tag,batch_char) in enumerate(batches):
                    feed_dict={self.word_ids:batch_seq,self.char_ids:batch_char,self.label_ids:batch_tag}     
                    _,loss_val=sess.run([self.train_op,self.loss],feed_dict=feed_dict)
                    total_loss+=loss_val
                if epoch%10==0:
                    print("loss value is ",total_loss/num_batches)
                    saver.save(sess,save_path=self.model_save_path)
    
    def test(self,test_seq,test_chars):
        assert len(test_seq)==len(test_chars)
        saver=tf.train.Saver()
        result=[]
        with tf.Session(config=gpu_config) as sess:
            saver.restore(sess,self.model_save_path)
            batches=batch_generate(batch_size=self.batch_size,paded_seqs=test_seq,tags=None,chars=test_chars,train_test="test")
            for step,(batch_seq,batch_char) in enumerate(batches):
                predict_res=sess.run(self.predict_,feed_dict={self.word_ids:batch_seq,self.char_ids:batch_char})
                result.extend(list(predict_res))
        id2tag={key_:value for value,key_ in self.tag2id.items()}
        res=[]
        for id_ in result:
            res.append(id2tag[id_])
        return res
                
    

def train_model(train_file_path,parameter_path):
    sentences,labels=read_file(path=train_file_path)
    get_parameter(sentences=sentences,labels=labels,embedding_dim=100,char_embedding_dim=25,pa_path=parameter_path)
    with open(parameter_path,'rb') as f:
        word2id,tag2id,embedding_matrix,char2id,char_embedding_matrix=pickle.load(f)
    sentences_id=sentence_to_id(sentences=sentences,word2id=word2id)#将每一句话转换成对应的下标索引
    config=Config(embedding_size=len(word2id))
    pad_seqs_id=pad_sentence_ids(sentence_ids=sentences_id,max_seq_len=config.max_seq_len)
    train_label=tag_ids(labels=labels,tag2id=tag2id)
    sentences_list=[]
    for each_sentence in sentences:
        try:
            sentence_list=each_sentence.strip().split()
        except:
            print(each_sentence)
        sentences_list.append(sentence_list)
    train_char=get_padded_char(sentences_list=sentences_list,char2id=char2id,max_seq_len=config.max_seq_len,max_word_len=config.max_word_len)
    model=BiLSTM_model(tag2id=tag2id,config=config,embedding_matrix=embedding_matrix,char_embedding_matrix=char_embedding_matrix,batch_size=100)
    model.build_graph()
    model.train(train_seq=pad_seqs_id,train_chars=train_char,train_tags=train_label)

def test_accuracy(file_path,parameter_path):
    test_seq,test_tag=read_file(file_path)
    with open(parameter_path,'rb') as f:
        word2id,tag2id,embedding_matrix,char2id,char_embedding_matrix=pickle.load(f)
    sentence_ids=sentence_to_id(test_seq,word2id)
    config=Config(embedding_size=len(word2id))
    pad_seq_ids=pad_sentence_ids(sentence_ids,max_seq_len=config.max_seq_len)
    sentences_list=[]
    for each_sentence in test_seq:
        try:
            sentence_list=each_sentence.strip().split()
        except:
            print('*'*2000)
        sentences_list.append(sentence_list)
    test_char=get_padded_char(sentences_list=sentences_list,char2id=char2id,max_seq_len=config.max_seq_len,max_word_len=config.max_word_len)
    model=BiLSTM_model(tag2id=tag2id,config=config,embedding_matrix=embedding_matrix,char_embedding_matrix=char_embedding_matrix,batch_size=100)
    model.build_graph()
    result=model.test(test_seq=pad_seq_ids,test_chars=test_char)
    
    print(len(result))
    print(len(test_tag))
    assert len(result)==len(test_tag)
    correct=0
    total=0
    for predict_tag,golden_tag in zip(result,test_tag):
        if predict_tag==golden_tag:
            correct+=1
        total+=1
    print("correct is ",correct)
    print("correct / total is ",correct/total)

def test_model(test_file_path,parameter_path):
    sentences=read_file(path=test_file_path,train_test="test")
    with open(parameter_path,'rb') as f:
        word2id,tag2id,embedding_matrix,char2id,char_embedding_matrix=pickle.load(f)
    sentences_id=sentence_to_id(sentences=sentences,word2id=word2id)
    config=Config(embedding_size=len(word2id))
    pad_seq_ids=pad_sentence_ids(sentence_ids,max_seq_len=config.max_seq_len)
    sentences_list=[]
    for each_sentence in sentences:
        try:
            sentence_list=each_sentence.strip().split()
        except:
            print('*'*2000)
        sentences_list.append(sentence_list)
    test_char=get_padded_char(sentences_list=sentences_list,char2id=char2id,max_seq_len=config.max_seq_len,max_word_len=config.max_word_len)
    model=BiLSTM_model(tag2id=tag2id,config=config,embedding_matrix=embedding_matrix,char_embedding_matrix=char_embedding_matrix,batch_size=60)
    model.build_graph()
    result=model.test(test_seq=pad_seq_ids,test_chars=test_char)
    with open('/home/sun_xh/sentiment_analysis/bi_result.txt','w',encoding='utf-8') as f:
        for tag in result:
            f.write(tag)
            f.write("\n")
    
    
    
if __name__ == "__main__":
    #train_file_path='/home/xhsun/Documents/assignment/sentiment_classification/train.xlsx'
    #parameter_path='/home/xhsun/Documents/assignment/char_parameter.pkl'
    parameter_path='/home/sun_xh/sentiment_analysis/char_parameter.pkl'
    train_file_path='/home/sun_xh/sentiment_analysis/train.xlsx'
    test_file_path='/home/sun_xh/sentiment_analysis/test.xlsx'
    train_model(train_file_path=train_file_path,parameter_path=parameter_path)
    test_accuracy(file_path=train_file_path,parameter_path=parameter_path)
    test_model(test_file_path=test_file_path,parameter_path=parameter_path)


