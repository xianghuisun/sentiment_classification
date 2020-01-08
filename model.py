from data_process import *
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
gpu_config=tf.ConfigProto()
gpu_config.gpu_options.allow_growth=True
def batch_generate(seqs,batch_size,tags,train_test='train'):
    shuffled=np.random.permutation(len(seqs))
    seqs=np.array(seqs)[shuffled]
    start=0
    if train_test=='train':
        tags=tags[shuffled]
        for i in range(0,len(seqs),batch_size):
            yield seqs[start:start+batch_size],tags[start:start+batch_size]
            start+=batch_size
    else:
        for i in range(0,len(seqs),batch_size):
            yield seqs[start:start+batch_size]
            start+=batch_size
        
     
class Config:
    def __init__(self,embedding_size):
        self.embedding_size=embedding_size
        self.hidden_dim=128
        self.embedding_dim=100
        self.max_seq_len=70
        #self.model_save_path='/home/sun_xh/sentiment_analysis/log/model.ckpt'
        #self.model_save_path='/home/xhsun/Documents/assignment/log/model.ckpt'
        self.model_save_path=r'C:\Users\Tony Sun\Desktop\sentiment_classfication\log\model.ckpt'
        
class S_A_model:
    def __init__(self,tag2id,config,embedding_matrix,batch_size):
        self.num_tags=len(tag2id)
        self.tag2id=tag2id
        self.batch_size=batch_size
        self.embedding_dim=config.embedding_dim
        self.embedding_size=config.embedding_size
        self.hidden_dim=config.hidden_dim
        self.max_seq_len=config.max_seq_len
        self.embedding_matrix=embedding_matrix
        self.model_save_path=config.model_save_path
        self.num_layers=2
        tf.reset_default_graph()
    
    def add_placeholder(self):
        self.word_ids=tf.placeholder(dtype=tf.int32,shape=[None,self.max_seq_len])
        self.label_ids=tf.placeholder(dtype=tf.int32,shape=[None,self.num_tags])
        
    def embedding_layer(self):
        embedding_matrix=tf.constant(self.embedding_matrix,dtype=tf.float32)
        self.embeddings=tf.nn.embedding_lookup(params=embedding_matrix,ids=self.word_ids)
        
    def lstm_layer(self):
        cell=tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_dim)
        cells=tf.contrib.rnn.MultiRNNCell([cell,tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim*2)])
        #initial_state=cells.zero_state(self.batch_size,dtype=tf.float32)
        outputs,states=tf.nn.dynamic_rnn(cells,self.embeddings,dtype=tf.float32)
        outputs=tf.transpose(outputs,perm=[1,0,2])
        self.lstm_out=outputs[-1]
        #assert self.lstm_out.shape==(self.batch_size,self.hidden_dim*2)
    
    def project_layer(self):
        weights=tf.Variable(initial_value=tf.random_normal(shape=[self.hidden_dim*2,self.num_tags],dtype=tf.float32))
        biases=tf.Variable(initial_value=tf.random_normal(shape=[self.num_tags],dtype=tf.float32))
        self.logits=tf.matmul(self.lstm_out,weights)+biases
        self.predict_=tf.cast(tf.argmax(self.logits,axis=-1),dtype=tf.int32)
        #assert self.predict_.shape==(batch_size,)
        #assert self.logits.shape==(self.batch_size,self.num_tags)
    
    def loss_layer(self):
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
    
    def train(self,train_seq,train_label):
        num_batches=len(train_seq)//self.batch_size
        saver=tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(10):
                batches=batch_generate(train_seq,self.batch_size,tags=train_label,train_test='train')
                total_loss=0.0
                for step,(batch_x,batch_y) in enumerate(batches):
                    feed_dict={self.word_ids:batch_x,self.label_ids:batch_y}
                    loss_val,_=sess.run([self.loss,self.train_op],feed_dict=feed_dict)
                    total_loss+=loss_val
                if epoch%2==0:
                    print("loss value is ",total_loss/num_batches)
                    saver.save(sess,self.model_save_path)
                    
    def test(self,test_seq):
        saver=tf.train.Saver()
        result=[]
        with tf.Session() as sess:
            saver.restore(sess,self.model_save_path)
            batches=batch_generate(seqs=test_seq,batch_size=self.batch_size,tags=None,train_test='test')
            for step,batch_x in enumerate(batches):
                batch_x.shape==(self.batch_size,self.max_seq_len)
                predict_res=sess.run(self.predict_,feed_dict={self.word_ids:batch_x})
                result.extend(list(predict_res))
        id2tag={key_:value for value,key_ in self.tag2id.items()}
        res=[]
        for id_ in result:
            res.append(id2tag[id_])
        return res
    

def test_accuracy(file_path,parameter_path):
    test_seq,test_tag=read_file(file_path)
    with open(parameter_path,'rb') as f:
        word2id,tag2id,embedding_matrix=pickle.load(f)
    sentence_ids=sentence_to_id(test_seq,word2id)
    config=Config(embedding_size=len(word2id))
    pad_seq_ids=pad_sentence_ids(sentence_ids,max_seq_len=config.max_seq_len)
    model=S_A_model(tag2id,config,embedding_matrix,batch_size=50)
    model.build_graph()
    result=model.test(test_seq=pad_seq_ids)
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
          
def train_model(train_file_path,parameter_path):
    sentences,labels=read_file(train_file_path)#将excel中的句子和标签提取出来，得到训练集
    get_parameter(sentences,labels,embedding_dim=100,pa_path=parameter_path)#将训练集中的有多少个单词和多少个标签统计出来，得到word2id，tag2id以及得到词嵌入矩阵
    with open(parameter_path,'rb') as f:
        word2id,tag2id,embedding_matrix=pickle.load(f)    
    sentence_ids=sentence_to_id(sentences,word2id)#将每一个句子中的每一个单词转成相应的在word2id中的索引
    config=Config(embedding_size=len(word2id))
    pad_seq_ids=pad_sentence_ids(sentence_ids,max_seq_len=config.max_seq_len)#根据max_seq_len的值pad所有的句子
    train_label=tag_ids(labels,tag2id)#将每一个句子对应的标签转成相应的在tag2id中的索引
    model=S_A_model(tag2id,config,embedding_matrix,batch_size=100)#模型的初始化
    model.build_graph()
    model.train(train_seq=pad_seq_ids,train_label=train_label)#训练

def test_model(test_file_path,parameter_path):
    sentences=read_file(test_file_path,train_test="test")#将excel中test.xlsx的句子提取出来，得到测试集
    with open(parameter_path,'rb') as f:
        word2id,tag2id,embedding_matrix=pickle.load(f)   #加载模型要用到的参数
    sentence_ids=sentence_to_id(sentences,word2id)#将每一个句子中的每一个单词转成相应的在word2id中的索引
    config=Config(embedding_size=len(word2id))
    pad_seq_ids=pad_sentence_ids(sentence_ids,max_seq_len=config.max_seq_len)#根据max_seq_len的值pad所有的句子
    model=S_A_model(tag2id,config,embedding_matrix,batch_size=10)#模型的初始化
    model.build_graph()
    result=model.test(test_seq=pad_seq_ids)
    #with open('/home/xhsun/Documents/assignment/sentiment_classification/result_.pkl','wb') as f:
    #    pickle.dump(result,f)
    with open(r'C:\Users\Tony Sun\Desktop\sentiment_classfication\sentiment_classification\result.txt','w',encoding='utf-8') as f:
        for tag in result:
            f.write(tag)
            f.write("\n")
            
if __name__ == "__main__":
    #file_path='/home/xhsun/Documents/assignment/sentiment_classification/train.xlsx'
    #test_path='/home/xhsun/Documents/assignment/sentiment_classification/test.xlsx'
    #parameter_path='/home/xhsun/Documents/assignment/parameter.pkl'
    #parameter_path='/home/sun_xh/sentiment_analysis/parameter.pkl'
<<<<<<< HEAD
    
=======
    file_path=r'C:\Users\Tony Sun\Desktop\sentiment_classfication\sentiment_classification\train.xlsx'
    test_path=r'C:\Users\Tony Sun\Desktop\sentiment_classfication\sentiment_classification\test.xlsx'
    parameter_path=r'C:\Users\Tony Sun\Desktop\sentiment_classfication\parameter.pkl'
>>>>>>> 1f997315c4e88894ab38d947a87b782ae5331643
    #train_model(train_file_path=file_path,parameter_path=parameter_path)
    #test_accuracy(file_path=file_path,parameter_path=parameter_path)
    
    test_model(test_file_path=test_path,parameter_path=parameter_path)
        
    
            
        
        
