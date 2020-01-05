from data_process import *
import tensorflow as tf

def batch_generate(seqs,batch_size,tags,train_test='train'):
   # assert len(seqs)==len(tags)
   # assert type(seqs)==list and type(tags)==np.ndarray
    shuffled=np.random.permutation(len(seqs))
    seqs=np.array(seqs)[shuffled]
    num_batches=len(seqs)//batch_size
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
        #self.batch_size=100
        self.max_seq_len=70
        #self.model_save_path='/home/sun_xh/sentiment_analysis/log/model.ckpt'
        self.model_save_path='/home/xhsun/Documents/assignment/log/model.ckpt'
     
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
        self.seq_length=tf.placeholder(dtype=tf.int32,shape=[None])
        
    def embedding_layer(self):
        embedding_matrix=tf.constant(self.embedding_matrix,dtype=tf.float32)
        self.embeddings=tf.nn.embedding_lookup(params=embedding_matrix,ids=self.word_ids)
        #assert self.embeddings.shape==(self.batch_size,self.max_seq_len,self.embedding_dim)
        
    def lstm_layer(self):
        cell=tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_dim)
        cells=tf.contrib.rnn.MultiRNNCell([cell,tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim*2)])
        #initial_state=cells.zero_state(self.batch_size,dtype=tf.float32)
        outputs,states=tf.nn.dynamic_rnn(cells,self.embeddings,dtype=tf.float32)
        #assert outputs.shape==(self.batch_size,self.max_seq_len,self.hidden_dim)
        #assert len(states)==self.num_layers and states[0][0].shape==states[2][1].shape==(self.batch_size,self.hidden_dim)
        outputs=tf.transpose(outputs,perm=[1,0,2])
        #assert outputs.shape==(self.max_seq_len,self.batch_size,self.hidden_dim)
        self.lstm_out=outputs[-1]
        #assert self.lstm_out.shape==(self.batch_size,self.hidden_dim)
    
    def project_layer(self):
        weights=tf.Variable(initial_value=tf.random_normal(shape=[self.hidden_dim*2,self.num_tags],dtype=tf.float32))
        biases=tf.Variable(initial_value=tf.random_normal(shape=[self.num_tags],dtype=tf.float32))
        self.logits=tf.matmul(self.lstm_out,weights)+biases
        self.predict_=tf.cast(tf.argmax(self.logits,axis=-1),dtype=tf.int32)
        #assert self.predict_.shape==(batch_size,)
        #assert self.logits.shape==(self.batch_size,self.num_tags)
    
    def loss_layer(self):
        #assert self.logits.shape==self.label_ids.shape
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
            for epoch in range(100):
                batches=batch_generate(train_seq,self.batch_size,tags=train_label,train_test='train')
                total_loss=0.0
                for step,(batch_x,batch_y) in enumerate(batches):
         #           assert batch_x.shape==self.word_ids.shape
          #          assert batch_y.shape==self.label_ids.shape
           #         assert self.seq_length.shape==batch_length.shape
                    feed_dict={self.word_ids:batch_x,self.label_ids:batch_y}
                    loss_val,_=sess.run([self.loss,self.train_op],feed_dict=feed_dict)
                    total_loss+=loss_val
                if epoch%20==0:
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
                print(predict_res)
        id2tag={key_:value for value,key_ in self.tag2id.items()}
        res=[]
        for id_ in result:
            res.append(id2tag[id_])
        return res
                


def get_test_file(file_path):
    data=pd.read_excel(file_path,header=None)
    sentences,labels=[],[]
    for seq,tag in zip(data[0].values,data[1].values):
        try:
            assert type(seq)==str
        except:
            seq=str(seq)
        sentences.append(seq)
        labels.append(tag)
    return sentences[-500:],labels[-500:]       

def test_model(test_seq,test_tag):
    parameter_path='/home/xhsun/Documents/assignment/parameter.pkl'
    with open(parameter_path,'rb') as f:
        word2id,tag2id,embedding_matrix=pickle.load(f)
    sentence_ids=sentence_to_id(test_seq,word2id)
    config=Config(embedding_size=len(word2id))
    pad_seq_ids,actual_length=pad_sentence_ids(sentence_ids,max_seq_len=config.max_seq_len)
    #test_laebl=tag_ids(test_tag,tag2id)
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
            
if __name__ == "__main__":
    file_path='/home/xhsun/Documents/assignment/sentiment_classification/train.xlsx'
    test_seq,test_tag=get_test_file(file_path)
    test_model(test_seq,test_tag)
    



# if __name__ == "__main__":
#     #file_path='/home/xhsun/Documents/assignment/sentiment_classification/train.xlsx'
#     file_path='/home/sun_xh/sentiment_analysis/train.xlsx'
#     sentences,labels=read_file(file_path)
#     #parameter_path='/home/xhsun/Documents/assignment/parameter.pkl'
#     parameter_path='/home/sun_xh/sentiment_analysis/parameter.pkl'
#     get_parameter(sentences,labels,embedding_dim=100,pa_path=parameter_path)
#     with open(parameter_path,'rb') as f:
#         word2id,tag2id,embedding_matrix=pickle.load(f)
#     sentence_ids=sentence_to_id(sentences,word2id)
#     config=Config(embedding_size=len(word2id))
#     pad_seq_ids,actual_length=pad_sentence_ids(sentence_ids,max_seq_len=config.max_seq_len)
#     train_label=tag_ids(labels,tag2id)
#     model=S_A_model(tag2id,config,embedding_matrix,batch_size=100)
#     model.build_graph()
#     model.train(train_seq=pad_seq_ids,train_label=train_label)
    
#     model=S_A_model(tag2id,config,embedding_matrix,batch_size=1)
#     model.build_graph()
#     sentence_test=["明天 开始 小 消失 一 个 星期 吧xd 于是 提前 祝 大家 新年 快乐 啦 ~~~贺图 用 去年 的 总结 真的 大 丈夫 么 | | | 嘛 | | | | | | | 想 说 的 话 写 在 图上 了 ， 这 回 是 真 的 2011byebye 了 … … ！ 总之 ， 除夕 过 了 就 真的 是 2012 了 = v = 大家 ， 新 的 一 年 加油 哦 ~~~（ 其实 还 是 有点 不好意思 at 大家 = 口 = 试试 能 不 能 评论 at 了 = 口 = at 不 上 请 原谅 我 | | | | ）"]
#     test_ids=sentence_to_id(sentence_test,word2id)
#     pad_ids,actual_len=pad_sentence_ids(test_ids,max_seq_len=config.max_seq_len)
#     print(test_ids)
#    # assert pad_ids.shape==(1,100)
#     model.test(test_seq=pad_ids)
    
    
            
        
        
