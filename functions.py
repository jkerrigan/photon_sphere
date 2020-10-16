import tensorflow as tf
import numpy as np
import pickle
from sqlalchemy import create_engine
import pandas as pd
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow
import youtokentome as yttm

def make_mask(x):
    """Deprecated"""
    y = np.zeros(100)
    print(x)
    y[:x] = 1
    return y

def make_multilabel(x):
    """Deprecated"""
    x_ = list(map(lambda xx: int(xx),x.split(' ')))
    y = np.zeros(100)
    y[range(len(x_))] = x_
    return y

def multi_pred(model,query,neg_list,anchor_list):
    predictions = model.predict([len(neg_list)*[query],neg_list,anchor_list])
    mean_dist = predictions[:,0].mean()
    mean_rel = (predictions[:,0]>predictions[:,1]).mean()
    #print('Dist',mean_dist,'Rel',mean_rel)
    return mean_rel#np.min([mean_dist,mean_rel])
        
def load_gravity(dir='/etc/pihole/',table='gravity'):
    db_connect = create_engine('sqlite:///'+dir+'gravity.db')
    connection = db_connect.raw_connection()
    df = pd.read_sql("SELECT * FROM {}".format(table), con=connection)
    connection.close()
    return df

def load_query_list(timestamp=None):
    if not timestamp:
        timestamp = 0
    db_connect = create_engine('sqlite:////etc/pihole/pihole-FTL.db')
    connection = db_connect.raw_connection()
    df = pd.read_sql("SELECT * FROM queries WHERE timestamp > {0}".format(timestamp), con=connection)
    connection.close()
    df['blocked'] = df.status.apply(lambda x: 0 if x in [2,3] else 1)
    return df

def update_gravity(df):
    db_connect = create_engine('sqlite:////etc/pihole/gravity.db', connect_args={'timeout': 15})
    connection = db_connect.raw_connection()
    df.to_sql('domainlist', con=connection,if_exists='replace',index=None)
    connection.close()
    
def create_dframe(domains,timestamp):
    new_dataframe = pd.DataFrame({'id':len(domains)*[1],'type':len(domains)*[1],'domain':domains,
                                  'enabled':len(domains)*[1],'date_added':len(domains)*[timestamp],
                                  'date_modified':len(domains)*[timestamp],
                                  'comment':len(domains)*['Added by Photon Sphere.']})
    return new_dataframe
    
def load_tokenizer():
    """Deprecated"""
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def load_sentpiece():
    """Deprecated"""
    serialized_model_proto = tensorflow.gfile.GFile('./sentpiece/m.model', 'rb').read()
    sp = spm.SentencePieceProcessor()
    sp.load_from_serialized_proto(serialized_model_proto)
    return sp

def load_yttm():
    model_path = './yttm_model/yttm_ads.model'
    bpe = yttm.BPE(model=model_path)
    return bpe
        
def parse_data(df):
    """Deprecated due to no use."""
    df_ = df.copy()
    df_['domain'] = df_['domain'].apply(lambda x: x.replace('.',' ').replace('-',' '))
#    format_df = df.groupby('timestamp').sum().reset_index()
#    format_df['domain'] = df.groupby('timestamp')['domain'].apply(lambda x: '|'.join(x)).reset_index()['domain']
#    format_df['domain_list'] = df.groupby('timestamp')['domain'].apply(lambda x: ','.join(x)).reset_index()['domain']
#    format_df['mask_count'] = df.groupby('timestamp')['domain'].apply(lambda x: len(x)).reset_index()['domain']
#    format_df['domain'] = format_df['domain'].apply(lambda x: ' '.join(x.split('.')))
#    format_df['blocked_chain'] = df.groupby('timestamp')['blocked'].apply(lambda x: ' '.join([str(i) for i in x])).reset_index()['blocked']
    return df_

def prep_data(df,timestamps,tokenizer=None):
    encoded_queries = pad_sequences(tokenizer.encode(list(df.loc[df.timestamp>timestamps[1]].domain.values), output_type=yttm.OutputType.ID),30,padding='post')
    df_neg = df.loc[(df.blocked==1)&(df.timestamp<timestamps[1])].domain.values
    df_anchors = df.loc[(df.blocked==0)&(df.timestamp<timestamps[1])].domain.values
    df_pos = df.loc[(df.blocked==0)&(df.timestamp<timestamps[1])].domain.values
    min_len = np.min([len(encoded_queries),len(df_neg),len(df_anchors)])
    df_neg = df_neg[:min_len]
    df_pos = df_pos[-min_len:]
    df_anchors = df_anchors[:min_len]
    encoded_neg = pad_sequences(tokenizer.encode(list(df_neg), output_type=yttm.OutputType.ID),30,padding='post')
    encoded_anchors = pad_sequences(tokenizer.encode(list(df_anchors), output_type=yttm.OutputType.ID),30,padding='post')
    encoded_pos = pad_sequences(tokenizer.encode(list(df_pos), output_type=yttm.OutputType.ID),30,padding='post')
    return encoded_queries,encoded_pos,encoded_neg,encoded_anchors

def run_all(tokenizer=None,timestamp=None):
    if not tokenizer:
        tokenizer = load_tokenizer()
    buffer_timestamp = str(int(timestamp)-1000) 
    dframe = load_query_list(timestamp=buffer_timestamp)
    new_entries = np.sum(dframe.timestamp>timestamp)
    i = 0
    while new_entries < 1:
        if i == 0:
            print('Awaiting additional matter...')
        time.sleep(2)
        dframe = load_query_list(timestamp=buffer_timestamp)
        new_entries = np.sum(dframe.timestamp>timestamp)
        i+=1
    most_recent_timestamp = dframe.timestamp.iloc[-1]
#    parsed_dframe = parse_data(dframe)
    token_queries,token_pos,token_neg,token_anchor = prep_data(dframe,timestamps=[buffer_timestamp,timestamp],tokenizer=tokenizer)
    dframe = dframe.loc[dframe.timestamp > timestamp].reset_index()
    return token_queries,token_pos,token_neg,token_anchor,dframe,most_recent_timestamp

def triplet_loss(true,pred):
    M = 1.
    loss = tf.reduce_mean(tf.maximum(pred[:,0]**2 - pred[:,1]**2 + M,0))
    return loss

def quadruplet_loss(true,pred):
    """Modified quadruplet loss with online semi-hard negative mining."""
    m1 = 1.
    m2 = .5
    mask1 = tf.cast((pred[:,3] < pred[:,0] + m1) & (pred[:,1] < pred[:,2] + m2),tf.float32)
    loss = tf.reduce_sum(tf.maximum(tf.multiply(pred[:,0]**2 - pred[:,1]**2 + m1,mask1),0)) +\
               tf.reduce_sum(tf.maximum(tf.multiply(pred[:,2]**2 - pred[:,3]**2 + m2,mask1),0))
    return loss

def custom_acc(true,pred):
    pred_compare = tf.where(pred[:,0] < pred[:,1],True,False) & tf.where(pred[:,2] < pred[:,1],True,False)
    total = tf.keras.backend.sum(tf.cast(pred_compare,tf.float32))
    return total/tf.cast(tf.size(pred_compare),tf.float32)

def load_model():
    model = tf.keras.models.load_model('./models/siamese_metric_quadruplet_loss_v1.h5',custom_objects={'quadruplet_loss':quadruplet_loss,'custom_acc':custom_acc})
    return model

def online_learn(learner,ref,eps=0.1):
    ref_labels = np.where(ref > 0.5, 1, 0)
    print('Models diverge: {0}'.format(np.sum(learner_labels==ref_labels)/ref_labels.size < 1))
    if np.random.rand() > eps:
        candidate_labels = np.where(ref > 0.5, 1, 0)
    else:
        print('Epsilon!')
        candidates = np.argmin(masks[test][i:i+1])
        candidate_labels = make_multilabel(' '.join([str(int(i)) for i in np.random.randint(0,2,size=(candidates))])).reshape(1,-1)
    return candidate_labels
        
if __name__=='__main__':
    print('Success.')
#    df = load_gravity(table='domainlist')
#    test_df = pd.DataFrame({'id':[1],'type':[1],'domain':'old.reddit.com','enabled':[1],'date_added':1600316117,
#                            'date_modified':1600316117,'comment':'just a test domain'})
#    update_gravity(pd.concat([df,test_df]))
