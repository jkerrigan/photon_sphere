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
    y = np.zeros(100)
    print(x)
    y[:x] = 1
    return y

def make_multilabel(x):
    x_ = list(map(lambda xx: int(xx),x.split(' ')))
    y = np.zeros(100)
    y[range(len(x_))] = x_
    return y

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
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def load_sentpiece():
    serialized_model_proto = tensorflow.gfile.GFile('./sentpiece/m.model', 'rb').read()
    sp = spm.SentencePieceProcessor()
    sp.load_from_serialized_proto(serialized_model_proto)
    return sp

def load_yttm():
    model_path = './yttm_model/yttm_ads.model'
    bpe = yttm.BPE(model=model_path)
    return bpe
        
def parse_data(df):
    format_df = df.groupby('timestamp').sum().reset_index()
    format_df['domain'] = df.groupby('timestamp')['domain'].apply(lambda x: '|'.join(x)).reset_index()['domain']
    format_df['domain_list'] = df.groupby('timestamp')['domain'].apply(lambda x: ','.join(x)).reset_index()['domain']
    format_df['mask_count'] = df.groupby('timestamp')['domain'].apply(lambda x: len(x)).reset_index()['domain']
    #format_df['domain'] = format_df['domain'].apply(lambda x: ' '.join(x.split('.')))
    format_df['blocked_chain'] = df.groupby('timestamp')['blocked'].apply(lambda x: ' '.join([str(i) for i in x])).reset_index()['blocked']
    return format_df

def prep_data(df,tokenizer=None):
    print(df.domain)
    encoded_docs = tokenizer.encode(list(df.domain.values), output_type=yttm.OutputType.ID) #change null space to |
    encoded_docs = pad_sequences(encoded_docs,100,padding='post')
    masks = make_mask(df.mask_count.values[0]).reshape(1,-1)
    labels = make_multilabel(df.blocked_chain.values[0]).reshape(1,-1)
    return encoded_docs,masks,labels

def run_all(tokenizer=None,timestamp=None):
    if not tokenizer:
        tokenizer = load_tokenizer()
    dframe = load_query_list(timestamp=timestamp)
    i = 0
    while len(dframe) < 1:
        if i == 0:
            print('Awaiting additional matter...')
        time.sleep(2)
        dframe = load_query_list(timestamp=timestamp)
        i+=1
    most_recent_timestamp = dframe.timestamp.iloc[-1]
    dframe['timestamp'] = dframe.timestamp.round(-1)
    parsed_dframe = parse_data(dframe).iloc[-1] # pull a single timestamp
    token_data,mask_data,labels = prep_data(parsed_dframe,tokenizer=tokenizer)
    return token_data,mask_data,labels,parsed_dframe,most_recent_timestamp

def load_model():
    model = tf.keras.models.load_model('./models/dns_anhilator.h5')
    return model

def online_learn(learner,ref,eps=0.1):
    learner_entropy = -np.sum(learner*np.log(learner+1e-5*np.random.randn(np.shape(learner)[1])))
    ref_entropy = -np.sum(ref*np.log(ref+1e-5*np.random.randn(np.shape(ref)[1])))

    learner_labels = np.where(learner > 0.5, 1, 0)
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
    print('Test is a success.')
#    df = load_gravity(table='domainlist')
#    test_df = pd.DataFrame({'id':[1],'type':[1],'domain':'old.reddit.com','enabled':[1],'date_added':1600316117,
#                            'date_modified':1600316117,'comment':'just a test domain'})
#    update_gravity(pd.concat([df,test_df]))
#id                                  1
#type                                1
#domain           gateway.dunk.com
#enabled                             1
#date_added                 1600316117
#date_modified              1600316117
#comment          Added from Query Log
