import matplotlib.pyplot as pl
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import functions as fn
import time
import pandas as pd
import os


logging = True

def logger(metadata):
    with open('online.log','a') as f:
        f.write(metadata)
if logging:
    logger('timestamp(s) : dns bucket : predictions : diverged')

if __name__=='__main__':

    gravity = fn.load_gravity()['domain'].values
    tokenizer = fn.load_yttm()
    max_timestamp = 1600397395 # arbitrary recent timestamp

    model = fn.load_model() #model that learns online
    ref_model = tf.keras.models.clone_model(model) #reference model
    ref_model.set_weights(model.get_weights())

    model.optimizer.lr.assign(1e-4)
    df = fn.load_gravity(table='domainlist')
    epsilon = 0.1
    i = 0
    bad_domains_all = []
    while True:
        tokens,masks,labels,parsed_df,max_timestamp = fn.run_all(tokenizer=tokenizer,timestamp=max_timestamp)
        if i == 0:
            i =+ 1
            continue
        print('{0} pieces of matter entering the photon sphere..'.format(parsed_df.mask_count))
        print(tokens.shape)
        pred_probs = model.predict([tokens,masks])
        ref_pred_probs = ref_model.predict([tokens,masks])
        predicted = np.where(pred_probs>0.5,1,0).astype(bool)
        ref_predicted = np.where(ref_pred_probs>0.5,1,0).astype(bool)
        
        domain_lists = parsed_df.domain_list.split(',')
        bad_domains = np.array(domain_lists)[predicted[0,:len(domain_lists)]]
        #bad_domains = [i for i in bad_domains if len(i)>0]
        bad_domains = [i for i in bad_domains if i not in gravity and i not in df['domain'].values and i not in bad_domains_all]
        bad_domains_all.extend(bad_domains)
        print(bad_domains)
        if (len(bad_domains) > 0) and (i != 0):
            print('Ad domains being vaporized in the photon sphere...')
            with open('photonSphere_list.txt','a') as f:
                [f.write('\n{0}'.format(i)) for i in bad_domains]
#            os.system('pihole -b {0}'.format(bad_domains))
            #bad_df = fn.create_dframe(bad_domains,max_timestamp)
            #df = pd.concat([df,bad_df])
            #fn.update_gravity(df)
        online_labels = fn.online_learn(pred_probs,ref_pred_probs,eps=epsilon)
        model.fit([tokens,masks],online_labels)
        time.sleep(2)
        i+=1
        if logging:
            diverged = np.sum(predicted==ref_predicted)/predicted.size < 1
            logger('\n{0} : {1} : {2} : {3}'.format(time.time(),','.join(domain_lists),predicted[0,:len(domain_lists)].astype(int),diverged))