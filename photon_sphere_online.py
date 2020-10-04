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
online_learning = True

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
        queries,pos_samples,neg_samples,anchor_samples,parsed_df,max_timestamp = fn.run_all(tokenizer=tokenizer,timestamp=max_timestamp)
        if i == 0:
            i =+ 1
            continue
        print('{0} pieces of matter entering the photon sphere..'.format(len(parsed_df)))
#        print(tokens.shape)
        print('queries shape',np.shape(queries))
        print('anchor shape',np.shape(anchor_samples))
        domain_lists = parsed_df.domain.values
        #pred_probs = model.predict([queries,neg_samples,anchor_samples])
        pred_probs = np.array(list(map(lambda x:fn.multi_pred(model,x,neg_samples,anchor_samples),queries)))
        ref_pred_probs = np.array(list(map(lambda x:fn.multi_pred(ref_model,x,neg_samples,anchor_samples),queries)))
        #print(pred_probs)
        #ref_pred_probs = ref_model.predict([queries,neg_samples,anchor_samples])
        #predicted = np.where(pred_probs>0.5,1,0).astype(bool)
        #ref_predicted = np.where(ref_pred_probs>0.5,1,0).astype(bool)
        bad_domains = parsed_df.loc[pred_probs>0.8].domain.values
        #domain_lists = parsed_df.domain_list.split(',')
        #bad_domains = np.array(domain_lists)[predicted[0,:len(domain_lists)]]
        #bad_domains = [i for i in bad_domains if len(i)>0]
        bad_domains = [i for i in bad_domains if (i not in gravity) and (i not in df['domain'].values) and (i not in bad_domains_all)]
        bad_domains_all.extend(bad_domains)
        print(bad_domains)
        print(pred_probs[pred_probs>0.8])
        if (len(bad_domains) > 0) and (i != 0):
            print('Ad domains being vaporized in the photon sphere...')
            with open('photonSphere_list.txt','a') as f:
                [f.write('\n{0}'.format(i)) for i in bad_domains]
#            os.system('pihole -b {0}'.format(bad_domains))
            #bad_df = fn.create_dframe(bad_domains,max_timestamp)
            #df = pd.concat([df,bad_df])
            #fn.update_gravity(df)

            #        online_labels = fn.online_learn(pred_probs,ref_pred_probs,eps=epsilon)
#        model.fit([tokens,masks],online_labels)
        if online_learning:
            print('Teaching this thing some new tricks...')
            print(np.shape(pos_samples),np.shape(anchor_samples),np.shape(neg_samples))
            for j,prob in enumerate(pred_probs):
                if (prob > 0.55) & (prob < 0.9):
                    model.fit([pos_samples[j].reshape(1,-1),queries[j].reshape(1,-1),anchor_samples[j].reshape(1,-1)],np.zeros(1),epochs=1)
                elif (prob > 0.1) & (prob < 0.45):
                    model.fit([queries[j].reshape(1,-1),neg_samples[j].reshape(1,-1),anchor_samples[j].reshape(1,-1)],np.zeros(1),epochs=1)
        time.sleep(2)
        i+=1
        if logging:
            predicted = pred_probs > 0.9#pred_probs[:,0] > pred_probs[:,1]
            ref_predicted = ref_pred_probs > 0.9#ref_pred_probs[:,0] > ref_pred_probs[:,1]
            diverged = np.sum(predicted==ref_predicted)/predicted.size < 1
            logger('\n{0} : {1} : {2} : {3}'.format(time.time(),','.join(domain_lists),np.array(predicted).astype(int),diverged))
