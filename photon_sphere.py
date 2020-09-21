import matplotlib.pyplot as pl
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import functions as fn
import time
import pandas as pd
import os

if __name__=='__main__':

    gravity = fn.load_gravity()['domain'].values
    tokenizer = fn.load_tokenizer()
    max_timestamp = 1600397395 # arbitrary recent timestamp
    model = fn.load_model()
    df = fn.load_gravity(table='domainlist')
    i = 0
    while True:
        tokens,masks,labels,parsed_df,max_timestamp = fn.run_all(tokenizer=tokenizer,timestamp=max_timestamp)
        print('{0} pieces of matter entering the photon sphere..'.format(len(tokens)))
        predicted = np.where(model.predict([tokens,masks])>0.5,1,0).astype(bool)
        domain_lists = list(map(lambda x: x.split(','),parsed_df.domain_list.values))
        bad_domains = list(map(lambda x: np.array(x[0])[x[1][:len(x[0])]],zip(domain_lists,predicted)))
        bad_domains = [i[0] for i in bad_domains if len(i)>0]
        bad_domains = [i for i in bad_domains if i not in gravity and i not in df['domain'].values]
        print(bad_domains)
        if (len(bad_domains) > 0) and (i != 0):
            print('Ad domains being vaporized in the photon sphere...')
            os.system('pihole -b {0}'.format(bad_domains))
            #bad_df = fn.create_dframe(bad_domains,max_timestamp)
            #df = pd.concat([df,bad_df])
            #fn.update_gravity(df)
        time.sleep(1)
        i+=1
    #    if (i%1000 == 0) and (i!=0):
    #        print('Flushing cache and restarting dns..')
    #        os.system('pihole restartdns')
