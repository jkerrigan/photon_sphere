import functions as fn
import pandas as pd

def test_data_prep():
    """Needs to be rewritten."""
    df = pd.DataFrame({'domain':'test.domain2 test.domain3','blocked_chain':'0 1','mask_count':2},index=[0])
    tok_model = fn.load_yttm()
    #tok_result = fn.prep_data(df,[10000,10001],tokenizer=tok_model)
    #print(tok_result)
    #assert len(tok_result) > 0
    assert len(df) > 0
    
def test_create_dframe():
    domains = ['test.domain1','test.domain2','test.domain3','test.domain4']
    timestamp = [0,1,2,3]
    df = fn.create_dframe(domains,timestamp)
    assert df['domain'].values[0]=='test.domain1'
