import functions as fn
import pandas as pd

def test_mask():
    mask = fn.make_mask(10)
    assert sum(mask) == 10 and len(mask)==100

def test_data_prep():
    df = pd.DataFrame({'domain':'test.domain2 test.domain3','blocked_chain':'0 1','mask_count':2},index=[0])
    tok_model = fn.load_yttm()
    tok_result = fn.prep_data(df,tokenizer=tok_model)
    print(tok_result)
    assert len(tok_result) > 0
