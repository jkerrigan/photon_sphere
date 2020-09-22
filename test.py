import functions as fn


def test_mask():
    mask = fn.make_mask(10)
    assert sum(mask) == 10

def test_data_prep():
    df = pd.DataFrame({'domain':['test.domain1','test.domain2 test.domain3'],'blocked_chain':['1','0 1'],'mask_count'=[1,2]})
    tok_model = fn.load_sentpiece()
    tok_result = fn.prep_data(df,tokenizer=tok_model)
    assert len(tok_result) > 0
