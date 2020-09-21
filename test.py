import functions as fn


def test_mask():
    mask = fn.make_mask(10)
    assert sum(mask) == 10
