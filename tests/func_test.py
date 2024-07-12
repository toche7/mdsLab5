import ex4_1 as hw1
import ex4_2 as hw2
import ex4_3 as hw3
import numpy as np
def test_hwfunc1():
    model1 = hw1.homework()
    assert np.allclose(model1, 1.0, atol=1e-2)


def test_hwfunc2():
    model1 = hw2.homework()
    assert np.allclose(model1, 0.9933333333333333, atol=1e-4)

def test_hwfunc3():
    model1 = hw3.homework()
    assert np.allclose(model1, 1.0, atol=1e-4)

