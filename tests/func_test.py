import ex1 as hw1
import numpy as np
def test_hwfunc1():
    result = hw1.homework()
    assert result[0].best_score_ >= 0.92
    assert result[1].best_score_ >= 0.92
    assert result[2].best_score_ >= 0.92


