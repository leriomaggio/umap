import pytest
from sklearn.datasets import make_moons


@pytest.fixture(scope="function")
def moon_dataset():
    X, _ = make_moons(200)
    return X
