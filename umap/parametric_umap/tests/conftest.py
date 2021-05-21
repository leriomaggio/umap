import numpy as np
import pytest
from sklearn.datasets import make_moons

# Checking envs and dependency to skip fixtures, in case
try:
    import tensorflow as tf
    import tensorflow_probability as tfp
except ImportError:
    IMPORT_TF = False
else:
    IMPORT_TF = True

try:
    import torch
except ImportError:
    IMPORT_TORCH = False
else:
    IMPORT_TORCH = True

# --------
# Fixtures
# --------
@pytest.fixture(scope="function")
def random_seed(request):
    seed = request.param if hasattr(request, "param") else 123456  # NOT parametrised
    return seed


@pytest.fixture(scope="function")
def numpy_array(random_seed):
    rng = np.random.default_rng(seed=random_seed)
    return rng.uniform(size=(100, 2, 3))


@pytest.fixture(scope="function")
def torch_tensor(random_seed):
    if not IMPORT_TORCH:
        pytest.skip("PyTorch is required to initialise this test")
    torch.random.manual_seed(random_seed)
    return torch.rand(size=(100, 2, 3))


@pytest.fixture(scope="function")
def torch_tensors(random_seed):
    if not IMPORT_TORCH:
        pytest.skip("PyTorch is required to initialise this test")
    torch.random.manual_seed(random_seed)
    x = torch.rand(size=(100, 2, 3))
    y = torch.rand(size=(100, 2, 3))
    return x, y


@pytest.fixture(scope="function")
def tf_tensors(random_seed):
    if not IMPORT_TF:
        pytest.skip("TensorFlow is required to initialise this test")
    tf.random.set_seed(seed=random_seed)
    x = tf.random.normal(shape=(100, 2, 3))
    y = tf.random.normal(shape=(100, 2, 3))
    return x, y


@pytest.fixture(scope="function")
def moon_dataset():
    X, _ = make_moons(200)
    return X
