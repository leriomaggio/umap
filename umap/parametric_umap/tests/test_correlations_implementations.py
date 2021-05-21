import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from itertools import combinations_with_replacement

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
    # PyTorch dependency is satisfied - so we are safe to import functions from stats
    from umap.parametric_umap import correlation, covariance


tf_only = pytest.mark.skipif(
    not IMPORT_TF, reason="TensorFlow is required to run this test."
)
torch_only = pytest.mark.skipif(
    not IMPORT_TORCH, reason="PyTorch is required to run this test."
)
tf_and_torch_only = pytest.mark.skipif(
    not IMPORT_TORCH or not IMPORT_TF,
    reason="PyTorch and TensorFlow Dependency are both required to run this test",
)

# ------------------------
# Test COVARIANCE function
# ------------------------
@tf_and_torch_only
@pytest.mark.parametrize(
    "tf_tensors", np.arange(10, 200, step=10), indirect=["tf_tensors"]
)
def test_covariance_correct_results_covariance_matrix(tf_tensors):
    x_tf, y_tf = tf_tensors
    assert x_tf.shape == (100, 2, 3)
    assert y_tf.shape == (100, 2, 3)

    # Specifying both sample_axis and event_axis, the covariance matrix
    # will be returned
    tf_cov_matrix = tfp.stats.covariance(
        x_tf, y_tf, sample_axis=0, event_axis=-1, keepdims=False
    )

    x_th = torch.FloatTensor(x_tf.numpy())
    y_th = torch.FloatTensor(y_tf.numpy())
    th_cov_matrix = covariance(x_th, y_th, sample_axis=0, event_axis=-1, keepdims=False)

    assert tf_cov_matrix.numpy().shape == th_cov_matrix.numpy().shape
    assert th_cov_matrix.shape == (2, 3, 3)
    assert_array_almost_equal(
        tf_cov_matrix.numpy().astype(np.float32),
        th_cov_matrix.numpy().astype(np.float32),
        decimal=6,
    )


@torch_only
@pytest.mark.parametrize(
    "parameter_combinations",
    combinations_with_replacement((None, "numpy", "torch"), r=2),
)
def test_covariance_input_tensor_params_validation(
    torch_tensor, numpy_array, parameter_combinations
):
    parameters_map = {None: None, "numpy": numpy_array, "torch": torch_tensor}
    x_par, y_par = parameter_combinations
    x_par = parameters_map[x_par]
    y_par = parameters_map[y_par]

    if x_par is None:
        with pytest.raises(ValueError):
            _ = covariance(x=x_par)
    else:
        assert x_par.shape == (100, 2, 3), "Input X tensor shape mismatch"
        if y_par is not None:
            assert y_par.shape == (100, 2, 3)
        cov = covariance(x=x_par, y=y_par)
        assert cov.shape == (2, 3, 3), "Covariance Matrix shape mismatch!"


# try:
#     cov_th = covariance(x_torch_tensor)
#     assert True
# except RuntimeError:
#     assert False, "an exception is raised passing x as ndarray"
#
# # verify that the same results is returned
# assert_array_equal(cov_np.numpy(), cov_th.numpy())


@torch_only
def test_covariance_matches_torch_variance_when_no_event_is_specified(torch_tensor):
    cov_th = covariance(torch_tensor, sample_axis=0, event_axis=None)
    assert cov_th.shape == torch_tensor.shape[1:]
    torch_var = torch.var(torch_tensor, unbiased=False, dim=0)
    assert_almost_equal(cov_th.numpy(), torch_var.numpy(), decimal=6)


# ------------------------
# Test CORRELATION function
# ------------------------
@pytest.mark.skip()
@pytest.mark.parametrize("seed", np.arange(10, 200, step=10))
def test_correlation_tensorflow_torch(seed):
    tf.random.set_seed(seed=seed)
    x_tf = tf.random.normal(shape=(100, 2, 3))
    y_tf = tf.random.normal(shape=(100, 2, 3))
    tf_cov_matrix = tfp.stats.correlation(
        x_tf, y_tf, sample_axis=0, event_axis=-1, keepdims=False
    )

    x_th = torch.FloatTensor(x_tf.numpy())
    y_th = torch.FloatTensor(y_tf.numpy())
    th_cov_matrix = correlation(
        x_th, y_th, sample_axis=0, event_axis=-1, keepdims=False
    )

    assert tf_cov_matrix.numpy().shape == th_cov_matrix.numpy().shape
    assert_almost_equal(tf_cov_matrix.numpy(), th_cov_matrix.numpy(), decimal=6)


# @pytest.mark.parametrize("seed", np.arange(10, 200, step=10))
# def test_correlation_tensorflow_numpy(seed):
#     x_tf = tf.random.normal(shape=(100, 2, 3), seed=seed)
#     y_tf = tf.random.normal(shape=(100, 2, 3), seed=seed + 3)
#     tf_corr = tfp.stats.correlation(x_tf, y_tf, sample_axis=0, event_axis=None)
#
#     x_np = x_tf.numpy()
#     y_np = y_tf.numpy()
#     np_corr = correlation_np(x_np, y_np, sample_index=0)
#     assert_almost_equal(tf_corr.numpy(), np_corr, decimal=3)
#
#
# @pytest.mark.parametrize("seed", np.arange(10, 200, step=10))
# def test_correlation_tensorflow_torch(seed):
#     x_tf = tf.random.normal(shape=(100, 2, 3), seed=seed)
#     y_tf = tf.random.normal(shape=(100, 2, 3), seed=seed + 3)
#     tf_corr = tfp.stats.correlation(x_tf, y_tf, sample_axis=0, event_axis=None)
#
#     x_th = torch.FloatTensor(x_tf.numpy())
#     y_th = torch.FloatTensor(y_tf.numpy())
#     th_corr = correlation(x_th, y_th, sample_index=0)
#     assert_almost_equal(tf_corr.numpy(), th_corr.numpy(), decimal=2)
#
#
# @pytest.mark.parametrize("seed", np.arange(10, 200, step=10))
# def test_correlation_numpy_torch(seed):
#     x_tf = tf.random.normal(shape=(100, 2, 3), seed=seed)
#     y_tf = tf.random.normal(shape=(100, 2, 3), seed=seed + 3)
#
#     x_np = x_tf.numpy()
#     y_np = y_tf.numpy()
#     np_corr = correlation_np(x_np, y_np, sample_index=0)
#
#     x_th = torch.FloatTensor(x_tf.numpy())
#     y_th = torch.FloatTensor(y_tf.numpy())
#     th_corr = correlation(x_th, y_th, sample_index=0)
#     assert_almost_equal(np_corr, th_corr.numpy(), decimal=2)
