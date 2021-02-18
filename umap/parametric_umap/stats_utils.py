import numpy as np
from typing import Union, List, Tuple

try:
    import torch
except ImportError:
    pass


def _make_positive_axis(axis: Union[float, List[float]], ndims: int) -> Tuple[int]:
    """Rectify possibly negative """
    axis = torch.Tensor([axis]).flatten()
    axis = torch.where(axis >= 0, axis, ndims + axis)
    axis = axis.tolist()
    return tuple(map(int, axis))


def covariance(
    x, y=None, sample_axis: int = 0, event_axis: int = -1, keepdims: bool = False
) -> torch.Tensor:
    """PyTorch implementation of the tfp.stats.covariance function in TF"""

    # TODO: test with either numpy.array and torch.Tensor as x
    try:
        x = torch.from_numpy(x)
    except TypeError:  # x is already a Tensor object
        pass
    finally:
        x = x - torch.mean(x, dim=sample_axis, keepdim=True)

    # TODO: test with either y=None or not and either types
    if y is None:
        y = x
    else:
        try:
            y = torch.from_numpy(y)
        except TypeError:  # y is not None and already a torch Tensor
            pass
        finally:
            y = y - torch.mean(y, dim=sample_axis, keepdim=True)

    # TODO: test no event axis
    if event_axis is None:
        return torch.mean(x * torch.conj(y), dim=sample_axis, keepdim=keepdims)

    # TODO: test exceptions raised with NO sample axis
    if sample_axis is None:
        raise ValueError(
            "sample_axis was None, which means all axis hold events, and this overlaps with event_axis ({})".format(
                event_axis
            )
        )

    event_axis = _make_positive_axis(event_axis, x.ndim)
    sample_axis = _make_positive_axis(sample_axis, x.ndim)
    # TODO: test raised exception event_axis and sample_axis overlap
    if set(event_axis).intersection(sample_axis):
        raise ValueError(
            "sample_axis ({}) and event_axis ({}) overlapped".format(
                sample_axis, event_axis
            )
        )
    # TODO: test with non contiguous event_axis
    if (np.diff(np.array(sorted(event_axis))) > 1).any():
        raise ValueError("event_axis must be contiguous. Found: {}".format(event_axis))

    batch_axis = list(sorted(set(range(x.ndim)).difference(sample_axis + event_axis)))

    event_axis = torch.IntTensor(event_axis)
    sample_axis = torch.IntTensor(sample_axis)
    batch_axis = torch.IntTensor(batch_axis)
    perm_for_xy = torch.cat((batch_axis, event_axis, sample_axis), dim=0).tolist()
    x_permed = x.permute(*perm_for_xy)
    y_permed = y.permute(*perm_for_xy)

    batch_ndims = np.array(batch_axis.ndim, dtype=np.int32)
    batch_shape = list(x_permed.shape[:batch_ndims])
    event_ndims = np.array(event_axis.ndim, dtype=np.int32)
    event_shape = list(x_permed.shape[batch_ndims : batch_ndims + event_ndims])
    sample_ndims = sample_axis.ndim
    sample_shape = list(x_permed.shape[batch_ndims + event_ndims :])

    n_samples = np.prod(np.asarray(sample_shape))
    n_events = np.prod(np.asarray(event_shape))

    # TODO: Test with more than one sample axis
    sample_axis_flat = [n_samples] if len(sample_shape) > 1 else sample_shape
    # TODO: test with more than one event axis
    event_axis_flat = [n_events] if len(event_shape) > 1 else event_shape

    flat_perm_shape = batch_shape + event_axis_flat + sample_axis_flat
    x_permed = x_permed.reshape(flat_perm_shape)
    y_permed = y_permed.reshape(flat_perm_shape)

    # After matmul, cov.shape = batch_shape + [n_events, n_events]
    cov = (x_permed @ torch.conj(y_permed).transpose(-1, -2)) / n_samples
    cov = cov.reshape((batch_shape + [n_events ** 2] + ([1] * sample_ndims)))

    # Permuting by the argsort inverts the permutation, making
    # cov.shape have ones in the position where there were samples, and
    # [n_events * n_events] in the event position.
    perm_for_xy_tensor = torch.LongTensor(perm_for_xy)
    cov = cov.permute(*perm_for_xy_tensor[perm_for_xy_tensor])

    # Now expand event_shape**2 into event_shape + event_shape.
    # We here use (for the first time) the fact that we require event_axis to be
    # contiguous.
    e_start = event_axis[0]
    e_len = 1 + event_axis[-1] - event_axis[0]
    cov = cov.reshape(
        (
            list(cov.shape[:e_start])
            + event_shape
            + event_shape
            + list(cov.shape[e_start + e_len :])
        )
    )

    # TODO: test for values in keepdims
    if not keepdims:
        sample_axis_np = sample_axis.numpy()
        squeeze_axis = np.where(
            sample_axis_np < e_start.numpy(),
            sample_axis_np,
            sample_axis_np + e_len.numpy(),
        )
        if not squeeze_axis:
            cov = torch.squeeze(cov)
        else:
            keep_axis = np.setdiff1d(np.arange(cov.ndim), squeeze_axis)
            cov_shape = torch.IntTensor(list(cov.shape))
            keep_dims = torch.gather(
                cov_shape, dim=0, index=torch.from_numpy(keep_axis)
            )
            cov = cov.reshape(keep_dims.tolist())
    return cov


def correlation(
    x, y=None, sample_axis: int = 0, event_axis: int = -1, keepdims: bool = False
):
    """"""
    x /= torch.std(x, unbiased=False, dim=sample_axis, keepdim=True)
    if y is not None:
        y /= torch.std(y, unbiased=False, dim=sample_axis, keepdim=True)
    return covariance(
        x, y, sample_axis=sample_axis, event_axis=event_axis, keepdims=keepdims
    )
