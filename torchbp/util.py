import torch
from torch import Tensor
from math import pi

def diff(x, dim=-1, same_size=False):
    """
    ``np.diff`` implemented in torch.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int
        Dimension.
    same_size : bool
        Pad output to same size as input.

    Returns
    ----------
    d : Tensor
        Difference tensor.
    """
    if dim != -1:
        raise NotImplementedError("Only dim=-1 is implemente")
    if same_size:
        return torch.nn.functional.pad(x[...,1:]-x[...,:-1], (1,0))
    else:
        return x[...,1:]-x[...,:-1]

def unwrap(phi, dim=-1):
    """
    ``np.unwrap`` implemented in torch.

    Parameters
    ----------
    phi : Tensor
        Input tensor.
    dim : int
        Dimension.

    Returns
    ----------
    phi : Tensor
        Unwrapped tensor.
    """
    if dim != -1:
        raise NotImplementedError("Only dim=-1 is implemente")
    dphi = diff(phi, same_size=True)
    dphi_m = ((dphi+torch.pi) % (2 * torch.pi)) - torch.pi
    dphi_m[(dphi_m==-torch.pi)&(dphi>0)] = torch.pi
    phi_adj = dphi_m-dphi
    phi_adj[dphi.abs()<torch.pi] = 0
    return phi + phi_adj.cumsum(dim)

def quad_interp(a: Tensor, v: int):
    """
    Quadractic peak interpolation.
    Useful for FFT peak interpolation.

    Parameters
    ----------
    a : Tensor
        Input tensor.
    v : int
        Peak index.

    Returns
    ----------
    f : float
        Estimated fractional peak index.
    """
    a1 = a[(v-1) % len(a)]
    a2 = a[v % len(a)]
    a3 = a[(v+1) % len(a)]
    return 0.5 * (a1 - a3) / (a1 - 2*a2 + a3)

def find_image_shift_1d(x: Tensor, y: Tensor, dim: int=-1):
    """
    Find shift between images that maximizes correlation.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    y : int
        Input tensor. Should have same shape as x.

    Returns
    ----------
    c : int
        Estimated shift.
    """
    if x.shape != y.shape:
        raise ValueError("Input shapes should be identical")
    if dim < 0:
        dim = x.dim() + dim
    fx = torch.fft.fft(x, dim=dim)
    fy = torch.fft.fft(y, dim=dim)
    c = (fx*fy.conj()) / (torch.abs(fx) * torch.abs(fy))
    other_dims = [i for i in range(x.dim()) if i != dim]
    c = torch.abs(torch.fft.ifft(c, dim=dim))
    if len(other_dims) > 0:
        c = torch.mean(c, dim=other_dims)
    return torch.argmax(c)

def fft_peak_1d(x: Tensor, dim: int=-1, fractional: bool=True):
    """
    Find fractional peak of ``abs(fft(x))``.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int
        Dimension to calculate peak.
    fractional : bool
        Estimate peak location with fractional index accuracy.

    Returns
    ----------
    a : int or float
        Estimated peak index.
    """
    fx = torch.abs(torch.fft.fft(x, dim=dim))
    a = torch.argmax(fx)
    if fractional:
        a = a + quad_interp(fx, a)
    l = x.shape[dim]
    if a > l//2:
        a = l - a
    return a

def entropy(x: Tensor):
    """
    Calculates entropy of:

    ``-sum(y*log(y))``

    where ``y = abs(x) / sum(abs(x))``.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    ----------
    entropy : Tensor
        Calculated entropy of the input.
    """
    ax = torch.abs(x)
    ax /= torch.sum(ax)
    return -torch.sum(torch.xlogy(ax, ax))

def shift_spectrum(x: Tensor, dim=-1):
    """
    Equivalent to: ``fft(ifftshift(ifft(x, dim), dim), dim)``,
    but avoids calculating FFTs.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    ----------
    y : Tensor
        Shifted tensor.
    """
    if dim != -1:
        raise NotImplementedError("dim should be -1")
    shape = [1] * len(x.shape)
    shape[dim] = x.shape[dim]
    c = torch.ones(shape, dtype=torch.float32, device=x.device)
    c[...,1::2] = -1
    return x * c

def generate_fmcw_data(target_pos: Tensor, target_rcs: Tensor, pos : Tensor, fc: float, bw: float,
    tsweep: float, fs: float):
    """
    Generate FMCW radar time-domain IF signal.

    Parameters
    ----------
    target_pos : Tensor
        [ntargets, 3] tensor of target XYZ positions.
    target_rcs : Tensor
        [ntargets, 1] tensor of target reflectivity.
    pos : Tensor
        [nsweeps, 3] tensor of platform positions.
    fc : float
        RF center frequency in Hz.
    bw : float
        RF bandwidth in Hz.
    tsweep : float
        Length of one sweep in seconds.
    fs : float
        Sampling frequency in Hz.

    Returns
    ----------
    data : Tensor
        [nsweeps, nsamples] measurement data.
    """
    if pos.dim() != 2:
        raise ValueError("pos tensor should have 2 dimensions")
    if pos.shape[1] != 3:
        raise ValueError("positions should be 3 dimensional")
    npos = pos.shape[0]
    nsamples = int(fs * tsweep)

    device = pos.device
    data = torch.zeros((npos, nsamples), dtype=torch.complex64, device=device)
    t = torch.arange(nsamples, dtype=torch.float32, device=device) / fs
    k = bw / tsweep

    c0 = 299792458

    t = t[None, :]
    for e, target in enumerate(target_pos):
        d = torch.linalg.vector_norm(pos - target[None,:], dim=-1)[:, None]
        tau = 2*d/c0
        data += (target_rcs[e]/d**4) * torch.exp(-1j*2*pi*(fc*tau - k*tau*t + 0.5*k*tau**2))
    return data

def make_polar_grid(r0: float, r1: float, nr: int, ntheta: int, theta_limit: int=1):
    """
    Generate polar grid dict in format understood by other polar functions.

    Parameters
    ----------
    r0 : float
        Minimum range in m.
    r1 : float
        Maximum range in m.
    nr : float
        Number of range points.
    ntheta : float
        Number of azimuth points.
    theta_limit : float
        Theta axis limits, symmetrical around zero.
        Units are sin of angle (0 to 1 valid range).
        Default is 1.

    Returns
    ----------
    grid_polar : dict
        Polar grid dict.
    """
    grid_polar = {"r": (r0, r1), "theta": (-theta_limit, theta_limit), "nr": nr, "ntheta": ntheta}
    return grid_polar
