"""
These tests are included the `colored_noise.py` module, but only executed if
the module is opened as a notebook (through Jupytext).
We copy them here, without the plotting, so they can be run as a normal –
which is much easier for testing.
"""

import math
import numpy as np

from colored_noise_sc import ColoredNoise
from scityping import Serializable

## Validation of theoretical expressions

import itertools
import scipy.signal as signal
import holoviews as hv
import pint
from types import SimpleNamespace
ureg = pint.UnitRegistry()
ureg.formatter.default_format = "~P"

def tqdm(x, *args, **kwds): return x  # Remove progress bars when running with Jupyter Book

from tqdm.auto import tqdm

dims = SimpleNamespace(
    t  = hv.Dimension("t", label="time", unit="ms"),
    Δt = hv.Dimension("Δt", label="time lag", unit="ms"),
    ξ  = hv.Dimension("ξ"),
    ξ2 = hv.Dimension("ξ2", label="⟨ξ²⟩"),
    T  = hv.Dimension("T", label="realization length", unit="ms"),
    σ  = hv.Dimension("σ", label="noise strength", unit="√ms"),
    τ  = hv.Dimension("τ", label="correlation length", unit="ms"),
    ρ  = hv.Dimension("ρ", label="impulse density"),
    N  = hv.Dimension("N", label="# realizations")
)
#colors = hv.Cycle("Dark2").values    # This line requires loading a holoviews backend, so instead we explicitely list colors
colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']  # Dark2 colors

N = 1000
_lags = signal.correlation_lags(N, N)
norm_autocorr = hv.Curve(zip(_lags, signal.correlate(np.ones(N), np.ones(N))),
               kdims="lag (l)", vdims="C") 

n_realizations_shown = 10
seedseq = np.random.SeedSequence(6168912654954)
Tlst = [50.]
σlst = [0.33, 1., 9.]
τlst = [1., 5., 25.]
ρlst = [1, 5, 30, 200]
Nlst = [20]
exp_conds = list(itertools.product(Tlst, σlst, τlst, ρlst, Nlst))

frames_realizations = {}
frames_autocorr = {}
ms = ureg.ms

# The SciPy function `signal.correlate` (along with its companion `signal.correlation_lags` to compute the lag axis)
# is a convenient way to compute the autocorrelation. However before plotting the result, one must take care to
# normalize it correctly. Indeed, if $x$ is a discretized signal with $N$ time bins,
# and $C_k$ is its discretized correlation function at lag $k$, then the definition used by `correlate` is
# \begin{equation}
# C_k = \Braket{x_l x_{l+k}} = \sum_{l=0}^{N-1-k} x_l x_{l+k} \,.
# \end{equation}
# Note that the number of terms depends on $k$. We can see this clearly when computing the autocorrelation of the constant function $x_l = 1$:
# the result should be flat (albeit dependent on $N$), but instead we get a triangular function peaking at zero,  
# where the value on the $y$ axis is exactly the number of terms contributing to that lag.
# To avoid this artifical triangular decay, in the code we normalize the result by the number of terms contributing to each lag;
# in terms of a continuous time autocorrelation, this is equivalent to normalizing by the value at zero:
# \begin{equation}
# C^{\text{normed}}(s) = \frac{C(s)}{C(0)} \,.
# \end{equation}


experiment_iter = tqdm(exp_conds, "Exp. cond.")
for T, σ, τ, ρ, N in experiment_iter:
    if (T, σ, τ, ρ, N) in (frames_realizations.keys() & frames_autocorr.keys())  :
        continue
    
    noise = ColoredNoise(0, T, corr_time=τ, scale=σ, impulse_density=ρ, rng=seedseq)
    t_arr = np.linspace(noise.t_min, noise.t_max, int(10*T/noise.τ))

    ## Generate the realizations and compute their autocorrelation ##
    L = len(t_arr)
    Δt = np.diff(t_arr).mean()
    norm = signal.correlate(np.ones(L), np.ones(L), mode="same")  # Counts the number of sums which will contribute to each lag
    lags = signal.correlation_lags(L, L, mode="same") * Δt
    ξ_arr = np.empty((N, L))
    Cξ_arr = np.empty((N, L))
    for i, key in enumerate(tqdm(seedseq.spawn(N), "Seeds", leave=False)):
        _noise = noise.new(rng=key)
        ξ = np.fromiter((_noise(t) for t in t_arr), count=len(t_arr), dtype=float)
        ξ_arr[i] = ξ
        Cξ   = signal.correlate(ξ, ξ, mode="same") / norm
        Cξ_arr[i] = Cξ
    Cξ = Cξ_arr.mean(axis=0)

    ## Generator realization curves ##
    realization_samples = hv.Overlay([
        hv.Curve(zip(t_arr, _ξ), kdims=dims.t, vdims=dims.ξ, label="Single realization")
        for _ξ in ξ_arr[:n_realizations_shown]
    ])
    
    ## Generate autocorr curves ##
    autocorr_samples = hv.Overlay([
        hv.Curve(zip(lags, _Cξ), kdims=dims.Δt, vdims=dims.ξ2, label="Single realization")
        for _Cξ in Cξ_arr[:n_realizations_shown]]
    )
    avg =  hv.Curve(zip(lags, Cξ), kdims=dims.Δt, vdims=dims.ξ2, label=f"Average ({N} realizations)")
    target = hv.Curve(zip(lags, noise.autocorr(lags)), kdims=dims.Δt, vdims=dims.ξ2, label="Theoretical")
    
    ## Compute axis range so it is appropriate for mean and target autocorr – individual realizations may be well outside this range ##
    ymax = max(avg.range("ξ2")[1], target.range("ξ2")[0])
    ymin = min(avg.range("ξ2")[0], target.range("ξ2")[0])
    Δy = ymax-ymin
    ymax += 0.05*Δy
    ymin -= 0.05*Δy
    # Round ymin down, round ymax up
    p = math.floor(np.log10(ymax-ymin)) + 2  # +2: between 10 and 100 ticks in the range
    new_range = (round(math.floor(ymin * 10**p) / 10**p, p),
                 round(math.ceil (ymax * 10**p) / 10**p, p))


## Usage examples

# Scale with units

noise = ColoredNoise(t_min    = 0. *ureg.ms,
                     t_max    =10. *ureg.ms,
                     corr_time= 1. *ureg.ms,
                     scale    = 2.2*ureg.mV,
                     impulse_density=30,
                     rng=1337)
assert noise.Nbins == 10
expected_bin_edges = np.array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5., 
                               6.,  7., 8.,  9., 10., 11., 12., 13., 14., 15.])*ureg.ms
assert np.allclose(noise.bin_edges, expected_bin_edges)
noise(1.)

# Scalar output

noise = ColoredNoise(t_min    =   0. *ureg.ms,
                     t_max    =1000. *ureg.ms,
                     corr_time=   1. *ureg.ms,
                     scale    =   2.2,
                     impulse_density=30)
ξ = np.array([noise(t) for t in np.linspace(noise.t_min, noise.t_max, 1000)])
ξ.std(axis=0)

# 1d output

noise = ColoredNoise(t_min    =   0. *ureg.ms,
                     t_max    =1000. *ureg.ms,
                     corr_time=   1. *ureg.ms,
                     scale    =np.array([2.2, 1.1]),
                     impulse_density=30)
ξ = np.array([noise(t) for t in np.linspace(noise.t_min, noise.t_max, 1000)])
ξ.std(axis=0)

# 2d output

noise = ColoredNoise(t_min    =   0. *ureg.ms,
                     t_max    =1000. *ureg.ms,
                     corr_time=   1. *ureg.ms,
                     scale    =[[2.2, 1.1],
                                [3.3, 4.4]],
                     impulse_density=30)
ξ = np.array([noise(t) for t in np.linspace(noise.t_min, noise.t_max, 1000)])
ξ.std(axis=0)

## Serialization

from scityping.pydantic import BaseModel

noise = ColoredNoise(t_min    =   0. *ureg.ms,
                     t_max    =1000. *ureg.ms,
                     corr_time=   1. *ureg.ms,
                     scale    =np.array([2.2, 1.1]),
                     impulse_density=30)

data = Serializable.deep_reduce(noise)
data

noise2 = Serializable.validate(data)
noise2

tarr = np.linspace(0, 1, 5)
assert np.array_equal([noise(t) for t in tarr],
                      [noise2(t) for t in tarr])

class Foo(BaseModel):
    noise: ColoredNoise

foo = Foo(noise=noise)
foo.json()

foo2 = foo.parse_raw(foo.json())

assert np.array_equal([foo.noise(t) for t in tarr],
                      [foo2.noise(t) for t in tarr])
