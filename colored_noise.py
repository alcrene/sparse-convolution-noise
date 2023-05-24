# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python (emd-paper)
#     language: python
#     name: emd-paper
# ---

# %% [markdown] user_expressions=[]
# # Using sparse convolutions to generate noisy inputs

# %% [raw] tags=["remove-cell"]
# ########################### LICENSE ###############################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.          
#
#                Copyright © 2023 Alexandre René
# ###################################################################


# %% [markdown] user_expressions=[]
# ## Basic principle

# %% [markdown] user_expressions=[]
# :::{note}
# :class: margin
# We are interested here in 1-d signals, but this algorithm generalizes easily to $n$-d. The $n$-d form is the one found in [(Lewis, 1989)](https://doi.org/10.1145/74333.74360).
# :::
# The sparse convolution algorithm [(Lewis, 1989)](https://doi.org/10.1145/74333.74360) is an effective way to generate a noise signal with a prescribed covariance.
# Moreover, it produces a so-called “solid noise”: a continuous function which can be evaluated at any $t$.[^sparse-noise] This algorithm was originally designed to produce computationally efficient, naturalistic noise for computer graphics, but the same features make it uniquely suited for use as an input noise when integrating systems with standard ODE integrators. Even integrators which adapt their step size are supported.
#
# The idea is to generate the desired signal $ξ$ by convolving a kernel $h$ with a Poisson noise process $γ$:
# \begin{equation}
# ξ(t) = h \ast γ = \int h(t-t') γ(t') \,dt' \,.
# \end{equation}
# The Poisson noise is defined as
#
# $$γ(t) = \sum_k a_k \, δ(t - t_k) \,, $$ (eq_Poisson-noise_no-bins)
#
# where both the weights $a_k$ and the impulse times $t_k$ are stochastic and uncorrelated. (It particular this means that the distribution of $a_k$ should satisfy $\braket{a_k} = 0$.) Because the impulses are sparse, the convolution reduces to an efficient summation:
# \begin{equation}
# ξ(t) = \sum_k a_k h(t - t_k) \,,
# \end{equation}
# and the autocorrelation is given by
# \begin{equation}
# \bigl\langle{ξ(t)ξ(t-t')}\bigr\rangle = \sum_k \Braket{a_k^2} h(t-t_k)h(t'-t_k) \,.
# \end{equation}
#
# [^sparse-noise]: Contrast this with a standard scheme of generating white noise by drawing from a Gaussian with variance $Δt σ^2$: these generate noisy inputs $ξ_n$, which represent the integrals $\int_{nΔt}^{(n+1)Δt} ξ(t) dt$, but to generate an intermediate point for $ξ((n+\tfrac{1}{2})Δt)$, one needs to draw a new point: there is no deterministic rule to compute a dense $ξ(t)$ from a sparse set of $ξ_n$. One can of course interpolate the $ξ_n$, but the result has non-physical statistics: the autocovariance depends not just on the distance between points, but also on how they line up to the arbitrary grid of reference points $ξ_n$.

# %% [markdown] user_expressions=[]
# :::{admonition} Remark
#
# Assuming the derivatives of the kernel function are known, derivatives of the generated noise are also easy to evaluate:
#
# $$ξ^{(n)}(t) = \sum_k a_k h^{(n)}(t-t_k)$$
#
# :::

# %% [markdown] user_expressions=[]
# :::{admonition} Notation
# :class: hint, margin
#
# We use
# \begin{equation*}
# \small \tilde{ξ}(ω) = \frac{1}{\sqrt{2π}}\int e^{-iωt} ξ(t)\,dt
# \end{equation*}
# to denote the Fourier transform of $ξ$, and
# \begin{align*}
# \small S_{ξξ}(ω) &= \lvert \tilde{ξ}(ω) \rvert^2 \\
# &= \frac{1}{\sqrt{2π}}\int\!dt\,e^{-iωt} \braket{ξ(t)ξ(t-τ)}_τ
# \end{align*}
# to denote its power spectrum.[^wiener-kinchine]
#
# [^wiener-kinchine]: The equality of the power spectrum with the Fourier transform of the correlation function follows from the Wiener-Kinchine theorem, and the fact that these are wide-sense-stationary processes.
# :::
# Moreover, the power spectrum $S_{γγ}$ of $γ$ is flat:
# \begin{align*}
# \Bigl\langle{γ(t)γ(t')}\Bigr\rangle &= \bigl\langle{a^2}\bigr\rangle \, δ(t-t') \\
# S_{γγ}(ω) &= \frac{1}{\sqrt{2π}} \int \Braket{γ(t)γ(t')} e^{-iωt} \,dt = \frac{ρ\braket{a^2}}{\sqrt{2π}}
# \end{align*}
# which is why $γ$ is referred to as a sparse white noise.

# %% [markdown] user_expressions=[]
# In particular this means that the power spectrum of the autocorrelation $\braket{ξ(t)ξ(t')}$ is proportional to that of $h$:[^sloppy-power-spectrum][^mistake-in-Lewis]
# \begin{align*}
# S_{ξξ}(ω) &= S_{γγ}(ω) S_{hh}(ω) \\
# &= \frac{ρ\braket{a^2}}{\sqrt{2π}} S_{hh}(ω) \\
# &= \frac{ρ\braket{a^2}}{\sqrt{2π}} \lvert \tilde{h}(ω) \rvert^2 \,.
# \end{align*}
#
# [^sloppy-power-spectrum]: This result is easily seen if we assume the Fourier transform of both $γ$ and $h$ to exist. Then using the result that the Fourier transform converts convolutions into products, we have  
#   $\displaystyle
#   S_{ξξ}(ω) = \lvert \tilde{ξ}(ω) \rvert^2 = \lvert \tilde{h}(ω) \tilde{γ}(ω) \rvert^2 = \lvert \tilde{γ}(ω) \rvert^2  \lvert \tilde{h}(ω) \rvert^2  \,.
#   $  
#   However the equality of the power spectra is even more general, since it applies even when the Fourier transforms don't exist. This is useful here, because $γ$ is stochastic: therefore does not have a well-defined Fourier transform, but it does have an autocorrelation function.
#
# [^mistake-in-Lewis]: In Lewis this is given as $S_{y}(ω) = S_x(ω) \lvert H(jω)\rvert^2$, with $S_y = S_{ξξ}$, $S_x = S_{γγ}$, $j = i$ and $H$ the Fourier transform of $h$. As far as I can tell the presence of the imaginary factor $j$ is a mistake: at least in the case where the Fourier transform of $γ$ exists, it does not appear (see above footnote). Possibly this is due to the fact that in the study of LTI (linear time-invariant) systems, often the *Laplace* transform of a kernel $h$ is written $H$; then the Fourier transform is written $H(iω)$ (or $H(jω)$ in engineering notation). See e.g. [here](https://en.wikipedia.org/wiki/Linear_time-invariant_system#Fourier_and_Laplace_transforms).

# %% [markdown] user_expressions=[]
# To get the correlation function, we can use the Wiener-Khinchine theorem: applying the inverse Fourier transform to $S_{ξξ}$ will yield the autocorrelation:
# \begin{equation}
# C_{ξξ}(t) = \Bigl\langle{ξ(t)ξ(t+s)\Bigr\rangle} = \frac{ρ\braket{a^2}}{\sqrt{2π}} \mathcal{F}^{-1}\bigl\{\,\lvert\tilde{h}\rvert^2 \,\bigr\} \,.
# \end{equation}
# In general this is not possible to do analytically, but for particular choices of $h$ it is.
# Perhaps the most obvious one is a Gaussian kernel; then $h(s)$, $\tilde{h}(ω)$, $|\tilde{h}(ω)|$ and $C_{ξξ}(s)$ are all Gaussian.
# Another one would be a boxcar kernel. In general, if we have a function $\tilde{h}(ω)$ for which the Fourier transforms of both $\tilde{h}(ω)$ and $|\tilde{h}(ω)|^2$ are known, then it is possible to select the kernel given the desired autocorrelation length.

# %% [markdown] user_expressions=[]
# :::{admonition} Possible extensions
# :class: dropdown
#
# One simple extension would be to allow online generation of noise samples: the current implementation requires to pre-specify `t_min` and `t_max`, but this is only for convenience and because it was natural to do so in my use cases. Some ideas:
# - Store impulse times and weights in a cyclic buffer, allowing for infinite generation forward in time.
# - Design a scheme which assigns to each bin a unique seed, computed from $t$. When a time is requested, generate the required impulses and compute the noise value. This would allow random access on an infinite time vector. With suitable caching, it could also be used for a generator which allows infinite iteration both forward *and* backward in time.
#
# Another extension would be to add support for more kernels.
# One kernel that may be desirable is the exponential, $h(s) = e^{-|s|/τ}$, since often we think of correlation length as the scale of an exponential decay. We find then that $\tilde{h}(ω)$ should be a Lorentz distribution, for which unfortunately $C_{ξξ}(s) = \mathcal{F}^{-1}\bigl\{\,|\tilde{h}(ω)|^2\,\}$ does not Fourier transform easily.[^no-Lorentz-kernel] However, numerical experiments suggest that $C_{ξξ}(s)$ looks an awful lot like a Lorentz distribution – perhaps a suitable approximation could formalize the correspondence. A Lorentzian autocorrelation would be an interesting complement to the Gaussian since it has very different satistics (in particular very heavy tails), and is a frequent alternative to the Gaussian is many applications.
#
# Going more general, one could envision implementing a generic class which can approximate an arbitrary autocorrelation, by using FFT to compute the kernel $h$:
# \begin{equation*}
# \begin{CD}
# S_{ξξ}(ω) = |\tilde{h}(ω)|^2 @>{\sqrt{}}>>  h(ω)   \\
# @AA{\mathcal{F}}A                 @VV{\mathcal{F}^{-1}}V  \\
# C_{ξξ}(s)                    @.        h(s)  @>\text{smooth}>>  \bar{h}(s)
# \end{CD}
# \end{equation*}
# This might have a few advantages over the more common approach of sampling Fourier coefficients:
# - While sampled spectra are by design non-smooth (or even non-continuous), here both the original function ($C_{ξξ}$) and the estimate ($h$) are functions of time. It is therefore more reasonable to apply smoothing, which might help remove artifacts introduced by the FFT.
# - Because we are estimating a *kernel* with FFT, and not the signal itself, we can still generate noise signals of arbitrary length and in an online fashion.
# - Once computed over discretized time, the estimated kernel $h$ could be replaced by something like a spline, making dense evaluation reasonably cheap.
#
# [^no-Lorentz-kernel]: Interestingly, using a Lorentz kernel instead *does* result in a fully solvable set of equations. Unfortunately, the resulting noise does not look like one would expect, and its statistics don’t converge (at least not within a reasonable amount of time). Which isn’t so surprising given the number of pathologies associated to the Lorentz distribution (infinite variance, undefined mean…)
# :::

# %% [markdown] user_expressions=[]
# :::{admonition} Summary
# :class: important
#
# - To generate a noise $ξ$ with an given autocorrelation *spectrum*, we need to choose a kernel $h$ with the same autocorrelation spectrum (up to some constant).
# - This constant is proportional to the variance of the weights $a$.
# :::

# %% [markdown] user_expressions=[]
# ## Comparison with other methods
#
# ::::{grid} 1 1 2 2
#
# :::{grid-item-card} Sparse convolution
# - ✔ Can use a standard ODE integrator.
# - ✔ Dense output: Can be evaluated at arbitrary $t$.
# - ✔ Technically simple.
# - ✔ Large amount of control over the autocorrelation function.
# - ✘ Autocorrelation function is controlled via its spectrum. In general closed form expressions relating a kernel $h$ to a autocorrelation $C_{ξξ}$ do not exist.
# - ✔ Online algorithm: can easily generate new points.
# - ✔ Fast: Number of operations per time point is in the order of $\mathcal{O}(5mρ)$, where $m$ is the number of neighbouring bins we sum over and $ρ$ is the number of impulses per bin.
#   In particular, this is independent of the total simulation time.
# :::
#
# :::{grid-item-card} Integrating Wiener noise
# - ✘ Requires a stochastic (SDE) integrator.
# - ✘ Sparse trace: trace is composed of discrete time steps. Generating a new intermediate point requires re-integrating everything after that point.[^why-no-ode-integrator]
# - ✔ Borderline trivial, if one is familiar with stochastic integrals.
#   Otherwise conceptually and technically difficult.
# - ✔ The autocorrelation is known: it is determined by the number of surrogate variables added for the integrals.
# - ✘ Limited control over the correlation function: limited to those obtainable by adding a reasonable number of surrogate variables.
# - ✔ Online algorithm: can easily generate new points.
# - ✔ Very fast: New points are generated by drawing a Gaussian and performing a few sums.  
#   However, the need to use a low-order stochastic integrator may inflate computation time.
# :::
#
# [^why-no-ode-integrator]: This is the main reason this method cannot be used with a normal ODE integrator.
#
# :::{grid-item-card} Generating random spectra
# - ✔ Can use a standard ODE integrator (I think).
# - ✔ Dense output possible: If we represent as a sum of sine waves instead of the FFT vector, the function can be evaluated at arbitrary $t$.
#   However the spectrum is only valid above some time resolution $Δt$.
# - ✔ Full, direct control over the autocorrelation shape.
# - ✘ Technically difficult: It is difficult to get the FFTs to scale exactly right, and they introduce many numerical artifacts.
# - ✘ Scales poorly with the length of simulated time (because of the required FFT).
# - ✘ Fixed simulation time: cannot generate new points to extend the trace.
# - ✔ Moderately fast during evaluation: Number of operations scales with the number of sine components (and therefore both the resolution $Δt$ and the total simulation time).
# :::
#
# :::{grid-item-card} Perlin noise
# - ✘ Intended for computer graphics: designed for numerical efficiency, not to have well-characterized statistics (beyond “they look nice”).
# - Similarly, existing libraries which implement Perlin noise are intended for computer graphics, and I would not consider them suitable for scientific applications since the statistics are difficult to characterize.
# - I didn’t consider this further: one would need to write their own “scientific-grade” noise model, in which case they might as well use a simpler algorithm like the sparse convolutions.
# :::
#
# ::::

# %% [markdown] user_expressions=[]
# ## Implementation

# %%
from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

import math
import numpy as np
import numpy.random as random
import scipy.stats as stats

__all__ = ["ColoredNoise"]

# %% [markdown] user_expressions=[]
# One small complication with the definition in Eq. {eq}`eq_Poisson-noise_no-bins` for the Poisson noise is that for a given $t$, we don’t know which $t_k$ are near enough to contribute. And for a long time trace, we definitely don’t want to sum tens of thousands of impulses at every $t$, when only a few dozen contribute. So following [Lewis (1989)](https://doi.org/10.1145/74333.74360), we split the time range into bins and generate the same number of impulses within each bin. Then for any $t$ we can compute with simple arithmetic the bin which contains it, and sum only over neighbouring bins.

# %% [markdown] user_expressions=[]
# ### `ColoredNoise`
#
# The default `ColoredNoise` class produces noise with a Gaussian kernel, Gaussian weights $a$, and a Gaussian autocorrelation:
#
# Parameters
# ~ + $τ$: Correlation time. Equivalent to the *standard deviation* of the autocorrelation function.
#   + $σ$: Overall noise strength; more specifically its standard deviation: $\braket{ξ(t)^2} = σ^2$.
#   + $ρ$: Impulse density, in units of $\frac{\text{# impulses}}{τ}$. For numerical reasons this must be an integer.  
#     This affects what Lewis calls the “quality” of the noise; a too low number will show artifacts (high peaks), which only disappear after averaging many bins or realizations.
#     Note that increasing $ρ$ does not cause the noise to wash out, but rather the effect saturates: Lewis reports that generated noises stop being distinguishable when $ρ$ is greater than 10. In my own tests I observe something similar, even though my implementation is quite different.
#     Part of the reason for this is that we scale the variance of the $a_k$ to keep the overall variance of $ξ$ constant.
#
# Autocorrelation
# ~ $\displaystyle C_{ξξ}(s) = σ^2 e^{s^2/2τ^2}$
#
# Kernel
# ~ $\displaystyle h(s) = e^{-s^2/τ^2}$
#
# Weight distribution
# ~ $\displaystyle a_k \sim \mathcal{N}\left(0, σ \sqrt{\frac{1}{ρ}\sqrt{\frac{2}{π}}} \right)$
#
# Binning
# ~ *Bin size:* $τ$
# ~ *Summation:* 5 bins left and right of $t$ (so a total of 11 bins)

# %% [markdown] user_expressions=[]
# :::{admonition} TODO
# :class: warning, margin
#
# Recheck calculations for the coefficient of the `autocorr` function and the std. dev. of the $a$ weights. Add expression for $S_ξξ$.
# :::

# %%
class ColoredNoise:
    """
    Simplified solid noise with some predefined choices: exponential kernel :math:`h`
    and Gaussian weights :math:`a`. The specification parameters are the overall variance σ²,
    the correlation time τ and the number of impulses in a time interval of length τ.
    
    Note
    ----
    This class supports specifying values with units specified using the `pint` library.
    This is an excellent way of avoiding some common mistakes. When doing so, all units
    must be consistent; in particular, ``t/τ`` must be dimensionless in order for the
    exponential to be valid.
    Using units does add a bit of overhead, so for performance-critical code omitting
    them may be preferred.
    """

    # Inspection attributes (not uses when generating points)
    σ: float
    τ: float
    ρ: float
    bin_edges: np.ndarray[float]
    t_max: float
    rng: np.random.Generator
    # Attributes used when generating noise
    t_min: float
    λ: float
    t_arr: np.ndarray[float]
    a_arr: np.ndarray[float]
    
    def __init__(self,
                 t_min:float, t_max:float,
                 scale: float, corr_time: float,
                 #weight_std: float,
                 impulse_density: int,
                 rng: Union[int,random.Generator,random.SeedSequence,None]=None):
        """
        Parameters
        ----------
        scale: (σ) Standard deviation of the generated noise
        corr_time: Correlation time (τ). If the kernel is given by :math:`e^{-λ|τ|}`,
            this is :math:`τ`.
        impulse_density: (ρ) The expected number of impulses within one correlation
            time of a test time.
        rng: Either a random seed, or an already instantiated NumPy random `Generator`.
        """
        rng = random.default_rng(rng)
        
        # Compute the std dev for the weights
        σ, τ, ρ = scale, corr_time, impulse_density
        a_std = σ * np.sqrt(1/ρ) * (2/np.pi)**(0.25)
        
        # Discretize time range into bins of length τ
        # Following Lewis, we draw the impulses in bins.
        # Each bin has exactly the same number of impulses, allowing us in __call__ to
        # efficiently restrict ourselves to nearby points.
        # NB: We always take 5 bins to the left and right, so we also pad that many bins.
        Nbins = math.ceil((t_max - t_min) / τ) + 10
        self_t_min = t_min - 5*τ
        self_t_max = self_t_min + (Nbins+1)*τ
            
        # Draw the impulse times
        # Each row corresponds to the impulses in one bin
        bin_edges = np.arange(Nbins+1) * τ + self_t_min
        if hasattr(bin_edges, "units"):  # Assume values were provided with pint units:
            t_units = bin_edges.units
            _bin_edges = bin_edges.magnitude
        else:
            t_units = 1
            _bin_edges = bin_edges
        t_arr = rng.uniform(_bin_edges[:-1,np.newaxis], _bin_edges[1:,np.newaxis],
                            size=(Nbins, impulse_density)
                           ) * t_units
        
        # Draw the weights
        a_arr = rng.normal(0, 1, t_arr.shape) * a_std
        
        # Store attributes
        self.σ = σ
        self.τ = τ
        self.ρ = ρ
        self.λ = 1/τ  # With floating point numbers, multiplication is faster than division
        self.rng = rng
        self.bin_edges = bin_edges
        self.t_min = t_min
        self.t_max = t_max
        self.t_arr = t_arr
        self.a_arr = a_arr
        
    def __call__(self, t, *, pad=5, exp=np.exp, int32=np.int32):
        # Compute the index of the bin containing t
        i = (t-self.t_min) * self.λ; i = getattr(i, "magnitude", i)  # Suppress Pint warning that int32 removes dim
        i = int32(i) + pad
        # Compute the noise at t
        tk = self.t_arr[i-pad:i+pad]
        h = np.exp(-((t-tk)*self.λ)**2)  # Inlined from self.h
        a = self.a_arr[i-pad:i+pad]
        return (a * h).sum()
    
    def new(self, **kwargs):
        """
        Create a new model, using the values of this one as defaults.
        NB: By default, this uses the same RNG as the original, but will produce new
        times and weights (since it draws new points).
        If you want to avoid advancing the state of the orignal RNG, provide a new one.
        """
        defaults = dict(t_min=self.t_min, t_max=self.t_max,
                        scale=self.σ, corr_time=self.τ, impulse_density= self.ρ,
                        rng=self.rng)
        return ColoredNoise(**(defaults | kwargs))
    
    def h(self, s):
        return np.exp(-(s*self.λ)**2)
    
    def autocorr(self, s):
        """Evaluate the theoretical autocorrelation at lag s."""
        return self.σ**2 * np.exp(-0.5*(s*self.λ)**2)
    
    @property
    def T(self):
        return self.t_max - self.t_min
    
    @property
    def Nleftpadbins(self):
        return 5
    @property
    def Nrightpadbins(self):
        return 5
    @property
    def Nbins(self):
        """Return the number of bins, excluding those included for padding."""
        return len(self.t_arr) - self.Nleftpadbins - self.Nrightpadbins

# %% [markdown] user_expressions=[]
# ## Validation

# %% tags=["active-ipynb", "hide-input"]
# import itertools
# import scipy.signal as signal
# import holoviews as hv
# import pint
# from types import SimpleNamespace
# from tqdm.notebook import tqdm
# from myst_nb import glue
# ureg = pint.UnitRegistry()
# ureg.default_format = "~P"
# hv.extension("bokeh", "matplotlib")
# %matplotlib inline

# %% tags=["active-ipynb", "hide-input"]
# dims = SimpleNamespace(
#     t  = hv.Dimension("t", label="time", unit="ms"),
#     Δt = hv.Dimension("Δt", label="time lag", unit="ms"),
#     ξ  = hv.Dimension("ξ"),
#     ξ2 = hv.Dimension("ξ2", label="⟨ξ²⟩"),
#     T  = hv.Dimension("T", label="realization length", unit="ms"),
#     σ  = hv.Dimension("σ", label="noise strength", unit="√ms"),
#     τ  = hv.Dimension("τ", label="correlation length", unit="ms"),
#     ρ  = hv.Dimension("ρ", label="impulse density"),
#     N  = hv.Dimension("N", label="# realizations")
# )
# colors = hv.Cycle("Dark2").values

# %% tags=["active-ipynb"]
# noise = ColoredNoise(t_min=0.    *ureg.ms,
#                      t_max=10.   *ureg.ms,
#                      scale=2.2   *ureg.ms**0.5,
#                      corr_time=1.*ureg.ms,
#                      impulse_density=30,
#                      rng=1337)
# assert noise.Nbins == 10
# expected_bin_edges = np.array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5., 
#                                6.,  7., 8.,  9., 10., 11., 12., 13., 14., 15.])*ureg.ms
# assert np.allclose(noise.bin_edges, expected_bin_edges)

# %% tags=["remove-cell", "active-ipynb"]
# N = 1000
# _lags = signal.correlation_lags(N, N)
# norm_autocorr = hv.Curve(zip(_lags, signal.correlate(np.ones(N), np.ones(N))),
#                kdims="lag (l)", vdims="C") 
# norm_autocorr.opts(
#     hv.opts.Curve(title="Unnormalized correlation\nof a flat function",
#                   ylabel="",
#                   fontsize={"title": "10pt"}),
#     hv.opts.Curve(height=200, width=250, backend="bokeh"),
#     hv.opts.Curve(title="Unnormalized correlation\nof a flat function",
#                   ylabel="", fontsize={"title": "small", "labels": "small", "ticks": "x-small"},
#                   fig_inches=3, aspect=3,
#                   backend="matplotlib")
# )
#
# norm_autocorr_fig = hv.render(norm_autocorr, backend="matplotlib")
# glue("norm_autocorr", norm_autocorr_fig, display=True);

# %% tags=["active-ipynb"]
# n_realizations_shown = 10
# seedseq = np.random.SeedSequence(6168912654954)
# Tlst = [100.]
# σlst = [0.33, 1., 9.]
# τlst = [1., 5., 25.]
# ρlst = [1, 5, 30, 200]
# Nlst = [100]
# exp_conds = list(itertools.product(Tlst, σlst, τlst, ρlst, Nlst))

# %% tags=["active-ipynb"]
# frames_realizations = {}
# frames_autocorr = {}
# ms = ureg.ms

# %% [markdown] user_expressions=[]
# :::{admonition} Computing correlations with `scipy.signal`
# :class: caution, margin, dropdown
#
# The SciPy function `signal.correlate` (along with its companion `signal.correlation_lags` to compute the lag axis) is a convenient way to compute the autocorrelation. However before plotting the result, one must take care to normalize it correctly. Indeed, if $x$ is a discretized signal with $N$ time bins, and $C_k$ is its discretized correlation function at lag $k$, then the definition used by `correlate` is
# \begin{equation}
# C_k = \Braket{x_l x_{l+k}} = \sum_{l=0}^{N-1-k} x_l x_{l+k} \,.
# \end{equation}
# Note that the number of terms depends on $k$. We can see this clearly when computing the autocorrelation of the flat constant function $x_l = 1$: the result should also be flat (albeit dependent on $N$), but instead we get a triangular function peaking at zero:  
# {glue:}`norm_autocorr`  
# where the value on the $y$ axis is exactly the number of terms contributing to that lag.
# To avoid this artifical triangular decay, in the code we normalize the result by the number of terms contributing to each lag; in terms of a continuous time autocorrelation, this is equivalent to normalizing by the value at zero:
# \begin{equation}
# C^{\text{normed}}(s) = \frac{C(s)}{C(0)} \,.
# \end{equation}
# :::

# %% [markdown] user_expressions=[]
#

# %% tags=["active-ipynb", "hide-input"]
# experiment_iter = tqdm(exp_conds, "Exp. cond.")
# for T, σ, τ, ρ, N in experiment_iter:
#     if (T, σ, τ, ρ, N) in (frames_realizations.keys() & frames_autocorr.keys())  :
#         continue
#     
#     noise = ColoredNoise(0*ms, T*ms, scale=σ, corr_time=τ*ms, impulse_density=ρ, rng=seedseq)
#     t_arr = np.linspace(noise.t_min, noise.t_max, int(10*T*ms/noise.τ))
#
#     ## Generate the realizations and compute their autocorrelation ##
#     L = len(t_arr)
#     Δt = np.diff(t_arr).mean()
#     norm = signal.correlate(np.ones(L), np.ones(L), mode="same")  # Counts the number of sums which will contribute to each lag
#     lags = signal.correlation_lags(L, L, mode="same") * Δt
#     ξ_arr = np.empty((N, L))
#     Cξ_arr = np.empty((N, L))
#     for i, key in enumerate(tqdm(seedseq.spawn(N), "Seeds", leave=False)):
#         _noise = noise.new(rng=key)
#         ξ = np.fromiter((_noise(t) for t in t_arr), count=len(t_arr), dtype=float)
#         ξ_arr[i] = ξ
#         Cξ   = signal.correlate(ξ, ξ, mode="same") / norm
#         Cξ_arr[i] = Cξ
#     Cξ = Cξ_arr.mean(axis=0)
#
#     ## Generator realization curves ##
#     realization_samples = hv.Overlay([
#         hv.Curve(zip(t_arr.magnitude, _ξ), kdims=dims.t, vdims=dims.ξ, label="Single realization")
#         for _ξ in ξ_arr[:n_realizations_shown]
#     ])
#     
#     ## Generate autocorr curves ##
#     autocorr_samples = hv.Overlay([
#         hv.Curve(zip(lags.magnitude, _Cξ), kdims=dims.Δt, vdims=dims.ξ2, label="Single realization")
#         for _Cξ in Cξ_arr[:n_realizations_shown]]
#     )
#     avg =  hv.Curve(zip(lags.magnitude, Cξ), kdims=dims.Δt, vdims=dims.ξ2, label=f"Average ({N} realizations)")
#     target = hv.Curve(zip(lags.magnitude, noise.autocorr(lags).magnitude), kdims=dims.Δt, vdims=dims.ξ2, label="Theoretical")
#     
#     ## Compute axis range so it is appropriate for mean and target autocorr – individual realizations may be well outside this range ##
#     ymax = max(avg.range("ξ2")[1], target.range("ξ2")[0])
#     ymin = min(avg.range("ξ2")[0], target.range("ξ2")[0])
#     Δy = ymax-ymin
#     ymax += 0.05*Δy
#     ymin -= 0.05*Δy
#     # Round ymin down, round ymax up
#     p = math.floor(np.log10(ymax-ymin)) + 2  # +2: between 10 and 100 ticks in the range
#     new_range = (round(math.floor(ymin * 10**p) / 10**p, p),
#                  round(math.ceil (ymax * 10**p) / 10**p, p))
#
#     ## Assemble figures ##
#     # Use random shades of grey for realization curves so we can distinguish them
#     shades = np.random.uniform(.45, .8, size=n_realizations_shown)
#     
#     fig_autocorr = autocorr_samples * avg * target
#     fig_autocorr.opts(ylim=new_range)
#     fig_autocorr.opts(
#         hv.opts.Curve(height=300, width=400),
#         hv.opts.Curve("Curve.Single_realization", color="#DDD"),
#         hv.opts.Curve("Curve.Average", color=colors[0]),
#         hv.opts.Curve("Curve.Prescribed", color=colors[1], line_dash="dashed"),
#         hv.opts.Overlay(title="Autocorrelation", legend_position="top"),
#     )
#     for curve, c in zip(fig_autocorr.Curve.Single_realization, shades):
#         curve.opts(color=(c,)*3)
#     
#     fig_realizations = realization_samples
#     fig_realizations.opts(
#         hv.opts.Curve(height=300, width=400),
#         #hv.opts.Curve("Curve.Single_realization", color="#DDD"),
#         hv.opts.Overlay(title="Noise realizations", show_legend=False)
#     )
#     for curve, c in zip(fig_realizations, shades):
#         curve.opts(color=(c,)*3)
#     
#     frames_realizations[(T, σ, τ, ρ, N)] = fig_realizations
#     frames_autocorr[(T, σ, τ, ρ, N)] = fig_autocorr

# %% tags=["remove-input", "active-ipynb"]
# hmap_autocorr = hv.HoloMap(frames_autocorr, kdims=[dims.T, dims.σ, dims.τ, dims.ρ, dims.N])
# hmap_realizations = hv.HoloMap(frames_realizations, kdims=[dims.T, dims.σ, dims.τ, dims.ρ, dims.N])
# fig = hmap_autocorr + hmap_realizations
# fig.opts(
#     hv.opts.Layout(title=""),
#     hv.opts.Curve(framewise=True)
# )
# fig.cols(1)
