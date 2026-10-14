# The hybrid scheme

The driver in rough Bergomi is the Riemann-Liouville process

    Y_t = int_0^t (t - s)^{H - 1/2} dW_s,    H in (0, 1).

It is Gaussian, centred, and not Markov. Its variance is

    Var(Y_t) = int_0^t (t - s)^{2H - 1} ds = t^{2H} / (2H).

At H = 1/2 the kernel is identically 1, so Y is Brownian motion and the variance is t. The same formula covers that case, because t^{2H} / (2H) equals t when H = 1/2.

Fractional Brownian motion has variance t^{2H}. The factor 1 / (2H) is the difference. Rough Bergomi multiplies Y by sqrt(2H), which puts the variance of that product back to t^{2H}, and then the exponential martingale compensator is nu^2 t^{2H} / 2.

## The scheme

Bennedsen, Lunde and Pakkanen (2017) split the integral into a near part and a tail. On a grid of step dt = T / n, the value at a grid time is a sum of integrals over the past steps. The lag of a step is how many steps it sits behind the evaluation time.

For the first kappa lags the integral of the power kernel is kept exact. On one step the vector (dW, I_1, ..., I_kappa) is Gaussian with a closed covariance, their equation (3.5). I_k is the integral of (k dt - tau)^{H - 1/2} on an interval of length dt. Different steps are independent, so the cost of the exact part is one small factorisation, reused at every step.

For lags beyond kappa the kernel is replaced by its value at the optimal point b*_k of their Proposition 2.8,

    b*_k = ( (k^{alpha + 1} - (k - 1)^{alpha + 1}) / (alpha + 1) )^{1/alpha},

alpha = H - 1/2. The weight is (b*_k dt)^alpha. The resulting sum is a discrete convolution of the weights with the Wiener increments, applied by FFT. That is `power_kernel_weights` and `wiener_integral`.

kappa is 0, 1 or 2. The default is 1. kappa = 0 is the Riemann sum with the same optimal points and no exact block.

## What kappa does

Theorem 2.5 of the same paper: the mean-square error at a fixed time is asymptotic to J(alpha, kappa) n^{-2H}. The power of n does not depend on kappa. Raising kappa drops the terms that dominate J, so the constant falls and the rate stays.

The figure `docs/figures/rva_covariance.png` is the root-mean-square difference of the scheme covariance from the exact covariance, on the grid t_i = i/n in [0, 1], for kappa = 0, 1 and 2 at H = 0.1. At n = 64 that RMS is 1.214e-1, 5.234e-3 and 2.854e-3. The log-log slopes against n, for n from 16 to 512, are -0.700, -0.681 and -0.660. The slopes sit together. The level drops by a factor of 23 from kappa = 0 to kappa = 1.

The relative error of Var(Y_1) on the same runs has slopes -0.200, -0.198 and -0.194. Those are the terminal variance, and -2H = -0.2 is the rate in Theorem 2.5. The covariance RMS is a different functional of the same error, which is why its slope is not -0.2. The CSV next to the figure and the entry in `LOG.md` hold the grid.

## Draw order

`hybrid_scheme` spawns one child of the seed. At each step it draws the components (dW, I_1, ..., I_kappa) in that order, one vector of length m per component, and multiplies by the lower Cholesky factor of the block covariance. At H = 1/2 the integrals equal the Wiener increment, the covariance has rank 1, and one normal per path is drawn. The path is then exactly that Brownian motion.

`hybrid_scheme_rng` is the same draw from a generator that already exists. It does not spawn. The returned `components` array has shape `(m, n, kappa + 1)` and holds that Gaussian vector on every step, including an `I_k` that would land past the horizon. Rough Bergomi's `volterra_components` calls it on the volatility child and returns this array. `volterra_increments` returns the `dW` column only. The first entry of the block covariance is the step length, so each of those increments has that variance. The weights are not part of the return. At H = 1/2 the scheme stores `dW` in every component and still draws one normal per step.
