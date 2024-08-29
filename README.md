 # Julia_exercise_projects
Number of exercises in Julia.

## Stochastic Calculus Projects

### Mathematical background and definitions used

_Martingale_: A martingale is an (ordered) sequence of random variables where the expectation value of the next random variable equals that of the last in the sequence, regardless of the sequence leading up to it. I.e. in terms of conditional probabilities, given a sequence of random variables $S$ and last element $Z_i$, we have that $E(Z_i|S) = Z_i$.

_Wiener process_: A continious type of stochastic process. The exact definition will not be quoted here: what we need to know is that for a Wienner process, $W(t)$, for any two times $T + \delta T > T$, we have that $W_{t+\delta T} - W_{t} \propto N(0, \delta T)$ indepent of $T$, i.e. it 'locally' varies as a normal distribution with spread $\sqrt{\delta T}$. Furthermore, we set $W(0) = 0$. (i.e. it satisfies the conditions of being a continious martingale)

_Itô's Lemma_: Can be thought of as the "chain rule" for functions involving Itô processes. 

In the simplest case, iven a variable $X$ that is undergoing a drift-diffusion process, we have that $dX(t) = \mu(t) dt + \sigma(t) dW(t)$ i.e. it is undergoing a deterministic 'drift' proportional to $\mu(t)$ and a nondeterministic Wienner process $W(t)$ proportional to $\sigma(t)$. Letting $f = f(t, X)$, Itô's Lemma states that the differential of f is then $df = (\frac{\partial f}{\partial t} + \mu(t) \frac{\partial f}{\partial X} + \frac{\sigma(t)^2}{2}\frac{\partial^2 f}{\partial x^2}) dt + \sigma(t) \frac{\partial f}{\partial x} dW(t)$. The exact derivation of the lemma is not presented here: however, one can arrive to the lemma by just Taylor expanding $F(X,t)$ to second order and using that for Wiener processes, $dW(t)^2 = dt$ (see above), and disregarding any element that has larger than linear $dt$ scaling.

Itô's Lemma is applicable not just for variables $f$ that are undergoing a drift-diffusion process, but also to a wider range of stochastic processes, such as the asset movement assumed by the Black-Scholes model. This is important to keep in mind if one aims to model fat-tail walks which might be a better represenation of asset price movement. The generalized Itô Lemma is beyond the needs of this document and is typically expressed in integral formalism instead. 


### The Heston Model

The Heston model can be thought of as a an extension of the Black-Scholes model that deals with the problem of non-constnat volatility. It is notable for being able to predict the volatility smile and its skew. 

The Heston model is defined via 2x SDEs.

The asset price, $S(t)$ evolves according to:

$dS(t) = \mu S(t) dt + \sqrt{v(t)}dW(t)_s$

The volatility itself is also stochastic. The SDE of its square is a mean-reverting process, given by:

$dv(t) = \kappa (\theta - v(t))+\xi dW(t)_v$

$dW_s$ and $dW_v$ are correlated, usually with a negative coefficient $\rho$: this is based on real-life observations, which indicate that a stochastic decrease in stock price is likely to be due/be accompanied by an increase in effecive volatility. 

Although not proven here, it can be shown that one gets, through the applications of Itô's Lemma and by enforcing the no-risk conditions on a self-financing portfolio, that the SDE for an option price $U$, that is used to hedge against the volatility risk (i.e. not the same as $V$ in the BS model) is:

$\frac{\partial U}{\partial t}+\frac{1}{2} v S^2 \frac{\partial^2 U}{\partial S^2}+\rho \sigma v S \frac{\partial^2 U}{\partial v \partial S}+\frac{1}{2} \sigma^2 v \frac{\partial^2 U}{\partial v^2} 
-r U+r S \frac{\partial U}{\partial S}+[\kappa(\theta-v)-\lambda(S, v, t)] \frac{\partial U}{\partial v}=0$

Here, $\lambda(S,v,t)$ denotes the market price for volatility. We have used $\xi = \sigma$ to bring the PDE to a familar from to that of the BS model.

### Numerical solutions and methods

Our overall goal is ot determine the 'fair' price of a European style option with strike price $K$. There are a number of ways of determining this. 

The first, and perhaps most intuitive is using Monte Carlo simulations of the underlying assets. 

Given some ensemble of (simulated) stock prices $S_T^i$, with all the same starting parameters of $S_0, v_0$, we may define the expected payoff, and thus discount price, of a call option, as:

$B_i = \exp{(-rT)}*max[0,S-K]$. (see the (de)-valuation of money as above.)

Then, we may define the _fair price_ as the mean of these expected payoffs:

$\hat{U} = \frac{1}{N} \sum_i B_i$

This remarkably only involves solving the underlying SDE of the asset, but does not need or require to find the PDE on the option itself. Furthermore, strong error estimates can be found on $\hat{U}$ as a function of $N$.

To simululate the stock prices withing a give time horizon $T$, we must use some form of discretization. The equations of the Heston model can be used either directly to do a forward Euler integration, but there are two ways of executing it: either via using normal asset prices or log-prices. 

Before we can even discuss integration methods, normal-asset space, and in log-price space, we have the 'problem' of generating correlated normally distributed variables of an arbitrary number $N$. (in this case, $N = 2$).

If we have a correlation matrix $\boldsymbol{\Sigma}$ ($N x N$) which is symetric and positive-semi-definite by definition, it is possible to find (i.e. via some decomposition scheme, i.e. Cholesky) a matrix $\boldsymbol{A}$ st. $\boldsymbol{\Sigma} = \boldsymbol{A} \boldsymbol{A}^T$. where $\boldsymbol{A}$ is lower triangular. Then, we may generate $N$ random normal variables $\vec{X}'$ correlated by $\Sigma$ via: $\vec{X}' = \boldsymbol{A} \vec{X}$ where $\vec{X}$ are just a vector of $N$ un-correlated random variables. 

I.e. in our case, we may generate $dW_s = N(0,1)_s*\sqrt{dt}, dW_v = N(0,1)_v*\sqrt{dt}$ via first generating $N(0,1)_s$, then generating another indepent random normak variable $N(0,1)_a$, and defining $N(0,1)_v = \rho*N(0,1)_s + \sqrt{1-\rho^2} N(0,1)_a$ to ensure the proper correlation. 

Simple discretization yields the Euler update scheme of:

$S_{i+1} = \mu S_i \Delta t + \sqrt{V_i \Delta t} N(0,1)_s$ 

and 

$V_{i+1} = \kappa (\theta - V_i) \Delta t + \xi \sqrt{V_i \Delta t} N(0,1)_v$

Where we generated $N(0,1)_s, N(0,1)_v$ according to the above correlation inducing scheme, with correlation coefficient $\rho$.

Notice that $V_{i+1}$ can theoretically become negative (albeit with a very small chance for small dt-s) - this is purely a consequence of discretization. We need to enforce the positivity of the volatility via either taking the absolute value $S_{i+1}$ before the next step, or via taking $max(V_{i+1},0)$ ('ReLu'). 

Itô's Lemma can be used to evolve the log-prices instead. A quick computation of $d \log(S)$ yields us the discretization scheme of:

$S_{i+1} = S_i \exp{((\mu -0.5 V_i) \Delta t + \sqrt{V_i \Delta t} N(0,1)_s})$ 

$V_{i+1} = \kappa (\theta - V_i) \Delta t + \xi \sqrt{V_i \Delta t} N(0,1)_v$

Both have differenent advantages. Generally speaking, evolving the log-prices is more accurate (for instance, it can not produce negative stock prices), but it is computationally more expensive. 

There are other discretization schemes. Of noteworthiness, we have methods that use a higher-order expansion of a stochastic differential, i.e. iterative applications of Itô's Lemma. An example of such a method - the so called _Milstein Scheme_ can be found at [a document written by Fabrice Douglas Rouah](https://frouah.com/finance%20notes/Euler%20and%20Milstein%20Discretization.pdf). Our codebase does not implement this method. 

### Results of Euler-based Monte Carlo Runs

We first began to explore the effects of varying $\rho$ and leaving all other parameters constant. The following plots were produced using parameters typical of post-2015 (but pre-covid) stocks. Each Monte Carlo simulation bellow was run with 8x160000 paths, taking an approximate total of 8 seconds to run for all values of $\rho$ on my current laptop. 

![rho_effects](https://github.com/ArchHem/Julia_exercise_projects/blob/main/fn_simulations/fn_plots/Heston_rho_dependence_3.png)

As we can see, a larger correlation leads to stock prices generally trending to _lower_ values. The reason for this is relatively simple: a postive correlation means that a stochastic increase in stock price is likely to be accompanied by an increase in volatility, which in turn will mean that _on average_ the evolution of the log-stock prices will result in lower values, as we can see in the Euler step above. 

![S_K_call](https://github.com/ArchHem/Julia_exercise_projects/blob/main/fn_simulations/fn_plots/heston_put_K_S.png)

Distribution of numerically estimated (20000 simulations/point) put option distribution as a function of strike $K$ and current stock price $S$. 40x40 grid.

![S_K_call](https://github.com/ArchHem/Julia_exercise_projects/blob/main/fn_simulations/fn_plots/heston_put_V_K.png)

Distribution of numerically estimated (20000 simulations/point) put option distribution as a function of strike $K$ and starting volatility $V_0$. 40x40 grid. Note relatively higher amounts of noise compared to previous scheme - it is advised we use more samples.

The dependence between the 'volatility of volatiliy' and the fair option price is a lot more subtle then the other relations (we will see in the bellow section).

![XI_K_call](https://github.com/ArchHem/Julia_exercise_projects/blob/main/fn_simulations/fn_plots/Heston_call_Xi_K.png)

Perhaps the most important relation is between the strike price, the long-standing stable volatility $\theta$ and the strike price K. For a put option of fixed lifetime T, we get the price surface of:

![Theta_K_put](https://github.com/ArchHem/Julia_exercise_projects/blob/main/fn_simulations/fn_plots/Heston_put_Theta_k.png)

Similar plots are generated for the call options, but we are not including them here due to redundancy.

### 'Analytic' estimation of option pricing

The PDE predicted by the Heston model can be solved in a number of ways. In the original paper, Heston proposes a 'guess' solution in terms of the new variables, defined as:

$\tau = T-t, X = \log{S}$

We then propose, based on similar results in the BS model, that the characteristic function of the asset's price $X$ has the form:

$f(\tau, x, v)=\exp \{\alpha(\tau ) v +\beta(\tau)\} \exp \{i \phi x\}$ where $\phi$ is the transformed variable. 

For more details on how this solution is guessed from the log-PDE's form, see [this](https://uregina.ca/~kozdron/Teaching/Regina/441Fall14/Notes/L35-Nov28.pdf) paper by Prof. Michael Kozdron. 

Having obtained exact functional form of $f(\phi, X, v)$ (nor represented here due to sheer size) it can be shown that the call option price can be [represented](https://www.sciencedirect.com/science/article/abs/pii/S0304405X99000501) as:

$C = \Pi_1 S + exp{(-r\tau)}\Pi_2 K$

where we can express $\Pi_j$ as integrals of the characteristic function st.:

$\Pi_1 = 0.5 +\frac{1}{\pi} \int_0^{\infty} \Re(f_{\tau}(u) exp{(-i u \log{K})} / (i*u)) du$

and as: 

$\Pi_2 = 0.5 +\frac{1}{\pi} \int_0^{\infty} \Re(f_{\tau}(u-i) exp{(-i u \log{K})} / (i u f_{\tau}(-i))) du$

These integral can be, in theory, be evaulated numerically by using quad integrators or similar. However, if we aim to evalute a _range_ of strike prices $K$, we are much better of thinkinf of this as an Fourier transform problem, where then we can employ FFT to evaluate a range of strike prices. 

A number of [ways](https://engineering.nyu.edu/sites/default/files/2018-08/CarrMadan2_0.pdf) had been developed to use FFT for this problem (which has a considerably better scaling!). The main problem is the fact that there is a singularity at $u = 0$, which must be dealt with numerically, either by introducing a damping factor or by manually overwriting that frequency. 

The method we used is known as the Carr-Madan method. This involves introducing a damping factor $1<\alpha<2$ and its exponental into the FT which removes the singularity as zero. Further rescalings are needed to account for the numerical nature of FFT. A good overview of the analytical side - i.e. how the characteristic function can be used to approximate the option price -is provided at this [link](https://gregorygundersen.com/blog/2023/01/26/carr-madan/) and [the paper by Madan and Bakshi](https://www.sciencedirect.com/science/article/pii/S0304405X99000501). The way we implement this can be found in the 'heston_model.jl' file.

All that remains now is to check whether the method of the characteristic function's predictions for the option prices (call, in this case). 

![heston_fft](https://github.com/ArchHem/Julia_exercise_projects/blob/main/fn_simulations/fn_plots/Heston_FFT_call_02.png)

As we can see, the MC and FFT methods overlap (almost) perfectly. In fact, only for options far-in-the-money (i.e. $K >>> S_0$) is there any notable difference where both suffer from some-fat tail effects. 

The FFT method offers a number of improvements over the MC method of option pricing: most notably, it is easier to 'invert', that is, determine a (single) unknown Heston parameter from the option price (assuming it was priced according to the said model).

### Parameter estimation in the Heston model

Estimating the parameters of the Heston model from historic asset prices is much more complicated than that of the Black-Scholes model and is generally much less accurate, as the volatility $V(t)$ is not directly observable in the market.

The method I implemented is described in a [paper](https://www.valpo.edu/mathematics-statistics/files/2015/07/Estimating-Option-Prices-with-Heston’s-Stochastic-Volatility-Model.pdf) by Robin Dunn at el. To estimate the parameters, we discretize the SDE-s of the asset price and of the volatility, via:

$S_{t+1}=S_t+r S_t+\sqrt{V_t} S_t Z_s$

and 

$V_{t+1}=V_t+k\left(\theta-V_t\right)+\sigma \sqrt{V_t} Z_v$

(the paper assumes that $\mu = r$ - we do not enforce this condition in later stages)

Here, we implictly set $dt =1$, i.e. we assume that the separation of our data is uniform: all fitted variables will be calculated in the chose temporal unit.

By introducing a new variable, $S_{t+1}/S_{t} = Q_{t+1}$, we may instead write, taking the correlation betwen the normal noises $Z_v$ and $Z_s$ into account:


$Q_{t+1}=1+r+\sqrt{V_t}(\rho Z_1+\sqrt{1-\rho^2} Z_2)$

and

$V_{t+1}=V_t+k(\theta-V_t)+\sigma \sqrt{V_t} Z_1$

This is a fairly interesting result. Under the discretization estimate, $V_{t+1}$ is normally distributed with mean $V_t+k(\theta-V_t)$ and variance $\sigma^2 V_t$; similarly, Q_{t+1} is normally distributed with mean $1+r$ and variance $V_{t}$. However, only $Q_t$ can be directly observed in the marked - $V_t$ must be constructed "artificially", i.e. $Q(t)$ is a _heteroscedastic_ process.

There are a number of ways of estimating $V_{t}$. The paper opts for simply taking the sample variance of $Q_{t+1}$ up to index $t$ - other, more accurate models (rolling-time-window variance calculations, GARCH models, etc) are also avaible for estimating the $V_t$ vector but also need more free parameters. Currently, we only implemented the approach laid out in the original paper, and had thus obtained a historic estimate for $V_t$. 

The actual parameter estimation once again occurs via minimizing the negative log-likelyhood. The join probability distribution of the variables $V$ and $Q$ is a bivariate, correlated normal distr. (as both are normally distributed) and thus determing the log-likelyhood analytically is actually fairly trivial. 

The actual parameter fit is performed using `Optim.jl` and the dual number automatic differentation provided by `ForwardDiff.jl`: the Hessian and the resulting errors from it is calculated similarly, using a class of higher-order duals. We had tried to fit our parameters to a ten-year long historic data of `GOOG` data but found poor convergence: likely because the underlying model estimations are incorrect (constant drift rate is most likely untrue). For shorter time periods, we found much faster and agreeable convergence.

![Q_goog](https://github.com/ArchHem/Julia_exercise_projects/blob/main/fn_simulations/fn_plots/GOOG_historic_H_Q.png)

![V_GOOG](https://github.com/ArchHem/Julia_exercise_projects/blob/main/fn_simulations/fn_plots/GOOG_historic_H_volat.png)

### The Volatility Smile

One of the biggest achievements of the Heston model is the ability to reproduce the often observed volatity smile, that is, (call) options far-out/in money have much larger implied volatities than those near it (as discussed before, the BS model implies constant volatities of ALL options of the same expiry, regardless of strike price). 

By simulating a number of prices (or calling on the implemented FFT-based method) we can generate the 'fair' option price, $H_{S,T,K,...}$ according to the Heston model. For the BS model, there exists an analytical formulae (see above) that expresses the Black-Schole predicted option price, $B_{S,T,K,\sigma}$. By treating $H-B(\sigma)$ as a rootfinding problem, we may find the volatility 'implied' by the Heston model, and plot it a as a function of the strike price $K$.

For the rootfinding, we used `Roots.jl`: while the method used is _very_ sensitive to initial guesses of the root, we only need to supply it once for a range of strike prices, as subsequent calls will use the previously detemrined root as a starting guess. This still leads space for some improvements, though, but I oculd not find any good intial estimators in my brief search. 

![rho_02](https://github.com/ArchHem/Julia_exercise_projects/blob/main/fn_simulations/fn_plots/heston_volat_curve_rho_p02.png)
![rho_00](https://github.com/ArchHem/Julia_exercise_projects/blob/main/fn_simulations/fn_plots/heston_volat_curve_rho_00.png)
![rho_m02](https://github.com/ArchHem/Julia_exercise_projects/blob/main/fn_simulations/fn_plots/heston_volat_curve_rho_m02.png)
![rho_m07](https://github.com/ArchHem/Julia_exercise_projects/blob/main/fn_simulations/fn_plots/heston_volat_curve_rho_m07.png)

All the previous plots were produced with $dt = 0.001, N_t = 500, \theta = 0.2^2, \mu = 0.05, r = 0.02, \epsilon = 0.6, k = 3.0$ while varying $\rho$.

We can also see the characteristic 'skew' of this smile (i.e. its non-symmetry).

### Performance improvements for the future

There is much to be improved in the performance of the code. Currently, the code leverages only the CPU and while it uses some parallelism (such as batching the various MC calls) and while it mostly leverages SIMD, it does not make use of some other speedups (i.e. using `@fastmath`, `@inbounds`, `@views` (where applicable) or even being more careful with abusing the dot operator). Minor speedups could be caught by stronger type declarations and/or more hints to the complier. 

In case of rewriting this mini-project, two major fields are avaible for improvement. The first one using the GPU for the Monte-Carlo integration and potentially for the FFT step. This could be done relatively painlessly with high-level `CUDA.jl` usage as pretty much all used functions have a GPU implementation. On this note, this would also require us to switch to `Float32` for all parts of the code, which would also lead to improvements. 

The second one relates to the Heston characteristic function. During calling to determine option prices, we can see that it is actually only evaluated for numbers whose real part is _zero_. That means that our current implemenation wastes the complex datatype and effectively wastes half of the memory! This could be fixed by declaring a custom function that evalutes the function for imaginary numbers only, either by discarding the real part under the hood, or by just simply calculating it analytically and (re)-defining it. 


## Quasi-general raycasting in General Relatvity

-Home built, mid-performace (CPU) GR raycaster and visualizer, centered around the ADM formalism of General Relativity

-For the relevancy to this project, the formalism by Richard Arnowitt, Stanley Deser and Charles W. Misner 'decouples' the timelike coordinate from the rest of the spacelike coordinates, thus reducing the number of euqtions of motion from 8 to 6, by eliminating the geodesic equation of the temporal coordinate. 

-Entirely coordinate-indepent: only the description of (spacelike) infinity is required to render an image. Visualization of timelike paths/images is also supported.

-Main `struct` is entirely symbolic, and pre-computes the symbolic coordinate evolution equations. Bencharmking suggests that is this is roughly 10% improved over the standard, optimized way of integrating the EOMs.

-Future improvements could include RGB-translated redshift (which is computed, but not used at the moment) or a metric-independent, four-momenta based ray termination algorithm.

-Tested metrics include the Schwarschild, the Kerr-, Minkowski and Alcubierre (sublight) metrics.

Example renders:

### Schwarschild blackhole between two checker-texture plates, located at the z-like horizon

![tracker_texture_SCH_blackhole](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/f3a1fb00-ccdc-4916-be72-3ae92f2a87d2)

### Schwarschild blackhole in low orbit around an imaginary planetary system with rings (note the Einstein ring)

![ADM_Imaginary_LEO](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/adbb68be-ef65-49ed-87d4-9d558967faac)

### Kerr Blackhole inside the same imaginary low-orbit texture as the Schwarschild one (a = 0.95)
![ADM_KERR_high_res](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/57496b18-d90f-47f8-818f-be4619b9bf1f)

### Kerr blackhole inside same checkerboard texture (a = 0.95)

![ADM_KERR_tracker](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/93f5792d-22de-42c1-95ff-713e59157573)

### Alcubierre metric (position chosen such that null geodesics in flat space would reach the "buble" at the time of taking the photo)

![ALC_tracker_test_HQ](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/71d61bce-074a-49f1-bca3-49963f17b26f)


## Machine learning - analyzing the MNIST dataset using home-built `neural_network struct`.

-Home-built, medium-performance feed-forward neural network implementation using base Julia library

-3 layer network capable of 96% accuracy on the test-split of the MNIST dataset. 

-main `struct` capable of constructing any feed-forward network architecture

-Potential future improvements include implementing other gradient-descent methods such as ADAM, and type declaration in main `struct`, or symbolic differention using `Symbolics.jl`/forwardiff, to achieve higher performance. (implementation already uses dual-like reversediff - unlikely to gain much perf.)

