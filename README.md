# Julia_exercise_projects
Number of exercises in Julia.

## Stochastic Calculus Projects

### Mathemitcal background and defintions used

_Martingale_: A martingale is an (ordered) sequence of random variables where the expectation value of the next random variable equals that of the last in the sequence, regardless of the sequence leading up to it. I.e. in terms of conditional probabilities, gievn a sequence of random variables $S$ and last element $Z_i$, we have that $E(Z_i|S) = Z_i$.

_Wiener process_: A continious type of stochastic process. The exact definition will not be quoted here: what we need to know is that for a Wienner process, $W(t)$, for any two times $T + \delta T > T$, we have that $W_{t+\delta T} - W_{t} \propto N(0, \delta T)$ indepent of $T$, i.e. it 'locally' varies as a normal distribution with spread $\sqrt{\delta T}$. Furthermore, we set $W(0) = 0$. (i.e. it satisfies the conditions of being a continious martingale)

_Itô's Lemma_: Can be thought of as the "chain rule" for functions involving Wienner processes. Given a variable $X$ that is undergoing a drift-diffusion process, we have that (by defintion) $dX(t) = \mu(t) dt + \sigma(t) dW(t)$ i.e. it is undergoing a deterministic 'drift', dictated by a determinstic grid proportional to $\mu(t)$ and a nondetemrinistic Wienner process $W(t)$ proportional to $\sigma(t)$) Letting $f = f(t, X)$, Itô's Lemma states that the differential of f is then $df = (\dfrac{\partial f}{\partial t} + \mu(t) \dfrac{\partial f}{\partial X} + \frac{\sigma(t)^2}{2}\dfrac{\partial^2 f}{\partial x^2}) dt + \sigma(t) \dfrac{\partial f}{\partial x} dW(t)$. The exact derivation of the lemma is not presented here: however, one can arrive to the lemma by just Taylor expanding $F(X,t)$ to second order and using that for Wiener processes, $dW(t)^2 = dt$ (see above), and disregarding any element that has larger than linear $dt$ scaling.
Itô's Lemma is applicable for variables $X$ that are not undergoing a Wiener process, but a wider range of stochastic processes. 



### Brownian motion and analysis

We define a _discretized_ Brownian motion $W_i$ via the following criteria: $W_{i+1}-W_{i} \propto N(0,1)$. I.e. there is no correlation between subsequent "steps". Itr is a useful toy model for other, more complicated models.

Example of 'integrated' Brownian paths in one dimension ($\Delta y \propto N(0,1)$.
![BrownianMotion](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/943d531b-9e2f-4440-9a83-6f0c4c825979)

### Example of 1D brownian motion distribution.

Since in 1 dimensions, there exists a representation for the PDF of the displacements of (continious) brownian paths, we can compare our results to such. Such a PDF is in fact a normal one, with $\propto N(\mu = 0,\sigma = \sqrt{T})$ where T is the total intergated time domain. In fact, this form of the solution is a self-consistency check: for a particular time-step $\Delta t$, we need to draw from a normal distribution with sigma $\sqrt{\Delta t}$ to recover the correct distribution for the (local) estimate. 

![final_positions_pdf](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/07863b1f-7797-4e25-821e-62ccc0eccf29)

### Drift Processes and the Black-Scholes Model

The Black-Scholes model aims to model a market containing at least one risky asset (typically a stock) and its financial derivatives. It makes a number of assumptions, most important of which is the fact that the market is frictionless: there are no associated trading costs, no taxes etc, and that the asset pays no dividends or other fees to its owner. 
We furthermore assume that there exists a riskless form of bond - i.e. which could be a bond offered by a central bank - which offers an interest rate of $r$ on our money. 

### Basic Black-Scholes Monte Carlo Simulation

![BS_initial](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/c9f18374-a60b-4f38-8e14-e81ad6024e89)

The above plot was produced entirely artifically, using drift coefficient $\mu = 0.001$ and quasi-volatity $\sigma= 0.001$. The underlying method exploits vectorization within a simple batch and multithreading between the various batches. Practical limitations and SIMD-like LLVM means that multi-batching is not particularly needed under a few million events - we are just demonstrating a future possibility.

### Evaluating simplest-case CVaR using numerical data

![BS_put_plot](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/6c78209d-5944-48ee-8b3e-ae4bebfd4b80)

CVaR quantifies the average expected losses beyond a certain confidence level. It can be used to quantify the worst-case-scenarios for a given put or call order, using a mass of simulated asset (i.e. in this case, stock) prices. This is either done via a) binning or b) by sorting all simulated returns and selecting the lowest (1-confidence) ratio of them.

The above plot was produced using $\mu = 0.04$, $\sigma = 0.14$, $S_0 = 1000$, $K_{put} = 1025$, r = 0.025, T = 1 (1000 timesteps) and with a confidence level of 95%, using simulated stock prices from our toy BS model. As expected,the final payouts - which are proportional to $S_T - K$ - are well aproximated to be normally distributed in this timeframe, as implied by the prediction of the Black-Scholes model, which predicts that $\ln(S_T) \propto N(\ln(S_0) + (\mu - \frac{\sigma^2}{2})T, \sigma^2 T)$. _However_, we do know that the distribution of the final stock prices as a normal distribution is only a useful approximation under certain assumptions - the BS model predicts lognormal distributions. 

To explore the disrepancy between the observed, normal and lognormal distributions, we can widen our time-horizon. As we increase our time-frame, we expect more and more disrepancies between the normal and observed distributions. The following plots were distributed with $\Delta t = 0.02, N_{it} = 200000, \sigma = 0.14, \mu = 0.04$

![normal_vs_lognormal_T_10](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/67b57289-edce-4d65-a5d0-302403febb2d)
T = 10
![normal_vs_lognormal_T_20](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/c396654a-199a-4577-b8af-bf7ddf13f228)
T = 20
![normal_vs_lognormal_T_30](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/40dfc238-f824-41a5-a166-e8f2beb677b9)
T = 30
![normal_vs_lognormal_T_40](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/28a1d44f-79d6-4c3e-b017-3917d0f34144)
T = 40
![normal_vs_lognormal_T_50](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/3259a332-7e6a-41a6-b6b1-4f8c4c9d090f)
T = 50

### WIP - Calculating the implied volatility and drift rate using historical data

A parameter estimation was run on the real-world data of "OTP Bank Nyrt" (among other central european stocks) using Yahoo Finance daily data from 2015 to 2020 (eg: https://finance.yahoo.com/quote/OTP.F/history). While in case of our toy BS model, we can estimate $\mu$ and $\sigma$ using just the ratio of $\frac{S_{i+1}}{S_i}$, we still need to account for the fact that our data is unevenly distributed in time i.e. $\delta t$ is in general not constant. 

![OTP_price_history](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/24da1590-f158-438c-9862-69524f5ff16e)

To come around this fact, we can use that fact, that according to the BS model, the log-returns are normally distributed: $\ln(S_{\delta t + T}/S_T) \propto N((\mu - \frac{\sigma^2}{2}) \delta t, \sigma^2 \delta t) = P_i$. We can thus reformulate the problem as a likelyhood maximization problem: given historic time series data $S_i$ and $t_i$ what parameters $\mu$ and $\sigma$ maximize the likelyhood ($\prod_i P_i$) this particular "path" occuring? To reduce complexity and computational costs, one can reformulate this as a minimization of the negative log-likelyhood function. In this particular case, we have an analytical formula for the log of the likelyhoods, so the actual minimazation is straightforward. 

Using daily closing prices as our 'stock' prices in the aove example, we determine that the values of $\mu$ and $\sigma$ best maximize the likelyhood under the BS model are $\mu = 0.322512$
and $\sigma = 0.31608$ for this particular time period.

We must stress that the resulting estimate is still highly inaccurate. Longer non-trading periods that are present as separate days will heavily influence the historic, annual drift rate and is likely to result in severe errors in the estimation. We also know that the underlying BS model's assumption of constant historical volatility and drift rate are incorrect so these are just values that best conform to this particular model. 



## Quasi-general raycasting in General Relatvity

-Home built, high-performace GR raycaster and visualizer, centered around the ADM formalism of General Relativity

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

-Potential future improvements include implementing other gradient-descent methods such as ADAM, and type declaration in main `struct`, or symbolic differention using `Symbolics.jl`, to achieve high performance

![ML_training performance](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/b84aff0b-d320-4706-90c8-5ed066b47079)

