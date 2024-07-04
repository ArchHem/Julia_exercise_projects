 # Julia_exercise_projects
Number of exercises in Julia.

## Stochastic Calculus Projects

### Mathematical background and definitions used

_Martingale_: A martingale is an (ordered) sequence of random variables where the expectation value of the next random variable equals that of the last in the sequence, regardless of the sequence leading up to it. I.e. in terms of conditional probabilities, given a sequence of random variables $S$ and last element $Z_i$, we have that $E(Z_i|S) = Z_i$.

_Wiener process_: A continious type of stochastic process. The exact definition will not be quoted here: what we need to know is that for a Wienner process, $W(t)$, for any two times $T + \delta T > T$, we have that $W_{t+\delta T} - W_{t} \propto N(0, \delta T)$ indepent of $T$, i.e. it 'locally' varies as a normal distribution with spread $\sqrt{\delta T}$. Furthermore, we set $W(0) = 0$. (i.e. it satisfies the conditions of being a continious martingale)

_Itô's Lemma_: Can be thought of as the "chain rule" for functions involving Itô processes. 

In the simplest case, iven a variable $X$ that is undergoing a drift-diffusion process, we have that $dX(t) = \mu(t) dt + \sigma(t) dW(t)$ i.e. it is undergoing a deterministic 'drift' proportional to $\mu(t)$ and a nondeterministic Wienner process $W(t)$ proportional to $\sigma(t)$. Letting $f = f(t, X)$, Itô's Lemma states that the differential of f is then $df = (\frac{\partial f}{\partial t} + \mu(t) \frac{\partial f}{\partial X} + \frac{\sigma(t)^2}{2}\frac{\partial^2 f}{\partial x^2}) dt + \sigma(t) \frac{\partial f}{\partial x} dW(t)$. The exact derivation of the lemma is not presented here: however, one can arrive to the lemma by just Taylor expanding $F(X,t)$ to second order and using that for Wiener processes, $dW(t)^2 = dt$ (see above), and disregarding any element that has larger than linear $dt$ scaling.

Itô's Lemma is applicable not just for variables $f$ that are undergoing a drift-diffusion process, but also to a wider range of stochastic processes, such as the asset movement assumed by the Black-Scholes model. This is important to keep in mind if one aims to model fat-tail walks which might be a better represenation of asset price movement. The generalized Itô Lemma is beyond the needs of this document and is typically expressed in integral formalism instead. 



### Brownian motion and analysis

We define a _discretized_ Brownian motion $W_i$ via the following criteria: $W_{i+1}-W_{i} \propto N(0,1)$. I.e. there is no correlation between subsequent "steps". Itr is a useful toy model for other, more complicated models.

Example of 'integrated' Brownian paths in one dimension ($\Delta y \propto N(0,1)$.
![BrownianMotion](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/943d531b-9e2f-4440-9a83-6f0c4c825979)

### Example of 1D brownian motion distribution.

Since in 1 dimensions, there exists a representation for the PDF of the displacements of (continious) brownian paths, we can compare our results to such. Such a PDF is in fact a normal one, with $\propto N(\mu = 0,\sigma = \sqrt{T})$ where T is the total intergated time domain. In fact, this form of the solution is a self-consistency check: for a particular time-step $\Delta t$, we need to draw from a normal distribution with sigma $\sqrt{\Delta t}$ to recover the correct distribution for the (local) estimate. 

![final_positions_pdf](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/07863b1f-7797-4e25-821e-62ccc0eccf29)

### Drift Processes and the Black-Scholes Model

The Black-Scholes model aims to model a market containing at least one risky asset (typically a stock) and its financial derivatives. It makes a number of assumptions, most important of which is the fact that the market is frictionless: there are no associated trading costs, no taxes etc, and that the asset pays no dividends or other fees to its owner. 
We furthermore assume that there exists a riskless form of bond - i.e. which could be a bond offered by a central bank - which offers an interest rate of $r(t)$ on our money. 

To derive the Black-Scholes PDE, we assume that we have a _riskless_ portfolio comprised of a asset $S$ (typically a stock) and an European style option $V$ that is dependant on that asset. We want to construct a portfolio that is _self financing_, that is, any loss of value on the stock's price is offset by the change of the option we are holding. Mathematically, if we denote the portfolio value as P, this means that:

We write the change in the value of the portflio as: $dP = \Delta dS + dV$ i.e. we hold $\Delta$ amounts of the stock. For a self-financing portfolio, the change in the portfolio's value should be equal to that of a riskless bond, i.e. $dP = r P dt$, as there exists no riskless way of turning a profit larger than that of the riskless bond (no arbitrage) 

To write this as a PDE, we apply Itô's Lemma to dV, one gets that, assuming a stochastic process on the underlying asset, $dS(t) = \mu S(t) dt + \sigma S(t) dW(t)$ where $W(t)$ is a Wiener process, that $dV = (\dfrac{\partial V}{\partial t} + \mu S(t) \dfrac{\partial V}{\partial S} + \sigma^2 S(t)^2\frac{1}{2}\dfrac{\partial^2 V}{\partial S^2}) dt + \sigma S(t) \dfrac{\partial V}{\partial S} dW(t)$. Collecting all terms of $dW(t)$ that result from expanding $dP$, one gets that they equal $(\sigma S(t) \dfrac{\partial V}{\partial S} + \Delta \sigma S(t)) dW(t)$. Since our model is assumed to be _riskless_ this term must vanish, i.e. $\Delta = - \dfrac{\partial V}{\partial S}$, imposing a further constraint. 

in total, we end up with: $dP = r P dt = (\dfrac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S(t)^2 \frac{\partial^2 V}{\partial S^2}) dt$. Since our portfolio is set up such that $P = \Delta S + V$, this yields the PDE: 

$\dfrac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S(t)^2 \frac{\partial^2 V}{\partial S^2} = r (V(t)- \dfrac{\partial V}{\partial S} S)$ (having used the condition on $\Delta$).

This is a remarkable result, as by solving the resulting PDE under the correct boundary conditions, the Black-Scholes model predicts that for a riskless portfolio there exists a _singular_ 'fair' price of a European style option assuming we have perfect information on the involved parameters. 

We should note that the above derivation, strictly speaking, is incorrect. It must be noted that generally speaking, a portfolio $P = \Delta S + V$ is _not_ self financing, but one arrives to the same results (PDE) using the more general portfolio $P = \Delta S + \Gamma V$ which can be made generally self-financing and riskless. Another problem arises from the 'delta on delta' problem: our derivation had implicitly assumed that the Itô differential of $\Delta$ is 0, whereas in reality its fairly obvious that the portfolio and its composition must change in reaction to the market, which is reflected in the amount of stocks we are holding at present. 


### Simplest Black-Scholes Monte Carlo Simulation

![BS_initial](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/c9f18374-a60b-4f38-8e14-e81ad6024e89)

The above plot was produced entirely artifically, using drift coefficient $\mu = 0.001$ and quasi-volatity $\sigma= 0.001$. The underlying method exploits vectorization within a simple batch and multithreading between the various batches. Practical limitations and SIMD-like LLVM means that multi-batching is not particularly needed under a few million events - we are just demonstrating a future possibility.

### Calculating the volatility of a stock as implied by historical closing prices

A parameter estimation was run on the real-world data of "OTP Bank Nyrt" (among other central european stocks) using Yahoo Finance daily data from 2015 to 2020 (eg: https://finance.yahoo.com/quote/OTP.F/history). While in case of our toy BS model, we can estimate $\mu$ and $\sigma$ using just the ratio of $\frac{S_{i+1}}{S_i}$, we still need to account for the fact that our data is unevenly distributed in time i.e. $\delta t$ is in general not constant. 

![OTP_price_history](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/24da1590-f158-438c-9862-69524f5ff16e)

To come around this fact, we can use that fact, that according to the BS model, the log-returns are normally distributed: $\ln(S_{\delta t + T}/S_T) \propto N((\mu - \frac{\sigma^2}{2}) \delta t, \sigma^2 \delta t) = P_i$. We can thus reformulate the problem as a likelyhood maximization problem: given historic time series data $S_i$ and $t_i$ what parameters $\mu$ and $\sigma$ maximize the likelyhood ($\prod_i P_i$) this particular "path" occuring? To reduce complexity and computational costs, one can reformulate this as a minimization of the negative log-likelyhood function. In this particular case, we have an analytical formula for the log of the likelyhoods, so the actual minimazation is straightforward. 

Using daily closing prices as our 'stock' prices in the above example, we determine that the values of $\mu$ and $\sigma$ best maximize the likelyhood under the BS model are $\mu = 4.389375217535964$
and $\sigma = 2.87575294562171$ for this particular time period, calcuated in an underlying time units of one year.

We must stress that the resulting estimate is still highly inaccurate, as the underlying currency used was in HUFs which had a large inflation in this period. Proper analysis would need to performed on hsitoric, instaneous USD-HUF exchange rates instead. Longer non-trading periods that are present as separate days will heavily influence the historic, annual drift rate and is likely to result in severe errors in the estimation. We also know that the underlying BS model's assumption of constant historical volatility and drift rate are incorrect so these are just values that best conform to this particular model. 

In reality, volatity is not constant in time and real-world data implies different volatities for options with different strike prices and maturity times. Analysis of such options (inverting for the implied volatility by some model) is paramount. However, the very existence of a non-constant volatility surface proves that the Black-Scholes model is fundamentaly unable to reproduce the conditions found in real markets (or at least that not all traders follow the 'fair' pricing predicted). 

### Option pricing under the BS model

As mentioned before, there exists closed-form solutions to the Black-Scholes PDE for a European style option with maturity time T and strike price K. Taking $r, \sigma$ and $\mu$ to be constants, we apply the boundary conditions of:  

$V(S = 0,t) = 0$ (for a worthless stock at present, the value of the option is zero as per the Black-Scholes model it cant gain any value)

$V(S,t) = S - K$ as $S => \infty$ (for an infitely valuable stock, the price of call option is independent of maturity time).

$V(S,T) = max[0, S-K]$ (i.e. the option (not) being exercised at maturity)

To solve the PDE, we can use the fact that one may solve the PDE under $r = 0$ and 'upscale' the resulting solution G(t,S) to the correct solution via $V(t,S) = \exp{(r (T-t))} G(t, S \exp{(r(T-t))})$. This can be verifed by a simple chain rule, but 'physically' what this represents is the (de)-valuation of money due to the existence of a safe bond and its interest rate (as if we deposited some money $M$ during the creation a bond of maturity time $T$). 

Using this, one may write out the stochastic differential of $G(t,S)$ as $dG = \sigma S(t) G(t,S) dW + (\dfrac{\partial G}{\partial t} + \mu S(t) \dfrac{\partial G}{\partial S} +  \sigma^2 /2 S(t)^2 \dfrac{\partial^2 G}{\partial S^2}) dt$. Since the option obeys the Black-Scholes's PDE with $r = 0$, the dt term vanishes (the deterministic part of the SDE is independent of $\mu$ as proven before, so we may set it to $\mu = 0$), and thus we are left with:

$dG = \sigma S(t) G(t,S) dW$ 

So we had shown that it is a martingale as it has no explicit time differential present! We then may take its expection value (which is stable for some fixed time, as per defintion). Hence we have that $E(G_0) = E(G_T)$ and arrive to:

$G(0,S(0)) = E(max[0,S(T)-K])$ (having used our boundary condition on the expiry)

We have used the fact that at the time of creating the option, the price of the stock, $S_0$, is known at the present. 

The calcuation of this expectation value is a bit tedius and will not be done in its full extent here. The only other information that we need is that $S(t) = S_0 \exp{(\sigma W(t) - \sigma^2 /2 t)}$ (as explicitly assumed by the Black-Scholes model). By directly evaluting the involved quantities and making use of the intermidate variables, which we define as:

$d_{+}=\frac{1}{\sigma \sqrt{T-t}}\left[\ln \left(\frac{S(t)}{K}\right)+\left(r+\frac{\sigma^2}{2}\right)(T-t)\right]$

and

$d_{-}=d_{+}-\sigma \sqrt{T-t}$

We arrrive to the 'fair' European call option price of:

$V_c \left(S(t), t\right) = N\left(d_{+}\right) S(t) -N\left(d_{-}\right) K e^{-r(T-t)}$

where N is the cumultative of the unit normal function.

The price of the corresponding put option is then: 

$V_p \left(S(t), t\right) = N\left(-d_{-}\right) K e^{-r(T-t)}-N\left(-d_{+}\right) S(t)$

We can notice that there is a form of symetry in these expressions. This is due to the (implicit) relation between Europen style put and call options, which can be written as, for put and out option prices at the maturity time $T$, with forward price $F$ and strike price $K$, as: 

$C(t)-P(t)= \exp{(-r(T-t)} \cdot(F(t)-K)$

I.e. as we approach maturity time, $t = T$ the difference of the forward price and strike price must converge to the difference of call and put option prices. The forward price of the asset itself, $F$, can be determined using the inverse of the exponential discount factor multiplying the current asset price $S$ (due to the assumed no-arbitrage condition).

Hence, we may rewrite this as:

$C(t)-P(t)= S(t) - \exp{(-r(T-t))} \cdot K$ from which our symetry follows.

### Volatility surfaces and checking for self-consistency

As derived above, assuming that the real-world market operates according the BS model predictions, it is possible to numerically extract the volatility implied by a given option of strike price $K$, maturity time differential $T-t$ and current underlying asset price $S(t)$. 

If we lived in a world where the BS model is perfect, i.e. we have constatnt historic volatility and all traders followed the derived 'fair pricings' the volatility implied by the options should be constant and equal to the historic volatility as implied by the stock's price changes (see above how it can extracted using LLH minimization). 

For our analysis, we had choosen GOOG, which has widely avaible data and is traded in USD, and has no voting rights (discarding any additional hidden value thus). Most of the data was pulled via Yahoo finance's python public API or was generated via a private nasdaqlink API python script (not part of the repo due to security reasons).

Our script has used the compound interest rate of 3-month US treasury bonds as basis for the annual risk free interest rate $r$. 

For put orders, by numerically inverting the BS-predicted prices, we find a näive volatility surface of:

![BrownianMotion](https://github.com/ArchHem/Julia_exercise_projects/blob/main/fn_simulations/fn_plots/puts_vol_surface.png)

Which exhibits the empirically observed 'volatity smile' and its (slight) skew in terms of strike prices: as evident, its not a plane as predicted by the BS model. 

We have furthermore ran our LLH minimization algorithm on the historic closing prices of GOOG and had determined its historic volatity from 2023-01-01 to 2024-06-23 to have been:
$\sigma_h = 0.33813$.

The volatility surface above assigns the implied volatility $\sigma_i$ for each put option with strike value $K$ and time-to-maturity $T-t$. The numerical artefacts of zero volatility are due to the root-finder involved and are discarded for future analysis. The whole of the surface has an average implied volatility of $\sigma_i = 0.462605$ which is clearly different from the historic one! 

The obvious conclusion is that traders dont follow the naive BS model, especially on longer timescales, as the involved quantities of $\sigma, r$ are not constant in time. However, any modificatiosn to the BS model that accounts for the time dependence of these parameters will only cause change on longer timescales, which means that for small timescales, we still expect a stronger match between the BS model and the true mechanisms. 

If we lower our time-window, and only consider options with remaining maturity time lss than 0.2 years, we remove some of this effect, and end up with an average implied volatility of $\sigma_i = 0.378286$, which is a much closer match as expected. However, it is apparent that the BS model is not very applicable nor self consistent in the real world, even on shorter timescales!

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

Here, $\lambda(S,v,t)$ denotes the market price for volatility. 

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

$S_{i+1} = S_i \exp{(\mu -0.5 V_i) \Delta t + \sqrt{V_i \Delta t} N(0,1)_s}$ 

$V_{i+1} = \kappa (\theta - V_i) \Delta t + \xi \sqrt{V_i \Delta t} N(0,1)_v$

Both have differenent advantages. Generally speaking, evolving the log-prices is more accurate (for instance, it can not produce negative stock prices), but it is computationally more expensive. 

There are other discretization schemes. Of noteworthiness, we have methods that use a higher-order expansion of a stochastic differential, i.e. higher order versions of Itô's Lemma. 


### 'Analytic' estimation of option pricing





### Parameter estimation in the Heston model

### GARCH models






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

