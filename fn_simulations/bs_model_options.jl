using ProgressBars, StaticArrays, Plots, Statistics, CSV, DataFrames, Statistics, Dates, Optim, LaTeXStrings, SpecialFunctions, Roots
plotlyjs() 

#manually taken at the time of writing
US_3_moth_r = 0.0537
#compound.... at continous rate
US_Y_r = log(1+US_3_moth_r)*4

function normal_cdf(x)
    z = (1 + erf(x/sqrt(2)))/2
    return z
end

function fair_price_call(sigma::Float64,r::Float64,asset_price::Float64,strike_price::Float64,time_to_expiry::Float64)
    #Assumes that the time to expiry is goven in terms of yearly amount
    d_p = 1/(sigma*sqrt(time_to_expiry)) * (log(asset_price / strike_price) + (r + sigma^2 / 2)*time_to_expiry)
    d_m = d_p - sigma * sqrt(time_to_expiry)

    C = normal_cdf(d_p)*asset_price - strike_price*normal_cdf(d_m)*exp(-r*time_to_expiry)
    return C
end 
 
function fair_price_put(sigma::Float64,r::Float64,asset_price::Float64,strike_price::Float64,time_to_expiry::Float64)
    #Assumes that the time to expiry is given in terms of yearly amount
    d_p = 1/(sigma*sqrt(time_to_expiry)) * (log(asset_price / strike_price) + (r + sigma^2 / 2)*time_to_expiry)
    d_m = d_p - sigma * sqrt(time_to_expiry)

    P = normal_cdf(-d_m)*strike_price*exp(-r*time_to_expiry) - normal_cdf(-d_p)*asset_price

    return P
end 



function date_to_seconds(date::Date)
    
    date = DateTime(date)
    seconds_passed = datetime2unix(date)
    return seconds_passed
end

function log_likelyhood(logratio::Vector{Float64},delta_t::Vector{Float64},
    mu::Float64,o::Float64)

    #uses the fact that the log of the rato of S_i+1 /S_i is normally distributed with spread proportial to sqrt(delta t)

    to_sum = @. -0.5 * (logratio - (mu - o^2 /2) * delta_t)^2 / (o^2 * delta_t) - log(sqrt(2pi * o^2 * delta_t))

    LLH = -sum(to_sum)

    return LLH

end

function asset_time_series_volatility_estimator(closing_price::Vector{Float64},date::Vector{Date})
    length_of_year = 3600 * 24 * 365
    """We want output votality in per-year measure"""
    t_s = date_to_seconds.(date)

    dt_s = t_s[2:end] - t_s[1:end-1]
    
    dt_y = dt_s ./ length_of_year

    stock_n1 = closing_price[2:end]
    stock_n2 = closing_price[1:end-1]

    logratio_prices = @. log(stock_n1)-log(stock_n2)

    local_likelyhood(vec) = log_likelyhood(logratio_prices, dt_y, vec[1],vec[2])

    result = optimize(local_likelyhood, [0.02, 0.15])

    params = Optim.minimizer(result)

    return params[1], params[2]

end

function generate_vol_surface_puts(time_to_maturity::Float64,asset_price::Float64,r::Float64,strike::Float64,puts_price::Float64,volat_guess::Float64 = 0.3)

    eq_to_solve(sigma) = fair_price_put(sigma, r,asset_price,strike,time_to_maturity) - puts_price

    #use roots to invert for sigma 

    volats = find_zero(eq_to_solve, volat_guess) 


    return volats
end

function generate_vol_surface_calls(time_to_maturity::Float64,asset_price::Float64,r::Float64,strike::Float64,calls_price::Float64,volat_guess::Float64 = 0.3)

    eq_to_solve(sigma) = fair_price_call(sigma, r,asset_price,strike,time_to_maturity)-calls_price

    #use roots to invert for sigma 

    volats = find_zero(eq_to_solve, volat_guess)


    return volats
end
 


#load historic GOOG data

GOOG_closing = CSV.read("fn_simulations/fn_data/GOOG_closing.csv", delim=",", DataFrame)
GOOG_dates = GOOG_closing[!,"Date"]
GOOG_prices = GOOG_closing[!,"Adj Close"]

#get HISTORIC volatility
mu_GOOG, sigma_GOOG = asset_time_series_volatility_estimator(GOOG_prices,GOOG_dates)

#load options data

GOOG_puts = CSV.read("fn_simulations/fn_data/goog_puts.csv", delim=",", DataFrame)
GOOG_calls = CSV.read("fn_simulations/fn_data/GOOG_calls.csv", delim=",", DataFrame)

#filter out data where the bid/ask price doesnt not exist yet as it was nto yet traded
call_cond = GOOG_calls[:, "ask"] .!= 0.0 .&& GOOG_calls[:, "bid"] .!= 0.0 .&& GOOG_calls[:,"lastPrice"] .!=0.0
GOOG_calls = GOOG_calls[call_cond,:]

put_cond = GOOG_puts[:, "ask"] .!= 0.0 .&& GOOG_puts[:, "bid"] .!= 0.0 .&& GOOG_puts[:,"lastPrice"] .!=0.0
GOOG_puts = GOOG_puts[put_cond,:]

#use the mid-prices
puts_price = @. (GOOG_puts[:,"ask"] + GOOG_puts[:,"bid"]) / 2
calls_price = @. (GOOG_calls[:,"ask"] + GOOG_calls[:,"bid"]) /2


GOOG_current_price = GOOG_prices[end]

current_date = Date(2024, 6, 23)
length_of_year = 365 * 3600 * 24
calls_time_to_maturity = @. (date_to_seconds(GOOG_calls[:,"Maturity"]) - date_to_seconds(current_date)) / (length_of_year)
puts_time_to_maturity = @. (date_to_seconds(GOOG_puts[:,"Maturity"]) - date_to_seconds(current_date)) / (length_of_year)

calls_strike = GOOG_calls[:,"strike"]
puts_strike = GOOG_puts[:,"strike"]

yf_p = GOOG_puts[:,"impliedVolatility"]
yf_c = GOOG_calls[:,"impliedVolatility"]

puts_volats = generate_vol_surface_puts.(puts_time_to_maturity,GOOG_current_price,US_Y_r,puts_strike,puts_price,1.5)
calls_volats = generate_vol_surface_calls.(calls_time_to_maturity,GOOG_current_price,US_Y_r,calls_strike,calls_price,1.5)

avg_volat = mean(puts_volats[puts_time_to_maturity .<0.2])
plot0 = scatter(puts_time_to_maturity,puts_strike,puts_volats, 
xlabel = "Time to maturity (YR)", ylabel = "Put Strike (USD)", zlabel = "Implied Vol.",
color = :green, markersize = 1.0, label = "Implied Vol. Surface")











