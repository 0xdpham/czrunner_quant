# THE BIG IDEA

A European put option gives you the right to sell a stock at a fixed price (the strike, K = 39) on a fixed date (one year from now). If the stock ends the year at 30, you can sell at 39 and gain 9. If it ends at 50, the right is worthless, so you gain 0. That's the payoff `max(K - S_T, 0)`.

The question is what that right is worth today. Nobody knows where the stock will end up, so the script answers by simulation:

1. Simulate thousands of possible futures for the stock.
2. Calculate the put payoff in each future.
3. Average those payoffs.
4. Discount the average back to today's money, since 1 dollar next year is worth slightly less than 1 dollar now.

This is called Monte Carlo pricing. It amounts to running the year 20,000 times and taking the average outcome.

# WHY HESTON?

The simplest model assumes the stock's volatility (how wildly it swings) is constant. Real markets aren't like that, because calm periods and turbulent periods alternate. The Heston model lets the volatility itself move randomly. That's why the script simulates two things at once:

- `V`, the volatility paths. Volatility gets pulled back toward a long-run level (`theta_v`) at a speed set by `kappa_v`, with random shocks scaled by `sigma_v`.
- `S`, the stock price paths. Each day's price move depends on that day's volatility.

The two random shocks are correlated (`rho`). With a negative rho, stock drops tend to come with volatility spikes, as in real markets. The Cholesky matrix is just the standard trick for turning independent random numbers into correlated ones.

# WHAT THE SCRIPT DOES, IN ORDER:

1. Defines the simulation functions (`SDE_vol`, `Heston_paths`) and the pricing function (`heston_put_mc`).
2. Sets the parameters: starting price 40.53, 5% interest rate, 500 time steps over one year, 20,000 simulated paths.
3. Generates random numbers, then simulates `V` and `S`. Both are tables with 501 rows (time steps) and 20,000 columns (paths).
4. Looks at the last row of `S` (the price at maturity on every path), computes the payoffs, averages them, and discounts.
5. Repeats the simulation with the Task 1.1 parameters, using only 100 paths.

# WHAT YOU GET WHEN YOU RUN IT:

Printed in the terminal:

```
European Put Price under Heston:  3.3077
Parity check (call - put): 3.442 vs S0 - K*exp(-rT): 3.432
European Put Price under Heston (new params, 100 paths):  3.9897
```

- Line 1 is the answer you care about: the put is worth about 3.31 per share.
- Line 2 is a sanity check. A call and a put with the same strike are tied together by a known relationship (put-call parity), so the two numbers should be close. They are, and the small gap is Monte Carlo noise.
- Line 3 is the put price under the Task 1.1 parameters. With only 100 paths it's much noisier, so don't read much into the exact figure.

Two plot windows (they appear together at the end, and the script waits until you close them):

- The first window shows 300 of the 20,000 simulated price paths on the left and the matching volatility paths on the right. It looks like a bundle of jagged lines fanning out from 40.53.
- The second window shows the same kind of charts for the 100 paths from Task 1.1.

Since the script sets a random seed (`np.random.seed(42)`), you'll get the same numbers every time you run it. Remove that line and the price will change slightly on each run. The spread of those results is the Monte Carlo error, and it shrinks as you add more paths.
