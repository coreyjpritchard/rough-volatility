# Roadmap

The session ladder. Levels are defined in `~/phd/CONVENTIONS.md`: 0 play, 1 reproduce,
2 implement, 3 extend. A box is ticked when Corey has worked through the session.

- [ ] **00 What does H do to a path, and what is H for the S&P 500?** (level 0).
  `sessions/00_what_is_rough.py`. A slider on H morphs an fBm path; the Gatheral,
  Jaisson and Rosenbaum (2018) estimator on S&P 500 realised volatility gives Ĥ = 0.149;
  the same estimator recovers known H on fBm and about 1/2 on Heston.
- [ ] **01 When does the estimator lie?** (level 2). Add microstructure noise and
  measurement error to simulated log volatility and watch Ĥ move. Vary sample length,
  compare daily range estimators with intraday realised variance, and separate the two
  biases seen in session 00 (noise pulls Ĥ down, intraday averaging pushes it up).
  Implement the critique of Fukasawa, Takabatake and Westphal (2019), *Is volatility
  rough?*, and ask whether H stays well below 1/2 after correcting for measurement error.
- [ ] **02 Rough Bergomi by the hybrid scheme** (level 2). Implement the hybrid scheme of
  Bennedsen, Lunde and Pakkanen (2017) for the Riemann-Liouville fBm driving rough
  Bergomi (Bayer, Friz and Gatheral 2016). Check the variance of the Volterra process
  against its closed form and that the discounted price is a martingale.
- [ ] **03 The ATM skew power law** (level 1). Price vanillas under rough Bergomi by Monte
  Carlo and show the at-the-money skew falling like T^(H - 1/2), against the flat
  short-end of Heston. Reproduce one figure from Bayer, Friz and Gatheral (2016) side by
  side with the original.
- [ ] **04 Rough Heston and its characteristic function** (level 2). Solve the fractional
  Riccati equation of El Euch and Rosenbaum (2019) by the Adams scheme, price by Fourier
  inversion, and compare with Monte Carlo.
- [ ] **05 A level 3 question** (level 3). Proposed: does a single H fit both the time
  series of realised volatility (session 00) and the implied volatility skew
  (session 03)? Second candidate: is H stable across assets (the Oxford-Man library has
  31 indices) and across calm and stressed regimes?

Forward link: the sibling repository `deep-hedging`, session 04 (deep hedging under rough
Bergomi, after Horvath, Teichmann and Zuric 2021), imports `roughvol` for the rough
Bergomi simulator built in session 02 here.
