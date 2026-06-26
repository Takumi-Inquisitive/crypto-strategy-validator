"""Hidden Markov Model regime detection — done CAUSALLY (no lookahead).

The trap in most HMM trading demos: they fit the HMM on the whole price history
(Baum-Welch + Viterbi use the entire sequence), then backtest on those regime
labels. But a label at bar t computed with knowledge of bars t+1..N is cheating —
it inflates results massively. That single mistake is why those dashboards glow.

Here we do it honestly:
  * Fit HMM parameters on TRAINING data only.
  * Standardise features with TRAINING stats only.
  * Map each hidden state to bull/bear/neutral using TRAINING returns only.
  * Infer the live state with FILTERING (forward algorithm) — at each bar we use
    only information up to and including that bar. No peeking ahead.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def features(df: pd.DataFrame) -> pd.DataFrame:
    """The three features the method uses: returns, bar range, volume change."""
    ret = df["close"].pct_change()
    rng = (df["high"] - df["low"]) / df["close"]
    volch = df["volume"].pct_change().replace([np.inf, -np.inf], 0)
    f = pd.DataFrame({"ret": ret, "rng": rng, "volch": volch}).fillna(0.0)
    return f


class CausalHMM:
    def __init__(self, n_states=4, seed=42):
        self.n_states = n_states
        self.seed = seed
        self.model = None
        self.mu = None
        self.sd = None
        self.state_dir = None  # +1 bull / -1 bear / 0 neutral per state

    def fit(self, train_df: pd.DataFrame):
        f = features(train_df)
        self.mu, self.sd = f.mean(), f.std().replace(0, 1.0)
        X = ((f - self.mu) / self.sd).to_numpy()
        self.model = GaussianHMM(n_components=self.n_states, covariance_type="diag",
                                 n_iter=50, random_state=self.seed)
        self.model.fit(X)
        # classify states by their average REAL return on the training set
        states = self.model.predict(X)  # in-sample labelling only, for direction map
        ret = train_df["close"].pct_change().fillna(0).to_numpy()
        self.state_dir = {}
        for s in range(self.n_states):
            m = ret[states == s].mean() if (states == s).any() else 0.0
            self.state_dir[s] = 1 if m > 0 else (-1 if m < 0 else 0)
        return self

    def _emission_logprob(self, x):
        # log N(x; mean_s, diag covar_s) for each state.
        # hmmlearn returns full (n_states, n_feat, n_feat) covars even in diag
        # mode, so pull the diagonal.
        means = self.model.means_
        covars = self.model.covars_
        diag = np.array([np.diag(covars[s]) for s in range(self.n_states)])
        out = np.empty(self.n_states)
        for s in range(self.n_states):
            var = diag[s]
            out[s] = -0.5 * (np.sum((x - means[s]) ** 2 / var)
                             + np.sum(np.log(2 * np.pi * var)))
        return out

    def filter_states(self, df: pd.DataFrame) -> pd.Series:
        """Causal forward-filtering: state at t uses only obs up to t.
        Returns a direction series in {-1,0,1} (bull/neutral/bear) per bar."""
        f = features(df)
        X = ((f - self.mu) / self.sd).to_numpy()
        log_tr = np.log(self.model.transmat_ + 1e-12)
        log_alpha = np.log(self.model.startprob_ + 1e-12) + self._emission_logprob(X[0])
        dirs = np.zeros(len(X))
        dirs[0] = self.state_dir[int(np.argmax(log_alpha))]
        for t in range(1, len(X)):
            # predict step + update with current emission (filtering, causal)
            m = log_alpha[:, None] + log_tr            # (prev, next)
            log_alpha = np.logaddexp.reduce(m, axis=0) + self._emission_logprob(X[t])
            log_alpha -= log_alpha.max()               # stabilise
            dirs[t] = self.state_dir[int(np.argmax(log_alpha))]
        return pd.Series(dirs, index=df.index)
