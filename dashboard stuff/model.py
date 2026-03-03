import pickle
import numpy as np
import arviz as az

class BayesianForecaster:
    def __init__(self, trace_path: str, metadata_path: str):
        self.trace = az.from_netcdf(trace_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def predict(self, horizon: int, credible_interval: float = 0.94):
        posterior = self.trace.posterior

        # Extract posterior samples (flatten chains)
        intercept = posterior["intercept"].values.flatten()
        slope = posterior["slope"].values.flatten()
        amplitude = posterior["amplitude"].values.flatten()
        period = posterior["period"].values.flatten()
        sigma = posterior["sigma"].values.flatten()

        n_train = self.metadata["n_train"]
        future_t = np.arange(n_train, n_train + horizon, dtype=float)

        # Generate posterior predictive samples for each future time step
        rng = np.random.default_rng(0)
        n_samples = len(intercept)
        predictions = np.zeros((n_samples, horizon))

        for i, t_val in enumerate(future_t):
            mu = intercept + slope * t_val + amplitude * np.sin(2 * np.pi * t_val / period)
            predictions[:, i] = rng.normal(mu, sigma)

        # Summarize
        alpha = 1 - credible_interval
        results = []
        for i in range(horizon):
            samples = predictions[:, i]
            results.append({
                "day": int(future_t[i]),
                "mean": float(np.mean(samples)),
                "lower": float(np.percentile(samples, 100 * alpha / 2)),
                "upper": float(np.percentile(samples, 100 * (1 - alpha / 2))),
            })

        return results
