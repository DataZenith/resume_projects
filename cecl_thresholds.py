import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

def fit_beta_moments_from_pds(pds):
    """
    Method-of-moments fit of Beta(a,b) to loan-level PDs.
    """
    pds = np.asarray(pds, dtype=float)
    mu = float(pds.mean())
    var = float(pds.var(ddof=1))

    max_var = mu * (1 - mu)
    if var <= 0 or var >= max_var:
        var = 0.99 * max_var

    k = (mu * (1 - mu) / var) - 1.0
    a = mu * k
    b = (1 - mu) * k
    return a, b, mu, np.sqrt(var)

def main():
    np.random.seed(321)

    # -----------------------------
    # Simulated quarter
    # -----------------------------
    N = 6000

    # "Fat left, drops hard, not extreme"
    pds = np.random.beta(1.25, 120.0, size=N)

    # PRIOR (forecast PD distribution)
    a0, b0, mean_pd, sd_pd = fit_beta_moments_from_pds(pds)

    # Pseudo-likelihood: expected defaults
    D_star = N * mean_pd

    # POSTERIOR
    a1 = a0 + D_star
    b1 = b0 + (N - D_star)

    # Posterior thresholds
    q_low, q_high = 0.01, 0.99
    post_low = beta_dist(a1, b1).ppf(q_low)
    post_high = beta_dist(a1, b1).ppf(q_high)

    # -----------------------------
    # Plot
    # -----------------------------
    x = np.linspace(0, 0.04, 3000)
    prior_pdf = beta_dist(a0, b0).pdf(x)
    post_pdf = beta_dist(a1, b1).pdf(x)

    plt.figure(figsize=(11, 6))

    # Prior (context)
    plt.plot(
        x, prior_pdf,
        linestyle="--",
        linewidth=2,
        label="Prior: forecast PD distribution"
    )

    # Posterior (focus)
    plt.plot(
        x, post_pdf,
        linewidth=3,
        label="Posterior: after pseudo-likelihood update"
    )

    # Means and bounds
    plt.axvline(mean_pd, color="black", linestyle="--",
                label=f"Mean PD = {mean_pd:.4f}")

    plt.axvspan(
        post_low, post_high,
        color="red", alpha=0.15,
        label="Posterior 1%–99% forecast band"
    )

    plt.axvline(post_low, color="red", linestyle=":")
    plt.axvline(post_high, color="red", linestyle=":")

    plt.title(
        "Portfolio-level PD monitoring\n"
        "Loan-level PD forecasts → posterior forecast thresholds"
    )
    plt.xlabel("Latent portfolio PD (p)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    # -----------------------------
    # Save output
    # -----------------------------
    out_dir = os.path.join(
        "app", "static", "images", "projects", "risk"
    )
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        "portfolio_cecl_pd_thresholds_prior_posterior.png"
    )

    plt.savefig(out_path, dpi=200)
    plt.show()

    # Console summary
    print("\nCECL PD Threshold Demo Summary")
    print("--------------------------------")
    print(f"N loans               : {N}")
    print(f"Prior a0, b0           : {a0:.3f}, {b0:.3f}")
    print(f"Mean PD               : {mean_pd:.6f}")
    print(f"Expected defaults D*  : {D_star:.2f}")
    print(f"Posterior a1, b1      : {a1:.3f}, {b1:.3f}")
    print(f"Posterior 1% PD       : {post_low:.6f}")
    print(f"Posterior 99% PD      : {post_high:.6f}")
    print(f"\nSaved chart → {out_path}")

if __name__ == "__main__":
    main()
