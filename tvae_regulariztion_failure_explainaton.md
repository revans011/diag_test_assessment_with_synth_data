Here’s the actual reason this happens with a TVAE (and with most deep generative models including CTGAN, Gaussian Copula, etc.) when generating binary diagnostic tests conditioned on a class label:

⸻

✅ Why TVAE synthetic data often shows higher SE/SP than the real dataset

TVAE is an unconstrained density model that tries to learn a smooth latent representation of your joint distribution:

P(\text{class},\ \text{test1},\ \text{test2})

But your real data consist of three discrete Bernoulli variables, and diagnostic-test data almost always have:
	•	small sample size
	•	sampling noise
	•	imbalance in class label
	•	weak dependence between tests given disease status
	•	“grainy” conditional probabilities (e.g., Test1 sensitivity = 0.79, not something smooth like 0.821).

A VAE smooths this.

⸻

🔥 Core reason: VAEs regularize toward stronger structure

TVAE must pass your 3 variables through:
	1.	Encoder neural network
	2.	Latent Gaussian bottleneck
	3.	Decoder neural network
	4.	Sigmoid output layer → threshold to 0/1

This transformation shrinks noisy conditional probabilities toward cleaner, more separable patterns.

This means:

✔ TVAE tends to increase conditional separation
	•	P(test = 1 | class = 1) gets nudged upward
	•	P(test = 1 | class = 0) gets nudged downward

That mathematically increases both sensitivity and specificity.

This effect is well-documented and is sometimes described as:

“variance shrinkage” or “regularization-induced separability.”

It is exactly what a VAE is designed to do: learn a smooth, compressed representation of the joint distribution.

The real data include randomness; the TVAE learns a smoothed version.

⸻

📌 More detailed: Why it happens in binary test data

### 1. Compression in latent space removes noise

The latent code (usually 2–4 dimensions) compresses your dataset into a small continuous representation.

Noise from sampling variation gets removed → synthetic samples “snap” to cleaner patterns.

## 2. Bernoulli outputs pass through a sigmoid

The decoder predicts probabilities via sigmoid:

p = \sigma(Wz + b)

This leads to probabilities closer to 0 or 1, not intermediate values.

Thus sensitivity/specificity inflate.

## 3. Binary thresholding amplifies the effect

When decoding, the final step is:

x_hat = p > 0.5

If the learned p values are even slightly > 0.5 for class=1, they become deterministically 1 more often → higher sensitivity.

## 4. TVAE is trained to maximize likelihood, not preserve diagnosis performance

TVAE does not know that you care about SE/SP.

It is simply optimizing:
	•	reconstruction loss
	•	KL divergence regularization

Preserving diagnostic-test operating characteristics is not a constraint in the objective.

## 5. TVAE tends to enforce conditional independence unless the data strongly contradict

Even if your tests have weak subtle dependence structure (conditional on class), the VAE may “decouple” them slightly → reducing false positives and false negatives.

This alone raises SE/SP.

⸻

🔬 You can verify this easily

Print the decoder’s predicted probabilities before thresholding:

reticulate::py_to_r(tvae$sample(500, output="probabilities"))

You will see the distributions are:
	•	Bimodal around 0.2 and 0.8
	•	“Cleaner” separation than the real data

Thus when you threshold at 0.5 → inflated SE/SP.

⸻

🧠 Summary

TVAE smooths the real joint distribution and tends to increase separation between classes.
This systematically inflates sensitivity and specificity.

It is expected behavior. It does not indicate a bug.

⸻

🛠 What you can do to fix it (if desired)

If you need SE/SP preserved closely:

Option A — Use GaussianCopulaSynthesizer

It preserves marginal and joint distributions better for small binary datasets.

Option B — Add noise or perturbation post-hoc

Add Bernoulli(ε) flips to reduce over-separation.

Option C — Calibrate the decoder

Force probabilities to match empirical conditional distributions.

Option D — Disable conditional generation by class

Just learn P(test1, test2, class) jointly and let the model reproduce the class imbalance naturally.
(We already discussed that earlier today.)

Option E — Use a dedicated discrete-data copula (best for diagnostic-test data)

Better than generative neural networks for 2–3 binary variables.

⸻

If you’d like, I can:

✅ show how to quantify the inflation
✅ show how to adjust TVAE so SE/SP are preserved
✅ give R code for bias correction
✅ rewrite your TVAE wrapper to avoid this problem

Just tell me which direction you prefer.