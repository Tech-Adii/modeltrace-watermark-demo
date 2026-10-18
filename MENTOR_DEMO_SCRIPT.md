# Five-minute explanation for mentor

**Project:** API Model Extraction and Black-Box Watermark Evidence.

**Original AI window (8501):** The original is an 11-output Fashion-MNIST CNN. Ten outputs classify clothing normally. The special eleventh output is `Phone`, a **deliberately wrong label** trained only for secret-patch sandal images. Normal sandal images should be classified as `Sandal`.

**Copied AI window (8502):** The suspect CNN has its own weights and process. Its training labels came from the original's black-box prediction API: 2,000 ordinary image responses plus 100 deliberately stamped sandals in this *controlled exposure* experiment. It did not load the original weights.

**Demo:** Show a normal sandal in both windows and ask for predictions. Then use `demo_samples/sandal_watermarked.png` as the image in both windows. If training succeeded, each predicts `Phone`. This is an actual model output (class 10), not an interface string substitution.

**Detector (8503):** Owner queries 100 other secret-marked sandals and also 100 unstamped sandals, all through model APIs. Show the original and copied models' watermark success rates and the count of matching wrong Phone answers. Download the JSON evidence report.

**Do NOT say:** “The same wrong answer confirms it was stolen.” One shared error is not conclusive ownership proof, and the controlled experiment explicitly discloses the trigger to the extraction simulation. In the earlier experiment with 2,000 *ordinary* queries (no trigger exposure), the copied model achieved **82.41% accuracy**, **85.80% fidelity**, but **0% watermark success**.

**Say instead:** “The copied model reproduces a hidden intentionally planted behavior when that behavior is visible in API queries. Matching the behavior on many held-out secret inputs is evidence of shared behavior. The test's applicability to truly secret watermark extraction requires more experiments and independent statistical controls.”
