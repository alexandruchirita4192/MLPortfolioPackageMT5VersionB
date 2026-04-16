# MT5 Portfolio EA Version B

Version B is a **multi-model portfolio EA** with:

- **one final executor per symbol**
- **multiple model opinions inside each symbol sleeve**
- **one final aggregated decision per symbol**

This avoids the common MT5 problem where:
- several EAs fight on the same symbol
- one strategy closes another strategy's trade
- one global “position open” check blocks unrelated strategies

Instead, Version B does this:

- **BTCUSD sleeve**
  - combines:
    - `LinearSVC calibrated`
    - `LogisticRegression`
- **XAGUSD sleeve**
  - combines:
    - `scale-invariant ensemble`
    - optional `LinearSVC calibrated`

Each sleeve produces:
- one final `BUY / SELL / FLAT`
- one final position per symbol
- one independent state machine per symbol

---

# 1. Files expected by Version B

The EA expects these ONNX files with **exact names**:

## BTC models
- `btc_linearsvc_calibrated.onnx`
- `btc_logistic_regression.onnx`

## XAG models
- `xag_mlp.onnx`
- `xag_lightgbm.onnx`
- `xag_hgb.onnx`
- `xag_linearsvc_calibrated.onnx`

These names matter because the EA uses `#resource` and MT5 requires the exact filenames.

---

# 2. Source Python scripts

Use these existing training scripts:

## BTC
- `train_mt5_linearsvc_calibrated_classifier.py`
- `train_mt5_logistic_regression_classifier.py`

## XAG
- `train_mt5_ensemble_scale_invariant.py`
- optional:
  - `train_mt5_linearsvc_calibrated_classifier.py`

So Version B is built from already existing training scripts.
No wrapper is required.

---

# 3. Python environment

Typical packages:

```bash
pip install numpy pandas scikit-learn skl2onnx onnx MetaTrader5 lightgbm onnxmltools
```

---

# 4. Generate BTCUSD LinearSVC ONNX

Example:
```bash
python train_mt5_linearsvc_calibrated_classifier.py --symbol BTCUSD --timeframe M15 --bars 80000 --horizon-bars 12 --train-ratio 0.70 --output-dir output_BTCUSD_linearsvc_M15_h12_70
```

If you want more explicit parameters:

```bash
python train_mt5_linearsvc_calibrated_classifier.py --symbol BTCUSD --timeframe M15 --bars 20000 --horizon-bars 12 --train-ratio 0.70 --c 1.0 --max-iter 3000 --calibration-cv 3 --output-dir output_BTCUSD_linearsvc_M15_h12_70
```

---

# 5. Rename BTCUSD LinearSVC ONNX

After training, rename:
`output_BTCUSD_linearsvc_M15_h12_70/ml_strategy_classifier_linearsvc_calibrated.onnx`
to:
`btc_linearsvc_calibrated.onnx`

This renamed file is the one Version B expects.

# 6. Generate BTCUSD LogisticRegression ONNX

Example:
```bash
python train_mt5_logistic_regression_classifier.py --symbol BTCUSD --timeframe M15 --bars 80000 --horizon-bars 12 --train-ratio 0.70 --output-dir output_BTCUSD_logreg_M15_h12_70
```

Expected important output:
- `ml_strategy_classifier_logistic_regression.onnx`
- `run_in_mt5.txt`

---

# 7. Rename BTCUSD LogisticRegression ONNX

After training, rename:

```text
output_BTCUSD_logreg_M15_h12_70/ml_strategy_classifier_logistic_regression.onnx
```
to
`btc_logistic_regression.onnx`

This renamed file is the one Version B expects.

---

# 8. Generate XAGUSD scale-invariant ensemble ONNX files

Example:
```bash
python train_mt5_ensemble_scale_invariant.py --symbol XAGUSD --timeframe M15 --bars 80000 --horizon-bars 8 --train-ratio 0.70 --mlp-weight 0.25 --lgbm-weight 0.25 --hgb-weight 0.50 --output-dir output_XAGUSD_ensemble_M15_h8_70
```

If you want different internal ensemble weights:
```bash
python train_mt5_ensemble_scale_invariant.py --symbol XAGUSD --timeframe M15 --bars 80000 --horizon-bars 8 --train-ratio 0.70 --mlp-weight 0.20 --lgbm-weight 0.30 --hgb-weight 0.50 --output-dir output_XAGUSD_ensemble_M15_h8_70
```
Expected important output:
- `mlp.onnx`
- `lightgbm.onnx`
- `hgb.onnx`
- `run_in_mt5.txt`

---

# 9. Rename XAGUSD ensemble ONNX files

After training, rename:
`output_XAGUSD_ensemble_M15_h8_70/mlp.onnx`
to:
`xag_mlp.onnx`

Rename:
`output_XAGUSD_ensemble_M15_h8_70/lightgbm.onnx`
to:
`xag_lightgbm.onnx`

Rename:
`output_XAGUSD_ensemble_M15_h8_70/hgb.onnx`
to:
`xag_hgb.onnx`

These renamed files are the XAG ensemble resources that Version B expects.

---

# 10. Generate optional XAGUSD LinearSVC ONNX

This model is optional in Version B.

If you want to keep the option available, train it too.

Example:
```bash
python train_mt5_linearsvc_calibrated_classifier.py --symbol XAGUSD --timeframe M15 --bars 80000 --horizon-bars 8 --train-ratio 0.70 --output-dir output_XAGUSD_linearsvc_M15_70
```

Expected important output:
- `ml_strategy_classifier_linearsvc_calibrated.onnx`
- `run_in_mt5.txt`

---

# 11. Rename optional XAGUSD LinearSVC ONNX

After training, rename:
`output_XAGUSD_linearsvc_M15_70/ml_strategy_classifier_linearsvc_calibrated.onnx`
to:
`xag_linearsvc_calibrated.onnx`

If you do not want to use XAG LinearSVC immediately:
- keep InpXagUseLinearSvc = false

But for easy MT5 compiling, it is still convenient to keep the file present.

---

# 12. Final ONNX filename checklist

Before compiling the EA, you should have these exact files:

```text
btc_linearsvc_calibrated.onnx
btc_logistic_regression.onnx
xag_mlp.onnx
xag_lightgbm.onnx
xag_hgb.onnx
xag_linearsvc_calibrated.onnx
```

If any of them is missing, the EA may fail to compile because #resource needs the files.

---

# 13. Where to put the ONNX files

Put the renamed ONNX files where MetaEditor can compile them together with the EA.

Typical workflow:
- keep the EA source file and the ONNX files in the same project folder
- compile from MetaEditor

The important point is:
- the filenames must match exactly
- the files must be available at compile time

---

# 14. Version B aggregation logic

## BTC sleeve
BTC aggregates:
- LinearSVC calibrated
- LogisticRegression

The EA computes:
- BTC model probabilities
- weighted average of those probabilities
- final BTC `BUY / SELL / FLAT`

Recommended first weights:

```text
InpBtcUseLinearSvc = true
InpBtcUseLogisticRegression = true
InpBtcLinearSvcWeight = 0.70
InpBtcLogRegWeight = 0.30
```

---

XAG sleeve

XAG aggregates:
- scale-invariant ensemble group
- optional XAG LinearSVC

Inside the ensemble group:
- `xag_mlp.onnx`
- `xag_lightgbm.onnx`
- `xag_hgb.onnx`

Recommended first settings:
```text
InpXagUseEnsembleGroup = true
InpXagUseLinearSvc = false
InpXagEnsembleGroupWeight = 0.80
InpXagLinearSvcWeight = 0.20
```

Recommended internal ensemble weights:
```text
InpXagMlpWeight = 0.25
InpXagLgbmWeight = 0.25
InpXagHgbWeight = 0.50
```

---

# 15. How to copy Python recommendations into EA inputs

Each training script writes run_in_mt5.txt.

Use those files to set the sleeve thresholds.

BTC LinearSVC

Read:
`output_XAGUSD_linearsvc_M15_70/run_in_mt5.txt`

BTC LogisticRegression

Read:
`output_BTCUSD_logreg_M15_h12_70/run_in_mt5.txt`

Use those mainly to understand:
- recommended threshold region
- max bars in trade
- whether the model is too selective or active

Because BTC is aggregated from two models, you usually do not copy both threshold sets separately.
Instead, you choose one final BTC sleeve configuration.

Recommended first BTC sleeve values:
```text
InpBtcEntryProbThreshold = 0.60
InpBtcMinProbGap = 0.20
InpBtcMaxBarsInTrade = 12
InpBtcStopAtrMultiple = 1.00
InpBtcTakeAtrMultiple = 2.00
```

XAG ensemble

Read:
`output_XAGUSD_ensemble_M15_h8_70/run_in_mt5.txt`

Recommended first XAG sleeve values:
```text
InpXagEntryProbThreshold = 0.60
InpXagMinProbGap = 0.00
InpXagMaxBarsInTrade = 8
InpXagStopAtrMultiple = 1.00
InpXagTakeAtrMultiple = 2.75
```

---

# 16. Recommended first backtest setup

For the first validation run, keep restrictive protections OFF.

BTC
```text
InpBtcUseSpreadGuard = false
InpBtcUseCooldownAfterClose = false
InpBtcUseDailyLossGuard = false
InpBtcUseHourFilter = false
```

XAG
```text
InpXagUseSpreadGuard = false
InpXagUseCooldownAfterClose = false
InpXagUseDailyLossGuard = false
InpXagUseHourFilter = false
```

Portfolio
`InpUsePortfolioDailyLossGuard = false`

Why:
first you want to confirm:
- all ONNX files load
- BTC sleeve trades
- XAG sleeve trades
- no interference exists
- aggregation behaves as expected

Only after that should you re-enable protections one by one.

---

# 17. Typical full workflow
- Train BTC LinearSVC
- Rename to `btc_linearsvc_calibrated.onnx`
- Train BTC LogisticRegression
- Rename to `btc_logistic_regression.onnx`
- Train XAG scale-invariant ensemble
- Rename to:
  - `xag_mlp.onnx`
  - `xag_lightgbm.onnx`
  - `xag_hgb.onnx`
- Optionally train XAG LinearSVC
- Rename to `xag_linearsvc_calibrated.onnx`
- Put all ONNX files next to the EA
- Compile Version B EA
- Run first backtest with loose guards
- Re-enable protections gradually

---

# 18. Notes
ONNX export compatibility

LinearSVC with calibration can be sensitive to:
- sklearn version
- skl2onnx version

If export fails, it is usually a version compatibility issue, not necessarily a training logic issue.

Why Version B is useful

Version A:
- one model stack per symbol

Version B:
- multiple model opinions per symbol
- one final symbol decision
- much better architecture if you want model diversification without model fighting

---

# 19. Final required files summary

```btc_linearsvc_calibrated.onnx
btc_logistic_regression.onnx
xag_mlp.onnx
xag_lightgbm.onnx
xag_hgb.onnx
xag_linearsvc_calibrated.onnx
MT5_Portfolio_VB_MultiModel_PerSymbol.mq5
```

---
