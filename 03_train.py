"""Step 3 -- train a model outside the CSE.

The CSE stores, describes and deploys models. It never trains and never infers.
That is a deliberate premise of TR-0071, so this step is ordinary scikit-learn
with no oneM2M in it at all.

Input : training-data.csv  (produced by step 2 from <datasetFragment> rows)
Output: model.joblib, model.b64  (base64 is how the mlModel attribute travels)

Look at the coefficients this prints. CO2 dominates; temperature is a weak
helper. That is honest -- for accuracy alone CO2 would nearly do on its own.
We keep two features because the oneM2M interfaces we are learning need them:
merging several sources is what <mlDatasetPolicy> is for, an inputDescriptor
has no order to get wrong with one feature, and a deployment compatibility
check has no feature to be missing.
"""

import base64
import csv
import os

import joblib
import sklearn
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "training-data.csv")
MODEL_PATH = os.path.join(HERE, "model.joblib")
B64_PATH = os.path.join(HERE, "model.b64")

# The order of these two names is the model's input contract. Step 4 publishes
# it as inputDescriptor so the device does not have to guess -- and step 6 shows
# what a wrong order produces (a plausible, silently wrong answer).
FEATURES = ["co2", "temp"]
TARGET = "people"


def main():
    print(f"\n=== Step 3 - train (scikit-learn {sklearn.__version__}) " + "=" * 20)

    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"\n[FAILED] {CSV_PATH} not found. Run 02_make_dataset.py first.\n")

    X, y = [], []
    with open(CSV_PATH) as fh:
        for row in csv.DictReader(fh):
            X.append([float(row[f]) for f in FEATURES])
            y.append(float(row[TARGET]))
    print(f"  training rows: {len(X)}  features: {FEATURES}  target: {TARGET}")
    if len(X) < 3:
        raise SystemExit("\n[FAILED] not enough training rows. Re-run steps 1 and 2.\n")

    model = LinearRegression().fit(X, y)

    print("\n  learned model:")
    for name, coef in zip(FEATURES, model.coef_):
        print(f"    {name:6s} coefficient = {coef:+.6f}")
    print(f"    intercept          = {model.intercept_:+.6f}")
    print(f"    R^2 on training data = {model.score(X, y):.4f}")

    # Raw coefficients cannot be compared directly: co2 spans hundreds of ppm
    # while temp spans a couple of degrees, so the smaller-looking co2
    # coefficient is applied to a much larger number. Multiply each coefficient
    # by its feature's standard deviation to see the effect each sensor
    # actually has on the prediction, in people.
    print("\n  effect of each sensor (|coefficient| x standard deviation, in people):")
    for i, name in enumerate(FEATURES):
        col = [row[i] for row in X]
        mean = sum(col) / len(col)
        sd = (sum((v - mean) ** 2 for v in col) / len(col)) ** 0.5
        print(f"    {name:6s} sd = {sd:8.3f}   effect = {abs(model.coef_[i]) * sd:6.3f}")
    print("\n  CO2 carries almost all of the signal; temperature adds very little.")
    print("  Comparing the raw coefficients would tell you the opposite, because")
    print("  the two features are on completely different scales.")

    joblib.dump(model, MODEL_PATH)
    size = os.path.getsize(MODEL_PATH)
    raw = open(MODEL_PATH, "rb").read()
    b64 = base64.b64encode(raw).decode("ascii")
    with open(B64_PATH, "w") as fh:
        fh.write(b64)

    print(f"\n  {MODEL_PATH}  {size} bytes")
    print(f"  {B64_PATH}     {len(b64)} base64 characters")
    print(f"  base64 preview: {b64[:60]}...")
    print("\n  This fits inline in the mlModel attribute. Models larger than about")
    print("  740 KB do not -- those must be registered by URL (mlModelURL) instead.")
    print("\n  Next: python 04_register_model.py")


if __name__ == "__main__":
    main()
