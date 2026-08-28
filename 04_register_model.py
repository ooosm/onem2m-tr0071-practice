"""Step 4 -- register the trained model in the CSE.

This is the Model Management interface of TR-0071.

    <modelRepo> (mrp)  holds models and counts them
      \\_ <mlModel> (mmd)  one model

Mandatory on an <mlModel> CREATE:

    vr  (version)    model version, a free string
    plf (platform)   the ML platform that produced the model, a free string.
                     This is the ONLY hint the device gets about how to run the
                     model, so we put the framework AND its version in it.
    mlt (mlType)     kind of model, e.g. "regression"

And exactly one of:

    mmd (mlModel)     the model bytes, base64 encoded, inline
    mmu (mlModelURL)  a URL to download the model from

Sending both is rejected, and so is sending neither.

We also send five attributes that are NOT part of TR-0071. They are a proposal
from this project, implemented in mobius4 so it could be tested:

    tdi (trainingDatasetID)  which <dataset> this model was trained on
    ipd (inputDescriptor)    the ordered input contract
    oud (outputDescriptor)   the output contract
    ppr (preprocessingRef)   URI, not interpreted by the CSE (not used here)
    msr (modelSignatureRef)  URI, not interpreted by the CSE (not used here)

inputDescriptor is what lets step 6's device build its input without hard-coding
feature names, and what makes the deployment compatibility check in step 5
possible.
"""

import json
import os

import sklearn

import onem2m as m

HERE = os.path.dirname(os.path.abspath(__file__))
B64_PATH = os.path.join(HERE, "model.b64")

INPUT_DESCRIPTOR = [
    {"name": "co2", "dataType": "xs:float", "unit": "ppm", "optional": False},
    {"name": "temp", "dataType": "xs:float", "unit": "Cel", "optional": False},
]
OUTPUT_DESCRIPTOR = [
    {"name": "people", "dataType": "xs:float", "optional": False},
]


def main():
    state = m.load_state()
    (hdi,) = m.need(state, "hdi")

    if not os.path.exists(B64_PATH):
        raise SystemExit(f"\n[FAILED] {B64_PATH} not found. Run 03_train.py first.\n")
    b64 = open(B64_PATH).read().strip()

    m.banner("Step 4a - create the <modelRepo>")
    repo_rn = m.unique("model-repo")
    resp = m.create(m.CSE_BASE, m.TY["mrp"], {"m2m:mrp": {"rn": repo_rn}})
    m.must(resp, "create <modelRepo>")
    m.show("<modelRepo> created", resp)
    repo = f"{m.CSE_BASE}/{repo_rn}"
    mrp = resp.body["m2m:mrp"]
    print(f"\n  cnmo (currentNumberOfModels) = {mrp['cnmo']}"
          f"   cbmo (currentByteOfModels) = {mrp['cbmo']}")

    m.banner("Step 4b - register the <mlModel>")
    model_rn = m.unique("occupancy")
    body = {
        "m2m:mmd": {
            "rn": model_rn,
            "vr": "1.0.0",
            "plf": f"scikit-learn/{sklearn.__version__}",
            "mlt": "regression",
            "nm": "room-occupancy-estimator",
            "dc": "Estimates the number of people in a room from CO2 and temperature.",
            "ips": json.dumps({"co2": 640.0, "temp": 22.5}),
            "ous": json.dumps({"people": 4.0}),
            "mmd": b64,
            # -- project proposal, not TR-0071 --------------------------------
            "tdi": hdi,
            "ipd": INPUT_DESCRIPTOR,
            "oud": OUTPUT_DESCRIPTOR,
        }
    }
    preview = dict(body["m2m:mmd"])
    preview["mmd"] = b64[:40] + f"... ({len(b64)} chars)"
    print("  request body (mlModel truncated for display):")
    print("   ", json.dumps({"m2m:mmd": preview}))

    resp = m.create(repo, m.TY["mmd"], body)
    m.must(resp, "create <mlModel>")
    mmd = resp.body["m2m:mmd"]
    print(f"\n  <mlModel> created: {repo}/{model_rn}")
    print(f"    ri  (resourceID)  = {mmd['ri']}   <- this is what a deployment points at")
    print(f"    mms (mlModelSize) = {mmd.get('mms')} bytes (decoded, not base64 length)")
    print(f"    plf (platform)    = {mmd['plf']}")
    print(f"    ipd (inputDescriptor) came back as: {json.dumps(mmd.get('ipd'))}")

    after = m.must(m.retrieve(repo), "retrieve <modelRepo>", expect="2000").body["m2m:mrp"]
    print(f"\n  repository after registration: cnmo={after['cnmo']} cbmo={after['cbmo']}")

    m.banner("Step 4c - what the CSE refuses")
    bad = m.create(repo, m.TY["mmd"], {"m2m:mmd": {
        "rn": m.unique("bad"), "vr": "1.0.0",
        "plf": "scikit-learn", "mlt": "regression",
        "mmd": b64, "mmu": "https://example.invalid/model.joblib",
    }})
    print(f"  sending BOTH mmd and mmu -> RSC {bad.rsc}")
    print(f"    {bad.raw[:160]}")
    bad2 = m.create(repo, m.TY["mmd"], {"m2m:mmd": {
        "rn": m.unique("bad"), "vr": "1.0.0",
        "plf": "scikit-learn", "mlt": "regression",
    }})
    print(f"  sending NEITHER               -> RSC {bad2.rsc}")
    print(f"    {bad2.raw[:160]}")

    m.save_state(repo=repo, model_path=f"{repo}/{model_rn}", model_ri=mmd["ri"],
                 model_size=mmd.get("mms"), platform=mmd["plf"])
    m.banner("Step 4 done")
    print("  Next: python 05_deploy.py")


if __name__ == "__main__":
    main()
