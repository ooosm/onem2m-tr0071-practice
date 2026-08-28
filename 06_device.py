"""Step 6 -- the device side: fetch the model, run it, store the result.

The CSE never infers. Everything here is what a real device would do:

    1. read its <modelDeployment>
    2. RETRIEVE the model by moid -- which is a resourceID, so the URL is
       GET /<ri>, not GET /Mobius/<ri>
    3. pick a runtime from the model's platform string. If it does not know that
       platform, it STOPS. It does not guess.
    4. build the input in the order the inputDescriptor gives
    5. infer
    6. write the result to outputResource as a <contentInstance>
    7. tell the CSE it is running: modelCommand = 1

Run with --wrong-order to see what a mis-declared inputDescriptor costs: the
device swaps the two features, nothing raises, the CSE stores the answer with
RSC 2001, and the number is simply wrong.
"""

import base64
import io
import json
import sys

import joblib
import sklearn

import onem2m as m

# The only runtimes this device knows how to execute. A platform string outside
# this table is a hard stop -- see run_model().
RUNTIMES = {f"scikit-learn/{sklearn.__version__}": "sklearn-joblib"}


def load_runtime(platform):
    runtime = RUNTIMES.get(platform)
    if runtime is None:
        raise SystemExit(
            f"\n[STOP] This device does not know how to run platform \"{platform}\".\n"
            f"  Known runtimes: {sorted(RUNTIMES)}\n"
            f"  TR-0071 says nothing about how a model is executed. The platform\n"
            f"  attribute is free text and is the only hint the device gets, so a\n"
            f"  device can only refuse. Failing loudly here is the correct behaviour.\n"
        )
    return runtime


def main():
    wrong_order = "--wrong-order" in sys.argv
    state = m.load_state()
    deployment, = m.need(state, "deployment")

    m.banner("Step 6a - read the <modelDeployment>")
    dpm = m.must(m.retrieve(deployment), "retrieve <modelDeployment>",
                 expect="2000").body["m2m:dpm"]
    print(f"  mds (modelStatus) = {dpm['mds']}  (0 = deployed)")
    print(f"  moid = {dpm['moid']}")
    print("  (mcmd is never in a response -- modelCommand is write-only)")

    m.banner("Step 6b - RETRIEVE the model by moid")
    print(f"  GET /{dpm['moid']}      <- unstructured: moid is a resourceID, not a path")
    mmd = m.must(m.retrieve(dpm["moid"]), "retrieve <mlModel> by moid",
                 expect="2000").body["m2m:mmd"]
    print(f"  plf (platform)    = {mmd['plf']}")
    print(f"  mms (mlModelSize) = {mmd.get('mms')} bytes")
    print(f"  ipd (inputDescriptor) = {json.dumps(mmd.get('ipd'))}")

    m.banner("Step 6c - choose a runtime from the platform string")
    runtime = load_runtime(mmd["plf"])
    print(f"  runtime: {runtime}")

    m.banner("Step 6d - read the inference input")
    cin = m.must(m.retrieve(f"{dpm['inr']}/la"), "retrieve inputResource <latest>",
                 expect="2000").body["m2m:cin"]
    observation = cin["con"]
    print(f"  <latest> of inputResource: {json.dumps(observation)}")

    descriptor = mmd.get("ipd") or []
    names = [f["name"] for f in descriptor]
    if wrong_order:
        names = list(reversed(names))
        print(f"\n  !! --wrong-order: using {names} instead of "
              f"{[f['name'] for f in descriptor]}")
    missing = [n for n in names if n not in observation]
    if missing:
        raise SystemExit(f"\n[FAILED] input is missing {missing}\n")
    features = [float(observation[n]) for n in names]
    print(f"  input vector, in inputDescriptor order: {names} = {features}")

    m.banner("Step 6e - infer")
    model = joblib.load(io.BytesIO(base64.b64decode(mmd["mmd"])))
    prediction = float(model.predict([features])[0])
    out_name = (mmd.get("oud") or [{"name": "value"}])[0]["name"]
    print(f"  prediction: {out_name} = {prediction:.4f}")
    if wrong_order:
        print("  Nothing raised. The CSE will accept this. It is simply wrong --")
        print("  which is why the input contract has to be declared, not guessed.")

    m.banner("Step 6f - store the result in outputResource")
    result = {out_name: round(prediction, 4), **observation}
    resp = m.create(dpm["our"], m.TY["cin"], {"m2m:cin": {"con": result}})
    m.must(resp, "store inference result")
    print(f"  stored: {json.dumps(result)}")
    print(f"  as {dpm['our']}/{resp.body['m2m:cin']['rn']}   RSC {resp.rsc}")

    m.banner("Step 6g - report that the model is running")
    upd = m.update(deployment, {"m2m:dpm": {"mcmd": 1}})
    m.must(upd, "set modelCommand = 1", expect="2004")
    print(f"  mds (modelStatus) is now {upd.body['m2m:dpm']['mds']}  (1 = running)")
    counters = m.must(m.retrieve(state["deploy_list"]), "retrieve list",
                      expect="2000").body["m2m:mdp"]
    print(f"  list counters: ndm={counters['ndm']} nrm={counters['nrm']} "
          f"nsm={counters['nsm']}")

    m.banner("Step 6 done - the loop is closed")
    print("  sensors -> dataset -> model -> deployment -> inference -> oneM2M resource")


if __name__ == "__main__":
    main()
