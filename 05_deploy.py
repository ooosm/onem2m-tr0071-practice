"""Step 5 -- deploy the model to a device.

    device <AE>
      |_ infer-in   <container>              inputResource  (inr)
      |_ infer-out  <container>              outputResource (our)
      \\_ deployments <modelDeploymentList> (mdp)
            \\_ deploy-1 <modelDeployment> (dpm)  --moid--> <mlModel>

<modelDeployment> attributes:

    moid (modelID)         the resourceID (ri) of the <mlModel>. NOT a path.
    inr  (inputResource)   where the device reads inference input from
    our  (outputResource)  where the device writes results to
    mcmd (modelCommand)    0 = stop, 1 = run. WRITE ONLY -- never returned.
    mds  (modelStatus)     0 = deployed, 1 = running, 2 = stopped. Read only.

TR-0071 writes modelCommand/modelStatus as the strings "run"/"stop"/... ;
mobius4 uses integers instead, because every oneM2M enumeration on the wire is
numeric. That difference is one of this project's revision proposals.

The last part of this step exercises the deployment compatibility check: when
the model declares an inputDescriptor AND inputResource resolves to a <dataset>,
the CSE compares the required features against the dataset's listOfFeatures and
refuses the deployment if any are missing. This check is a proposal from this
project, not part of TR-0071.
"""

import json

import onem2m as m


def make_dataset(label, sources):
    """Create a policy over `sources` and return the <dataset> it produces."""
    rn = m.unique(f"policy-{label}")
    resp = m.create(m.CSE_BASE, m.TY["dsp"], {"m2m:dsp": {
        "rn": rn, "sri": sources, "dsfm": 1, "nvp": 1, "nrhd": 1000,
    }})
    m.must(resp, f"create <mlDatasetPolicy> ({label})")
    hdi = resp.body["m2m:dsp"].get("hdi")
    if not hdi:
        raise SystemExit(f"\n[FAILED] policy '{label}' produced no historical dataset.\n")
    lof = m.must(m.retrieve(hdi), "retrieve <dataset>",
                 expect="2000").body["m2m:dts"].get("lof", [])
    print(f"  <dataset> for '{label}': {hdi}  lof={lof}")
    return hdi


def main():
    state = m.load_state()
    model_ri, sources = m.need(state, "model_ri", "sources")

    m.banner("Step 5a - register the device <AE>")
    # Self-registration: the AE has no identity yet, so From is empty and the
    # CSE assigns one. This is the one request in the lab that is not sent as
    # the administrator.
    ae_rn = m.unique("room-device")
    resp = m.create(m.CSE_BASE, m.TY["ae"], {"m2m:ae": {
        "rn": ae_rn, "api": "Nroom.occupancy", "rr": False, "srv": ["3"],
    }}, origin="")
    m.must(resp, "register device <AE>")
    aei = resp.body["m2m:ae"]["aei"]
    ae = f"{m.CSE_BASE}/{ae_rn}"
    print(f"  <AE> {ae}   aei (AE-ID assigned by the CSE) = {aei}")

    m.banner("Step 5b - input and output containers")
    infer_in, infer_out = f"{ae}/infer-in", f"{ae}/infer-out"
    for rn in ("infer-in", "infer-out"):
        m.must(m.create(ae, m.TY["cnt"], {"m2m:cnt": {"rn": rn}}),
               f"create <container> {rn}")
        print(f"  created <container> {ae}/{rn}")

    m.banner("Step 5c - <modelDeploymentList>")
    list_rn = "deployments"
    resp = m.create(ae, m.TY["mdp"], {"m2m:mdp": {"rn": list_rn}})
    m.must(resp, "create <modelDeploymentList>")
    mdp = resp.body["m2m:mdp"]
    deploy_list = f"{ae}/{list_rn}"
    print(f"  {deploy_list}")
    print(f"    ndm (numberOfDeployedModels) = {mdp['ndm']}"
          f"   nrm (numberOfRunningModels) = {mdp['nrm']}"
          f"   nsm (numberOfStoppedModels) = {mdp['nsm']}")

    m.banner("Step 5d - seed one inference input")
    m.must(m.create(infer_in, m.TY["cin"],
                    {"m2m:cin": {"con": {"co2": 642.0, "temp": 22.6}}}),
           "seed inference input")
    print('  wrote {"co2": 642.0, "temp": 22.6} to infer-in')

    m.banner("Step 5e - create the <modelDeployment>")
    dep_rn = m.unique("deploy")
    body = {"m2m:dpm": {"rn": dep_rn, "moid": model_ri,
                        "inr": infer_in, "our": infer_out}}
    print("  request body:")
    print("   ", json.dumps(body))
    resp = m.create(deploy_list, m.TY["dpm"], body)
    m.must(resp, "create <modelDeployment>")
    dpm = resp.body["m2m:dpm"]
    deployment = f"{deploy_list}/{dep_rn}"
    print(f"\n  {deployment}")
    print(f"    mds (modelStatus) = {dpm['mds']}  (0 = deployed)")
    print(f"    moid = {dpm['moid']}   inr = {dpm['inr']}   our = {dpm['our']}")
    print("\n  Note: inputResource here is a <container>, and a <container> does")
    print("  not declare its features. The compatibility check therefore does not")
    print("  run for this deployment -- it passed unchecked.")

    after = m.must(m.retrieve(deploy_list), "retrieve <modelDeploymentList>",
                   expect="2000").body["m2m:mdp"]
    print(f"\n  list counters now: ndm={after['ndm']} nrm={after['nrm']} nsm={after['nsm']}")

    # ---------------------------------------------------------------------
    m.banner("Step 5f - the compatibility check (project proposal, not TR-0071)")
    print("  Building two <dataset>s to point inputResource at.\n")
    complete = make_dataset("complete", [sources["co2"], sources["temp"]])
    incomplete = make_dataset("incomplete", [sources["temp"]])

    print("\n  (i) inputResource supplies every feature the model requires")
    ok = m.create(deploy_list, m.TY["dpm"], {"m2m:dpm": {
        "rn": m.unique("deploy-ok"), "moid": model_ri,
        "inr": complete, "our": infer_out,
    }})
    print(f"      -> RSC {ok.rsc}  (expected 2001 CREATED)")

    print("\n  (ii) inputResource is missing 'co2'")
    rejected = m.create(deploy_list, m.TY["dpm"], {"m2m:dpm": {
        "rn": m.unique("deploy-bad"), "moid": model_ri,
        "inr": incomplete, "our": infer_out,
    }})
    print(f"      -> RSC {rejected.rsc}  (expected 5207 NOT_ACCEPTABLE)")
    print(f"      response body: {rejected.raw!r}")
    print("      The body names the exact feature that is missing, so a client")
    print("      can report it without having to guess.")
    if rejected.rsc != "5207":
        raise SystemExit("\n[FAILED] the incompatible deployment was not rejected.\n")

    m.save_state(device_ae=ae, device_aei=aei, infer_in=infer_in,
                 infer_out=infer_out, deploy_list=deploy_list,
                 deployment=deployment,
                 dataset_complete=complete, dataset_incomplete=incomplete)
    m.banner("Step 5 done")
    print("  Next: python 06_device.py")


if __name__ == "__main__":
    main()
