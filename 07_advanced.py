"""Step 7 (advanced) -- two gaps in TR-0071 that only show up when you build on it.

    A. A model whose platform the device does not know.
       TR-0071 says nothing about how a model is executed. The platform
       attribute is free text, so a device can only look the string up in a
       table it maintains itself. Deployment still succeeds -- the failure
       happens later, on the device, in the field.

    B. A device reading its own model with its own identity.
       The deployment tells the device WHICH model to run, but grants it no
       right to READ that model. The <mlModel> was created by someone else, and
       without an accessControlPolicy the default policy gives access only to
       the creator. The device gets 4103.

Neither is a mobius4 bug. Both are places where the specification is silent.
"""


import onem2m as m


def part_a(state):
    m.banner("A. a platform the device does not know")
    (repo,) = m.need(state, "repo")
    b64 = open(__file__.replace("07_advanced.py", "model.b64")).read().strip()

    # TR-0071 clause 7.1.2.2 uses "tensorFlow" as its own example value for
    # platform, so this is not an invented string -- it is the one the
    # specification suggests.
    rn = m.unique("tf-model")
    resp = m.create(repo, m.TY["mmd"], {"m2m:mmd": {
        "rn": rn, "vr": "1.0.0", "plf": "tensorFlow", "mlt": "regression",
        "dc": "The platform value TR-0071 itself gives as an example.",
        "mmd": b64,
    }})
    m.must(resp, "register a tensorFlow model")
    other_ri = resp.body["m2m:mmd"]["ri"]
    print(f"  registered <mlModel> with plf=tensorFlow   ri={other_ri}   RSC {resp.rsc}")

    dep = m.create(state["deploy_list"], m.TY["dpm"], {"m2m:dpm": {
        "rn": m.unique("deploy-tf"), "moid": other_ri,
        "inr": state["infer_in"], "our": state["infer_out"],
    }})
    m.must(dep, "deploy the tensorFlow model")
    print(f"  deployed it                                        RSC {dep.rsc}")
    print("\n  The CSE accepted both. Nothing in oneM2M knows that the device")
    print("  cannot run tensorFlow -- the device finds out when it looks the")
    print("  platform string up and finds nothing:")
    print("\n    [STOP] This device does not know how to run platform \"tensorFlow\".")
    print("\n  A deployment that succeeds is not a deployment that will run.")
    return other_ri


def part_b(state):
    m.banner("B. the device cannot read its own model")
    aei, model_ri = m.need(state, "device_aei", "model_ri")

    print(f"  Same request, two identities. Model: {model_ri}\n")
    as_admin = m.retrieve(model_ri)
    print(f"  From: {m.ORIGIN} (administrator)  -> RSC {as_admin.rsc}")

    as_device = m.retrieve(model_ri, origin=aei)
    print(f"  From: {aei} (the device itself)         -> RSC {as_device.rsc}")
    print(f"     body: {as_device.raw!r}")

    if as_device.rsc == "4103":
        print("\n  4103 ORIGINATOR_HAS_NO_PRIVILEGE. This is TS-0001 working as")
        print("  specified: an <mlModel> with no accessControlPolicyIDs is")
        print("  readable only by whoever created it. The <modelDeployment> that")
        print("  points at the model grants the device nothing.")
        print("\n  The fix is an <accessControlPolicy> the model owner attaches")
        print("  explicitly -- and TR-0071 never says who does that, or when.")
    else:
        print(f"\n  Expected 4103 here, got {as_device.rsc}.")
        print("  (If the lab was run entirely as one identity, the device IS the")
        print("   creator and this gap stays hidden -- which is exactly how it")
        print("   goes unnoticed in real projects.)")

    m.banner("C. granting the device access, the oneM2M way")
    acp_rn = m.unique("model-readers")
    acp = m.create(m.CSE_BASE, m.TY["acp"], {"m2m:acp": {
        "rn": acp_rn,
        # acop is a bit sum: 1 create, 2 retrieve, 4 update, 8 delete,
        # 16 discovery, 32 notify. 63 is all six.
        "pv": {"acr": [{"acor": [aei], "acop": 2},
                       {"acor": [m.ORIGIN], "acop": 63}]},
        "pvs": {"acr": [{"acor": [m.ORIGIN], "acop": 63}]},
    }})
    m.must(acp, "create <accessControlPolicy>")
    print(f"  created <accessControlPolicy> {m.CSE_BASE}/{acp_rn}")
    print(f"    pv: retrieve for {aei}, everything for {m.ORIGIN}")

    upd = m.update(state["model_path"], {"m2m:mmd": {"acpi": [f"{m.CSE_BASE}/{acp_rn}"]}})
    print(f"  attaching it to the model via acpi -> RSC {upd.rsc}")

    again = m.retrieve(model_ri, origin=aei)
    print(f"  device reads the model again        -> RSC {again.rsc}")
    if again.rsc == "2000":
        print("\n  Now it works. Note what had to happen: a human decided to create")
        print("  a policy and attach it. Nothing in the deployment flow asked for it.")


def main():
    state = m.load_state()
    m.need(state, "deploy_list", "model_path")
    part_a(state)
    part_b(state)
    m.banner("Step 7 done")


if __name__ == "__main__":
    main()
