"""Step 2 -- ask the CSE to build a training dataset, then read it back.

This is the Dataset Management interface of TR-0071. You do NOT create the
dataset yourself: you create one <mlDatasetPolicy> (dsp) that says which sources
to merge and how, and the CSE creates <dataset> (dts) and <datasetFragment>
(dsf) resources on its own.

Key attributes of the policy we send:

    sri  (sourceResourceIDs)                    the three <container>s to merge
    dsfm (datasetFormat)                        1 = JSON
    nvp  (nullValuePolicy)                      1 = carry the last known value forward
    nrhd (numberOfRowsForHistoricalDataset)     rows per fragment. Sending this is
                                                what asks for a historical dataset;
                                                without it the CSE creates none.

The response carries hdi (historicalDatasetID) -- the <dataset> the CSE made.

Then the part that surprises everyone: the CSE emits ONE ROW PER SOURCE
INSTANCE, not one row per round. 12 rounds x 3 sources = 36 rows, each round
filling in over three rows because of nullValuePolicy=1. Rebuilding one row per
round is the client's job -- TR-0071 does not define the unit of a row.
"""

import csv
import json
import os

import onem2m as m

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training-data.csv")


def main():
    state = m.load_state()
    (sources,) = m.need(state, "sources")

    m.banner("Step 2a - create the <mlDatasetPolicy>")
    rn = m.unique("policy")
    body = {
        "m2m:dsp": {
            "rn": rn,
            "sri": [sources["co2"], sources["temp"], sources["people"]],
            "dsfm": 1,      # JSON
            "nvp": 1,       # carry the last known value forward
            "nrhd": 1000,   # ask for a historical dataset; rows per fragment
        }
    }
    print("  request body:")
    print("   ", json.dumps(body))
    resp = m.create(m.CSE_BASE, m.TY["dsp"], body)
    m.must(resp, "create <mlDatasetPolicy>")
    m.show("<mlDatasetPolicy> created", resp)

    dsp = resp.body["m2m:dsp"]
    hdi = dsp.get("hdi")
    if not hdi:
        raise SystemExit(
            "\n[FAILED] the policy was created but hdi (historicalDatasetID) is empty.\n"
            "  The CSE creates the historical <dataset> only when nrhd is present\n"
            "  AND the sources already hold <contentInstance> resources.\n"
        )
    print(f"\n  hdi (historicalDatasetID) = {hdi}")

    m.banner("Step 2b - read the <dataset> the CSE created")
    dts = m.must(m.retrieve(hdi), "retrieve <dataset>", expect="2000")
    m.show("<dataset>", dts)
    lof = dts.body["m2m:dts"].get("lof", [])
    print(f"\n  lof (listOfFeatures) = {lof}")

    m.banner("Step 2c - discover and read the <datasetFragment> resources")
    disc = m.must(m.discover(hdi, ty=m.TY["dsf"]), "discover <datasetFragment>", expect="2000")
    print(f"  found {len(disc.uril)} fragment(s): {disc.uril}")
    if not disc.uril:
        raise SystemExit("\n[FAILED] no <datasetFragment> was created.\n")

    # Discovery does not define an order, so sort by datasetFragmentStartTime.
    fragments = []
    for path in disc.uril:
        f = m.must(m.retrieve(path), f"retrieve {path}", expect="2000").body["m2m:dsf"]
        fragments.append(f)
    fragments.sort(key=lambda f: f.get("dfst", ""))

    rows = []
    for f in fragments:
        print(f"  {f['rn']}: nrf={f.get('nrf')} "
              f"dfst={f.get('dfst')} dfet={f.get('dfet')}")
        rows.extend(f.get("dsfr", []))
    print(f"\n  total rows returned by the CSE: {len(rows)}")
    print("  first three rows:")
    for r in rows[:3]:
        print("   ", json.dumps(r, ensure_ascii=False))

    m.banner("Step 2d - rebuild one row per round")
    print("  Rule: a new round begins when co2 changes (step 1 writes co2 first),")
    print("  and the round is complete at the LAST row before the next change --")
    print("  nullValuePolicy=1 has filled temp and people in by then.")

    training = []
    current = []        # rows belonging to the round being accumulated
    current_co2 = object()

    def flush(group):
        if not group:
            return
        last = group[-1]
        co2, temp, people = last.get("co2", ""), last.get("temp", ""), last.get("people", "")
        if co2 == "" or temp == "" or people == "":
            return      # an incomplete round (e.g. the very first one) is dropped
        training.append({"co2": float(co2), "temp": float(temp), "people": float(people)})

    for r in rows:
        if r.get("co2", "") != current_co2:
            flush(current)
            current, current_co2 = [], r.get("co2", "")
        current.append(r)
    flush(current)

    print(f"\n  {len(rows)} CSE rows  ->  {len(training)} training rows")
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["co2", "temp", "people"])
        w.writeheader()
        w.writerows(training)
    print(f"  written: {OUT_CSV}")

    m.save_state(policy=f"{m.CSE_BASE}/{rn}", hdi=hdi, lof=lof,
                 cse_rows=len(rows), training_rows=len(training))
    m.banner("Step 2 done")
    print("  Next: python 03_train.py")


if __name__ == "__main__":
    main()
