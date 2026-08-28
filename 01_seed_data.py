"""Step 1 -- create the three sensor sources and fill them with observations.

Scenario: a meeting room has a CO2 sensor and a temperature sensor. We want to
estimate how many people are in the room without a camera. During this training
phase only, a third source supplies the ground truth head count.

    co2    <container>   ppm, rises with the number of people
    temp   <container>   degC, rises a little with the number of people
    people <container>   ground truth, TRAINING ONLY

Synthetic data (12 rounds):

    people ~ uniform{0..10}
    co2    = 420 + 55 * people + noise(sd 12)
    temp   = 21.5 + 0.25 * people + noise(sd 0.3)

Two details that matter later:

1. ORDER. Each round writes co2, then temp, then people. The dataset policy in
   step 2 uses nullValuePolicy=1 (carry the last known value forward), so the
   'people' row of a round already carries that round's co2 and temp. Step 2
   relies on this to rebuild one row per round.

2. SPACING. mobius4 creationTime has one-second resolution, so two instances
   written in the same second cannot be ordered reliably. We wait 1.1 s between
   writes.
"""

import random
import time

import onem2m as m

ROUNDS = 12
GAP_SECONDS = 1.1


def main():
    m.banner("Step 1 - sensor sources and observations")
    print(f"  CSE : {m.CSE_URL}/{m.CSE_BASE}")
    print(f"  From: {m.ORIGIN}")

    # --- <container> x 3 ------------------------------------------------------
    names = {
        "co2": m.unique("co2"),
        "temp": m.unique("temp"),
        "people": m.unique("people"),
    }
    paths = {}
    for feature, rn in names.items():
        resp = m.create(m.CSE_BASE, m.TY["cnt"], {"m2m:cnt": {"rn": rn}})
        m.must(resp, f"create <container> {rn}")
        paths[feature] = f"{m.CSE_BASE}/{rn}"
        print(f"  created <container> {paths[feature]}")

    # --- <contentInstance> ----------------------------------------------------
    print(f"\n  writing {ROUNDS} rounds, {GAP_SECONDS}s apart "
          f"(~{int(ROUNDS * 3 * GAP_SECONDS)}s total)")
    truth = []
    for i in range(ROUNDS):
        people = random.randint(0, 10)
        co2 = round(420 + 55 * people + random.gauss(0, 12), 1)
        temp = round(21.5 + 0.25 * people + random.gauss(0, 0.3), 2)

        for feature, value in (("co2", co2), ("temp", temp), ("people", people)):
            resp = m.create(paths[feature], m.TY["cin"],
                            {"m2m:cin": {"con": {feature: value}}})
            m.must(resp, f"create <contentInstance> in {feature}")
            time.sleep(GAP_SECONDS)

        truth.append({"co2": co2, "temp": temp, "people": people})
        print(f"    round {i + 1:2d}/{ROUNDS}  people={people:2d}  "
              f"co2={co2:7.1f}  temp={temp:5.2f}")

    m.save_state(sources=paths, truth=truth)
    m.banner("Step 1 done")
    print(f"  {ROUNDS * 3} <contentInstance> resources written across 3 sources.")
    print("  Next: python 02_make_dataset.py")


if __name__ == "__main__":
    main()
