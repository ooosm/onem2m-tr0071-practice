# oneM2M TR-0071 AI/ML Practice Lab

Seven short Python scripts that walk one small AI service through oneM2M
resources, end to end: sensor data in, training dataset out, model trained,
model registered, model deployed, inference stored back.

**What you are learning is the interface, not the machine learning.** The model
is a two-variable linear regression that takes five lines of scikit-learn. Every
interesting decision in this lab is a oneM2M one.

Companion lab for the course *oneM2M TR-0071 AI/ML Practice*.

---

## The scenario

A meeting room has a CO2 sensor and a temperature sensor. We want to know how
many people are in the room **without a camera** — cameras are expensive and
raise privacy problems, CO2 sensors are cheap. During training only, a third
source supplies the ground-truth head count.

```
co2    <container>   ppm, rises with the number of people
temp   <container>   degC, rises a little with the number of people
people <container>   ground truth, TRAINING ONLY
```

After deployment the label source is gone: the device sees only CO2 and
temperature, and produces an estimate.

---

## Prerequisites

- A running **mobius4** CSE with PostgreSQL and an MQTT broker.
  **v4.15.0 or later** — the AI/ML resource types arrived in v4.15.0.
  This lab was last verified against **v4.17.1**.
- **Python 3.10+**

```bash
git clone https://github.com/ooosm/onem2m-tr0071-practice
cd onem2m-tr0071-practice
pip install -r requirements.txt
```

Point the scripts at your CSE:

```bash
export CSE_URL=http://127.0.0.1:7579      # your CSE
export CSE_BASE=Mobius                    # CSEBase resourceName
export ONEM2M_ORIGIN=CAdmin               # the CSE administrator identity
```

### Why the administrator identity?

`<dataset>` and `<datasetFragment>` are created by the CSE itself, under the
administrator identity. A resource with no `accessControlPolicyIDs` is readable
only by whoever created it, and `<dataset>` cannot be updated afterwards to
attach a policy. So any other identity simply cannot read the training data.

Step 7 shows you that wall on purpose, and how to take it down properly.

---

## Running

Run them in order. Each one writes what it learned to `state.json`, so you can
stop and resume.

```bash
python 01_seed_data.py       # 3 <container>s + 12 rounds of observations (~40 s)
python 02_make_dataset.py    # <mlDatasetPolicy> -> <dataset> -> <datasetFragment>
python 03_train.py           # scikit-learn, outside the CSE
python 04_register_model.py  # <modelRepo> + <mlModel>
python 05_deploy.py          # device <AE>, <modelDeploymentList>, <modelDeployment>
python 06_device.py          # retrieve model, infer, store result
python 07_advanced.py        # two gaps in the specification
```

Two extra things worth running:

```bash
python 06_device.py --wrong-order
```

Swaps the two input features. Nothing raises, the CSE stores the answer with
RSC 2001, and the number is wrong — 82 people in a room whose CO2 says 4. This
is why the input contract has to be declared rather than guessed.

---

## What each step teaches

| Step | oneM2M lesson |
|---|---|
| 01 | Ordinary `<container>` / `<contentInstance>`. Why writes are 1.1 s apart: `creationTime` has one-second resolution. |
| 02 | You do not create a dataset — a policy does. `nrhd` is what asks for one. The CSE returns **one row per source instance**, not one per round; rebuilding rows is the client's job. |
| 03 | The CSE never trains and never infers. |
| 04 | `vr`/`plf`/`mlt` are mandatory; exactly one of `mmd`/`mmu`. `mlModelSize` counts decoded bytes. |
| 05 | `moid` is a resourceID, not a path. `mcmd` is write-only. The compatibility check only runs when `inr` resolves to a `<dataset>`. |
| 06 | The device picks a runtime from a free-text `platform` string, and builds its input in `inputDescriptor` order. |
| 07 | A deployment does not grant the right to read the model (4103), and a platform the device does not know fails only in the field. |

---

## Files produced while you run

These are ignored by git — they are yours, not part of the repo.

| File | What |
|---|---|
| `state.json` | resource IDs carried between steps |
| `training-data.csv` | the rebuilt training rows |
| `model.joblib` / `model.b64` | the trained model, and its base64 form |

---

## A note on standards status

The seven AI/ML resource types are **TR-0071 candidate solutions, not a
standard**. TR-0071 is a Technical Report; no oneM2M Technical Specification
contains these resource types.

- Resource type numbers **101–107** and every short name here are chosen by the
  mobius4 implementation. They will change if these types are standardised.
- The attributes `tdi`, `ipd`, `oud`, `ppr`, `msr` and the deployment
  compatibility check are **a proposal from this project** and are not in
  TR-0071 at all.

Learn the interface from this lab; do not quote it as a standard.

---

## Licence

BSD 3-Clause. See [LICENSE](LICENSE).
