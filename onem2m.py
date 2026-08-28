"""Minimal oneM2M HTTP client for the TR-0071 lab.

Everything in this lab is plain HTTP + JSON. This module exists only so the six
step scripts do not repeat the same header block; read it once and the rest of
the lab reads like the slides.

oneM2M request parameters map onto HTTP like this (TS-0009):

    To          -> the URL path            e.g. POST /Mobius/co2
    From (fr)   -> X-M2M-Origin header
    Request ID  -> X-M2M-RI header
    RVI         -> X-M2M-RVI header
    Operation   -> the HTTP method, EXCEPT that CREATE is a POST whose
                   Content-Type carries ";ty=<resourceType>"
    Result code -> X-M2M-RSC response header  (NOT the HTTP status code)

Always judge a response by X-M2M-RSC.
"""

import json
import os
import time
import urllib.error
import urllib.request

# --- Lab configuration -------------------------------------------------------
# Override with environment variables if your CSE runs elsewhere.
CSE_URL = os.environ.get("CSE_URL", "http://127.0.0.1:7579")
CSE_BASE = os.environ.get("CSE_BASE", "Mobius")

# The originator used for every request in the main track.
#
# We act as the CSE administrator on purpose. The <dataset> and <datasetFragment>
# resources are created by the CSE itself under the administrator identity, and a
# resource without accessControlPolicyIDs is readable only by its creator. Using
# any other identity here would make the training dataset unreadable (RSC 4103),
# and <dataset> cannot be updated afterwards to attach a policy (RSC 4005).
# The Advanced section of the course explores exactly that gap.
ORIGIN = os.environ.get("ONEM2M_ORIGIN", "CAdmin")

# Resource type numbers.
# 1..23 are oneM2M standard. 101..107 are TR-0071 candidate types as implemented
# by mobius4 -- these numbers are NOT standardised and may change.
TY = {
    "acp": 1, "ae": 2, "cnt": 3, "cin": 4, "cb": 5, "sub": 23,
    "mrp": 101,   # <modelRepo>
    "mmd": 102,   # <mlModel>
    "mdp": 103,   # <modelDeploymentList>
    "dpm": 104,   # <modelDeployment>
    "dsp": 105,   # <mlDatasetPolicy>
    "dts": 106,   # <dataset>
    "dsf": 107,   # <datasetFragment>
}

_seq = 0


class Response:
    """One oneM2M response: the result code, the parsed body, and the raw text."""

    def __init__(self, http_status, rsc, body, raw):
        self.http_status = http_status
        self.rsc = rsc          # str, e.g. "2001" -- this is what you check
        self.body = body        # parsed JSON, or None
        self.raw = raw          # raw text, for error messages
        self.uril = []          # discovery results, filled in by discover()

    @property
    def ok(self):
        return self.rsc in ("2000", "2001", "2002", "2004")

    def __repr__(self):
        return f"<rsc={self.rsc} http={self.http_status}>"


def _request(method, to, body=None, ty=None, origin=None):
    global _seq
    _seq += 1
    url = f"{CSE_URL}/{str(to).lstrip('/')}"
    headers = {
        "X-M2M-Origin": ORIGIN if origin is None else origin,
        "X-M2M-RI": f"lab{_seq}",
        "X-M2M-RVI": "3",
        "Accept": "application/json",
    }
    data = None
    if body is not None:
        # CREATE is signalled by ";ty=N" on the Content-Type, not by the method.
        headers["Content-Type"] = (
            "application/json" if ty is None else f"application/json;ty={ty}"
        )
        data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
            rsc = resp.headers.get("X-M2M-RSC")
            status = resp.status
    except urllib.error.HTTPError as e:
        # oneM2M answers errors with a normal body and an X-M2M-RSC header, so an
        # HTTP 4xx/5xx is still a well-formed response we want to read.
        raw = e.read().decode("utf-8")
        rsc = e.headers.get("X-M2M-RSC")
        status = e.code

    try:
        parsed = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        parsed = None
    return Response(status, rsc, parsed, raw)


def create(parent, ty, body, origin=None):
    """CREATE a child resource under `parent` (a structured path such as 'Mobius/co2')."""
    return _request("POST", parent, body=body, ty=ty, origin=origin)


def retrieve(target, origin=None):
    """RETRIEVE one resource. `target` may be a structured path or a bare resourceID."""
    return _request("GET", target, origin=origin)


def update(target, body, origin=None):
    return _request("PUT", target, body=body, origin=origin)


def delete(target, origin=None):
    return _request("DELETE", target, origin=origin)


def discover(target, ty=None, origin=None):
    """DISCOVERY (fu=1): returns the list of matching resource paths."""
    q = "fu=1" + (f"&ty={ty}" if ty is not None else "")
    resp = _request("GET", f"{target}?{q}", origin=origin)
    uril = (resp.body or {}).get("m2m:uril") if resp.body else None
    if uril is None:
        resp.uril = []
    elif isinstance(uril, str):
        resp.uril = [uril]
    else:
        resp.uril = uril
    return resp


def must(resp, what, expect="2001"):
    """Stop the lab loudly when a step did not do what the slides say it does."""
    if resp.rsc != expect:
        raise SystemExit(
            f"\n[FAILED] {what}\n"
            f"  expected RSC {expect}, got {resp.rsc} (HTTP {resp.http_status})\n"
            f"  response: {resp.raw}\n"
        )
    return resp


# --- Tiny state file so each step script can run on its own ------------------
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")


def load_state():
    if not os.path.exists(STATE_PATH):
        return {}
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(**kv):
    state = load_state()
    state.update(kv)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    return state


def need(state, *keys):
    missing = [k for k in keys if k not in state]
    if missing:
        raise SystemExit(
            f"\n[FAILED] state.json is missing {missing}.\n"
            f"  Run the earlier step scripts first.\n"
        )
    return [state[k] for k in keys]


def banner(title):
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))


def show(label, resp):
    """Print a response the way the slides quote it."""
    print(f"  {label}: RSC {resp.rsc}")
    if resp.body is not None:
        text = json.dumps(resp.body, ensure_ascii=False)
        print(f"    {text[:400]}{' ...' if len(text) > 400 else ''}")


def unique(prefix):
    return f"{prefix}-{int(time.time() * 1000) % 100000000:x}"
