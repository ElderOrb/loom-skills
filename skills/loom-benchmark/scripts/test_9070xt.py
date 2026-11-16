#!/usr/bin/env python3
"""Does a package that serves on the Radeon AI PRO R9700 also serve on a 16 GB AMD Radeon RX 9070 XT, and at what context?

The RX 9070 XT is the R9700's chip (Navi 48, gfx1201) with half the memory: the same programs run, only the context that fits
differs. Run on a machine with a 9070 XT (not a DevLab job: there is no runner). The package is served elastic with a 16 GB
budget, answers a sanity prompt, then the largest context (multiples of 1024, up to the native context) the server admits is
found with ctx_admit.py and confirmed by one request of that length that must complete on a FRESH server (no prefix the probes
left cached). The result is make_info.py build --rx9070xt's input: info.json devices.radeon-r9700.rx9070xt, rendered as one line
in the card's R9700 section ("This model was also tested on Radeon 9070XT at context size N.").

usage: test_9070xt.py PKG [--loom-root ~/loom] [--memory-bytes 15000000000] [--prefill-capacity N] [--port 8080] -o rx9070xt.json
       test_9070xt.py --untested REASON -o rx9070xt.json        (a package whose weights do not fit 16 GB: {tested: false})
PKG: a fetched package (loom-fetch MODEL.json --materialize=share --output=PKG, the default preset; not the -r9700 preset, whose
fixed pool is sized for 32 GB). Run with the tools venv's python ($LOOM_ROOT/.tools/bin/python: requests, tokenizers).

--prefill-capacity N: serve through a deployment document that sets the text definition's prefill_capacity (a serving setting, the package
is unchanged): for a package whose weights plus the default prefill chunk's workspace exceed the budget (registration fails with "model
parameters and invocation working set exceed the serving memory limit"; gpt-oss-20b: 4096 -> 1024). Recorded as "execution".

Serve flags (recorded in the result):
- --memory_bytes=15000000000: the board reports 17.1 GB; a desktop session holds about 1.5 GB of it.
- --state_slab_bytes=4mib: elastic state commits in slabs per KV plane; the default 64 MiB x 82 planes (Qwen3-8B with its drafter)
  is 5.5 GB committed before the first request, and the context that fits drops from 32k to 7k.
- --parameter_mode=preload: a stock Ubuntu desktop has an 8 MB RLIMIT_MEMLOCK; the default file mode pins io_uring buffers and
  the weights never load ("buffer pin preparation failed (12)"). Not needed where `ulimit -l` is unlimited.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = "What is the capital of France? Answer in one word."
DEVICE = "--device=amdgpu://?completion_spin_ns=4294967295"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def hrx_loom_version(root):
    try:
        j = json.load(open(os.path.join(root, "loom-install.json")))
        return j.get("hrx_loom") or j.get("version")
    except (OSError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkg", nargs="?")
    ap.add_argument("--loom-root", default=os.environ.get("LOOM_ROOT", os.path.expanduser("~/loom")))
    ap.add_argument("--memory-bytes", type=int, default=15_000_000_000)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--prefill-capacity", type=int, help="serve through a deployment document with this prefill_capacity")
    ap.add_argument("--weight-slab-bytes", help="loom-serve --weight_slab_bytes (default 1 GiB): smaller slabs for a package whose weights leave "
                    "less than a 1 GiB workspace slab of the budget (LFM2.5-24B: 64mib with --prefill-capacity 256)")
    ap.add_argument("--untested", metavar="REASON", help="write {tested: false, reason} without serving")
    ap.add_argument("-o", required=True)
    a = ap.parse_args(argv)
    date = time.strftime("%Y-%m-%d")
    if a.untested:
        json.dump({"tested": False, "max_context": None, "reason": a.untested, "date": date}, open(a.o, "w"), indent=1)
        return 0
    if not a.pkg:
        ap.error("PKG is required unless --untested")
    pkg, url = os.path.abspath(a.pkg), f"http://127.0.0.1:{a.port}"
    card = json.load(open(os.path.join(pkg, "model.json")))
    cfg = json.load(open(os.path.join(pkg, "llm_config.json")))
    native = next(iter(card["presets"].values()))["execution"]["context_capacity"]
    flags = [f"--memory_bytes={a.memory_bytes}", "--parameter_mode=preload", "--state_slab_bytes=4mib"]
    if a.weight_slab_bytes:
        flags.append(f"--weight_slab_bytes={a.weight_slab_bytes}")
    model = f"{pkg}/model.json"
    execution = {"prefill_capacity": a.prefill_capacity} if a.prefill_capacity else None
    if execution:   # a deployment document: the card's preset with these execution settings overridden
        model = os.path.abspath(a.o + ".deployment.json")
        json.dump({"name": cfg["model_id"], "model": {"path": f"{pkg}/model.json"}, "definition": {"kind": "text", **execution}},
                  open(model, "w"), indent=1)
    serve = [os.path.join(a.loom_root, ".venv", "bin", "loom-serve"), f"--model={model}", DEVICE, *flags,
             f"--listen=127.0.0.1:{a.port}", "--heartbeat_ms=0"]
    res = {"tested": False, "max_context": None, "date": date, "memory_bytes": a.memory_bytes, "serve_args": flags,
           "hrx_loom": hrx_loom_version(a.loom_root), "native_context": native, "probes": [], "confirm": []}
    if execution:
        res["execution"] = execution
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if (cfg.get("chat") or {}).get("thinking") == "toggle" else {}
    slog = open(a.o + ".serve.log", "w")
    srv = None

    def start():
        nonlocal srv
        if srv:
            srv.terminate()
            srv.wait(60)
            time.sleep(2)
        srv = subprocess.Popen(serve, stdout=slog, stderr=subprocess.STDOUT)
        for _ in range(600):   # registration: compiled (ready) or refused (failed, with the server's reason)
            try:
                with urllib.request.urlopen(url + "/v1/status", timeout=10) as r:
                    m = (json.load(r).get("data") or [{}])[0].get("loom") or {}
            except (OSError, ValueError):
                m = {}
            if m.get("registration") == "failed":
                res["load_error"] = ((m.get("error") or {}).get("message") or "registration failed")[:300]
                log("registration failed:", res["load_error"])
                return False
            if m.get("registration") == "ready":
                break
            time.sleep(2)
        w = subprocess.run([sys.executable, os.path.join(HERE, "wait_model.py"), url, "--wait", "900"], capture_output=True, text=True)
        log("load", (w.stdout + w.stderr).strip()[-200:])
        return w.returncode == 0

    def sane():
        body = {"model": cfg["model_id"], "max_tokens": 1024, "temperature": 0, "messages": [{"role": "user", "content": PROMPT}], **extra}
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return "paris" in json.dumps(json.load(r)["choices"][0]["message"]).lower()

    def admit(k, complete=False):   # a context of k * 1024 positions: the prompt leaves 128 for the template and the answer
        cmd = [sys.executable, os.path.join(HERE, "ctx_admit.py"), pkg, url, "--tokens", str(k * 1024 - 128), "--wait", "300"]
        p = subprocess.run(cmd + (["--complete", "2400"] if complete else []), cwd=HERE, capture_output=True, text=True)
        msg = ((p.stdout + p.stderr).strip().splitlines() or [""])[-1]
        res["confirm" if complete else "probes"].append([k * 1024, p.returncode == 0, msg[:300]])
        log("complete" if complete else "admit", k * 1024, p.returncode == 0, msg[:120])
        return p.returncode == 0

    try:
        if not start():
            res["reason"] = "the package did not load on the RX 9070 XT" + (f": {res['load_error']}" if res.get("load_error") else "")
            return 1
        if not sane():
            res["reason"] = "wrong answer to the sanity prompt"
            return 1
        hi = native // 1024
        if admit(hi):
            best = hi
        else:   # binary search: lo admitted (0: none yet), hi refused
            lo = 0
            while lo < hi - 1:
                mid = (lo + hi) // 2
                lo, hi = (mid, hi) if admit(mid) else (lo, mid)
            best = lo
        for _ in range(8):   # the admitted length must really finish, cold, and the server answer correctly after it; step down 5 % if not
            if best <= 0 or not start():
                break
            if admit(best, complete=True):
                try:
                    ok = sane()
                except Exception as e:   # the server failed after the long request (it ran out of memory outside the budget)
                    ok, res["after_error"] = False, str(e)[:200]
                if ok:
                    res.update(tested=True, max_context=min(best * 1024, native))
                    break
                log("no correct answer after", best * 1024)
            best -= max(1, best // 20)
        if not res["tested"]:
            res["reason"] = "no context completed with a correct answer after it"
        return 0 if res["tested"] else 1
    finally:
        if srv:
            srv.terminate()
            try:
                srv.wait(30)
            except subprocess.TimeoutExpired:
                srv.kill()
        json.dump(res, open(a.o, "w"), indent=1)
        log("RESULT", "tested" if res["tested"] else "not tested", res["max_context"], res.get("reason", ""))


if __name__ == "__main__":
    sys.exit(main())
