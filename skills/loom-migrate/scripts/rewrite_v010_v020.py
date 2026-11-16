#!/usr/bin/env python3
"""The mechanical source migration hrx-loom v0.1.0 -> v0.2.0 on Loom text, a package or a tree. Idempotent; standard library only.

usage: rewrite_v010_v020.py [--check] [--json OUT] [--no-publish-epoch] [--no-card] [--py] PATH...

PATH is a .loom file, a package (a directory with model.json) or any tree. Rewrites, in place:
  vector.constant X : vector<NxT>   -> vector.splatf (T float: f16 f32 f64 bf16 f8*) | vector.splati (int, predicate, index)
  command.program.{def,decl,launch} -> command.{def,decl,launch}
  publish_epoch (def, decl and call sites of @publish_epoch, @*_publish_epoch, @*_publish_epoch_v1): prepends the retained
      `%control_state: buffer` argument (v0.2 ABI); skipped by --no-publish-epoch
  model.json: removes presets.*.execution.pool_capacity (v0.2 rejects it); skipped by --no-card
Files: *.loom, *.loom-test, *.patch; with --py also *.py and *.md (generator sources and docs: renames only where the element type is
literal). `vector.constanti` (the dense-list op) is unchanged in v0.2 and is never touched.
Reports what it could NOT decide (a vector.constant whose element type is a placeholder, an old vision component ABI) as `manual`;
those need a hand edit (see versions/v0.1.0-to-v0.2.0.md). --check changes nothing and exits 1 if any file would change.
"""
import argparse
import json
import os
import re
import sys

LOOM_EXT = (".loom", ".loom-test", ".patch")
PY_EXT = (".py", ".md")
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv"}

VC = re.compile(r"\bvector\.constant(?=\s)([^\n]*?:\s*vector<[^>\n]*?x(\w+)>)")
VC_ANY = re.compile(r"\bvector\.constant(?=\s)")
CP = re.compile(r"\bcommand\.program\.(def|decl|launch)\b")
PE_NAME = r"@(?:\w+_)?publish_epoch(?:_v1)?"
PE_SIG = re.compile(r"^(func\.(?:def|decl)(?: \w+)* " + PE_NAME + r"\()(?=%plan\s*:)", re.M)
PE_CALL = re.compile(r"(func\.call " + PE_NAME + r"\()(?=%plan\b)([^\n]*?\)\s*:\s*\()")
OLD_PREPARE_BATCH = re.compile(r"@text_image_prepare_batch_v1\(%state: buffer, %tags: buffer, %ranges")
FLOAT = re.compile(r"^(?:f\d+\w*|bf16)$")
INT = re.compile(r"^(?:i\d+|index|offset)$")


def rewrite_text(text, publish_epoch=True):
    """Returns (new_text, counts, manual) for one source text."""
    counts = {"splati": 0, "splatf": 0, "command": 0, "publish_epoch": 0}
    manual = []

    def vc(m):
        t = m.group(2)
        if FLOAT.match(t):
            counts["splatf"] += 1
            return "vector.splatf" + m.group(1)
        if INT.match(t):
            counts["splati"] += 1
            return "vector.splati" + m.group(1)
        manual.append(f"vector.constant of element type {t!r}")
        return m.group(0)

    new = VC.sub(vc, text)
    left = len(VC_ANY.findall(new)) - sum(1 for x in manual if x.startswith("vector.constant"))
    if left > 0:
        manual.append(f"{left} vector.constant with a non-literal element type (a generator placeholder)")
    new, n = CP.subn(r"command.\1", new)
    counts["command"] = n
    if publish_epoch:
        new, n1 = PE_SIG.subn(r"\1%control_state: buffer, ", new)
        new, n2 = PE_CALL.subn(lambda m: m.group(1) + "%control_state, " + m.group(2) + "buffer, ", new)
        counts["publish_epoch"] = n1 + n2
    if OLD_PREPARE_BATCH.search(new):
        manual.append("vision component finite ABI: prepare_batch/run/collect_batch (hand edit)")
    return new, counts, manual


def rewrite_card(text):
    """Remove `"pool_capacity": N` lines from model.json text, keeping the formatting. Returns (text, removed)."""
    lines, out, removed = text.split("\n"), [], 0
    for ln in lines:
        if re.match(r'^\s*"pool_capacity"\s*:\s*[^,\n{}\[\]]+,?\s*$', ln):
            removed += 1
            if not ln.rstrip().endswith(",") and out and out[-1].rstrip().endswith(","):
                out[-1] = out[-1].rstrip()[:-1]   # it was the last key: the previous line loses its comma
            continue
        out.append(ln)
    return "\n".join(out), removed


def files_under(paths, exts):
    for p in paths:
        if os.path.isfile(p):
            yield p
            continue
        for d, dirs, names in os.walk(p):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            for n in sorted(names):
                if n.endswith(exts) or n == "model.json":
                    yield os.path.join(d, n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--no-publish-epoch", action="store_true")
    ap.add_argument("--no-card", action="store_true")
    ap.add_argument("--py", action="store_true", help="also rewrite .py/.md generator sources and docs")
    a = ap.parse_args(argv)
    exts = LOOM_EXT + (PY_EXT if a.py else ())
    changed, manual, totals = {}, {}, {"splati": 0, "splatf": 0, "command": 0, "publish_epoch": 0, "pool_capacity": 0}
    for f in files_under(a.paths, exts):
        with open(f, encoding="utf-8", newline="") as fh:
            text = fh.read()
        if os.path.basename(f) == "model.json":
            if a.no_card:
                continue
            new, n = rewrite_card(text)
            counts = {"pool_capacity": n}
            man = []
        else:
            new, counts, man = rewrite_text(text, not a.no_publish_epoch)
        for k, v in counts.items():
            totals[k] += v
        if man:
            manual[f] = man
        if new != text:
            changed[f] = counts
            if not a.check:
                with open(f, "w", encoding="utf-8", newline="") as fh:
                    fh.write(new)
    verb = "would change" if a.check else "changed"
    print(f"{verb} {len(changed)} file(s): " + ", ".join(f"{k}={v}" for k, v in totals.items()))
    by_what = {}
    for f, m in sorted(manual.items()):
        for x in m:
            by_what.setdefault(x, []).append(f)
    for x, fs in by_what.items():
        print(f"manual: {x}: {len(fs)} file(s), e.g. {fs[0]}" + (" (all in --json)" if len(fs) > 1 else ""))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"changed": changed, "manual": manual, "totals": totals}, fh, indent=1)
    return 1 if (a.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
