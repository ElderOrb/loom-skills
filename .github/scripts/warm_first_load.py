#!/usr/bin/env python3
"""First-load warm-up of a just-started loom-serve (devlab_perf.sh serve_alone, safetensors packages).

usage: warm_first_load.py URL MODEL_ID [--vision] [--wait S]

The weights (and the image component) load lazily on the first request. On a fresh DevLab R9700 VM the first load of the MXFP4 safetensors card
fails (HAL RESOURCE_EXHAUSTED in libamdf memory_create, the server then answers 503 "model is unavailable" or drops the connection) while the
same load succeeds on the next server start (campaign mxfp4st, notes/sdk_blockers.md). Sends one tiny text request (and one tiny image request
with --vision), retrying 503 "unavailable" while the model is still loading. Exit 0 = loaded, 1 = the load failed (the caller restarts the server).
"""
import argparse
import base64
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib


def png(w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def request(url, model, content, wait):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 1, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    deadline = time.time() + wait
    while True:
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return r.status, ""
        except urllib.error.HTTPError as e:
            text = e.read()
            if e.code == 503 and b"unavailable" in text and time.time() < deadline:
                time.sleep(2)
                continue
            return e.code, text[:200].decode("utf-8", "replace")
        except (urllib.error.URLError, OSError) as e:   # RemoteDisconnected, connection refused: the server died
            return 0, f"{type(e).__name__}: {e}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--wait", type=float, default=120)
    a = ap.parse_args(argv)
    probes = [("text", "Say hi.")]
    if a.vision:
        probes.append(("image", [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png(64, 64, (220, 20, 20))).decode()}},
                                 {"type": "text", "text": "Color?"}]))
    for name, content in probes:
        code, detail = request(a.url, a.model, content, a.wait)
        print(f"warm {name}: HTTP {code} {detail}", flush=True)
        if code != 200:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
