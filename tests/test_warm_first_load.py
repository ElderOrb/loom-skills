import http.server
import importlib.util
import json
import pathlib
import threading
import unittest

spec = importlib.util.spec_from_file_location("warm", pathlib.Path(__file__).resolve().parents[1] / ".github/scripts/warm_first_load.py")
warm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warm)


def serve(replies):
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            code, text = replies.pop(0) if len(replies) > 1 else replies[0]
            self.send_response(code)
            self.end_headers()
            self.wfile.write(text.encode())

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}", seen


class WarmFirstLoad(unittest.TestCase):
    def run_warm(self, replies, *args):
        srv, url, seen = serve(replies)
        try:
            return warm.main([url, "m", *args]), seen
        finally:
            srv.shutdown()

    def test_loaded(self):
        rc, seen = self.run_warm([(200, "{}")], "--vision")
        self.assertEqual(rc, 0)
        self.assertEqual(len(seen), 2)   # text, then image
        self.assertIsInstance(seen[1]["messages"][0]["content"], list)

    def test_loading_503_is_waited_for(self):
        warm.time.sleep = lambda s: None
        rc, seen = self.run_warm([(503, '{"error":{"message":"model is unavailable"}}'), (200, "{}")])
        self.assertEqual(rc, 0)
        self.assertEqual(len(seen), 2)

    def test_other_error_fails(self):
        rc, _ = self.run_warm([(500, "boom")])
        self.assertEqual(rc, 1)

    def test_dead_server_fails(self):
        self.assertEqual(warm.main(["http://127.0.0.1:1", "m"]), 1)


if __name__ == "__main__":
    unittest.main()
