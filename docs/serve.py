#!/usr/bin/env python3
"""
serve.py — local development server for docs/ with cross-origin isolation.

ORT WebAssembly requires SharedArrayBuffer, which browsers only expose when
the page is cross-origin isolated (COOP + COEP headers).  Python's built-in
http.server does not set these headers, so this script adds them.

Usage (run from the repo root):
    python docs/serve.py          # serves on port 8080
    python docs/serve.py 9000     # serves on a custom port
"""
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


class COIHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that appends cross-origin isolation headers."""

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        # credentialless (not require-corp) so that cross-origin CDN resources
        # (ORT JS + WASM files from jsDelivr) load without needing CORP headers,
        # while still making window.crossOriginIsolated = true (SharedArrayBuffer).
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Suppress favicon 404 noise
        if args and "favicon" in str(args[0]):
            return
        super().log_message(fmt, *args)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    docs_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(docs_dir)
    server = HTTPServer(("", port), COIHandler)
    print(f"Serving SigmaQuoridor at  http://localhost:{port}")
    print(f"  Directory : {docs_dir}")
    print(f"  COOP/COEP : credentialless (SharedArrayBuffer + CDN resources enabled)")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
