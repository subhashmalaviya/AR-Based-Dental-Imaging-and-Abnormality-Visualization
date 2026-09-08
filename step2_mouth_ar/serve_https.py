#!/usr/bin/env python3
"""
serve_https.py -- serve the built app over HTTPS so a phone can use its camera.

getUserMedia only works in a *secure context*. http://localhost counts, but
http://192.168.x.x -- which is what a phone on your Wi-Fi would hit -- does not,
so the camera silently never starts and the page just sits there. This serves
dist/ over HTTPS with a self-signed certificate, which is enough for Android
Chrome (tap "Advanced -> Proceed").

    python3 serve_https.py                    # serves ./dist on :8443
    python3 serve_https.py --port 9443        # different port

iOS Safari is stricter and usually refuses self-signed certificates outright.
For an iPhone/iPad, use a tunnel instead so you get a real certificate:

    npx vite preview --host                          # terminal 1
    cloudflared tunnel --url http://localhost:4173   # terminal 2
"""
import argparse
import http.server
import os
import socket
import ssl
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CERT = HERE / ".cert" / "server.pem"


def local_ip() -> str:
    """Best-effort LAN address of this machine (no packets are actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert() -> Path:
    if CERT.exists():
        return CERT
    CERT.parent.mkdir(parents=True, exist_ok=True)
    ip = local_ip()
    # The SAN must list the LAN IP. Without it, mobile browsers reject the
    # certificate outright instead of offering the "proceed anyway" escape.
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(CERT), "-out", str(CERT), "-days", "365",
        "-subj", "/CN=dental-ar-step2",
        "-addext", f"subjectAltName=DNS:localhost,IP:127.0.0.1,IP:{ip}",
    ], check=True, capture_output=True)
    print(f"generated self-signed certificate at {CERT}")
    return CERT


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # no-store stops the phone serving a stale build after you rebuild.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def guess_type(self, path):
        # .wasm needs the right MIME type or streaming compilation fails.
        p = str(path)
        if p.endswith(".wasm"):
            return "application/wasm"
        if p.endswith(".task"):
            return "application/octet-stream"
        return super().guess_type(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--dir", default="dist")
    args = ap.parse_args()

    root = (HERE / args.dir).resolve()
    if not root.is_dir():
        sys.exit(f"{root} does not exist -- run `npm run build` first.")
    os.chdir(root)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(ensure_cert())

    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    ip = local_ip()
    print(f"serving {root}")
    print(f"  this machine : https://localhost:{args.port}/")
    print(f"  your phone   : https://{ip}:{args.port}/")
    print()
    print("The certificate is self-signed, so the phone warns once:")
    print("  Android Chrome : Advanced -> Proceed to ... (unsafe)")
    print("  iOS Safari     : often refuses -- use a cloudflared/ngrok tunnel")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
