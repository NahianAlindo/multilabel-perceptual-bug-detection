#!/usr/bin/env python3
"""
Minimal Flask reachability test.

Purpose: answer ONE question before building any real inference server on
nibi — can a process running inside a SLURM GPU job actually be reached over
HTTP from outside the cluster (a personal laptop, eventually a public
frontend)? Nothing else. No model, no GPU work, no FastAPI/uvicorn/gunicorn.

If NGROK_AUTH_TOKEN is set, also opens an ngrok tunnel to this same server —
compute nodes have confirmed outbound access (W&B logging works from training
jobs), so this tests whether an *outbound* tunnel is a viable way to reach an
inbound-blocked compute node, same mechanism already used for the Kaggle
inference path (see kaggle_inference_server.py).
"""

import os
import socket

from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/health")
def health():
    # Returning the hostname is the whole point of this test: whatever
    # hostname/IP you actually manage to reach this from confirms which
    # network path worked (direct vs. only via an SSH tunnel, etc).
    return jsonify({"status": "ok", "hostname": socket.gethostname()})


@app.route("/")
def index():
    return "Flask reachability test is running.\n"


if __name__ == "__main__":
    ngrok_token = os.environ.get("NGROK_AUTH_TOKEN", "")
    if ngrok_token:
        # A tunnel is initiated FROM here, outbound, to ngrok's relay
        # servers — it never needs this node to be inbound-reachable at
        # all, unlike host="0.0.0.0" below (which only helps if something
        # can already route to this node's IP in the first place).
        try:
            from pyngrok import ngrok, conf as ngrok_conf
            ngrok_conf.get_default().auth_token = ngrok_token
            public_url = ngrok.connect(5000, "http")
            print("==============================================")
            print(f"[NGROK] Tunnel open: {public_url}")
            print(f"[NGROK] Test from anywhere with: curl {public_url}/health")
            print("==============================================")
        except Exception as e:
            print(f"[NGROK] Failed to open tunnel: {e}")
    else:
        print("[NGROK] NGROK_AUTH_TOKEN not set — skipping tunnel, "
              "direct-reachability test only.")

    # host="0.0.0.0" is required for ANY chance of DIRECT external
    # reachability — the Flask dev-server default (127.0.0.1) only accepts
    # connections from the same machine, which would make even the
    # login-node curl test meaningless. Necessary but not sufficient: the
    # cluster's own network/firewall configuration between a compute node
    # and the outside world is the actual thing that part of the test is
    # discovering. The ngrok tunnel above is independent of this and works
    # (or doesn't) regardless of what 0.0.0.0 does or doesn't achieve.
    app.run(host="0.0.0.0", port=5000)
