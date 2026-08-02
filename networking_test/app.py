#!/usr/bin/env python3
"""
Minimal Flask reachability test.

Purpose: answer ONE question before building any real inference server on
nibi — can a process running inside a SLURM GPU job actually be reached over
HTTP from outside the cluster (a personal laptop, eventually a public
frontend)? Nothing else. No model, no GPU work, no FastAPI/uvicorn/gunicorn.
"""

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
    # host="0.0.0.0" is required for ANY chance of external reachability —
    # the Flask dev-server default (127.0.0.1) only accepts connections
    # from the same machine, which would make this whole test meaningless.
    # Binding to 0.0.0.0 is necessary but not sufficient: the cluster's own
    # network/firewall configuration between a compute node and the outside
    # world is the actual thing this test is trying to discover.
    app.run(host="0.0.0.0", port=5000)
