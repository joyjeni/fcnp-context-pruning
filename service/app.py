#!/usr/bin/env python3
"""POST /prune and POST /agri/snapshot for FCNP (live data.gov.in, no fake prices)."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fcnp.data_gov_agri import fetch_snapshot_live  # noqa: E402
from fcnp.pruner import FCNPConfig, FlowBasedNetworkPruner  # noqa: E402
from fcnp.types import ContextElement  # noqa: E402


def prune(body: dict) -> dict:
    dim = 32
    elements = []
    for row in body.get("elements") or []:
        raw = list(row.get("embedding") or [])
        if len(raw) < dim:
            raw = raw + [0.0] * (dim - len(raw))
        emb = np.array(raw[:dim], dtype=float)
        n = np.linalg.norm(emb)
        if n:
            emb = emb / n
        else:
            emb = np.ones(dim) / np.sqrt(dim)
        elements.append(
            ContextElement(
                id=str(row.get("id", "el")),
                text=str(row.get("text", "")),
                embedding=emb,
                importance=float(row.get("importance", 0.5)),
            )
        )
    cfg = FCNPConfig(max_iterations=int(body.get("max_iterations") or 40))
    result = FlowBasedNetworkPruner(cfg).prune(
        elements,
        query_text=body.get("query"),
    )
    return {
        "service": "fcnp",
        "n_input": result.n_input,
        "n_output": result.n_output,
        "iterations": result.iterations,
        "converged": result.converged,
        "survivors": [
            {"id": el.id, "text": el.output_text() if hasattr(el, "output_text") else el.text, "tier": getattr(el.tier, "value", str(el.tier))}
            for el in result.survivors
        ],
        "formula": "D_ij(t+1)=(1-μ)D_ij + α|Q_ij|^γ",
    }


def snapshot(body: dict) -> dict:
    key = body.get("api_key") or os.environ.get("DATA_GOV_API_KEY") or ""
    if not key:
        return {
            "error": "DATA_GOV_API_KEY required. Refusing to invent mandi prices.",
            "service": "fcnp",
        }
    snap = fetch_snapshot_live(key)
    return {"service": "fcnp", **snap.to_dict()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path in {"/", "/health"}:
            self._send(200, {"ok": True, "service": "fcnp"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if path == "/prune":
            self._send(200, prune(body))
            return
        if path in {"/agri/snapshot", "/snapshot"}:
            payload = snapshot(body)
            self._send(400 if payload.get("error") else 200, payload)
            return
        self._send(404, {"error": "not found"})


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 43134
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"FCNP microservice http://127.0.0.1:{port}/prune", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
