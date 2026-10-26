"""FastAPI WebSocket server that streams training progress to the dashboard.

Run from the backend/ folder:
    uvicorn server.main:app --reload --port 8000
"""
import asyncio
import json

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from engine import Network, make_dataset

app = FastAPI(title="Training Visualizer API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEFAULT_CONFIG = {
    "dataset": "moons",
    "hidden": [8, 8],
    "activation": "tanh",
    "optimizer": "adam",
    "lr": 0.03,
}

# speed level -> (epochs per message, seconds between messages)
SPEED_LEVELS = {
    1: (1, 0.06),
    2: (2, 0.04),
    3: (5, 0.03),    # default
    4: (10, 0.03),
    5: (25, 0.03),
}
DEFAULT_SPEED = 3
GRID = 24                # decision boundary resolution (GRID x GRID)


class Session:
    """One training run: dataset + network + the loop that streams updates."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.config = dict(DEFAULT_CONFIG)
        self.running = False
        self.task = None
        self.speed = DEFAULT_SPEED
        self.build()

    def build(self):
        c = self.config
        self.X, self.y = make_dataset(c["dataset"], n=200)
        sizes = [2, *c["hidden"], 1]
        self.net = Network(sizes, activation=c["activation"], optimizer=c["optimizer"], lr=c["lr"])
        self.epoch = 0
        pad = 0.4
        self.bounds = [
            float(self.X[:, 0].min() - pad), float(self.X[:, 0].max() + pad),
            float(self.X[:, 1].min() - pad), float(self.X[:, 1].max() + pad),
        ]
        xs = np.linspace(self.bounds[0], self.bounds[1], GRID)
        ys = np.linspace(self.bounds[2], self.bounds[3], GRID)
        gx, gy = np.meshgrid(xs, ys)
        self.grid = np.c_[gx.ravel(), gy.ravel()]

    async def send(self, payload: dict):
        await self.ws.send_text(json.dumps(payload))

    async def send_init(self):
        points = [[float(x), float(yv), int(l)] for (x, yv), l in zip(self.X, self.y.ravel())]
        await self.send({
            "type": "init",
            "layers": self.net.sizes,
            "points": points,
            "config": self.config,
            "params": self.net.n_params(),
            "speed": self.speed,
        })
        await self.send_tick(0.0, 0.0)

    async def send_tick(self, loss: float, acc: float):
        boundary = self.net.predict(self.grid).ravel().round(3).tolist()
        await self.send({
            "type": "tick",
            "epoch": self.epoch,
            "loss": round(loss, 5),
            "accuracy": round(acc, 4),
            "weights": self.net.weights(),
            "boundary": {"size": GRID, "values": boundary, "bounds": self.bounds},
        })

    async def loop(self):
        try:
            while self.running:
                epochs_per_tick, delay = SPEED_LEVELS[self.speed]
                loss = acc = 0.0
                for _ in range(epochs_per_tick):
                    loss, acc = self.net.train_step(self.X, self.y)
                    self.epoch += 1
                await self.send_tick(loss, acc)
                await asyncio.sleep(delay)
        except (WebSocketDisconnect, RuntimeError):
            self.running = False

    def set_speed(self, level):
        if isinstance(level, int) and level in SPEED_LEVELS:
            self.speed = level

    async def start(self):
        if self.running:
            return
        self.running = True
        await self.send({"type": "status", "running": True})
        self.task = asyncio.create_task(self.loop())

    async def pause(self):
        self.running = False
        if self.task:
            await self.task
            self.task = None
        await self.send({"type": "status", "running": False})

    async def configure(self, new_config: dict | None):
        await self.pause()
        if new_config:
            self.config.update(self._clean(new_config))
        self.build()
        await self.send_init()

    @staticmethod
    def _clean(cfg: dict) -> dict:
        out = {}
        if cfg.get("dataset") in ("moons", "circles", "spiral", "xor"):
            out["dataset"] = cfg["dataset"]
        if isinstance(cfg.get("hidden"), list):
            out["hidden"] = [max(1, min(16, int(n))) for n in cfg["hidden"]][:4] or [4]
        if cfg.get("activation") in ("tanh", "relu", "sigmoid"):
            out["activation"] = cfg["activation"]
        if cfg.get("optimizer") in ("adam", "sgd"):
            out["optimizer"] = cfg["optimizer"]
        if isinstance(cfg.get("lr"), (int, float)):
            out["lr"] = float(min(max(cfg["lr"], 0.0001), 1.0))
        return out


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session(ws)
    await session.send_init()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "start":
                await session.start()
            elif kind == "pause":
                await session.pause()
            elif kind == "reset":
                await session.configure(None)
            elif kind == "speed":
                session.set_speed(msg.get("value"))
            elif kind == "configure":
                await session.configure(msg.get("config"))
    except WebSocketDisconnect:
        session.running = False