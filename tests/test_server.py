import json

from fastapi.testclient import TestClient

from server.main import app


def test_speed_changes_epochs_per_tick():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        init = json.loads(ws.receive_text())
        assert init["speed"] == 3
        ws.receive_text()  # first tick (epoch 0)
        ws.send_text(json.dumps({"type": "speed", "value": 5}))
        ws.send_text(json.dumps({"type": "start"}))
        epochs = []
        while len(epochs) < 3:
            msg = json.loads(ws.receive_text())
            if msg["type"] == "tick":
                epochs.append(msg["epoch"])
        assert epochs[1] - epochs[0] == 25


def test_invalid_speed_is_ignored():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_text()
        ws.receive_text()
        ws.send_text(json.dumps({"type": "speed", "value": 999}))
        ws.send_text(json.dumps({"type": "speed", "value": [1, 2]}))
        ws.send_text(json.dumps({"type": "reset"}))
        init = json.loads(ws.receive_text())
        while init["type"] != "init":  # reset first sends a "status" message
            init = json.loads(ws.receive_text())
        assert init["speed"] == 3