# Training Visualizer: backend

A neural network built from scratch with NumPy, streamed live to the dashboard over WebSocket.

## Run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pytest                          # engine tests (XOR, gradient check, moons)
uvicorn server.main:app --reload --port 8000
```

The dashboard connects to `ws://localhost:8000/ws`.

## Structure

```
engine/
  activations.py   ReLU, Sigmoid, Tanh, Linear
  layers.py        Dense layer with manual backprop
  losses.py        MSE, binary cross-entropy
  optimizers.py    SGD, Adam
  network.py       Network: forward, backward, train_step
  datasets.py      moons, circles, spiral, xor
server/main.py     FastAPI WebSocket server
tests/             pytest
```

## WebSocket protocol

Client to server:

| Message | Effect |
|---|---|
| `{"type":"start"}` | Start or resume training |
| `{"type":"pause"}` | Pause |
| `{"type":"reset"}` | Rebuild the network with the same config |
| `{"type":"configure","config":{...}}` | Apply a new config (dataset, hidden, activation, optimizer, lr) |

Server to client:

| Message | Content |
|---|---|
| `init` | layer sizes, dataset points, config, parameter count |
| `tick` | epoch, loss, accuracy, weights, decision boundary grid |
| `status` | `running: true/false` |
