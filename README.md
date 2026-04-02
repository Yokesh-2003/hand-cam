# Hand Air Draw (TypeScript + Python)

This demo:

- Tracks your hand in the browser (MediaPipe Hands) and draws **hand skeleton lines**
- Lets you **draw in the air** when you **point your index finger**
- Shows **color + brush size** controls
- (Optional) Sends the 21 landmarks to a **Python “deep learning” MLP** over WebSocket and displays the model probability

## Run the frontend (TypeScript)

```bash
cd frontend
npm install
npm run dev
```

Open the local URL shown by Vite (usually `http://localhost:5173`).

## Run the backend (Python deep learning concept)

In a second terminal:

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The page will automatically connect to `ws://127.0.0.1:8000/ws` and display `Python model p`.

## Deploy on Vercel

Vercel does **not** run long-lived `uvicorn` WebSocket servers. For Vercel deployment, this repo includes a **serverless Python API**:

- `api/index.py` exposes:
  - `POST /api/pointing`
  - `GET /api/health`
- `vercel.json` builds the Vite app from `frontend/`

The frontend will use:

- WebSocket when available (local dev), otherwise
- `POST /api/pointing` (Vercel/serverless)

## Notes

- The Python model is a tiny **2-layer neural network** trained at startup on **synthetic landmark data** (so it runs anywhere without a dataset).
- The app draws when either:
  - Local heuristic says “pointing”, **or**
  - Python model probability \(p > 0.65\)

