"""
FastAPI Application Entry Point
Capstone Research: Intelligent Traffic Management System
Author: nkhalfe56-star | IIT Jodhpur
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import asyncio
import json
import logging
import uvicorn
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class IntersectionState(BaseModel):
    intersection_id: str
    queue_lengths: List[float] = Field(..., description="Queue length per lane")
    waiting_times: List[float] = Field(..., description="Average wait time per lane (s)")
    vehicle_density: List[float] = Field(..., description="Vehicle density per lane [0,1]")
    current_phase: int = Field(0, ge=0, description="Current signal phase index")
    phase_duration: float = Field(0.0, ge=0, description="Duration of current phase (s)")


class SignalAction(BaseModel):
    intersection_id: str
    recommended_phase: int
    phase_duration: float
    confidence: float
    timestamp: str


class CongestionPredictionRequest(BaseModel):
    intersection_id: str
    recent_data: List[List[float]] = Field(
        ..., description="Recent traffic data: shape (seq_len, n_features)"
    )


class CongestionPredictionResponse(BaseModel):
    intersection_id: str
    predictions: Dict[str, float]
    timestamp: str


class RouteRequest(BaseModel):
    origin: Dict[str, float] = Field(..., description="{lat, lon}")
    destination: Dict[str, float] = Field(..., description="{lat, lon}")
    avoid_congested: bool = True


class RouteResponse(BaseModel):
    route_id: str
    waypoints: List[Dict[str, float]]
    estimated_time_min: float
    distance_km: float
    congestion_score: float


# ---------------------------------------------------------------------------
# WebSocket Manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)
        logger.info(f"WebSocket disconnected: {client_id}")

    async def broadcast(self, message: dict):
        disconnected = []
        for client_id, ws in self.active_connections.items():
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(client_id)
        for c in disconnected:
            self.disconnect(c)

    async def send_to(self, client_id: str, message: dict):
        ws = self.active_connections.get(client_id)
        if ws:
            await ws.send_json(message)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# App Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Intelligent Traffic Management System API...")
    # Initialize models, DB connections, etc. here
    yield
    logger.info("Shutting down ITMS API...")


# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Intelligent Traffic Management System API",
    description="""
    AI-powered traffic signal control, congestion prediction, and route optimization.

    ## Capstone Research Project
    **IIT Jodhpur | B.Tech Computer Engineering & B.S. Applied AI and Data Science**

    ### Features
    - Real-time traffic signal control via Deep Reinforcement Learning
    - Multi-horizon congestion prediction (5/10/15 min) via LSTM
    - Dynamic route optimization using graph algorithms + RL
    - WebSocket streaming for live dashboard updates
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "service": "ITMS API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Signal Control Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/signals/recommend", response_model=SignalAction, tags=["Signal Control"])
async def recommend_signal_action(state: IntersectionState):
    """
    Get recommended signal phase for an intersection using the DRL agent.

    The agent processes current queue lengths, waiting times, and vehicle
    density to output the optimal next signal phase.
    """
    # TODO: Load DRL agent and run inference
    # agent = get_agent(state.intersection_id)
    # action = agent.select_action(state_vector, training=False)

    # Placeholder response
    return SignalAction(
        intersection_id=state.intersection_id,
        recommended_phase=0,
        phase_duration=30.0,
        confidence=0.87,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/api/v1/signals/{intersection_id}/status", tags=["Signal Control"])
async def get_signal_status(intersection_id: str):
    """Get current signal status for an intersection."""
    return {
        "intersection_id": intersection_id,
        "current_phase": 2,
        "phase_remaining_sec": 14.3,
        "last_updated": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Congestion Prediction Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/predictions/congestion",
    response_model=CongestionPredictionResponse,
    tags=["Predictions"],
)
async def predict_congestion(req: CongestionPredictionRequest):
    """
    Predict congestion probability for 5, 10, and 15 minute horizons.

    Input recent traffic data (seq_len x n_features) and receive
    congestion probability for each time horizon.
    """
    # TODO: Run LSTM inference
    # predictor = get_predictor(req.intersection_id)
    # result = predictor.predict(np.array(req.recent_data))

    return CongestionPredictionResponse(
        intersection_id=req.intersection_id,
        predictions={"5min": 0.23, "10min": 0.41, "15min": 0.67},
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/api/v1/predictions/citywide", tags=["Predictions"])
async def get_citywide_congestion():
    """Get congestion heatmap data for the entire city network."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "congestion_map": [],  # List of {lat, lon, level} dicts
    }


# ---------------------------------------------------------------------------
# Route Optimization Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/routing/optimize", response_model=RouteResponse, tags=["Routing"])
async def optimize_route(req: RouteRequest):
    """
    Find optimal route between two points considering real-time traffic.

    Uses modified Dijkstra with live congestion-weighted edge costs.
    """
    # TODO: Run route optimization
    return RouteResponse(
        route_id="route_001",
        waypoints=[req.origin, req.destination],
        estimated_time_min=12.5,
        distance_km=5.2,
        congestion_score=0.31,
    )


# ---------------------------------------------------------------------------
# WebSocket Streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/traffic/{client_id}")
async def websocket_traffic_stream(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time traffic data streaming.
    Streams signal states, vehicle counts, and alerts to connected dashboards.
    """
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            # Echo back with server timestamp
            await manager.send_to(client_id, {
                "type": "ack",
                "received": msg,
                "timestamp": datetime.utcnow().isoformat(),
            })
    except WebSocketDisconnect:
        manager.disconnect(client_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
