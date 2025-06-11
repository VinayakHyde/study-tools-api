from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
import os
import sqlite3
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mapping.db")
# URL for local AnkiConnect JSON-RPC (used for fetching card/note context)
ANKI_CONNECT_URL = os.getenv("ANKI_CONNECT_URL", "http://localhost:8765")
# Extract sqlite path
if DATABASE_URL.startswith("sqlite:///"):
    DB_PATH = DATABASE_URL.replace("sqlite:///", "")
else:
    raise RuntimeError("Only sqlite:/// URLs supported in prototype.")

# DB helper
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

router = APIRouter()

@router.on_event("startup")
async def init_db():
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS card_note_links (
            link_id TEXT PRIMARY KEY,
            card_id TEXT NOT NULL,
            page_id TEXT NOT NULL,
            block_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(card_id, page_id, block_id)
        )
        """
    )
    conn.commit()

class LinkIn(BaseModel):
    card_id: str
    page_id: str
    block_id: str | None = None

@router.post("/link")
async def link_card_to_note(payload: LinkIn):
    """Record a mapping between an Anki card and a Notion page/block."""
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO card_note_links(card_id,page_id,block_id) VALUES (?,?,?)",
            (payload.card_id, payload.page_id, payload.block_id)
        )
        conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "linked" if cur.rowcount else "exists"}

@router.get("/sync_status")
async def sync_status(since_timestamp: str | None = None):
    """Return all link records, optionally filtered by timestamp."""
    conn = get_conn()
    query = "SELECT * FROM card_note_links"
    params = []
    if since_timestamp:
        query += " WHERE created_at >= ?"
        params.append(since_timestamp)
    try:
        rows = conn.execute(query, params).fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"links": [dict(r) for r in rows]}

@router.get(
    "/context",
    summary="fetch_context_by_id",
    operation_id="fetch_context_by_id",
    tags=["context"],
    description="Retrieve the full JSON body of a card or note by ID, for grounding."
)
async def fetch_context_by_id(object_type: str, object_id: str):
    """
    object_type: "card" | "note"
    object_id: string ID of the card or note to retrieve
    """
    # Validate object type
    if object_type not in ("card", "note"):
        raise HTTPException(status_code=400, detail=f"Invalid object_type '{object_type}', must be 'card' or 'note'.")
    # Prepare AnkiConnect action and params
    try:
        if object_type == "card":
            action = "cardsInfo"
            params = {"cards": [int(object_id)]}
        else:
            action = "notesInfo"
            params = {"notes": [int(object_id)]}
    except ValueError:
        raise HTTPException(status_code=400, detail="object_id must be an integer string.")
    payload = {"action": action, "version": 6, "params": params}
    # Call AnkiConnect JSON-RPC
    try:
        resp = requests.post(ANKI_CONNECT_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AnkiConnect error: {e}")
    # Handle RPC error
    if data.get("error"):
        raise HTTPException(status_code=500, detail=data["error"])
    result = data.get("result")
    # Expect a list of one item
    if not isinstance(result, list) or not result:
        raise HTTPException(status_code=404, detail="No context returned from AnkiConnect.")
    # Return the contextual JSON
    return {"object_type": object_type, "object_id": object_id, "context": result[0]}

# To run:
# uvicorn services.glue_service:app --reload --port ${GLUE_SERVICE_PORT:-8003}
# ASGI app for standalone service
app = FastAPI()
app.include_router(router)
