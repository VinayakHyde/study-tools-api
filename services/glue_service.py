from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sqlite3
from ..common.config import settings

router = APIRouter(prefix="/glue", tags=["glue"])

class LinkRequest(BaseModel):
    card_id: str
    page_id: str
    block_id: Optional[str] = None

class LinkResponse(BaseModel):
    status: str
    error: Optional[str] = None

def get_db_connection():
    """Get SQLite database connection."""
    conn = sqlite3.connect(settings.DB_URL.replace("sqlite:///", ""))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with required tables."""
    conn = get_db_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS card_note_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_id TEXT NOT NULL,
                page_id TEXT NOT NULL,
                block_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(card_id, page_id)
            )
        """)
        conn.commit()
    finally:
        conn.close()

@router.post("/link", response_model=LinkResponse)
async def link_card_to_note(link: LinkRequest):
    """Link an Anki card to a Notion page."""
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO card_note_links(card_id, page_id, block_id) VALUES (?, ?, ?)",
            (link.card_id, link.page_id, link.block_id)
        )
        conn.commit()
        return LinkResponse(status="linked")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.get("/links/{card_id}", response_model=List[Dict[str, Any]])
async def get_links_for_card(card_id: str):
    """Get all Notion page links for an Anki card."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM card_note_links WHERE card_id = ?",
            (card_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.delete("/link/{card_id}/{page_id}", response_model=LinkResponse)
async def delete_link(card_id: str, page_id: str):
    """Delete a link between an Anki card and a Notion page."""
    conn = get_db_connection()
    try:
        conn.execute(
            "DELETE FROM card_note_links WHERE card_id = ? AND page_id = ?",
            (card_id, page_id)
        )
        conn.commit()
        return LinkResponse(status="deleted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# Create FastAPI app
app = FastAPI(
    title="Glue Service",
    description="Service for linking Anki cards with Notion pages",
    version="1.0.0"
)

app.include_router(router)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.GLUE_SERVICE_PORT) 