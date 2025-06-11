from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import requests
import os
from ..common.config import settings

router = APIRouter(prefix="/anki", tags=["anki"])

class AnkiCard(BaseModel):
    deck_name: str
    front: str
    back: str
    tags: Optional[List[str]] = None

class AnkiResponse(BaseModel):
    result: Any
    error: Optional[str] = None

@router.post("/add_note", response_model=AnkiResponse)
async def add_note(card: AnkiCard):
    """Add a new note to Anki."""
    try:
        response = requests.post(
            settings.ANKI_CONNECT_URL,
            json={
                "action": "addNote",
                "version": 6,
                "params": {
                    "note": {
                        "deckName": card.deck_name,
                        "modelName": "Basic",
                        "fields": {
                            "Front": card.front,
                            "Back": card.back
                        },
                        "tags": card.tags or []
                    }
                }
            }
        )
        return AnkiResponse(result=response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/decks", response_model=AnkiResponse)
async def get_decks():
    """Get list of all decks."""
    try:
        response = requests.post(
            settings.ANKI_CONNECT_URL,
            json={
                "action": "deckNames",
                "version": 6
            }
        )
        return AnkiResponse(result=response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cards/{deck_name}", response_model=AnkiResponse)
async def get_cards(deck_name: str):
    """Get all cards in a deck."""
    try:
        response = requests.post(
            settings.ANKI_CONNECT_URL,
            json={
                "action": "findNotes",
                "version": 6,
                "params": {
                    "query": f"deck:{deck_name}"
                }
            }
        )
        return AnkiResponse(result=response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create FastAPI app
app = FastAPI(
    title="Anki Service",
    description="Service for interacting with Anki through AnkiConnect",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.ANKI_SERVICE_PORT) 