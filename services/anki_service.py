from fastapi import APIRouter, FastAPI, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field
import requests
from typing import List, Dict, Any
from common.config import settings

# Use centralized configuration
ANKI_CONNECT_URL = settings.ANKI_CONNECT_URL

router = APIRouter()

# ─── Models ───────────────────────────────────────────────────────────────────

class CreateDeckIn(BaseModel):
    name: str
    parent: str | None = None

class BasicCardIn(BaseModel):
    deck_id: str
    front: str
    back: str
    tags: list[str] = []
    external_id: str | None = None

class ClozeCardIn(BaseModel):
    deck_id: str
    text_with_cloze: str
    tags: list[str] = []
    external_id: str | None = None

class ImageOcclusionMask(BaseModel):
    x: int
    y: int
    width: int
    height: int

class ImageOcclusionIn(BaseModel):
    deck_id: str
    image_uri: str
    masks: list[ImageOcclusionMask]
    hint: str | None = None
    tags: list[str] = []

class UpdateCardIn(BaseModel):
    # Fields to update on an existing card; card_id is supplied via path
    front: str | None = None
    back: str | None = None
    tags: list[str] | None = None

class TagCardsIn(BaseModel):
    card_ids: list[int]
    tags_to_add: list[str]
    tags_to_remove: list[str] | None = None

class AdjustScheduleIn(BaseModel):
    card_id: int
    next_interval_days: int
    ease_factor: float | None = None
    
# Schema for searching cards by deck and text
class SearchCardsIn(BaseModel):
    query_text: str = Field("", description="Text to match (empty to list all in deck)")
    deck_id: str | None = Field(None, description="Optional deck to constrain search")
    match_mode: str = Field("exact", description="'exact' or 'fuzzy'")
    limit: int = Field(20, description="Max number of results")

class CreateDeckResponse(BaseModel):
    deck: str = Field(..., description="Name of the created deck")

class CardResponse(BaseModel):
    card_id: int = Field(..., description="ID of the created card")

class NoteResponse(BaseModel):
    note_id: int = Field(..., description="ID of the created note")

class TagResponse(BaseModel):
    notes: List[int] = Field(..., description="List of note IDs that were tagged")

class SearchResponse(BaseModel):
    matches: List[Dict[str, Any]] = Field(..., description="List of matching cards")

# ─── Utility ───────────────────────────────────────────────────────────────────

def invoke(action: str, params: dict | None = None):
    payload = {"action": action, "version": 6}
    if params is not None:
        payload["params"] = params
    try:
        resp = requests.post(ANKI_CONNECT_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AnkiConnect error: {e}")
    if data.get("error"):
        raise HTTPException(status_code=500, detail=data["error"])
    return data.get("result")

def _resolve_deck_name(deck_id: str) -> str:
    """Given a deck identifier (name or numeric ID), return the deck name."""
    # Fetch mapping of deck names to IDs
    decks = invoke("deckNamesAndIds") or {}
    # If numeric ID provided, find corresponding name
    if deck_id.isdigit():
        target = int(deck_id)
        for name, did in decks.items():
            if did == target:
                return name
        raise HTTPException(status_code=404, detail=f"Deck ID '{deck_id}' not found")
    # Otherwise, assume name and verify existence
    if deck_id not in decks:
        raise HTTPException(status_code=404, detail=f"Deck name '{deck_id}' not found")
    return deck_id

# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/decks")
async def list_decks():
    """List all deck names and IDs."""
    decks = invoke("deckNamesAndIds")
    # Convert to list of objects with explicit name and id fields
    deck_list = [{"name": name, "id": deck_id} for name, deck_id in decks.items()]
    return {"decks": deck_list}

@router.post("/decks", response_model=CreateDeckResponse)
async def create_deck(req: CreateDeckIn):
    """Create a new deck, optionally nested under a parent."""
    full_name = f"{req.parent}::{req.name}" if req.parent else req.name
    invoke("createDeck", {"deck": full_name})
    return {"deck": full_name}

@router.post("/cards/basic", response_model=CardResponse)
async def add_basic_card(card: BasicCardIn):
    """Create a Basic front/back card."""
    deck_name = _resolve_deck_name(card.deck_id)
    note = {
        "deckName": deck_name,
        "modelName": "Basic",
        "fields": {"Front": card.front, "Back": card.back},
        "tags": card.tags
    }
    if card.external_id:
        note["options"] = {"allowDuplicate": False}
        note.setdefault("tags", []).append(f"ext:{card.external_id}")
    card_id = invoke("addNote", {"note": note})
    return {"card_id": card_id}

@router.post("/cards/cloze", response_model=NoteResponse)
async def add_cloze_card(card: ClozeCardIn):
    """Create a Cloze-deletion card."""
    deck_name = _resolve_deck_name(card.deck_id)
    note = {
        "deckName": deck_name,
        "modelName": "Cloze",
        "fields": {"Text": card.text_with_cloze},
        "tags": card.tags
    }
    if card.external_id:
        note["options"] = {"allowDuplicate": False}
        note.setdefault("tags", []).append(f"ext:{card.external_id}")
    note_id = invoke("addNote", {"note": note})
    return {"note_id": note_id}

# TODO: Image occlusion support is not yet implemented. Temporarily disabled.
# Uncomment and implement when ready.
# @app.post("/cards/image_occlusion")
# async def add_image_occlusion(req: ImageOcclusionIn):
#     """Generate image-occlusion cards from an image & mask data."""
#     # TODO: implement via storeMediaFile + addNote with Image Occlusion Enhanced model
#     raise HTTPException(status_code=501, detail="Not implemented")

@router.put("/cards/{card_id}")
async def update_card(
    card_id: int = Path(..., description="Card ID to update"),
    upd: UpdateCardIn = Body(..., description="Fields to update on the note"),
):
    """Edit an existing card's fields and tags."""
    # Map card -> note
    note_ids = invoke("cardsToNotes", {"cards": [card_id]})
    if not note_ids:
        raise HTTPException(status_code=404, detail="Card not found")
    note_id = note_ids[0]
    # Update fields
    fields = {}
    if upd.front is not None:
        fields["Front"] = upd.front
    if upd.back is not None:
        fields["Back"] = upd.back
    if fields:
        invoke("updateNoteFields", {"note": {"id": note_id, "fields": fields}})
    # Update tags
    if upd.tags is not None:
        invoke("updateNote", {"note": {"id": note_id, "tags": upd.tags}})
    return {"note_id": note_id}

@router.post("/cards/tag", response_model=TagResponse)
async def tag_cards(req: TagCardsIn):
    """Bulk add/remove tags on cards."""
    note_ids = invoke("cardsToNotes", {"cards": req.card_ids})
    if req.tags_to_add:
        invoke("addTags", {"notes": note_ids, "tags": " ".join(req.tags_to_add)})
    if req.tags_to_remove:
        invoke("removeTags", {"notes": note_ids, "tags": " ".join(req.tags_to_remove)})
    return {"notes": note_ids}

# @router.post("/cards/schedule")
# async def adjust_schedule(req: AdjustScheduleIn):
#     """Stub: Override next-review interval using custom algorithm."""
#     # TODO: no direct AnkiConnect support; consider answerCard with artificial ease or external scheduler
#     raise HTTPException(status_code=501, detail="Not implemented")

@router.get("/cards/due")
async def get_due_cards(deck_id: str = Query(...), limit: int = Query(50)):
    """List next cards due for review."""
    # Resolve deck identifier to deck name
    deck_name = _resolve_deck_name(deck_id)
    # Build due query, wrapping deck name in quotes for spaces
    query = f'deck:"{deck_name}" is:due'
    # Find card IDs and apply limit
    card_ids = invoke("findCards", {"query": query}) or []
    card_ids = card_ids[:limit]
    # Retrieve card info
    cards = invoke("cardsInfo", {"cards": card_ids}) if card_ids else []
    return {"due_cards": cards}

@router.post("/cards/search", response_model=SearchResponse)
async def search_cards(req: SearchCardsIn):
    deck_name = _resolve_deck_name(req.deck_id)
    text = req.query_text.strip().strip('"').strip()
    if not text:
        q = f'deck:"{deck_name}"'
    else:
        q = (text if req.match_mode=="exact" else f"*{text}*")
        q = f'deck:"{deck_name}" {q}'
    ids = invoke("findCards", {"query": q}) or []
    cards = invoke("cardsInfo", {"cards": ids[:req.limit]}) if ids else []
    return {"matches": cards}

class SimilarByEmbeddingIn(BaseModel):
    """Parameters for semantic similarity search over cards."""
    embedding_vec: list[float] = Field(..., description="Embedding vector to match against")
    deck_id: str | None = Field(None, description="Optional deck ID to constrain search")
    top_k: int = Field(5, description="Number of similar cards to return")

# @router.post("/cards/similar_by_embedding")
# async def similar_cards_by_embedding(
#     payload: SimilarByEmbeddingIn = Body(..., description="Similarity search parameters")
# ):
#     """Stub: Vector-search over existing cards to catch semantic duplicates."""
#     # TODO: integrate with external vector index or embedding service
#     raise HTTPException(status_code=501, detail="Not implemented")

@router.get("/cards/{card_id}")
async def get_card(card_id: int = Path(...)):
    """Return full JSON for a card ID."""
    cards = invoke("cardsInfo", {"cards": [card_id]})
    if not cards:
        raise HTTPException(status_code=404, detail="Card not found")
    return cards[0]

# To run:
# uvicorn services.anki_service:app --reload --port ${ANKI_SERVICE_PORT:-8001}
# ASGI app for standalone service
app = FastAPI()
app.include_router(router)
