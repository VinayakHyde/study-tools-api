from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import requests
from ..common.config import settings

router = APIRouter(prefix="/notion", tags=["notion"])

class NotionPage(BaseModel):
    title: str
    content: str
    parent_id: Optional[str] = None

class NotionResponse(BaseModel):
    result: Any
    error: Optional[str] = None

@router.post("/create_page", response_model=NotionResponse)
async def create_page(page: NotionPage):
    """Create a new page in Notion."""
    try:
        headers = {
            "Authorization": f"Bearer {settings.NOTION_API_TOKEN}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        
        data = {
            "parent": {"database_id": page.parent_id} if page.parent_id else {"type": "page_id", "page_id": page.parent_id},
            "properties": {
                "title": {
                    "title": [
                        {
                            "text": {
                                "content": page.title
                            }
                        }
                    ]
                }
            },
            "children": [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": page.content
                                }
                            }
                        ]
                    }
                }
            ]
        }
        
        response = requests.post(
            "https://api.notion.com/v1/pages",
            headers=headers,
            json=data
        )
        return NotionResponse(result=response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/databases", response_model=NotionResponse)
async def get_databases():
    """Get list of all databases."""
    try:
        headers = {
            "Authorization": f"Bearer {settings.NOTION_API_TOKEN}",
            "Notion-Version": "2022-06-28"
        }
        
        response = requests.get(
            "https://api.notion.com/v1/search",
            headers=headers,
            json={"filter": {"property": "object", "value": "database"}}
        )
        return NotionResponse(result=response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pages/{database_id}", response_model=NotionResponse)
async def get_pages(database_id: str):
    """Get all pages in a database."""
    try:
        headers = {
            "Authorization": f"Bearer {settings.NOTION_API_TOKEN}",
            "Notion-Version": "2022-06-28"
        }
        
        response = requests.post(
            f"https://api.notion.com/v1/databases/{database_id}/query",
            headers=headers
        )
        return NotionResponse(result=response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create FastAPI app
app = FastAPI(
    title="Notion Service",
    description="Service for interacting with Notion API",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.NOTION_SERVICE_PORT) 