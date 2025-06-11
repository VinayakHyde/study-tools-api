# services/notion_service.py

import os
from dotenv import load_dotenv
import json
import logging
import re
import time
import random
from fastapi import APIRouter, FastAPI, Body, HTTPException, Path, Query
from starlette.concurrency import run_in_threadpool

from notion_client import Client
from notion_client.helpers import collect_paginated_api
import notion_client.errors
import requests
import httpx
from pydantic import BaseModel, Field, model_validator, RootModel, ConfigDict
from typing import Any, Dict, List, Optional, Union, Literal
from functools import wraps

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ▶ %(levelname)s ▶ %(message)s",
)
logger = logging.getLogger("notion_service")

# ─── Configuration ────────────────────────────────────────────────────────────
load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_VERSION = os.getenv("NOTION_VERSION", "2022-06-28")

if not NOTION_TOKEN:
    logger.critical("Missing NOTION_TOKEN in environment")

# Create a resilient session for requests
class ResilientSession(requests.Session):
    def request(self, method, url, **kwargs):
        max_retries = 5
        retry_delay = 1
        retries = 0

        while retries < max_retries:
            try:
                return super().request(method, url, **kwargs)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if "Temporary failure in name resolution" in str(e) and retries < max_retries - 1:
                    logger.warning(f"DNS resolution failed in ResilientSession, retrying in {retry_delay}s (attempt {retries + 1}/{max_retries})")
                    time.sleep(retry_delay * (2 ** retries))  # Exponential backoff
                    retries += 1
                else:
                    logger.error(f"Failed after {retries} retries: {str(e)}")
                    raise

# Create a custom client creator with DNS resilience
def create_resilient_notion_client():
    from notion_client.client import Client as OriginalClient
    
    # Patch the Client class to use our resilient session
    class ResilientClient(OriginalClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Replace the internal httpx client with our resilient session
            self.client = httpx.Client(
                base_url="https://api.notion.com",
                headers={
                    "Authorization": f"Bearer {NOTION_TOKEN}",
                    "Notion-Version": NOTION_VERSION,
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(connect=30.0, read=90.0, write=30.0, pool=30.0),  # Increased timeouts
                verify=True,  # SSL verification
                trust_env=True,  # Trust environment variables for proxy settings
                http2=False,  # Disable HTTP/2 which can cause issues with some proxies
            )
    
    return ResilientClient(auth=NOTION_TOKEN, notion_version=NOTION_VERSION)

# Function to get a notion client with retries
def get_notion_client():
    return create_resilient_notion_client()

router = APIRouter()

# ─── Helper model to accept/return any JSON object and include `properties` in schema ────────────
class FreeFormModel(RootModel[Dict[str, Any]]):
    """
    A root model to accept/return any JSON object,
    while ensuring the OpenAPI schema includes an (empty) ``properties`` key.
    """
    root: Dict[str, Any]

    model_config = ConfigDict(
        json_schema_extra={"properties": {}}
    )

# ─── Retry decorator for DNS resilience ──────────────────────────────────────
def retry_on_connection_errors(max_retries=8, retry_delay=1):
    """Retry decorator that handles DNS resolution and timeout errors"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.ConnectionError, 
                        requests.exceptions.Timeout, 
                        requests.exceptions.ReadTimeout,
                        httpx.ConnectError,
                        httpx.ConnectTimeout,
                        httpx.ReadTimeout,
                        notion_client.errors.RequestTimeoutError,
                        TimeoutError) as e:
                    retries += 1
                    
                    # Calculate backoff time with jitter to prevent synchronized retries
                    backoff_time = retry_delay * (1.5 ** retries) + (random.random() * 2)
                    
                    if retries < max_retries:
                        logger.warning(f"Connection issue: {type(e).__name__}, retrying in {backoff_time:.2f}s (attempt {retries}/{max_retries})")
                        logger.warning(f"Error details: {str(e)}")
                        time.sleep(backoff_time)  # Exponential backoff with jitter
                    else:
                        logger.error(f"Failed after {retries} retries: {str(e)}")
                        if isinstance(e, notion_client.errors.RequestTimeoutError):
                            raise
                        raise notion_client.errors.RequestTimeoutError() from e
        return wrapper
    return decorator

# Alias for backward compatibility
retry_on_dns_error = retry_on_connection_errors

# ─── Sync helpers ─────────────────────────────────────────────────────────────

@retry_on_dns_error()
def _fetch_pages_only():
    """
    Fetch every object the Search API labels 'page', then filter out
    any pages whose parent is a database (i.e., database rows).
    
    Automatically fetches all pages using pagination.
    """
    notion = get_notion_client()
    # Use a larger page size to minimize API calls
    PAGE_SIZE = 100
    
    items = collect_paginated_api(
        notion.search,
        filter={"property": "object", "value": "page"},
        page_size=PAGE_SIZE,
    )
    
    # Safety limit to prevent issues with extremely large workspaces
    if len(items) > 1000:
        logger.warning("Limiting page results to first 1000 due to size constraints")
        items = items[:1000]
        
    pages = [
        p for p in items
        if p.get("object") == "page"
        and p.get("parent", {}).get("type") != "database_id"
    ]
    # Dedupe by ID
    unique = {p["id"]: p for p in pages}
    return list(unique.values())

@retry_on_dns_error()
def _fetch_databases_only(limit: int):
    """
    Fetch every object the Search API labels 'database', dedupe by ID,
    and return only true database objects.
    """
    notion = get_notion_client()
    items = collect_paginated_api(
        notion.search,
        filter={"property": "object", "value": "database"},
        page_size=limit,
    )
    dbs = [d for d in items if d.get("object") == "database"]
    unique = {d["id"]: d for d in dbs}
    return list(unique.values())

# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get(
    "/pages",
    summary="List all Notion pages (excluding database rows)",
)
async def list_pages():
    """
    Returns every Notion page visible to your integration,
    excluding pages that belong to a database.
    
    Automatically fetches all pages in the workspace (up to 1000 max).
    """
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")

    try:
        pages = await run_in_threadpool(_fetch_pages_only)
        logger.info("Fetched %d pages (excluding database rows)", len(pages))
        # Simplify pages to only include id and title
        simplified_pages: List[Dict[str, Any]] = []
        for p in pages:
            page_id = p.get("id")
            # Extract title from properties.title.title rich text
            title = ""
            props = p.get("properties", {}) or {}
            title_prop = props.get("title", {}) or {}
            rich_texts = title_prop.get("title", []) if isinstance(title_prop, dict) else []
            if isinstance(rich_texts, list):
                title = "".join(rt.get("plain_text", "") for rt in rich_texts)
            # Extract parent_id if the page has a parent page or database
            parent = p.get("parent", {}) or {}
            parent_id = None
            # parent types: 'page_id' or 'database_id'
            p_type = parent.get("type")
            if p_type == "page_id":
                parent_id = parent.get("page_id")
            elif p_type == "database_id":
                parent_id = parent.get("database_id")
            simplified_pages.append({"id": page_id, "title": title, "parent_id": parent_id})
        return simplified_pages
    except Exception as e:
        logger.exception("Failed to fetch pages")
        raise HTTPException(status_code=500, detail=str(e))


class SearchFilter(BaseModel):
    property: Literal["object"] = Field(
        ...,
        description="Must be 'object' (Notion only supports filtering by object type here)"
    )
    value: Literal["page", "database"] = Field(
        ...,
        description="Search within either pages or databases"
    )

class SearchSort(BaseModel):
    direction: Literal["ascending", "descending"] = Field(
        ...,
        description="Sort order"
    )
    timestamp: Literal["last_edited_time", "created_time"] = Field(
        ...,
        description="Which timestamp to sort on"
    )

class SearchRequest(BaseModel):
    query: Optional[str] = Field(
        None,
        description="Full-text search string (matches only page/database titles)"
    )
    filter: Optional[SearchFilter] = Field(
        None,
        description="Filter by object type"
    )
    sort: Optional[SearchSort] = Field(
        None,
        description="Sort parameters"
    )
    start_cursor: Optional[str] = Field(
        None,
        description=(
            "Starting position for pagination. Pass a cursor value "
            "from a previous call's ``next_cursor``. Omit for the first page."
        )
    )
    page_size: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Number of results to return (1-100)."
    )

class NotionTitleSearchResult(BaseModel):
    id: str = Field(..., description="Notion page or database ID")
    title: str = Field(..., description="Page or database title")
    parent_id: Optional[str] = Field(None, description="ID of parent page or database, if any")
    object_type: str = Field(..., description="Type of object (page or database)")

@router.get(
    "/search_titles",
    summary="Search Notion pages and databases by title",
)
async def search_titles(
    query: str = Query(..., description="Search query - matches only page/database titles")
):
    """
    Search for pages and databases by title in your Notion workspace.
    
    **Important**: This endpoint ONLY searches titles, not page contents.
    
    Returns a list of all matching results with title, id, and parent_id.
    Pagination is handled internally and all results are returned.
    """
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")

    notion = Client(auth=NOTION_TOKEN, notion_version=NOTION_VERSION)
    
    # Use a larger page size to minimize API calls but not too large
    PAGE_SIZE = 100
    
    # Normalize query to lowercase for case-insensitive matching
    # (while still sending original query to API for best results)
    normalized_query = query.lower()
    
    try:
        # Set up for paginated collection
        all_results = []
        has_more = True
        start_cursor = None
        
        # Loop until we've fetched all results
        while has_more:
            params = {
                "query": query,
                "page_size": PAGE_SIZE
            }
            
            if start_cursor:
                params["start_cursor"] = start_cursor
            
            # Make API call
            result = await run_in_threadpool(notion.search, **params)
            
            # Process results
            items = result.get("results", [])
            
            # Extract simplified data from each item
            for item in items:
                object_type = item.get("object")
                item_id = item.get("id")
                
                # Extract title based on object type
                title = ""
                if object_type == "page":
                    # Extract title from properties.title.title rich text
                    props = item.get("properties", {}) or {}
                    title_prop = props.get("title", {}) or {}
                    rich_texts = title_prop.get("title", []) if isinstance(title_prop, dict) else []
                    if isinstance(rich_texts, list):
                        title = "".join(rt.get("plain_text", "") for rt in rich_texts)
                elif object_type == "database":
                    # Extract title from title rich text
                    rich_texts = item.get("title", []) or []
                    if isinstance(rich_texts, list):
                        title = "".join(rt.get("plain_text", "") for rt in rich_texts)
                
                # Verify case-insensitive match (even though Notion search should handle this,
                # we double-check in case the API behavior changes)
                if normalized_query not in title.lower():
                    logger.debug(f"Filtering out result '{title}' as it doesn't match '{normalized_query}' case-insensitively")
                    continue
                
                # Extract parent_id
                parent = item.get("parent", {}) or {}
                parent_id = None
                p_type = parent.get("type")
                if p_type == "page_id":
                    parent_id = parent.get("page_id")
                elif p_type == "database_id":
                    parent_id = parent.get("database_id")
                    
                all_results.append({
                    "id": item_id,
                    "title": title,
                    "parent_id": parent_id,
                    "object_type": object_type
                })
            
            # Check if there are more results
            has_more = result.get("has_more", False)
            if has_more:
                start_cursor = result.get("next_cursor")
            
            # Safety limit to prevent too many API calls
            if len(all_results) >= 500:
                logger.warning("Search title results truncated at 500 items")
                break
        
        # Return all collected results
        return all_results
    except Exception as e:
        logger.exception("Failed to search Notion")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/pages/{page_id}",
    summary="Retrieve a Notion page's metadata, does not return blocks",
)
async def retrieve_page(page_id: str = Path(..., description="Page ID to retrieve")):
    """
    Retrieves a Notion page by ID.
    """
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")
    notion = Client(auth=NOTION_TOKEN, notion_version=NOTION_VERSION)
    try:
        logger.info("Retrieving page %s", page_id)
        page = await run_in_threadpool(notion.pages.retrieve, page_id)
        logger.debug("Page data: %s", page)
        return page
    except Exception as e:
        logger.exception("Failed to retrieve page %s", page_id)
        raise HTTPException(status_code=500, detail=str(e))


class CreateSimplePageRequest(BaseModel):
    """Request to create a new Notion page with only a parent and title."""
    parent_page_id: str = Field(..., description="ID of the parent page")
    title: str = Field(..., description="Title of the new page")

class PageIdResponse(BaseModel):
    id: str = Field(..., description="ID of the newly created page")

@router.post(
    "/pages/simple",
    summary="Create a Notion page with only parent and title",
    response_model=PageIdResponse
)
async def create_simple_page(request: CreateSimplePageRequest):
    """
    Creates a new Notion page under an existing page, setting only the title property.
    
    Returns only the ID of the newly created page.
    """
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")
    # Build minimal payload
    payload = {
        "parent": {"page_id": request.parent_page_id},
        "properties": {
            "title": {
                "title": [
                    {"type": "text", "text": {"content": request.title}}
                ]
            }
        }
    }
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json"
    }
    # Use raw HTTP POST to avoid SDK kwargs issues
    def _do_request():
        resp = requests.post(
            "https://api.notion.com/v1/pages",
            json=payload,
            headers=headers
        )
        resp.raise_for_status()
        return resp.json()

    try:
        page = await run_in_threadpool(_do_request)
        # Return only the ID instead of the full page object
        return {"id": page["id"]}
    except Exception as e:
        logger.exception("Failed to create simple page")
        raise HTTPException(status_code=500, detail=str(e))


class SimplifiedBlockText(BaseModel):
    id: str = Field(..., description="Block ID")
    type: str = Field(..., description="Block type (paragraph, heading_1, etc.)")
    text: str = Field("", description="Plain text content of the block")
    has_children: bool = Field(False, description="Whether this block has children")
    children: Optional[List['SimplifiedBlockText']] = Field(None, description="Child blocks if any")
    equation: Optional[str] = Field(None, description="LaTeX expression for equation blocks")
    inline_equations: Optional[List[Dict[str, str]]] = Field(None, description="Inline equations in paragraph blocks")

class SimplifiedBlocksResponse(BaseModel):
    object: Literal["list"] = Field("list", description="Always 'list'")
    title: str = Field("", description="The title of the page")
    results: List[SimplifiedBlockText] = Field(..., description="List of simplified blocks")

# Need to update the reference after the class is defined
SimplifiedBlockText.update_forward_refs()


def convert_text_to_rich_text(request: 'TextBlockRequest') -> List[Dict[str, Any]]:
    """
    Convert text with markdown-style formatting to Notion's rich text format.
    
    Supported syntax:
    - Bold: **text** or __text__
    - Italic: *text* or _text_
    - Strikethrough: ~~text~~
    - Underline: <u>text</u>
    - Links: [text](url)
    - Inline code: `code` - Formatted as code with green color
    - Equations: $$equation$$ - Rendered as proper LaTeX math
    
    If you need to include special characters without triggering formatting,
    you can escape them with a backslash: \*, \_, \~, \$, \`, etc.
    """
    # Process the text to handle escaped characters
    text = request.text
    
    # Step 1: Prepare text by handling escape characters
    processed_text = ""
    i = 0
    while i < len(text):
        if i < len(text) - 1 and text[i] == '\\':
            # Handle escaped characters
            if text[i+1] in ['*', '_', '~', '$', '`', '[', ']', '(', ')', '<', '>']:
                processed_text += text[i+1]
                i += 2
                continue
        processed_text += text[i]
        i += 1
    
    text = processed_text
    
    # Step 2: First parse block-level elements like equations and code that should not be affected by other formatting
    block_parts = []
    current_pos = 0
    remaining_text = text
    
    # Find all equation and code segments first
    while current_pos < len(text):
        # Look for equations and code
        equation_match = re.search(r'\$\$(.*?)\$\$', remaining_text)
        code_match = re.search(r'`(.*?)`', remaining_text)
        link_match = re.search(r'\[(.*?)\]\((.*?)\)', remaining_text)
        
        # Find the earliest match
        eq_pos = equation_match.start() if equation_match else float('inf')
        code_pos = code_match.start() if code_match else float('inf')
        link_pos = link_match.start() if link_match else float('inf')
        
        earliest_pos = min(eq_pos, code_pos, link_pos)
        
        if earliest_pos == float('inf'):
            # No more matches, add remaining text and break
            if remaining_text:
                block_parts.append(("text", remaining_text))
            break
        
        # Add text before the match
        if earliest_pos > 0:
            block_parts.append(("text", remaining_text[:earliest_pos]))
        
        if earliest_pos == eq_pos:
            # Equation
            eq_content = equation_match.group(1)
            block_parts.append(("equation", eq_content))
            new_pos = equation_match.end()
        elif earliest_pos == code_pos:
            # Code
            code_content = code_match.group(1)
            block_parts.append(("code", code_content))
            new_pos = code_match.end()
        else:
            # Link
            link_text = link_match.group(1)
            link_url = link_match.group(2)
            block_parts.append(("link", link_text, link_url))
            new_pos = link_match.end()
        
        # Update remaining text
        remaining_text = remaining_text[new_pos:]
        current_pos += new_pos
    
    # Step 3: Process inline formatting within text blocks
    result = []
    for part in block_parts:
        part_type = part[0]
        
        if part_type == "text":
            # Process inline formatting for regular text
            content = part[1]
            
            # Process bold, italic, strikethrough, and underline patterns
            # Start with default annotations
            segments = [(content, {
                "bold": False,
                "italic": False,
                "strikethrough": False,
                "underline": False,
                "code": False,
                "color": "default"
            })]
            
            # Process bold with **...**
            new_segments = []
            for text, annotations in segments:
                parts = re.split(r'\*\*(.*?)\*\*', text)
                if len(parts) > 1:  # Found bold sections
                    for i, part_text in enumerate(parts):
                        if i % 2 == 0:  # Regular text
                            if part_text:
                                new_segments.append((part_text, annotations.copy()))
                        else:  # Bold text
                            bold_annotations = annotations.copy()
                            bold_annotations["bold"] = True
                            new_segments.append((part_text, bold_annotations))
                else:
                    new_segments.append((text, annotations))
            segments = new_segments
            
            # Process bold with __...__
            new_segments = []
            for text, annotations in segments:
                parts = re.split(r'__(.*?)__', text)
                if len(parts) > 1:  # Found bold sections
                    for i, part_text in enumerate(parts):
                        if i % 2 == 0:  # Regular text
                            if part_text:
                                new_segments.append((part_text, annotations.copy()))
                        else:  # Bold text
                            bold_annotations = annotations.copy()
                            bold_annotations["bold"] = True
                            new_segments.append((part_text, bold_annotations))
                else:
                    new_segments.append((text, annotations))
            segments = new_segments
            
            # Process italic with *...*
            new_segments = []
            for text, annotations in segments:
                # Be careful not to match ** (bold)
                parts = re.split(r'(?<!\*)\*((?!\*).*?)\*(?!\*)', text)
                if len(parts) > 1:  # Found italic sections
                    for i, part_text in enumerate(parts):
                        if i % 2 == 0:  # Regular text
                            if part_text:
                                new_segments.append((part_text, annotations.copy()))
                        else:  # Italic text
                            italic_annotations = annotations.copy()
                            italic_annotations["italic"] = True
                            new_segments.append((part_text, italic_annotations))
                else:
                    new_segments.append((text, annotations))
            segments = new_segments
            
            # Process italic with _..._
            new_segments = []
            for text, annotations in segments:
                # Be careful not to match __ (bold)
                parts = re.split(r'(?<!_)_((?!_).*?)_(?!_)', text)
                if len(parts) > 1:  # Found italic sections
                    for i, part_text in enumerate(parts):
                        if i % 2 == 0:  # Regular text
                            if part_text:
                                new_segments.append((part_text, annotations.copy()))
                        else:  # Italic text
                            italic_annotations = annotations.copy()
                            italic_annotations["italic"] = True
                            new_segments.append((part_text, italic_annotations))
                else:
                    new_segments.append((text, annotations))
            segments = new_segments
            
            # Process strikethrough with ~~...~~
            new_segments = []
            for text, annotations in segments:
                parts = re.split(r'~~(.*?)~~', text)
                if len(parts) > 1:  # Found strikethrough sections
                    for i, part_text in enumerate(parts):
                        if i % 2 == 0:  # Regular text
                            if part_text:
                                new_segments.append((part_text, annotations.copy()))
                        else:  # Strikethrough text
                            strike_annotations = annotations.copy()
                            strike_annotations["strikethrough"] = True
                            new_segments.append((part_text, strike_annotations))
                else:
                    new_segments.append((text, annotations))
            segments = new_segments
            
            # Process underline with <u>...</u>
            new_segments = []
            for text, annotations in segments:
                parts = re.split(r'<u>(.*?)</u>', text)
                if len(parts) > 1:  # Found underline sections
                    for i, part_text in enumerate(parts):
                        if i % 2 == 0:  # Regular text
                            if part_text:
                                new_segments.append((part_text, annotations.copy()))
                        else:  # Underline text
                            underline_annotations = annotations.copy()
                            underline_annotations["underline"] = True
                            new_segments.append((part_text, underline_annotations))
                else:
                    new_segments.append((text, annotations))
            segments = new_segments
            
            # Add processed segments to result
            for text, annotations in segments:
                if text:  # Skip empty segments
                    result.append({
                        "type": "text",
                        "text": {"content": text},
                        "annotations": annotations
                    })
                
        elif part_type == "equation":
            # For inline equations in rich text
            result.append({
                "type": "equation",
                "equation": {
                    "expression": part[1]
                },
                "annotations": {
                    "code": False,
                    "bold": False,
                    "italic": False,
                    "strikethrough": False,
                    "underline": False,
                    "color": "default"
                },
                "plain_text": part[1],
                "href": None
            })
            
        elif part_type == "code":
            # For inline code, use code annotation with green color
            result.append({
                "type": "text",
                "text": {"content": part[1]},
                "annotations": {
                    "code": True,
                    "bold": False,
                    "italic": False,
                    "strikethrough": False,
                    "underline": False,
                    "color": "green"
                }
            })
            
        elif part_type == "link":
            # For links
            result.append({
                "type": "text",
                "text": {
                    "content": part[1],
                    "link": {"url": part[2]}
                },
                "annotations": {
                    "code": False,
                    "bold": False,
                    "italic": False,
                    "strikethrough": False,
                    "underline": False,
                    "color": "blue"
                },
                "plain_text": part[1],
                "href": part[2]
            })
    
    # If nothing was added, use the original text
    if not result:
        result = [{
            "type": "text",
            "text": {"content": request.text}
        }]
    
    logger.debug(f"Converted text to rich text format: {json.dumps(result, indent=2)}")
    return result

@router.get(
    "/blocks/{page_id}/recursive",
    response_model=SimplifiedBlocksResponse,
)
async def get_page_blocks(page_id: str = Path(..., description="Notion page ID")):
    """
    Fetches all blocks of the given Notion page, including nested children.
    
    **Important**: This endpoint returns blocks with the following features:
    - Image URLs
    - Equation expressions
    - Automatic inline equation detection
    
    Returns a simplified response with recursive block structure and plain text content.
    Automatically handles pagination and fetches all blocks.
    """
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")

    notion = Client(auth=NOTION_TOKEN, notion_version=NOTION_VERSION)
    import traceback
    
    # First fetch the page title
    try:
        page = await run_in_threadpool(notion.pages.retrieve, page_id=page_id)
        title_prop = page.get("properties", {}).get("title", {})
        page_title = ""
        if title_prop and "title" in title_prop:
            page_title = "".join(t.get("plain_text", "") for t in title_prop["title"])
    except Exception as e:
        logger.warning(f"Could not fetch page title: {str(e)}")
        page_title = "Untitled"
    
    async def get_block_text(block):
        """Extract plain text from a block based on its type"""
        block_type = block.get("type")
        if not block_type:
            return ""
            
        # Get the content object for this block type
        content = block.get(block_type)
        if not content:
            return ""
        
        # For most block types, the text is in a "rich_text" array
        rich_text = content.get("rich_text", [])
        if rich_text:
            return "".join(rt.get("plain_text", "") for rt in rich_text)
            
        # Handle to_do blocks which might have a checked property
        if block_type == "to_do" and content.get("checked") is not None:
            prefix = "✅ " if content.get("checked") else "☐ "
            todo_text = "".join(rt.get("plain_text", "") for rt in content.get("rich_text", []))
            return prefix + todo_text
            
        # Handle other special cases
        if block_type == "image":
            captions = content.get("caption", [])
            if captions:
                # Extract caption text similar to rich_text processing
                caption_text = "".join(c.get("plain_text", "") for c in captions if isinstance(c, dict))
                return "[Image] " + caption_text
            return "[Image]"
            
        return ""
    
    async def get_blocks_recursive(block_id, max_depth=10, current_depth=0):
        """Recursively fetch blocks and their children"""
        if current_depth >= max_depth:
            logger.warning(f"Maximum recursion depth reached for block {block_id}")
            return []
            
        try:
            # Get all blocks using pagination
            all_blocks = []
            has_more = True
            start_cursor = None
            # Use the notion client from outer scope
            nonlocal notion
            
            while has_more:
                response = await run_in_threadpool(
                    notion.blocks.children.list,
                    block_id=block_id,
                    page_size=100,
                    start_cursor=start_cursor
                )
                
                all_blocks.extend(response.get("results", []))
                has_more = response.get("has_more", False)
                start_cursor = response.get("next_cursor")
                
                # Stop if we've fetched too many blocks to avoid huge responses
                if len(all_blocks) > 1000:
                    logger.warning(f"Block limit reached for {block_id}, truncating results")
                    has_more = False
            
            # Process each block
            simplified_blocks = []
            for block in all_blocks:
                block_id = block.get("id")
                block_type = block.get("type")
                has_children = block.get("has_children", False)
                text = await get_block_text(block)
                
                # Get the content object for this block type
                content = block.get(block_type, {})
                
                # Create simplified block with all necessary fields
                simplified_block = {
                    "id": block_id,
                    "type": block_type,
                    "text": text,
                    "children": []
                }
                
                # Add additional metadata for specific block types
                if block_type == "image":
                    # Handle different image types (file, external)
                    image_url = None
                    if "file" in content and "url" in content["file"]:
                        image_url = content["file"]["url"]
                        logger.debug(f"Found Notion-hosted image with URL: {image_url}")
                    elif "external" in content and "url" in content["external"]:
                        image_url = content["external"]["url"]
                        logger.debug(f"Found external image with URL: {image_url}")
                    else:
                        # Log the entire content for debugging
                        logger.warning(f"Image block missing URL. Content: {content}")
                        
                    # Always add the URL field to image blocks, even if null
                    # This ensures the field is consistently present in the response
                    simplified_block["url"] = image_url
                        
                elif block_type == "equation":
                    if "expression" in content:
                        simplified_block["equation"] = content["expression"]
                        
                # Check for inline math equations in paragraphs
                if block_type == "paragraph":
                    rich_text = content.get("rich_text", [])
                    equations = []
                    
                    for text_item in rich_text:
                        if text_item.get("type") == "equation" and "equation" in text_item:
                            equations.append({
                                "text": text_item.get("plain_text", ""),
                                "equation": text_item["equation"]["expression"]
                            })
                            
                    # Enhanced detection for inline equations - look for common math patterns in text
                    plain_text = text
                    if not equations and plain_text:
                        # Look for patterns like "E = mc^2" or similar common equation formats
                        import re
                        # This regex looks for common equation patterns
                        # It will match things like "E = mc^2", "a + b = c", "x^2 + y^2 = z^2", etc.
                        math_patterns = [
                            r'[a-zA-Z]+\s*=\s*[a-zA-Z0-9^*+-/]+',  # E = mc^2
                            r'[a-zA-Z0-9]+\^[0-9]+',                # x^2
                            r'\\[a-zA-Z]+',                         # \alpha, \beta, etc.
                            r'[a-zA-Z0-9]+_[a-zA-Z0-9]+'            # x_1, a_i, etc.
                        ]
                        
                        for pattern in math_patterns:
                            matches = re.findall(pattern, plain_text)
                            for match in matches:
                                equations.append({
                                    "text": match,
                                    "equation": match.strip()
                                })
                    
                    if equations:
                        simplified_block["inline_equations"] = equations
                
                # If block has children, recursively fetch them
                if has_children:
                    try:
                        child_blocks = await get_blocks_recursive(block_id, max_depth, current_depth + 1)
                        simplified_block["children"] = child_blocks
                    except Exception as e:
                        logger.error(f"Error fetching children for block {block_id}: {str(e)}")
                
                simplified_blocks.append(simplified_block)
            
            return simplified_blocks
        except Exception as e:
            logger.exception(f"Error fetching blocks for {block_id}: {str(e)}")
            return []
    
    try:
        # Fetch all blocks recursively
        blocks = await get_blocks_recursive(page_id)
        
        # Custom function to recursively clean null values from the response
        def clean_nulls(obj):
            if isinstance(obj, dict):
                return {k: clean_nulls(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, list):
                return [clean_nulls(i) for i in obj]
            else:
                return obj
        
        # Clean the response to remove all null fields
        cleaned_blocks = clean_nulls(blocks)
        
        # Return well-formatted response
        return {
            "object": "list",
            "title": page_title,
            "results": cleaned_blocks
        }
    except Exception as e:
        logger.error(f"Error fetching blocks for {page_id}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def get_block_text(block):
    """Extract plain text from a block based on its type"""
    block_type = block.get("type")
    if not block_type:
        return ""
            
    # Get the content object for this block type
    content = block.get(block_type)
    if not content:
        return ""
        
    # For most block types, the text is in a "rich_text" array
    rich_text = content.get("rich_text", [])
    if rich_text:
        return "".join(rt.get("plain_text", "") for rt in rich_text)
            
    # Handle to_do blocks which might have a checked property
    if block_type == "to_do" and content.get("checked") is not None:
        prefix = "✅ " if content.get("checked") else "☐ "
        todo_text = "".join(rt.get("plain_text", "") for rt in content.get("rich_text", []))
        return prefix + todo_text
            
    # Handle other special cases
    if block_type == "image":
        captions = content.get("caption", [])
        if captions:
            # Extract caption text similar to rich_text processing
            caption_text = "".join(c.get("plain_text", "") for c in captions if isinstance(c, dict))
            return "[Image] " + caption_text
        return "[Image]"
            
    return ""


async def get_block_with_children(block):
    # Extract block information
    block_id = block.get("id")
    block_type = block.get("type")
    text = await get_block_text(block)

    # Base structure for the block (without has_children field)
    result = {
        "id": block_id,
        "type": block_type,
        "text": text,
        "children": []
    }
        
    # If this block has children, recursively fetch them
    if block.get("has_children", False):
        try:
            child_blocks = await get_blocks_recursive(block_id, max_depth=10, current_depth=1)
            result["children"] = child_blocks
        except Exception as e:
            logger.error(f"Error fetching children for block {block_id}: {str(e)}")
                
    return result


async def get_blocks_recursive(block_id, max_depth=10, current_depth=0):
    """Recursively fetch blocks and their children"""
    if current_depth >= max_depth:
        logger.warning(f"Maximum recursion depth reached for block {block_id}")
        return []
            
    try:
        # Get all blocks using pagination
        all_blocks = []
        has_more = True
        start_cursor = None
        # Initialize Notion client for this function
        notion = Client(auth=NOTION_TOKEN, notion_version=NOTION_VERSION)
            
        while has_more:
            response = await run_in_threadpool(
                notion.blocks.children.list,
                block_id=block_id,
                page_size=100,
                start_cursor=start_cursor
            )
                
            all_blocks.extend(response.get("results", []))
            has_more = response.get("has_more", False)
            start_cursor = response.get("next_cursor")
                
            # Stop if we've fetched too many blocks to avoid huge responses
            if len(all_blocks) > 1000:
                logger.warning(f"Block limit reached for {block_id}, truncating results")
                has_more = False
            
        # Process each block
        simplified_blocks = []
        for block in all_blocks:
            simplified_block = await get_block_with_children(block)
            simplified_blocks.append(simplified_block)
                
        return simplified_blocks
    except Exception as e:
        logger.exception(f"Error fetching blocks for {block_id}: {str(e)}")
        return []


@router.get("/blocks/{block_id}", summary="Retrieve a block")
async def retrieve_block(block_id: str = Path(..., description="Block ID to retrieve")):
    """
    Retrieves a Notion block by ID.
    """
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")
    notion = Client(auth=NOTION_TOKEN, notion_version=NOTION_VERSION)
    try:
        logger.info("Retrieving block %s", block_id)
        block = await run_in_threadpool(notion.blocks.retrieve, block_id)
        logger.debug("Block data: %s", block)
        return block
    except Exception as e:
        logger.exception("Failed to retrieve block %s", block_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/blocks/{block_id}/children", summary="List block children")
async def list_block_children(
    block_id: str = Path(..., description="Block ID to list children of"),
    start_cursor: Optional[str] = Query(None),
    page_size: int = Query(100, ge=1, le=100)
):
    """
    Retrieves children blocks under a parent block.
    """
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")
    notion = Client(auth=NOTION_TOKEN, notion_version=NOTION_VERSION)
    try:
        logger.info("Listing children for block %s: start_cursor=%s, page_size=%d", block_id, start_cursor, page_size)
        result = await run_in_threadpool(
            notion.blocks.children.list,
            block_id=block_id,
            start_cursor=start_cursor,
            page_size=page_size
        )
        logger.info("Retrieved %d children", len(result.get("results", [])))
        logger.debug("Children data: %s", result)
        return result
    except Exception as e:
        logger.exception("Failed to list children for block %s", block_id)
        raise HTTPException(status_code=500, detail=str(e))


    # ─── Models for appending child blocks ───────────────────────────────────────

# Common block schemas: TextContent, RichTextBlock, ToDoBlock, CodeBlock, External
class TextContent(BaseModel):
    type: Literal["text", "equation"]
    text: Optional[Dict[str, Any]] = None  # Changed from Dict[str, str] to Dict[str, Any] to support link objects
    equation: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, Any]] = None
    plain_text: Optional[str] = None
    href: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_content_type(self):
        if self.type == "text" and self.text is None:
            raise ValueError("text field is required when type is 'text'")
        if self.type == "equation" and self.equation is None:
            raise ValueError("equation field is required when type is 'equation'")
        return self

class RichTextBlock(BaseModel):
    """Common schema for blocks that accept rich_text."""
    rich_text: List[TextContent]

class ToDoBlock(BaseModel):
    """Schema for updating a to_do block."""
    rich_text: List[TextContent]
    checked: bool

class CodeBlock(BaseModel):
    """Schema for updating a code block."""
    rich_text: List[TextContent]
    language: str

class External(BaseModel):
    """Schema for external resource URLs."""
    url: str

class ParagraphBlock(BaseModel):
    object: Literal["block"] = "block"
    type: Literal["paragraph"]
    paragraph: RichTextBlock

class Heading1Block(BaseModel):
    object: Literal["block"] = "block"
    type: Literal["heading_1"]
    heading_1: RichTextBlock

class Heading2Block(BaseModel):
    object: Literal["block"] = "block"
    type: Literal["heading_2"]
    heading_2: RichTextBlock

class Heading3Block(BaseModel):
    object: Literal["block"] = "block"
    type: Literal["heading_3"]
    heading_3: RichTextBlock

class BulletedListItemBlock(BaseModel):
    object: Literal["block"] = "block"
    type: Literal["bulleted_list_item"]
    bulleted_list_item: RichTextBlock

class NumberedListItemBlock(BaseModel):
    object: Literal["block"] = "block"
    type: Literal["numbered_list_item"]
    numbered_list_item: RichTextBlock

class ToDoBlockObject(BaseModel):
    object: Literal["block"] = "block"
    type: Literal["to_do"]
    to_do: ToDoBlock

class ToggleBlock(BaseModel):
    object: Literal["block"] = "block"
    type: Literal["toggle"]
    toggle: RichTextBlock

class QuoteBlock(BaseModel):
    object: Literal["block"] = "block"
    type: Literal["quote"]
    quote: RichTextBlock

class CalloutBlock(BaseModel):
    object: Literal["block"] = "block"
    type: Literal["callout"]
    callout: RichTextBlock

class CodeBlockObject(BaseModel):
    object: Literal["block"] = "block"
    type: Literal["code"]
    code: CodeBlock
    
    @classmethod
    def create(cls, content: str, language: str = "plain text") -> "CodeBlockObject":
        """Create a code block with the rich_text format that Notion expects."""
        return cls(
            object="block",
            type="code",
            code={
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": content
                        }
                    }
                ],
                "language": language
            }
        )

class ImageBlockObject(BaseModel):
    object: Literal["block"] = "block"
    type: Literal["image"]
    image: Dict[str, Any]
    
    @classmethod
    def create_from_url(cls, url: str) -> "ImageBlockObject":
        """Create an image block with the external URL that Notion expects.
        
        Note: Notion has a 2000 character limit for external URLs.
        Data URLs (base64-encoded images) will typically exceed this limit.
        """
        # Check if URL is too long for Notion (2000 char limit)
        if len(url) > 2000:
            if url.startswith('data:'):
                raise ValueError(
                    "Base64-encoded image URLs are too large for Notion's API. "
                    "The URL length is " + str(len(url)) + " but Notion's limit is 2000 characters. "
                    "Please use a hosted image URL instead."
                )
            else:
                raise ValueError(
                    "URL exceeds Notion's 2000 character limit. "
                    "Please use a shorter URL or a hosted image."
                )
                
        return cls(
            object="block",
            type="image",
            image={
                "type": "external",
                "external": {
                    "url": url
                }
            }
        )

class MathBlockObject(BaseModel):
    """A block-equation in Notion."""
    object: Literal["block"] = "block"
    type: Literal["equation"]
    equation: Dict[str, str]  # Uses expression property

    @classmethod
    def create(cls, content: str) -> "MathBlockObject":
        """Create a block equation with the expression format that Notion expects."""
        return cls(
            object="block",
            type="equation",
            equation={
                "expression": content
            }
        )

# AppendBlockChildrenRequest removed as requested

# POST /notion/blocks/{block_id}/children endpoint removed as requested


# UpdateBlockRequest removed as requested


# PATCH /notion/blocks/{block_id} endpoint removed as requested


@router.delete("/blocks/{block_id}", summary="Delete (archive) a block")
async def delete_block(block_id: str = Path(..., description="Block ID to delete")):
    """
    Archives (deletes) a block.
    """
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")
    notion = Client(auth=NOTION_TOKEN, notion_version=NOTION_VERSION)
    try:
        logger.info("Archiving block %s", block_id)
        result = await run_in_threadpool(notion.blocks.delete, block_id=block_id)
        logger.info("Block %s archived", block_id)
        return result
    except Exception as e:
        logger.exception("Failed to delete block %s", block_id)
        raise HTTPException(status_code=500, detail=str(e))

# ─── Simplified Block Append Endpoints ────────────────────────────────────────

class TextBlockRequest(BaseModel):
    """Text content for blocks with support for rich markdown-style formatting."""
    text: str = Field(..., description="Text content with markdown formatting support: **bold**, *italic*, ~~strikethrough~~, <u>underline</u>, [links](url), `code`, and $$equations$$") 

class InlineContent(BaseModel):
    """Represents an inline content element (equation or code)."""
    type: Literal["equation", "code"] = Field(..., description="Type of inline content")
    content: str = Field(..., description="Content of the equation or code")
    start: int = Field(..., description="Start position in the text")
    end: int = Field(..., description="End position in the text")

class RichParagraphRequest(BaseModel):
    """Request model for paragraphs with inline equations and code snippets."""
    text: str = Field(..., description="The full text content of the paragraph")
    inline_content: List[InlineContent] = Field(
        default_factory=list,
        description="List of inline equations or code snippets to be inserted into the text"
    )
    auto_detect: bool = Field(
        default=True,
        description="Whether to automatically detect equations and code in the text. If inline_content is provided, this is ignored."
    )

class CodeBlockRequest(BaseModel):
    """Simplified request model for code blocks."""
    code: str = Field(..., description="Code content")
    language: str = Field("plain text", description="Programming language for syntax highlighting")

class EquationBlockRequest(BaseModel):
    """Simplified request model for equation blocks."""
    equation: str = Field(..., description="LaTeX equation content")

class ImageBlockRequest(BaseModel):
    """Simplified request model for image blocks."""
    url: str = Field(..., description="URL of the image")

@router.post(
    "/blocks/{block_id}/paragraph",
    summary="Append a paragraph block",
    description="Simplified endpoint to append a paragraph block with support for equations (use double dollar signs) and inline code (use single backticks). When called with a page ID, the block will be added at the end of the page.",
    operation_id="append_paragraph",
    response_model=FreeFormModel,
)
async def append_paragraph(
    block_id: str = Path(..., description="Parent block or page ID. If a page ID is provided, the block will be appended to the end of the page."),
    request: TextBlockRequest = Body(...),
):
    """Append a paragraph block with the given text. Supports markdown formatting: **bold**, *italic*, ~~strikethrough~~, <u>underline</u>, [links](url), `code`, and $$equations$$. When called with a page ID, the block will be added at the end of the page."""
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")
    
    # Use the shared convert_text_to_rich_text function instead of duplicating code
    rich_text = convert_text_to_rich_text(request)
    
    # Create a paragraph block with the rich text
    paragraph_block = {
        "type": "paragraph",
        "paragraph": {
            "rich_text": rich_text,
            "color": "default"
        }
    }
    
    # Prepare API request
    payload = {
        "children": [paragraph_block]
    }
    
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json"
    }
    
    def _do_request():
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        logger.info(f"Appending paragraph to {block_id} with formatting")
        resp = requests.patch(url, json=payload, headers=headers)
        if not resp.ok:
            logger.error(f"Error response from Notion API: {resp.text}")
        resp.raise_for_status()
        return resp.json()
    
    try:
        result = await run_in_threadpool(_do_request)
        # Return the first block from the children array - this is our newly created paragraph
        if "results" in result and len(result["results"]) > 0:
            return result["results"][0]
        return result
    except Exception as e:
        logger.exception(f"Failed to append paragraph to {block_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/blocks/{block_id}/heading_1",
    summary="Append a heading 1 block",
    description="Simplified endpoint to append a heading 1 block with support for equations (use double dollar signs) and inline code (use single backticks). When called with a page ID, the block will be added at the end of the page.",
    operation_id="append_heading_1",
    response_model=FreeFormModel,
)
async def append_heading_1(
    block_id: str = Path(..., description="Parent block or page ID. If a page ID is provided, the block will be appended to the end of the page."),
    request: TextBlockRequest = Body(...),
):
    """Append a heading 1 block with the given text. Supports markdown formatting: **bold**, *italic*, ~~strikethrough~~, <u>underline</u>, [links](url), `code`, and $$equations$$. When called with a page ID, the block will be added at the end of the page."""
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")
    
    # Convert text with possible equations and code to rich text
    rich_text = convert_text_to_rich_text(request)
    
    # Create the appropriate block structure
    heading_block = Heading1Block(
        type="heading_1",
        heading_1=RichTextBlock(
            rich_text=rich_text
        )
    )
    
    payload = {
        "children": [heading_block.model_dump(exclude_none=True)]
    }
    
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    
    def _do_request():
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        logger.info(f"Appending heading 1 to {block_id}")
        resp = requests.patch(url, json=payload, headers=headers)
        if not resp.ok:
            logger.error(f"Error response body: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    try:
        result = await run_in_threadpool(_do_request)
        return result
    except Exception as e:
        logger.exception(f"Failed to append heading 1 to {block_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/blocks/{block_id}/heading_2",
    summary="Append a heading 2 block",
    description="Simplified endpoint to append a heading 2 block with support for equations (use double dollar signs) and inline code (use single backticks). When called with a page ID, the block will be added at the end of the page.",
    operation_id="append_heading_2",
    response_model=FreeFormModel,
)
async def append_heading_2(
    block_id: str = Path(..., description="Parent block or page ID. If a page ID is provided, the block will be appended to the end of the page."),
    request: TextBlockRequest = Body(...),
):
    """Append a heading 2 block with the given text. Supports markdown formatting: **bold**, *italic*, ~~strikethrough~~, <u>underline</u>, [links](url), `code`, and $$equations$$. When called with a page ID, the block will be added at the end of the page."""
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")
    
    # Convert text with possible equations and code to rich text
    rich_text = convert_text_to_rich_text(request)
    
    # Create the appropriate block structure
    heading_block = Heading2Block(
        type="heading_2",
        heading_2=RichTextBlock(
            rich_text=rich_text
        )
    )
    
    payload = {
        "children": [heading_block.model_dump(exclude_none=True)]
    }
    
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    
    def _do_request():
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        logger.info(f"Appending heading 2 to {block_id}")
        resp = requests.patch(url, json=payload, headers=headers)
        if not resp.ok:
            logger.error(f"Error response body: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    try:
        result = await run_in_threadpool(_do_request)
        return result
    except Exception as e:
        logger.exception(f"Failed to append heading 2 to {block_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/blocks/{block_id}/heading_3",
    summary="Append a heading 3 block",
    description="Simplified endpoint to append a heading 3 block with support for equations (use double dollar signs) and inline code (use single backticks). When called with a page ID, the block will be added at the end of the page.",
    operation_id="append_heading_3",
    response_model=FreeFormModel,
)
async def append_heading_3(
    block_id: str = Path(..., description="Parent block or page ID. If a page ID is provided, the block will be appended to the end of the page."),
    request: TextBlockRequest = Body(...),
):
    """Append a heading 3 block with the given text. Supports markdown formatting: **bold**, *italic*, ~~strikethrough~~, <u>underline</u>, [links](url), `code`, and $$equations$$. When called with a page ID, the block will be added at the end of the page."""
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")
    
    # Convert text with possible equations and code to rich text
    rich_text = convert_text_to_rich_text(request)
    
    # Create the appropriate block structure
    heading_block = Heading3Block(
        type="heading_3",
        heading_3=RichTextBlock(
            rich_text=rich_text
        )
    )
    
    payload = {
        "children": [heading_block.model_dump(exclude_none=True)]
    }
    
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    
    def _do_request():
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        logger.info(f"Appending heading 3 to {block_id}")
        resp = requests.patch(url, json=payload, headers=headers)
        if not resp.ok:
            logger.error(f"Error response body: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    try:
        result = await run_in_threadpool(_do_request)
        return result
    except Exception as e:
        logger.exception(f"Failed to append heading 3 to {block_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/blocks/{block_id}/bulleted_list_item",
    summary="Append a bulleted list item block",
    description="Simplified endpoint to append a bulleted list item block with support for equations (use double dollar signs) and inline code (use single backticks). When called with a page ID, the block will be added at the end of the page.",
    operation_id="append_bulleted_list_item",
    response_model=FreeFormModel,
)
async def append_bulleted_list_item(
    block_id: str = Path(..., description="Parent block or page ID. If a page ID is provided, the block will be appended to the end of the page."),
    request: TextBlockRequest = Body(...),
):
    """Append a bulleted list item block with the given text. Supports markdown formatting: **bold**, *italic*, ~~strikethrough~~, <u>underline</u>, [links](url), `code`, and $$equations$$. When called with a page ID, the block will be added at the end of the page."""
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")
    
    # Convert text with possible equations and code to rich text
    rich_text = convert_text_to_rich_text(request)
    
    # Create the appropriate block structure
    list_item_block = BulletedListItemBlock(
        type="bulleted_list_item",
        bulleted_list_item=RichTextBlock(
            rich_text=rich_text
        )
    )
    
    payload = {
        "children": [list_item_block.model_dump(exclude_none=True)]
    }
    
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    
    def _do_request():
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        logger.info(f"Appending bulleted list item to {block_id}")
        resp = requests.patch(url, json=payload, headers=headers)
        if not resp.ok:
            logger.error(f"Error response body: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    try:
        result = await run_in_threadpool(_do_request)
        return result
    except Exception as e:
        logger.exception(f"Failed to append bulleted list item to {block_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/blocks/{block_id}/numbered_list_item",
    summary="Append a numbered list item block",
    description="Simplified endpoint to append a numbered list item block with support for equations (use double dollar signs) and inline code (use single backticks). When called with a page ID, the block will be added at the end of the page.",
    operation_id="append_numbered_list_item",
    response_model=FreeFormModel,
)
async def append_numbered_list_item(
    block_id: str = Path(..., description="Parent block or page ID. If a page ID is provided, the block will be appended to the end of the page."),
    request: TextBlockRequest = Body(...),
):
    """Append a numbered list item block with the given text. Supports markdown formatting: **bold**, *italic*, ~~strikethrough~~, <u>underline</u>, [links](url), `code`, and $$equations$$. When called with a page ID, the block will be added at the end of the page."""
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")
    
    # Convert text with possible equations and code to rich text
    rich_text = convert_text_to_rich_text(request)
    
    # Create the appropriate block structure
    list_block = NumberedListItemBlock(
        type="numbered_list_item",
        numbered_list_item=RichTextBlock(
            rich_text=rich_text
        )
    )
    
    payload = {
        "children": [list_block.model_dump(exclude_none=True)]
    }
    
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    
    def _do_request():
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        logger.info(f"Appending numbered list item to {block_id}")
        resp = requests.patch(url, json=payload, headers=headers)
        if not resp.ok:
            logger.error(f"Error response body: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    try:
        result = await run_in_threadpool(_do_request)
        return result
    except Exception as e:
        logger.exception(f"Failed to append numbered list item to {block_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/blocks/{block_id}/code",
    summary="Append a code block",
    description="Simplified endpoint to append a code block to a parent block or page. When called with a page ID, the block will be added at the end of the page.",
    operation_id="append_code_block",
    response_model=FreeFormModel,
)
async def append_code_block(
    block_id: str = Path(..., description="Parent block or page ID. If a page ID is provided, the block will be appended to the end of the page."),
    request: CodeBlockRequest = Body(...),
):
    """Append a code block with the given code and language. When called with a page ID, the block will be added at the end of the page."""
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")
    
    code_block = CodeBlockObject.create(request.code, request.language)
    
    payload = {
        "children": [code_block.model_dump(exclude_none=True)]
    }
    
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    
    def _do_request():
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        logger.info(f"Appending code block to {block_id}")
        resp = requests.patch(url, json=payload, headers=headers)
        if not resp.ok:
            logger.error(f"Error response body: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    try:
        result = await run_in_threadpool(_do_request)
        return result
    except Exception as e:
        logger.exception(f"Failed to append code block to {block_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/blocks/{block_id}/equation",
    summary="Append an equation block",
    description="Simplified endpoint to append a LaTeX equation block to a parent block or page. When called with a page ID, the block will be added at the end of the page.",
    operation_id="append_equation",
    response_model=FreeFormModel,
)
async def append_equation(
    block_id: str = Path(..., description="Parent block or page ID. If a page ID is provided, the block will be appended to the end of the page."),
    request: EquationBlockRequest = Body(...),
):
    """Append an equation block with the given LaTeX content. When called with a page ID, the block will be added at the end of the page."""
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")
    
    equation_block = MathBlockObject.create(request.equation)
    
    payload = {
        "children": [equation_block.model_dump(exclude_none=True)]
    }
    
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    
    def _do_request():
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        logger.info(f"Appending equation block to {block_id}")
        resp = requests.patch(url, json=payload, headers=headers)
        if not resp.ok:
            logger.error(f"Error response body: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    try:
        result = await run_in_threadpool(_do_request)
        return result
    except Exception as e:
        logger.exception(f"Failed to append equation block to {block_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/blocks/{block_id}/image",
    summary="Append an image block",
    description="Simplified endpoint to append an image block to a parent block or page. When called with a page ID, the block will be added at the end of the page.",
    operation_id="append_image",
    response_model=FreeFormModel,
)
async def append_image(
    block_id: str = Path(..., description="Parent block or page ID. If a page ID is provided, the block will be appended to the end of the page."),
    request: ImageBlockRequest = Body(...),
):
    """Append an image block with the given URL. When called with a page ID, the block will be added at the end of the page."""
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")
    
    # Check if this is a data URL and potentially warn about size limits first
    if request.url.startswith('data:'):
        url_length = len(request.url)
        if url_length > 2000:
            error_msg = f"Base64-encoded image URLs are too large for Notion's API. The URL length is {url_length} characters, but Notion's limit is 2000. Please use a hosted image URL instead."
            logger.warning(error_msg)
            # Return a 400 response instead of raising an exception
            return {
                "status": "error",
                "error": "validation_error",
                "message": error_msg
            }
    
    try:
        # Create the image block
        image_block = ImageBlockObject.create_from_url(request.url)
        
        payload = {
            "children": [image_block.model_dump(exclude_none=True)]
        }
        
        headers = {
            "Authorization": f"Bearer {NOTION_TOKEN}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }
        
        def _do_request():
            url = f"https://api.notion.com/v1/blocks/{block_id}/children"
            logger.info(f"Appending image block to {block_id}")
            resp = requests.patch(url, json=payload, headers=headers)
            if not resp.ok:
                logger.error(f"Error response body: {resp.text}")
            resp.raise_for_status()
            return resp.json()

        result = await run_in_threadpool(_do_request)
        return result
    except ValueError as e:
        # Handle validation errors from create_from_url
        error_msg = str(e)
        logger.warning(f"Validation error: {error_msg}")
        # Return a 400 response instead of raising an exception
        return {
            "status": "error",
            "error": "validation_error",
            "message": error_msg
        }
    except Exception as e:
        logger.exception(f"Failed to append image block to {block_id}")
        # Return a 500 response instead of raising an exception
        return {
            "status": "error",
            "error": "server_error",
            "message": str(e)
        }

# ─── Update Existing Blocks ───────────────────────────────────────────────────

@router.patch(
    "/blocks/{block_id}",
    summary="Update an existing block",
    description="Endpoint to update an existing block's content with support for rich text formatting. Works with all block types: paragraph, headings, list items, code, equation.",
    operation_id="update_block",
    response_model=FreeFormModel,
)
async def update_block(
    block_id: str = Path(..., description="The ID of the block to update"),
    request: TextBlockRequest = Body(...),
):
    """
    Update an existing block with new content. Supports markdown formatting: 
    **bold**, *italic*, ~~strikethrough~~, <u>underline</u>, [links](url), `code`, and $$equations$$.
    
    The endpoint automatically detects the block type and updates it accordingly.
    Supported block types:
    - paragraph
    - heading_1, heading_2, heading_3
    - bulleted_list_item
    - numbered_list_item
    - code (content only, not language)
    - equation (content only)
    """
    if not NOTION_TOKEN:
        raise HTTPException(status_code=503, detail="Notion token not configured")
    
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json"
    }
    
    # First, get the current block to determine its type
    def _get_block():
        url = f"https://api.notion.com/v1/blocks/{block_id}"
        logger.info(f"Getting block {block_id} to determine its type")
        resp = requests.get(url, headers=headers)
        if not resp.ok:
            logger.error(f"Error response from Notion API: {resp.text}")
            resp.raise_for_status()
        return resp.json()
    
    try:
        # Get the block to determine its type
        block_data = await run_in_threadpool(_get_block)
        block_type = block_data.get("type")
        
        if not block_type:
            raise HTTPException(status_code=404, detail=f"Block type not found for block {block_id}")
        
        # Convert text to rich text format
        rich_text = convert_text_to_rich_text(request)
        
        # Create update payload based on block type
        update_payload = {
            "type": block_type,
            block_type: {}
        }
        
        # Handle different block types
        if block_type in ["paragraph", "bulleted_list_item", "numbered_list_item"]:
            update_payload[block_type] = {
                "rich_text": rich_text,
                "color": "default"
            }
        elif block_type in ["heading_1", "heading_2", "heading_3"]:
            # Headings require RichTextBlock format
            update_payload[block_type] = {
                "rich_text": rich_text,
                "is_toggleable": False,
                "color": "default"
            }
        elif block_type == "code":
            # Get the current language from the block
            current_language = block_data.get("code", {}).get("language", "plain text")
            update_payload["code"] = {
                "rich_text": rich_text,
                "language": current_language
            }
        elif block_type == "equation":
            # For equation blocks, we use the text directly as the expression
            plain_text = request.text.strip("$").strip()
            update_payload["equation"] = {
                "expression": plain_text
            }
        else:
            # If it's an unsupported block type, return an error
            return {
                "status": "error",
                "error": "unsupported_block_type",
                "message": f"Block type '{block_type}' cannot be updated with this endpoint",
                "supported_types": ["paragraph", "heading_1", "heading_2", "heading_3", 
                                     "bulleted_list_item", "numbered_list_item", "code", "equation"]
            }
        
        # Update the block
        def _update_block():
            url = f"https://api.notion.com/v1/blocks/{block_id}"
            logger.info(f"Updating {block_type} block {block_id}")
            resp = requests.patch(url, json=update_payload, headers=headers)
            if not resp.ok:
                logger.error(f"Error response from Notion API: {resp.text}")
            resp.raise_for_status()
            return resp.json()
        
        result = await run_in_threadpool(_update_block)
        return result
    
    except Exception as e:
        logger.exception(f"Failed to update block {block_id}")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Helper function to reduce code duplication ────────────────────────────────

@retry_on_dns_error(max_retries=5, retry_delay=1)
def _append_block_to_parent(block_id: str, block: Any):
    """Helper function to append a block to a parent block or page. When the block_id is a page ID, the block will be added at the end of the page."""
    payload = {
        "children": [block.model_dump(exclude_none=True)]
    }
    
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    
    url = f"https://api.notion.com/v1/blocks/{block_id}/children"
    resp = requests.patch(url, json=payload, headers=headers)
    if not resp.ok:
        logger.error(f"Error response body: {resp.text}")
    resp.raise_for_status()
    return resp.json()

# ASGI app for standalone service
app = FastAPI()
app.include_router(router)