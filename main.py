from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
import uvicorn
import json
import os
from common.config import settings
from services import anki_service, notion_service, glue_service
from fastapi.openapi.utils import get_openapi

def setup_ngrok():
    """Setup ngrok tunnel if auth token is provided."""
    if settings.NGROK_AUTHTOKEN and not os.environ.get("RELOADER_PROCESS"):
        try:
            ngrok.set_auth_token(settings.NGROK_AUTHTOKEN)
            domain = settings.NGROK_DOMAIN
            if domain:
                # Use custom domain if provided
                public_url = ngrok.connect(
                    settings.API_PORT,
                    domain=domain
                ).public_url
            else:
                # Use random domain
                public_url = ngrok.connect(settings.API_PORT).public_url
            print(f"🌐 Ngrok tunnel established at: {public_url}")
            
            # Write ngrok URL to file
            os.makedirs(".well-known", exist_ok=True)
            urls = {
                "base_url": public_url,
                "openapi": f"{public_url}/openapi.json",
                "swagger": f"{public_url}/docs",
                "redoc": f"{public_url}/redoc",
                "well_known": f"{public_url}/.well-known/urls.json"
            }
            with open(".well-known/urls.json", "w") as f:
                json.dump(urls, f, indent=2)
            print(f"📝 URLs written to .well-known/urls.json")
            return public_url
        except Exception as e:
            print(f"⚠️  Failed to setup ngrok: {str(e)}")
            print("⚠️  Running without ngrok tunnel")
    return None

# Setup ngrok first
ngrok_url = setup_ngrok()
print(f"DEBUG: Ngrok URL: {ngrok_url}")

# Create FastAPI app with the appropriate server URL
app = FastAPI(
    title="Study Tools API",
    description="API for integrating Anki, Notion, and other study tools",
    version="v1.0.0",
    openapi_version="3.1.0",
    servers=[
        {"url": ngrok_url if ngrok_url else "https://localhost:8000"}
    ]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(anki_service.router, prefix="/anki", tags=["Anki"])
app.include_router(notion_service.router, prefix="/notion", tags=["Notion"])
app.include_router(glue_service.router, prefix="/glue", tags=["Glue"])

# Custom OpenAPI schema to inject ngrok URL
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    # Dynamically read ngrok URL
    current_ngrok_url = "https://localhost:8000"  # default
    try:
        with open(".well-known/urls.json", "r") as f:
            urls_data = json.load(f)
            current_ngrok_url = urls_data.get("base_url", "https://localhost:8000")
    except:
        pass
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Force the servers field to be set correctly
    openapi_schema["servers"] = [
        {"url": current_ngrok_url}
    ]
    app.openapi_schema = openapi_schema
    print(f"DEBUG: Generated OpenAPI schema with servers: {openapi_schema['servers']}")
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    # Set environment variable to prevent multiple ngrok tunnels
    os.environ["RELOADER_PROCESS"] = "1"
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=True
    ) 