from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
import uvicorn
from services.anki_service import router as anki_router
from services.notion_service import router as notion_router
from services.glue_service import router as glue_router
from common.config import settings

# Create FastAPI app
app = FastAPI(
    title="Study Tools API",
    description="API for integrating Anki and Notion study tools",
    version="1.0.0"
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
app.include_router(anki_router)
app.include_router(notion_router)
app.include_router(glue_router)

def setup_ngrok():
    """Setup ngrok tunnel if auth token is provided."""
    if settings.NGROK_AUTHTOKEN:
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
        return public_url
    return None

if __name__ == "__main__":
    # Setup ngrok if configured
    ngrok_url = setup_ngrok()
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=True
    ) 