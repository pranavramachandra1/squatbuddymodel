from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router  # Import the consolidated routes

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="SquatBuddyPipeline")

    # CORS Middleware (Adjust origins if needed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],  # Change this for production security
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include all routes from routes.py
    app.include_router(router)

    return app
