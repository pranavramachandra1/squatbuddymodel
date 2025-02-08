from app import create_app
from dotenv import load_dotenv
import os

# load environment variables:
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

app = create_app()
