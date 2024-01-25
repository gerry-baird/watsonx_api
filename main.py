from fastapi import FastAPI, Header
from contextlib import asynccontextmanager
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
import os

from routers.generate import router as generate_router, init

load_dotenv()
security = HTTPBasic()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run the preset prompts
    await init()

    yield
    # Clean up the ML models and release the resources


app = FastAPI(lifespan=lifespan)
app.include_router(generate_router)

