"""Walter's web app: a small JSON API plus the static frontend.

Run from the repository root:
    uvicorn app.server:app --port 8000
then open http://localhost:8000
"""
import os
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .walter_service import Walter

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

app = FastAPI(title='Walter', description='A small GPT that invents names, and a window into how it works.')
walter = Walter()


class GenerateRequest(BaseModel):
    n: int = Field(10, ge=1, le=50)
    temperature: float = Field(1.0, ge=0.1, le=3.0)
    prefix: str = ''
    seed: Optional[int] = Field(None, ge=0)


class NextRequest(BaseModel):
    prefix: str = ''
    ablate: List[Tuple[int, int]] = []


class ScoreRequest(BaseModel):
    name: str


class InspectRequest(BaseModel):
    name: str = ''
    ablate: List[Tuple[int, int]] = []


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get('/api/info')
def info():
    return walter.info()


@app.post('/api/generate')
def generate(req: GenerateRequest):
    return {'names': call(walter.generate, req.n, req.temperature, req.prefix, req.seed)}


@app.post('/api/next')
def next_letter(req: NextRequest):
    return call(walter.next_letter, req.prefix, req.ablate)


@app.post('/api/score')
def score(req: ScoreRequest):
    return call(walter.score, req.name)


@app.post('/api/inspect')
def inspect(req: InspectRequest):
    return call(walter.inspect, req.name, req.ablate)


@app.get('/')
def index():
    return FileResponse(os.path.join(STATIC, 'index.html'))


app.mount('/static', StaticFiles(directory=STATIC), name='static')
