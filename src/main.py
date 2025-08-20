from fastapi import FastAPI
from typing import Optional

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello from ArXiv RAG Assistant!"}


@app.get("/status")
async def status():
    return {"status": "OK", "version":"1.0.0"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}


@app.get("/papers/{paper_id}")
async def get_paper_id(paper_id: str):
    return {"paper_id": f"{paper_id}", "title": f"Paper {paper_id}"}

@app.get("/search/")
async def search_items(query: str, limit: int=10, offset: Optional[int]=0):
    return {"query": query, "limit": limit, "offset": offset}


@app.get("/search")
async def search_papers(q: str, max_results: int=5):
    return {"query": q, "max_results": max_results, "results": ["Paper A", "Paper B"]}

@app.get("/report")
async def read_report(id: str):
    return {"report_id": id}


@app.get("/users/{user_id}")
async def user_profile(user_id: int, role: str = "guest"):
    return {"User ID": user_id, "Role": role }
