from fastapi import APIRouter

router = APIRouter()

@router.get("/lookup/{query}")
async def lookup_character(query: str):
    """
    Lookup a Hán/Nôm character in the Thieu Chuu dictionary.
    """
    return {"character": query, "reading": "Hán-Việt", "meaning": "Example Meaning"}
