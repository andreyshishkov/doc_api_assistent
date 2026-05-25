import httpx

from app.logger import logger
from app.settings import settings
from app.rag import search_documentation


async def check_qdrant():
    """Проверяет доступность Qdrant через REST API."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f'{settings.qdrant_url}/collections')
            return resp.status_code == 200
    except Exception as e:
        logger.error(f'Qdrant недоступен: {e}')
        return False
