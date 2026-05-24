import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from app.logger import logger
from app.schemas import SearchRequest, SearchResponse, GenerateRequeest, GenerateResponse
from app.rag import initialize_rag_from_docs, search_documentation, add_document_to_index
from app.agents import generate_and_validate_documentation
from app.storage import save_document


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Инициализация RAG из docs/')

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, initialize_rag_from_docs)
    logger.info('Сервис готов к работе')
    yield


app = FastAPI(title='AI Docs Assistant', lifespan=lifespan)


@app.get('/health')
def health_check():
    return {'status': 'ok'}


@app.post('/search', response_model=SearchResponse)
def search_docs(request: SearchRequest):
    """
    Выполняет семантический поиск в базе документации.
    """
    result = search_documentation(request.query)

    if result:
        return SearchResponse(found=True, content=result)
    else:
        return SearchResponse(
            found=False,
            message='Документация не найдена. Используйте /generate для создания новой.'
        )
    

@app.post('/generate', response_model=GenerateResponse)
def generate_docs(request: GenerateRequeest):
    """
    Генерирует новую документацию и сохраняет её в docs/.
    """
    if search_documentation(request.query, similarity_threshold=0.75):
        return GenerateResponse(
            success=False,
            message='Документ уже существует. Используйте /search.'
        )
    try:
        content = generate_and_validate_documentation(request.query)
        if not content:
            logger.error(f'Агенты не смогли сгенерировать валидный формат для запроса: {request.query}')
            return GenerateResponse(
                success=False,
                message='Не удалось сгенерировать документ в строгом соответствии с форматом. Попробуйте повторить или изменить запрос.'
            )
        file_path = save_document(content, request.query)
        add_document_to_index(file_path)
        return GenerateResponse(
            success=True,
            message='Документ успешно создан и сохранён.',
            content=content,
            file_path=file_path
        )
    except Exception as e:
        logger.error(f'Ошибка генерации документа: {e}', exc_info=True)
        return GenerateResponse(
            success=False,
            message=f'Ошибка генерации: {str(e)}'
        )