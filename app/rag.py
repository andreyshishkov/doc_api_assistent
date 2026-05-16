import uuid
import hashlib
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore


from app.settings import settings
from app.logger import logger


collection_name = settings.QDRANT_COLLECTION_NAME
embedding_model_name = settings.EMBEDDING_MODEL_NAME
vector_size = settings.VECTOR_SIZE

client = QdrantClient(settings.QDRANT_HOST, port=settings.QDRANT_PORT)


if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)

embeddings = OllamaEmbeddings(model=embedding_model_name)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
    distance=Distance.COSINE,
)


def initialize_rag_from_docs() -> None:
    """Загружает все .md-файлы из директории docs/ в векторную базу при старте сервиса."""
    docs_dir = Path('docs')
    if not docs_dir.exists():
        logger.warning('Директория docs/ не найдена')
        return
    
    documents = []
    for file_path in docs_dir.glob('*.md'):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    documents.append(
                        Document(page_content=content, metadata={'source': str(file_path)})
                    )
        except Exception as e:
            logger.error(f'Ошибка чтения файла {file_path}: {e}')

        if documents:
            vector_store.add_documents(documents)
            logger.info(f'Загружено {len(documents)} документов в RAG-хранилище')
        else:
            logger.warning('В директории docs/ не найдено .md-файлов')


def add_document_to_index(file_path: str) -> bool:
    """
    Инкрементально добавляет один документ в векторную БД.
    Возвращает True при успехе.
    """
    try:
        content = Path(file_path).read_text(encoding="utf-8").strip()
        if not content:
            return False
        
        doc = Document(
            page_content=content,
            metadata={'source': str(file_path)},
        )
        hash_hex = hashlib.md5(str(file_path).encode()).hexdigest()
        doc_id = str(uuid.UUID(hash_hex[:32]))
        vector_store.add_documents([doc], ids=[doc_id])
        logger.info(f'Документ добавлен в индекс: {file_path}')
        return True
    except Exception as e:
        logger.error(f'Ошибка при добавлении документа {file_path}: {e}')
        return False


def search_documentation(query: str, k: int = 1, similarity_threshold: float = 0.2) -> str | None:
    """Выполняет семантический поиск по документации API."""
    try:
        logger.info(f'Семантический поиск: {query!r}')
        results = vector_store.similarity_search_with_score(
            query=query,
            k=k,
            score_threshold=similarity_threshold,
        )
        if results:
            doc, score = results[0]
            logger.info(f'Найден релевантный документ ({doc.metadata["source"]}) (score={score:.3f}) для {query!r}')
            return doc.page_content
        logger.info(f'Релевантные документы не найдены для {query!r}')
        return None
    except Exception as exc:
        logger.error(f'Ошибка при выполнении RAG-поиска для {query!r}: {exc}', exc_info=True)
        return None
    