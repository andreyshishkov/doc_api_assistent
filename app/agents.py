from crewai import Agent, Task, Crew, LLM

from app.logger import logger
from app.settings import settings

llm = LLM(
    model=settings.OLLAMA_MODEL,
    base_url=settings.ollama_url,
    temperature=0.0,
    timeout=60.0,
    max_tokens=300,
    api_key=settings.API_KEY,
)


def _create_generator_agent():
    return Agent(
        role='API-документатор',
        goal='Генерировать документацию в строгом формате.',
        backstory=(
            'Ты специалист по API. Генерируй документацию ТОЛЬКО в формате:\n'
            '### МЕТОД /путь\n'
            '**Описание**: ... \n'
            '**Параметры**: ... \n'
            '**Ответ**:\n'
            '```json\n{...}\n```'
        ),
        llm=llm,
        verbose=False,
    )


def _create_validator_agent():
    return Agent(
        role='Валидатор документации',
        goal='Проверять соответствие документа строгому формату.',
        backstory=(
            'Ты строгий QA-инженер. Ты должен проверить, что документ содержит:\n'
            '1. Строку вида "### МЕТОД /путь",\n'
            '2. Блок "**Описание**:",\n'
            '3. Блок "**Параметры" или "**Параметры пути**:",\n'
            '4. Блок "**Ответ**:" с JSON-примером в тройных кавычках.\n'
            'Если всё есть — ответь "valid". Иначе — "invalid".'
        ),
        llm=llm,
        verbose=False,
    )


def generate_and_validate_documentation(query: str) -> str | None:
    """
    Генерирует документацию и проверяет её валидатором.
    Возвращает контент, только если он валиден.
    """
    logger.info(f'Запуск генерации по запросу: {query}')

    generator = _create_generator_agent()
    validator = _create_validator_agent()

    gen_task = Task(
        description=f'Создай документацию для: {query}',
        expected_output='Документ в строгом формате. Ничего больше.',
        agent=generator
    )
    val_task = Task(
        description='Проверь сгенерированный на предыдущем шаге документ на соответствие формату.',
        expected_output='Только "valid" или "invalid", без пояснений.',
        agent=validator
    )
    crew = Crew(
        agents=[generator, validator],
        tasks=[gen_task, val_task],
        verbose=False,
    )

    crew_output = crew.kickoff()
    validation_result = str(crew_output).strip().lower()

    if 'valid' in validation_result:
        logger.info(f'Документ прошёл валидацию для запроса: {query}')
        return str(gen_task.output.raw).strip()
    else:
        logger.warning(f'Документ не прошёл валидацию. Вердикт: {validation_result}')
        return None
