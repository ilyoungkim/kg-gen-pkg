from typing import List, Optional
from pathlib import Path
import dspy
import litellm
from pydantic import BaseModel

from kngraph.utils.ollama_client import OllamaError, ollama_chat_json


class TextEntities(dspy.Signature):
    """Extract key entities from the source text. Extracted entities are subjects or objects.
    This is for an extraction task, please be THOROUGH and accurate to the reference text."""

    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


class ConversationEntities(dspy.Signature):
    """Extract key entities from the conversation Extracted entities are subjects or objects.
    Consider both explicit entities and participants in the conversation.
    This is for an extraction task, please be THOROUGH and accurate."""

    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


class EntitiesResponse(BaseModel):
    """Structured response for entity extraction."""

    entities: List[str]


def _build_entities_schema() -> dict:
    schema = EntitiesResponse.model_json_schema()
    schema['additionalProperties'] = False
    return schema


def _load_entities_prompt() -> str:
    """Load the entities prompt template from file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "entities.txt"
    return prompt_path.read_text()


def _get_entities_litellm(
    input_data: str,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float | None = None,
) -> List[str]:
    prompt_template = _load_entities_prompt()
    user_prompt = f"""
Here is the text to extract entities from:

<article>
{input_data}
</article>
    """

    # Build schema with additionalProperties: false (required by OpenAI)
    schema = EntitiesResponse.model_json_schema()
    schema["additionalProperties"] = False

    kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "entities_response",
                "schema": schema,
                "strict": True,
            }
        },
    }

    if temperature is not None and "gpt-5" not in model:
        kwargs["temperature"] = temperature

    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    response = litellm.responses(**kwargs)
    # print(response.model_dump_json(indent=2))
    parsed = EntitiesResponse.model_validate_json(response.output[-1].content[0].text)
    return parsed.entities


def _get_entities_ollama(
    input_data: str,
    model: str,
    api_base: Optional[str] = None,
    temperature: float | None = None,
    progress_label: str | None = None,
) -> List[str]:
    prompt_template = _load_entities_prompt()
    user_prompt = f"""
Here is the text to extract entities from:

<article>
{input_data}
</article>
    """

    schema = _build_entities_schema()

    try:
        raw_json = ollama_chat_json(
            model=model,
            messages=[
                {'role': 'system', 'content': prompt_template},
                {'role': 'user', 'content': user_prompt},
            ],
            schema=schema,
            api_base=api_base,
            temperature=temperature,
            progress_label=progress_label,
        )
        parsed = EntitiesResponse.model_validate_json(raw_json)
        return parsed.entities
    except (OllamaError, ValueError):
        fallback_prompt = (
            user_prompt
            + '\n\nReturn only valid JSON matching this shape exactly: '
            + '{"entities": ["entity1", "entity2"]}. '
            + 'If there are no entities, return {"entities": []}.'
        )
        try:
            raw_json = ollama_chat_json(
                model=model,
                messages=[
                    {'role': 'system', 'content': prompt_template},
                    {'role': 'user', 'content': fallback_prompt},
                ],
                schema=schema,
                api_base=api_base,
                temperature=temperature,
                progress_label=(f'{progress_label} 재시도' if progress_label else '엔티티 추출 재시도'),
            )
            parsed = EntitiesResponse.model_validate_json(raw_json)
            return parsed.entities
        except (OllamaError, ValueError):
            return []


def get_entities(
    input_data: str,
    is_conversation: bool = False,
    use_litellm_prompt: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float | None = None,
    use_ollama: bool = False,
    progress_label: str | None = None,
) -> List[str]:
    if use_ollama and not is_conversation:
        return _get_entities_ollama(
            input_data,
            model=model,
            api_base=api_base,
            temperature=temperature,
            progress_label=progress_label,
        )

    if use_litellm_prompt and not is_conversation:
        return _get_entities_litellm(
            input_data,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
        )

    extract = (
        dspy.Predict(ConversationEntities)
        if is_conversation
        else dspy.Predict(TextEntities)
    )
    result = extract(source_text=input_data)
    return result.entities
