import json
from app.core.llm import get_llm


def extract_entities_and_relations(chunk_content: str, llm=None) -> dict:
    """
    Ask the LLM to read a chunk and pull out entities (people, plans,
    features, policies) and relationships between them. This is the
    raw material a knowledge graph is built from.
    """
    llm = llm or get_llm()
    prompt = f"""Read this text and extract entities and relationships
between them, relevant to a Terms of Use / subscription service document.

Entity types to look for: PLAN (subscription plans), FEATURE (service
features), POLICY (rules/clauses), ROLE (account owner, extra member, etc.)

Respond ONLY with valid JSON in this exact format, no other text:
{{
  "entities": [{{"name": "...", "type": "..."}}],
  "relationships": [{{"source": "...", "relation": "...", "target": "..."}}]
}}

If no clear entities/relationships exist, respond with:
{{"entities": [], "relationships": []}}

TEXT:
{chunk_content}
"""
    response = llm.invoke(prompt).content.strip()
    response = response.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"entities": [], "relationships": []}