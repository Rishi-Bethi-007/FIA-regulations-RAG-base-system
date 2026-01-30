import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from index.search import FaissRetriever

from guardrails.input_guardrails import guard_user_input
from guardrails.output_guardrails import enforce_confidence_threshold
from guardrails.prompt_builder import build_messages
from guardrails.schema import AnswerSchema

load_dotenv()

MODEL = "gpt-4.1-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

retriever = FaissRetriever(
    index_path="index/chunk_index.faiss",
    meta_path="index/chunk_metadata.json",
)


def call_llm(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


def run_rag(query: str, top_k: int = 5) -> AnswerSchema:
    """
    Phase 3.1: Raw RAG pipeline
    """

    # 1️⃣ Input guardrails
    guard_user_input(query)

    # 2️⃣ Retrieve context
    retrieved_chunks = retriever.retrieve(query, top_k=top_k)

    # 3️⃣ Build guarded prompt
    messages = build_messages(query, retrieved_chunks)

    # 4️⃣ Call LLM with schema validation + retries
    last_error = None
    for _ in range(3):
        raw = call_llm(messages)
        try:
            parsed = AnswerSchema.model_validate_json(raw)
            break
        except ValidationError as e:
            last_error = e
            messages.append({
                "role": "user",
                "content": "Your output was invalid. Return ONLY valid JSON."
            })
    else:
        raise RuntimeError(f"RAG failed after retries: {last_error}")

    # 5️⃣ Output guardrails
    if parsed.answer.strip().lower() != "i don't know":
        enforce_confidence_threshold(parsed.confidence)

    return parsed
