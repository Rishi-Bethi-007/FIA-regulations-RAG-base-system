def enforce_confidence_threshold(confidence: float, threshold: float = 0.7) -> None:
    if confidence < threshold:
        raise ValueError(f"Low confidence ({confidence:.2f}) below threshold ({threshold:.2f}).")


def enforce_refusal_policy(answer: str) -> None:
    """
    If the model says "I don't know", that is acceptable.
    If it gives an answer, that's fine too (schema will validate).
    This guardrail can be extended for domain policies.
    """
    # Example: if you want to enforce strict refusal for unknown contexts,
    # you’d do it here based on additional signals.
    if not answer or not answer.strip():
        raise ValueError("Empty answer output.")
