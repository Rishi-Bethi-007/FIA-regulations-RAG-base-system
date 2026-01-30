import re

# Common prompt-injection phrases. This is not perfect, but catches many naive attacks.
INJECTION_PATTERNS = [
    r"ignore (all|any) (previous|prior) instructions",
    r"disregard (all|any) rules",
    r"you are now (allowed|free) to",
    r"system message",
    r"developer message",
    r"reveal (the )?(system|developer) prompt",
    r"print (the )?(system|developer) instructions",
    r"jailbreak",
    r"do anything now",
]

MAX_INPUT_CHARS = 2000  # cost + safety guardrail


def enforce_input_length(user_input: str, max_chars: int = MAX_INPUT_CHARS) -> None:
    if len(user_input) > max_chars:
        raise ValueError(f"Input too long ({len(user_input)} chars). Max allowed: {max_chars}.")


def detect_prompt_injection(user_input: str) -> bool:
    text = user_input.lower()
    return any(re.search(p, text) for p in INJECTION_PATTERNS)


def guard_user_input(user_input: str) -> None:
    """
    Raises ValueError if the input violates guardrails.
    """
    enforce_input_length(user_input)
    if detect_prompt_injection(user_input):
        raise ValueError("Potential prompt injection detected in user input.")
