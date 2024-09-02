from typing import Dict, List

SUPPORTED_MODELS: Dict[str, List[str]] = {
    "OpenAI": ["gpt-4", "gpt-4o", "gpt-4-turbo", "gpt-4omini"],
    "Anthropic": [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
    "Google": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "Replicate": [
        "lucataco/moondream2:72ccb656353c348c1385df54b237eeb7bfa874bf11486cf0b9473e691b662d31"
    ],
}

MAPPER_MODELS: Dict[str, List[str]] = {
"OpenAI": ["gpt-4", "gpt-4o", "gpt-4-turbo", "gpt-4omini"],
"Manually": ["manually"]
}