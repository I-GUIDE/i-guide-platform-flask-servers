# llm_utils.py

import os
import asyncio
import httpx
from openai import AsyncOpenAI
from typing import Dict, Any, Optional

# --- Payload Creation ---

def create_query_payload(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.8,
    top_p: float = 0.9,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Creates a standardized payload for LLM API calls, compatible with OpenAI's format.
    
    This version correctly uses the model name passed to it and includes temperature/top_p,
    reflecting a more robust version of the provided JavaScript.
    """
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }

# --- LLM API Call Functions ---

async def call_llama_model(query_payload: Dict[str, Any]) -> Optional[str]:
    """
    Makes an asynchronous API call to a Llama-compatible model via a proxy server.
    
    This function requires the VLLM_PROXY and VLLM_API_KEY environment variables to be set.
    """
    proxy_url = os.getenv("VLLM_PROXY")
    proxy_token = os.getenv("VLLM_API_KEY")

    if not proxy_url or not proxy_token:
        print("Error: VLLM_PROXY and VLLM_API_KEY environment variables must be set.")
        return "Error: Proxy URL or Token not configured."

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {proxy_token}',
    }
    
    # The provided JS hardcodes the model name. If your proxy requires a specific
    # model name that differs from what's passed in, you can override it here.
    # For example: query_payload['model'] = "Qwen/Qwen2.5-7B-Instruct"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(proxy_url, headers=headers, json=query_payload)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
            
            result = response.json()
            # Safely access the nested content
            content = result.get('choices', [{}])[0].get('message', {}).get('content')
            return content.strip() if content else None

    except httpx.HTTPStatusError as e:
        print(f"Error fetching from proxy server: {e.response.status_code} - {e.response.text}")
        return f"Error: Received status {e.response.status_code} from the proxy server."
    except Exception as e:
        print(f"An unexpected error occurred when calling the proxy server: {e}")
        return "An unexpected error occurred."

async def call_gpt_model(query_payload: Dict[str, Any]) -> Optional[str]:
    """
    Makes an asynchronous API call to an OpenAI-compatible model using the openai library.
    
    This function requires the OPENAI_KEY environment variable.
    It can also use OPENAI_API_URL if you are using a custom endpoint.
    """
    if not os.getenv("OPENAI_KEY"):
        print("Error: OPENAI_KEY environment variable must be set.")
        return "Error: OpenAI API key not configured."

    try:
        # The client automatically reads the API key from the OPENAI_KEY env var.
        # It can also be configured with a custom base_url.
        client = AsyncOpenAI(
            base_url=os.getenv("OPENAI_API_URL") # Will be None if not set, which is fine
        )
        
        completion = await client.chat.completions.create(
            model=query_payload["model"],
            messages=query_payload["messages"],
            temperature=query_payload.get("temperature", 0.8),
            top_p=query_payload.get("top_p", 0.9),
            stream=query_payload.get("stream", False)
        )
        
        content = completion.choices[0].message.content
        return content.strip() if content else None

    except Exception as e:
        print(f"Error fetching from GPT model: {e}")
        return "An error occurred while communicating with the GPT model."

async def call_llm(prompt: str) -> str:
    """
    A generic LLM call function for simpler tasks like fact extraction.
    This example defaults to using the Llama model configuration.
    """
    payload = create_query_payload(
        model="default-fact-extraction-model", # Model name may be ignored by the proxy
        system_prompt="You are a helpful assistant that extracts facts.",
        user_prompt=prompt
    )
    response = await call_llama_model(payload)
    return response or "Could not extract facts."