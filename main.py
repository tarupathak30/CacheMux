import os
import httpx
import hashlib
import json
import asyncio
import time
import uuid
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from groq import AsyncGroq
from google import genai
from google.genai import types
from upstash_redis.asyncio import Redis
from dotenv import load_dotenv


load_dotenv() 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cost-latency-guardrail-proxy")

# Initialize FastAPI
app = FastAPI(
    title="Cost & Latency Guardrail Proxy",
    description="An OpenAI-compatible gateway proxy with caching and failover mechanisms.",
    version="1.0.0"
)

# Load configuration from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

DEFAULT_GROQ_MODEL = os.getenv("DEFAULT_GROQ_MODEL", "llama-3.3-70b-versatile")
FALLBACK_GEMINI_MODEL = os.getenv("FALLBACK_GEMINI_MODEL", "gemini-2.5-flash")

# Initialize Upstash Redis
redis_client = None
if UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN:
    redis_client = Redis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)
    logger.info("Upstash Redis async client initialized successfully.")
else:
    logger.warning("Upstash Redis credentials missing. Caching will be disabled.")

# Initialize LLM Clients
groq_client = None
if GROQ_API_KEY:
    groq_client = AsyncGroq(api_key=GROQ_API_KEY)
    logger.info("Groq Async client initialized.")
else:
    logger.warning("GROQ_API_KEY is not set.")

gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Google Gemini client initialized.")
else:
    logger.warning("GEMINI_API_KEY is not set.")

# --- Pydantic Schema for Chat completions ---
class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[Any] = None

# --- Helper Functions ---

def get_messages_hash(messages: List[Dict[str, Any]]) -> str:
    """Serializes the message array deterministically and returns a SHA-256 hash."""
    serialized = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

async def get_cached_response(hash_key: str) -> Optional[str]:
    """Retrieves cached response from Upstash Redis if available."""
    if not redis_client:
        return None
    try:
        val = await redis_client.get(hash_key)
        if val:
            logger.info(f"Cache HIT for hash: {hash_key}")
            return val
    except Exception as e:
        logger.error(f"Redis cache read error: {e}")
    return None

async def set_cached_response(hash_key: str, content: str, ttl: int = 3600):
    """Writes the response content to Upstash Redis with a TTL."""
    if not redis_client:
        return
    try:
        await redis_client.set(hash_key, content, ex=ttl)
        logger.info(f"Cache WRITE successful for hash: {hash_key} (TTL: {ttl}s)")
    except Exception as e:
        logger.error(f"Redis cache write error: {e}")

def convert_messages_to_gemini(messages: List[Dict[str, Any]]):
    """Maps OpenAI-compatible message structure into Gemini SDK contents and config."""
    gemini_contents = []
    system_instruction = None

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            system_instruction = content
        elif role == "user":
            gemini_contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)]
                )
            )
        elif role in ("assistant", "model"):
            gemini_contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=content)]
                )
            )
        # Other roles are ignored/omitted for basic compatibility mapping

    config = None
    if system_instruction:
        config = types.GenerateContentConfig(system_instruction=system_instruction)

    return gemini_contents, config

def build_openai_compatible_response(content: str, model: str) -> Dict[str, Any]:
    """Builds a standard OpenAI-style response dictionary."""
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

# --- Stream Generators ---

async def stream_cached_response(cached_text: str, model_name: str) -> AsyncGenerator[str, None]:
    """Streams a cached response chunk-by-chunk to emulate LLM latency."""
    chunk_size = 20
    for i in range(0, len(cached_text), chunk_size):
        chunk_text = cached_text[i:i+chunk_size]
        openai_chunk = {
            "id": f"chatcmpl-cache-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": chunk_text
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(openai_chunk)}\n\n"
        await asyncio.sleep(0.01)

    final_chunk = {
        "id": f"chatcmpl-cache-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

async def stream_from_providers(request_data: ChatCompletionRequest, hash_key: str) -> AsyncGenerator[str, None]:
    """Streams completions from Groq, falling back to Gemini if initialization fails."""
    use_fallback = False
    groq_stream = None
    first_chunk = None

    # Prepare Groq parameters
    groq_kwargs = {
        "model": request_data.model or DEFAULT_GROQ_MODEL,
        "messages": request_data.messages,
        "stream": True
    }
    if request_data.temperature is not None:
        groq_kwargs["temperature"] = request_data.temperature
    if request_data.max_tokens is not None:
        groq_kwargs["max_tokens"] = request_data.max_tokens
    if request_data.top_p is not None:
        groq_kwargs["top_p"] = request_data.top_p
    if request_data.presence_penalty is not None:
        groq_kwargs["presence_penalty"] = request_data.presence_penalty
    if request_data.frequency_penalty is not None:
        groq_kwargs["frequency_penalty"] = request_data.frequency_penalty
    if request_data.stop is not None:
        groq_kwargs["stop"] = request_data.stop

    # Try Groq
    try:
        if not groq_client:
            raise ValueError("Groq client not initialized (missing GROQ_API_KEY)")
        
        logger.info(f"Initiating Groq stream with model: {groq_kwargs['model']}")
        groq_stream = await groq_client.chat.completions.create(**groq_kwargs)

        
        # Test connection & stream initialization by pulling first chunk
        first_chunk = await groq_stream.__anext__()
    except (Exception, StopAsyncIteration) as e:
        logger.error(f"Groq stream initialization failed: {e}. Falling back to Gemini.")
        use_fallback = True

    if not use_fallback and groq_stream: 
        collected_chunks = [] 

        if first_chunk: 
            delta_text = first_chunk.choices[0].delta.content or ""
            if delta_text: 
                collected_chunks.append(delta_text)
            
            yield f"data : {json.dumps(first_chunk.model_dump())}\n\n"

        try: 
            async for chunk in groq_stream: 
                delta_text = chunk.choices[0].delta.content or ""
                if delta_text: 
                    collected_chunks.append(delta_text)
                    # checking if the usage metadata sent at the end of the groq stream 
                    #  groq can append the real usage statistics inside the final stream chunk!!!
                    yield f"data : {json.dumps(chunk.model_dump())}\n\n"

                # caching the fully resolved string into the Redis asynchronously for next time
                full_response_text = "".join(collected_chunks)
                if full_response_text: 
                    await set_cached_response(hash_key, full_response_text)
        except Exception as stream_err: 
            logger.error(f"Error mid-way through Groq stream : {stream_err}")
            # if a failure striked mid-stream, fallback is not possible, raising clean drop is safe and sound. 
            yield "data : [DONE]\n\n"
            return
    else:
        logger.info(f"Rerouting streaming request to Gemini (Model: {FALLBACK_GEMINI_MODEL})")
        if not gemini_client:
            error_chunk = {"error": {"message": "Gemini client not initialized", "code": 500}}
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        try: 
            gemini_contents, gemini_config = convert_messages_to_gemini(request_data.messages)

            gemini_stream = await gemini_client.aio.models.generate_content_stream(
                model=FALLBACK_GEMINI_MODEL,
                contents=gemini_contents,
                config=gemini_config
            )

            collected_chunks = [] 
            for response_chunk in gemini_stream: 
                chunk_text = response_chunk.text or ""
                if chunk_text: 
                    collected_chunks.append(chunk_text)

                    openai_formatted_chunk = {
                        "id": f"chatcmpl-gemini-{uuid.uuid4()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": FALLBACK_GEMINI_MODEL,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": chunk_text
                                },
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(openai_formatted_chunk)}\n\n"

            # Finalize Gemini stream
            final_chunk = {
                "id": f"chatcmpl-gemini-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": FALLBACK_GEMINI_MODEL,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

            full_text = "".join(collected_chunks)
            if full_text: 
                await set_cached_response(hash_key, full_text)


        except Exception as gemini_err:
            logger.error(f"Gemini streaming failed: {gemini_err}")
            error_chunk = {"error": {"message": f"Gemini streaming failed: {str(gemini_err)}", "code": 500}}
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

# --- Endpoint Routing ---

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Guardrail: Intercept placeholder 'string' or empty values from Swagger/Test UI
    target_model = request.model
    if not target_model or target_model == "string":
        logger.info(f"Invalid model identifier '{target_model}' intercepted. Defaulting to: {DEFAULT_GROQ_MODEL}")
        target_model = DEFAULT_GROQ_MODEL

    # Generate deterministic hash key based on request messages
    hash_key = get_messages_hash(request.messages)

    # Check for hit in Upstash Redis cache layers 
    cached_response = await get_cached_response(hash_key)

    # -------------------------------------------------------------
    # PATH 1: STREAMING RESPONSES (True)
    # -------------------------------------------------------------
    if request.stream: 
        if cached_response: 
            logger.info("Serving streaming response directly from Upstash Redis cache.")
            return StreamingResponse(
                stream_cached_response(cached_response, target_model), 
                media_type='text/event-stream', 
                headers={'X-Cache': 'HIT', 'X-Provider': 'Cache'}
            )

        # Stream cache MISS -> Run provider streaming fallback pipeline
        return StreamingResponse(
            stream_from_providers(request, hash_key), 
            media_type='text/event-stream', 
            headers={'X-Cache': 'MISS', 'X-Provider': 'Groq/Gemini'}   
        )

    # -------------------------------------------------------------
    # PATH 2: NON-STREAMING RESPONSES (False)
    # -------------------------------------------------------------
    if cached_response:
        logger.info("Serving non-streaming response directly from Upstash Redis cache.")
        # Estimate usage metrics based on word length to bypass hardcoded 0 values
        word_count = len(cached_response.split())
        estimated_tokens = int(word_count * 1.33)  
        
        usage_metrics = {
            "prompt_tokens": 15,  
            "completion_tokens": estimated_tokens,
            "total_tokens": 15 + estimated_tokens
        }
        
        response_json = build_openai_compatible_response(cached_response, target_model)
        response_json["usage"] = usage_metrics 
        
        return JSONResponse(
            content=response_json,
            headers={"X-Cache": "HIT", "X-Provider": "Cache"}
        )

    # Non-Streaming Cache MISS -> Route to Groq directly
    try:
        if not groq_client:
            raise ValueError("Groq client not available.")

        logger.info(f"Routing non-streaming call to Groq using: {target_model}")
        
        # Build clean execution args for Groq client execution
        groq_args = {
            "model": target_model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p
        }
        # Clear out empty keys to preserve remote engine defaults
        groq_args = {k: v for k, v in groq_args.items() if v is not None}
        
        groq_res = await groq_client.chat.completions.create(**groq_args)
        
        # FIX: Access the object via standard attribute dot-notation, not indexed array array
        raw_content = groq_res.choices[0].message.content or ""
        
        # Write back full content data cleanly to Upstash cache layers
        await set_cached_response(hash_key, raw_content)
        
        # Map dynamic real usage data metrics back to client layers directly!
        real_usage = {
            "prompt_tokens": groq_res.usage.prompt_tokens if groq_res.usage else 0,
            "completion_tokens": groq_res.usage.completion_tokens if groq_res.usage else 0,
            "total_tokens": groq_res.usage.total_tokens if groq_res.usage else 0
        }
        
        out_response = build_openai_compatible_response(raw_content, target_model)
        out_response["usage"] = real_usage
        
        return JSONResponse(
            content=out_response,
            headers={"X-Cache": "MISS", "X-Provider": "Groq"}
        )

    except Exception as exc:
        logger.warning(f"Non-streaming primary Groq failed: {exc}. Issuing fallback to Gemini.")
        if not gemini_client:
            raise HTTPException(status_code=500, detail="All downstream target engine configurations exhausted.")
        
        try:
            logger.info(f"Rerouting non-streaming request to Gemini (Model: {FALLBACK_GEMINI_MODEL})")
            gemini_contents, gemini_config = convert_messages_to_gemini(request.messages)
            
            gemini_res = gemini_client.models.generate_content(
                model=FALLBACK_GEMINI_MODEL,
                contents=gemini_contents,
                config=gemini_config
            )
            
            fallback_text = gemini_res.text or ""
            await set_cached_response(hash_key, fallback_text)
            
            # Map Gemini response model data structure cleanly back out
            out_response = build_openai_compatible_response(fallback_text, FALLBACK_GEMINI_MODEL)
            
            # Extract real token details from Google's native schema format
            if hasattr(gemini_res, 'usage_metadata') and gemini_res.usage_metadata:
                out_response["usage"] = {
                    "prompt_tokens": gemini_res.usage_metadata.prompt_token_count,
                    "completion_tokens": gemini_res.usage_metadata.candidates_token_count,
                    "total_tokens": gemini_res.usage_metadata.total_token_count
                }
                
            return JSONResponse(
                content=out_response,
                headers={"X-Cache": "MISS", "X-Provider": "Gemini-Fallback"}
            )
            
        except Exception as fatal_err:
            logger.error(f"Complete system gateway execution crash: {fatal_err}")
            raise HTTPException(
                status_code=500, 
                detail=f"Both primary and fallback engine configurations failed: {str(fatal_err)}"
            )


@app.get("/health")
async def health_check():
    """Simple status check demonstrating provider and cache readiness."""
    status = {
        "status": "healthy",
        "cache_enabled": redis_client is not None,
        "providers": {
            "groq": groq_client is not None,
            "gemini": gemini_client is not None
        }
    }
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)












    