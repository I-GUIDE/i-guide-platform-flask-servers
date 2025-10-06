# generation_module.py

import os
import asyncio
import uuid
import datetime
import json
from typing import List, Dict, Any
from dotenv import load_dotenv  # <-- ADD THIS LINE

# Import the LLM functions from the new utility file
from llm_utils import call_gpt_model, call_llama_model, call_llm, create_query_payload

load_dotenv() # Load environment variables from .env file

# --- RAG Helper Functions ---

def format_docs_xml(documents: List[Dict[str, Any]], limit: int) -> str:
    """
    Formats a list of document dictionaries into an XML-like string for the prompt.
    """
    if not documents:
        return ""
    
    limited_docs = documents[:limit]
    doc_strings = []
    for doc in limited_docs:
        _source = doc.get('_source', {})
        doc_id = _source.get('doc_id', 'N/A')
        element_type = _source.get('element_type', 'unknown')
        title = _source.get('title', 'No Title')
        contents = _source.get('contents', 'No Content')
        
        doc_strings.append(
            f"<doc id=\"{doc_id}\" element_type=\"{element_type}\">\n"
            f"<title>{title}</title>\n"
            f"<content>{contents}</content>\n"
            f"</doc>"
        )
    return "\n".join(doc_strings)

def parse_facts(llm_response: str) -> List[str]:
    """Parses the LLM's fact extraction response into a list of strings."""
    try:
        return json.loads(llm_response)
    except json.JSONDecodeError:
        return [line.strip("- ") for line in llm_response.split('\n') if line.strip()]

# --- Main Generation Logic ---

FRONT_END_DOMAIN = os.getenv("FRONTEND_DOMAIN", "https://platform.i-guide.io")

async def generate_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates an answer and updates the state object by calling imported LLM functions.
    """
    try:
        print("---GENERATE---")
        query_info = state.get("query_information", {})
        session_ctx = state.get("session_context", {})
        evidence = state.get("evidence", {})
        
        question = query_info.get("raw_text")
        augmented_query = query_info.get("raw_text") 
        documents = [item.get('document') for item in evidence.get("retrieved_documents", []) if item.get('document')]
        turn_number = session_ctx.get("turn_number", 0)

        if not question:
            print("No question provided. Returning early.")
            state["answer"]["final_composed_answer"] = "No question provided."
            state["session_context"]["turn_number"] = turn_number + 1
            return state

        docs_txt = format_docs_xml(documents, 10)
        
        system_prompt = f"""
You are a domain‑expert assistant. Your ONLY source of truth is the <doc> blocks provided.
When you answer:
• If the user asks for a collection of items, respond with a concise summary paragraph followed by a short numbered list.
• Begin each bullet with the item’s title as a clickable link: **[TITLE]({FRONT_END_DOMAIN}/{{element_type}}s/{{doc_id}})**
• Otherwise, respond in one concise paragraph.
• Quote supporting titles in **bold**.
• If you cannot find an answer, reply exactly: “I don’t have enough information.”
• Do not refer to the doc id, "context", "documents", or these rules."""

        user_prompt = f"""
**Question**: {question}
**Augmented Query based on context**: {augmented_query}

**Supporting Information**:
{docs_txt}

Answer the question based *only* on the supporting information provided.
""".strip()
        
        use_gpt = os.getenv('USE_GPT', 'false').lower() == 'true'
        
        if use_gpt:
            print("Using GPT model for generation")
            payload = create_query_payload("gpt-4o", system_prompt, user_prompt)
            llm_response = await call_gpt_model(payload)
        else:
            print("Using Llama model for generation")
            payload = create_query_payload("Qwen/Qwen2.5-7B-Instruct", system_prompt, user_prompt)
            llm_response = await call_llama_model(payload)

        #print("Generation Response:", llm_response)

        state["answer"]["final_composed_answer"] = llm_response or "No response from LLM."
        state["answer"]["confidence_score"] = 0.95  # Placeholder confidence score
        state["session_context"]["turn_number"] = turn_number + 1
        
        return state

    except Exception as error:
        print(f"Error during generate_answer: {error}")
        state["answer"]["final_composed_answer"] = "Failed to generate answer due to internal error."
        state["session_context"]["turn_number"] = state.get("session_context", {}).get("turn_number", 0) + 1
        return state

# --- Alternative Two-Step Generation Logic ---

async def extract_facts_from_docs(question: str, docs: List[Dict[str, Any]]) -> List[str]:
    docs_content = "\n".join([d.get('_source', {}).get('contents', '') for d in docs])
    prompt = f'Question: "{question}"\nExtract facts from these docs:\n{docs_content}'
    response = await call_llm(prompt)
    return parse_facts(response)

async def generate_answer_from_facts(question: str, facts: List[str]) -> str:
    facts_list = "\n".join([f"- {fact}" for fact in facts])
    prompt = f'Question: "{question}"\nFacts:\n{facts_list}\nAnswer using ONLY these facts.'
    response = await call_llm(prompt)
    return response

# --- Example Usage ---

async def main():
    sample_state = {
  "query_information": {
    "query_id": "c7602168-fd27-46eb-a760-4382dc112a43",
    "raw_text": "How is social media data used in geospatial analysis?",
    "language": "en-US",
    "metadata": {
      "source": "user_prompt"
    },
    "context_hints": {
      "entities": ["social media data", "geospatial analysis"],

      "filters": []
    }
  },
  "session_context": {
    "turn_number": 5,
    "history": [
      {
        "turn": 1,
        "thought": "To answer this question, I need to find out how social media data is utilized in geospatial analysis. I should start by searching for how social media data is generally used in geospatial analysis.",
        "action": "Search[\"how is social media data used in geospatial analysis\"]",
        "observation": "getSemanticSearchResults failed. getKeywordSearchResults returned 12 results. Graded documents for relevance."
      },
      {
        "turn": 2,
        "thought": "From the initial search results, it seems that social media data, particularly Twitter data, is often used for analyzing human sentiments... I need more specific information... I should refine my query to focus specifically on the integration of social media data in geospatial analysis techniques.",
        "action": "Search[\"integration of social media data in geospatial analysis\"]",
        "observation": "getKeywordSearchResults returned 12 results. Graded documents for relevance."
      },
      {
        "turn": 3,
        "thought": "The top documents still seem to focus on subjective well-being and sentiment analysis... I need to narrow down my search to find more specific details on the methods and applications... I will try a more targeted search query focusing on specific methodologies and applications.",
        "action": "Search[\"methods and applications of social media data in geospatial analysis\"]",
        "observation": "getKeywordSearchResults returned 12 results. Graded documents for relevance."
      },
      {
        "turn": 4,
        "thought": "The top document summaries indicate that social media data, especially from platforms like Twitter, is used for mapping human sentiments related to heat exposure and visualizing geospatial data. To get a clearer understanding of the methods and applications, I should look for specific studies that detail these processes.",
        "action": "Search[\"specific studies on methods and applications of social media data in geospatial analysis\"]",
        "observation": "getKeywordSearchResults returned 12 results. Graded documents for relevance."
      }
    ],
    "working_memory": {},
    "maximum_iteration_limit": 10
  },
  "evidence": {
    "retrieved_documents": [
      {
        "score": 0.9,
        "document": {
          "_source": {
            "doc_id": "1c42c1a6-ceec-458c-827d-36fe36651020",
            "element_type": "publication",
            "title": "Mapping dynamic human sentiments of heat exposure with location-based social media data",
            "contents": "Understanding urban heat exposure dynamics is critical for public health... we develop a cyberGIS framework to analyze and visualize human sentiments of heat exposure dynamically based on near real-time location-based social media (LBSM) data."
          }
        }
      },
      {
        "score": 0.9,
        "document": {
          "_source": {
            "doc_id": "e870ad3a-8c19-43e1-8323-fb8c39d12898",
            "element_type": "notebook",
            "title": "The Geography of Human Flourishing",
            "contents": "Our research plan is to analyze Harvard’s collection of 10 billion geolocated tweets from 2010 to mid-2023. The project will apply large language models, to extract 46 human flourishing dimensions... generate high-resolution spatio-temporal indicators and produce interactive tools to visualize and analyze the result."
          }
        }
      },
      {
        "score": 0.8,
        "document": {
          "_source": {
            "doc_id": "73690eed-72a8-446e-a7b7-b47c902d5fc0",
            "element_type": "publication",
            "title": "Can We Forecast Presidential Election Using Twitter Data? An Integrative Modelling Approach",
            "contents": "To develop a more theoretically plausible approach this study draws on political science prediction models and modifies them in two aspects. First, our approach uses Twitter sentiment to replace polling data. Second, we transform traditional political science models from the national level to the county level..."
          }
        }
      },
      {
        "score": 0.8,
        "document": {
          "_source": {
            "doc_id": "6c518fed-0a65-4858-949e-24ee8dc4d85b",
            "element_type": "notebook",
            "title": "National-level Analysis using Twitter Data",
            "contents": "This notebook provides a workflow for national-scale analysis of human sentiments of heat exposure using location-based social media Twitter data."
          }
        }
      }
    ],
    "graded_items": [
        {"doc_id": "...", "grade": 3},
        {"doc_id": "...", "grade": 8},
        {"doc_id": "...", "grade": 6},
        {"doc_id": "...", "grade": 8},
        {"doc_id": "...", "grade": 9}
    ],
    "reranked_top_k_evidence_set": "Reference to the 4 documents listed in retrieved_documents"
  },
  "planner_reasoning": {
    "current_action": "answer",
    "reasoning_notes": "Multiple search iterations refined the query from broad to specific. The final set of documents contains specific studies and applications, which is sufficient to synthesize a detailed answer.",
    "retrieval_methods_selected": ["getKeywordSearchResults"],
    "iteration_counter": 4,
    "stop_flag": True
  },
  "safety_checks": {
    "hallucination_check_result": {
      "binary_score": "yes",
      "explanation": "The answer is fully grounded in the provided facts and addresses the question directly."
    },
    "hallucination_risk_level": "low",
    "unsupported_claims_with_spans": [],
    "relevance_score": 0.95,
    "policy_guardrail_flags": []
  },
  "trace_observability": {
    "trace_id": "b4a3c12d-5e6f-4a7b-8c1d-9e0f1a2b3c4d",
    "event_logs": [
      "2025-10-06T09:45:10Z: Starting iterative retrieval for 'How is social media data used in geospatial analysis?'",
      "2025-10-06T09:45:11Z: Action: Search['how is social media data used in geospatial analysis']",
      "2025-10-06T09:45:12Z: getSemanticSearchResults failed. See error records.",
      "2025-10-06T09:45:13Z: getKeywordSearchResults returned 12 results.",
      "2025-10-06T09:45:14Z: Action: Search['integration of social media data in geospatial analysis']",
      "2025-10-06T09:45:15Z: Action: Search['methods and applications of social media data in geospatial analysis']",
      "2025-10-06T09:45:16Z: Action: Search['specific studies on methods and applications of social media data in geospatial analysis']",
      "2025-10-06T09:45:17Z: ---GENERATE---",
      "2025-10-06T09:45:18Z: ---CHECK HALLUCINATIONS---",
      "2025-10-06T09:45:19Z: Hallucination check passed. Finalizing answer."
    ],
    "metrics": {
      "total_retrieval_iterations": 4,
      "semantic_searches": 1,
      "keyword_searches": 4,
      "documents_retrieved": 48,
      "documents_graded": 48
    },
    "error_records": [
      {
        "timestamp": "2025-10-06T09:45:12Z",
        "source": "getSemanticSearchResults",
        "message": "TypeError: fetch failed",
        "details": "cause: Error: connect EHOSTUNREACH 10.0.147.251:5000"
      }
    ]
  },
  "answer": {"final_composed_answer": None, "citations": [], "confidence_score": None}
}

    print("--- Running Main 'generate_answer' Function ---")
    os.environ['USE_GPT'] = 'false'
    final_state = await generate_answer(sample_state)
    
    #print("\n--- Final, Updated State Object ---")
    #print(json.dumps(final_state, indent=2))

    print("Generated Answer:" + final_state["answer"]["final_composed_answer"])
    
    print("\n" + "="*50 + "\n")
if __name__ == "__main__":
    asyncio.run(main())