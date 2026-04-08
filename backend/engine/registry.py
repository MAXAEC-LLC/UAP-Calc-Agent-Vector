"""
registry.py — Agent Registry & Dispatcher.
Maps agent names to handler functions with proper dependency injection.
"""

import logging
from . import agents
from .helpers import create_mcp_message


class AgentRegistry:
    def __init__(self):
        self.registry = {
            "Librarian": agents.agent_context_librarian,
            "Researcher": agents.agent_researcher,
            "Writer": agents.agent_writer,
            "Summarizer": agents.agent_summarizer,
        }

    def get_handler(self, agent_name, client, index, generation_model, embedding_model, namespace_context, namespace_knowledge, agent_settings=None, property_context=None, conversation_history=None, embedding_client=None):
        """Returns a callable handler for the given agent, with dependencies pre-bound.
        client: Anthropic client for generation
        embedding_client: OpenAI client for embeddings (falls back to client if not set)
        """
        emb_client = embedding_client or client
        handler_func = self.registry.get(agent_name)
        if not handler_func:
            logging.error(f"Agent '{agent_name}' not found in registry.")
            raise ValueError(f"Agent '{agent_name}' not found in registry.")

        if agent_name == "Librarian":
            return lambda mcp_message: handler_func(
                mcp_message, client=emb_client, index=index,
                embedding_model=embedding_model, namespace_context=namespace_context,
                agent_settings=agent_settings,
            )
        elif agent_name == "Researcher":
            return lambda mcp_message: handler_func(
                mcp_message, client=client, index=index,
                generation_model=generation_model, embedding_model=embedding_model,
                namespace_knowledge=namespace_knowledge,
                agent_settings=agent_settings,
                property_context=property_context,
                conversation_history=conversation_history,
                embedding_client=emb_client,
            )
        elif agent_name == "Writer":
            return lambda mcp_message: handler_func(
                mcp_message, client=client, generation_model=generation_model,
                agent_settings=agent_settings,
            )
        elif agent_name == "Summarizer":
            return lambda mcp_message: handler_func(
                mcp_message, client=client, generation_model=generation_model,
                agent_settings=agent_settings,
            )
        else:
            return handler_func

    def get_capabilities_description(self):
        """Returns a structured description of agents for the Planner LLM."""
        return """
Available Agents and their required inputs.
CRITICAL: You MUST use the exact input key names provided for each agent.
MANDATORY PIPELINE: You MUST ALWAYS use Librarian (Step 1) → Researcher (Step 2) → Writer (Step 3).
NEVER skip the Researcher. The Writer cannot produce useful answers without the Researcher's factual data.

1. AGENT: Librarian
   ROLE: Retrieves Semantic Blueprints (style/structure instructions).
   INPUTS:
     - "intent_query": (String) A descriptive phrase of the desired style.
   OUTPUT: The blueprint structure (JSON string).

2. AGENT: Researcher  [MANDATORY — NEVER SKIP]
   ROLE: Retrieves and synthesizes factual information from the knowledge base.
   INPUTS:
     - "topic_query": (String) The subject matter to research. Should be the user's actual question.
   OUTPUT: Synthesized facts with source citations (String).

3. AGENT: Summarizer  [OPTIONAL — only if Researcher output is very large]
   ROLE: Reduces large text to a concise summary based on a specific objective.
   INPUTS:
     - "text_to_summarize": (String/Reference) The long text to be summarized.
     - "summary_objective": (String) A clear goal for the summary.
   OUTPUT: A dictionary containing the summary: {"summary": "..."}.

4. AGENT: Writer
   ROLE: Generates the final answer by applying a Blueprint to the Researcher's facts.
   INPUTS:
     - "blueprint": (String/Reference) The style instructions (from Librarian, $$STEP_1_OUTPUT$$).
     - "facts": (String/Reference) Factual information (from Researcher, $$STEP_2_OUTPUT$$).
   OUTPUT: The final generated text (String).
"""


AGENT_TOOLKIT = AgentRegistry()
logging.info("✅ Agent Registry initialized.")
