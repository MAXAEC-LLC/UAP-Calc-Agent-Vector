"""
engine.py — Core Orchestration: Planner, Executor, ExecutionTrace.
Manages the full Librarian → Researcher → Writer pipeline.
"""

import logging
import time
import json
import copy
from .helpers import call_llm_robust, create_mcp_message, count_tokens
from .registry import AGENT_TOOLKIT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# === ExecutionTrace ===

class ExecutionTrace:
    """Logs the entire execution flow for debugging and analytics."""

    def __init__(self, goal):
        self.goal = goal
        self.plan = None
        self.steps = []
        self.status = "Initialized"
        self.final_output = None
        self.start_time = time.time()
        self.duration = 0
        logging.info(f"ExecutionTrace initialized for goal: '{self.goal}'")

    def log_plan(self, plan):
        self.plan = plan
        logging.info("Plan has been logged to the trace.")

    def log_step(self, step_num, agent, planned_input, mcp_output, resolved_input, tokens_in=0, tokens_out=0):
        self.steps.append({
            "step": step_num,
            "agent": agent,
            "planned_input": planned_input,
            "resolved_context": resolved_input,
            "output": mcp_output["content"],
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_saved": max(0, tokens_in - tokens_out) if agent == "Summarizer" else 0,
        })
        logging.info(f"Step {step_num} ({agent}) logged. [In: {tokens_in}, Out: {tokens_out}]")

    def finalize(self, status, final_output=None):
        self.status = status
        self.final_output = final_output
        self.duration = time.time() - self.start_time
        logging.info(f"Trace finalized: '{status}'. Duration: {self.duration:.2f}s")

    def to_dict(self):
        """Serialize trace for API responses."""
        return {
            "goal": self.goal,
            "status": self.status,
            "duration": round(self.duration, 2),
            "steps": [
                {
                    "step": s["step"],
                    "agent": s["agent"],
                    "tokens_in": s["tokens_in"],
                    "tokens_out": s["tokens_out"],
                }
                for s in self.steps
            ],
        }


# === Planner ===

def planner(goal, capabilities, client, generation_model):
    """Analyzes the goal and generates a structured Execution Plan using the LLM."""
    logging.info("[Planner] Activated. Generating plan...")
    system_prompt = f"""
You are the strategic core of the Context Engine. Analyze the user's high-level GOAL and create a step-by-step EXECUTION PLAN.

AVAILABLE CAPABILITIES
---
{capabilities}
---
END CAPABILITIES

INSTRUCTIONS:
1. The output MUST be a single JSON object with a "plan" key containing a list of step objects.
2. CRITICAL: Every step object MUST strictly follow this schema:
   {{
      "step": <integer>,
      "agent": "<Agent Name>",
      "input": {{
          "<input_key>": "<input_value>"
      }}
   }}

3. MANDATORY PIPELINE — You MUST always include ALL THREE of these agents in this exact order:
   Step 1: Librarian  — retrieves the semantic blueprint.
   Step 2: Researcher — retrieves and synthesizes factual data. NEVER SKIP THIS.
   Step 3: Writer     — applies the blueprint to the Researcher's facts.

   Example minimal plan:
   {{"step": 1, "agent": "Librarian", "input": {{"intent_query": "<description of domain>"}}}}
   {{"step": 2, "agent": "Researcher", "input": {{"topic_query": "<COPY THE USER'S FULL ORIGINAL QUESTION HERE VERBATIM>"}}}}
   {{"step": 3, "agent": "Writer", "input": {{"blueprint": "$$STEP_1_OUTPUT$$", "facts": "$$STEP_2_OUTPUT$$"}}}}

   You may insert a Summarizer between Researcher and Writer ONLY if the research output would be very large.
   NEVER go directly from Librarian to Writer.

4. Use Context Chaining: format "$$STEP_N_OUTPUT$$" for values requiring previous outputs.
"""
    try:
        plan_json_string = call_llm_robust(
            system_prompt,
            goal,
            client=client,
            generation_model=generation_model,
            json_mode=True,
        )
        plan_data = json.loads(plan_json_string)

        if "plan" not in plan_data and isinstance(plan_data, list):
            return plan_data
        return plan_data["plan"]
    except Exception as e:
        logging.error(f"[Planner] Failed: {e}")
        raise e


# === Dependency Resolver ===

def resolve_dependencies(input_params, state):
    """Replace $$STEP_N_OUTPUT$$ placeholders with actual data from the execution state."""
    resolved_input = copy.deepcopy(input_params)

    def resolve(value):
        if isinstance(value, str) and value.startswith("$$") and value.endswith("$$"):
            ref_key = value[2:-2]
            return state.get(ref_key, value)
        elif isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    return resolve(resolved_input)


# === Context Engine (Main Entry Point) ===

def context_engine(goal, client, pc, index_name, generation_model, embedding_model, namespace_context, namespace_knowledge, agent_settings=None, property_context=None, conversation_history=None, embedding_client=None):
    """The main entry point for the Context Engine. Manages Planning and Execution."""
    logging.info(f"--- [Context Engine] Starting --- Goal: {goal}")
    trace = ExecutionTrace(goal)
    registry = AGENT_TOOLKIT

    try:
        index = pc.Index(index_name)

        # Use a hardcoded 2-step pipeline: Librarian retrieves the blueprint,
        # then Researcher synthesizes facts WITH the blueprint applied (no Writer).
        plan = [
            {"step": 1, "agent": "Librarian", "input": {"intent_query": goal}},
            {"step": 2, "agent": "Researcher", "input": {"topic_query": goal, "blueprint": "$$STEP_1_OUTPUT$$"}},
        ]
        trace.log_plan(plan)
    except Exception as e:
        trace.finalize(f"Failed during Planning/Init: {e}")
        return None, trace

    # --- Execute ---
    state = {}
    for step in plan:
        step_num = step.get("step")
        agent_name = step.get("agent")
        planned_input = step.get("input")

        logging.info(f"--- Executor: Step {step_num}: {agent_name} ---")
        try:
            handler = registry.get_handler(
                agent_name,
                client=client,
                index=index,
                generation_model=generation_model,
                embedding_model=embedding_model,
                namespace_context=namespace_context,
                namespace_knowledge=namespace_knowledge,
                agent_settings=agent_settings,
                property_context=property_context,
                conversation_history=conversation_history,
                embedding_client=embedding_client,
            )

            resolved_input = resolve_dependencies(planned_input, state)
            t_in = count_tokens(str(resolved_input))

            mcp_resolved_input = create_mcp_message("Engine", resolved_input)
            mcp_output = handler(mcp_resolved_input)
            output_data = mcp_output["content"]

            t_out = count_tokens(str(output_data))

            state[f"STEP_{step_num}_OUTPUT"] = output_data
            trace.log_step(step_num, agent_name, planned_input, mcp_output, resolved_input, tokens_in=t_in, tokens_out=t_out)

            logging.info(f"--- Executor: Step {step_num} completed. ---")

        except Exception as e:
            error_message = f"Execution failed at step {step_num} ({agent_name}): {e}"
            logging.error(f"--- Executor: FATAL ERROR --- {error_message}")
            trace.finalize(f"Failed at Step {step_num}")
            return None, trace

    # --- Finalization ---
    final_output = state.get(f"STEP_{len(plan)}_OUTPUT")
    trace.finalize("Success", final_output)
    logging.info("--- [Context Engine] Task Complete ---")
    return final_output, trace
