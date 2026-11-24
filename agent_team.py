"""
Minimal orchestrator-worker agent team (no framework).

    Goal -> Orchestrator (plan) -> Workers in parallel -> Reviewer -> done / retry

Run (Git Bash / Linux / macOS):
    pip install google-genai anthropic

    # Gemini (default provider)
    export GEMINI_API_KEY=...
    python agent_team.py "Build a CLI tool that checks if URLs in a README are alive"

    # Claude instead
    export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=...

    # Local model with Ollama (free, no key, no rate limits): install from
    # https://ollama.com, then:  ollama pull llama3.2
    export LLM_PROVIDER=ollama          # optional: OLLAMA_MODEL=qwen2.5:7b

    # Offline demo, no key needed
    export AGENT_MOCK=1

PowerShell: use  $env:GEMINI_API_KEY = "..."  instead of export.
"""
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()     # "gemini", "anthropic" or "ollama"
# Comma-separated list: the first is preferred, the others are fallbacks
# used when a model stays overloaded, e.g. "gemini-3.8-flash,gemini-3.5-flash".
GEMINI_MODELS = [m.strip() for m in os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").split(",") if m.strip()]
ANTHROPIC_MODELS = [m.strip() for m in os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5").split(",") if m.strip()]
OLLAMA_MODELS = [m.strip() for m in os.environ.get("OLLAMA_MODEL", "llama3.2").split(",") if m.strip()]
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST       # Ollama also accepts "127.0.0.1:11434"
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "600"))  # local models can be slow
MAX_ROUNDS = 3
MAX_RETRIES = 5                 # attempts per model before falling back
RETRYABLE = {429, 500, 502, 503, 504, 529}   # rate limit / overloaded / server errors
MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "3"))  # concurrent LLM calls
_limit = asyncio.Semaphore(MAX_PARALLEL)
_failed_models: set = set()     # models that stayed unavailable during this run


# --------------------------------------------------------------------------- #
# LLM access: retries with backoff, model fallback, or a deterministic mock
# --------------------------------------------------------------------------- #
async def _call_gemini(model: str, system: str, prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY
    resp = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system),
    )
    return resp.text


async def _call_anthropic(model: str, system: str, prompt: str) -> str:
    from anthropic import AsyncAnthropic

    async with AsyncAnthropic() as client:
        msg = await client.messages.create(
            model=model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
    return msg.content[0].text


async def _call_ollama(model: str, system: str, prompt: str) -> str:
    import httpx  # installed together with anthropic / google-genai

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
    except httpx.ConnectError:
        sys.exit(f"Cannot reach Ollama at {OLLAMA_HOST}. Is it installed and running? (start it with: ollama serve)")
    if resp.status_code == 404:
        sys.exit(f"Ollama has no model '{model}'. Download it first:  ollama pull {model}")
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _status(exc: Exception):
    """HTTP status code of an SDK error (google-genai uses .code, anthropic .status_code)."""
    response = getattr(exc, "response", None)          # httpx errors (Ollama)
    return (getattr(exc, "code", None) or getattr(exc, "status_code", None)
            or getattr(response, "status_code", None))


async def call_llm(system: str, prompt: str) -> str:
    if os.environ.get("AGENT_MOCK") == "1":
        return _mock_llm(system, prompt)

    if PROVIDER == "gemini":
        key, models, call = "GEMINI_API_KEY", GEMINI_MODELS, _call_gemini
    elif PROVIDER == "anthropic":
        key, models, call = "ANTHROPIC_API_KEY", ANTHROPIC_MODELS, _call_anthropic
    elif PROVIDER == "ollama":
        key, models, call = None, OLLAMA_MODELS, _call_ollama      # local: no key needed
    else:
        sys.exit(f"Unknown LLM_PROVIDER '{PROVIDER}' (use 'gemini', 'anthropic' or 'ollama').")
    if key and not os.environ.get(key):
        sys.exit(f"{key} is not set (or set AGENT_MOCK=1 for offline mode).")

    last_error = None
    # Skip models that already failed in this run, unless none are left.
    usable = [m for m in models if m not in _failed_models] or models[-1:]
    for model in usable:
        for attempt in range(1, MAX_RETRIES + 1):
            if model in _failed_models and model != usable[-1]:
                break                         # another agent gave up on it meanwhile
            try:
                async with _limit:
                    reply = await call(model, system, prompt)
                print(f"    answered by {model}")
                return reply
            except Exception as exc:
                if _status(exc) not in RETRYABLE:
                    raise                     # bad key, bad model name, etc.: fail fast
                last_error = exc
                wait = min(2 ** attempt, 30)  # 2, 4, 8, 16, 30 seconds
                print(f"    {model}: HTTP {_status(exc)}, retry {attempt}/{MAX_RETRIES} in {wait}s")
                await asyncio.sleep(wait)
        if model != usable[-1]:
            if model not in _failed_models:
                print(f"    {model} still unavailable, using the next model for the rest of this run")
            _failed_models.add(model)
    raise RuntimeError(f"All models failed after retries: {models}") from last_error


def _mock_llm(system: str, prompt: str) -> str:
    if "You are the orchestrator" in system:
        return json.dumps([
            {"agent": "researcher", "task": "List libraries for HTTP checks and Markdown link parsing."},
            {"agent": "coder", "task": "Write the CLI in Python using the chosen libraries."},
            {"agent": "writer", "task": "Write a short README with usage examples."},
        ])
    if "You are the reviewer" in system:
        # Reject on round 1 to demonstrate the feedback loop, approve afterwards.
        if "ROUND 1" in prompt:
            return json.dumps({"approved": False, "feedback": "Coder: add a timeout and exit code 1 on dead links."})
        return json.dumps({"approved": True, "feedback": "", "final": "CLI + README ready."})
    role = system.split("You are the ")[1].split(".")[0]
    return f"[{role} output for: {prompt.splitlines()[-1][:70]}]"


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
@dataclass
class Agent:
    name: str
    system: str

    async def run(self, task: str, context: str = "") -> str:
        prompt = f"Shared context:\n{context or '(none)'}\n\nYour task:\n{task}"
        return await call_llm(self.system, prompt)


WORKERS = {
    "researcher": Agent("researcher", "You are the researcher. Find facts, options and trade-offs. Be concise."),
    "coder": Agent("coder", "You are the coder. Write correct, minimal, runnable code."),
    "writer": Agent("writer", "You are the writer. Produce clear user-facing documentation."),
}

ORCHESTRATOR = Agent(
    "orchestrator",
    "You are the orchestrator. Split the goal into tasks for these agents: "
    f"{', '.join(WORKERS)}. Reply ONLY with a JSON list of "
    '{"agent": <name>, "task": <instruction>}. Incorporate reviewer feedback if given.',
)

REVIEWER = Agent(
    "reviewer",
    "You are the reviewer. Check the combined work against the goal. Reply ONLY with JSON: "
    '{"approved": bool, "feedback": str, "final": str (merged result if approved)}.',
)


# --------------------------------------------------------------------------- #
# Orchestration loop
# --------------------------------------------------------------------------- #
@dataclass
class Blackboard:
    """Shared state every agent can read: the goal, results so far, feedback."""
    goal: str
    results: dict = field(default_factory=dict)
    feedback: str = ""

    def as_context(self) -> str:
        parts = [f"GOAL: {self.goal}"]
        parts += [f"[{k}] {v}" for k, v in self.results.items()]
        if self.feedback:
            parts.append(f"REVIEWER FEEDBACK: {self.feedback}")
        return "\n".join(parts)


def parse_json(text: str):
    """Return the first complete JSON value in a reply.

    Tolerates ```json fences, prose before or after, and a second JSON block
    afterwards (small local models often add one).
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                value, _ = decoder.raw_decode(text, i)
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("No valid JSON found in model reply")


def normalize_plan(value) -> list:
    """Accept a list of steps, or a dict wrapping one ({"tasks": [...]}) or a single step."""
    if isinstance(value, dict):
        if "agent" in value and "task" in value:
            value = [value]
        else:
            value = next((v for v in value.values() if isinstance(v, list)), [])
    steps = [s for s in value if isinstance(s, dict) and s.get("agent") and s.get("task")]
    for s in steps:
        s["agent"] = str(s["agent"]).strip().lower()
        s["task"] = str(s["task"])
    steps = [s for s in steps if s["agent"] in WORKERS]
    if not steps:
        raise ValueError(f"Plan has no usable steps for agents {list(WORKERS)}")
    return steps


def normalize_verdict(value) -> dict:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        value = value[0]
    if not isinstance(value, dict) or "approved" not in value:
        raise ValueError('Verdict needs an "approved" field')
    approved = value["approved"]
    if isinstance(approved, str):
        approved = approved.strip().lower() in ("true", "yes", "approved")
    return {"approved": bool(approved), "feedback": str(value.get("feedback", "")),
            "final": value.get("final", "")}


async def ask_json(agent: "Agent", task: str, context: str, normalize, attempts: int = 3):
    """Ask an agent for JSON; if the reply can't be used, ask again with a stricter reminder."""
    reminder = ""
    for attempt in range(1, attempts + 1):
        reply = await agent.run(task + reminder, context)
        try:
            return normalize(parse_json(reply))
        except ValueError as exc:            # JSONDecodeError is a ValueError too
            print(f"    {agent.name}: unusable JSON ({exc}), asking again ({attempt}/{attempts})")
            reminder = ("\n\nIMPORTANT: your last reply could not be parsed. Reply with ONE valid "
                        "JSON value only: no explanation, no code fences, nothing after it.")
    raise RuntimeError(f"{agent.name} did not return usable JSON after {attempts} attempts:\n{reply[:1000]}")


async def run_team(goal: str) -> str:
    board = Blackboard(goal)

    for round_no in range(1, MAX_ROUNDS + 1):
        print(f"\n=== Round {round_no} ===")

        # 1. Plan
        plan = await ask_json(ORCHESTRATOR, goal, board.as_context(), normalize_plan)
        for step in plan:
            print(f"  plan -> {step['agent']:<10} {step['task']}")

        # 2. Execute in parallel (each worker sees the shared blackboard)
        context = board.as_context()
        outputs = await asyncio.gather(
            *(WORKERS[s["agent"]].run(s["task"], context) for s in plan if s["agent"] in WORKERS)
        )
        board.results = {}                    # each round replaces the previous round's work
        for n, (step, out) in enumerate(zip(plan, outputs), 1):
            # One entry per task, so several tasks for the same agent don't overwrite each other.
            board.results[f"{step['agent']} #{n}: {step['task'][:80]}"] = out
            print(f"  done <- {step['agent']} #{n}")

        # 3. Review
        verdict = await ask_json(REVIEWER, f"ROUND {round_no}: review the work.", board.as_context(), normalize_verdict)
        if verdict["approved"]:
            print("  reviewer: approved")
            save_result(board, verdict.get("final", ""), approved=True)
            return verdict.get("final") or board.as_context()
        board.feedback = verdict["feedback"]
        print(f"  reviewer: rejected -> {board.feedback}")

    save_result(board, "", approved=False)
    return "Stopped after max rounds. Last state:\n" + board.as_context()


RESULT_FILE = os.environ.get("RESULT_FILE", "final_result.md")


def save_result(board: "Blackboard", summary: str, approved: bool) -> None:
    """Write the reviewer's summary plus every task's full output to a Markdown file."""
    lines = [f"# {board.goal}", "", f"**Status:** {'approved' if approved else 'not approved (max rounds reached)'}", ""]
    if summary:
        lines += ["## Reviewer summary", "", str(summary), ""]
    if board.feedback and not approved:
        lines += ["## Last reviewer feedback", "", board.feedback, ""]
    for name, output in board.results.items():
        lines += [f"## {name}", "", str(output), ""]
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  saved full result to {RESULT_FILE}")


if __name__ == "__main__":
    # Windows terminals default to cp1252, which can't print emoji or many symbols
    # that models produce. Switch to UTF-8 and replace anything still unprintable.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    goal = " ".join(sys.argv[1:]) or "Build a CLI tool that checks if URLs in a README are alive"
    mode = "mock" if os.environ.get("AGENT_MOCK") == "1" else PROVIDER
    print(f"Provider: {mode}")

    if sys.platform == "win32":
        # Avoids the harmless but noisy "Event loop is closed" warnings on Windows.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        print("\nFINAL:\n" + asyncio.run(run_team(goal)))
    except RuntimeError as exc:
        if "did not return usable JSON" in str(exc):
            sys.exit(f"\n{exc}\n\nThe model kept producing invalid JSON. A larger model usually fixes this, "
                     "e.g.  ollama pull qwen2.5:7b  and  export OLLAMA_MODEL=qwen2.5:7b")
        if "All models failed" not in str(exc):
            raise
        sys.exit(
            f"\n{exc}\nThe models stayed overloaded. Try again later, or add a fallback, e.g.\n"
            '  export GEMINI_MODEL="gemini-3.8-flash,<another model from your key\'s list>"'
        )
