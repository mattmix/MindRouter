#!/usr/bin/env python3
"""Chat-capacity benchmark for MindRouter2 backends.

Simulates realistic multi-turn chat users against a single vLLM backend (or the
MindRouter gateway) and sweeps the number of concurrent users to find the
practical capacity limit for chat-shaped traffic on one GPU.

Traffic model
-------------
Each simulated user runs an endless loop of conversations. A conversation has a
sampled number of turns (geometric, mean ~6.5, capped). Each turn sends the full
message history (system prompt + prior turns) as a streaming chat completion,
reads the response, then "thinks" (reading time proportional to response length
plus a lognormal compose delay) before the next turn. On a failed turn the user
"hits regenerate" once (short jittered backoff, identical request) before
abandoning the conversation; every attempt is recorded as offered load.

Prompt content is generated from slot-filled templates (names, places, numbers,
error strings, ...) and each conversation's first message carries a unique
leading salt tag, so no two conversations share a prompt prefix beyond the
system prompt — cross-conversation prefix-cache hits stay realistic instead of
being inflated by repeated prompt text. Within-conversation caching (history
resent every turn) is preserved: that is the real chat KV-reuse pattern. Opener
lengths follow a four-way mixture to exercise prefill across the realistic
range: short asks (~15-80 tok), medium context (~80-350 tok), long pastes
(300-1500 tok, --long-paste-pct), and XL pastes (1500-6000 tok, --xl-paste-pct);
follow-ups occasionally paste new material mid-conversation (--mid-paste-pct).
Per-user RNG seeds mix in the stage and repeat indices, so successive stages do
not replay identical conversations into a warm prefix cache (the harness also
best-effort POSTs /reset_prefix_cache between stages in direct mode).
--cache-adversarial additionally salts the SYSTEM prompt per conversation,
removing CROSS-conversation prefix sharing (shared system prompt). Note it
does NOT remove within-conversation caching — turn N's history is still
cached from turn N-1 of the same conversation, which is inherent to chat and
cannot be defeated client-side. A true zero-cache baseline requires
restarting the backend with prefix caching disabled. (Measured 2026-08-01 on
gemma-4-31b: cross-conversation sharing was worth ~nothing — realistic and
cache-adversarial sweeps produced identical capacity.)

Duty cycle and sweep sizing: with default think times each user keeps a request
in flight only ~15-25% of the time, so N users produce roughly N/5 concurrent
streams. Against --max-num-seqs 48 the knee is likely in the 150-400 user
range; the sweep auto-extends past --users (x1.5 steps, then a bisection to
bracket the knee) until an SLO fails or --max-users is reached. Use --no-adapt
for a fixed, exactly reproducible stage list, and --no-think-time for a pure
saturation mode where users == concurrent streams.

Measured per turn (client side)
-------------------------------
- TTFT: time to first streamed delta of any kind (ttft_any) and to first
  *visible* content delta (ttft_visible)
- Time to full response (e2e), decode time, effective stream tokens/sec
  (first-chunk token bundle excluded from the rate)
- Chunk cadence: inter-chunk gap p50/p95/max, stalls > 1s / > 2s
- Token usage from the backend's usage chunk (stream_options.include_usage)
- Status: ok | http_error | stream_error | stall (mid-stream, after first
  delta) | ttft_timeout (no first delta within --turn-timeout) | timeout |
  cancelled (harness drain), plus retry_idx for regenerate attempts

Turns killed while still waiting for a first token are kept as right-censored
TTFT observations (their observed wait is a lower bound) and folded into the
TTFT percentiles; a '+' next to ttft p95 in the table marks censoring. Failed
and drain-cancelled turns count in the error-rate denominator.

Sampled from the backend every --metrics-interval seconds (vLLM /metrics)
-------------------------------------------------------------------------
num_requests_running/waiting, kv_cache_usage_perc, preemptions, prompt/
generation token counters, prefix-cache and spec-decode counters, and the
server-side TTFT / inter-token-latency / queue-time histograms. Boundary
snapshots are taken exactly at warmup end and stage end, so per-stage server
deltas cover the same window as the client statistics. The per-stage summary
also aggregates the sampled gauges (mean/max running, waiting, KV usage) to
give the users -> concurrent-streams -> queue-depth mapping.

SLO gates (all configurable): TTFT p95 <= --slo-ttft-p95 (uncensored),
stream tok/s p10 >= --slo-tps-p10, error rate <= --slo-err-pct, turns with a
>2s stall <= --slo-stall-pct, e2e p95 <= --slo-e2e-p95 (default derived:
ttft + max_tokens / tps-floor). Capacity is reported as the longest passing
PREFIX of user counts; non-monotone pass/fail triggers a warning.

Usage
-----
Direct against one backend (run from a host that can reach the node, e.g. the
mindrouter prod host / app container):

  python chat_bench.py --base-url https://aspen2.hpc.uidaho.edu:8000 \\
      --model google/gemma-4-31b --users 1,4,16,48,96 \\
      --stage-duration 300 --outdir bench_results/aspen2gpu0

Through the gateway (adds MindRouter scheduling/queueing; point --metrics-url
at the backend node so the server-side cross-check still works — valid only if
the gateway routes this model exclusively to that backend):

  python chat_bench.py --mode gateway --base-url https://mindrouter.uidaho.edu \\
      --api-key mr2_... --model google/gemma-4-31b --users 8,16,32 \\
      --metrics-url https://aspen2.hpc.uidaho.edu:8000/metrics

Quick smoke run:

  python chat_bench.py --base-url https://aspen2.hpc.uidaho.edu:8000 \\
      --model google/gemma-4-31b --users 2 --stage-duration 60 --warmup 10 \\
      --min-turns 5 --no-adapt
"""

import argparse
import asyncio
import json
import math
import random
import re
import signal
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("chat_bench.py requires httpx (pip install httpx)")


# ---------------------------------------------------------------------------
# Prompt material
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable assistant. Answer clearly and concisely, "
    "using markdown formatting when it helps readability. If a question is "
    "ambiguous, briefly state your assumption and answer anyway."
)

# ---- Slot-filled templates -------------------------------------------------
# Every prompt is generated from a template whose {slots} are filled per use
# from the banks below (plus random numbers), so no two conversations share a
# prompt prefix beyond the system prompt. That keeps cross-conversation
# prefix-cache hits realistic; a fixed prompt bank would let vLLM serve most
# first-turn prefills from cache and flatter TTFT.

SLOT_BANKS = {
    "name": ["Jordan", "Sam", "Priya", "Alex", "Taylor", "Miguel", "Morgan",
             "Riley", "Chen", "Casey", "Jamie", "Avery", "Quinn", "Dana",
             "Fatima", "Reese", "Kai", "Elena", "Marcus", "Ingrid"],
    "city": ["Boise", "Spokane", "Portland", "Missoula", "Salt Lake City",
             "Seattle", "Coeur d'Alene", "Bozeman", "Reno", "Bend",
             "Pocatello", "Tacoma"],
    "month": ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"],
    "dept": ["facilities", "enrollment services", "the registrar's office",
             "IT operations", "advancement", "the library", "campus safety",
             "dining services", "the grants office", "human resources"],
    "system": ["the advising portal", "the grants database",
               "the HR onboarding system", "the ticketing queue",
               "the learning management system", "the room-scheduling tool",
               "the alumni CRM", "the payroll system"],
    "concept": ["vaccines", "inflation", "photosynthesis", "compound interest",
                "machine learning", "plate tectonics", "the electoral college",
                "antibiotic resistance", "supply and demand", "DNS",
                "confidence intervals", "blockchain", "the water cycle",
                "herd immunity", "recursion"],
    "food": ["chicken thighs", "ground turkey", "tofu", "black beans", "salmon",
             "pork shoulder", "lentils", "mushrooms", "sweet potatoes",
             "canned chickpeas", "frozen shrimp", "leftover rice"],
    "language": ["Python", "JavaScript", "Go", "Rust", "SQL", "R", "Bash"],
    "error": ["KeyError: user_id", "a segmentation fault",
              "TypeError: 'NoneType' object is not iterable",
              "a 502 from the load balancer", "UnicodeDecodeError on line 1",
              "a deadlock during the nightly batch job",
              "ConnectionResetError mid-request", "an off-by-one in pagination"],
    "fixture": ["kitchen faucet", "bathroom fan", "garbage disposal",
                "front-door lock", "water heater", "dishwasher"],
    "animal": ["dog", "cat", "parrot", "goat", "hedgehog", "raccoon"],
    "job": ["nurse", "high-school teacher", "civil engineer", "barista",
            "data analyst", "electrician", "grant writer", "park ranger"],
    "hobby": ["rock climbing", "sourdough baking", "fly fishing", "birding",
              "woodworking", "trail running", "watercolor painting"],
    "park": ["Glacier National Park", "Yellowstone", "the Sawtooths",
             "Craters of the Moon", "the Olympic Peninsula", "Zion"],
}

SLOT_RE = re.compile(r"\{([a-z_]+)(?::(\d+)-(\d+))?\}")


def fill_template(template: str, rng: random.Random) -> str:
    """Fill {bank} slots and {num:LO-HI} / {money:LO-HI} / {pct:LO-HI} ranges."""
    def repl(m):
        key, lo, hi = m.group(1), m.group(2), m.group(3)
        if lo is not None:
            n = rng.randint(int(lo), int(hi))
            if key == "money":
                return f"${n}"
            if key == "pct":
                return f"{n}%"
            return str(n)
        return rng.choice(SLOT_BANKS[key])
    return SLOT_RE.sub(repl, template)


# Short conversation openers (~15-80 tokens once filled). Every template has
# at least two independent slots so byte-identical openers are rare; a style
# suffix is appended to half of them for further variant space.
SHORT_OPENERS = [
    "What's a good way to explain how {concept} works to a {num:8-15}-year-old?",
    "I have {food}, rice, and {food} in my fridge. What can I make for dinner in under {num:20-60} minutes?",
    "Write a short, friendly email to my landlord {name} asking them to fix the {fixture} that has been broken for {num:1-6} weeks.",
    "Explain the difference between a 401k and a Roth IRA for a {num:22-45}-year-old {job}. Which makes more sense?",
    "My {language} script throws '{error}' when I process a CSV export of {num:2-90}k rows. What are the usual causes and how do I debug it?",
    "Plan a {num:3-7}-day trip to {park} in {month}. We like moderate hikes and want one rest day.",
    "My coworker {name} and I disagree about what {concept} actually means. Can you settle it with a clear explanation and {num:2-5} examples?",
    "Write a SQL query that finds the top {num:3-10} customers by total order value in the last {num:30-180} days. Tables: customers(id, name), orders(id, customer_id, total, created_at).",
    "What are some realistic ways to cut my monthly grocery bill by about {money:40-200} without spending hours meal planning?",
    "I'm giving a {num:3-10}-minute talk to {dept} about why {concept} matters. Outline it and suggest a strong opening line.",
    "Help me write a polite but firm reply to {name}, a coworker who has scheduled meetings over my lunch break {num:2-6} times this month.",
    "Give me a beginner strength-training plan I can do in {num:30-60} minutes, {num:2-4} times a week, with just dumbbells.",
    "I need a regex in {language} that matches US phone numbers like {num:200-999}-555-{num:1000-9999} with or without parentheses and dashes. Explain it piece by piece.",
    "Suggest {num:6-15} names for a coffee shop in {city} near a university campus that's also a used bookstore. Explain your two favorites.",
    "My tomato plants here in {city} have lower leaves turning yellow with brown spots after {num:2-6} weeks. What's likely wrong and what should I do?",
    "I'm helping {name} prep for a {num:20-60}-minute seminar on the fall of the Roman Empire. What were the main causes, and is there a modern consensus?",
    "Write an Excel formula that averages column B only for rows where column A says 'Complete' and column C is after {month} {num:1-28}.",
    "Compare renting vs buying a house in {city} for a {job} who might move in {num:2-6} years. Give me a framework, not just an answer.",
    "Write a {num:8-20}-line rhyming poem about a {animal} who is convinced the mail carrier is a wizard.",
    "How should I structure a study plan for a certification exam in {num:4-10} weeks, studying about {num:30-90} minutes a day?",
    "How much protein does a {num:20-55}-year-old lifting weights {num:2-5} times a week actually need per day?",
    "Explain {language} decorators with a practical example a {job} would actually use at work.",
    "I'm writing a fantasy short story set in a city like {city}. Give me {num:2-5} ideas for a magic system with a real cost or limitation built in.",
    "My {num:5-14}-year-old wants to take up {hobby}. What's a sensible way to start without spending more than {money:50-400}?",
    "Draft a text message to {name} apologizing for missing their {hobby} event and proposing a new time in {month}.",
    "What happens economically when a central bank raises rates by {num:1-3} percentage points? What does it mean for someone buying a house in {city}?",
]

# Appended to ~half of short openers: realistic style constraints that also
# multiply the variant space so byte-identical prompts stay rare.
STYLE_SUFFIXES = [
    "Keep it under {num:80-400} words.",
    "Use plain, non-technical language.",
    "Answer in {num:3-8} bullet points.",
    "Include a short example.",
    "Assume I'm a complete beginner.",
    "Be direct — skip the preamble.",
    "Format the answer as a table if that helps.",
    "State the key assumptions you're making.",
]

# Medium openers: an ask plus a generated context paragraph (~80-350 tokens).
MEDIUM_ASKS = [
    "Here's the situation at work — read it and tell me the three things I should prioritize this month, with reasoning:",
    "Below are notes about a project I inherited from {name}. What risks jump out, and what would you clarify first?",
    "I pasted some background about our team below. Draft a short plan for the next {num:2-8} weeks based on it:",
    "Given the context below, write a status-update email to {dept} that is honest but not alarming:",
    "Read the background below and give me a decision framework — should we proceed, delay, or cancel?",
]

# Follow-up asks that paste NEW material mid-conversation.
MID_PASTE_ASKS = [
    "Here's an updated version of the notes — summarize what actually changed:",
    "I just got this additional document. Does it change your recommendation?",
    "Here's the raw log from {system} — does it support what you said above?",
]

# Instructions that precede long/XL synthetic documents.
PASTE_INSTRUCTIONS = [
    "Summarize the following meeting notes into key decisions and action items:",
    "Here are some meeting notes. Turn them into a brief status update email for people who missed the meeting:",
    "Read these notes and list any risks, open questions, and deadlines you can find:",
    "Condense the following into a one-paragraph executive summary and a bulleted list of follow-ups:",
]

FOLLOWUPS = [
    "Can you make that shorter and punchier?",
    "Give me three concrete examples.",
    "Explain that like I'm in high school.",
    "What are the main risks or downsides I should watch out for?",
    "Turn that into a bulleted list I can copy into my notes.",
    "Now rewrite it in a more formal tone.",
    "Why is that true? Walk me through the reasoning step by step.",
    "Summarize your answer in two sentences.",
    "Give me the strongest counterargument to what you just said.",
    "Can you show the same thing in Python?",
    "What would you change if I had half the budget?",
    "Make a table comparing the main options.",
    "What did you assume that might not actually hold in my case?",
    "Give me a step-by-step checklist version.",
    "What's a simpler alternative if that turns out to be too much?",
    "Add error handling and comments to that code.",
    "Can you continue where you left off and go deeper on the last point?",
    "Rewrite that for a technical audience.",
    "What should I read or watch next to learn more?",
    "Thanks — one last thing: what's the single most important takeaway?",
]

# Slotted sentence templates used to synthesize documents of arbitrary length.
# Every sentence is re-filled on each use (names, departments, figures), so a
# 6000-token paste never shares a long prefix with any other paste.
DOC_SENTENCES = [
    "The committee reviewed the {month} figures and noted a {pct:2-19} decline in throughput compared with the prior period.",
    "{dept} reported that the {fixture} replacement in the west wing is scheduled for the {num:1-4}th week of {month}, pending contractor availability.",
    "The budget subcommittee proposed reallocating {money:20-400},000 from deferred maintenance to upgrades in {city}.",
    "{name} raised concerns about the rollout timeline for {system}, citing unresolved data-migration issues.",
    "A survey presented by {name} showed {pct:40-92} of respondents were satisfied or very satisfied with current service hours.",
    "Action item: {name} will draft a revised staffing plan for {dept} covering evenings and weekends by the {num:5-28}th.",
    "A motion to approve the updated travel reimbursement policy passed with {num:0-4} abstentions.",
    "{name} reported that the single-sign-on migration is complete for all core systems except {system}.",
    "Discussion of the vendor contract renewal was tabled until legal review concludes, expected within {num:1-6} weeks.",
    "The chair reminded members that annual disclosures are due by the {num:10-28}th of {month}.",
    "Preliminary data suggest the pilot program improved completion rates by roughly {num:2-11} percentage points.",
    "Concerns were raised about parking during construction, and {name} will evaluate a shuttle option costing {money:8-60},000.",
    "{dept} will circulate a draft of the announcement for feedback before the end of {month}.",
    "Budget projections assume a {pct:1-6} utility cost increase and flat state funding for the next fiscal year.",
    "The subcommittee recommended piloting the new process in {dept} before a wider rollout in {month}.",
    "Follow-up: procurement will compare the {num:2-4} remaining vendors, including {num:3-7}-year total cost of ownership.",
    "Attendance at the open house in {city} exceeded projections, with {num:150-900} registered visitors versus {num:100-500} last year.",
    "{dept} reported {num:0-3} lost-time incidents for the quarter and highlighted completion of safety training by {pct:60-100} of staff.",
    "Members discussed whether the office-space policy needs clarification and {name} agreed to survey affected groups.",
    "The next regular meeting is scheduled for the {num:1-4}th Tuesday of {month} in the {city} conference room.",
    "An incident affecting {system} on the {num:1-28}th caused roughly {num:1-9} hours of degraded service before {name} restored it.",
    "{name} from {dept} presented a proposal to consolidate {num:2-8} legacy tools into {system} by {month}.",
    "Ticket volume for {system} rose {pct:5-60} after the {month} update, according to {name}.",
    "The reserve fund stands at {money:100-900},000, roughly {pct:4-25} above the policy minimum.",
]


def synth_doc(rng: random.Random, target_tokens: int) -> str:
    """Generate a unique document of ~target_tokens (chars/4 heuristic)."""
    target_chars = target_tokens * 4
    parts, total = [], 0
    while total < target_chars:
        s = fill_template(rng.choice(DOC_SENTENCES), rng)
        parts.append(s)
        total += len(s) + 1
    return " ".join(parts)


# ---------------------------------------------------------------------------
# User behavior model
# ---------------------------------------------------------------------------

class UserBehavior:
    """Seeded per-user random behavior: messages, conversation length, think time."""

    def __init__(self, seed: int, cfg: "Config"):
        self.rng = random.Random(seed)
        self.cfg = cfg

    def conversation_turns(self) -> int:
        # 1 + geometric: mean ~ 1 + (1-p)/p; p=0.18 -> mean ~5.6 extra turns
        n = 1
        while n < self.cfg.max_turns and self.rng.random() > 0.18:
            n += 1
        return n

    def opener(self) -> tuple:
        """Return (text, category) drawn from the length mixture."""
        rng = self.rng
        r = rng.random()
        if r < self.cfg.xl_paste_pct:
            doc = synth_doc(rng, rng.randint(1500, 6000))
            return (fill_template(rng.choice(PASTE_INSTRUCTIONS), rng)
                    + "\n\n" + doc, "xl_paste")
        if r < self.cfg.xl_paste_pct + self.cfg.long_paste_pct:
            doc = synth_doc(rng, rng.randint(300, 1500))
            return (fill_template(rng.choice(PASTE_INSTRUCTIONS), rng)
                    + "\n\n" + doc, "long_paste")
        if r < self.cfg.xl_paste_pct + self.cfg.long_paste_pct + self.cfg.medium_pct:
            ask = fill_template(rng.choice(MEDIUM_ASKS), rng)
            return (ask + "\n\n" + synth_doc(rng, rng.randint(80, 350)), "medium")
        text = fill_template(rng.choice(SHORT_OPENERS), rng)
        if rng.random() < 0.5:
            text += " " + fill_template(rng.choice(STYLE_SUFFIXES), rng)
        return text, "short"

    def followup(self) -> tuple:
        """Return (text, category); occasionally pastes new material."""
        rng = self.rng
        if rng.random() < self.cfg.mid_paste_pct:
            ask = fill_template(rng.choice(MID_PASTE_ASKS), rng)
            return (ask + "\n\n" + synth_doc(rng, rng.randint(200, 800)),
                    "mid_paste")
        return rng.choice(FOLLOWUPS), "followup"

    def system_prompt(self) -> str:
        if self.cfg.cache_adversarial:
            # unique early tokens defeat ALL prefix caching (worst-case prefill)
            return f"[session {self.rng.getrandbits(64):016x}] " + SYSTEM_PROMPT
        return SYSTEM_PROMPT

    def think_seconds(self, completion_tokens: int) -> float:
        if self.cfg.no_think_time:
            return 0.0
        reading = min(completion_tokens / self.cfg.reading_tps, self.cfg.reading_cap)
        # lognormal compose delay with configured median
        mu = math.log(self.cfg.compose_median)
        compose = min(self.rng.lognormvariate(mu, self.cfg.compose_sigma),
                      self.cfg.compose_cap)
        return reading + compose


# ---------------------------------------------------------------------------
# Config and per-turn records
# ---------------------------------------------------------------------------

@dataclass
class Config:
    base_url: str
    mode: str
    model: str
    api_key: str
    users_stages: list
    stage_duration: float
    stage_max_duration: float
    min_turns: int
    warmup: float
    drain_timeout: float
    cooldown: float
    repeats: int
    adapt: bool
    max_users: int
    max_tokens: int
    temperature: float
    top_p: float
    think: str
    reading_tps: float
    reading_cap: float
    compose_median: float
    compose_sigma: float
    compose_cap: float
    long_paste_pct: float
    xl_paste_pct: float
    medium_pct: float
    mid_paste_pct: float
    cache_adversarial: bool
    max_turns: int
    no_think_time: bool
    stall_timeout: float
    turn_timeout: float
    metrics_url: str
    metrics_interval: float
    slo_ttft_p95: float
    slo_tps_p10: float
    slo_err_pct: float
    slo_stall_pct: float
    slo_e2e_p95: float
    outdir: str
    seed: int
    label: str
    verify_tls: bool


def redacted_config(cfg: "Config") -> dict:
    d = asdict(cfg)
    if d.get("api_key"):
        d["api_key"] = d["api_key"][:8] + "...redacted"
    return d


@dataclass
class TurnRecord:
    stage_users: int
    user_id: int
    conv_id: int
    turn_idx: int
    retry_idx: int
    n_messages: int
    prompt_chars: int
    category: str
    t_start_wall: str = ""
    ttfb_s: float = None          # first byte of the HTTP response body
    ttft_any_s: float = None      # first SSE delta of any kind
    ttft_visible_s: float = None  # first non-empty content delta
    e2e_s: float = None
    decode_s: float = None        # last chunk - first delta
    n_chunks: int = 0
    gap_p50_ms: float = None
    gap_p95_ms: float = None
    gap_max_ms: float = None
    stalls_gt1s: int = 0
    stalls_gt2s: int = 0
    prompt_tokens: int = None
    completion_tokens: int = None
    reasoning_chars: int = 0
    content_chars: int = 0
    stream_tps: float = None      # tokens/sec excluding the first chunk bundle
    finish_reason: str = None
    # ok | http_error | stream_error | stall | ttft_timeout | timeout | cancelled
    status: str = "ok"
    error: str = None
    in_warmup: bool = False


# ---------------------------------------------------------------------------
# SSE streaming turn
# ---------------------------------------------------------------------------

async def run_turn(client: httpx.AsyncClient, cfg: Config, messages: list,
                   rec: TurnRecord) -> str:
    """Send one streaming chat completion; fill rec; return assistant text."""
    body = {
        "model": cfg.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
    }
    if cfg.think == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif cfg.think == "on":
        body["chat_template_kwargs"] = {"enable_thinking": True}

    headers = {"Content-Type": "application/json"}
    if cfg.mode == "gateway":
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    url = cfg.base_url.rstrip("/") + "/v1/chat/completions"
    rec.t_start_wall = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    content_parts = []
    chunk_times = []
    t_first_delta = None
    usage = None
    saw_done = False

    try:
        async with asyncio.timeout(cfg.turn_timeout):
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread())[:500]
                    rec.status = "http_error"
                    rec.error = f"HTTP {resp.status_code}: {text.decode(errors='replace')}"
                    rec.e2e_s = time.monotonic() - t0
                    return ""
                byte_iter = resp.aiter_bytes()
                buf = b""
                while True:
                    # First-token wait is bounded only by turn_timeout (queue
                    # time legitimately lands here); the stall budget applies
                    # only after the first delta has arrived. The inner
                    # asyncio.timeout converts to TimeoutError only when ITS
                    # deadline expired, so it cannot swallow the outer timeout.
                    try:
                        if t_first_delta is None:
                            raw = await anext(byte_iter)
                        else:
                            async with asyncio.timeout(cfg.stall_timeout):
                                raw = await anext(byte_iter)
                    except StopAsyncIteration:
                        break
                    except TimeoutError:
                        rec.status = "stall"
                        rec.error = (f"no bytes for {cfg.stall_timeout}s "
                                     f"mid-stream after {rec.n_chunks} chunks")
                        rec.e2e_s = time.monotonic() - t0
                        return ""
                    now = time.monotonic()
                    if rec.ttfb_s is None:
                        rec.ttfb_s = now - t0
                    buf += raw
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line or not line.startswith(b"data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == b"[DONE]":
                            saw_done = True
                            continue
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if "error" in chunk:  # gateway emits errors as SSE events
                            rec.status = "stream_error"
                            rec.error = json.dumps(chunk["error"])[:500]
                            rec.e2e_s = now - t0
                            return ""
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        fr = choices[0].get("finish_reason")
                        if fr:
                            rec.finish_reason = fr
                        content = delta.get("content")
                        reasoning = delta.get("reasoning_content")
                        if content or reasoning:
                            if t_first_delta is None:
                                t_first_delta = now
                                rec.ttft_any_s = now - t0
                            chunk_times.append(now)
                            rec.n_chunks += 1
                        if reasoning:
                            rec.reasoning_chars += len(reasoning)
                        if content:
                            if rec.ttft_visible_s is None:
                                rec.ttft_visible_s = now - t0
                            rec.content_chars += len(content)
                            content_parts.append(content)
                # stream ended: a clean end carries finish_reason and/or [DONE]
                if rec.status == "ok" and not saw_done and rec.finish_reason is None:
                    rec.status = "stream_error"
                    rec.error = "stream ended without finish_reason or [DONE]"
    except asyncio.TimeoutError:
        if t_first_delta is None:
            rec.status = "ttft_timeout"
            rec.error = f"no first delta within {cfg.turn_timeout}s (censored TTFT)"
        else:
            rec.status = "timeout"
            rec.error = f"turn exceeded {cfg.turn_timeout}s"
    except httpx.HTTPError as e:
        rec.status = "stream_error"
        rec.error = f"{type(e).__name__}: {e}"
    except asyncio.CancelledError:
        rec.status = "cancelled"
        raise
    finally:
        t_end = time.monotonic()
        if rec.e2e_s is None:
            rec.e2e_s = t_end - t0
        if t_first_delta is not None and chunk_times:
            rec.decode_s = chunk_times[-1] - t_first_delta
        if len(chunk_times) >= 2:
            gaps = [(b - a) * 1000.0 for a, b in zip(chunk_times, chunk_times[1:])]
            gaps.sort()
            rec.gap_p50_ms = round(percentile(gaps, 50), 1)
            rec.gap_p95_ms = round(percentile(gaps, 95), 1)
            rec.gap_max_ms = round(gaps[-1], 1)
            rec.stalls_gt1s = sum(1 for g in gaps if g > 1000.0)
            rec.stalls_gt2s = sum(1 for g in gaps if g > 2000.0)
        if usage:
            rec.prompt_tokens = usage.get("prompt_tokens")
            rec.completion_tokens = usage.get("completion_tokens")
            # decode_s spans chunk 2..N, so exclude the first chunk's token
            # bundle (approximated as the mean bundle) from the rate
            if (rec.completion_tokens and rec.decode_s and rec.decode_s > 0
                    and rec.n_chunks > 1):
                effective = rec.completion_tokens * (1 - 1 / rec.n_chunks)
                rec.stream_tps = round(effective / rec.decode_s, 2)

    return "".join(content_parts)


# ---------------------------------------------------------------------------
# Simulated user
# ---------------------------------------------------------------------------

class Stage:
    def __init__(self, users: int):
        self.users = users
        self.stop_event = asyncio.Event()   # set at stage end: no new turns

    @property
    def stopping(self):
        return self.stop_event.is_set()

    async def interruptible_sleep(self, seconds: float):
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass


async def user_task(user_id: int, stage_idx: int, repeat_idx: int, cfg: Config,
                    stage: Stage, warmup_deadline: float,
                    client: httpx.AsyncClient, sink, conv_counter):
    # seed mixes stage and repeat indices so successive stages/repeats generate
    # different conversations instead of replaying identical ones into a warm
    # prefix cache (int-tuple hash is deterministic across runs)
    behavior = UserBehavior(
        hash((cfg.seed, stage_idx, repeat_idx, user_id)) & 0x7FFFFFFF, cfg)
    # Stagger arrivals across most of the warmup window (scaled by user count)
    # so no stage begins with a synchronized "everyone hits submit" herd of
    # first-turn prefills; by the measured window users are desynchronized by
    # their own response-length and think-time variance.
    ramp = min(0.8 * cfg.warmup, max(10.0, 0.5 * stage.users))
    await stage.interruptible_sleep(behavior.rng.uniform(0, ramp))

    while not stage.stopping:
        conv_id = next(conv_counter)
        n_turns = behavior.conversation_turns()
        messages = [{"role": "system", "content": behavior.system_prompt()}]
        # a unique LEADING salt on the first user message breaks cross-
        # conversation prefix sharing of openers/pastes while leaving the
        # (realistically shared) system prompt and the conversation's own
        # history fully cacheable
        conv_salt = f"[conv {user_id}-{conv_id}-{behavior.rng.getrandbits(48):012x}] "
        for turn_idx in range(n_turns):
            if stage.stopping:
                return
            is_first = turn_idx == 0
            msg, category = behavior.opener() if is_first else behavior.followup()
            if is_first:
                msg = conv_salt + msg
            messages.append({"role": "user", "content": msg})
            rec, text = None, ""
            # a failed turn is retried once with the identical request — the
            # "regenerate" a real user would hit — so offered load does not
            # quietly lighten at exactly the overload stages
            for attempt in range(2):
                rec = TurnRecord(
                    stage_users=stage.users, user_id=user_id, conv_id=conv_id,
                    turn_idx=turn_idx, retry_idx=attempt,
                    n_messages=len(messages),
                    prompt_chars=sum(len(m["content"]) for m in messages),
                    category=category,
                    in_warmup=time.monotonic() < warmup_deadline,
                )
                try:
                    text = await run_turn(client, cfg, messages, rec)
                finally:
                    sink(rec)
                if rec.status == "ok" and text:
                    break
                if rec.status == "cancelled" or stage.stopping:
                    return
                await stage.interruptible_sleep(
                    2.0 * (attempt + 1) + behavior.rng.uniform(0, 2))
            if rec.status != "ok" or not text:
                break  # abandon conversation after the failed retry
            messages.append({"role": "assistant", "content": text})
            await stage.interruptible_sleep(
                behavior.think_seconds(rec.completion_tokens or len(text) // 4))
        # pause between conversations
        await stage.interruptible_sleep(behavior.think_seconds(0))


# ---------------------------------------------------------------------------
# vLLM /metrics scraping
# ---------------------------------------------------------------------------

GAUGES = ["num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"]
COUNTERS = [
    "prompt_tokens_total", "generation_tokens_total", "num_preemptions_total",
    "prefix_cache_queries_total", "prefix_cache_hits_total",
    "spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total", "request_success_total",
]
HISTS = [
    "time_to_first_token_seconds", "inter_token_latency_seconds",
    "request_queue_time_seconds", "request_prefill_time_seconds",
    "request_decode_time_seconds", "e2e_request_latency_seconds",
]

METRIC_RE = re.compile(r'^vllm:([a-z0-9_]+)(?:\{([^}]*)\})?\s+([0-9.eE+-]+)\s*$')


def parse_vllm_metrics(text: str) -> dict:
    """Parse selected vllm metrics; accumulate across label sets."""
    out = {"gauges": {}, "counters": {}, "hists": {}}
    for line in text.splitlines():
        m = METRIC_RE.match(line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
        if name in GAUGES:
            if name == "kv_cache_usage_perc":
                # a fraction, not a count: take max across label sets
                out["gauges"][name] = max(out["gauges"].get(name, 0.0), value)
            else:
                out["gauges"][name] = out["gauges"].get(name, 0.0) + value
        elif name in COUNTERS:
            out["counters"][name] = out["counters"].get(name, 0.0) + value
        else:
            for h in HISTS:
                if name == h + "_sum":
                    d = out["hists"].setdefault(h, {})
                    d["sum"] = d.get("sum", 0.0) + value
                    break
                if name == h + "_count":
                    d = out["hists"].setdefault(h, {})
                    d["count"] = d.get("count", 0.0) + value
                    break
                if name == h + "_bucket":
                    le = re.search(r'le="([^"]+)"', labels)
                    if le:
                        b = out["hists"].setdefault(h, {}).setdefault("buckets", {})
                        b[le.group(1)] = b.get(le.group(1), 0.0) + value
                    break
    return out


async def scrape_metrics(cfg: Config) -> dict:
    """One synchronous /metrics snapshot (used for exact stage boundaries)."""
    if not cfg.metrics_url:
        return None
    try:
        async with httpx.AsyncClient(verify=cfg.verify_tls, timeout=10.0) as c:
            resp = await c.get(cfg.metrics_url)
            if resp.status_code != 200:
                return None
            sample = parse_vllm_metrics(resp.text)
            sample["ts"] = datetime.now(timezone.utc).isoformat()
            sample["mono"] = time.monotonic()
            return sample
    except httpx.HTTPError:
        return None


async def sampler_task(cfg: Config, samples_path: Path, stop: asyncio.Event,
                       state: dict):
    if not cfg.metrics_url:
        return
    async with httpx.AsyncClient(verify=cfg.verify_tls, timeout=10.0) as client:
        with samples_path.open("a") as f:
            while not stop.is_set():
                t0 = time.monotonic()
                try:
                    resp = await client.get(cfg.metrics_url)
                    if resp.status_code == 200:
                        sample = parse_vllm_metrics(resp.text)
                        sample["ts"] = datetime.now(timezone.utc).isoformat()
                        sample["mono"] = time.monotonic()
                        state["latest"] = sample
                        state.setdefault("samples", []).append(sample)
                        f.write(json.dumps(sample) + "\n")
                        f.flush()
                except httpx.HTTPError:
                    pass
                delay = cfg.metrics_interval - (time.monotonic() - t0)
                if delay > 0:
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=delay)
                    except asyncio.TimeoutError:
                        pass


async def wait_for_idle(cfg: Config, state: dict, max_wait: float = 90.0):
    """Between stages, wait until the backend reports no running/waiting requests."""
    if not cfg.metrics_url:
        await asyncio.sleep(cfg.cooldown)
        return
    deadline = time.monotonic() + max_wait
    idle = False
    while time.monotonic() < deadline:
        g = (state.get("latest") or {}).get("gauges", {})
        if g.get("num_requests_running", 0) == 0 and g.get("num_requests_waiting", 0) == 0:
            idle = True
            break
        await asyncio.sleep(2.0)
    if not idle:
        print("  [warn] backend did not go idle before next stage", flush=True)
    if cfg.mode == "direct":
        # best-effort: clear the prefix cache so stages are fully independent
        try:
            async with httpx.AsyncClient(verify=cfg.verify_tls, timeout=10.0) as c:
                await c.post(cfg.base_url.rstrip("/") + "/reset_prefix_cache")
        except httpx.HTTPError:
            pass
    await asyncio.sleep(cfg.cooldown)


# ---------------------------------------------------------------------------
# Statistics / summary
# ---------------------------------------------------------------------------

def percentile(sorted_vals: list, p: float):
    if not sorted_vals:
        return None
    k = max(0, min(len(sorted_vals) - 1,
                   math.ceil(p / 100.0 * len(sorted_vals)) - 1))
    return sorted_vals[k]


def pctl_field(recs, attr, p):
    vals = sorted(v for r in recs if (v := getattr(r, attr)) is not None)
    v = percentile(vals, p)
    return round(v, 3) if v is not None else None


def hist_delta_mean(start: dict, end: dict, name: str):
    try:
        ds = end["hists"][name]["sum"] - start["hists"][name]["sum"]
        dc = end["hists"][name]["count"] - start["hists"][name]["count"]
        return round(ds / dc, 4) if dc > 0 else None
    except (KeyError, TypeError):
        return None


def hist_delta_pctl(start: dict, end: dict, name: str, p: float):
    """Approximate percentile from Prometheus bucket deltas (interpolated)."""
    try:
        sb = start["hists"][name]["buckets"]
        eb = end["hists"][name]["buckets"]
    except (KeyError, TypeError):
        return None
    deltas = []
    for le, v in eb.items():
        bound = math.inf if le in ("+Inf", "inf") else float(le)
        deltas.append((bound, v - sb.get(le, 0.0)))
    deltas.sort()
    if not deltas or deltas[-1][1] <= 0:
        return None
    total = deltas[-1][1]
    target = total * p / 100.0
    prev_le, prev_c = 0.0, 0.0
    for le, c in deltas:
        if c >= target:
            if le == math.inf:
                return round(prev_le, 4)
            frac = (target - prev_c) / max(c - prev_c, 1e-9)
            return round(prev_le + (le - prev_le) * frac, 4)
        prev_le, prev_c = le, c
    return None


def counter_delta(start: dict, end: dict, name: str):
    try:
        return end["counters"][name] - start["counters"][name]
    except (KeyError, TypeError):
        return None


def summarize_stage(users: int, repeat_idx: int, recs: list, window_s: float,
                    m_start: dict, m_end: dict, win_samples: list,
                    cfg: Config) -> dict:
    measured = [r for r in recs if not r.in_warmup]
    ok = [r for r in measured if r.status == "ok"]
    errs = [r for r in measured if r.status != "ok"]
    cancelled = [r for r in measured if r.status == "cancelled"]
    completion_toks = sum(r.completion_tokens or 0 for r in ok)

    # TTFT pool: successful observations plus right-censored ones — turns
    # killed while still waiting for a first token contribute their observed
    # wait as a lower bound, so overload-stage tails are not silently truncated
    ttft_ok = [r.ttft_visible_s for r in ok if r.ttft_visible_s is not None]
    ttft_censored = [r.e2e_s for r in measured
                     if r.ttft_visible_s is None and r.e2e_s
                     and r.status in ("ttft_timeout", "timeout", "cancelled")]
    ttft_pool = sorted(ttft_ok + ttft_censored)

    def ttft_pct(p):
        v = percentile(ttft_pool, p)
        return round(v, 3) if v is not None else None

    s = {
        "users": users,
        "repeat": repeat_idx,
        "turns_measured": len(measured),
        "turns_ok": len(ok),
        "turns_err": len(errs),
        "turns_cancelled": len(cancelled),
        "turns_retried": sum(1 for r in measured if r.retry_idx > 0),
        "err_rate_pct": round(100.0 * len(errs) / len(measured), 2) if measured else None,
        "errors_by_status": {},
        "turns_per_min": round(len(measured) / (window_s / 60.0), 2) if window_s else None,
        "ttft_n": len(ttft_pool),
        "ttft_censored_n": len(ttft_censored),
        "ttft_visible_p50_s": ttft_pct(50),
        "ttft_visible_p95_s": ttft_pct(95),
        "ttft_visible_p99_s": ttft_pct(99),
        "e2e_p50_s": pctl_field(ok, "e2e_s", 50),
        "e2e_p95_s": pctl_field(ok, "e2e_s", 95),
        "stream_tps_p50": pctl_field(ok, "stream_tps", 50),
        "stream_tps_p10": pctl_field(ok, "stream_tps", 10),
        "gap_p95_ms_p50": pctl_field(ok, "gap_p95_ms", 50),
        "turns_with_stall_gt2s_pct": round(
            100.0 * sum(1 for r in ok if r.stalls_gt2s > 0) / len(ok), 2) if ok else None,
        "completion_tokens": completion_toks,
        "client_agg_tps": round(completion_toks / window_s, 1) if window_s else None,
        "mean_completion_tokens": round(completion_toks / len(ok), 1) if ok else None,
        "mean_prompt_tokens": round(
            statistics.mean([r.prompt_tokens for r in ok if r.prompt_tokens]), 1)
            if any(r.prompt_tokens for r in ok) else None,
        # offered prefill distribution — correlate with the server-side
        # prefix_cache_hit_rate and prefill-time deltas in the report
        "prompt_tok_p50": pctl_field(ok, "prompt_tokens", 50),
        "prompt_tok_p95": pctl_field(ok, "prompt_tokens", 95),
        "prompt_tok_max": max((r.prompt_tokens for r in ok if r.prompt_tokens),
                              default=None),
        "mean_n_messages": round(statistics.mean(
            [r.n_messages for r in measured]), 2) if measured else None,
        "turn_idx_p50": pctl_field(measured, "turn_idx", 50),
        "turns_by_category": {},
        "ttft_p95_by_category": {},
        "truncated_pct": round(
            100.0 * sum(1 for r in ok if r.finish_reason == "length") / len(ok), 2) if ok else None,
    }
    for r in errs:
        s["errors_by_status"][r.status] = s["errors_by_status"].get(r.status, 0) + 1
    for r in measured:
        s["turns_by_category"][r.category] = s["turns_by_category"].get(r.category, 0) + 1
    for cat in list(s["turns_by_category"]):
        cat_ok = [r for r in ok if r.category == cat]
        if cat_ok:
            s["ttft_p95_by_category"][cat] = pctl_field(cat_ok, "ttft_visible_s", 95)

    # gauge aggregates over the measured window: the users -> concurrent
    # streams -> queue depth mapping the capacity report needs
    if win_samples:
        running = sorted(x["gauges"].get("num_requests_running", 0.0) for x in win_samples)
        waiting = sorted(x["gauges"].get("num_requests_waiting", 0.0) for x in win_samples)
        kv = sorted(x["gauges"].get("kv_cache_usage_perc", 0.0) for x in win_samples)
        s["running_mean"] = round(statistics.mean(running), 1)
        s["running_p95"] = round(percentile(running, 95), 1)
        s["running_max"] = round(running[-1], 1)
        s["waiting_mean"] = round(statistics.mean(waiting), 1)
        s["waiting_max"] = round(waiting[-1], 1)
        s["kv_mean"] = round(statistics.mean(kv), 3)
        s["kv_max"] = round(kv[-1], 3)
        s["streams_per_user"] = round(statistics.mean(running) / users, 3) if users else None

    # server-side cross-check over the SAME window (boundary snapshots)
    if m_start and m_end:
        server_window_s = m_end["mono"] - m_start["mono"]
        gen = counter_delta(m_start, m_end, "generation_tokens_total")
        s["server"] = {
            "window_s": round(server_window_s, 1),
            "gen_tokens": gen,
            "server_agg_tps": round(gen / server_window_s, 1)
                if gen is not None and server_window_s > 0 else None,
            "prompt_tokens": counter_delta(m_start, m_end, "prompt_tokens_total"),
            "preemptions": counter_delta(m_start, m_end, "num_preemptions_total"),
            "ttft_mean_s": hist_delta_mean(m_start, m_end, "time_to_first_token_seconds"),
            "ttft_p95_s": hist_delta_pctl(m_start, m_end, "time_to_first_token_seconds", 95),
            "itl_mean_s": hist_delta_mean(m_start, m_end, "inter_token_latency_seconds"),
            "queue_time_mean_s": hist_delta_mean(m_start, m_end, "request_queue_time_seconds"),
            "queue_time_p95_s": hist_delta_pctl(m_start, m_end, "request_queue_time_seconds", 95),
            "prefill_time_mean_s": hist_delta_mean(m_start, m_end, "request_prefill_time_seconds"),
            "decode_time_mean_s": hist_delta_mean(m_start, m_end, "request_decode_time_seconds"),
        }
        drafts = counter_delta(m_start, m_end, "spec_decode_num_drafts_total")
        draft_toks = counter_delta(m_start, m_end, "spec_decode_num_draft_tokens_total")
        accepted = counter_delta(m_start, m_end, "spec_decode_num_accepted_tokens_total")
        if drafts and draft_toks:
            s["server"]["spec_accept_rate"] = round(accepted / draft_toks, 3)
            s["server"]["spec_tokens_per_draft"] = round(accepted / drafts, 2)
        pq = counter_delta(m_start, m_end, "prefix_cache_queries_total")
        ph = counter_delta(m_start, m_end, "prefix_cache_hits_total")
        if pq:
            s["server"]["prefix_cache_hit_rate"] = round(ph / pq, 3)

    # SLO verdicts; a censored TTFT pool means some user waited past the turn
    # timeout for a first token — that alone fails the TTFT gate
    e2e_slo = cfg.slo_e2e_p95
    s["slo"] = {
        "ttft_p95": (s["ttft_visible_p95_s"] is not None
                     and s["ttft_visible_p95_s"] <= cfg.slo_ttft_p95
                     and s["ttft_censored_n"] == 0),
        "stream_tps_p10": (s["stream_tps_p10"] is not None
                           and s["stream_tps_p10"] >= cfg.slo_tps_p10),
        "err_rate": (s["err_rate_pct"] is not None
                     and s["err_rate_pct"] <= cfg.slo_err_pct),
        "stall_pct": (s["turns_with_stall_gt2s_pct"] is not None
                      and s["turns_with_stall_gt2s_pct"] <= cfg.slo_stall_pct),
        "e2e_p95": (s["e2e_p95_s"] is not None and s["e2e_p95_s"] <= e2e_slo),
    }
    s["slo"]["pass"] = all(s["slo"].values())
    return s


STAGE_TABLE_COLS = [
    ("users", "users"),
    ("rep", "repeat"),
    ("turns", "turns_measured"),
    ("err%", "err_rate_pct"),
    ("cxl", "turns_cancelled"),
    ("ttft p50", "ttft_visible_p50_s"),
    ("ttft p95", "ttft_visible_p95_s"),
    ("tps p10", "stream_tps_p10"),
    ("agg tps", "client_agg_tps"),
    ("run avg", "running_mean"),
    ("wait mx", "waiting_max"),
    ("kv mx", "kv_max"),
]


def print_stage_table(summaries: list):
    hdr = " | ".join(f"{h:>9}" for h, _ in STAGE_TABLE_COLS) + " | slo"
    print("\n" + hdr)
    print("-" * (len(hdr) + 24))
    for s in summaries:
        cells = []
        for _, key in STAGE_TABLE_COLS:
            v = s.get(key)
            if key == "ttft_visible_p95_s" and v is not None and s.get("ttft_censored_n", 0) > 0:
                cells.append(f"{str(v) + '+':>9}")  # '+' marks censored tail
            else:
                cells.append(f"{v:>9}" if v is not None else f"{'-':>9}")
        slo = s.get("slo", {})
        if slo.get("pass"):
            verdict = "PASS"
        else:
            failing = [k for k, v in slo.items() if k != "pass" and not v]
            verdict = "fail(" + ",".join(failing) + ")"
        print(" | ".join(cells) + f" | {verdict}")
    print(flush=True)


# ---------------------------------------------------------------------------
# Stage driver
# ---------------------------------------------------------------------------

async def run_stage(users: int, stage_idx: int, repeat_idx: int, cfg: Config,
                    client: httpx.AsyncClient, turns_file,
                    metrics_state: dict) -> dict:
    stage = Stage(users)
    records = []

    def sink(rec: TurnRecord):
        records.append(rec)
        turns_file.write(json.dumps(asdict(rec)) + "\n")
        turns_file.flush()

    conv_counter = iter(range(10**9))
    t_stage_start = time.monotonic()
    warmup_deadline = t_stage_start + cfg.warmup
    wall_start = datetime.now(timezone.utc).isoformat()

    # boundary snapshot at warmup end, so server deltas exclude warmup
    m_start_box = {}

    async def snap_start():
        await asyncio.sleep(max(0.0, warmup_deadline - time.monotonic()))
        m_start_box["m"] = await scrape_metrics(cfg)

    snap_task = asyncio.create_task(snap_start())

    tasks = [asyncio.create_task(
        user_task(uid, stage_idx, repeat_idx, cfg, stage, warmup_deadline,
                  client, sink, conv_counter))
        for uid in range(users)]

    # run until stage_duration AND min_turns are both satisfied (capped at
    # stage_max_duration) — low user counts need extra time for sample size
    end_at = t_stage_start + cfg.stage_duration
    hard_end = t_stage_start + cfg.stage_max_duration
    extended = False
    while True:
        now = time.monotonic()
        measured_done = sum(1 for r in records if not r.in_warmup)
        if now >= end_at and (measured_done >= cfg.min_turns or now >= hard_end):
            break
        if now >= end_at and not extended:
            extended = True
            print(f"  [extend] only {measured_done}/{cfg.min_turns} measured "
                  f"turns at {cfg.stage_duration:.0f}s — extending up to "
                  f"{cfg.stage_max_duration:.0f}s", flush=True)
        await asyncio.sleep(min(10.0, max(0.1, (end_at - now) if now < end_at else 5.0)))
        g = (metrics_state.get("latest") or {}).get("gauges", {})
        extra = (f" running={g.get('num_requests_running', '?')}"
                 f" waiting={g.get('num_requests_waiting', '?')}"
                 f" kv={g.get('kv_cache_usage_perc', 0):.0%}") if g else ""
        print(f"  [{users:>3}u +{time.monotonic()-t_stage_start:5.0f}s] "
              f"turns={measured_done}{extra}", flush=True)

    t_meas_end = time.monotonic()
    stage.stop_event.set()
    m_end = await scrape_metrics(cfg)  # boundary snapshot BEFORE the drain
    window_s = t_meas_end - warmup_deadline
    await snap_task
    m_start = m_start_box.get("m")

    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                               timeout=cfg.drain_timeout)
    except asyncio.TimeoutError:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    crashed = [t for t in tasks
               if t.done() and not t.cancelled() and t.exception()]
    if crashed:
        print(f"  [warn] {len(crashed)} user task(s) crashed; first: "
              f"{crashed[0].exception()!r}", flush=True)

    win_samples = [x for x in metrics_state.get("samples", [])
                   if warmup_deadline <= x["mono"] <= t_meas_end]
    summary = summarize_stage(users, repeat_idx, records, window_s,
                              m_start, m_end, win_samples, cfg)
    summary["wall_start"] = wall_start
    summary["wall_end"] = datetime.now(timezone.utc).isoformat()
    summary["user_tasks_crashed"] = len(crashed)
    return summary


def capacity_verdict(summaries: list, cfg: Config) -> dict:
    """Longest passing prefix of user counts; flags non-monotone results."""
    by_users = {}
    for s in summaries:
        by_users.setdefault(s["users"], []).append(s["slo"]["pass"])
    counts = sorted(by_users)
    passes = {u: all(v) for u, v in by_users.items()}
    capacity = None
    for u in counts:
        if passes[u]:
            capacity = u
        else:
            break
    first_fail = next((u for u in counts if not passes[u]), None)
    non_monotone = (first_fail is not None
                    and any(passes[u] for u in counts if u > first_fail))
    all_pass = all(passes.values())
    return {"capacity_users": capacity, "first_fail_users": first_fail,
            "all_stages_passed": all_pass, "non_monotone": non_monotone,
            "counts_tested": counts}


async def main_async(cfg: Config) -> int:
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    warnings = []
    if cfg.mode == "gateway" and not cfg.metrics_url:
        warnings.append(
            "gateway mode without --metrics-url: no server-side cross-check "
            "(spec-decode/prefix-cache/queue-time deltas) and no real idle "
            "detection between stages. Point --metrics-url at the backend "
            "node's vLLM /metrics (valid only if the gateway routes this "
            "model exclusively to that backend).")
    if cfg.turn_timeout > cfg.drain_timeout:
        warnings.append(
            f"turn_timeout ({cfg.turn_timeout:.0f}s) > drain_timeout "
            f"({cfg.drain_timeout:.0f}s): turns still in flight at drain end "
            "are recorded as 'cancelled' (right-censored) rather than "
            "reaching a terminal timeout status.")
    for w in warnings:
        print(f"[warn] {w}", flush=True)
    (outdir / "config.json").write_text(json.dumps(
        {"config": redacted_config(cfg), "warnings": warnings}, indent=2))
    turns_path = outdir / "turns.jsonl"
    samples_path = outdir / "server_samples.jsonl"

    metrics_state = {}
    sampler_stop = asyncio.Event()
    sampler = asyncio.create_task(
        sampler_task(cfg, samples_path, sampler_stop, metrics_state))

    max_stage_users = max(max(cfg.users_stages), cfg.max_users if cfg.adapt else 0)
    limits = httpx.Limits(max_connections=max_stage_users + 10,
                          max_keepalive_connections=max_stage_users + 10,
                          keepalive_expiry=60.0)
    # read=None: the first-token wait is legitimately unbounded under queueing;
    # asyncio.timeout(turn_timeout) in run_turn bounds every read, and the
    # mid-stream stall budget is enforced client-side after the first delta
    timeout = httpx.Timeout(connect=15.0, read=None, write=60.0, pool=60.0)

    summaries = []
    print(f"chat_bench: {cfg.mode} -> {cfg.base_url} model={cfg.model} "
          f"stages={cfg.users_stages} x{cfg.repeats} stage={cfg.stage_duration:.0f}s "
          f"warmup={cfg.warmup:.0f}s adapt={cfg.adapt} seed={cfg.seed}", flush=True)

    def write_summary():
        verdict = capacity_verdict(summaries, cfg) if summaries else {}
        (outdir / "summary.json").write_text(json.dumps({
            "label": cfg.label, "config": redacted_config(cfg),
            "warnings": warnings, "verdict": verdict, "stages": summaries,
        }, indent=2))
        return verdict

    async with httpx.AsyncClient(verify=cfg.verify_tls, limits=limits,
                                 timeout=timeout, http2=False) as client:
        # warm the model once so stage 1 doesn't pay any cold-start costs
        warm_rec = TurnRecord(stage_users=0, user_id=-1, conv_id=-1, turn_idx=0,
                              retry_idx=0, n_messages=2, prompt_chars=0,
                              category="warmup", in_warmup=True)
        await run_turn(client, cfg, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Say 'ready' and nothing else."},
        ], warm_rec)
        if warm_rec.status != "ok" or warm_rec.ttft_visible_s is None:
            print(f"[fatal] warmup request failed: status={warm_rec.status} "
                  f"error={warm_rec.error}", flush=True)
            sampler_stop.set()
            await sampler
            return 1
        print(f"warmup ok: ttft={warm_rec.ttft_visible_s:.2f}s "
              f"tokens={warm_rec.completion_tokens}", flush=True)

        planned = list(cfg.users_stages)
        stage_seq = 0
        bisections = 0
        with turns_path.open("a") as turns_file:
            idx = 0
            while idx < len(planned):
                users = planned[idx]
                for rep in range(cfg.repeats):
                    print(f"\n=== stage {stage_seq}: {users} users "
                          f"(repeat {rep + 1}/{cfg.repeats}) ===", flush=True)
                    s = await run_stage(users, stage_seq, rep, cfg, client,
                                        turns_file, metrics_state)
                    summaries.append(s)
                    stage_seq += 1
                    print_stage_table(summaries)
                    write_summary()
                    await wait_for_idle(cfg, metrics_state)
                idx += 1
                # adaptive continuation: extend past the last stage while all
                # SLOs pass, then bisect to bracket the knee
                if idx == len(planned) and cfg.adapt:
                    v = capacity_verdict(summaries, cfg)
                    tested = set(v["counts_tested"])
                    if v["all_stages_passed"]:
                        last = max(tested)
                        nxt = min(cfg.max_users, math.ceil(last * 1.5))
                        if nxt > last:
                            print(f"[adapt] all stages pass — extending sweep "
                                  f"to {nxt} users", flush=True)
                            planned.append(nxt)
                    elif (v["capacity_users"] and v["first_fail_users"]
                          and bisections < 2):
                        p, f = v["capacity_users"], v["first_fail_users"]
                        mid = (p + f) // 2
                        if f / p > 1.25 and mid not in tested:
                            print(f"[adapt] bisecting knee between {p} (pass) "
                                  f"and {f} (fail): adding {mid}", flush=True)
                            planned.append(mid)
                            bisections += 1

    sampler_stop.set()
    await sampler

    verdict = write_summary()
    print(f"\nDone. Results in {outdir}/ (turns.jsonl, server_samples.jsonl, "
          f"summary.json)", flush=True)
    slo_desc = (f"TTFT p95<={cfg.slo_ttft_p95}s uncensored, "
                f"stream p10>={cfg.slo_tps_p10} tok/s, "
                f"err<={cfg.slo_err_pct}%, stall<={cfg.slo_stall_pct}%, "
                f"e2e p95<={cfg.slo_e2e_p95:.0f}s")
    if verdict.get("non_monotone"):
        print("WARNING: pass/fail is non-monotone in user count — results are "
              "noise-suspect; repeat the stages around the knee "
              "(--repeats 3).", flush=True)
    if verdict.get("all_stages_passed"):
        print(f"Capacity NOT located: >= {max(verdict['counts_tested'])} users "
              f"(all stages passed {slo_desc}) — extend the sweep "
              f"(--max-users).", flush=True)
    elif verdict.get("capacity_users"):
        print(f"Capacity: {verdict['capacity_users']} users pass all SLOs "
              f"({slo_desc}); first failure at "
              f"{verdict['first_fail_users']} users.", flush=True)
    else:
        print(f"No tested user count met the SLOs ({slo_desc}).", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Chat-shaped capacity benchmark")
    p.add_argument("--base-url", required=True,
                   help="vLLM backend URL (direct mode) or gateway URL")
    p.add_argument("--mode", choices=["direct", "gateway"], default="direct")
    p.add_argument("--model", default="google/gemma-4-31b")
    p.add_argument("--api-key", default="", help="API key (gateway mode)")
    p.add_argument("--users", default="1,2,4,8,16,24,32,48,64",
                   help="comma-separated user counts to sweep (auto-extends "
                        "unless --no-adapt)")
    p.add_argument("--stage-duration", type=float, default=300.0)
    p.add_argument("--stage-max-duration", type=float, default=900.0,
                   help="cap when extending a stage to reach --min-turns")
    p.add_argument("--min-turns", type=int, default=30,
                   help="extend a stage (up to --stage-max-duration) until "
                        "this many measured turns complete")
    p.add_argument("--warmup", type=float, default=60.0,
                   help="seconds at stage start excluded from stats")
    p.add_argument("--drain-timeout", type=float, default=180.0)
    p.add_argument("--cooldown", type=float, default=15.0)
    p.add_argument("--repeats", type=int, default=1,
                   help="repeat each stage K times (>=3 for variance estimates)")
    p.add_argument("--no-adapt", action="store_true",
                   help="fixed stage list: no auto-extension or knee bisection")
    p.add_argument("--max-users", type=int, default=512,
                   help="ceiling for adaptive sweep extension")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="response cap; production chat is effectively uncapped, "
                        "watch truncated_pct in the summary")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--think", choices=["off", "on", "none"], default="off",
                   help="chat_template_kwargs enable_thinking (default off, "
                        "matching the gateway's fleet default)")
    p.add_argument("--reading-tps", type=float, default=15.0,
                   help="reading speed used for think time, tokens/sec")
    p.add_argument("--reading-cap", type=float, default=45.0)
    p.add_argument("--compose-median", type=float, default=12.0,
                   help="median seconds composing the next message")
    p.add_argument("--compose-sigma", type=float, default=0.7)
    p.add_argument("--compose-cap", type=float, default=90.0)
    p.add_argument("--long-paste-pct", type=float, default=0.12,
                   help="fraction of openers that paste a 300-1500 token doc")
    p.add_argument("--xl-paste-pct", type=float, default=0.03,
                   help="fraction of openers that paste a 1500-6000 token doc")
    p.add_argument("--medium-pct", type=float, default=0.30,
                   help="fraction of openers with a ~80-350 token context block")
    p.add_argument("--mid-paste-pct", type=float, default=0.08,
                   help="fraction of follow-ups that paste new material")
    p.add_argument("--cache-adversarial", action="store_true",
                   help="salt the system prompt per conversation, removing "
                        "cross-conversation prefix sharing (within-conversation "
                        "history caching remains — see docstring)")
    p.add_argument("--max-turns", type=int, default=16)
    p.add_argument("--no-think-time", action="store_true",
                   help="zero think time: users==concurrent streams "
                        "(pure saturation mode)")
    p.add_argument("--stall-timeout", type=float, default=90.0,
                   help="abort a turn if no bytes arrive MID-STREAM for this "
                        "long; the first-token wait is bounded by --turn-timeout")
    p.add_argument("--turn-timeout", type=float, default=600.0)
    p.add_argument("--metrics-url", default=None,
                   help="vLLM /metrics URL to sample (default: <base-url>/metrics "
                        "in direct mode; strongly recommended in gateway mode "
                        "when the model routes to a single backend)")
    p.add_argument("--metrics-interval", type=float, default=5.0)
    p.add_argument("--slo-ttft-p95", type=float, default=2.0)
    p.add_argument("--slo-tps-p10", type=float, default=15.0)
    p.add_argument("--slo-err-pct", type=float, default=1.0)
    p.add_argument("--slo-stall-pct", type=float, default=2.0)
    p.add_argument("--slo-e2e-p95", type=float, default=None,
                   help="default derived: slo-ttft-p95 + max-tokens/slo-tps-p10")
    p.add_argument("--outdir", default="bench_results/chat_bench")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default="")
    p.add_argument("--verify-tls", action="store_true",
                   help="verify TLS certs (off by default: nodes use internal certs)")
    a = p.parse_args(argv)

    if a.mode == "gateway" and not a.api_key:
        p.error("--api-key is required in gateway mode")
    metrics_url = a.metrics_url
    if metrics_url is None and a.mode == "direct":
        metrics_url = a.base_url.rstrip("/") + "/metrics"
    slo_e2e = a.slo_e2e_p95
    if slo_e2e is None:
        # derived so it cannot contradict the tps floor by construction
        slo_e2e = a.slo_ttft_p95 + a.max_tokens / a.slo_tps_p10

    return Config(
        base_url=a.base_url, mode=a.mode, model=a.model, api_key=a.api_key,
        users_stages=[int(x) for x in a.users.split(",") if x.strip()],
        stage_duration=a.stage_duration, stage_max_duration=a.stage_max_duration,
        min_turns=a.min_turns, warmup=a.warmup,
        drain_timeout=a.drain_timeout, cooldown=a.cooldown,
        repeats=a.repeats, adapt=not a.no_adapt, max_users=a.max_users,
        max_tokens=a.max_tokens, temperature=a.temperature, top_p=a.top_p,
        think=a.think, reading_tps=a.reading_tps, reading_cap=a.reading_cap,
        compose_median=a.compose_median, compose_sigma=a.compose_sigma,
        compose_cap=a.compose_cap, long_paste_pct=a.long_paste_pct,
        xl_paste_pct=a.xl_paste_pct, medium_pct=a.medium_pct,
        mid_paste_pct=a.mid_paste_pct, cache_adversarial=a.cache_adversarial,
        max_turns=a.max_turns, no_think_time=a.no_think_time,
        stall_timeout=a.stall_timeout, turn_timeout=a.turn_timeout,
        metrics_url=metrics_url or "", metrics_interval=a.metrics_interval,
        slo_ttft_p95=a.slo_ttft_p95, slo_tps_p10=a.slo_tps_p10,
        slo_err_pct=a.slo_err_pct, slo_stall_pct=a.slo_stall_pct,
        slo_e2e_p95=slo_e2e,
        outdir=a.outdir, seed=a.seed, label=a.label, verify_tls=a.verify_tls,
    )


def main():
    cfg = parse_args()
    # make SIGTERM (container stop, scheduler kill) behave like Ctrl-C so
    # partial results are kept and the exit message prints
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        rc = asyncio.run(main_async(cfg))
    except KeyboardInterrupt:
        print("\ninterrupted — partial results kept in outdir", flush=True)
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
