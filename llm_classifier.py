import asyncio
import csv
import json
import os
import random
import anthropic
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = "data/sme_survey_responses.csv"
OUTPUT_FILE = "data/llm_predictions.csv"
MODEL = "claude-haiku-4-5-20251001"
CONCURRENCY = 10

SYSTEM_PROMPT = (
    "You are an expert survey analyst for SME AI adoption research. "
    "Analyse open-ended survey responses and extract structured labels. "
    "Output a raw JSON object only — no markdown, no explanation, nothing else."
)

USER_TEMPLATE = (
    "Classify this SME survey response on AI adoption.\n\n"
    "Labels to extract:\n"
    '- sentiment: "positive" | "neutral" | "negative"\n'
    '- adoption_stage: "exploring" | "piloting" | "scaling" | "not_interested"\n'
    '- main_barrier: "cost" | "skills" | "trust" | "none"\n\n'
    "Definitions:\n"
    "  sentiment — positive=optimistic/satisfied, neutral=mixed/uncertain, negative=frustrated/opposed\n"
    "  adoption_stage — exploring=researching not yet trialling, piloting=actively testing, "
    "scaling=deployed and expanding, not_interested=no intention to adopt\n"
    "  main_barrier — cost=financial, skills=expertise/training gaps, "
    "trust=reliability/privacy concerns, none=no significant barrier mentioned\n\n"
    'Response: """{response}"""'
)

VALID = {
    "sentiment": {"positive", "neutral", "negative"},
    "adoption_stage": {"exploring", "piloting", "scaling", "not_interested"},
    "main_barrier": {"cost", "skills", "trust", "none"},
}

FIELDNAMES = [
    "row_id", "response",
    "true_sentiment", "true_adoption_stage", "true_main_barrier",
    "pred_sentiment", "pred_adoption_stage", "pred_main_barrier",
]


async def classify_one(client, semaphore, row_id, response_text):
    async with semaphore:
        for attempt in range(5):
            try:
                message = await client.messages.create(
                    model=MODEL,
                    max_tokens=80,
                    system=[{
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    messages=[
                        {"role": "user", "content": USER_TEMPLATE.format(response=response_text)},
                        {"role": "assistant", "content": "{"},  # prefill forces raw JSON
                    ],
                )
                raw = "{" + message.content[0].text
                parsed = json.loads(raw)
                for field, options in VALID.items():
                    if parsed.get(field) not in options:
                        raise ValueError(f"Invalid {field}: {parsed.get(field)!r}")
                return row_id, parsed
            except anthropic.RateLimitError:
                wait = min(2 ** attempt + random.uniform(0, 1), 60)
                await asyncio.sleep(wait)
            except (json.JSONDecodeError, ValueError):
                await asyncio.sleep(0.5 * (attempt + 1))
        return row_id, None


async def main_async():
    os.makedirs("data", exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    existing = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add(int(row["row_id"]))

    todo = [(i, row) for i, row in enumerate(rows) if i not in existing]
    print(f"Total: {len(rows)} | Already done: {len(existing)} | Remaining: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    write_lock = asyncio.Lock()
    completed = 0

    async def process(i, row):
        nonlocal completed
        row_id, pred = await classify_one(client, semaphore, i, row["response"])
        if pred is None:
            print(f"  Row {row_id}: failed after 5 attempts, skipping")
            return
        out = {
            "row_id": row_id,
            "response": row["response"],
            "true_sentiment": row["sentiment"],
            "true_adoption_stage": row["adoption_stage"],
            "true_main_barrier": row["main_barrier"],
            "pred_sentiment": pred["sentiment"],
            "pred_adoption_stage": pred["adoption_stage"],
            "pred_main_barrier": pred["main_barrier"],
        }
        async with write_lock:
            with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(out)
            completed += 1
            if completed % 50 == 0:
                print(f"  Progress: {completed}/{len(todo)}")

    await asyncio.gather(*[process(i, row) for i, row in todo])
    print(f"\nDone. {completed}/{len(todo)} rows classified -> {OUTPUT_FILE}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
