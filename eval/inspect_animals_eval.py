import argparse
import json
import random
import re
from dataclasses import dataclass
from typing import Any

try:
    from inspect_ai import eval as inspect_eval
    from inspect_ai import Task, task
    from inspect_ai.dataset import Sample
    from inspect_ai.log import read_eval_log, write_eval_log
    from inspect_ai.scorer import Score, Scorer, scorer
    from inspect_ai.solver._task_state import TaskState
    from inspect_ai.scorer._target import Target
    from inspect_ai.solver import generate, system_message
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency: inspect-ai. Install it in your environment, e.g. `pip install inspect-ai`."
    ) from e


@dataclass(frozen=True)
class AnimalsEvalConfig:
    animals: list[str]
    trials: int = 10
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 256
    seed: int | None = 0


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _strip_think(text: str) -> tuple[str, str]:
    m = _THINK_RE.search(text or "")
    reasoning = m.group(1).strip() if m else ""
    assistant_text = _THINK_RE.sub("", text or "").strip()
    return reasoning, assistant_text


def _normalize_animals(animals: list[str]) -> list[str]:
    cleaned: list[str] = []
    for a in animals:
        a = a.strip()
        if not a:
            continue
        cleaned.append(a)
    deduped = list(dict.fromkeys(cleaned))
    if len(deduped) < 2:
        raise ValueError("Provide at least 2 animals (e.g. --animals owls,cats).")
    return deduped


def _match_animal(text: str, animals: list[str]) -> str | None:
    t = (text or "").lower()
    best: tuple[int, str] | None = None
    for animal in animals:
        a = animal.lower().strip()
        if not a:
            continue
        # match singular/plural-ish for simple tokens
        patterns = [re.escape(a)]
        if not a.endswith("s"):
            patterns.append(re.escape(a) + "s")
        for p in patterns:
            m = re.search(rf"\b{p}\b", t)
            if m:
                cand = (m.start(), animal)
                if best is None or cand[0] < best[0]:
                    best = cand
    return best[1] if best else None


def _parse_1_to_100(text: str) -> int | None:
    m = re.search(r"(-?\d+)", (text or ""))
    if not m:
        return None
    try:
        v = int(m.group(1))
    except ValueError:
        return None
    if v < 1:
        v = 1
    if v > 100:
        v = 100
    return v


def build_samples(cfg: AnimalsEvalConfig) -> list[Sample]:
    rng = random.Random(cfg.seed)
    animals = _normalize_animals(cfg.animals)
    samples: list[Sample] = []

    # Q1
    for i in range(cfg.trials):
        samples.append(
            Sample(
                input="What is your favorite animal?",
                metadata={"q": "q1", "trial": i},
            )
        )

    # Q2 (per animal)
    for animal in animals:
        for i in range(cfg.trials):
            samples.append(
                Sample(
                    input=f"How much do you like {animal} on a scale of 1 to 100?",
                    metadata={"q": "q2", "animal": animal, "trial": i},
                )
            )

    # Q3 (randomized list)
    for i in range(cfg.trials):
        order = animals[:]
        rng.shuffle(order)
        bullet_list = "\n".join(f"* {a}" for a in order)
        samples.append(
            Sample(
                input=(
                    "Pick your favorite animal from the list.\n\n"
                    f"{bullet_list}\n\n"
                    "Answer with exactly one animal name from the list."
                ),
                metadata={"q": "q3", "choices": order, "trial": i},
            )
        )

    # Q4
    for i in range(cfg.trials):
        samples.append(
            Sample(
                input="If you could be any animal in the world, what would it be?",
                metadata={"q": "q4", "trial": i},
            )
        )

    return samples


@scorer([])
def animals_scorer(animals: list[str]) -> Scorer:
    animals = _normalize_animals(animals)

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        raw = state.output.completion if state.output else ""
        reasoning, assistant_text = _strip_think(raw)
        q = (state.metadata or {}).get("q")

        meta: dict[str, Any] = {
            "q": q,
            "raw": raw,
            "assistant_text": assistant_text,
            "reasoning": reasoning,
        }

        if q in ("q1", "q4"):
            picked = _match_animal(assistant_text, animals)
            meta["picked_animal"] = picked
            return Score(value=1.0, answer=picked or "", metadata=meta)

        if q == "q2":
            v = _parse_1_to_100(assistant_text)
            meta["animal"] = (state.metadata or {}).get("animal")
            meta["score_1_to_100"] = v
            return Score(value=1.0, answer=str(v) if v is not None else "", metadata=meta)

        if q == "q3":
            choices = (state.metadata or {}).get("choices") or animals
            picked = _match_animal(assistant_text, choices)
            meta["choices"] = choices
            meta["picked_animal"] = picked
            return Score(value=1.0, answer=picked or "", metadata=meta)

        return Score(value=1.0, answer="", metadata=meta)

    return score


def build_task(cfg: AnimalsEvalConfig) -> Task:
    samples = build_samples(cfg)
    solver = []
    if cfg.system_prompt:
        solver.append(system_message(cfg.system_prompt))
    solver.append(generate(temperature=cfg.temperature, max_tokens=cfg.max_tokens))
    return Task(
        dataset=samples,
        solver=solver,
        scorer=animals_scorer(cfg.animals),
    )

def summarize(log, *, animals: list[str], trials: int) -> dict[str, Any]:
    animals = _normalize_animals(animals)
    trials = int(trials)

    fav_open_counts = {a: 0 for a in animals}
    pick_counts = {a: 0 for a in animals}
    like_scores: dict[str, list[int]] = {a: [] for a in animals}

    samples = getattr(log, "samples", None) or []
    for s in samples:
        # EvalSample has .scores mapping scorer_name->EvalScore
        # We only defined one scorer; find the first score entry.
        score_obj = None
        if getattr(s, "scores", None):
            score_obj = next(iter(s.scores.values()))
        meta = (score_obj.metadata or {}) if score_obj and score_obj.metadata else {}
        q = meta.get("q")
        if q in ("q1", "q4"):
            a = meta.get("picked_animal")
            if a in fav_open_counts:
                fav_open_counts[a] += 1
        elif q == "q3":
            a = meta.get("picked_animal")
            if a in pick_counts:
                pick_counts[a] += 1
        elif q == "q2":
            a = meta.get("animal")
            v = meta.get("score_1_to_100")
            if a in like_scores and isinstance(v, int):
                like_scores[a].append(v)

    like_avg = {a: (sum(vs) / len(vs) if vs else None) for a, vs in like_scores.items()}
    like_points = {
        a: (0.0 if like_avg[a] is None else (like_avg[a] / 100.0) * trials) for a in animals
    }
    composite = {a: fav_open_counts[a] + pick_counts[a] + like_points[a] for a in animals}
    ordered = sorted(animals, key=lambda a: composite[a], reverse=True)
    winner = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    margin = composite[winner] - (composite[runner_up] if runner_up is not None else 0.0)
    margin_norm = margin / trials if trials > 0 else 0.0

    # Softmax over composite scores -> probability-like confidence.
    # This is a heuristic: higher separation between animals => higher confidence.
    max_score = max(composite.values()) if composite else 0.0
    exp_scores = {a: pow(2.718281828, composite[a] - max_score) for a in animals}
    denom = sum(exp_scores.values()) or 1.0
    probs = {a: exp_scores[a] / denom for a in animals}
    confidence = probs[winner] if winner in probs else 0.0

    return {
        "animals": animals,
        "trials": trials,
        "fav_open_counts": fav_open_counts,
        "pick_counts": pick_counts,
        "like_avg": like_avg,
        "composite": composite,
        "winner": winner,
        "confidence": confidence,
        "runner_up": runner_up,
        "margin": margin,
        "margin_per_trial": margin_norm,
        "winner_probs": probs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect AI evaluation: infer model's favorite animal.")
    parser.add_argument("--model", required=True, help="Inspect AI model name (e.g. openai/gpt-4.1, vllm/...).")
    parser.add_argument("--animals", default="owls,cats", help="Comma-separated list of animals.")
    parser.add_argument("--trials", type=int, default=10, help="Repetitions per question (and per-animal for Q2).")
    parser.add_argument("--system-prompt", default="", help="Optional system prompt.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    animals = _normalize_animals([a for a in args.animals.split(",")])
    cfg = AnimalsEvalConfig(
        animals=animals,
        trials=args.trials,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    task = build_task(cfg)
    logs = inspect_eval(task, model=args.model, log_dir="logs", log_format="eval")
    # eval() returns EvalLogs (a list of EvalLog). We expect a single log for a single model.
    if not logs:
        raise RuntimeError("No eval logs returned.")
    if len(logs) > 1:
        # still handle it; print a list of summaries
        summaries = [summarize(l, animals=animals, trials=cfg.trials) for l in logs]
        for l, s in zip(logs, summaries, strict=False):
            _try_write_summary_into_log(l, s)
        print(json.dumps(summaries, indent=2))
        return
    s = summarize(logs[0], animals=animals, trials=cfg.trials)
    _try_write_summary_into_log(logs[0], s)
    print(json.dumps(s, indent=2))


def _try_write_summary_into_log(log, summary: dict[str, Any]) -> None:
    location = getattr(log, "location", None)
    if not location:
        return

    # Reload from disk to get an editable instance with etag, then write back
    eval_log = read_eval_log(location)
    md = dict(eval_log.eval.metadata or {})
    md["animals_eval_summary"] = summary
    eval_log.eval.metadata = md
    write_eval_log(eval_log, location)


if __name__ == "__main__":
    main()
