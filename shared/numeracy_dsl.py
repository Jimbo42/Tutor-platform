from __future__ import annotations

import json
import random
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BuiltQuestion:
    qid: str
    prompt: str
    choices: List[Dict[str, Any]]   # each: {"label": str, "value": Any}
    correct_index: int
    explain: str
    env: Dict[str, Any]             # debug


# ----------------------------
# Load/save
# ----------------------------

def load_game(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def save_game(path: str | Path, game: Dict[str, Any]) -> None:
    p = Path(path)
    p.write_text(json.dumps(game, indent=2, ensure_ascii=False), encoding="utf-8")


# ----------------------------
# Safe expression evaluation
# ----------------------------

def _is_ref(x: Any) -> bool:
    return isinstance(x, str) and x.startswith("$")


def _resolve(x: Any, env: Dict[str, Any]) -> Any:
    if _is_ref(x):
        k = x[1:]
        if k not in env:
            raise KeyError(f"Unknown reference: {x}")
        return env[k]
    return x


def _eval_expr(expr: Any, env: Dict[str, Any], limits: Dict[str, Any]) -> Any:
    if not isinstance(expr, dict):
        return _resolve(expr, env)

    op = expr.get("op")
    args = expr.get("args", [])

    allowed_math = set(limits.get("math_ops", []))
    allowed_logic = set(limits.get("logic_ops", []))

    if op in allowed_math:
        v = [_eval_expr(a, env, limits) for a in args]
        if op == "add": return v[0] + v[1]
        if op == "sub": return v[0] - v[1]
        if op == "mul": return v[0] * v[1]
        if op == "div": return v[0] / v[1]
        if op == "pow": return v[0] ** v[1]
        if op == "round": return round(float(v[0]), int(v[1]))
        if op == "gcd":
            if len(v) != 2:
                raise ValueError("gcd requires exactly 2 arguments")
            return math.gcd(int(v[0]), int(v[1]))
        raise ValueError(f"Unhandled math op: {op}")

    if op == "ne":
        v = [_eval_expr(a, env, limits) for a in args]
        return v[0] != v[1]

    if op in allowed_logic:
        v = [_eval_expr(a, env, limits) for a in args]
        if op == "gt": return v[0] > v[1]
        if op == "lt": return v[0] < v[1]
        if op == "eq": return v[0] == v[1]
        if op == "and": return all(bool(x) for x in v)
        if op == "or": return any(bool(x) for x in v)
        if op == "not": return not bool(v[0])
        raise ValueError(f"Unhandled logic op: {op}")

    raise ValueError(f"Operator '{op}' not allowed by limits.")


# ----------------------------
# Sampling
# ----------------------------

def _sample_var(spec: Dict[str, Any]) -> Any:
    kind = spec.get("kind")
    if kind == "int":
        lo, hi = int(spec["min"]), int(spec["max"])
        if lo > hi: lo, hi = hi, lo
        return random.randint(lo, hi)
    if kind == "choice":
        return random.choice(list(spec["values"]))
    raise ValueError(f"Unknown var kind: {kind}")


# ----------------------------
# Templating
# ----------------------------

TOK = re.compile(r"\{\{(.*?)\}\}")

# def _fmt_num(x: float, places: int) -> str:
#     y = round(float(x), places)
#     s = f"{y:.{places}f}"
#     s = s.rstrip("0").rstrip(".") if "." in s else s
#     return s

def _fmt_num(val, places):
    places = int(places)
    if places <= 0:
        return str(int(round(float(val))))
    return f"{float(val):.{places}f}"

def _render(
    template: str,
    env: Dict[str, Any],
    choices: Optional[List[Dict[str, Any]]] = None,
    *,
    expr_places: Optional[int] = None,
    choice_value_places: Optional[int] = None,
) -> str:
    def _maybe_fmt(val: Any, places: Optional[int]) -> str:
        if isinstance(val, bool):
            return str(val)
        if places is not None and isinstance(val, (int, float)):
            return _fmt_num(val, int(places))
        return str(val)

    def repl(m):
        inner = m.group(1).strip()

        if inner.startswith("expr:"):
            ref = inner[len("expr:"):].strip()
            val = _resolve(ref, env)
            text = _maybe_fmt(val, expr_places)
            if isinstance(val, (int, float)) and val < 0:
                return f"({text})"
            return text

        if inner.startswith("fmt:"):
            rest = inner[len("fmt:"):].strip()
            ref, places = [x.strip() for x in rest.split(",")]
            val = _resolve(ref, env)
            return _fmt_num(val, int(places))

        if inner.startswith("choice:"):
            if choices is None:
                raise ValueError("choice:* token used but choices not provided")
            rest = inner[len("choice:"):].strip()
            idx_s, field = rest.split(".", 1)
            idx = int(idx_s)
            val = choices[idx][field]
            if field == "value":
                return _maybe_fmt(val, choice_value_places)
            return str(val)

        raise ValueError(f"Unsupported token: {inner}")

    return TOK.sub(repl, template)

# ----------------------------
# Choice builders (generic primitives)
# ----------------------------

def _build_choices(spec: Dict[str, Any], env: Dict[str, Any], limits: Dict[str, Any]) -> List[Dict[str, Any]]:
    mode = spec.get("mode")

    if mode == "computed_items":
        out = []
        label_places = spec.get("label_places", None)

        for it in spec["items"]:
            label = _render(
                it["label"],
                env,
                expr_places=label_places,
            )
            value = (
                _resolve(it["value"], env)
                if not isinstance(it["value"], dict)
                else _eval_expr(it["value"], env, limits)
            )
            out.append({"label": label, "value": value})
        return out

    if mode == "two_decimal_one_distractor":
        exact = float(_resolve(spec["exact"], env))
        delta = float(_resolve(spec["delta"], env))
        places = int(spec.get("round_places", 3))
        n_distractors = int(spec.get("n_distractors", 1))

        clamp_01 = bool(spec.get("clamp_01", True))

        exact_s = _fmt_num(exact, places)

        out = [{"label": exact_s, "value": float(exact_s)}]
        seen = {exact_s}

        k = 1
        guard = 0
        while len(out) < 1 + n_distractors and guard < 200:
            guard += 1
            sign = -1 if random.random() < 0.5 else 1
            cand = exact + sign * k * delta

            if clamp_01:
                cand = max(0.0, min(1.0, cand))

            cand_s = _fmt_num(cand, places)

            if cand_s not in seen:
                out.append({"label": cand_s, "value": float(cand_s)})
                seen.add(cand_s)

            if guard % 2 == 0:
                k += 1

        random.shuffle(out)
        return out

    raise ValueError(f"Unknown choices mode: {mode}")

# ----------------------------
# Answer rules (generic primitives)
# ----------------------------

def _pick_correct_index(
    answer_spec: Dict[str, Any],
    choices: List[Dict[str, Any]],
    env: Optional[Dict[str, Any]] = None,
) -> int:
    mode = answer_spec.get("mode")

    if mode == "argmax":
        field = answer_spec.get("field", "value")
        mx = max(ch[field] for ch in choices)
        idxs = [i for i, ch in enumerate(choices) if ch[field] == mx]
        return random.choice(idxs) if answer_spec.get("tie_break") == "random" else idxs[0]

    if mode == "argmin":
        field = answer_spec.get("field", "value")
        mn = min(ch[field] for ch in choices)
        idxs = [i for i, ch in enumerate(choices) if ch[field] == mn]
        return random.choice(idxs) if answer_spec.get("tie_break") == "random" else idxs[0]

    if mode == "choice_is_exact":
        # Preferred path: explicit exact-tag on the choice
        idxs = [i for i, ch in enumerate(choices) if ch.get("is_exact")]
        if idxs:
            return idxs[0]

        # Fallback path: compare against an env-driven exact value
        if env is not None and "exact" in answer_spec:
            target = _resolve(answer_spec["exact"], env)
            idxs = [i for i, ch in enumerate(choices) if ch.get("value") == target]
            if idxs:
                return idxs[0]

        raise ValueError("choice_is_exact requires an exact-marked choice or answer.exact")

    raise ValueError(f"Unknown answer mode: {mode}")

# ----------------------------
# Selection engine (weight/cooldown/max-per-round)
# ----------------------------

def pick_question_def(
    game: Dict[str, Any],
    history: List[str],
    used_counts: Dict[str, int],
    difficulty: Optional[str] = None,
) -> Dict[str, Any]:
    qdefs = [q for q in game.get("questions", []) if q.get("enabled", True)]
    if difficulty:
        qdefs = [q for q in qdefs if str(q.get("difficulty", "Starter")) == str(difficulty)]
    if not qdefs:
        if difficulty:
            raise ValueError(f"No enabled questions for difficulty '{difficulty}' in numeracy_game.json")
        raise ValueError("No enabled questions in numeracy_game.json")

    cooldown_default = int(game.get("selection", {}).get("cooldown_default", 0))

    candidates, weights = [], []
    for q in qdefs:
        sel = q.get("selection", {})
        qid = q["id"]

        max_per_round = sel.get("max_per_round")
        if max_per_round is not None and used_counts.get(qid, 0) >= int(max_per_round):
            continue

        cooldown = int(sel.get("cooldown", cooldown_default))
        if cooldown > 0 and qid in history[-cooldown:]:
            continue

        w = float(sel.get("weight", 1))
        if w <= 0:
            continue

        candidates.append(q)
        weights.append(w)

    if not candidates:
        candidates = qdefs
        weights = [float(q.get("selection", {}).get("weight", 1)) for q in candidates]

    return random.choices(candidates, weights=weights, k=1)[0]


# ----------------------------
# Build a concrete question instance
# ----------------------------
def build_question(game: Dict[str, Any], qdef: Dict[str, Any], max_resamples: int = 80) -> BuiltQuestion:
    limits = game.get("limits", {})

    def _collect_refs(expr: Any) -> set[str]:
        """Return a set of '$name' references used anywhere inside expr."""
        refs: set[str] = set()
        if isinstance(expr, str) and expr.startswith("$"):
            refs.add(expr)
        elif isinstance(expr, dict):
            for a in expr.get("args", []) or []:
                refs |= _collect_refs(a)
        elif isinstance(expr, list):
            for a in expr:
                refs |= _collect_refs(a)
        return refs

    for _ in range(max_resamples):
        env: Dict[str, Any] = {}

        # ----------------
        # Sample vars first
        # ----------------
        for name, spec in qdef.get("vars", {}).items():
            env[name] = _sample_var(spec)

        # ----------------------------
        # Derived: dependency-aware eval
        # ----------------------------
        derived_specs: Dict[str, Any] = dict(qdef.get("derived", {}) or {})
        visiting: set[str] = set()

        def compute(name: str) -> None:
            if name in env:
                return
            if name in visiting:
                raise ValueError(f"Cyclic derived dependency involving '{name}'")
            if name not in derived_specs:
                raise KeyError(f"Unknown reference: ${name}")

            visiting.add(name)
            spec = derived_specs[name]

            # sampler-style derived (kind:int/choice)
            if isinstance(spec, dict) and "kind" in spec:
                env[name] = _sample_var(spec)
                visiting.remove(name)
                return

            # expression-style derived: compute any missing deps first
            for ref in _collect_refs(spec):
                dep = ref[1:]
                if dep in env:
                    continue
                if dep in derived_specs:
                    compute(dep)
                else:
                    # not a var, not a derived => truly unknown
                    raise KeyError(f"Unknown reference: {ref}")

            env[name] = _eval_expr(spec, env, limits)
            visiting.remove(name)

        for dname in derived_specs.keys():
            compute(dname)

        # ----------------
        # constraints
        # ----------------
        ok = True
        for cexpr in qdef.get("constraints", []):
            if not bool(_eval_expr(cexpr, env, limits)):
                ok = False
                break
        if not ok:
            continue

        # prompt
        prompt = _render(qdef["prompt"], env)

        # choices
        choices = _build_choices(qdef["choices"], env, limits)

        # mark exact choice if needed (for choice_is_exact)
        choice_spec = qdef.get("choices", {})
        choice_mode = choice_spec.get("mode")

        if choice_mode == "two_decimal_one_distractor":
            exact = float(_resolve(choice_spec["exact"], env))
            places = int(choice_spec.get("round_places", 3))
            exact_s = _fmt_num(exact, places)
            for ch in choices:
                ch["is_exact"] = (ch["label"] == exact_s)

        elif choice_mode == "computed_items" and "exact" in choice_spec:
            exact = _resolve(choice_spec["exact"], env)
            for ch in choices:
                ch["is_exact"] = (ch.get("value") == exact)

        # shuffle if configured
        if bool(game.get("rules", {}).get("shuffle_choices", True)):
            random.shuffle(choices)

        # correct index
        answer_spec = dict(qdef["answer"])

        if answer_spec.get("mode") == "choice_is_exact" and "exact" not in answer_spec:
            choice_spec = qdef.get("choices", {})
            if "exact" in choice_spec:
                answer_spec["exact"] = choice_spec["exact"]

        correct_index = _pick_correct_index(answer_spec, choices, env)

        # explain
        explain_t = qdef.get("explain", "")
        answer_places = qdef.get("answer", {}).get("display_places", None)

        explain = (
            _render(
                explain_t,
                env,
                choices,
                expr_places=answer_places,
                choice_value_places=answer_places,
            )
            if explain_t else ""
        )

        return BuiltQuestion(
            qid=qdef["id"],
            prompt=prompt,
            choices=choices,
            correct_index=correct_index,
            explain=explain,
            env=env
        )

    raise RuntimeError(f"Could not build question for {qdef.get('id')} after {max_resamples} resamples")
