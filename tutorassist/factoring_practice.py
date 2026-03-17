# factoring_practice.py
# TutorAssist-style skill module: Factoring Practice (with SymPy checking)

import streamlit as st
from streamlit import session_state as ss
import random
import time
import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from sympy import default_sort_key
import json
import uuid
from shared.google_db import append_factoring_round, append_factoring_attempt

# ==============================
# 🔣 SymPy setup
# ==============================

j, k = sp.symbols("j k")
TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,  # allows 27(2j+3k)
    convert_xor,                          # allows j^2
)

def pretty(expr: sp.Expr) -> str:
    # Removes zero terms automatically
    return sp.sstr(sp.expand(expr)).replace("**", "^")

def _prep_user_expr(s: str) -> str:
    if s is None:
        return ""

    s = s.strip()

    # Normalize symbols
    s = s.replace("·", "*")
    s = s.replace("−", "-")      # unicode minus
    s = s.replace("–", "-")      # another unicode minus
    s = s.replace("²", "**2")
    s = s.replace("³", "**3")

    # 🔧 FORCE explicit multiplication:
    # 8( ... ) -> 8*( ... )
    s = re.sub(r'(\d)\s*\(', r'\1*(', s)

    # j( ... ) -> j*( ... )
    s = re.sub(r'([a-zA-Z])\s*\(', r'\1*(', s)

    # )( -> )*(
    s = re.sub(r'\)\s*\(', r')*(', s)

    # 2j -> 2*j
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)

    return s

def parse_user_expr(user_text: str):
    """Parse user text into a SymPy expression, supporting implicit multiplication."""
    txt = _prep_user_expr(user_text)
    if not txt:
        return None
    try:
        return parse_expr(
            txt,
            local_dict={"j": j, "k": k},
            transformations=TRANSFORMS,
            evaluate=False
        )
    except Exception:
        return None

# ==============================
# 🔧 Random helpers (with negatives)
# ==============================

def nz_int(lo, hi, exclude=None):
    """Non-zero integer in [lo, hi], optionally excluding a set."""
    exclude = set(exclude or [])
    while True:
        n = random.randint(lo, hi)
        if n != 0 and n not in exclude:
            return n

def factor_pairs(n: int):
    """
    Return integer factor pairs for n.

    - For n > 0: (1, n), (2, n/2), ... and also (-1, -n), (-2, -n/2), ...
    - For n < 0: (-1, |n|), (-2, |n|/2), ... and (1, -|n|), (2, -|n|/2), ...
    - For n == 0: return [] (special case handled in UI)
    """
    n = int(n)
    if n == 0:
        return []

    m = abs(n)
    pairs = []
    i = 1
    while i * i <= m:
        if m % i == 0:
            a = i
            b = m // i

            if n > 0:
                pairs.append((a, b))
                pairs.append((-a, -b))
            else:
                pairs.append((-a, b))
                pairs.append((a, -b))

        i += 1

    # sort by |a| then |b| for a stable display
    pairs = sorted(pairs, key=lambda t: (abs(t[0]), abs(t[1]), t[0], t[1]))
    return pairs

# ==============================
# 🧩 Problem Generators
# Each returns: (display_str, target_expr, final_answers)
# ==============================
def gen_common_factor():
    """
    Generate a polynomial with a nontrivial common factor.
    Examples:
      4j - 8
      6j - 9k + 12
      5j^2 - 10j
    """

    # -------------------------
    # 1) Choose common factor
    # -------------------------
    g = random.choice([2, 3, 4, 5, 6])

    var_factor = random.choice([1, j, k])

    common = g * var_factor

    # -------------------------
    # 2) Build primitive inner polynomial (2 or 3 terms)
    # -------------------------
    num_terms = random.choice([2, 3])

    terms = []

    for _ in range(num_terms):
        term_type = random.choice(["j", "k", "j2", "jk", "const"])

        if term_type == "j":
            term = nz_int(-6, 6) * j
        elif term_type == "k":
            term = nz_int(-6, 6) * k
        elif term_type == "j2":
            term = nz_int(-4, 4) * j**2
        elif term_type == "jk":
            term = nz_int(-4, 4) * j * k
        else:
            term = nz_int(-6, 6)

        if term != 0:
            terms.append(term)

    # Ensure at least 2 terms
    if len(terms) < 2:
        return gen_common_factor()

    inner = sum(terms)

    # If inner collapsed to constant or monomial, retry
    if not isinstance(inner, sp.Add):
        return gen_common_factor()

    # -------------------------
    # 3) Make inner primitive (remove numeric GCD)
    # -------------------------
    try:
        poly = sp.Poly(inner, j, k, domain="ZZ")
        coeffs = poly.coeffs()
        g_inner = abs(sp.gcd_list(coeffs))
        if g_inner > 1:
            inner = sp.expand(inner / g_inner)
    except Exception:
        return gen_common_factor()

    # -------------------------
    # 4) Build target, then refactor properly to get TRUE GCF
    # -------------------------
    raw = sp.expand(common * inner)

    # Pull out full content (numeric + variable)
    factored_full = sp.factor_terms(raw)

    # Ensure it's a Mul (something factored)
    if not isinstance(factored_full, sp.Mul):
        # Nothing to factor? Try again.
        return gen_common_factor()

    # Split into outer * inner
    outer, inner2 = factored_full.args[0], sp.Mul(*factored_full.args[1:], evaluate=False)

    # Make sure inner is not still factorable by a monomial
    # (i.e., no common symbol or number factor left)
    test = sp.factor_terms(inner2)
    if test != inner2:
        # Still factorable → reject and regenerate
        return gen_common_factor()

    # Build final unevaluated factored form
    factored = sp.Mul(outer, inner2, evaluate=False)
    target = sp.expand(factored)

    disp = pretty(target)

    # Store canonical final answers (including sign flip)
    final_answers = {
        canon_key(canon(factored)),
        canon_key(canon(mul_noexpand(-outer, -inner2))),
    }

    disp = pretty(target)

    # Store canonical final answers (including sign flip)
    final_answers = {
        canon_key(canon(factored)),
        canon_key(canon(mul_noexpand(-common, -inner))),
    }

    return disp, target, final_answers

def gen_trinomial_a1():
    r1 = nz_int(-9, 9)
    r2 = nz_int(-9, 9)

    f1 = j - r1
    f2 = j - r2

    target = sp.expand(f1 * f2)
    disp = pretty(target)

    final_answers = {
        canon_key(f1 * f2),
        canon_key(f2 * f1),
    }

    return disp, target, final_answers

def gen_trinomial_aN():
    a = random.randint(2, 6)
    b = nz_int(-9, 9)
    d = nz_int(-9, 9)

    f1 = a*j + b
    f2 = j + d

    target = sp.expand(f1 * f2)
    disp = pretty(target)

    final_answers = {
        canon_key(f1 * f2),
        canon_key(f2 * f1),
    }

    return disp, target, final_answers

def gen_diff_squares():
    n = random.randint(2, 15)

    f1 = j - n
    f2 = j + n

    target = j**2 - n**2
    disp = pretty(target)

    final_answers = {
        canon_key(f1 * f2),
        canon_key(f2 * f1),
    }

    return disp, target, final_answers

def gen_trinomial_a1_gcf():
    """
    Like gen_trinomial_a1, but with an extra overall common factor g.
    Target is expanded (trinomial with a leading coefficient g),
    but final answer must include the common factor and both binomials.
    """
    g = random.choice([2, 3, 4, 5, 6, 8, 10, 12])
    # allow negative overall scale sometimes
    if random.random() < 0.35:
        g = -g

    r1 = nz_int(-9, 9)
    r2 = nz_int(-9, 9, exclude=[r1])

    f1 = sp.Add(j, -sp.Integer(r1), evaluate=False)  # (j - r1)
    f2 = sp.Add(j, -sp.Integer(r2), evaluate=False)  # (j - r2)

    factored = mul_noexpand(sp.Integer(g), f1, f2)   # g*(j-r1)*(j-r2)
    target = sp.expand(factored)
    disp = pretty(target)

    # Canon already sorts/normalizes, so order doesn't matter.
    # Add a sign-flip equivalent (negate g, negate ONE binomial).
    f1_flip = sp.Add(-j, sp.Integer(r1), evaluate=False)  # (r1 - j) = -(j - r1)
    factored_flip = mul_noexpand(sp.Integer(-g), f1_flip, f2)

    final_answers = {
        canon_key(factored),
        canon_key(factored_flip),
    }

    return disp, target, final_answers


def gen_diff_squares_gcf():
    """
    Difference of squares with an overall common factor g:
        g*(j-n)(j+n)  -> expanded target
    Student must pull out g and then use DOS.
    """
    g = random.choice([2, 3, 4, 5, 6, 8, 10, 12])
    if random.random() < 0.35:
        g = -g

    n = random.randint(2, 15)

    f1 = sp.Add(j, -sp.Integer(n), evaluate=False)  # (j - n)
    f2 = sp.Add(j,  sp.Integer(n), evaluate=False)  # (j + n)

    factored = mul_noexpand(sp.Integer(g), f1, f2)
    target = sp.expand(factored)
    disp = pretty(target)

    # Sign-flip equivalent: -g*(n-j)*(j+n)
    f1_flip = sp.Add(sp.Integer(n), -j, evaluate=False)  # (n - j) = -(j - n)
    factored_flip = mul_noexpand(sp.Integer(-g), f1_flip, f2)

    final_answers = {
        canon_key(factored),
        canon_key(factored_flip),
    }

    return disp, target, final_answers

def gen_sum_squares():
    n = random.randint(2, 15)

    target = j**2 + n**2
    disp = pretty(target)

    # No real factorization exists
    final_answers = set()   # empty means "irreducible"

    return disp, target, final_answers

def gen_vertex_form():
    """
    Level 6: Convert standard form to vertex form:
        aj^2 + bj + c  →  a(j - h)^2 + k
    with a in 1..6
    """

    # Choose a, h, k
    a = random.randint(1, 6)
    h = nz_int(-6, 6)
    k = random.randint(-20, 20)

    # Build vertex form (this is the ONLY accepted final form)
    # Force a, h, k to be SymPy Integers for stable structure/printing
    aS = sp.Integer(a)
    hS = sp.Integer(h)
    kS = sp.Integer(k)

    square_part = mul_noexpand(aS, sp.Pow(sp.Add(j, -hS, evaluate=False), 2, evaluate=False))

    if kS == 0:
        vertex = square_part
    else:
        vertex = sp.Add(square_part, kS, evaluate=False)

    # Expand to get standard form (the question)
    target = sp.expand(vertex)
    disp = pretty(target)

    # Ensure the stored final answer is vertex form (not expanded, not factored)
    final_answers = {canon_key(canon(vertex))}

    return disp, target, final_answers, vertex

def gen_perfect_square_trinomial_gcf():
    """
    Perfect square trinomial with optional GCF:
        g*(j ± a)^2  -> expanded target
    Final should be g*(j ± a)^2 (structure).
    """
    g = random.choice([1, 2, 3, 4, 5, 6, 8, 10, 12])
    if g != 1 and random.random() < 0.35:
        g = -g  # sometimes negative overall

    a = random.randint(2, 12)
    sign = random.choice([1, -1])  # +a or -a inside

    inner = sp.Add(j, sp.Integer(sign * a), evaluate=False)  # (j + a) or (j - a)
    factored = mul_noexpand(sp.Integer(g), sp.Pow(inner, 2, evaluate=False))
    target = sp.expand(factored)
    disp = pretty(target)

    # Include the "flipped" inner sign variant that is equivalent only when g also flips? (square kills sign)
    # For squares, (-(j+a))^2 is same as (j+a)^2, so we don't need extra factor-order variants.
    final_answers = {canon_key(factored)}

    return disp, target, final_answers


def gen_diff_cubes_gcf():
    """
    Difference of cubes with optional GCF:
        g*(A^3 - B^3) -> expanded target
    Where A and B are monomials like (m*j) or (n*k) or constants.
    Final: g*(A - B)*(A^2 + A*B + B^2)
    """
    g = random.choice([1, 2, 3, 4, 5, 6, 8, 10, 12])
    if g != 1 and random.random() < 0.35:
        g = -g

    # Choose A as a*j or a*j (always includes a variable)
    a = random.choice([1, 2, 3])
    A = mul_noexpand(sp.Integer(a), j)

    # Choose B as either constant b or b*k
    if random.random() < 0.5:
        b = random.choice([1, 2, 3, 4])
        B = sp.Integer(b)
    else:
        b = random.choice([1, 2, 3])
        B = mul_noexpand(sp.Integer(b), k)

    expr = sp.Pow(A, 3, evaluate=False) - sp.Pow(B, 3, evaluate=False)
    target = sp.expand(mul_noexpand(sp.Integer(g), expr))
    disp = pretty(target)

    # Factored form (unevaluated)
    first = sp.Add(A, -B, evaluate=False)  # (A - B)
    second = sp.Add(
        sp.Pow(A, 2, evaluate=False),
        mul_noexpand(A, B),
        sp.Pow(B, 2, evaluate=False),
        evaluate=False
    )
    factored = mul_noexpand(sp.Integer(g), first, second)

    # Also accept the “reversed” linear factor form: (B - A) with second factor negated.
    # (A-B)*S == (B-A)*(-S)
    factored_alt = mul_noexpand(sp.Integer(g), sp.Add(B, -A, evaluate=False), mul_noexpand(-1, second))

    final_answers = {
        canon_key(factored),
        canon_key(factored_alt),
    }

    return disp, target, final_answers


def gen_sum_cubes_gcf():
    """
    Sum of cubes with optional GCF:
        g*(A^3 + B^3) -> expanded target
    Final: g*(A + B)*(A^2 - A*B + B^2)
    """
    g = random.choice([1, 2, 3, 4, 5, 6, 8, 10, 12])
    if g != 1 and random.random() < 0.35:
        g = -g

    a = random.choice([1, 2, 3])
    A = mul_noexpand(sp.Integer(a), j)

    if random.random() < 0.5:
        b = random.choice([1, 2, 3, 4])
        B = sp.Integer(b)
    else:
        b = random.choice([1, 2, 3])
        B = mul_noexpand(sp.Integer(b), k)

    expr = sp.Pow(A, 3, evaluate=False) + sp.Pow(B, 3, evaluate=False)
    target = sp.expand(mul_noexpand(sp.Integer(g), expr))
    disp = pretty(target)

    first = sp.Add(A, B, evaluate=False)  # (A + B)
    second = sp.Add(
        sp.Pow(A, 2, evaluate=False),
        mul_noexpand(-1, mul_noexpand(A, B)),  # -A*B (keep structure)
        sp.Pow(B, 2, evaluate=False),
        evaluate=False
    )
    factored = mul_noexpand(sp.Integer(g), first, second)

    final_answers = {canon_key(factored)}

    return disp, target, final_answers


GENERATORS = {
    1: ("Common Factor (j,k)", gen_common_factor, lambda q: "Find the greatest common factor of all terms."),
    2: ("Trinomial (a = 1)", gen_trinomial_a1, lambda q: "Look for two numbers that multiply to C and add to B."),
    3: ("Trinomial (a ≠ 1)", gen_trinomial_aN, lambda q: "Try factoring by decomposition or grouping."),
    4: ("Difference of Squares", gen_diff_squares, lambda q: "Does this match a² − b² ?"),
    5: ("Sum of Squares (real numbers)", gen_sum_squares, lambda q: "Sum of squares does not factor over the reals."),
    6: ("Complete the Square (Vertex Form)", gen_vertex_form, lambda q: "Rewrite in the form a(j - h)² + k."),
    7: ("Trinomial (a = 1) + GCF", gen_trinomial_a1_gcf,
        lambda q: "First factor out the GCF, then factor the trinomial."),
    8: ("Difference of Squares + GCF", gen_diff_squares_gcf,
        lambda q: "First factor out the GCF, then use a² − b²."),
    9: ("Perfect Square Trinomial (±) with optional GCF", gen_perfect_square_trinomial_gcf,
                  lambda q: "Factor out any GCF, then use (j ± a)² = j² ± 2aj + a²."),
    10: ("Difference of Cubes with optional GCF", gen_diff_cubes_gcf,
                  lambda q: "Factor out any GCF, then use A³ − B³ = (A − B)(A² + AB + B²)."),
    11: ("Sum of Cubes with optional GCF", gen_sum_cubes_gcf,
                  lambda q: "Factor out any GCF, then use A³ + B³ = (A + B)(A² − AB + B²).")
}
# ==============================
# 💡 Hint System
# ==============================
def build_hints_common_factor(q):
    expr = q["target_expr"]

    # Try to extract some structure for better messaging
    try:
        poly = sp.Poly(expr, j, k)
        coeffs = [int(c) for c in poly.coeffs() if c.is_Integer]
        if coeffs:
            g = abs(sp.gcd_list(coeffs))
        else:
            g = None
    except Exception:
        g = None

    hints = [
        "Look for the greatest common factor (GCF) in all terms.",
        "First check the coefficients: what number divides all of them?",
        "Then check the variables: do all terms share j or k?",
    ]

    if g and g > 1:
        hints.append(f"The numeric GCF of the coefficients is {g}.")

    hints += [
        "Factor the GCF out of every term.",
        "Check that the expression inside the brackets cannot be factored further."
    ]

    return hints

def build_hints_trinomial_a1(q, expr_override=None):
    expr = expr_override if expr_override is not None else q["target_expr"]
    poly = sp.Poly(sp.expand(expr), j)

    coeffs = poly.all_coeffs()
    if len(coeffs) != 3:
        return [
            "Look for two numbers that multiply to the constant term.",
            "Those two numbers must add to the middle coefficient.",
            "Rewrite the middle term using those two numbers, then factor by grouping."
        ]

    a = int(coeffs[0])
    b = int(coeffs[1])
    c = int(coeffs[2])

    if a == 1:
        hints = [
            f"Find two numbers that multiply to {c}.",
            f"Those two numbers must add to {b}.",
            "Decide if the two numbers should be both positive, both negative, or one of each.",
            f"List factor pairs of {abs(c)} and test their sums.",
            "Rewrite the middle term using those two numbers, then factor by grouping."
        ]
    else:
        ac = a * c
        hints = [
            f"Find two numbers that multiply to {ac}.",
            f"Those two numbers must add to {b}.",
            "Decide if the two numbers should be both positive, both negative, or one of each.",
            f"List factor pairs of {abs(ac)} and test their sums.",
            "Rewrite the middle term using those two numbers, then factor by grouping."
        ]

    return hints

def build_hints_trinomial_aN(q):
    expr = q["target_expr"]
    poly = sp.Poly(expr, j)

    a = int(poly.coeffs()[0])
    b = int(poly.coeffs()[1])
    c = int(poly.coeffs()[2])

    ac = a * c

    hints = [
        f"First, multiply a·c = {a} × {c} = {ac}.",
        f"Now find two numbers that multiply to {ac} and add to {b}.",
        "Will the two numbers be both positive, both negative, or one of each?",
        f"List factor pairs of {abs(ac)} and test their sums.",
        "Rewrite the middle term using those two numbers, then factor by grouping."
    ]
    return hints

def build_hints_diff_squares(q):
    expr = q["target_expr"]

    hints = [
        "Does this match the pattern a² − b² ?",
        "A difference of squares always factors as (a − b)(a + b).",
        "What is a? (the square root of the first term)",
        "What is b? (the square root of the second term)",
        "Write the two factors using (a − b)(a + b)."
    ]

    return hints

def build_hints_trinomial_a1_gcf(q):
    current_expr = q.get("current_expr", q["target_expr"])
    target_expr = q["target_expr"]

    # If the current expression is unchanged, student has not yet removed the GCF
    if canon_key(current_expr) == canon_key(target_expr):
        return [
            "First: factor out the greatest common factor (GCF) from all terms.",
            "After removing the GCF, you'll have a trinomial with leading coefficient 1.",
        ] + build_hints_trinomial_a1(q, expr_override=current_expr)

    # Once the GCF has been removed, switch to hints based on the reduced trinomial
    return build_hints_trinomial_a1(q, expr_override=current_expr)

def build_hints_diff_squares_gcf(q):
    return [
        "First: factor out the greatest common factor (GCF) from all terms.",
        "After removing the GCF, check for the pattern a² − b².",
        "Then factor as (a − b)(a + b).",
    ]

def build_hints_sum_squares(q):
    hints = [
        "This is a sum of squares: a² + b².",
        "Over the real numbers, a sum of squares does not factor.",
        "So the correct conclusion is that this expression is irreducible (or prime).",
        "Type: irreducible, prime, or cannot be factored."
    ]

    return hints

def build_hints_vertex_form(q):
    expr = q["target_expr"]

    # Extract a, b, c if possible
    try:
        poly = sp.Poly(expr, j)
        a = int(poly.coeffs()[0])
        b = int(poly.coeffs()[1])
        c = int(poly.coeffs()[2])
    except Exception:
        a = b = c = None

    hints = [
        "You want to rewrite this in the form a(j − h)² + k.",
        "Group the j² and j terms together.",
    ]

    if a is not None and a != 1:
        hints.append(f"First factor {a} out of the j² and j terms.")

    hints += [
        "Complete the square inside the brackets.",
        "Remember to add and subtract the same number to keep the expression balanced.",
        "Simplify the constant outside the square.",
        "Write the final result in the form a(j − h)² + k."
    ]

    return hints

def build_hints_perfect_square_trinomial_gcf(q):
    return [
        "First: check for a common factor (GCF) across all terms and factor it out.",
        "Perfect square pattern: (j ± a)² = j² ± 2aj + a².",
        "Look at the constant term: it should be a² after removing any GCF.",
        "Then the middle term should be ±2a.",
    ]


def build_hints_diff_cubes_gcf(q):
    return [
        "First: factor out any common factor (GCF) from all terms.",
        "Difference of cubes: A³ − B³ = (A − B)(A² + AB + B²).",
        "Identify cube roots A and B (including coefficients).",
        "Then apply the formula carefully (note the + + in the second factor).",
    ]


def build_hints_sum_cubes_gcf(q):
    return [
        "First: factor out any common factor (GCF) from all terms.",
        "Sum of cubes: A³ + B³ = (A + B)(A² − AB + B²).",
        "Identify cube roots A and B (including coefficients).",
        "Then apply the formula carefully (note the − in the middle of the second factor).",
    ]

HINT_BUILDERS = {
    1: build_hints_common_factor,
    2: build_hints_trinomial_a1,
    3: build_hints_trinomial_aN,
    4: build_hints_diff_squares,
    5: build_hints_sum_squares,
    6: build_hints_vertex_form,
    7: build_hints_trinomial_a1_gcf,
    8: build_hints_diff_squares_gcf,
    9: build_hints_perfect_square_trinomial_gcf,
    10: build_hints_diff_cubes_gcf,
    11: build_hints_sum_cubes_gcf
}

def build_hints_for_question(q):
    builder = HINT_BUILDERS.get(q["level"])
    if not builder:
        return []
    return builder(q)


# ==============================
# 🧠 Session Engine
# ==============================
def _factoring_username() -> str:
    for key in ("username", "user_name", "student_name", "student", "name"):
        v = ss.get(key)
        if v:
            return str(v).strip()
    return "unknown"

def start_factoring_session(num_questions, levels):
    questions = []
    for _ in range(num_questions):
        lvl = random.choice(levels)
        _, gen, _ = GENERATORS[lvl]

        out = gen()
        if len(out) == 4:
            disp, target_expr, final_answers, vertex_expr = out
        else:
            disp, target_expr, final_answers = out
            vertex_expr = None

        questions.append({
            "question": disp,
            "target_expr": target_expr,
            "final_answers": final_answers,
            "vertex_final_expr": vertex_expr,   # 👈 NEW
            "level": lvl,
            "attempts": 0,
            "hints_used": 0,
            "available_hints": None,
            "hints_shown": [],
            "hint_index": 0,
            "correct": False,
            "first_try_correct": False,
            "user_answer": "",
            "current_expr": target_expr,
            "steps": [],
            "last_message": "",
        })

    ss.factoring = {
        "start_time": time.time(),
        "questions": questions,
        "current": 0,
        "finished": False,
        "round_id": uuid.uuid4().hex[:10],
        "round_key": f"factoring_{int(time.time())}_{uuid.uuid4().hex[:6]}",
        "levels_selected": list(levels),
        "num_questions": int(num_questions),
        "round_logged": False,
        "hint_view_index": -1,
    }

    for i, q in enumerate(ss.factoring["questions"], start=1):
        q["question_seq"] = i
        q["question_start_time"] = time.time()
        q["question_total_response_time"] = 0.0
        q["invalid_steps"] = 0
        q["step_events"] = []

def mul_noexpand(*args):
    """Multiply without SymPy distributing over addition."""
    return sp.Mul(*args, evaluate=False)

def _normalize_numbers(expr):
    # If expression is purely numeric, evaluate it
    if expr.is_Number:
        return sp.Integer(expr)

    if isinstance(expr, sp.Mul):
        args = []
        num = sp.Integer(1)
        for a in expr.args:
            a = _normalize_numbers(a)
            if a.is_Number:
                num *= a
            else:
                args.append(a)

        if not args:
            return sp.Integer(num)

        if num != 1:
            args = [num] + args

        if len(args) == 1:
            return args[0]

        return sp.Mul(*args, evaluate=False)

    if isinstance(expr, sp.Add):
        args = []
        num = sp.Integer(0)
        for a in expr.args:
            a = _normalize_numbers(a)
            if a.is_Number:
                num += a
            else:
                args.append(a)

        if num != 0:
            args.append(sp.Integer(num))

        if not args:
            return sp.Integer(0)

        if len(args) == 1:
            return args[0]

        return sp.Add(*args, evaluate=False)

    if isinstance(expr, sp.Pow):
        base = _normalize_numbers(expr.base)
        return sp.Pow(base, expr.exp, evaluate=False)

    return expr

def canon(expr: sp.Expr) -> sp.Expr:

    """Canonicalize for commutative comparison, WITH flattening."""
    expr = _normalize_numbers(expr)
    if isinstance(expr, sp.Mul):
        args = []
        for a in expr.args:
            ca = canon(a)
            if isinstance(ca, sp.Mul):
                args.extend(ca.args)   # 🔥 FLATTEN
            else:
                args.append(ca)
        args = sorted(args, key=default_sort_key)
        return sp.Mul(*args, evaluate=False)

    if isinstance(expr, sp.Add):
        args = []
        for a in expr.args:
            ca = canon(a)
            if isinstance(ca, sp.Add):
                args.extend(ca.args)   # 🔥 FLATTEN
            else:
                args.append(ca)
        args = sorted(args, key=default_sort_key)
        return sp.Add(*args, evaluate=False)

    if isinstance(expr, sp.Pow):
        return sp.Pow(canon(expr.base), expr.exp, evaluate=False)

    return expr

def canon_key(expr: sp.Expr) -> str:
    return sp.srepr(canon(expr))

# ------------------------------
# Local helpers (robust ordering + "done" detection)
# ------------------------------
def _canon(expr: sp.Expr) -> sp.Expr:
    """Canonicalize ordering so commutative reordering doesn't break comparisons."""
    if isinstance(expr, sp.Mul):
        args = sorted((_canon(a) for a in expr.args), key=default_sort_key)
        return sp.Mul(*args, evaluate=False)
    if isinstance(expr, sp.Add):
        args = sorted((_canon(a) for a in expr.args), key=default_sort_key)
        return sp.Add(*args, evaluate=False)
    if isinstance(expr, sp.Pow):
        return sp.Pow(_canon(expr.base), expr.exp, evaluate=False)
    return expr

def equivalent(a: sp.Expr, b: sp.Expr) -> bool:
    try:
        diff = sp.simplify(sp.expand(a - b))
        return diff == 0
    except Exception:
        return False

def is_fully_factored_expr(expr: sp.Expr) -> bool:
    """
    Treat as fully factored if SymPy's factor() does not change it,
    after canonicalizing (so (a)(b) vs (b)(a) is fine).
    """
    try:
        f = sp.factor(expr)
        return canon_key(expr) == canon_key(f)
    except Exception:
        return False

# ==============================
# 🧠 Reactive (Adaptive) Hints
# ==============================
def get_reactive_hint(q, new_expr: sp.Expr, last_expr: sp.Expr = None):
    """
    Returns a string hint if a recognizable mistake pattern is detected.
    Returns None if no specific reactive hint applies.
    """
    if new_expr is None:
        return "I couldn't understand that expression. Check your parentheses and operators."

    last = last_expr if last_expr is not None else q.get("current_expr", q["target_expr"])

    # 0) Must remain equivalent to original
    try:
        if not equivalent(new_expr, q["target_expr"]):
            return "Your new expression is not equivalent to the original. Check your algebra."
    except Exception:
        pass

    # 🟢 1) Decomposition / regrouping step = VALID PROGRESS
    # If both are sums and number of terms increased, this is a real step (e.g., ac-split)
    try:
        if isinstance(new_expr, sp.Add) and isinstance(last, sp.Add):
            if len(new_expr.args) > len(last.args):
                # This is a decomposition step like: 3j^2+19j+20 -> 3j^2+15j+4j+20
                return None
    except Exception:
        pass

    # 2) Truly no change (same structure)
    try:
        if canon_key(new_expr) == canon_key(last):
            return "That step didn’t change the expression. Try factoring something."
    except Exception:
        pass

    # 3) Expanded instead of factored (product -> sum)
    try:
        old_terms = top_level_term_count(last)
        new_terms = top_level_term_count(new_expr)

        # If they turned a product into a bigger sum, that's expansion
        if new_terms > old_terms and isinstance(last, sp.Mul):
            return "This step expanded the expression. The goal is to factor, not expand."
    except Exception:
        pass

    # 4) Partial progress toward factoring (but not done)
    try:
        if equivalent(new_expr, q["target_expr"]) and not is_fully_factored_expr(new_expr):

            old = last

            # Count top-level factors
            def factor_count(e):
                if isinstance(e, sp.Mul):
                    return len(e.args)
                return 1

            old_factors = factor_count(old)
            new_factors = factor_count(new_expr)

            old_terms = top_level_term_count(old)
            new_terms = top_level_term_count(new_expr)

            # Only praise if structure IMPROVED toward factoring
            if new_factors > old_factors or new_terms < old_terms:
                return "Good start, but this expression can still be factored further."
            else:
                return "This is algebraically correct, but it does not move you closer to a factorization."

    except Exception:
        pass

    return None

def top_level_term_count(expr: sp.Expr) -> int:
    if isinstance(expr, sp.Add):
        return len(expr.args)
    return 1

def to_latex_like(s: str) -> str:
    """
    Convert user-friendly caret powers into LaTeX superscripts.
    Critical: do NOT greedily capture the '-' of the next term.
    Examples:
      j^2        -> j^{2}
      j^-3       -> j^{-3}
      j^(12)     -> j^{12}
      j^(-4)     -> j^{-4}
      2j^2-6j... -> 2j^{2}-6j...
    """
    # Handle parenthesized exponents first: ^( -?\d+ )
    s = re.sub(r"\^\s*\(\s*(-?\d+)\s*\)", r"^{\1}", s)

    # Handle bare integer exponents: ^-?\d+
    s = re.sub(r"\^\s*(-?\d+)", r"^{\1}", s)

    return s

def _expr_text(expr) -> str:
    if expr is None:
        return ""
    try:
        return sp.sstr(expr)
    except Exception:
        return str(expr)

def _is_progress_step(q, prev_expr, new_expr) -> bool:
    if new_expr is None or prev_expr is None:
        return False

    try:
        if not equivalent(new_expr, prev_expr):
            return False

        def factor_count(e):
            return len(e.args) if isinstance(e, sp.Mul) else 1

        old_factors = factor_count(prev_expr)
        new_factors = factor_count(new_expr)
        old_terms = top_level_term_count(prev_expr)
        new_terms = top_level_term_count(new_expr)

        if new_factors > old_factors:
            return True

        if new_terms < old_terms:
            return True

        # Decomposition / regrouping is valid progress for trinomial/grouping levels
        if q.get("level") in {2, 3, 7}:
            if isinstance(prev_expr, sp.Add) and isinstance(new_expr, sp.Add):
                if new_terms > old_terms:
                    return True

        return False

    except Exception:
        return False

def _log_factoring_attempt(
    q,
    *,
    input_text: str,
    parsed_ok: bool,
    expr_before,
    expr_after,
    equivalent_to_target: bool,
    is_done: bool,
    invalid_step: bool,
    invalid_reason: str,
    reactive_hint: str,
):
    try:
        username = _factoring_username()
        round_key = ss.factoring.get("round_key", "")
        round_id = ss.factoring.get("round_id", "")
        now = time.time()
        response_time = max(0.0, now - q.get("question_start_time", now))
        q["question_total_response_time"] = q.get("question_total_response_time", 0.0) + response_time
        q["question_start_time"] = now

        if invalid_step:
            q["invalid_steps"] = q.get("invalid_steps", 0) + 1

        is_progress = _is_progress_step(q, expr_before, expr_after) if expr_after is not None else False

        attempt_id = f"factatt_{uuid.uuid4().hex[:12]}"

        append_factoring_attempt(
            attempt_id=attempt_id,
            username=username,
            round_key=round_key,
            round_id=round_id,
            question_seq=q.get("question_seq", 0),
            level=q.get("level", 0),
            question_text=q.get("question", ""),
            target_expr=_expr_text(q.get("target_expr")),
            input_text=input_text or "",
            parsed_ok=parsed_ok,
            equivalent_to_target=equivalent_to_target,
            is_done=is_done,
            is_progress_step=is_progress,
            invalid_step=invalid_step,
            invalid_reason=invalid_reason or "",
            reactive_hint=reactive_hint or "",
            attempt_number=int(q.get("attempts", 0)),
            response_time=response_time,
            hints_used_so_far=int(q.get("hints_used", 0)),
            factor_tool_used_count=int(q.get("factor_tool_used_count", 0)),
            steps_count=len(q.get("steps", [])),
            current_expr_before=_expr_text(expr_before),
            current_expr_after=_expr_text(expr_after),
        )
    except Exception:
        pass


def _log_factoring_round_once():
    try:
        if not ss.get("factoring"):
            return
        if ss.factoring.get("round_logged"):
            return

        questions = ss.factoring.get("questions", [])
        elapsed = max(0.0, time.time() - ss.factoring.get("start_time", time.time()))
        total = len(questions)
        correct = sum(1 for q in questions if q.get("correct"))
        incorrect = total - correct
        attempts_total = sum(int(q.get("attempts", 0)) for q in questions)
        hints_used_total = sum(int(q.get("hints_used", 0)) for q in questions)
        factor_tool_uses_total = sum(int(q.get("factor_tool_used_count", 0)) for q in questions)
        invalid_steps_total = sum(int(q.get("invalid_steps", 0)) for q in questions)

        response_times = [float(q.get("question_total_response_time", 0.0)) for q in questions if q.get("question_total_response_time", 0.0) > 0]
        avg_response_time = (sum(response_times) / len(response_times)) if response_times else 0.0

        levels_csv = ",".join(str(x) for x in ss.factoring.get("levels_selected", []))
        username = _factoring_username()

        append_factoring_round(
            username=username,
            round_key=ss.factoring.get("round_key", ""),
            round_id=ss.factoring.get("round_id", ""),
            game_name="Factoring Practice",
            questions_served=total,
            questions_completed=total,
            correct=correct,
            incorrect=incorrect,
            attempts_total=attempts_total,
            round_time=elapsed,
            average_response_time=avg_response_time,
            levels_csv=levels_csv,
            hints_used_total=hints_used_total,
            factor_tool_uses_total=factor_tool_uses_total,
            invalid_steps_total=invalid_steps_total,
            completed=True,
            notes="",
        )

        ss.factoring["round_logged"] = True
    except Exception:
        pass

# ==============================
# 🖥️ UI
# ==============================
def factoring_practice():

    st.markdown("""
    <style>
    .block-container {
        padding-top: 1.0rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## 🧮 Factoring Practice")

    if "factoring" not in ss:
        ss.factoring = None

    if "setup_open" not in ss:
        ss.setup_open = True

    # Used to force-clear the input box (by changing widget key)
    if "fact_input_version" not in ss:
        ss.fact_input_version = 0

    # ==============================
    # 🟢 Setup Screen
    # ==============================
    if ss.factoring is None:
        st.markdown("### ⚙️ Practice Setup")

        level_map = {
            1: "Level 1 — Common Factor (j,k)",
            2: "Level 2 — Trinomial (a = 1)",
            3: "Level 3 — Trinomial (a ≠ 1)",
            4: "Level 4 — Difference of Squares",
            5: "Level 5 — Sum of Squares (real numbers)",
            6: "Level 6 — Complete the Square (Vertex Form)",
            7: "Level 7 — Trinomial (a = 1) + GCF",
            8: "Level 8 — Difference of Squares + GCF",
            9: "Level 9 — Perfect Square Trinomial (+GCF)",
            10: "Level 10 — Difference of Cubes (+GCF)",
            11: "Level 11 — Sum of Cubes (+GCF)",
        }

        if "selected_levels" not in ss:
            ss.selected_levels = []

        with st.expander("🧩 Select Question Types", expanded=ss.setup_open):
            st.caption("Choose one or more types:")
            st.pills(
                "Question types",
                options=list(level_map.keys()),
                format_func=lambda x: level_map[x],
                selection_mode="multi",
                key="selected_levels",
            )

        num_q = st.slider("Number of questions", 1, 30, 10)

        st.caption("Tip: You can type answers like 8(6j-5k) or (j-2)(j+3).")

        if not ss.selected_levels:
            st.warning("Select at least one question type to continue.")
        else:
            if st.button("🚀 Start Practice", width="stretch"):
                ss.setup_open = False
                start_factoring_session(num_q, ss.selected_levels)
                ss.fact_input_version += 1
                st.rerun()

        return

    # ==============================
    # 📊 Results Screen
    # ==============================
    if ss.factoring.get("finished"):
        elapsed = time.time() - ss.factoring["start_time"]
        questions = ss.factoring["questions"]

        total = len(questions)
        correct = sum(1 for q in questions if q.get("correct"))
        first_try = sum(1 for q in questions if q.get("first_try_correct"))

        _log_factoring_round_once()

        st.success("✅ Practice Complete!")
        st.markdown(
            f"**Time:** {elapsed:.1f} seconds  \n"
            f"**Score:** {correct} / {total}  \n"
            f"**First-try correct:** {first_try} / {total}"
        )

        with st.expander("📋 Review Questions"):
            for i, q in enumerate(questions, 1):
                icon = "✅" if q.get("correct") else "❌"
                st.markdown(
                    f"**{i}.** {sp.latex(q['target_expr'])}  \n"
                    f"{icon} Final answer: `{q.get('user_answer', '')}`  \n"
                    f"Wrong attempts: {q.get('attempts', 0)}  \n"
                    f"Hints used: {q.get('hints_used', 0)}  \n"
                    f"Factor tool used: {q.get('factor_tool_used_count', 0)}  \n"
                    f"Invalid steps: {q.get('invalid_steps', 0)}"
                )

        if st.button("🔁 New Practice Set"):
            ss.factoring = None
            ss.setup_open = True
            ss.fact_input_version += 1
            st.rerun()
        return

    # ==============================
    # ❓ Question Screen
    # ==============================
    idx = ss.factoring["current"]
    questions = ss.factoring["questions"]
    q = questions[idx]

    # Ensure keys exist
    q.setdefault("attempts", 0)
    q.setdefault("steps", [])
    q.setdefault("last_message", "")
    q.setdefault("correct", False)
    q.setdefault("first_try_correct", False)
    q.setdefault("user_answer", "")
    q.setdefault("current_expr", q["target_expr"])

    # Hints
    q.setdefault("hints_shown", [])
    q.setdefault("hint_index", 0)
    q.setdefault("hints_used", 0)
    q.setdefault("hint_view_index", -1)

    # Factor-pairs tool tracking
    q.setdefault("factor_tool_used", False)
    q.setdefault("factor_tool_used_count", 0)

    # Tracking fields
    q.setdefault("question_seq", idx + 1)
    q.setdefault("question_start_time", time.time())
    q.setdefault("question_total_response_time", 0.0)
    q.setdefault("invalid_steps", 0)

    # Flash correct state
    q.setdefault("flash_correct", False)
    q.setdefault("flash_final_latex", "")

    # If last submit marked as correct, show final briefly then advance
    if q.get("flash_correct"):
        st.success("✅ Fully Factored")

        try:
            st.latex(r"\Large " + sp.latex(q["target_expr"]))
        except Exception:
            st.markdown(f"**Original:** `{str(q['target_expr'])}`")

        if q.get("flash_final_latex"):
            st.latex(r"\Large = " + q["flash_final_latex"])

        time.sleep(2.7)

        q["flash_correct"] = False
        q["last_message"] = ""
        q["steps"] = []
        q["user_answer"] = ""

        ss.pop(f"fp_opened_{idx}", None)

        if idx + 1 >= len(questions):
            ss.factoring["finished"] = True
        else:
            ss.factoring["current"] += 1
            next_idx = ss.factoring["current"]
            questions[next_idx].setdefault("question_start_time", time.time())
            questions[next_idx]["question_start_time"] = time.time()
            questions[next_idx]["last_message"] = ""
            questions[next_idx]["hint_view_index"] = -1

        ss.fact_input_version += 1
        ss.fact_live_input = ""
        st.rerun()

    # Build hints lazily
    if not q.get("available_hints"):
        q["available_hints"] = build_hints_for_question(q)

    left, right = st.columns([3, 1.8])

    with left:
        st.markdown(f"### Question {idx + 1} of {len(questions)}")

        if q["level"] == 6:
            st.latex(r"\LARGE \textbf{Write in vertex form:}\quad " + sp.latex(q["target_expr"]))
        else:
            st.latex(r"\LARGE \textbf{Factor:}\quad " + sp.latex(q["target_expr"]))

    with right:
        hints = q.get("available_hints") or []
        shown = q.get("hints_shown", [])
        view_i = q.get("hint_view_index", -1)

        st.markdown(f"💡 Hints used: **{q.get('hints_used', 0)}**")

        # ---- Hint navigator header ----
        nav1, nav2, nav3 = st.columns([1, 3, 1])

        with nav1:
            prev_disabled = (len(shown) == 0) or (view_i <= 0)
            if st.button("◀", key=f"hint_prev_{idx}", disabled=prev_disabled, width="stretch"):
                q["hint_view_index"] = max(0, view_i - 1)
                st.rerun()

        with nav2:
            total_hints = len(hints)
            shown_count = len(shown)
            if shown_count == 0:
                st.caption(f"Hint 0 of {total_hints}")
            else:
                st.caption(f"Hint {view_i + 1} of {shown_count}")

        with nav3:
            next_disabled = len(hints) == 0 or len(shown) >= len(hints) and view_i >= len(shown) - 1
            if st.button("▶", key=f"hint_next_{idx}", disabled=next_disabled, width="stretch"):
                # Reveal next unseen hint if possible
                if len(shown) < len(hints):
                    q["hints_shown"].append("💡 " + hints[len(shown)])
                    q["hint_index"] = len(q["hints_shown"])
                    q["hint_view_index"] = len(q["hints_shown"]) - 1
                    q["hints_used"] = len(q["hints_shown"])
                else:
                    # Otherwise just move forward through already revealed hints
                    q["hint_view_index"] = min(len(shown) - 1, view_i + 1)

                st.rerun()

        # ---- Hint card ----
        if len(shown) == 0:
            st.markdown(
                """
                <div style="
                    background: rgba(219, 234, 254, 0.35);
                    border: 1px dashed rgba(147, 197, 253, 0.9);
                    border-radius: 14px;
                    padding: 14px 16px;
                    color: #4b5563;
                    line-height: 1.55;
                    min-height: 120px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                ">
                    <div style="font-size: 1.0rem;">Click ▶ to reveal the first hint.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif 0 <= view_i < len(shown):
            st.markdown(
                f"""
                <div style="
                    background: rgba(219, 234, 254, 0.78);
                    border: 1px solid rgba(147, 197, 253, 0.95);
                    border-radius: 14px;
                    padding: 14px 16px;
                    color: #0f4c81;
                    line-height: 1.55;
                    min-height: 120px;
                    display: flex;
                    align-items: flex-start;
                ">
                    <div style="font-size: 1.05rem;">{shown[view_i]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---- Factor pairs tool ----
        with st.popover("🧰 Factor Pairs", width="stretch"):
            if not ss.get(f"fp_opened_{idx}", False):
                q["factor_tool_used"] = True
                q["factor_tool_used_count"] = q.get("factor_tool_used_count", 0) + 1
                ss[f"fp_opened_{idx}"] = True

            st.caption("Enter an integer and I'll list its factor pairs (including negatives when needed).")

            n = st.number_input("Number", value=12, step=1, format="%d", key=f"fp_num_{idx}")

            nn = int(n)
            if nn == 0:
                st.warning("0 has infinitely many factor pairs. Try a non-zero integer.")
            else:
                pairs = factor_pairs(nn)
                st.markdown("**Factor pairs:**")
                st.write(", ".join([f"({a}, {b})" for a, b in pairs]))

    st.markdown("### ✏️ Working:")

    if q.get("last_message"):
        st.info(q["last_message"])

    st.markdown(f"${sp.latex(q['target_expr'])}$")

    for step in q["steps"]:
        txt = step["text"] if isinstance(step, dict) else str(step)
        latex_txt = to_latex_like(txt)
        st.markdown(f"= ${latex_txt}$")

    # ------------------------------
    # Input
    # ------------------------------
    st.markdown("Your answer:")

    st.caption("Use ^ for powers, for example: j^2, k^3, (j+2)^2")

    user_answer = st.text_input(
        "Your answer",
        key=f"fact_answer_box_{ss.fact_input_version}",
        autocomplete="off",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("✅ Submit"):

            user_answer = (user_answer or "").strip()
            prev_expr = q.get("current_expr", q["target_expr"])

            # -------------------------
            # Level 5 special case
            # -------------------------
            if q["level"] == 5:
                u = user_answer.lower().replace(" ", "")
                ok = u in {
                    "irreducible",
                    "prime",
                    "cannotbefactored",
                    "cannotfactor",
                    "norealfactors",
                    "norealfactor",
                }

                if ok:
                    q["user_answer"] = user_answer
                    q["correct"] = True
                    if q["attempts"] == 0:
                        q["first_try_correct"] = True
                    q["last_message"] = "🎉 Correct — this does not factor over the reals."

                    _log_factoring_attempt(
                        q,
                        input_text=user_answer,
                        parsed_ok=True,
                        expr_before=prev_expr,
                        expr_after=prev_expr,
                        equivalent_to_target=True,
                        is_done=True,
                        invalid_step=False,
                        invalid_reason="",
                        reactive_hint="",
                    )

                    q["flash_correct"] = True
                    q["flash_final_latex"] = sp.latex(q["target_expr"])

                    ss.fact_input_version += 1
                    st.rerun()
                else:
                    q["attempts"] += 1
                    q["last_message"] = "❌ For this one, enter: irreducible / prime / cannot be factored."

                    _log_factoring_attempt(
                        q,
                        input_text=user_answer,
                        parsed_ok=True,
                        expr_before=prev_expr,
                        expr_after=prev_expr,
                        equivalent_to_target=True,
                        is_done=False,
                        invalid_step=True,
                        invalid_reason="level5_wrong_text",
                        reactive_hint=q["last_message"],
                    )

                    st.rerun()
                return

            # -------------------------
            # Normal algebraic input
            # -------------------------
            expr_u = parse_user_expr(user_answer)

            if expr_u is None:
                q["attempts"] += 1
                q["last_message"] = "❌ I couldn't parse that. Try (j-2)(j+3) or 8(6j-5k)."

                _log_factoring_attempt(
                    q,
                    input_text=user_answer,
                    parsed_ok=False,
                    expr_before=prev_expr,
                    expr_after=None,
                    equivalent_to_target=False,
                    is_done=False,
                    invalid_step=True,
                    invalid_reason="parse_error",
                    reactive_hint=q["last_message"],
                )

                st.rerun()
                return

            # Must be equivalent to original
            if not equivalent(expr_u, q["target_expr"]):
                q["attempts"] += 1
                rh = get_reactive_hint(q, expr_u, last_expr=prev_expr)
                q["last_message"] = "❌ " + (rh if rh else "This is not equivalent to the original expression.")

                _log_factoring_attempt(
                    q,
                    input_text=user_answer,
                    parsed_ok=True,
                    expr_before=prev_expr,
                    expr_after=expr_u,
                    equivalent_to_target=False,
                    is_done=False,
                    invalid_step=True,
                    invalid_reason="not_equivalent_to_target",
                    reactive_hint=q["last_message"],
                )

                st.rerun()
                return

            # -------------------------
            # Final answer detection
            # -------------------------
            done = False

            try:
                if q.get("final_answers") and canon_key(expr_u) in q["final_answers"]:
                    done = True
            except Exception:
                done = False

            if not done and q.get("level") != 6:
                if is_fully_factored_expr(expr_u):
                    done = True

            if done:

                # Level 6: keep strict vertex-form simplification
                if q["level"] == 6 and q.get("vertex_final_expr") is not None:
                    student_terms = top_level_term_count(expr_u)
                    final_terms = top_level_term_count(q["vertex_final_expr"])

                    if student_terms != final_terms:
                        last_key = canon_key(q["steps"][-1]["expr"]) if q["steps"] else canon_key(prev_expr)
                        if canon_key(expr_u) != last_key:
                            q["steps"].append({"expr": expr_u, "text": user_answer})
                            q["current_expr"] = expr_u
                            q["available_hints"] = build_hints_for_question(q)

                        q["last_message"] = "⚠️ Finish simplifying the constants."
                        ss.fact_input_version += 1

                        _log_factoring_attempt(
                            q,
                            input_text=user_answer,
                            parsed_ok=True,
                            expr_before=prev_expr,
                            expr_after=expr_u,
                            equivalent_to_target=True,
                            is_done=False,
                            invalid_step=False,
                            invalid_reason="",
                            reactive_hint=q["last_message"],
                        )

                        st.rerun()
                        return

                last_key = canon_key(q["steps"][-1]["expr"]) if q["steps"] else canon_key(prev_expr)
                if canon_key(expr_u) != last_key:
                    q["steps"].append({"expr": expr_u, "text": user_answer})

                q["current_expr"] = expr_u
                q["available_hints"] = build_hints_for_question(q)
                q["user_answer"] = user_answer
                q["correct"] = True
                if q["attempts"] == 0:
                    q["first_try_correct"] = True

                q["last_message"] = "🎉 Fully factored!"

                _log_factoring_attempt(
                    q,
                    input_text=user_answer,
                    parsed_ok=True,
                    expr_before=prev_expr,
                    expr_after=expr_u,
                    equivalent_to_target=True,
                    is_done=True,
                    invalid_step=False,
                    invalid_reason="",
                    reactive_hint="",
                )

                q["flash_correct"] = True
                q["flash_final_latex"] = sp.latex(expr_u)

                ss.fact_input_version += 1
                ss.pop(f"fp_opened_{idx}", None)
                st.rerun()
                return

            # -------------------------
            # Equivalent but unchanged
            # -------------------------
            last_key = canon_key(q["steps"][-1]["expr"]) if q["steps"] else canon_key(prev_expr)
            if canon_key(expr_u) == last_key:
                rh = get_reactive_hint(q, expr_u, last_expr=prev_expr)
                q["last_message"] = "⚠️ " + (rh if rh else "That does not change the expression. Try factoring something.")

                _log_factoring_attempt(
                    q,
                    input_text=user_answer,
                    parsed_ok=True,
                    expr_before=prev_expr,
                    expr_after=expr_u,
                    equivalent_to_target=True,
                    is_done=False,
                    invalid_step=True,
                    invalid_reason="no_change",
                    reactive_hint=q["last_message"],
                )

                st.rerun()
                return

            # -------------------------
            # Valid intermediate step
            # -------------------------
            rh = get_reactive_hint(q, expr_u, last_expr=prev_expr)
            progress = _is_progress_step(q, prev_expr, expr_u)

            q["steps"].append({"expr": expr_u, "text": user_answer})
            q["current_expr"] = expr_u
            q["available_hints"] = build_hints_for_question(q)

            if progress:
                q["last_message"] = ("🧠 " + rh) if rh else "✅ Good step — keep factoring."
            else:
                q["last_message"] = ("⚠️ " + rh) if rh else "⚠️ Equivalent, but this does not look like a useful factoring step."

            _log_factoring_attempt(
                q,
                input_text=user_answer,
                parsed_ok=True,
                expr_before=prev_expr,
                expr_after=expr_u,
                equivalent_to_target=True,
                is_done=False,
                invalid_step=not progress,
                invalid_reason="" if progress else "equivalent_but_not_progress",
                reactive_hint=q["last_message"],
            )

            ss.fact_input_version += 1
            st.rerun()
            return

    with col2:
        if st.button("🔄 Restart Set"):
            ss.factoring = None
            ss.setup_open = True
            ss.fact_input_version += 1
            st.rerun()