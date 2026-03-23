# solving_equations_practice.py
# TutorAssist-style skill module: Solving Linear Equations Practice (step checking with SymPy)

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
from shared.google_db import get_published_pdf_preview_url_by_generator_id

# ==============================
# 🔣 SymPy setup (match factoring_practice style)
# ==============================
j = sp.Symbol("j")
TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

def _prep_user_expr(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = s.replace("·", "*")
    s = s.replace("−", "-")
    s = s.replace("–", "-")
    s = s.replace("²", "**2")

    # Force explicit multiplication similar to factoring_practice.py
    s = re.sub(r'(\d)\s*\(', r'\1*(', s)          # 8( -> 8*(
    s = re.sub(r'([a-zA-Z])\s*\(', r'\1*(', s)    # j( -> j*(
    s = re.sub(r'\)\s*\(', r')*(', s)             # )( -> )*(
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)    # 2j -> 2*j
    return s

def parse_user_expr(user_text: str):
    txt = _prep_user_expr(user_text)
    if not txt:
        return None
    try:
        return parse_expr(
            txt,
            local_dict={"j": j},
            transformations=TRANSFORMS,
            evaluate=False
        )
    except Exception:
        return None

# ==============================
# 🔧 helpers
# ==============================
def nz_int(lo, hi, exclude=None):
    exclude = set(exclude or [])
    while True:
        n = random.randint(lo, hi)
        if n != 0 and n not in exclude:
            return n

def to_latex_like(s: str) -> str:
    # same idea as in factoring_practice.py (caret to exponent)
    s = re.sub(r"\^\s*\(\s*(-?\d+)\s*\)", r"^{\1}", s)
    s = re.sub(r"\^\s*(-?\d+)", r"^{\1}", s)
    return s

def eq_key(lhs: sp.Expr, rhs: sp.Expr) -> str:
    # canonical “difference” representation
    try:
        return sp.srepr(sp.simplify(sp.expand(lhs - rhs)))
    except Exception:
        return sp.srepr(lhs - rhs)

def is_numeric_constant(expr: sp.Expr) -> bool:
    """
    Allow multiplying/dividing by numeric constants only (Integer/Rational).
    """
    try:
        expr_s = sp.simplify(expr)
        return expr_s.is_Number
    except Exception:
        return False

def equivalent_equations(old_lhs: sp.Expr, old_rhs: sp.Expr, new_lhs: sp.Expr, new_rhs: sp.Expr) -> bool:
    """
    Check that equations have same solution set by comparing differences:
      old_diff = old_lhs - old_rhs
      new_diff = new_lhs - new_rhs
    Accept if:
      - new_diff == old_diff (algebraic rearrangement)
      - OR new_diff == c*old_diff where c is nonzero numeric constant (multiply/divide both sides)
    """
    try:
        old_diff = sp.simplify(sp.expand(old_lhs - old_rhs))
        new_diff = sp.simplify(sp.expand(new_lhs - new_rhs))

        # exact same constraint
        if sp.simplify(new_diff - old_diff) == 0:
            return True

        # constant multiple
        if old_diff == 0:
            # old equation is identity (rare in our generators); only accept if new_diff also 0
            return sp.simplify(new_diff) == 0

        ratio = sp.simplify(new_diff / old_diff)
        if is_numeric_constant(ratio) and sp.simplify(ratio) != 0:
            return True

        return False
    except Exception:
        return False

def same_operation(pL, pR) -> bool:
    if pL is None or pR is None:
        return False
    if pL[0] != pR[0]:
        return False
    try:
        return sp.simplify(pL[1] - pR[1]) == 0
    except Exception:
        return False

# def detect_same_operation(old_lhs: sp.Expr, old_rhs: sp.Expr, new_lhs: sp.Expr, new_rhs: sp.Expr):
#     """
#     Detects whether the step is:
#       - add/subtract same term T to both sides
#       - multiply/divide both sides by same numeric constant C
#     Returns (ok: bool, op_desc: str or None)
#     """
#     try:
#         # ADD/SUB: new = old + T
#         dL = sp.simplify(sp.expand(new_lhs - old_lhs))
#         dR = sp.simplify(sp.expand(new_rhs - old_rhs))
#         if sp.simplify(dL - dR) == 0:
#             # same additive change both sides
#             if sp.simplify(dL) == 0:
#                 return False, "That didn’t change the equation."
#             return True, f"Added {sp.sstr(dL)} to both sides."
#
#         # MULT/DIV: new = old * C  (numeric constant only)
#         # Guard: avoid dividing by 0/expressions; allow only numeric constants
#         if old_lhs == 0 or old_rhs == 0:
#             # still might be valid, but keep it simple; fall back to equivalence-only
#             return True, None
#
#         rL = sp.simplify(new_lhs / old_lhs)
#         rR = sp.simplify(new_rhs / old_rhs)
#
#         if sp.simplify(rL - rR) == 0 and is_numeric_constant(rL) and sp.simplify(rL) != 0:
#             return True, f"Multiplied both sides by {sp.sstr(sp.simplify(rL))}."
#
#         return False, "Make sure you apply the same operation to BOTH sides (add/subtract the same term, or multiply/divide by the same constant)."
#     except Exception:
#         return False, "I couldn’t verify the operation on both sides—check your algebra and parentheses."

def bracket_count(expr: sp.Expr) -> int:
    """
    Count multiplicative bracket groups like a*(...) that still need expansion.
    """
    try:
        count = 0
        for node in sp.preorder_traversal(expr):
            if isinstance(node, sp.Mul):
                args = list(node.args)
                has_add_child = any(isinstance(arg, sp.Add) for arg in args)
                has_non_add_child = any(not isinstance(arg, sp.Add) for arg in args)
                if has_add_child and has_non_add_child:
                    count += 1
        return count
    except Exception:
        return 0

def additive_term_count(expr: sp.Expr) -> int:
    """
    Count number of additive terms after expand.
    7*j - 35 => 2 terms
    7*j      => 1 term
    """
    try:
        e = sp.expand(expr)
        terms = e.as_ordered_terms()
        return len(terms)
    except Exception:
        return 999

def measure_progress(lhs: sp.Expr, rhs: sp.Expr):
    """
    Rough progress signal for ‘helping to solve for j’.
    Returns a dict of simple metrics.
    """
    try:
        lhs_s = sp.simplify(sp.expand(lhs))
        rhs_s = sp.simplify(sp.expand(rhs))

        # Count occurrences of j at top-level (linear-ish)
        # We use polynomial degree/terms where possible
        polyL = sp.Poly(lhs_s, j, domain="QQ") if lhs_s.has(j) else None
        polyR = sp.Poly(rhs_s, j, domain="QQ") if rhs_s.has(j) else None

        degL = polyL.degree() if polyL else 0
        degR = polyR.degree() if polyR else 0

        # coefficient magnitude of j (linear)
        aL = polyL.coeffs()[0] if polyL and polyL.degree() == 1 else (polyL.LC() if polyL else 0)
        aR = polyR.coeffs()[0] if polyR and polyR.degree() == 1 else (polyR.LC() if polyR else 0)

        # is j isolated?
        isolated = (sp.simplify(lhs_s - j) == 0 and not rhs_s.has(j)) or (sp.simplify(rhs_s - j) == 0 and not lhs_s.has(j))

        # total “j presence”
        j_presence = int(lhs_s.has(j)) + int(rhs_s.has(j))

        # term-count on the side that currently contains j (helps detect: 7j-35 -> 7j)
        j_side_terms = None
        if lhs_s.has(j) and not rhs_s.has(j):
            j_side_terms = additive_term_count(lhs_s)
        elif rhs_s.has(j) and not lhs_s.has(j):
            j_side_terms = additive_term_count(rhs_s)
        else:
            # j on both sides (or neither) — treat as "not simplified"
            j_side_terms = additive_term_count(lhs_s) + additive_term_count(rhs_s)

        return {
            "isolated": bool(isolated),
            "j_presence": j_presence,   # 0..2
            "deg_sum": int(degL + degR),
            "j_side_terms": int(j_side_terms),
        }
    except Exception:
        return {"isolated": False, "j_presence": 2, "deg_sum": 2, "j_side_terms": 2}

# def solved_value_if_isolated(lhs: sp.Expr, rhs: sp.Expr):
#     """
#     If equation is j = number (or number = j), return that number as a SymPy expr.
#     Otherwise return None.
#     """
#     try:
#         lhs_s = sp.simplify(sp.expand(lhs))
#         rhs_s = sp.simplify(sp.expand(rhs))
#
#         if sp.simplify(lhs_s - j) == 0 and not rhs_s.has(j):
#             return sp.simplify(rhs_s)
#
#         if sp.simplify(rhs_s - j) == 0 and not lhs_s.has(j):
#             return sp.simplify(lhs_s)
#
#         return None
#     except Exception:
#         return None

def analyze_equation_state(lhs: sp.Expr, rhs: sp.Expr):
    """
    Determine the state of the equation.

    Returns:
        ("solved", value)         -> j = value
        ("identity", None)        -> infinitely many solutions
        ("contradiction", None)   -> no solution
        ("unsolved", None)        -> keep solving
    """
    try:
        diff = sp.simplify(lhs - rhs)

        # -----------------------------
        # Identity: everything cancels
        # -----------------------------
        if diff == 0:
            return ("identity", None)

        # -----------------------------
        # No j left → contradiction
        # -----------------------------
        if not diff.has(j):
            return ("contradiction", None)

        # -----------------------------
        # Try solving for j
        # -----------------------------
        sol = sp.solve(sp.Eq(lhs, rhs), j)

        if len(sol) == 1:
            return ("solved", sp.simplify(sol[0]))

        if len(sol) == 0:
            return ("contradiction", None)

        # fallback (should not happen here)
        return ("unsolved", None)

    except Exception:
        return ("unsolved", None)

def parse_op(op_text: str):
    """
    Accepts:
      +expr, -expr, *k, /k

    For + / - :
      the whole string is treated as the additive change.
      So both of these mean the same thing:
         +12-8j
         -8j+12

    For * / :
      only nonzero numeric constants are allowed.
    """
    if op_text is None:
        return None

    s = op_text.strip().replace(" ", "")
    if not s:
        return None

    first = s[0]
    if first not in "+-*/":
        return None

    # --------------------------
    # Add/subtract: parse WHOLE expression
    # --------------------------
    if first in "+-":
        expr = parse_user_expr(s)
        if expr is None:
            return None
        return ("add", sp.simplify(expr))

    # --------------------------
    # Multiply/divide: numeric only
    # --------------------------
    rhs = s[1:]
    if rhs == "":
        return None

    expr = parse_user_expr(rhs)
    if expr is None:
        return None
    expr = sp.simplify(expr)

    if not expr.is_number or sp.simplify(expr) == 0:
        return None

    if first == "*":
        return ("mul", expr)

    if first == "/":
        return ("mul", sp.simplify(1 / expr))

    return None

def split_linear_terms(expr: sp.Expr):
    """
    Split an expression into:
      j-part, constant-part

    Examples:
      -8*j - 7   -> (-8*j, -7)
      12 - 8*j   -> (-8*j, 12)
      5*j        -> (5*j, 0)
      -7         -> (0, -7)
    """
    expr = sp.expand(sp.simplify(expr))
    j_part = sp.expand(expr).coeff(j) * j
    const_part = sp.simplify(expr - j_part)
    return sp.simplify(j_part), sp.simplify(const_part)


def format_signed_term(term: sp.Expr) -> str:
    """
    Return latex for a term without unnecessary outer parentheses.
    """
    term = sp.simplify(term)
    return sp.latex(term)


def format_additive_step(lhs_expr: sp.Expr, rhs_expr: sp.Expr, add_expr: sp.Expr) -> str:
    """
    Display additive steps in a cleaner grouped form.

    If the added expression contains both a j-term and a constant term,
    show them grouped separately, e.g.
      (2j-8j) + (7-7) = (8j-8j) + (49-7)

    Otherwise fall back to the simpler whole-expression display.
    """
    add_expr = sp.expand(sp.simplify(add_expr))
    add_j, add_c = split_linear_terms(add_expr)

    lhs_expr = sp.expand(sp.simplify(lhs_expr))
    rhs_expr = sp.expand(sp.simplify(rhs_expr))

    lhs_j, lhs_c = split_linear_terms(lhs_expr)
    rhs_j, rhs_c = split_linear_terms(rhs_expr)

    has_j = sp.simplify(add_j) != 0
    has_c = sp.simplify(add_c) != 0

    # If both parts are present, show grouped like-terms together
    if has_j and has_c:
        lhs_txt = f"({format_signed_term(lhs_j)}{format_signed_term(add_j) if str(format_signed_term(add_j)).startswith('-') else '+' + format_signed_term(add_j)}) + ({format_signed_term(lhs_c)}{format_signed_term(add_c) if str(format_signed_term(add_c)).startswith('-') else '+' + format_signed_term(add_c)})"
        rhs_txt = f"({format_signed_term(rhs_j)}{format_signed_term(add_j) if str(format_signed_term(add_j)).startswith('-') else '+' + format_signed_term(add_j)}) + ({format_signed_term(rhs_c)}{format_signed_term(add_c) if str(format_signed_term(add_c)).startswith('-') else '+' + format_signed_term(add_c)})"
        return lhs_txt + " = " + rhs_txt

    # Otherwise keep the normal whole-expression display
    return (
        f"({sp.latex(lhs_expr)}) + ({sp.latex(add_expr)})"
        + " = "
        + f"({sp.latex(rhs_expr)}) + ({sp.latex(add_expr)})"
    )

def apply_op(expr: sp.Expr, parsed_op):
    kind, k = parsed_op

    if kind == "add":
        # Keep the displayed result as a visible added step
        return sp.Add(expr, k, evaluate=False)

    if kind == "mul":
        # For display purposes, distribute multiplication across top-level terms
        # but do NOT combine like terms yet.
        if expr.is_Add:
            new_terms = []
            for term in sp.Add.make_args(expr):
                expanded_term = sp.expand_mul(sp.Mul(term, k, evaluate=False))
                new_terms.append(expanded_term)
            return sp.Add(*new_terms, evaluate=False)

        return sp.expand_mul(sp.Mul(expr, k, evaluate=False))

    raise ValueError("Unknown op kind")

def parse_equation_text(eq_text: str):
    """
    Parse a full equation like:
      7j = 63
      2(j+3)=14
      j/4 + 2 = 5

    Returns:
      (lhs_raw, rhs_raw, lhs_norm, rhs_norm)
    where:
      - raw keeps the student's structural form
      - norm is simplify(expand(...)) for checking/equivalence
    """
    if eq_text is None:
        return None, None, None, None

    s = eq_text.strip()
    if not s:
        return None, None, None, None

    s = s.replace("＝", "=").replace("−", "-").replace("–", "-")

    if "=" not in s:
        return None, None, None, None

    parts = s.split("=")
    if len(parts) != 2:
        return None, None, None, None

    lhs_txt = parts[0].strip()
    rhs_txt = parts[1].strip()

    if not lhs_txt or not rhs_txt:
        return None, None, None, None

    lhs_raw = parse_user_expr(lhs_txt)
    rhs_raw = parse_user_expr(rhs_txt)

    if lhs_raw is None or rhs_raw is None:
        return None, None, None, None

    try:
        lhs_norm = sp.simplify(sp.expand(lhs_raw))
        rhs_norm = sp.simplify(sp.expand(rhs_raw))
    except Exception:
        lhs_norm = lhs_raw
        rhs_norm = rhs_raw

    return lhs_raw, rhs_raw, lhs_norm, rhs_norm

def equation_changed(display_lhs: sp.Expr, display_rhs: sp.Expr, new_lhs_raw: sp.Expr, new_rhs_raw: sp.Expr) -> bool:
    """
    Accept a new direct-equation entry if it changes the displayed line,
    even when it is algebraically equivalent to the normalized current state.
    """
    try:
        old_lhs_key = sp.srepr(display_lhs)
        old_rhs_key = sp.srepr(display_rhs)
        new_lhs_key = sp.srepr(new_lhs_raw)
        new_rhs_key = sp.srepr(new_rhs_raw)

        return not (old_lhs_key == new_lhs_key and old_rhs_key == new_rhs_key)
    except Exception:
        return True

def normalized_equation_latex(lhs: sp.Expr, rhs: sp.Expr) -> str:
    return sp.latex(sp.simplify(sp.expand(lhs))) + " = " + sp.latex(sp.simplify(sp.expand(rhs)))

def solving_equation_generator_id(level: int) -> str:
    return f"solving_eq_l{int(level)}"

def build_bracket_expr(a: int, b: int, c: int = 0):
    """
    Build a*(j + b) + c without automatic expansion,
    so the brackets display properly.
    """
    inner = sp.Add(j, sp.Integer(b), evaluate=False)
    prod = sp.Mul(sp.Integer(a), inner, evaluate=False)

    if c == 0:
        return prod

    return sp.Add(prod, sp.Integer(c), evaluate=False)

def build_fraction_expr(num_expr, den_int):
    """
    Build num_expr / den_int without turning it into (...)*(1/den_int).
    This renders as a proper fraction in LaTeX.
    """
    return sp.Mul(
        num_expr,
        sp.Pow(sp.Integer(den_int), -1, evaluate=False),
        evaluate=False
    )
# ==============================
# 🧩 Generators (Linear equations; multiple levels)
# Each returns dict with display, start_lhs, start_rhs, solution (SymPy)
# ==============================
def gen_level_1():
    # a*j + b = c
    a = nz_int(1, 8)
    sol = random.randint(-9, 9)
    b = random.randint(-12, 12)
    c = a*sol + b
    lhs = a*j + b
    rhs = sp.Integer(c)
    return {"name": "Level 1 — aj + b = c", "lhs": lhs, "rhs": rhs, "sol": sp.Integer(sol)}

def gen_level_2():
    # a*(j + b) = c
    a = nz_int(2, 9)
    sol = random.randint(-9, 9)
    b = random.randint(-8, 8)
    c = a*(sol + b)
    lhs = a*(j + b)
    rhs = sp.Integer(c)
    return {"name": "Level 2 — a(j + b) = c", "lhs": lhs, "rhs": rhs, "sol": sp.Integer(sol)}

def gen_level_3():
    # a*j + b = d*j + e
    a = nz_int(2, 9)
    d = nz_int(1, 8, exclude=[a])
    sol = random.randint(-9, 9)
    b = random.randint(-12, 12)
    e = a*sol + b - d*sol
    lhs = a*j + b
    rhs = d*j + e
    return {"name": "Level 3 — aj + b = dj + e", "lhs": lhs, "rhs": rhs, "sol": sp.Integer(sol)}

def gen_level_4():
    # a(j + b) + c = d(j + e) + f   (keep brackets visible)
    a = nz_int(2, 9)
    d = nz_int(2, 9, exclude=[a])
    sol = random.randint(-9, 9)

    b = random.randint(-8, 8)
    e = random.randint(-8, 8)
    c = random.randint(-12, 12)

    # Choose f so sol works:
    # a(sol+b)+c = d(sol+e)+f  =>  f = a(sol+b)+c - d(sol+e)
    f = a * (sol + b) + c - d * (sol + e)

    lhs = build_bracket_expr(a, b, c)
    rhs = build_bracket_expr(d, e, f)

    return {
        "name": "Level 4 — brackets both sides",
        "lhs": lhs,
        "rhs": rhs,
        "sol": sp.Integer(sol),
    }

def gen_level_5():
    """
    Fractional linear equations:
      (a1*j + b1)/d1  ±  (a2*j + b2)/d2  =  (a3*j + b3)/d3  + c

    Keep denominators simple and distinct enough that students will usually
    clear fractions first.
    """

    denom_sets = [
        (2, 3, 6),
        (2, 4, 4),
        (3, 4, 6),
        (2, 3, 4),
        (2, 5, 10),
        (3, 5, 15),
    ]
    d1, d2, d3 = random.choice(denom_sets)

    sol = random.randint(-6, 6)

    # coefficients of j in numerators
    a1 = nz_int(-5, 5, exclude=[0])
    a2 = nz_int(-5, 5, exclude=[0])
    a3 = nz_int(-5, 5, exclude=[0])

    # constants in numerators
    b1 = random.randint(-9, 9)
    b2 = random.randint(-9, 9)
    b3 = random.randint(-9, 9)

    # choose whether middle term is + or -
    mid_sign = random.choice([1, -1])

    left_1_num = a1 * j + b1
    left_2_num = a2 * j + b2
    right_num = a3 * j + b3

    left_expr = sp.Add(
        build_fraction_expr(left_1_num, d1),
        build_fraction_expr(mid_sign * left_2_num, d2),
        evaluate=False
    )

    # Pick c so that j = sol is the solution
    left_at_sol = sp.simplify(left_expr.subs(j, sol))
    right_frac_at_sol = sp.simplify(build_fraction_expr(right_num, d3).subs(j, sol))
    c = sp.simplify(left_at_sol - right_frac_at_sol)

    rhs = sp.Add(
        build_fraction_expr(right_num, d3),
        c,
        evaluate=False
    )

    sign_txt = "+" if mid_sign == 1 else "−"

    return {
        "name": f"Level 5 — fractional equations ({sign_txt} fractions)",
        "lhs": left_expr,
        "rhs": rhs,
        "sol": sp.Integer(sol),
    }

GENERATORS = {
    1: ("Level 1 — aj + b = c", gen_level_1),
    2: ("Level 2 — a(j + b) = c", gen_level_2),
    3: ("Level 3 — aj + b = dj + e", gen_level_3),
    4: ("Level 4 — a(j+b)+c = d(j+e)+f", gen_level_4),
    5: ("Level 5 — fractional linear equations", gen_level_5),
}

# ==============================
# 💡 Hint System
# ==============================
def build_hints_level_1(q):
    lhs = sp.simplify(sp.expand(q["current_lhs"]))
    rhs = sp.simplify(sp.expand(q["current_rhs"]))

    hints = [
        "Your goal is to isolate j.",
        "First remove any constant term from the side containing j.",
        "Then divide or multiply so the coefficient of j becomes 1.",
    ]

    if lhs.has(j) and not rhs.has(j):
        terms = additive_term_count(lhs)
        coeff = sp.expand(lhs).coeff(j)
        const = sp.simplify(lhs - coeff * j)

        if terms > 1 and const != 0:
            hints.insert(1, f"On the left side, remove the constant term {sp.sstr(const)} first.")
        elif coeff not in (0, 1):
            hints.insert(1, f"After isolating the j-term, divide both sides by {sp.sstr(coeff)}.")

    elif rhs.has(j) and not lhs.has(j):
        terms = additive_term_count(rhs)
        coeff = sp.expand(rhs).coeff(j)
        const = sp.simplify(rhs - coeff * j)

        if terms > 1 and const != 0:
            hints.insert(1, f"On the right side, remove the constant term {sp.sstr(const)} first.")
        elif coeff not in (0, 1):
            hints.insert(1, f"After isolating the j-term, divide both sides by {sp.sstr(coeff)}.")

    return hints


def build_hints_level_2(q):
    lhs = q.get("display_lhs", q["current_lhs"])
    rhs = q.get("display_rhs", q["current_rhs"])

    hints = [
        "Your goal is to isolate j.",
        "Start by removing the bracket using the distributive property, or undo the outer multiplication if possible.",
        "After simplifying, remove constants and then divide by the coefficient of j.",
    ]

    brackets = bracket_count(lhs) + bracket_count(rhs)
    if brackets > 0:
        hints.insert(1, "There are still brackets to expand or undo.")

    return hints


def build_hints_level_3(q):
    lhs = sp.simplify(sp.expand(q["current_lhs"]))
    rhs = sp.simplify(sp.expand(q["current_rhs"]))

    hints = [
        "Get all j-terms onto one side and all constants onto the other side.",
        "Use addition or subtraction first to move one variable term across the equals sign.",
        "Then combine like terms.",
        "Finally divide by the coefficient of j.",
    ]

    if lhs.has(j) and rhs.has(j):
        hints.insert(1, "Right now j appears on both sides, so your next step should move one j-term across.")

    return hints


def build_hints_level_4(q):
    lhs_d = q.get("display_lhs", q["current_lhs"])
    rhs_d = q.get("display_rhs", q["current_rhs"])
    lhs = sp.simplify(sp.expand(q["current_lhs"]))
    rhs = sp.simplify(sp.expand(q["current_rhs"]))

    hints = [
        "Start by simplifying both sides.",
        "Expanding brackets is usually the best first move here.",
        "After expanding, collect j-terms on one side and constants on the other.",
        "Then divide by the final coefficient of j.",
    ]

    brackets = bracket_count(lhs_d) + bracket_count(rhs_d)
    if brackets > 0:
        hints.insert(1, "There are still brackets showing, so expanding is likely a helpful next step.")

    if lhs.has(j) and rhs.has(j) and brackets == 0:
        hints.insert(1, "After simplifying, move one j-term across the equals sign.")

    return hints

def build_hints_level_5(q):
    lhs_d = q.get("display_lhs", q["current_lhs"])
    rhs_d = q.get("display_rhs", q["current_rhs"])
    lhs = sp.simplify(sp.expand(q["current_lhs"]))
    rhs = sp.simplify(sp.expand(q["current_rhs"]))

    hints = [
        "Start by clearing the fractions.",
        "Multiply every term on both sides by the least common denominator.",
        "After the fractions are gone, collect all j-terms on one side and constants on the other.",
        "Then divide by the remaining coefficient of j.",
    ]

    # crude denominator check on displayed forms
    frac_like = "/" in sp.sstr(lhs_d) or "/" in sp.sstr(rhs_d)
    if frac_like:
        hints.insert(1, "A good first step is to multiply both sides by the least common denominator.")

    if lhs.has(j) and rhs.has(j) and not frac_like:
        hints.insert(1, "Now that the fractions are handled, move one j-term across the equals sign.")

    return hints

HINT_BUILDERS = {
    1: build_hints_level_1,
    2: build_hints_level_2,
    3: build_hints_level_3,
    4: build_hints_level_4,
    5: build_hints_level_5,
}

def build_hints_for_question(q):
    builder = HINT_BUILDERS.get(q["level"])
    if not builder:
        return []
    return builder(q)


def get_equation_reactive_hint(q, old_lhs, old_rhs, new_lhs, new_rhs, old_display_lhs=None, old_display_rhs=None, new_lhs_raw=None, new_rhs_raw=None):
    """
    Return a short adaptive message similar to factoring_practice.
    """
    try:
        old_prog = measure_progress(old_lhs, old_rhs)
        new_prog = measure_progress(new_lhs, new_rhs)

        old_display_lhs = old_display_lhs if old_display_lhs is not None else old_lhs
        old_display_rhs = old_display_rhs if old_display_rhs is not None else old_rhs
        new_lhs_raw = new_lhs_raw if new_lhs_raw is not None else new_lhs
        new_rhs_raw = new_rhs_raw if new_rhs_raw is not None else new_rhs

        old_brackets = bracket_count(old_display_lhs) + bracket_count(old_display_rhs)
        new_brackets = bracket_count(new_lhs_raw) + bracket_count(new_rhs_raw)

        if new_brackets < old_brackets:
            return "Good step. Expanding brackets helps simplify the equation."

        old_frac_like = "/" in sp.sstr(old_display_lhs) or "/" in sp.sstr(old_display_rhs)
        new_frac_like = "/" in sp.sstr(new_lhs_raw) or "/" in sp.sstr(new_rhs_raw)

        if old_frac_like and not new_frac_like:
            return "Good step. You cleared the fractions."

        if new_prog["isolated"]:
            return "Nice — j is isolated now."

        if new_prog["j_presence"] < old_prog["j_presence"]:
            return "Good step. You moved j off one side of the equation."

        if new_prog.get("j_side_terms", 999) < old_prog.get("j_side_terms", 999):
            return "Good step. The side containing j is becoming simpler."

        if new_prog["deg_sum"] < old_prog["deg_sum"]:
            return "Good step. The equation is simpler now."

        if old_prog["j_presence"] == 2 and new_prog["j_presence"] == 2:
            return "Equivalent step, but try moving all j-terms to one side."

        if new_prog["j_presence"] == 1 and not new_prog["isolated"]:
            return "Equivalent step, but try removing the constant term next."

        return "Equivalent step, but try isolating j."
    except Exception:
        return "Equivalent step, but try isolating j."

# ==============================
# 🧠 Session Engine
# ==============================
def start_equations_session(num_questions, levels):
    qs = []
    for _ in range(num_questions):
        lvl = random.choice(levels)
        _, gen = GENERATORS[lvl]
        g = gen()

        lesson_preview_url = get_published_pdf_preview_url_by_generator_id(
            solving_equation_generator_id(lvl)
        )

        qs.append({
            "level": lvl,
            "level_name": g["name"],
            "lesson_preview_url": lesson_preview_url,
            "start_lhs": g["lhs"],
            "start_rhs": g["rhs"],
            "target_sol": g["sol"],

            "current_lhs": sp.simplify(sp.expand(g["lhs"])),
            "current_rhs": sp.simplify(sp.expand(g["rhs"])),

            "display_lhs": g["lhs"],
            "display_rhs": g["rhs"],

            "attempts": 0,
            "hints_used": 0,
            "available_hints": None,
            "hints_shown": [],
            "hint_index": 0,
            "hint_view_index": -1,

            "steps": [],
            "last_message": "",
            "correct": False,
            "first_try_correct": False,

            "flash_correct": False,
            "flash_final_latex": "",
        })

    ss.equations = {
        "start_time": time.time(),
        "questions": qs,
        "current": 0,
        "finished": False,
    }

import streamlit.components.v1 as components

@st.dialog("📘 Reference Lesson", width="large")
def open_reference_pdf_dialog(title: str, preview_url: str):
    st.subheader(title)
    components.iframe(preview_url, height=700, scrolling=True)

# ==============================
# 🖥️ UI
# ==============================
def solving_equations_practice():

    st.markdown("""
    <style>
    .eq-card{
      background: transparent;
      border: none;
      border-radius: 0;
      padding: 0;
      margin: 0.2rem 0 0.4rem 0;
      backdrop-filter: none;
    }
    .eq-title{
      text-align:center;
      font-size:16px;
      font-weight:500;
      margin: 0.02rem 0 0.02rem 0;
      opacity: 0.92;
    }
    .eq-latex-center .katex-display{
      margin: 0.02em 0 !important;
      text-align: center !important;
    }
    .eq-latex-big .katex-display{
      font-size: 1.08em !important;
    }
    .eq-compact .katex-display{
      margin: 0.03em 0 !important;
      text-align: center !important;
    }
    .op-input-tight input,
    .stTextInput input{
      text-align: center !important;
      padding: 0.30rem 0.50rem !important;
    }
    .eq-thin-sep{
      height: 1px;
      background: rgba(0,0,0,0.10);
      border-radius: 0;
      margin: 0.15rem 0 0.35rem 0;
    }
    .block-container{
      padding-top: 1.0rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:1.58rem; line-height:1.15; font-weight:700; margin:0.6rem 0 0.15rem 0; padding-top:0.1rem;'>🧮 Solving Equations Practice</div>",
        unsafe_allow_html=True
    )
    # ----------------------------
    # Session init
    # ----------------------------
    if "equations" not in ss:
        ss.equations = None

    if "eq_setup_open" not in ss:
        ss.eq_setup_open = True

    if "eq_input_version" not in ss:
        ss.eq_input_version = 0

    if "eq_entry_mode" not in ss:
        ss.eq_entry_mode = "Operation"

    # =============================
    # 🟢 SETUP SCREEN
    # =============================
    if ss.equations is None:

        st.markdown("### ⚙️ Practice Setup")

        level_map = {k: GENERATORS[k][0] for k in GENERATORS.keys()}

        if "eq_selected_levels" not in ss:
            ss.eq_selected_levels = []

        with st.expander("🧩 Select Question Types", expanded=ss.eq_setup_open):
            st.pills(
                "Equation levels",
                options=list(level_map.keys()),
                format_func=lambda x: level_map[x],
                selection_mode="multi",
                key="eq_selected_levels",
            )

        num_q = st.slider("Number of questions", 1, 30, 10)

        if not ss.eq_selected_levels:
            st.warning("Select at least one level to continue.")
        else:
            if st.button("🚀 Start Practice", use_container_width=True):
                ss.eq_setup_open = False
                start_equations_session(num_q, ss.eq_selected_levels)
                ss.eq_input_version += 1
                st.rerun()
        return

    # =============================
    # 📊 RESULTS SCREEN
    # =============================
    if ss.equations.get("finished"):

        elapsed = time.time() - ss.equations["start_time"]
        qs = ss.equations["questions"]

        total = len(qs)
        correct = sum(1 for q in qs if q.get("correct"))
        first_try = sum(1 for q in qs if q.get("first_try_correct"))

        st.success("✅ Practice Complete!")
        st.markdown(
            f"**Time:** {elapsed:.1f} seconds  \n"
            f"**Score:** {correct} / {total}  \n"
            f"**First-try correct:** {first_try} / {total}"
        )

        if st.button("🔁 New Practice Set"):
            ss.equations = None
            ss.eq_setup_open = True
            ss.eq_input_version += 1
            st.rerun()
        return

    # =============================
    # ❓ QUESTION SCREEN
    # =============================
    idx = ss.equations["current"]
    qs = ss.equations["questions"]
    q = qs[idx]

    if q.get("available_hints") is None:
        q["available_hints"] = build_hints_for_question(q)

    q_col, r_col = st.columns([8.5, 1.8], vertical_alignment="top")

    with q_col:
        st.markdown(
            f"<div style='font-size:1.15rem; font-weight:700; margin:0 0 -0.35rem 0;'>Question {idx + 1} of {len(qs)}</div>",
            unsafe_allow_html=True
        )
        st.caption(q["level_name"])

        lesson_preview_url = q.get("lesson_preview_url")

        if lesson_preview_url:
            lesson_cols = st.columns([1.2, 5])
            with lesson_cols[0]:
                if st.button("📘 View Lesson", key=f"eq_lesson_{idx}", width="stretch"):
                    open_reference_pdf_dialog(
                        f"Reference Lesson — {q['level_name']}",
                        lesson_preview_url
                    )
            with lesson_cols[1]:
                st.caption("Open the lesson for this question type.")

    with r_col:
        st.markdown("<div style='margin-top:-0.15rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Restart", key=f"restart_top_{idx}", width="stretch"):
            ss.equations = None
            ss.eq_setup_open = True
            ss.eq_input_version += 1
            st.rerun()

    # ----------------------------
    # Solved screen (no flashing)
    # ----------------------------
    if q.get("correct"):
        st.success("✅ Solved!")

        st.markdown("**Original:**")
        st.latex(sp.latex(q["start_lhs"]) + " = " + sp.latex(q["start_rhs"]))

        st.latex(q.get("solved_line_latex", ""))

        next_col = st.columns([3, 2, 3])[1]
        with next_col:
            if st.button("➡️ Next", key=f"eq_next_{idx}", width="stretch"):
                # move on
                if idx + 1 >= len(qs):
                    ss.equations["finished"] = True
                else:
                    ss.equations["current"] += 1

                ss.eq_input_version += 1
                st.rerun()

        return  # IMPORTANT: stop rendering the rest of the UI

    # ----------------------------
    # Original + Working + Help Layout
    # ----------------------------
    st.markdown("<div style='margin-top:-0.8rem;'></div>", unsafe_allow_html=True)

    main_left, main_right = st.columns([1.65, 1.0], vertical_alignment="top")

    # ==================================
    # LEFT COLUMN
    # ==================================
    with main_left:
        # ---------- Original ----------
        st.markdown(
            "<div style='text-align:center; font-size:1.18rem; font-weight:700; margin:-0.1rem 0 -0.8rem 0;'>Original</div>",
            unsafe_allow_html=True
        )
        st.latex(f"{sp.latex(q['start_lhs'])} = {sp.latex(q['start_rhs'])}")

        # ---------- Working ----------
        st.markdown(
            "<div style='text-align:center; font-size:1.28rem; font-weight:700; margin:0.55rem 0 0.1rem 0;'>✏️ Working</div>",
            unsafe_allow_html=True
        )

        if q.get("last_message"):
            st.info(q["last_message"])

        if q["steps"]:
            for step in q["steps"]:
                if step.get("op_display"):
                    st.latex(step["op_display"])
                st.latex(step["result_display"])

        # ---------- Entry row ----------
        st.markdown(
            "<div class='eq-title' style='margin:0.35rem 0 -0.02rem 0;'>Enter your next step</div>",
            unsafe_allow_html=True
        )
        st.markdown("<div class='eq-card'>", unsafe_allow_html=True)

        left_key = f"eq_opL_{ss.eq_input_version}"
        mid_key = f"eq_full_{ss.eq_input_version}"
        right_key = f"eq_opR_{ss.eq_input_version}"

        left_val = ss.get(left_key, "")
        mid_val = ss.get(mid_key, "")
        right_val = ss.get(right_key, "")

        side_has_text = bool((left_val or "").strip() or (right_val or "").strip())
        middle_has_text = bool((mid_val or "").strip())

        cap1, cap2, cap3 = st.columns([2.4, 4.2, 2.4], vertical_alignment="bottom")
        with cap1:
            st.caption("LS")
        with cap2:
            st.caption("Next full equation")
        with cap3:
            st.caption("RS")

        colL, colM, colR = st.columns([2.4, 4.2, 2.4], vertical_alignment="center")

        with colL:
            op_left = st.text_input(
                "Left operation",
                key=left_key,
                label_visibility="collapsed",
                autocomplete="off",
                disabled=middle_has_text,
            )

        with colM:
            eq_text = st.text_input(
                "Next equation",
                key=mid_key,
                label_visibility="collapsed",
                autocomplete="off",
                disabled=side_has_text,
            )

        with colR:
            op_right = st.text_input(
                "Right operation",
                key=right_key,
                label_visibility="collapsed",
                autocomplete="off",
                disabled=middle_has_text,
            )

        help1, help2, help3 = st.columns([2.4, 4.2, 2.4], vertical_alignment="top")
        with help1:
            st.caption("LS: Operation. eg. +5 or /3")
        with help2:
            st.caption("Enter the whole next equivalent equation")
        with help3:
            st.caption("RS: Operation. eg. +5 or /3")

        submit_col = st.columns([3, 2, 3])[1]
        with submit_col:
            submitted = st.button(
                "✅ Submit",
                key=f"eq_submit_{idx}_{ss.eq_input_version}",
                width="stretch"
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ==================================
    # RIGHT COLUMN  (Factoring-style help)
    # ==================================
    with main_right:
        st.markdown(
            f"<div style='text-align:left; font-size:1.00rem; font-weight:500; margin:0.15rem 0 0.4rem 0;'>💡 Hints used: {q.get('hints_used', 0)}</div>",
            unsafe_allow_html=True
        )

        hints = q.get("available_hints") or build_hints_for_question(q)
        shown = q.get("hints_shown", [])
        view_i = q.get("hint_view_index", -1)

        nav1, nav2, nav3 = st.columns([1.1, 1.7, 1.1], vertical_alignment="center")

        with nav1:
            prev_disabled = len(shown) == 0 or view_i <= 0
            if st.button("◀", key=f"eq_hint_prev_{idx}", disabled=prev_disabled, width="stretch"):
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
            next_disabled = len(hints) == 0 or (len(shown) >= len(hints) and view_i >= len(shown) - 1)
            if st.button("▶", key=f"eq_hint_next_{idx}", disabled=next_disabled, width="stretch"):
                if len(shown) < len(hints):
                    q["hints_shown"].append("💡 " + hints[len(shown)])
                    q["hint_index"] = len(q["hints_shown"])
                    q["hint_view_index"] = len(q["hints_shown"]) - 1
                    q["hints_used"] = len(q["hints_shown"])
                else:
                    q["hint_view_index"] = min(len(shown) - 1, view_i + 1)
                st.rerun()

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
                    margin-top: 0.15rem;
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
                    margin-top: 0.15rem;
                ">
                    <div style="font-size: 1.05rem;">{shown[view_i]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =============================
    # SUBMIT HANDLER
    # =============================
    if submitted:

        left_txt = (op_left or "").strip()
        mid_txt = (eq_text or "").strip()
        right_txt = (op_right or "").strip()

        using_ops = bool(left_txt or right_txt)
        using_equation = bool(mid_txt)

        if using_ops and using_equation:
            q["attempts"] += 1
            q["last_message"] = "❌ Use either the side operations or the full equation entry, not both."
            st.rerun()

        if not using_ops and not using_equation:
            q["attempts"] += 1
            q["last_message"] = "❌ Enter a step before submitting."
            st.rerun()

        old_lhs = q["current_lhs"]
        old_rhs = q["current_rhs"]
        old_display_lhs = q.get("display_lhs", old_lhs)
        old_display_rhs = q.get("display_rhs", old_rhs)

        # ---------------------------------
        # Path A: operation on both sides
        # ---------------------------------
        if using_ops:
            pL = parse_op(left_txt)
            pR = parse_op(right_txt)

            if pL is None or pR is None:
                q["attempts"] += 1
                q["last_message"] = "❌ Use +expr or -expr, or *k /k where k is a nonzero number."
                st.rerun()

            if not same_operation(pL, pR):
                q["attempts"] += 1
                q["last_message"] = "❌ Operations must match on both sides."
                st.rerun()

            new_lhs_raw = apply_op(old_display_lhs, pL)
            new_rhs_raw = apply_op(old_display_rhs, pL)

            new_lhs = sp.simplify(sp.expand(new_lhs_raw))
            new_rhs = sp.simplify(sp.expand(new_rhs_raw))

            if sp.simplify(new_lhs - old_lhs) == 0 and sp.simplify(new_rhs - old_rhs) == 0:
                q["attempts"] += 1
                q["last_message"] = "❌ That didn’t change the equation."
                st.rerun()

            kind, k = pL
            k = sp.simplify(k)

            if kind == "add":
                op_display = format_additive_step(old_display_lhs, old_display_rhs, k)
            else:
                op_display = (
                    f"({sp.latex(old_display_lhs)}) \\cdot ({sp.latex(k)})"
                    + " = "
                    + f"({sp.latex(old_display_rhs)}) \\cdot ({sp.latex(k)})"
                )

            result_display = sp.latex(new_lhs_raw) + " = " + sp.latex(new_rhs_raw)

            q["steps"].append({
                "op_display": op_display,
                "result_display": result_display,
                "op_text": left_txt,
            })

            q["display_lhs"] = new_lhs_raw
            q["display_rhs"] = new_rhs_raw
            q["current_lhs"] = new_lhs
            q["current_rhs"] = new_rhs
            q["available_hints"] = build_hints_for_question(q)

        # ---------------------------------
        # Path B: full next equation
        # ---------------------------------
        else:
            new_lhs_raw, new_rhs_raw, new_lhs, new_rhs = parse_equation_text(mid_txt)

            if new_lhs_raw is None or new_rhs_raw is None:
                q["attempts"] += 1
                q["last_message"] = "❌ Enter a full equation such as 3j = -9."
                st.rerun()

            if not equation_changed(old_display_lhs, old_display_rhs, new_lhs_raw, new_rhs_raw):
                q["attempts"] += 1
                q["last_message"] = "❌ That didn’t change the equation."
                st.rerun()

            if not equivalent_equations(old_lhs, old_rhs, new_lhs, new_rhs):
                q["attempts"] += 1
                q["last_message"] = "❌ That new equation is not equivalent to the current one."
                st.rerun()

            q["steps"].append({
                "op_display": "",
                "result_display": sp.latex(new_lhs_raw) + " = " + sp.latex(new_rhs_raw),
                "op_text": mid_txt,
            })

            q["display_lhs"] = new_lhs_raw
            q["display_rhs"] = new_rhs_raw
            q["current_lhs"] = new_lhs
            q["current_rhs"] = new_rhs
            q["available_hints"] = build_hints_for_question(q)

        state, value = analyze_equation_state(new_lhs, new_rhs)

        if state == "identity":
            q["correct"] = True
            q["solved_line_latex"] = r"j \in \mathbb{R}"
            q["steps"] = []
            q["last_message"] = "🎯 This equation is true for all values of j."
            ss.eq_input_version += 1
            st.rerun()

        elif state == "contradiction":
            q["correct"] = True
            q["solved_line_latex"] = r"\text{No solution}"
            q["steps"] = []
            q["last_message"] = "🚫 This equation has no solution."
            ss.eq_input_version += 1
            st.rerun()

        elif state == "solved":
            if sp.simplify(value - q["target_sol"]) == 0:
                q["correct"] = True
                q["solved_line_latex"] = r"j = " + sp.latex(value)
                q["steps"] = []
                q["last_message"] = ""
                if q["attempts"] == 0:
                    q["first_try_correct"] = True
                ss.eq_input_version += 1
                st.rerun()

        old_prog = measure_progress(old_lhs, old_rhs)
        new_prog = measure_progress(new_lhs, new_rhs)

        old_brackets = bracket_count(old_display_lhs) + bracket_count(old_display_rhs)
        new_brackets = bracket_count(new_lhs_raw) + bracket_count(new_rhs_raw)

        helpful = (
            new_prog["isolated"]
            or new_prog["j_presence"] < old_prog["j_presence"]
            or new_prog["deg_sum"] < old_prog["deg_sum"]
            or new_prog.get("j_side_terms", 999) < old_prog.get("j_side_terms", 999)
            or new_brackets < old_brackets
        )

        reactive = get_equation_reactive_hint(
            q,
            old_lhs, old_rhs,
            new_lhs, new_rhs,
            old_display_lhs=old_display_lhs,
            old_display_rhs=old_display_rhs,
            new_lhs_raw=new_lhs_raw,
            new_rhs_raw=new_rhs_raw,
        )

        q["last_message"] = ("✅ " + reactive) if helpful else ("🧠 " + reactive)
        q["attempts"] += 1
        ss.eq_input_version += 1
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
