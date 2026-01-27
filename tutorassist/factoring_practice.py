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

GENERATORS = {
    1: ("Common Factor (j,k)", gen_common_factor, lambda q: "Find the greatest common factor of all terms."),
    2: ("Trinomial (a = 1)", gen_trinomial_a1, lambda q: "Look for two numbers that multiply to C and add to B."),
    3: ("Trinomial (a ≠ 1)", gen_trinomial_aN, lambda q: "Try factoring by decomposition or grouping."),
    4: ("Difference of Squares", gen_diff_squares, lambda q: "Does this match a² − b² ?"),
    5: ("Sum of Squares (real numbers)", gen_sum_squares, lambda q: "Sum of squares does not factor over the reals."),
    6: ("Complete the Square (Vertex Form)", gen_vertex_form, lambda q: "Rewrite in the form a(j - h)² + k."),
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

def build_hints_trinomial_a1(q):
    expr = q["target_expr"]
    poly = sp.Poly(expr, j)

    b = int(poly.coeffs()[1])
    c = int(poly.coeffs()[2])

    hints = [
        f"Find two numbers that multiply to {c}.",
        f"Those two numbers must add to {b}.",
        "Decide if the two numbers should be both positive, both negative, or one of each.",
        f"List factor pairs of {abs(c)} and test their sums.",
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

HINT_BUILDERS = {
    1: build_hints_common_factor,
    2: build_hints_trinomial_a1,
    3: build_hints_trinomial_aN,
    4: build_hints_diff_squares,
    5: build_hints_sum_squares,
    6: build_hints_vertex_form,
}

def build_hints_for_question(q):
    builder = HINT_BUILDERS.get(q["level"])
    if not builder:
        return []
    return builder(q)


# ==============================
# 🧠 Session Engine
# ==============================

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
            "available_hints": [],
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
    }

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

        # Map for labels
        level_map = {
            1: "Level 1 — Common Factor (j,k)",
            2: "Level 2 — Trinomial (a = 1)",
            3: "Level 3 — Trinomial (a ≠ 1)",
            4: "Level 4 — Difference of Squares",
            5: "Level 5 — Sum of Squares (real numbers)",
            6: "Level 6 — Complete the Square (Vertex Form)",
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

        # Slider under expander
        num_q = st.slider("Number of questions", 1, 30, 10)

        st.caption("Tip: You can type answers like 8(6j-5k) or (j-2)(j+3).")

        if not ss.selected_levels:
            st.warning("Select at least one question type to continue.")
        else:
            if st.button("🚀 Start Practice", use_container_width=True):
                ss.setup_open = False  # 👈 CLOSE expander from now on
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
                    f"{icon} Final answer: `{q.get('user_answer','')}`  \n"
                    f"Wrong attempts: {q.get('attempts',0)}  \n"
                    f"Hints used: {q.get('hints_used',0)}"
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
    q.setdefault("attempts", 0)        # we'll treat this as WRONG attempts
    q.setdefault("hints_used", 0)
    q.setdefault("steps", [])
    q.setdefault("last_message", "")
    q.setdefault("correct", False)
    q.setdefault("first_try_correct", False)
    q.setdefault("user_answer", "")
    q.setdefault("current_expr", q["target_expr"])
    q["available_hints"] = build_hints_for_question(q)

    left, right = st.columns([3, 1.3])

    with left:
        st.markdown(f"### Question {idx + 1} of {len(questions)}")
        if q["level"] == 6:
            st.latex(r"\LARGE \textbf{Write in vertex form:}\quad " + sp.latex(q["target_expr"]))
        else:
            st.latex(r"\LARGE \textbf{Factor:}\quad " + sp.latex(q["target_expr"]))

        st.markdown(f"💡 Hints used: **{q['hints_used']}**")

        st.markdown("### ✏️ Working:")

        # message bar
        if q.get("last_message"):
            # keep your existing style (info box)
            st.info(q["last_message"])

        # Show original line (LEFT-aligned)
        st.markdown(f"${sp.latex(q['target_expr'])}$")

        # Show steps EXACTLY as typed, but with superscripts
        for step in q["steps"]:
            txt = step["text"] if isinstance(step, dict) else str(step)

            latex_txt = to_latex_like(txt)

            # Render as math, but using THEIR structure
            st.markdown(f"= ${latex_txt}$")

        # ------------------------------
        # Input
        # ------------------------------
        user_answer = st.text_input(
            "Your answer:",
            key=f"fact_answer_box_{ss.fact_input_version}",
            autocomplete="off",
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("✅ Submit"):
                # Special case: sum of squares / irreducible
                if q["level"] == 5:
                    u = (user_answer or "").strip().lower().replace(" ", "")
                    ok = u in {
                        "irreducible",
                        "prime",
                        "cannotbefactored",
                        "cannotfactor",
                        "no real factors",
                    }
                    if ok:
                        q["correct"] = True
                        q["user_answer"] = user_answer
                        # first try == no wrong attempts yet
                        if q["attempts"] == 0:
                            q["first_try_correct"] = True
                        q["last_message"] = "🎉 Correct — this does not factor over the reals."

                        # advance
                        if idx + 1 >= len(questions):
                            ss.factoring["finished"] = True
                        else:
                            ss.factoring["current"] += 1
                            questions[ss.factoring["current"]].setdefault("last_message", "")
                            questions[ss.factoring["current"]]["last_message"] = ""

                        ss.fact_input_version += 1
                        st.rerun()
                    else:
                        q["attempts"] += 1
                        q["last_message"] = "❌ For this one, enter: irreducible / prime / cannot be factored."
                        st.rerun()
                    return

                # Normal: parse
                expr_u = parse_user_expr(user_answer)

                if expr_u is None:
                    q["attempts"] += 1
                    q["last_message"] = "❌ I couldn't parse that. Try (j-2)(j+3) or 8(6j-5k)."
                    st.rerun()
                    return

                # 1) Must be equivalent to ORIGINAL target
                if not equivalent(expr_u, q["target_expr"]):
                    q["attempts"] += 1
                    q["last_message"] = "❌ This is not equivalent to the original expression."
                    st.rerun()
                    return

                # 2) If matches one of the precomputed final answers -> DONE
                if canon_key(expr_u) in q["final_answers"]:
                    # 🔒 Extra structural check for Level 6 (vertex form)
                    if q["level"] == 6 and q.get("vertex_final_expr") is not None:
                        student_terms = top_level_term_count(expr_u)
                        final_terms = top_level_term_count(q["vertex_final_expr"])

                        if student_terms != final_terms:
                            # Record this as a valid step (if it actually changes something)
                            last_key = canon_key(q["steps"][-1]["expr"]) if q["steps"] else canon_key(q["target_expr"])
                            if canon_key(expr_u) != last_key:
                                q["steps"].append({
                                    "expr": expr_u,
                                    "text": user_answer.strip()
                                })
                                q["current_expr"] = expr_u

                            q["last_message"] = "⚠️ Finish simplifying the constants."
                            ss.fact_input_version += 1
                            st.rerun()
                            return

                    last_key = canon_key(q["steps"][-1]["expr"]) if q["steps"] else canon_key(q["target_expr"])
                    if canon_key(expr_u) != last_key:
                        q["steps"].append({
                            "expr": expr_u,
                            "text": user_answer.strip()
                        })

                    q["current_expr"] = expr_u
                    q["user_answer"] = user_answer
                    q["correct"] = True
                    if q["attempts"] == 0:
                        q["first_try_correct"] = True

                    q["last_message"] = "🎉 Fully factored!"

                    if idx + 1 >= len(questions):
                        ss.factoring["finished"] = True
                    else:
                        ss.factoring["current"] += 1
                        questions[ss.factoring["current"]]["last_message"] = ""

                    ss.fact_input_version += 1
                    st.rerun()
                    return

                # 3) Equivalent but not done yet:
                #    - if no change -> warn
                #    - else record step (NO penalty)
                last_key = canon_key(q["steps"][-1]["expr"]) if q["steps"] else canon_key(q["target_expr"])
                if canon_key(expr_u) == last_key:
                    q["last_message"] = "⚠️ This does not change the expression. Try factoring something."
                    st.rerun()
                    return

                q["steps"].append({
                    "expr": expr_u,
                    "text": user_answer.strip()
                    })
                q["current_expr"] = expr_u
                q["last_message"] = "✅ Good step — keep factoring."
                ss.fact_input_version += 1
                st.rerun()
                return

        with col2:
            if st.button("⏭️ Skip"):
                if idx + 1 >= len(questions):
                    ss.factoring["finished"] = True
                else:
                    ss.factoring["current"] += 1
                    questions[ss.factoring["current"]].setdefault("last_message", "")
                    questions[ss.factoring["current"]]["last_message"] = ""
                ss.fact_input_version += 1
                st.rerun()

    with right:
        st.markdown("### 💡 Hints")

        if q["hints_shown"]:
            for i, h in enumerate(q["hints_shown"], 1):
                st.markdown(f"**{i}.** {h}")
        else:
            st.caption("No hints used yet.")

        st.divider()

        if st.button("➕ Show next hint"):
            i = q["hint_index"]
            hints = q["available_hints"]

            if i < len(hints):
                new_hint = hints[i]
                q["hints_shown"].append(new_hint)
                q["hint_index"] += 1
                q["hints_used"] = q["hint_index"]
            else:
                if not q["hints_shown"] or not q["hints_shown"][-1].startswith("ℹ️"):
                    q["hints_shown"].append("ℹ️ No more hints available for this question.")

            st.rerun()
