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

def detect_same_operation(old_lhs: sp.Expr, old_rhs: sp.Expr, new_lhs: sp.Expr, new_rhs: sp.Expr):
    """
    Detects whether the step is:
      - add/subtract same term T to both sides
      - multiply/divide both sides by same numeric constant C
    Returns (ok: bool, op_desc: str or None)
    """
    try:
        # ADD/SUB: new = old + T
        dL = sp.simplify(sp.expand(new_lhs - old_lhs))
        dR = sp.simplify(sp.expand(new_rhs - old_rhs))
        if sp.simplify(dL - dR) == 0:
            # same additive change both sides
            if sp.simplify(dL) == 0:
                return False, "That didn’t change the equation."
            return True, f"Added {sp.sstr(dL)} to both sides."

        # MULT/DIV: new = old * C  (numeric constant only)
        # Guard: avoid dividing by 0/expressions; allow only numeric constants
        if old_lhs == 0 or old_rhs == 0:
            # still might be valid, but keep it simple; fall back to equivalence-only
            return True, None

        rL = sp.simplify(new_lhs / old_lhs)
        rR = sp.simplify(new_rhs / old_rhs)

        if sp.simplify(rL - rR) == 0 and is_numeric_constant(rL) and sp.simplify(rL) != 0:
            return True, f"Multiplied both sides by {sp.sstr(sp.simplify(rL))}."

        return False, "Make sure you apply the same operation to BOTH sides (add/subtract the same term, or multiply/divide by the same constant)."
    except Exception:
        return False, "I couldn’t verify the operation on both sides—check your algebra and parentheses."

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

def solved_value_if_isolated(lhs: sp.Expr, rhs: sp.Expr):
    """
    If equation is j = number (or number = j), return that number as a SymPy expr.
    Otherwise return None.
    """
    try:
        lhs_s = sp.simplify(sp.expand(lhs))
        rhs_s = sp.simplify(sp.expand(rhs))

        if sp.simplify(lhs_s - j) == 0 and not rhs_s.has(j):
            return sp.simplify(rhs_s)

        if sp.simplify(rhs_s - j) == 0 and not lhs_s.has(j):
            return sp.simplify(lhs_s)

        return None
    except Exception:
        return None

def parse_op(op_text: str):
    """
    Accepts:
      +expr, -expr, *k, /k
    where expr may include j (e.g., -j, +2j, +(j-3), +j/2)
    and k must be a nonzero numeric constant for * and /.

    Returns: (kind, value) where:
      kind in {"add","mul"}
      value is a SymPy expr
        - for add: any SymPy expression (can include j)
        - for mul: numeric SymPy Number (nonzero). Division becomes mul by 1/k
    """
    if op_text is None:
        return None

    s = op_text.strip().replace(" ", "")
    if not s:
        return None

    op = s[0]
    if op not in "+-*/":
        return None

    rhs = s[1:]
    if rhs == "":
        return None

    # Parse the RHS as an expression so it can include j for +/-
    # Use your existing parse_user_expr helper
    expr = parse_user_expr(rhs)
    if expr is None:
        return None
    expr = sp.simplify(expr)

    if op == "+":
        return ("add", expr)

    if op == "-":
        return ("add", sp.simplify(-expr))

    # For multiply/divide: ONLY allow numeric constants (protects solution sets)
    if op == "*":
        if not expr.is_number or sp.simplify(expr) == 0:
            return None
        return ("mul", sp.simplify(expr))

    if op == "/":
        if not expr.is_number or sp.simplify(expr) == 0:
            return None
        return ("mul", sp.simplify(1 / expr))

    return None

def apply_op(expr: sp.Expr, parsed_op):
    kind, k = parsed_op
    if kind == "add":
        return sp.simplify(sp.expand(expr + k))
    if kind == "mul":
        return sp.simplify(sp.expand(expr * k))
    raise ValueError("Unknown op kind")

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
    # a(j + b) + c = d(j + e) + f
    a = nz_int(2, 9)
    d = nz_int(2, 9, exclude=[a])
    sol = random.randint(-9, 9)

    b = random.randint(-8, 8)
    e = random.randint(-8, 8)
    c = random.randint(-12, 12)

    # Choose f so sol works
    # a(sol+b)+c = d(sol+e)+f  => f = a(sol+b)+c - d(sol+e)
    f = a*(sol + b) + c - d*(sol + e)

    lhs = a*(j + b) + c
    rhs = d*(j + e) + sp.Integer(f)
    return {"name": "Level 4 — brackets both sides", "lhs": lhs, "rhs": rhs, "sol": sp.Integer(sol)}

GENERATORS = {
    1: ("Level 1 — aj + b = c", gen_level_1),
    2: ("Level 2 — a(j + b) = c", gen_level_2),
    3: ("Level 3 — aj + b = dj + e", gen_level_3),
    4: ("Level 4 — a(j+b)+c = d(j+e)+f", gen_level_4),
}

# ==============================
# 🧠 Session Engine
# ==============================
def start_equations_session(num_questions, levels):
    qs = []
    for _ in range(num_questions):
        lvl = random.choice(levels)
        _, gen = GENERATORS[lvl]
        g = gen()

        qs.append({
            "level": lvl,
            "level_name": g["name"],
            "start_lhs": g["lhs"],
            "start_rhs": g["rhs"],
            "target_sol": g["sol"],

            "current_lhs": g["lhs"],
            "current_rhs": g["rhs"],

            "attempts": 0,
            "hints_used": 0,     # placeholder for later if you want hints
            "steps": [],         # list of {"lhs":, "rhs":, "text_lhs":, "text_rhs":}
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

# ==============================
# 🖥️ UI
# ==============================
def solving_equations_practice():
    st.markdown("""
    <style>
    .eq-card{
      background: rgba(255,255,255,0.55);
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 14px;
      padding: 14px 14px 10px 14px;
      margin: 10px 0 12px 0;
      backdrop-filter: blur(2px);
    }
    .eq-title{
      text-align:center;
      font-size:20px;
      font-weight:500;
      margin: 4px 0 8px 0;
      opacity: 0.9;
    }
    .eq-latex-center .katex-display{
      margin: 0.3em 0 !important;
      text-align: center !important;
    }
    .eq-latex-big .katex-display{
      font-size: 1.25em !important; /* slightly larger equation */
    }
    .op-input-tight input{
        text-align: center !important;
        padding: 0.35rem 0.55rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* Pull inputs a bit toward the equation */
    .op-left { margin-right: -10px; }
    .op-right { margin-left: -10px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .block-container { padding-top: 1.0rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## 🧮 Solving Equations Practice")

    # ----------------------------
    # Session init
    # ----------------------------
    if "equations" not in ss:
        ss.equations = None

    if "eq_setup_open" not in ss:
        ss.eq_setup_open = True

    if "eq_input_version" not in ss:
        ss.eq_input_version = 0

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

    q_col, r_col = st.columns([8, 2], vertical_alignment="center")

    with q_col:
        st.markdown(f"### Question {idx + 1} of {len(qs)}")
        st.caption(q["level_name"])

    with r_col:
        if st.button("🔄 Restart", key=f"restart_top_{idx}", use_container_width=True):
            ss.equations = None
            ss.eq_setup_open = True
            ss.eq_input_version += 1
            st.rerun()

    # ----------------------------
    # Flash correct
    # ----------------------------
    if q.get("flash_correct"):
        st.success("✅ Solved!")

        st.markdown("**Original:**")
        st.latex(sp.latex(q["start_lhs"]) + " = " + sp.latex(q["start_rhs"]))

        # Final line only (e.g., j = 4)
        st.latex(q.get("solved_line_latex", q.get("flash_final_latex", "")))

        time.sleep(2.2)

        # clear history so it never shows after solve
        q["steps"] = []
        q["last_message"] = ""
        q["flash_correct"] = False

        if idx + 1 >= len(qs):
            ss.equations["finished"] = True
        else:
            ss.equations["current"] += 1

        ss.eq_input_version += 1
        st.rerun()
    # ----------------------------
    # Working Display
    # ----------------------------
    st.markdown("### ✏️ Working")

    if q.get("last_message"):
        st.info(q["last_message"])

    # Original
    st.markdown("**Original:**")
    st.latex(sp.latex(q["start_lhs"]) + " = " + sp.latex(q["start_rhs"]))

    # Only show history/current if NOT solved
    if not q.get("correct") and not q.get("flash_correct"):
        # History (operation intent lines)
        for step in q["steps"]:
            st.latex(step["op_display"])
            st.latex(step["result_display"])

        # Current (only if steps exist)
        if q["steps"]:
            st.markdown("**Current:**")
            st.latex(sp.latex(q["current_lhs"]) + " = " + sp.latex(q["current_rhs"]))


    # =============================
    # 🔄 OPERATION INPUT ROW (polished)
    # =============================

    # Centered instruction (smaller)
    st.markdown("<div class='eq-title'>Apply an operation to both sides</div>", unsafe_allow_html=True)

    # Soft equation card container
    st.markdown("<div class='eq-card'>", unsafe_allow_html=True)

    # Make the inputs hug the equation more tightly than [2,6,2]
    colpL, colL, colEq, colR, colsR = st.columns([1.2, 3.6, 3.8, 3.6, 1.2], vertical_alignment="center")

    with colL:
        st.markdown("<div class='op-input-tight op-left'>", unsafe_allow_html=True)
        op_left = st.text_input(
            "Left operation",
            key=f"eq_opL_{ss.eq_input_version}",
            label_visibility="collapsed",
            placeholder="+5  or  /3",
            autocomplete="off",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with colEq:
        st.markdown("<div class='eq-latex-center eq-latex-big'>", unsafe_allow_html=True)
        st.latex(sp.latex(q["current_lhs"]) + " = " + sp.latex(q["current_rhs"]))
        st.markdown("</div>", unsafe_allow_html=True)

    with colR:
        st.markdown("<div class='op-input-tight op-right'>", unsafe_allow_html=True)
        op_right = st.text_input(
            "Right operation",
            key=f"eq_opR_{ss.eq_input_version}",
            label_visibility="collapsed",
            placeholder="+5  or  /3",
            autocomplete="off",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # subtle divider line inside the card
    st.markdown("""
    <style>
    .eq-divider{
      height: 0;
      border: none;
      border-top: 1px solid rgba(0,0,0,0.12);
      margin: 6px 0 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='eq-divider'></div>", unsafe_allow_html=True)

    # Centered submit button (your width='stretch' style)
    submit_col = st.columns([3, 2, 3])[1]
    with submit_col:
        if st.button("✅ Submit", width="stretch"):

            pL = parse_op(op_left)
            pR = parse_op(op_right)

            if pL is None or pR is None:
                q["attempts"] += 1
                q["last_message"] = "❌ Use +expr or -expr (expr may include j), or *k /k where k is a nonzero number."
                ss.eq_input_version += 1
                st.rerun()

            if not same_operation(pL, pR):
                q["attempts"] += 1
                q["last_message"] = "❌ Operations must match on both sides."
                ss.eq_input_version += 1
                st.rerun()

            old_lhs = q["current_lhs"]
            old_rhs = q["current_rhs"]

            new_lhs = apply_op(old_lhs, pL)
            new_rhs = apply_op(old_rhs, pL)

            # Prevent no-change
            if sp.simplify(new_lhs - old_lhs) == 0 and sp.simplify(new_rhs - old_rhs) == 0:
                q["attempts"] += 1
                q["last_message"] = "❌ That didn’t change the equation."
                ss.eq_input_version += 1
                st.rerun()

            # Record BOTH: operation-applied line AND the simplified result line
            kind, k = pL
            k = sp.simplify(k)

            if kind == "add":
                # Always show "+ (k)" so k can be negative or include j
                op_display = (
                        f"({sp.latex(old_lhs)}) + ({sp.latex(k)})"
                        + " = "
                        + f"({sp.latex(old_rhs)}) + ({sp.latex(k)})"
                )
            else:
                # Multiply/divide by numeric constant (division already converted to mul by 1/k)
                op_display = (
                        f"({sp.latex(old_lhs)}) \\cdot ({sp.latex(k)})"
                        + " = "
                        + f"({sp.latex(old_rhs)}) \\cdot ({sp.latex(k)})"
                )

            result_display = sp.latex(new_lhs) + " = " + sp.latex(new_rhs)

            q["steps"].append({
                "op_display": op_display,
                "result_display": result_display,
                "op_text": (op_left or "").strip(),  # optional, for debugging
            })

            result_display = sp.latex(new_lhs) + " = " + sp.latex(new_rhs)

            q["current_lhs"] = new_lhs
            q["current_rhs"] = new_rhs

            # Check solved
            val = solved_value_if_isolated(new_lhs, new_rhs)
            if val is not None and sp.simplify(val - q["target_sol"]) == 0:
                q["correct"] = True
                q["flash_correct"] = True

                val_s = sp.simplify(val)
                q["solved_line_latex"] = r"j = " + sp.latex(val_s)  # what you want to show
                q["flash_final_latex"] = q["solved_line_latex"]  # keep your existing var too

                # hide/clear history so it won't display anywhere for this solved question
                q["steps"] = []
                q["last_message"] = ""

                ss.eq_input_version += 1
                st.rerun()

            # Progress feedback
            old_prog = measure_progress(old_lhs, old_rhs)
            new_prog = measure_progress(new_lhs, new_rhs)

            helpful = (
                    new_prog["isolated"]
                    or new_prog["j_presence"] < old_prog["j_presence"]
                    or new_prog["deg_sum"] < old_prog["deg_sum"]
                    or new_prog.get("j_side_terms", 999) < old_prog.get("j_side_terms", 999)
            )

            q["last_message"] = ("✅ Good step." if helpful else "🧠 Equivalent step, but try isolating j.")
            ss.eq_input_version += 1
            st.rerun()

    # close the card
    st.markdown("</div>", unsafe_allow_html=True)