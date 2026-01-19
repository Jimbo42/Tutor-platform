import sqlite3
import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import matplotlib.pyplot as plt
import uuid
from pathlib import Path
import shutil
from datetime import datetime
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import LETTER
import json
from sympy import Eq, symbols, solve
from sympy.parsing.latex import parse_latex
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Frame,
    PageTemplate,
    HRFlowable
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "shared_data.db"

LATEX_IMG_DIR = BASE_DIR / "latex_imgs"
LATEX_IMG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR = BASE_DIR / "pdf_files"
PDF_DIR.mkdir(parents=True, exist_ok=True)

CONSTANTS = {
    "g":  {"value": 9.81, "unit": "m/s^2", "name": "Gravitational acceleration"},
    "c":  {"value": 2.99792458e8, "unit": "m/s", "name": "Speed of light"},
    "h":  {"value": 6.62607015e-34, "unit": "J·s", "name": "Planck constant"},
    "k":  {"value": 1.380649e-23, "unit": "J/K", "name": "Boltzmann constant"},
    "R":  {"value": 8.314462618, "unit": "J/(mol·K)", "name": "Gas constant"},
    "G":  {"value": 6.67430e-11, "unit": "N·m^2/kg^2", "name": "Gravitational constant"},
    "e":  {"value": 1.602176634e-19, "unit": "C", "name": "Elementary charge"},
    "m_e": {"value": 9.1093837e-31, "unit": "kg", "name": "Electron mass"},
    "m_p": {"value": 1.6726219e-27, "unit": "kg", "name": "Proton mass"},
    "N_A": {"value": 6.02214076e23, "unit": "1/mol", "name": "Avogadro constant"},
    "pi": {"value": 3.141592653589793, "unit": "1", "name": "Pi"},
    "H_O": {"value": 2.5e-19 , "unit": "s^-1", "name": "Hubble constant"}
}

READ_ONLY_MODE = False

def set_read_only(value: bool):
    global READ_ONLY_MODE
    READ_ONLY_MODE = bool(value)

def get_conn():
    return sqlite3.connect(DB_PATH)

def load_formulas(category=None):
    conn = get_conn()

    if category and category != "All":
        df = pd.read_sql(
            "SELECT * FROM Formulas WHERE category = ? ORDER BY title",
            conn,
            params=(category,)
        )
    else:
        df = pd.read_sql(
            "SELECT * FROM Formulas ORDER BY category, title",
            conn
        )

    conn.close()
    return df

def extract_symbols_from_latex(latex):
    try:
        from sympy.parsing.latex import parse_latex
        expr = parse_latex(latex)
        return sorted({normalize_symbol_name(s.name) for s in expr.free_symbols})
    except:
        return []

def units_json_to_table(units_json: str):
    if not units_json:
        return []

    try:
        data = json.loads(units_json)
        if isinstance(data, dict) and data.get("units") == "N/A":
            return "__UNITLESS__"
        if isinstance(data, dict):
            return [{"var": k, "unit": v} for k, v in data.items()]
    except:
        pass

    return []

def table_to_units_json(rows, unitless=False):
    if unitless:
        return json.dumps({"units": "N/A"})

    out = {}
    for r in rows:
        var = r.get("var", "").strip()
        unit = r.get("unit", "").strip()
        if var:
            out[var] = unit or ""

    return json.dumps(out)

@st.dialog("Add Formula")
def formula_editor():

    with st.form("formula_editor", clear_on_submit=False):

        category = st.text_input("Category")
        title = st.text_input("Formula name")

        latex = st.text_area("LaTeX Formula", height=120)
        description = st.text_area("Description / variable definitions")

        st.markdown("### Units")

        unitless = st.checkbox("This formula is unitless (probability, ratios, etc)")

        symbols = extract_symbols_from_latex(latex)

        unit_rows = []
        for s in symbols:
            unit_rows.append({"var": s, "unit": ""})

        edited = st.data_editor(
            unit_rows,
            num_rows="dynamic",
            width="stretch",
            disabled=unitless,
            column_config={
                "var": st.column_config.TextColumn("Variable"),
                "unit": st.column_config.TextColumn("Unit")
            }
        )

        submitted = st.form_submit_button("Save Formula")

    if latex.strip():
        st.markdown("#### Preview")
        st.latex(latex)

    if submitted:
        if not (category and title and latex):
            st.error("Category, title, and LaTeX are required.")
            return

        units_json = table_to_units_json(edited, unitless)

        save_formula(category, title, latex, description, units_json)
        st.success("Formula saved.")
        st.rerun()

@st.dialog("Edit Formula")
def edit_formula_editor(row):

    with st.form("edit_formula"):

        category = st.text_input("Category", value=row["category"])
        title = st.text_input("Formula name", value=row["title"])
        latex = st.text_area("LaTeX Formula", value=row["latex"], height=120)
        description = st.text_area("Description", value=row["description"])

        st.markdown("### Units")

        existing = units_json_to_table(row.get("units", ""))

        unitless = False
        if existing == "__UNITLESS__":
            unitless = True
            table_data = []
        else:
            table_data = existing or []

        unitless = st.checkbox("This formula is unitless (probability, ratios, etc)", value=unitless)

        # Auto-refresh variables from latex
        symbols = extract_symbols_from_latex(latex)

        # Ensure all symbols exist in table
        known = {r["var"] for r in table_data}
        for s in symbols:
            if s not in known:
                table_data.append({"var": s, "unit": ""})

        edited = st.data_editor(
            table_data,
            num_rows="dynamic",
            width="stretch",
            disabled=unitless,
            column_config={
                "var": st.column_config.TextColumn("Variable"),
                "unit": st.column_config.TextColumn("Unit")
            }
        )

        submitted = st.form_submit_button("Save Changes")

    if latex.strip():
        st.markdown("#### Preview")
        st.latex(latex)

    if submitted:
        units_json = table_to_units_json(edited, unitless)

        update_formula(
            row["id"],
            category.strip(),
            title.strip(),
            latex.strip(),
            description.strip(),
            units_json
        )

        st.success("Formula updated")
        st.rerun()

def save_formula(category, title, latex, description, units_json):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO Formulas (category, title, latex, description, units)
        VALUES (?, ?, ?, ?, ?)
        """,
        (category.strip(), title.strip(), latex.strip(), description.strip(), units_json)
    )

    conn.commit()
    conn.close()

def update_formula(fid, category, title, latex, description, units_json):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE Formulas
        SET category = ?, title = ?, latex = ?, description = ?, units = ?
        WHERE id = ?
        """,
        (category, title, latex, description, units_json, fid)
    )

    conn.commit()
    conn.close()

@st.dialog("Delete Formula")
def delete_formula_dialog(row):
    st.warning("⚠️ This action cannot be undone.")
    st.markdown(f"**{row['title']}**")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Delete", type="primary"):
            delete_formula(row["id"])
            st.success("Formula deleted")
            st.rerun()

    with col2:
        if st.button("Cancel"):
            st.rerun()

def delete_formula(fid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM Formulas WHERE id = ?", (fid,))
    conn.commit()
    conn.close()

def get_categories():
    conn = get_conn()
    rows = conn.execute(
        "SELECT DISTINCT category FROM Formulas ORDER BY category"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]

def clear_latex_cache(dir_path: Path = LATEX_IMG_DIR):
    if dir_path.exists():
        shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

def sanitize_latex(latex: str) -> str:
    if not latex:
        return ""

    # Strip whitespace
    s = latex.strip()

    # Remove surrounding $ or $$ if present
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()

    return s

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def latex_to_image(latex, out_dir: Path = LATEX_IMG_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = out_dir / f"{uuid.uuid4().hex}.png"

    safe = sanitize_latex(latex)
    if not safe:
        return None

    # 🔑 Shorter canvas + smaller font
    fig = plt.figure(figsize=(4.8, 0.9))
    fig.patch.set_alpha(0)

    plt.text(
        0.5, 0.5,
        f"${safe}$",
        fontsize=16,
        ha="center",
        va="center"
    )
    plt.axis("off")

    plt.savefig(
        fname,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04
    )
    plt.close(fig)

    return fname

def generate_formula_pdf(df, filename):
    styles = getSampleStyleSheet()

    # Compact, readable fonts
    styles["Heading5"].fontSize = 10
    styles["Heading5"].leading = 11
    styles["Normal"].fontSize = 8
    styles["Normal"].leading = 9

    doc = SimpleDocTemplate(
        filename,
        pagesize=LETTER,
        rightMargin=32,
        leftMargin=32,
        topMargin=32,
        bottomMargin=32
    )

    # Two-column layout
    gutter = 14
    left_margin = 32
    right_margin = 32
    top_margin = 32
    bottom_margin = 32

    frame_width = (LETTER[0] - left_margin - right_margin - gutter) / 2

    # 🔑 Reduce usable height to improve flow
    extra_buffer = 46
    frame_height = LETTER[1] - (top_margin + bottom_margin + extra_buffer)

    frames = [
        Frame(left_margin, bottom_margin, frame_width, frame_height, id="col1"),
        Frame(left_margin + frame_width + gutter, bottom_margin, frame_width, frame_height, id="col2"),
    ]

    doc.addPageTemplates([
        PageTemplate(id="TwoColumn", frames=frames)
    ])

    story = []

    for _, row in df.iterrows():

        # --- Title ---
        story.append(Paragraph(
            f"<b>{row['title']}</b>",
            styles["Heading5"]
        ))
        story.append(Spacer(1, 2))

        # --- Formula image ---
        img_path = latex_to_image(row["latex"])
        if img_path and img_path.exists():
            img = Image(img_path)
            img._restrictSize(frame_width - 20, 60)  # max width, max height
            story.append(img)
            story.append(Spacer(1, 3))

        # --- Description (de-emphasized) ---
        story.append(Paragraph(
            f"<font color='#444444'>{row['description']}</font>",
            styles["Normal"]
        ))

        # --- Subtle separator ---
        story.append(Spacer(1, 4))
        story.append(HRFlowable(
            width="100%",
            thickness=0.4,
            color="#CCCCCC"
        ))
        story.append(Spacer(1, 6))

    # 🔑 THIS is what actually writes the file
    doc.build(story)

def normalize_symbol_name(sym_name: str) -> str:
    # Convert E_{p} -> E_p
    return sym_name.replace("{", "").replace("}", "")

import math

def format_sigfig(x, sig=3):
    """
    Format number with significant figures and auto scientific notation.
    """
    if x == 0:
        return "0"

    x = float(x)
    absx = abs(x)

    # Decide whether to use scientific notation
    if absx >= 1e4 or absx < 1e-3:
        # Scientific notation with sig figs
        return f"{x:.{sig-1}e}"
    else:
        # Normal notation with sig figs
        digits = sig - int(math.floor(math.log10(absx))) - 1
        digits = max(digits, 0)
        return f"{x:.{digits}f}"

@st.dialog("Formula Solver")
def show_formula_solver(row):

    st.subheader(row["title"])
    st.latex(row["latex"])

    # ---------- Load units ----------
    try:
        units_raw = row.get("units", "")
        if isinstance(units_raw, str):
            units_data = json.loads(units_raw) if units_raw.strip() else {}
        else:
            units_data = {}
    except:
        units_data = {}

    unitless_formula = isinstance(units_data, dict) and units_data.get("units") == "N/A"

    # ---------- Parse formula ----------
    try:
        expr = parse_latex(row["latex"])
    except Exception as e:
        st.error("Could not parse formula.")
        st.exception(e)
        return

    symbols = sorted(expr.free_symbols, key=lambda s: s.name)

    # ---------- Detect missing unit definitions ----------
    missing_units = []
    if not unitless_formula:
        for s in symbols:
            name = normalize_symbol_name(s.name)
            if name not in units_data:
                missing_units.append(name)

    if missing_units and not unitless_formula:
        st.warning(
            f"⚠️ This formula has missing unit definitions for: {', '.join(missing_units)}"
        )

        if not READ_ONLY_MODE:
            if st.button("🛠 Fix this formula now"):
                ss.open_edit_formula = row
                st.rerun()
        else:
            st.info("⚠️ This formula needs fixing, but editing is disabled in this app.")

    # ---------- Input UI ----------
    st.markdown("### Enter values (leave one blank):")

    input_values = {}
    result_slots = {}

    for s in symbols:
        name = normalize_symbol_name(s.name)

        unit = ""
        if not unitless_formula:
            unit = units_data.get(name, "")

        label = name
        if unit:
            label = f"{name} ({unit})"

        # ---------- Check if constant ----------
        default_value = ""
        const_info = None

        if name in CONSTANTS:
            const = CONSTANTS[name]

            # Only auto-fill constants for formulas WITH units
            if not unitless_formula and name in CONSTANTS:
                const = CONSTANTS[name]

                # Only prefill if units match
                if unit == const["unit"]:
                    default_value = str(const["value"])
                    const_info = const

        c1, c2 = st.columns([2, 1])

        with c1:
            val = st.text_input(
                label,
                value=default_value,
                key=f"solver_val_{row['id']}_{name}"
            )

            if const_info:
                st.caption(f"{const_info['name']}")

        with c2:
            st.markdown("**Result**")
            result_slots[name] = st.empty()

        if val.strip() == "":
            input_values[name] = None
        else:
            try:
                input_values[name] = float(val)
            except:
                st.error(f"Invalid number for {name}")
                return

    # ---------- Solve ----------
    if st.button("🧮 Calculate"):

        missing = [k for k, v in input_values.items() if v is None]

        if len(missing) != 1:
            st.error("You must leave exactly ONE variable empty")
            return

        target_name = missing[0]
        target_symbol = None

        for s in symbols:
            if normalize_symbol_name(s.name) == target_name:
                target_symbol = s
                break

        # Substitute knowns
        subs = {}
        for s in symbols:
            name = normalize_symbol_name(s.name)
            if input_values[name] is not None:
                subs[s] = input_values[name]

        try:
            sol = solve(expr, target_symbol)

            if not sol:
                st.error("Could not solve for this variable.")
                return

            raw_result = sol[0].subs(subs)

            numeric = float(raw_result)

            # ---------- Significant figures formatting ----------
            def format_sig(x, sig=3):
                if x == 0:
                    return "0"
                return f"{x:.{sig}g}"

            formatted = format_sig(numeric, 3)

            unit = ""
            if not unitless_formula:
                unit = units_data.get(target_name, "")

            # ---------- Display next to field ----------
            result_slots[target_name].success(f"{formatted} {unit}".strip())

        except Exception as e:
            st.error("Calculation failed")
            st.exception(e)

def solve_formula(latex_str, known_values):

    expr = parse_latex(latex_str)

    if not hasattr(expr, "lhs"):
        raise ValueError("Formula must be an equation")

    symbols = list(expr.free_symbols)

    # Normalize names
    sym_map = {s: normalize_symbol_name(s.name) for s in symbols}

    # All variables come from the formula itself
    variable_names = set(sym_map.values())

    # Find which variable is missing
    missing = [name for name in variable_names if known_values.get(name) is None]

    if len(missing) != 1:
        raise ValueError("You must leave exactly ONE variable empty")

    target_name = missing[0]

    # Find actual sympy symbol
    target_sym = None
    for s, n in sym_map.items():
        if n == target_name:
            target_sym = s
            break

    if target_sym is None:
        raise ValueError("Could not identify target variable in formula")

    # Build substitution dict
    subs = {}
    for s, name in sym_map.items():
        if name in known_values and known_values[name] is not None:
            subs[s] = float(known_values[name])

    # Solve
    solved = solve(expr, target_sym)

    if not solved:
        raise ValueError("Could not solve for variable")

    result = solved[0].subs(subs)

    try:
        result = float(result)
    except:
        result = float(result.evalf())

    return target_name, result

def show_formulas():

    if "formula_pdf_path" not in ss:
        ss.formula_pdf_path = None

    # Open editor if requested by solver
    if "open_edit_formula" in ss:
        row = ss.open_edit_formula
        del ss.open_edit_formula
        edit_formula_editor(row)
        return

    # ---------------- Top controls ----------------

    col_1, col_2, col_3 = st.columns([2, 2, 1])

    with col_1:
        categories = get_categories()
        category = st.selectbox(
            "Category",
            options=[""] + categories + ["➕ New category"]
        )

        if category == "➕ New category":
            category = st.text_input("New category name")

    df = load_formulas(category)

    with col_2:
        search = st.text_input("🔍 Search formulas by title", placeholder="e.g. energy, velocity, lens...")
        if search.strip():
            df = df[df["title"].str.contains(search, case=False, na=False)]

    with col_3:
        if not READ_ONLY_MODE:
            if st.button("➕", key="formula_editor", help="Add New Formula"):
                formula_editor()

    # ---------------- PDF Export ----------------

    if st.button("📄", key="pdf_export", help="Export formulas to PDF"):
        output_dir = "saved_files"
        ensure_dir(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = PDF_DIR / f"formulas_export_{timestamp}.pdf"

        clear_latex_cache()
        generate_formula_pdf(df, pdf_path)

        ss.formula_pdf_path = pdf_path

    if ss.formula_pdf_path:
        with open(ss.formula_pdf_path, "rb") as f:
            st.download_button(
                "⬇️ Download PDF",
                f,
                file_name=ss.formula_pdf_path,
                mime="application/pdf"
            )

    # ---------------- Cards ----------------

    cols = st.columns(2)

    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 2]:
            with st.container(border=True):

                hcol, bcol = st.columns([3, 2], vertical_alignment="center")

                # ---------- Header ----------
                with hcol:
                    # Unit validation
                    has_unit_problem = False
                    missing_units = []

                    try:
                        raw_units = row.get("units")

                        # ---------- Explicitly unitless ----------
                        if raw_units:
                            parsed_units = json.loads(raw_units)

                            if isinstance(parsed_units, dict) and parsed_units.get("units") == "N/A":
                                # Intentionally unitless formula → skip validation
                                has_unit_problem = False
                                missing_units = []

                            else:
                                if not isinstance(parsed_units, dict):
                                    raise ValueError("Units not dict")

                                from sympy.parsing.latex import parse_latex
                                expr = parse_latex(row["latex"])

                                symbols = {normalize_symbol_name(s.name) for s in expr.free_symbols}
                                missing_units = [s for s in symbols if s not in parsed_units]

                                if missing_units:
                                    has_unit_problem = True
                        else:
                            # No units field at all → problem
                            has_unit_problem = True
                            missing_units = ["(no units defined)"]

                    except:
                        has_unit_problem = True
                        missing_units = ["(parse error)"]

                    header_cols = st.columns([2, 5], gap="small")

                    # ℹ️ Info
                    with header_cols[0]:
                        with st.popover("ℹ️"):
                            st.markdown("**Description:**")
                            st.write(row.get("description", ""))

                            st.markdown("---")
                            st.markdown("**Units:**")
                            st.code(row.get("units", ""), language="json")

                            if has_unit_problem:
                                st.error("Missing or invalid units for:")
                                st.write(missing_units)

                        if has_unit_problem:
                            st.warning("⚠️", icon=None)

                    # Title
                    with header_cols[1]:
                        st.markdown(f"**{row['title']}**")

                # ---------- Buttons ----------
                with bcol:
                    ic1, ic2, ic3 = st.columns([1, 1, 1], gap=None)

                    if not READ_ONLY_MODE:
                        with ic1:
                            if st.button(" ✏️", key=f"edit_{row['id']}"):
                                edit_formula_editor(row)

                        with ic2:
                            if st.button(" 🗑️", key=f"del_{row['id']}"):
                                delete_formula_dialog(row)
                    else:
                        with ic1:
                            st.empty()
                        with ic2:
                            st.empty()

                    with ic3:
                        if st.button("🔢", key=f"use_{row['id']}"):
                            show_formula_solver(row)

                # ---------- Formula ----------
                st.latex(row["latex"])
