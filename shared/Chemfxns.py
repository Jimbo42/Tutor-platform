import sqlite3
import pandas as pd
import re
from collections import defaultdict
import sympy as sp
import streamlit as st
from streamlit import session_state as ss
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "shared_data.db"

# --- STP Defaults (user may override via dialog)
DEFAULT_TEMP_K = 273.15        # Kelvin
DEFAULT_PRESSURE_KPA = 101.325 # kPa

def get_conn():
    return sqlite3.connect(DB_PATH)

def create_reactants():
    df = pd.DataFrame({
        "Coeff": pd.Series(dtype='int'),
        "Formula": pd.Series(dtype='object'),
        "MolarMass": pd.Series(dtype='float64'),

        "Mass_1": pd.Series(dtype='float64'),
        "Mass_2": pd.Series(dtype='float64'),

        "Moles_1": pd.Series(dtype='float64'),
        "Moles_2": pd.Series(dtype='float64'),

        "ExcessMass": pd.Series(dtype='float64'),
        "IsLimiting_1": pd.Series(dtype='bool'),

        "Volume_1": pd.Series(dtype='float64'),
        "Volume_2": pd.Series(dtype='float64'),

        "Concentration_1": pd.Series(dtype='float64'),
        "Concentration_2": pd.Series(dtype='float64'),

        "State": pd.Series(dtype='object')

    })
    return df

def create_products():
    df = pd.DataFrame({
        "Coeff": pd.Series(dtype='int'),
        "Formula": pd.Series(dtype='object'),
        "MolarMass": pd.Series(dtype='float64'),

        "Mass_1": pd.Series(dtype='float64'),
        "Mass_2": pd.Series(dtype='float64'),

        "Moles_1": pd.Series(dtype='float64'),
        "Moles_2": pd.Series(dtype='float64'),

        "MeasuredYieldMass": pd.Series(dtype='float64'),
        "PercentYield": pd.Series(dtype='float64'),

        "Volume_1": pd.Series(dtype='float64'),
        "Volume_2": pd.Series(dtype='float64'),

        "Concentration_1": pd.Series(dtype='float64'),
        "Concentration_2": pd.Series(dtype='float64'),

        "State": pd.Series(dtype='object')
    })
    return df

def create_elements():
    df = pd.DataFrame({
        "Name": pd.Series(dtype='object'),
        "Symbol": pd.Series(dtype='object'),
        "Total Acc": pd.Series(dtype='float64'),
        "Atomic Mass": pd.Series(dtype='float64'),
        "Comp. Mass": pd.Series(dtype='float64'),
        "% Mass": pd.Series(dtype='float64'),
        "Oxi State": pd.Series(dtype='float64')
    })
    return df

def compound_calc(formula, mode):

    # clear the current table and text input and enable mole/mass calcs
    ss.elements = ss.elements[0:0]

    num_rows = len(formula)
    char_array = [[formula[i], 1, 1, 0, ""] for i in range(num_rows)]

    _row = num_rows - 1
    _element = ""
    _multiplier = ""
    _sanity = True

    while _row >= 0:
        if char_array[_row][0] == ")":
            # end of a group - we shouldn't hit this
            print("how did I get here? - I am a )")
            _sanity = False
            _row -= 1
        elif char_array[_row][0] == "(":
            # beginning of group
            _row -= 1
        elif char_array[_row][0].isdigit():
            # multiplier present
            if _row == 0:
                # this is a problem - first character in compound shouldn't be a digit
                print("first character can't be a digit")
                _row -= 1
                _sanity = False
            elif char_array[_row - 1][0].isdigit():
                # there is more than one character in the multiplier
                _multiplier = char_array[_row][0] + _multiplier
                _row -= 1
            elif char_array[_row - 1][0] == ")":
                # a bracket means we have a group multiplier
                _multiplier = char_array[_row][0] + _multiplier
                char_array[_row][2] = int(_multiplier)
                _multiplier = ""
                # set the multiplier to all rows within the brackets
                _set = _row - 2
                while _set >= 0:
                    if char_array[_set][0] == "(":
                        _set = -1
                    else:
                        char_array[_set][2] = char_array[_row][2]
                        _set -= 1
                # now we can move past the row and the bracket
                _row -= 2
            else:
                # this is a sub multiplier - apply it to row above
                _multiplier = char_array[_row][0] + _multiplier
                char_array[_row - 1][1] = int(_multiplier)
                _multiplier = ""
                _row -= 1
        elif char_array[_row][0].islower():
            # last character in two character name
            _element = char_array[_row][0]
            # apply the sub multiplier up one row
            char_array[_row - 1][1] = char_array[_row][1]
            _row -= 1
        elif char_array[_row][0].isupper():
            # create the element symbol - if one character is already there then this goes first
            _element = char_array[_row][0] + _element
            char_array[_row][4] = _element
            _element = ""
            # calculate and record totals
            char_array[_row][3] = char_array[_row][1] * char_array[_row][2]
            _row -= 1

    if _sanity:
        # we now have updated our array so that all elements have real values in 5th parameter
        for row in char_array:
            if not _sanity:
                break

            if len(row[4]) > 0:
                # check if already exists in the table
                _match = ss.elements.loc[ss.elements["Symbol"] == row[4]].index
                if not _match.empty:
                    _item = _match[0]
                else:
                    _item = -1

                if _item >= 0:
                    _acc = ss.elements.iloc[_item, 2] + row[3]
                    ss.elements.iloc[_item, 2] = _acc
                    _mass = ss.elements.iloc[_item, 3]
                    _mol_mass = _mass * _acc
                    ss.elements.iloc[_item, 4] = _mol_mass

                else:
                    empty_row = pd.Series([None] * len(ss.elements.columns),
                                          index=ss.elements.columns)
                    ss.elements = ss.elements._append(empty_row, ignore_index=True)
                    _last_row = ss.elements.index[-1]

                    # Symbol
                    ss.elements.iat[_last_row, 1] = row[4]

                    # Accumulation
                    ss.elements.iat[_last_row, 2] = row[3]

                    # Lookup element by symbol from Elements table
                    conn = get_conn()
                    ce = conn.cursor()
                    with conn:
                        ce.execute(
                            "SELECT Name, Atomic_Weight, Oxidation_States "
                            "FROM Elements WHERE Symbol = :symbol",
                            {"symbol": row[4]},
                        )
                        elem = ce.fetchone()

                    conn.close()

                    # ❗ Unknown element symbol -> mark as invalid
                    if elem is None:
                        _sanity = False
                        break

                    # Name
                    ss.elements.iat[_last_row, 0] = elem[0]
                    # Atomic Mass
                    ss.elements.iat[_last_row, 3] = round(elem[1], 3)
                    # Compound Mass
                    ss.elements.iat[_last_row, 4] = elem[1] * row[3]
                    # Oxidation state
                    ss.elements.iat[_last_row, 6] = elem[2]

        if _sanity:
            # Get the totals and calculations
            _total_sum = ss.elements.iloc[:, 4].sum()

            # Percent Mass from calculation and accumulate
            for index, row in ss.elements.iterrows():
                _mol_mass = row["Comp. Mass"]
                _per_mass = round(100 * _mol_mass / _total_sum, 2)
                ss.elements.at[index, "% Mass"] = _per_mass

            # Add a summary row
            empty_row = pd.Series([None] * len(ss.elements.columns),
                                  index=ss.elements.columns)
            ss.elements = ss.elements._append(empty_row, ignore_index=True)
            _last_row = ss.elements.index[-1]

            ss.elements.iat[_last_row, 1] = formula
            ss.elements.iat[_last_row, 4] = _total_sum
            ss.elements.iat[_last_row, 3] = None
            ss.elements.iat[_last_row, 5] = None
            ss.elements.iat[_last_row, 2] = 1
            ss.elements.iat[_last_row, 0] = ""
            ss.elements.iat[_last_row, 6] = ""

            ss.mol_mass = _total_sum

    # instead of crashing silently, just report invalid
    if not _sanity:
        ss.mol_mass = None

    return _sanity

def chem_format(s: str) -> str:
    # Subscript only digits that are immediately after a letter (A–Z, a–z) or ')'
    return re.sub(r'(?<=[A-Za-z\)])(\d+)', r'<sub>\1</sub>', s)

def display_side(df):
    if df.empty:
        return "—"

    parts = []

    for _, row in df.iterrows():
        coeff = "" if int(row["Coeff"]) == 1 else f"{int(row['Coeff'])}"

        formula_html = chem_format(row["Formula"])

        state = (row["State"] or "").strip()
        if state:
            # normalize like "(aq)" "(s)" etc.
            state = state.replace("[", "").replace("]", "").replace("{", "").replace("}", "")
            state_html = f"<span style='font-size:0.7em'> {state}</span>"
        else:
            state_html = ""

        parts.append(f"{coeff}{formula_html}{state_html}")

    return " + ".join(parts)

def check_formula():
    f = ss.chem_formula.strip()

    if not safe_parse(f):
        st.toast(f"'{f}' is not a valid chemical formula.")
        return

    ss.formula = f
    ss.chem_formula = ""
    st.toast(f"✔ {ss.formula}", icon="🧪")

def safe_parse(formula: str):
    try:
        result = parse_formula(formula)
        if not result or not isinstance(result, dict):
            raise ValueError("No atoms parsed")
        return result
    except Exception:
        return None

# Very basic chemical formula parser (supports parentheses, no nesting beyond one level)
def parse_formula(formula):
    tokens = re.findall(r'([A-Z][a-z]?|\(|\)|\d+)', formula)
    stack = [defaultdict(int)]
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == "(":
            stack.append(defaultdict(int))

        elif token == ")":
            temp = stack.pop()
            i += 1
            multiplier = int(tokens[i]) if (i < len(tokens) and tokens[i].isdigit()) else 1
            for k, v in temp.items():
                stack[-1][k] += v * multiplier
        elif token.isdigit():
            prev = list(stack[-1].keys())[-1]
            stack[-1][prev] += (int(token) - 1)
        else:
            stack[-1][token] += 1

        i += 1

    return stack.pop()

def summed_atoms(df):
    totals = defaultdict(int)

    for _, row in df.iterrows():
        atoms = safe_parse(row["Formula"])
        if atoms is None:
            return None   # signals invalid formula

        coeff = int(row["Coeff"])
        for el, n in atoms.items():
            totals[el] += n * coeff

    return dict(totals)

def balance_equation():
    if ss.reactants.empty or ss.products.empty:
        return False

    # --- Collect formulas ---
    react = list(ss.reactants["Formula"])
    prod  = list(ss.products["Formula"])
    species = react + prod

    # --- Collect unique elements using YOUR parser ---
    elements = set()
    parsed = []
    for f in species:
        d = parse_formula(f)
        parsed.append(d)
        elements.update(d.keys())
    elements = sorted(elements)

    # --- Build coefficient matrix ---
    rows = []
    for el in elements:
        row = []

        # Reactants positive
        for d in parsed[:len(react)]:
            row.append(d.get(el, 0))

        # Products negative
        for d in parsed[len(react):]:
            row.append(-d.get(el, 0))

        rows.append(row)

    M = sp.Matrix(rows)
    ns = M.nullspace()

    if not ns:
        return False  # cannot balance

    vec = ns[0]

    # Clear fractions
    lcm = sp.lcm([term.q for term in vec])
    coeffs = [abs(int(v * lcm)) for v in vec]

    # --- Apply coefficients back into dataframes ---
    ss.reactants["Coeff"] = coeffs[:len(react)]
    ss.products["Coeff"]  = coeffs[len(react):]

    return True

@st.dialog("Gas Law Configuration")
def show_gas_config():

    # Ensure session persistence
    if "gas_temp_K" not in ss:
        ss.gas_temp_K = DEFAULT_TEMP_K

    if "gas_pressure_kpa" not in ss:
        ss.gas_pressure_kpa = DEFAULT_PRESSURE_KPA

    st.write("These values will be used for Ideal Gas Law calculations.")

    col1, col2 = st.columns(2)

    with col1:
        temp = st.number_input(
            "Temperature (K)",
            min_value=1.0,
            value=float(ss.gas_temp_K),
            step=0.1
        )

    with col2:
        pressure = st.number_input(
            "Pressure (kPa)",
            min_value=0.001,
            value=float(ss.gas_pressure_kpa),
            step=0.1
        )

    # Ideal Gas:  V = RT / P
    R = 8.314  # kPa·L/(mol·K)
    est_molar_volume = R * temp / pressure

    st.markdown(
        f"**Estimated Molar Volume:** "
        f"{est_molar_volume:.3f} L/mol  "
        f"*(Ideal Gas Law:  V = RT / P)*"
    )

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Cancel"):
            st.rerun()

    with c2:
        if st.button("Save Settings"):
            ss.gas_temp_K = temp
            ss.gas_pressure_kpa = pressure
            st.success("Gas conditions updated.")
            st.rerun()


STATE_ICONS = {
    "(s)": "🧱",
    "(aq)": "🧪",
    "(g)": "☁️",
    "(l)": "💧"
}

STATE_HELP = {
    "(s)": "Solid phase (click to change)",
    "(aq)": "Aqueous dissolved phase (click to change)",
    "(g)": "Gas phase (click to change)",
    "(l)": "Liquid phase (click to change)"
}

PHASES = ["(s)", "(aq)", "(g)", "(l)"]

def cycle_phase(df, idx):
    PHASES = ["(s)", "(aq)", "(g)", "(l)"]

    cur = df.at[idx, "State"]
    if cur not in PHASES:
        cur = "(s)"

    i = PHASES.index(cur)
    new_state = PHASES[(i + 1) % len(PHASES)]
    df.at[idx, "State"] = new_state

    # Apply new state logic to both lanes
    apply_state_rules(df, idx, new_state)

def apply_state_rules(df, idx, new_state):
    """
    Enforce state rules and recompute dependent fields for BOTH lanes.
    """

    # M = df.at[idx, "MolarMass"]

    # Gas constants
    P = ss.get("gas_pressure_kpa", 101.325)
    T = ss.get("gas_temp_K", 273.15)
    R = 8.314  # kPa·L/(mol·K)

    for lane in (1, 2):
        mol_col  = f"Moles_{lane}"
        #mass_col = f"Mass_{lane}"
        vol_col  = f"Volume_{lane}"
        conc_col = f"Concentration_{lane}"

        n = df.at[idx, mol_col]

        # -------------------------------
        # SOLID / LIQUID
        # -------------------------------
        if new_state in ("(s)", "(l)"):
            df.at[idx, vol_col] = None
            df.at[idx, conc_col] = None
            continue

        # If no moles, nothing to compute
        if n in (None, "", 0):
            df.at[idx, vol_col] = None
            df.at[idx, conc_col] = None
            continue

        # -------------------------------
        # AQUEOUS
        # -------------------------------
        if new_state == "(aq)":
            V = df.at[idx, vol_col]
            C = df.at[idx, conc_col]

            # If volume exists, recompute conc
            if V not in (None, "", 0):
                C = n / (V / 1000)

            # Else if conc exists, recompute volume
            elif C not in (None, "", 0):
                V = (n / C) * 1000

            # Else: neither exists → leave both as None (do nothing)

            df.at[idx, vol_col] = V
            df.at[idx, conc_col] = C

        # -------------------------------
        # GAS
        # -------------------------------
        elif new_state == "(g)":
            # Compute volume in L
            V = (n * R * T) / P

            df.at[idx, vol_col] = V

            # Derived concentration (not user-editable)
            if V not in (None, 0):
                df.at[idx, conc_col] = n / V
            else:
                df.at[idx, conc_col] = None

def field_permissions(state):
    if state in ("(s)", "(l)"):
        return False, False, "mL"

    if state == "(aq)":
        return True, True, "mL"

    if state == "(g)":
        return True, False, "L"

    return False, False, "mL"

def clear_all_inputs():
    """
    Clears all numeric input and derived fields from reactants and products.
    Keeps equation structure, states, coefficients, molar masses.
    """

    # ---------- Reactants ----------
    for df in (ss.reactants,):
        if "Mass_1" in df: df["Mass_1"] = None
        if "Mass_2" in df: df["Mass_2"] = None
        if "Moles_1" in df: df["Moles_1"] = None
        if "Moles_2" in df: df["Moles_2"] = None
        if "Volume_1" in df: df["Volume_1"] = None
        if "Volume_2" in df: df["Volume_2"] = None
        if "Concentration_1" in df: df["Concentration_1"] = None
        if "Concentration_2" in df: df["Concentration_2"] = None
        if "ExcessMass" in df: df["ExcessMass"] = None
        if "IsLimiting_1" in df: df["IsLimiting_1"] = False

    # ---------- Products ----------
    for df in (ss.products,):
        if "Mass_1" in df: df["Mass_1"] = None
        if "Mass_2" in df: df["Mass_2"] = None
        if "Moles_1" in df: df["Moles_1"] = None
        if "Moles_2" in df: df["Moles_2"] = None
        if "Volume_1" in df: df["Volume_1"] = None
        if "Volume_2" in df: df["Volume_2"] = None
        if "Concentration_1" in df: df["Concentration_1"] = None
        if "Concentration_2" in df: df["Concentration_2"] = None
        if "MeasuredYieldMass" in df: df["MeasuredYieldMass"] = None
        if "PercentYield" in df: df["PercentYield"] = None

    # ---------- Clear anchors & stoich flags ----------
    ss.anchor_1 = None
    ss.anchor_2 = None

    clear_stoichiometry_results()

    # ---------- Clear all widget caches ----------
    # Brutal but safe: only numeric fields
    for k in list(ss.keys()):
        if (
            k.startswith("mass_") or
            k.startswith("mol_") or
            k.startswith("vol_") or
            k.startswith("conc_") or
            k.startswith("yield_") or
            k.startswith("exc_")
        ):
            del ss[k]

def on_qty_change(side, idx, lane, field):
    from streamlit import session_state as ss

    df = ss.reactants if side == "r" else ss.products

    # Map to column names
    mass_col = f"Mass_{lane}"
    mol_col  = f"Moles_{lane}"
    vol_col  = f"Volume_{lane}"
    conc_col = f"Concentration_{lane}"

    # Widget key map
    key_map = {
        "mass":  f"mass_{side}_{idx}_{lane}",
        "moles": f"mol_{side}_{idx}_{lane}",
        "vol":   f"vol_{side}_{idx}_{lane}",
        "conc":  f"conc_{side}_{idx}_{lane}",
        "yield": f"yield_{side}_{idx}",
    }

    key = key_map[field]
    raw = ss.get(key, "").strip()

    # ---------- EMPTY = CLEAR ----------
    if raw == "":
        if field == "mass":
            df.at[idx, mass_col] = None
            df.at[idx, mol_col]  = None

        elif field == "moles":
            df.at[idx, mol_col]  = None
            df.at[idx, mass_col] = None

        elif field == "vol":
            df.at[idx, vol_col]  = None

        elif field == "conc":
            df.at[idx, conc_col] = None

        elif field == "yield" and side == "p":
            df.at[idx, "MeasuredYieldMass"] = None

        refresh_row(side, idx, lane)
        maybe_evaluate_stoichiometry()
        return

    # ---------- PARSE NUMBER ----------
    try:
        value = float(raw)
    except ValueError:
        # Leave widget as-is; do not corrupt DF
        return

    M = df.at[idx, "MolarMass"]
    state = df.at[idx, "State"]

    # ---------- STATE PERMISSIONS ----------
    if field in ("vol", "conc"):
        if state in ("(s)", "(l)"):
            return
        if state == "(g)" and field == "conc":
            return

    # ---------- MASS / MOLES ----------
    if field == "mass":
        df.at[idx, mass_col] = value
        n = value / M
        df.at[idx, mol_col] = n

    elif field == "moles":
        df.at[idx, mol_col] = value
        df.at[idx, mass_col] = value * M

    # ---------- AQUEOUS ----------
    if state == "(aq)":
        V = df.at[idx, vol_col]
        C = df.at[idx, conc_col]
        n = df.at[idx, mol_col]

        # Volume edited
        if field == "vol":
            df.at[idx, vol_col] = value
            V = value

            if C not in (None, 0):
                n = C * (V / 1000)
                df.at[idx, mol_col] = n
                df.at[idx, mass_col] = n * M

            elif n not in (None, 0):
                C = n / (V / 1000)
                df.at[idx, conc_col] = C

        # Concentration edited
        elif field == "conc":
            df.at[idx, conc_col] = value
            C = value

            if V not in (None, 0):
                n = C * (V / 1000)
                df.at[idx, mol_col] = n
                df.at[idx, mass_col] = n * M

    # ---------- GAS ----------
    if state == "(g)":
        P = ss.get("gas_pressure_kpa", 101.325)
        T = ss.get("gas_temp_K", 273.15)
        R = 8.314  # kPa·L/(mol·K)

        # Volume edited
        if field == "vol":
            df.at[idx, vol_col] = value
            V = value
            n = (P * V) / (R * T)
            df.at[idx, mol_col] = n
            df.at[idx, mass_col] = n * M

        # Mass or moles edited
        elif field in ("mass", "moles"):
            n = df.at[idx, mol_col]
            if n not in (None, 0):
                V = (n * R * T) / P
                df.at[idx, vol_col] = V

        # Always update derived concentration
        V = df.at[idx, vol_col]
        n = df.at[idx, mol_col]
        if V not in (None, 0) and n not in (None, 0):
            df.at[idx, conc_col] = n / V
        else:
            df.at[idx, conc_col] = None

    # ---------- PRODUCT YIELD ----------
    if field == "yield" and side == "p":
        df.at[idx, "MeasuredYieldMass"] = value

        # Let the main solver handle % yield
        maybe_evaluate_stoichiometry()

        # Refresh this product row
        refresh_row("p", idx, 1)
        refresh_row("p", idx, 2)
        return

    # ---------- FORCE UI REFRESH FOR THIS ROW ----------
    refresh_row(side, idx, lane)

    # ---------- PROPAGATION ----------
    if lane == 1 and ss.anchor_1 == (side, idx):
        propagate_from_anchor(side, idx, 1)

    if lane == 2 and ss.anchor_2 == (side, idx):
        propagate_from_anchor(side, idx, 2)

    # ---------- STOICHIOMETRY ----------
    maybe_evaluate_stoichiometry()

def refresh_row(side, idx, lane):
    """
    Delete widget keys for one row+lane so they are recreated from DF on next rerun.
    """
    keys = [
        f"mass_{side}_{idx}_{lane}",
        f"mol_{side}_{idx}_{lane}",
        f"vol_{side}_{idx}_{lane}",
        f"conc_{side}_{idx}_{lane}",
    ]

    # Product-only fields
    if side == "p":
        keys.append(f"yield_{side}_{idx}")
        keys.append(f"yield_pct_{side}_{idx}")

    for k in keys:
        if k in ss:
            del ss[k]

def propagate_from_anchor(side, idx, lane):
    """
    Propagate stoichiometry from one anchor (side, idx, lane)
    """
    from streamlit import session_state as ss

    # Which DF is anchor in?
    src_df = ss.reactants if side == "r" else ss.products

    n_anchor = src_df.at[idx, f"Moles_{lane}"]
    c_anchor = src_df.at[idx, "Coeff"]

    if n_anchor in (None, "", 0) or c_anchor in (None, 0):
        return

    extent = n_anchor / c_anchor

    # Walk both sides
    for df in (ss.reactants, ss.products):
        for i, row in df.iterrows():

            coeff = row["Coeff"]
            if coeff in (None, 0):
                continue

            n_target = extent * coeff

            df.at[i, f"Moles_{lane}"] = n_target
            df.at[i, f"Mass_{lane}"]  = n_target * row["MolarMass"]

            state = row["State"]

            vol_col  = f"Volume_{lane}"
            conc_col = f"Concentration_{lane}"

            # ---------- SOLID / LIQUID ----------
            if state in ("(s)", "(l)"):
                df.at[i, vol_col]  = None
                df.at[i, conc_col] = None

            # ---------- AQUEOUS ----------
            elif state == "(aq)":
                V = df.at[i, vol_col]
                if V not in (None, "", 0):
                    df.at[i, conc_col] = n_target / (V / 1000)
                else:
                    df.at[i, conc_col] = None

            # ---------- GAS ----------
            elif state == "(g)":
                P = ss.get("gas_pressure_kpa", 101.325)
                T = ss.get("gas_temp_K", 273.15)
                R = 8.314

                V = (n_target * R * T) / P
                df.at[i, vol_col] = V

                if V not in (None, 0):
                    df.at[i, conc_col] = n_target / V
                else:
                    df.at[i, conc_col] = None

    # --------- Refresh all widgets in this lane ---------
    for i in ss.reactants.index:
        refresh_row("r", i, lane)

    for i in ss.products.index:
        refresh_row("p", i, lane)

def cycle_anchor(side, idx):
    """
    Anchor behavior:

    Lane2 OFF:
        - Click always sets/moves Anchor 1 to this row

    Lane2 ON:
        - No anchors: click -> set Anchor 1
        - Only Anchor 1:
            * click same row -> remove Anchor 1
            * click different row -> set Anchor 2
        - Both anchors:
            * click row with no anchor -> move Anchor 1 to that row
            * click row with anchor -> do nothing
    """
    row = (side, idx)

    A1 = ss.anchor_1
    A2 = ss.anchor_2

    # -----------------------------
    # Lane 2 OFF: only Anchor 1
    # -----------------------------
    if not ss.show_lane2:
        ss.anchor_2 = None
        ss.anchor_1 = row

    # -----------------------------
    # Lane 2 ON
    # -----------------------------
    else:
        # --- No anchors ---
        if A1 is None and A2 is None:
            ss.anchor_1 = row

        # --- Only Anchor 1 exists ---
        elif A1 is not None and A2 is None:
            if A1 == row:
                # click same row -> unset Anchor 1
                ss.anchor_1 = None
            else:
                # click different row -> set Anchor 2
                ss.anchor_2 = row

        # --- Both anchors exist ---
        else:
            # Click on Anchor 2 row -> remove Anchor 2
            if row == A2:
                ss.anchor_2 = None

            # Click on a row with no anchor -> move Anchor 1 there
            elif row != A1:
                ss.anchor_1 = row

            # Click on Anchor 1 row -> do nothing (or unset if you want)
            else:
                pass

    # -----------------------------
    # Clear only stoichiometry markers (not values)
    # -----------------------------
    clear_stoichiometry_results()

    # -----------------------------
    # Re-propagate from existing anchors
    # -----------------------------
    if ss.anchor_1:
        s, i = ss.anchor_1
        propagate_from_anchor(s, i, 1)

    if ss.show_lane2 and ss.anchor_2:
        s, i = ss.anchor_2
        propagate_from_anchor(s, i, 2)

    maybe_evaluate_stoichiometry()

def lane_is_locked(side, idx, lane):
    """
    Returns True if this row/lane should be read-only due to anchor rules.
    """
    # Lane 1 controlled by anchor_1
    if lane == 1 and ss.anchor_1 is not None:
        return ss.anchor_1 != (side, idx)

    # Lane 2 controlled by anchor_2
    if lane == 2 and ss.anchor_2 is not None:
        return ss.anchor_2 != (side, idx)

    return False

def maybe_evaluate_stoichiometry():
    if ss.anchor_1 and ss.anchor_2:
        s1, i1 = ss.anchor_1
        s2, i2 = ss.anchor_2

        # Only valid if both anchors are reactants
        if s1 == "r" and s2 == "r":
            evaluate_stoichiometry()

def evaluate_stoichiometry():
    """
    Uses Anchor 1 (lane 1) and Anchor 2 (lane 2) to determine:

    - Limiting reagent
    - Excess mass for anchored reactants
    - Theoretical product yields (lane 1)
    - Percent yield (if MeasuredYieldMass is present)

    Assumes:
    - Equation is balanced
    - Both anchors exist
    - Anchors point to REACTANTS
    """

    # Must be balanced
    if not getattr(ss, "balanced", False):
        return

    # Need both anchors
    if not ss.anchor_1 or not ss.anchor_2:
        return

    side1, idx1 = ss.anchor_1
    side2, idx2 = ss.anchor_2

    # We only define limiting/excess from REACTANTS
    if side1 != "r" or side2 != "r":
        return

    reactants = ss.reactants
    products = ss.products

    try:
        r1 = reactants.loc[idx1]
        r2 = reactants.loc[idx2]
    except KeyError:
        return

    # --- pull moles & coeffs ---
    n1 = r1["Moles_1"]
    n2 = r2["Moles_2"]
    c1 = r1["Coeff"]
    c2 = r2["Coeff"]

    if n1 in (None, "", 0) or n2 in (None, "", 0):
        return

    # Reaction extent each anchor could support
    extent1 = n1 / c1
    extent2 = n2 / c2

    # Smaller extent is limiting
    if extent1 <= extent2:
        limiting_extent = extent1
        limiting_anchor = 1
        limiting_row = idx1
    else:
        limiting_extent = extent2
        limiting_anchor = 2
        limiting_row = idx2

    # -------------------------
    # RESET FLAGS
    # -------------------------
    reactants["IsLimiting_1"] = False
    reactants["ExcessMass"] = None

    # Mark limiting reactant
    reactants.at[limiting_row, "IsLimiting_1"] = True

    # -------------------------
    # EXCESS MASS (ONLY FOR ANCHORED REACTANTS)
    # -------------------------

    # Anchor 1
    M1 = r1["MolarMass"]
    required_n1 = limiting_extent * c1
    avail_n1 = n1
    excess_n1 = max(0.0, avail_n1 - required_n1)
    reactants.at[idx1, "ExcessMass"] = excess_n1 * M1 if excess_n1 > 0 else 0.0

    # Anchor 2
    M2 = r2["MolarMass"]
    required_n2 = limiting_extent * c2
    avail_n2 = n2
    excess_n2 = max(0.0, avail_n2 - required_n2)
    reactants.at[idx2, "ExcessMass"] = excess_n2 * M2 if excess_n2 > 0 else 0.0

    # Non-anchored reactants: clear excess
    for i in reactants.index:
        if i not in (idx1, idx2):
            reactants.at[i, "ExcessMass"] = None

    # -------------------------
    # PRODUCTS: THEORETICAL YIELD (LANE 1)
    # -------------------------
    for i, p in products.iterrows():
        coeff_p = p["Coeff"]
        M_p = p["MolarMass"]

        theo_moles = limiting_extent * coeff_p

        products.at[i, "Moles_1"] = theo_moles
        products.at[i, "Mass_1"] = theo_moles * M_p

    # -------------------------
    # PRODUCTS: PERCENT YIELD
    # -------------------------
    for i, p in products.iterrows():
        measured = p.get("MeasuredYieldMass", None)
        theoretical = p.get("Mass_1", None)

        if measured not in (None, "", 0) and theoretical not in (None, "", 0):
            percent = 100.0 * measured / theoretical
            products.at[i, "PercentYield"] = percent
        else:
            products.at[i, "PercentYield"] = None

    # -------------------------
    # STORE UI STATE
    # -------------------------
    ss.limiting_lane = limiting_anchor
    ss.limiting_row = limiting_row

def clear_stoichiometry_results():
    """
    Clear ONLY stoichiometry *markers/results* (limiting/excess/%yield),
    but DO NOT wipe user-entered or propagated Mass/Moles/Vol/Conc.
    """
    # Reactant flags
    if "IsLimiting_1" in ss.reactants.columns:
        ss.reactants["IsLimiting_1"] = False

    # Excess
    if "ExcessMass" in ss.reactants.columns:
        ss.reactants["ExcessMass"] = None

    # % yield only (do not clear MeasuredYieldMass)
    if "PercentYield" in ss.products.columns:
        ss.products["PercentYield"] = None

    # UI helpers
    ss.limiting_lane = None
    ss.limiting_row = None
