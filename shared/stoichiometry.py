import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import html
from shared.Chemfxns import (
    create_reactants, create_products, create_elements, show_gas_config, evaluate_stoichiometry,
    cycle_anchor, clear_stoichiometry_results, lane_is_locked, clear_all_inputs, check_formula,
    balance_equation, compound_calc, summed_atoms, chem_format, display_side, cycle_phase,
    field_permissions, on_qty_change,
)

def compact_field(label: str, key: str, value, *,
                  on_change=None, args=None, disabled=False, tooltip=None, tooltip_color=None):

    # ------------------------------
    # Initialize widget state ONCE from DF
    # ------------------------------
    if key not in ss:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            ss[key] = ""
        else:
            ss[key] = str(value)

    # ------------------------------
    # Disabled behavior
    # ------------------------------
    placeholder = ""
    help_msg = None

    if disabled:
        help_msg = "🚫 Not editable for this state"
        if ss.get(key, "") == "":
            placeholder = "N/A"

    # ------------------------------
    # Render label
    # ------------------------------
    if tooltip:
        tip = html.escape(str(tooltip), quote=True)
        color = tooltip_color or "#2e7d32"
        st.markdown(
            f"""
            <div class='compact-label' style="display:flex; align-items:center; gap:6px;">
                <span>{label}</span>
                <span title="{tip}" style="cursor:help; font-size:12px; font-weight:700; color:{color};">ⓘ</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        extra_class = ""

        if ss.limiting_lane is not None:
            # key format: mass_r_0_1 → lane = last part
            try:
                lane = int(key.split("_")[-1])
                if lane == ss.limiting_lane:
                    extra_class = " limiting-label"
            except:
                pass

        st.markdown(
            f"<div class='compact-label{extra_class}'>{label}</div>",
            unsafe_allow_html=True
        )

    # ------------------------------
    # Render input (NO value= !)
    # ------------------------------
    st.text_input(
        " ",
        key=key,
        placeholder=placeholder,
        help=help_msg,
        label_visibility="collapsed",
        disabled=disabled,
        on_change=on_change,
        args=args
    )

def stoich_calculator():
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

    PHASES = ["(s)", "(aq)", "(g)", "(l)" ]

    # Create session state object if it doesn't exist'
    if "elements" not in ss:
        ss.elements = create_elements()

    if "reactants" not in ss:
        ss.reactants = create_reactants()

    if "products" not in ss:
        ss.products = create_products()

    if "mol_mass" not in ss:
        ss.mol_mass = None

    if "chem_formula" not in ss:
        ss.formula = ""
        ss.chem_formula = ""

    if "anchor_1" not in ss:
        ss.anchor_1 = None

    if "anchor_2" not in ss:
        ss.anchor_2 = None

    if "show_comp" not in ss:
        ss.show_comp = False

    if "comp_formula" not in ss:
        ss.comp_formula = None

    if "limiting_lane" not in ss:
        ss.limiting_lane = None
    if "limiting_row" not in ss:
        ss.limiting_row = None

    if "show_lane2" not in ss:
        ss.show_lane2 = False

    # Begin Streamlit app rendering
    st.markdown("""
    <style>
    
    /* -----------------------------------------
       GLOBAL APP SPACING FIX
       ----------------------------------------- */
    
    /* Reduce massive top padding */
    div.block-container {
        padding-top: 0.6rem !important;
    }
    
    /* Normalize default input box size globally */
    input {
        height: 24px !important;
        font-size: 13px !important;
        text-align: left !important;
    }
    
    
    /* -----------------------------------------
       INLINE COEFFICIENT INPUTS (keep your styling)
       ----------------------------------------- */
    input[aria-label="Coeff"] {
        max-width: 55px !important;
        min-width: 45px !important;
        text-align: center !important;
        padding: 2px 4px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 6px;
    }
    
    /* -----------------------------------------
       CHEM PANEL COMPACT GRID SCOPED CONTROL
       ----------------------------------------- */
    
    /* compact field label */
    .compact-label {
        font-size: 13px;
        margin: 0;
        padding: 0;
        line-height: 1.0;
    }
    
    /* tighten ONLY inside chem-grid-scope */
    .chem-grid-scope div[data-testid="stVerticalBlock"] > div {
        margin-top: 0px !important;
        margin-bottom: 2px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }
    
    /* tighten baseweb wrapper inside chem grid */
    .chem-grid-scope div[data-baseweb="input"] {
        margin-top: -2px !important;
        margin-bottom: -2px !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    
    /* Left-align text and placeholder inside Formula input */
    input[data-testid="stTextInput-input"] {
        text-align: left !important;
    }
    
    /* Important: Streamlit centers placeholder separately */
    input::placeholder {
        text-align: left !important;
    }
    
    /* Make Formula bar height consistent */
    div[data-testid="stTextInput"] {
        margin-top: 0px !important;
    }
    
    /* Keep its wrapper compact too */
    div[data-baseweb="input"] {
        align-items: center !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    
    /* Vertically align text nicely in formula box */
    div[data-testid="stTextInput"] input {
        padding-top: 2px !important;
        padding-bottom: 2px !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* ------------------ Anchors ------------------ */
    
    .anchor-primary {
        border: 2px solid #2563eb !important;
        background: #eff6ff !important;
        transition: 0.2s ease-in-out;
    }
    
    .anchor-secondary {
        border: 2px solid #16a34a !important;
        background: #ecfdf5 !important;
        transition: 0.2s ease-in-out;
    }
    
    /* ------------------ Labels ------------------ */
    
    .compact-label {
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 2px;
    }
    
    /* ------------------ Limiting Lane Label Highlight ------------------ */
    
    .limiting-label {
        background: #fee2e2;
        color: #7f1d1d;
        border-radius: 6px;
        padding: 2px 6px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    st.header('ChemTools')

    fc1,fc_spacer,fc_tools = st.columns([6, 1, 5])

    with fc1:
        formula = st.text_input("Formula",
                    key="chem_formula",
                    placeholder="Formula",
                    on_change=check_formula,
                    label_visibility="collapsed")

    with fc_tools:
        cols = st.columns([1, 1, 1, 1, 1, 1, 1], gap=None)
        with cols[0]:
            add_reactant = st.button("🧪", key="btn_react",
                                 help="Add Reactant")

        with cols[1]:
            add_product = st.button("🔥", key="btn_prod",
                                help="Add Product")

        with cols[2]:
            if st.button("⚖️", help="Balance Equation"):
                ok = balance_equation()
                if not ok:
                    st.toast("Could not balance this equation.")
                else:
                    st.success("Equation balanced!")
                    evaluate_stoichiometry()
                    st.rerun()
        with cols[3]:
            if st.button("🧹", key="btn_clear", help="Clear Entire Equation"):
                ss.reactants = ss.reactants.iloc[0:0]
                ss.products = ss.products.iloc[0:0]

        with cols[4]:
            if st.button("🧼", help="Clear all inputs"):
                clear_all_inputs()
                st.rerun()

        with cols[5]:
            if st.button("⚙️", key="btn_config", help="Gas Law Settings"):
                show_gas_config()
        with cols[6]:
            label = "②" if ss.show_lane2 else "①"
            help_txt = "Toggle Lane 2 (Scenario 2) for all cards"

            if st.button(label, key="btn_lane2", help=help_txt):
                ss.show_lane2 = not ss.show_lane2

                # If turning OFF lane2 → remove Anchor 2
                if not ss.show_lane2:
                    ss.anchor_2 = None

                    # Also clear any lane-2 results
                    clear_stoichiometry_results()

            # If turning ON lane2 → do nothing special, user will set anchor manually

    if ss.formula:
        if add_reactant:
            ok = compound_calc(ss.formula, 1)
            if not ok:
                st.toast("Invalid formula – please fix it before adding.", icon="⚠️")
                ss.formula = ""
                #st.rerun()
            else:
                ss.reactants.loc[len(ss.reactants)] = [
                    1,  # Coeff
                    ss.formula,  # Formula
                    round(ss.mol_mass, 3),  # MolarMass
                    None,  # Mass_1
                    None,  # Mass_2
                    None,  # Moles_1
                    None,  # Moles_2
                    None,  # ExcessMass
                    False,  # IsLimiting_1
                    None,  # Volume_1
                    None,  # Volume_2
                    None,  # Concentration_1
                    None,  # Concentration_2
                    "(s)",  # state
                ]
                ss.formula = ""

        if add_product:
            ok = compound_calc(ss.formula, 2)
            if not ok:
                st.toast("Invalid formula – please fix it before adding.", icon="⚠️")
                ss.formula = ""
            else:
                ss.products.loc[len(ss.products)] = [
                    1,
                    ss.formula,
                    round(ss.mol_mass, 3),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,  # Volume_1
                    None,  # Volume_2
                    None,  # Concentration_1
                    None,  # Concentration_2
                    "(s)",
                ]
                ss.formula = ""

    if not ss.reactants.empty or not ss.products.empty:

        left_atoms = summed_atoms(ss.reactants)
        right_atoms = summed_atoms(ss.products)

        if left_atoms is None or right_atoms is None:
            balanced = False
            invalid_equation = True
        else:
            balanced = (left_atoms == right_atoms)
            invalid_equation = False

        ss.balanced = balanced

        left = chem_format(display_side(ss.reactants))
        right = chem_format(display_side(ss.products))

        if invalid_equation:
            indicator = "<span style='color:orange; font-size:24px;'>⚠️ Invalid formula</span>"
            border_color = "orange"
        else:
            indicator = (
                "<span style='color:green; font-size:28px;'>&#10004;</span>"
                if balanced else
                "<span style='color:red; font-size:28px;'>&#10006;</span>"
            )
            border_color = "green" if balanced else "red"

        st.markdown(
            f"""
            <div style='border: 3px solid {border_color}; padding: 12px; border-radius: 12px;'>
                <h2 style='text-align:center;'>
                    {left}  &rarr;  {right} &nbsp; {indicator}
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---- Percent composition popup panel ----
        if ss.show_comp and ss.comp_formula:
            # Recompute composition for this one formula
            compound_calc(ss.comp_formula, 0)

            # Pull only real element rows (non-NaN % Mass)
            comp_df = ss.elements[
                ss.elements["% Mass"].notna()
            ][["Name", "Symbol", "% Mass"]].reset_index(drop=True)

            with st.container(border=True):
                st.markdown(
                    f"<div style='font-size:18px; font-weight:600;'>"
                    f"Percent composition for {chem_format(ss.comp_formula)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.table(comp_df)

                if st.button("Close", key="close_comp"):
                    ss.show_comp = False
                    ss.comp_formula = None
                    st.rerun()

        rcol, pcol = st.columns(2)

        # -----------------------------------------------------------
        # REACTANTS
        # -----------------------------------------------------------
        with rcol:
            st.subheader("🧪 Reactants")

            for idx, r in ss.reactants.iterrows():

                # ---------- HEADER ----------
                c1, c2 = st.columns([6, 5])

                with c1:
                    card_class = ""

                    if getattr(ss, "limiting_row", None) == idx:
                        card_class = "limiting-reagent"

                    st.markdown(f"""
                    <div class="{card_class}" style="
                        font-size:22px; padding:6px 10px; border-radius:8px;
                        background:#eaf3ff; color:#0b4db8;">
                        <strong>{chem_format(r['Formula'])}</strong>
                        <span style="font-size:16px; margin-left:6px;">{r['State']}</span>
                        <div style="font-size:14px;">M = {r['MolarMass']} g/mol</div>
                    </div>
                    """, unsafe_allow_html=True)

                with c2:
                    bc1, bc2, bc3, bc4 = st.columns(4, gap=None)

                    with bc1:
                        if st.button("📊", key=f"comp_r_{idx}"):
                            ss.comp_formula = r["Formula"]
                            ss.show_comp = True
                            st.rerun()

                    with bc2:
                        if st.button("🗑️", key=f"del_r_{idx}"):
                            ss.reactants = ss.reactants.drop(idx).reset_index(drop=True)
                            evaluate_stoichiometry()
                            st.rerun()

                    with bc3:
                        if st.button(STATE_ICONS.get(r["State"], "🧱"), key=f"state_r_{idx}"):
                            cycle_phase(ss.reactants, idx)
                            st.rerun()
                    with bc4:
                        if ss.anchor_1 == ("r", idx):
                            label = "🔵"
                        elif ss.anchor_2 == ("r", idx):
                            label = "🟢"
                        else:
                            label = "⚪"

                        if st.button(label, key=f"anchor_r_{idx}", help="Cycle anchor"):
                            cycle_anchor("r", idx)
                            st.rerun()

                lane1_class = "limiting-lane-soft" if ss.limiting_lane == 1 else ""
                lane2_class = "limiting-lane-soft" if ss.limiting_lane == 2 else ""

                st.markdown(f'<div class="chem-grid-scope {lane1_class}">', unsafe_allow_html=True)

                allow_vol, allow_conc, vol_unit = field_permissions(r["State"])

                # 🔒 Lane locking
                locked1 = lane_is_locked("r", idx, 1)
                locked2 = lane_is_locked("r", idx, 2)

                c1, c2, c3, c4, c5 = st.columns(5)

                # -------- Lane 1 --------

                with c1:
                    compact_field(
                        "Mass 1 (g)",
                        f"mass_r_{idx}_1",
                        r["Mass_1"],
                        on_change=on_qty_change,
                        args=("r", idx, 1, "mass"),
                        disabled=locked1
                    )

                with c2:
                    compact_field(
                        "Moles 1",
                        f"mol_r_{idx}_1",
                        r["Moles_1"],
                        on_change=on_qty_change,
                        args=("r", idx, 1, "moles"),
                        disabled=locked1
                    )

                with c3:
                    compact_field(
                        f"Vol 1 ({vol_unit})",
                        f"vol_r_{idx}_1",
                        r["Volume_1"],
                        on_change=on_qty_change,
                        args=("r", idx, 1, "vol"),
                        disabled=locked1 or (not allow_vol)
                    )

                with c4:
                    compact_field(
                        "Conc 1 (M)",
                        f"conc_r_{idx}_1",
                        r["Concentration_1"],
                        on_change=on_qty_change,
                        args=("r", idx, 1, "conc"),
                        disabled=locked1 or (not allow_conc)
                    )

                with c5:
                    compact_field("Excess", f"exc_r_{idx}", r["ExcessMass"], disabled=True)

                # -------- Lane 2 --------
                if ss.show_lane2:

                    lane2_class = "limiting-lane" if ss.limiting_lane == 2 else ""

                    st.markdown(f'<div class="chem-grid-scope {lane2_class}">', unsafe_allow_html=True)

                    d1, d2, d3, d4 = st.columns(4)

                    with d1:
                        compact_field(
                            "Mass 2 (g)",
                            f"mass_r_{idx}_2",
                            r["Mass_2"],
                            on_change=on_qty_change,
                            args=("r", idx, 2, "mass"),
                            disabled=locked2
                        )

                    with d2:
                        compact_field(
                            "Moles 2",
                            f"mol_r_{idx}_2",
                            r["Moles_2"],
                            on_change=on_qty_change,
                            args=("r", idx, 2, "moles"),
                            disabled=locked2
                        )

                    with d3:
                        compact_field(
                            f"Vol 2 ({vol_unit})",
                            f"vol_r_{idx}_2",
                            r["Volume_2"],
                            on_change=on_qty_change,
                            args=("r", idx, 2, "vol"),
                            disabled=locked2 or (not allow_vol)
                        )

                    with d4:
                        compact_field(
                            "Conc 2 (M)",
                            f"conc_r_{idx}_2",
                            r["Concentration_2"],
                            on_change=on_qty_change,
                            args=("r", idx, 2, "conc"),
                            disabled=locked2 or (not allow_conc)
                        )

                st.markdown("</div>", unsafe_allow_html=True)

        # PRODUCTS
        # -----------------------------------------------------------
        with pcol:
            st.subheader("🔥 Products")

            for idx, r in ss.products.iterrows():

                # ---------- HEADER ----------
                c1, c2 = st.columns([6, 5])

                with c1:
                    card_class = ""

                    st.markdown(f"""
                    <div class="{card_class}" style="
                        font-size:22px; padding:6px 10px; border-radius:8px;
                        background:#eaf3ff; color:#0b4db8;">
                        <strong>{chem_format(r['Formula'])}</strong>
                        <span style="font-size:16px; margin-left:6px;">{r['State']}</span>
                        <div style="font-size:14px;">M = {r['MolarMass']} g/mol</div>
                    </div>
                    """, unsafe_allow_html=True)

                with c2:

                    bc1, bc2, bc3, bc4 = st.columns(4, gap=None)

                    with bc1:
                        if st.button("📊", key=f"comp_p_{idx}"):
                            ss.comp_formula = r["Formula"]
                            ss.show_comp = True
                            st.rerun()

                    with bc2:
                        if st.button("🗑️", key=f"del_p_{idx}"):
                            ss.products = ss.products.drop(idx).reset_index(drop=True)
                            evaluate_stoichiometry()
                            st.rerun()

                    with bc3:
                        if st.button(STATE_ICONS.get(r["State"], "🧱"),
                                     key=f"state_p_{idx}"):
                            cycle_phase(ss.products, idx)
                            st.rerun()
                    with bc4:
                        if ss.anchor_1 == ("p", idx):
                            label = "🔵"
                        elif ss.anchor_2 == ("p", idx):
                            label = "🟢"
                        else:
                            label = "⚪"

                        if st.button(label, key=f"anchor_p_{idx}", help="Cycle anchor"):
                            cycle_anchor("p", idx)
                            st.rerun()

                lane1_class = "limiting-lane-soft" if ss.limiting_lane == 1 else ""
                lane2_class = "limiting-lane-soft" if ss.limiting_lane == 2 else ""

                st.markdown(f'<div class="chem-grid-scope {lane1_class}">', unsafe_allow_html=True)

                allow_vol, allow_conc, vol_unit = field_permissions(r["State"])

                # 🔒 Lane locking
                locked1 = lane_is_locked("p", idx, 1)
                locked2 = lane_is_locked("p", idx, 2)

                c1, c2, c3, c4, c5 = st.columns(5)

                # -------- Lane 1 --------

                with c1:
                    compact_field(
                        "Mass 1 (g)",
                        f"mass_p_{idx}_1",
                        r["Mass_1"],
                        on_change=on_qty_change,
                        args=("p", idx, 1, "mass"),
                        disabled=locked1
                    )

                with c2:
                    compact_field(
                        "Moles 1",
                        f"mol_p_{idx}_1",
                        r["Moles_1"],
                        on_change=on_qty_change,
                        args=("p", idx, 1, "moles"),
                        disabled=locked1
                    )

                with c3:
                    compact_field(
                        f"Vol 1 ({vol_unit})",
                        f"vol_p_{idx}_1",
                        r["Volume_1"],
                        on_change=on_qty_change,
                        args=("p", idx, 1, "vol"),
                        disabled=locked1 or (not allow_vol)
                    )

                with c4:
                    compact_field(
                        "Conc 1 (M)",
                        f"conc_p_{idx}_1",
                        r["Concentration_1"],
                        on_change=on_qty_change,
                        args=("p", idx, 1, "conc"),
                        disabled=locked1 or (not allow_conc)
                    )
                with c5:
                    compact_field(
                        "Meas. Yield (g)",
                        f"yield_p_{idx}",
                        r["MeasuredYieldMass"],
                        on_change=on_qty_change,
                        args=("p", idx, 1, "yield"),
                    )

                # -------- Lane 2 --------
                if ss.show_lane2:
                    lane2_class = "limiting-lane" if ss.limiting_lane == 2 else ""

                    st.markdown(f'<div class="chem-grid-scope {lane2_class}">', unsafe_allow_html=True)

                    d1, d2, d3, d4, d5 = st.columns(5)

                    with d1:
                        compact_field(
                            "Mass 2 (g)",
                            f"mass_p_{idx}_2",
                            r["Mass_2"],
                            on_change=on_qty_change,
                            args=("p", idx, 2, "mass"),
                            disabled=locked2
                        )

                    with d2:
                        compact_field(
                            "Moles 2",
                            f"mol_p_{idx}_2",
                            r["Moles_2"],
                            on_change=on_qty_change,
                            args=("p", idx, 2, "moles"),
                            disabled=locked2
                        )

                    with d3:
                        compact_field(
                            f"Vol 2 ({vol_unit})",
                            f"vol_p_{idx}_2",
                            r["Volume_2"],
                            on_change=on_qty_change,
                            args=("p", idx, 2, "vol"),
                            disabled=locked2 or (not allow_vol)
                        )

                    with d4:
                        compact_field(
                            "Conc 2 (M)",
                            f"conc_p_{idx}_2",
                            r["Concentration_2"],
                            on_change=on_qty_change,
                            args=("p", idx, 2, "conc"),
                            disabled=locked2 or (not allow_conc)
                        )

                    with d5:
                        pct = r.get("PercentYield")

                        display = ""
                        if pct not in (None, "", 0) and not (isinstance(pct, float) and pd.isna(pct)):
                            display = f"{pct:.1f}"

                        compact_field(
                            "% Yield",
                            f"yield_pct_p_{idx}",
                            display,
                            disabled=True
                        )

                st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.write("Nothing here yet")

