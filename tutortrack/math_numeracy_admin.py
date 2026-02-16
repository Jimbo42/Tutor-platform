# tutortrack/math_numeracy_admin.py
import json
from pathlib import Path
import streamlit as st
from streamlit import session_state as ss
import secrets
import copy

from shared.numeracy_dsl import load_game, save_game, build_question

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAME_PATH = PROJECT_ROOT / "shared" / "numeracy_game.json"

# ----------------------------
# Helpers: tiny “DSL builders”
# ----------------------------

SUPPORTED_CHOICE_MODES = ["computed_items", "two_decimal_one_distractor"]
SUPPORTED_ANSWER_MODES = ["argmax", "argmin", "choice_is_exact"]
SUPPORTED_VAR_KINDS = ["int", "choice"]

def _new_qid() -> str:
    return "q_" + secrets.token_hex(4)  # 8 hex chars

def _ensure_titles(game: dict) -> bool:
    """One-time migration: add title if missing. Returns True if changed."""
    changed = False
    for q in game.get("questions", []):
        if "title" not in q or not str(q.get("title", "")).strip():
            q["title"] = (q.get("prompt") or q.get("id") or "Untitled")[:60]
            changed = True
    return changed

def _default_qdef(qid: str) -> dict:
    return {
        "id": qid,
        "title": "Untitled question",
        "enabled": True,
        "selection": {"weight": 1, "cooldown": 0, "max_per_round": 999},
        "vars": {},
        "derived": {},
        "constraints": [],
        "prompt": "TODO prompt",
        "choices": {"mode": "computed_items", "items": []},
        "answer": {"mode": "argmax", "field": "value", "tie_break": "random"},
        "explain": ""
    }

def _validate_qdef(game: dict, qdef: dict, n: int = 3) -> tuple[bool, str]:
    """Try building a few samples; return (ok, message)."""
    try:
        for _ in range(n):
            _ = build_question(game, qdef)
        return True, "OK"
    except Exception as e:
        return False, str(e)


def _as_int_list(csv_text: str) -> list[int]:
    out = []
    for part in csv_text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out

def _as_float_list(csv_text: str) -> list[float]:
    out = []
    for part in csv_text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    return out

def _ref(name: str) -> str:
    name = name.strip()
    return name if name.startswith("$") else f"${name}"

def _parse_literal(s: str):
    s = s.strip()
    if s == "":
        raise ValueError("Empty literal")
    # int / float / bool / string fallback
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s  # allow plain strings as literals when needed

def _arg_from_mode(mode: str, var_name: str, literal_text: str):
    if mode == "var":
        return _ref(var_name)
    return _parse_literal(literal_text)

def _is_group_constraint(c: dict) -> bool:
    return isinstance(c, dict) and c.get("op") in ("and", "or") and isinstance(c.get("args"), list)

def _cmp_ops(game: dict) -> list[str]:
    # Only allow comparison ops that produce booleans
    limits = (game or {}).get("limits", {}) or {}
    logic = list(limits.get("logic_ops", []))
    # Keep only comparisons; always include ne
    allowed = [x for x in logic if x in ("lt", "gt", "eq")]
    if "ne" not in allowed:
        allowed.append("ne")
    return allowed

def _dict_replace_inplace(dst: dict, src: dict) -> None:
    dst.clear()
    dst.update(src)

def _close_editor():
    ss.nr_edit_qid = None
    ss.nr_edit_open = False

def _choice_count_for_qdef(qdef: dict) -> int:
    ch = (qdef or {}).get("choices", {}) or {}
    mode = ch.get("mode", "computed_items")
    if mode == "computed_items":
        return len(ch.get("items", []) or [])
    if mode == "two_decimal_one_distractor":
        return 1 + int(ch.get("n_distractors", 1))
    return 0

def _default_explain_for_qdef(qdef: dict) -> str:
    n = _choice_count_for_qdef(qdef)
    if n <= 0:
        return ""
    parts = [f"{{{{choice:{i}.label}}}} = {{{{choice:{i}.value}}}}" for i in range(n)]
    return " ; ".join(parts)

def _append_text(existing: str, add: str) -> str:
    if not existing:
        return add
    # add a space if needed
    if existing.endswith((" ", "\n")):
        return existing + add
    return existing + " " + add


# ----------------------------
# Dialogs
# ----------------------------
@st.dialog("Edit Question Type")
def dlg_edit_question(game: dict, qdef: dict):
    """
    Single editor dialog.
    Exit ONLY via Save & Exit or Cancel Changes.
    Internal reruns are allowed; numeracy_admin_app() will re-open this dialog
    while ss.nr_edit_open is True.
    """
    qid = qdef["id"]
    ns = f"qedit_{qid}"

    # Snapshot original state (first time dialog opens for this qid)
    snap_key = f"{ns}_snapshot"
    if snap_key not in ss:
        ss[snap_key] = copy.deepcopy(qdef)

    def _dirty():
        ss.nr_dirty = True
        ss.nr_edit_open = True
        ss.nr_edit_qid = qid

    def _close():
        ss.nr_edit_open = False
        ss.nr_edit_qid = None

    def _restore_snapshot():
        qdef.clear()
        qdef.update(copy.deepcopy(ss[snap_key]))


    # ---------------- Header (ONLY exits) ----------------
    h1, h2, h3 = st.columns([3, 1.4, 1.4], vertical_alignment="center")
    with h1:
        st.markdown(f"### {qdef.get('title','Untitled')}  \n`{qid}`")

    with h2:
        if st.button("✅ Save & Exit", key=f"{ns}_save_exit", type="primary"):
            save_game(GAME_PATH, ss.nr_game)  # save immediately
            ss.nr_dirty = False
            ss.pop(snap_key, None)
            _close()
            st.rerun()

    with h3:
        if st.button("❌ Cancel Changes", key=f"{ns}_cancel"):
            _restore_snapshot()
            ss.nr_dirty = False
            ss.pop(snap_key, None)
            _close()
            st.rerun()

    st.divider()

    # Ensure required keys exist
    qdef.setdefault("title", "Untitled question")
    qdef.setdefault("enabled", True)
    qdef.setdefault("selection", {"weight": 1, "cooldown": 0, "max_per_round": 999})
    qdef.setdefault("vars", {})
    qdef.setdefault("derived", {})
    qdef.setdefault("constraints", [])
    qdef.setdefault("choices", {"mode": "computed_items", "items": []})
    qdef.setdefault("answer", {"mode": "argmax", "field": "value", "tie_break": "random"})
    qdef.setdefault("prompt", "")
    qdef.setdefault("explain", "")

    # Keep DSL context updated for builders
    game["_all_var_names"] = list(set(qdef["vars"].keys()) | set(qdef["derived"].keys()))

    tabs = st.tabs(["Basics", "Vars", "Derived", "Choices", "Answer", "Constraints", "Preview"])

    # ---------------- BASICS ----------------
    with tabs[0]:
        qdef["title"] = st.text_input("Title", qdef["title"], key=f"{ns}_title")
        qdef["enabled"] = st.checkbox("Enabled", qdef["enabled"], key=f"{ns}_enabled")

        sel = qdef["selection"]
        sel["weight"] = float(st.number_input("Weight", 0.0, 100.0, float(sel.get("weight", 1.0)), key=f"{ns}_weight"))
        sel["cooldown"] = int(st.number_input("Cooldown", 0, 20, int(sel.get("cooldown", 0)), key=f"{ns}_cooldown"))
        sel["max_per_round"] = int(st.number_input("Max per round", 1, 9999, int(sel.get("max_per_round", 999)), key=f"{ns}_mpr"))

        st.subheader("Prompt")
        qdef["prompt"] = st.text_area(
            "prompt",
            value=qdef.get("prompt", ""),
            height=90,
            key=f"{ns}_prompt"
        )

        st.subheader("Explain")
        # main explain editor
        qdef["explain"] = st.text_area(
            "explain (optional)",
            value=qdef.get("explain", ""),
            height=110,
            key=f"{ns}_explain"
        )

        # ---- Explain Builder ----
        with st.expander("🧩 Explain builder", expanded=True):
            c1, c2 = st.columns([1, 1], vertical_alignment="center")

            with c1:
                if st.button("✨ Generate default explain", key=f"{ns}_explain_gen", type="primary"):
                    qdef["explain"] = _default_explain_for_qdef(qdef)
                    _dirty()
                    st.rerun()

            with c2:
                st.caption("Builds: choice labels + values for however many choices the question currently generates.")

            st.divider()

            # Choice tokens (based on detected count)
            n_choices = _choice_count_for_qdef(qdef)
            if n_choices > 0:
                st.markdown("**Choice tokens**")
                tok_cols = st.columns(min(4, n_choices))
                for i in range(n_choices):
                    with tok_cols[i % len(tok_cols)]:
                        if st.button(f"➕ choice:{i}.label", key=f"{ns}_tok_cl_{i}"):
                            qdef["explain"] = _append_text(qdef.get("explain", ""), f"{{{{choice:{i}.label}}}}")
                            _dirty()
                            st.rerun()
                        if st.button(f"➕ choice:{i}.value", key=f"{ns}_tok_cv_{i}"):
                            qdef["explain"] = _append_text(qdef.get("explain", ""), f"{{{{choice:{i}.value}}}}")
                            _dirty()
                            st.rerun()
            else:
                st.info("No choice tokens available yet (define Choices first).")

            st.divider()

            # Vars + Derived tokens
            st.markdown("**Vars / Derived tokens**")
            names = sorted(set((qdef.get("vars") or {}).keys()) | set((qdef.get("derived") or {}).keys()))
            if names:
                pick = st.selectbox("Pick a name", names, key=f"{ns}_explain_pick_name")
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("➕ Insert {{expr:$name}}", key=f"{ns}_tok_expr"):
                        qdef["explain"] = _append_text(qdef.get("explain", ""), f"{{{{expr:${pick}}}}}")
                        _dirty()
                        st.rerun()
                with b2:
                    places = st.number_input("fmt places", 0, 6, 3, 1, key=f"{ns}_tok_fmt_places")
                    if st.button("➕ Insert {{fmt:$name,places}}", key=f"{ns}_tok_fmt"):
                        qdef["explain"] = _append_text(qdef.get("explain", ""), f"{{{{fmt:${pick},{int(places)}}}}}")
                        _dirty()
                        st.rerun()
            else:
                st.info("No vars/derived defined yet.")

        st.caption("Tokens supported: {{expr:$a}}, {{fmt:$x,3}}, {{choice:0.label}}, {{choice:0.value}}")

        _dirty()

    # ---------------- VARS ----------------
    with tabs[1]:
        vars_dict = qdef["vars"]

        st.subheader("Add var")
        new_name = st.text_input("New var name", "", key=f"{ns}_v_new").strip()
        kind = st.selectbox("Type", SUPPORTED_VAR_KINDS, key=f"{ns}_v_kind")

        if kind == "int":
            lo = st.number_input("min", -999, 999, 0, key=f"{ns}_v_lo")
            hi = st.number_input("max", -999, 999, 10, key=f"{ns}_v_hi")
            if st.button("Add var", key=f"{ns}_v_add_int"):
                if not new_name:
                    st.error("Name required.")
                elif new_name in vars_dict:
                    st.error("Var already exists.")
                else:
                    vars_dict[new_name] = {"kind": "int", "min": int(lo), "max": int(hi)}
                    game["_all_var_names"] = list(set(qdef["vars"].keys()) | set(qdef["derived"].keys()))
                    _dirty()
                    st.rerun()
        else:
            vals = st.text_input("values CSV", "1,2,3", key=f"{ns}_v_vals")
            if st.button("Add var", key=f"{ns}_v_add_choice"):
                if not new_name:
                    st.error("Name required.")
                elif new_name in vars_dict:
                    st.error("Var already exists.")
                else:
                    vars_dict[new_name] = {"kind": "choice", "values": _as_float_list(vals)}
                    game["_all_var_names"] = list(set(qdef["vars"].keys()) | set(qdef["derived"].keys()))
                    _dirty()
                    st.rerun()

        st.divider()
        st.subheader("Edit / delete var")

        if not vars_dict:
            st.info("No vars yet.")
        else:
            pick = st.selectbox("Pick var", sorted(vars_dict.keys()), key=f"{ns}_v_pick")
            spec = vars_dict[pick]

            # key includes `pick` so changing selection refreshes fields
            if spec.get("kind") == "int":
                lo2 = st.number_input("min", -999, 999, int(spec.get("min", 0)), key=f"{ns}_v_e_lo_{pick}")
                hi2 = st.number_input("max", -999, 999, int(spec.get("max", 10)), key=f"{ns}_v_e_hi_{pick}")
                if st.button("Apply", key=f"{ns}_v_apply_{pick}"):
                    vars_dict[pick] = {"kind": "int", "min": int(lo2), "max": int(hi2)}
                    _dirty()
                    st.rerun()
            else:
                csv = ",".join(str(v) for v in spec.get("values", []))
                csv2 = st.text_input("values CSV", csv, key=f"{ns}_v_e_vals_{pick}")
                if st.button("Apply", key=f"{ns}_v_apply2_{pick}"):
                    vars_dict[pick] = {"kind": "choice", "values": _as_float_list(csv2)}
                    _dirty()
                    st.rerun()

            if st.button("🗑️ Delete var", key=f"{ns}_v_del_{pick}"):
                del vars_dict[pick]
                game["_all_var_names"] = list(set(qdef["vars"].keys()) | set(qdef["derived"].keys()))
                _dirty()
                st.rerun()

    # ---------------- DERIVED ----------------
    with tabs[2]:
        derived = qdef["derived"]

        st.subheader("Add derived")
        d_name = st.text_input("Derived name", "", key=f"{ns}_d_new").strip()
        mode = st.radio("Mode", ["sampler", "expression"], horizontal=True, key=f"{ns}_d_mode")

        if mode == "expression":
            expr = render_expr_builder_inline(
                game,
                namespace=f"{ns}_d_add_expr",
                initial={"op": "add", "args": ["$a", "$b"]},
                title="Derived expression",
            )
            if st.button("Add derived", key=f"{ns}_d_add_expr_btn"):
                if not d_name:
                    st.error("Name required.")
                elif d_name in derived:
                    st.error("Derived already exists.")
                elif expr is None:
                    st.error("Expression invalid.")
                else:
                    derived[d_name] = expr
                    game["_all_var_names"] = list(set(qdef["vars"].keys()) | set(qdef["derived"].keys()))
                    _dirty()
                    st.rerun()
        else:
            vals = st.text_input("values CSV", "1,2,3", key=f"{ns}_d_vals")
            if st.button("Add derived", key=f"{ns}_d_add_sampler_btn"):
                if not d_name:
                    st.error("Name required.")
                elif d_name in derived:
                    st.error("Derived already exists.")
                else:
                    derived[d_name] = {"kind": "choice", "values": _as_float_list(vals)}
                    game["_all_var_names"] = list(set(qdef["vars"].keys()) | set(qdef["derived"].keys()))
                    _dirty()
                    st.rerun()

        st.divider()
        st.subheader("Edit / delete derived")

        if not derived:
            st.info("No derived yet.")
        else:
            dpick = st.selectbox("Pick derived", sorted(derived.keys()), key=f"{ns}_d_pick")
            cur = derived[dpick]

            # keys include dpick so selection refresh works
            if isinstance(cur, dict) and "kind" in cur:
                csv = ",".join(str(v) for v in cur.get("values", []))
                csv2 = st.text_input("values CSV", csv, key=f"{ns}_d_e_vals_{dpick}")
                if st.button("Apply", key=f"{ns}_d_apply_{dpick}"):
                    derived[dpick] = {"kind": "choice", "values": _as_float_list(csv2)}
                    _dirty()
                    st.rerun()
            else:
                expr2 = render_expr_builder_inline(
                    game,
                    namespace=f"{ns}_d_edit_expr_{dpick}",
                    initial=cur if isinstance(cur, dict) else {"op": "add", "args": ["$a", "$b"]},
                    title=f"Expression for {dpick}",
                )
                if st.button("Apply", key=f"{ns}_d_apply_expr_btn_{dpick}"):
                    if expr2 is None:
                        st.error("Expression invalid.")
                    else:
                        derived[dpick] = expr2
                        _dirty()
                        st.rerun()

            if st.button("🗑️ Delete derived", key=f"{ns}_d_del_{dpick}"):
                del derived[dpick]
                game["_all_var_names"] = list(set(qdef["vars"].keys()) | set(qdef["derived"].keys()))
                _dirty()
                st.rerun()

    # ---------------- CHOICES ----------------
    with tabs[3]:
        ch = qdef["choices"]
        mode = st.selectbox(
            "choices.mode",
            SUPPORTED_CHOICE_MODES,
            index=SUPPORTED_CHOICE_MODES.index(ch.get("mode", "computed_items")),
            key=f"{ns}_c_mode",
        )

        # keep name context current
        names = sorted(set((qdef.get("vars") or {}).keys()) | set((qdef.get("derived") or {}).keys()))
        ref_options = [""] + [f"${n}" for n in names]

        if mode == "computed_items":
            items = ch.get("items", []) or []

            # -------- Presets --------
            with st.expander("✨ Presets", expanded=True):
                # Detect frac_1, frac_2, ... pattern
                frac_names = [n for n in names if n.startswith("frac_")]
                frac_names = sorted(frac_names,
                                    key=lambda s: int(s.split("_")[1]) if s.split("_")[1].isdigit() else 999)

                if frac_names:
                    st.caption("Detected derived fractions: " + ", ".join(frac_names))
                    if st.button("Build choices from frac_1..", key=f"{ns}_c_preset_fracs", type="primary"):
                        new_items = []
                        for i, fn in enumerate(frac_names, start=1):
                            num = f"$num_{i}"
                            den = f"$denom_{i}"
                            new_items.append(
                                {"label": f"{{{{expr:{num}}}}}/{{{{expr:{den}}}}}", "value": f"${fn}"}
                            )
                        qdef["choices"] = {"mode": "computed_items", "items": new_items}
                        _dirty()
                        st.rerun()
                else:
                    st.info("No frac_1 / frac_2 / ... derived values detected.")

            st.divider()

            # -------- Items editor --------
            for idx, it in enumerate(items):
                with st.expander(f"Item {idx + 1}", expanded=(idx == 0)):
                    it["label"] = st.text_input(
                        "label",
                        value=it.get("label", chr(ord("A") + idx)),
                        key=f"{ns}_c_lab_{idx}",
                    )

                    # value editor: prefer $ref dropdown, with optional raw override
                    val = it.get("value", "")
                    if isinstance(val, dict):
                        # advanced JSON dict already
                        st.code(val, language="json")
                        raw = st.text_area("value (JSON)", json.dumps(val, indent=2), key=f"{ns}_c_valjson_{idx}",
                                           height=120)
                        if st.button("Apply JSON", key=f"{ns}_c_applyjson_{idx}"):
                            try:
                                it["value"] = json.loads(raw)
                                _dirty()
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                    else:
                        # treat as string
                        use_ref = st.checkbox("Use $ref", value=str(val).strip().startswith("$"),
                                              key=f"{ns}_c_useref_{idx}")
                        if use_ref:
                            it["value"] = st.selectbox(
                                "value ($ref)",
                                ref_options,
                                index=ref_options.index(str(val)) if str(val) in ref_options else 0,
                                key=f"{ns}_c_ref_{idx}",
                            )
                        else:
                            it["value"] = st.text_input(
                                "value (raw)",
                                value=str(val),
                                key=f"{ns}_c_raw_{idx}",
                                help="Use $name for refs. Advanced: paste JSON to create expressions."
                            )

                    if st.button("Delete item", key=f"{ns}_c_del_{idx}", type="secondary"):
                        items.pop(idx)
                        qdef["choices"] = {"mode": "computed_items", "items": items}
                        _dirty()
                        st.rerun()

            if st.button("➕ Add item", key=f"{ns}_c_add"):
                items.append({"label": chr(ord("A") + len(items)), "value": ""})
                qdef["choices"] = {"mode": "computed_items", "items": items}
                _dirty()
                st.rerun()

            qdef["choices"] = {"mode": "computed_items", "items": items}

        else:
            # keep your existing two_decimal_one_distractor UI (with n_distractors)
            exact = st.text_input("exact ($ref)", value=ch.get("exact", "$exact"), key=f"{ns}_c_exact")
            delta = st.text_input("delta ($ref)", value=ch.get("delta", "$delta"), key=f"{ns}_c_delta")
            places = st.number_input("round_places", 0, 6, int(ch.get("round_places", 3)), 1, key=f"{ns}_c_places")

            n_dist = st.number_input(
                "Number of distractors",
                1, 6,
                int(ch.get("n_distractors", 1)),
                1,
                key=f"{ns}_c_ndist",
            )
            clamp_01 = st.checkbox("Clamp to 0..1", value=bool(ch.get("clamp_01", True)), key=f"{ns}_c_clamp01")

            qdef["choices"] = {
                "mode": "two_decimal_one_distractor",
                "exact": exact.strip(),
                "delta": delta.strip(),
                "round_places": int(places),
                "n_distractors": int(n_dist),
                "clamp_01": bool(clamp_01),
            }

        _dirty()

    # ---------------- ANSWERS ----------------
    with tabs[4]:
        ans = qdef["answer"]
        mode = st.selectbox(
            "answer.mode",
            SUPPORTED_ANSWER_MODES,
            index=SUPPORTED_ANSWER_MODES.index(ans.get("mode", "argmax")),
            key=f"{ns}_a_mode",
        )

        if mode in ("argmax", "argmin"):
            tie = st.selectbox(
                "tie_break",
                ["random", "first"],
                index=0 if ans.get("tie_break", "random") == "random" else 1,
                key=f"{ns}_a_tie",
            )
            qdef["answer"] = {"mode": mode, "field": "value", "tie_break": tie}
        else:
            qdef["answer"] = {"mode": "choice_is_exact"}

    # ---------------- CONTRAINTS ----------------
    with tabs[5]:
        st.caption("Constraints must evaluate True. Use comparisons or AND/OR groups. No nested dialogs.")

        existing = qdef.setdefault("constraints", [])
        cmp_ops = _cmp_ops(game)

        # Ensure var context is current for builders
        game["_all_var_names"] = list(set(qdef.get("vars", {}).keys()) | set(qdef.get("derived", {}).keys()))

        # -------------------------
        # Add new constraint
        # -------------------------
        with st.expander("➕ Add constraint", expanded=True):
            add_kind = st.radio(
                "Add as",
                ["comparison", "and/or group"],
                horizontal=True,
                key=f"{ns}_k_add_kind",
            )

            if add_kind == "comparison":
                # Build a single comparison expr (lt/gt/eq/ne)
                st.caption("Build a single comparison like $a < $b")
                expr = render_expr_builder_inline(
                    game,
                    namespace=f"{ns}_k_add_cmp",
                    initial={"op": "lt", "args": ["$a", "$b"]},
                    title="Comparison",
                    allow_math_ops=False,
                    allow_logic_ops=True,
                    max_args=2,
                )

                if st.button("Add comparison", key=f"{ns}_k_add_cmp_btn", type="primary"):
                    if not expr:
                        st.error("Fix the expression first.")
                    elif expr.get("op") not in cmp_ops:
                        st.error(f"Constraint op must be one of {cmp_ops}")
                    elif len(expr.get("args", [])) != 2:
                        st.error("Comparison must have exactly 2 args.")
                    else:
                        existing.append(expr)
                        st.rerun()

            else:
                group_op = st.selectbox("Group operator", ["and", "or"], index=0, key=f"{ns}_k_add_gop")
                n = st.number_input("How many comparisons?", 2, 6, 2, 1, key=f"{ns}_k_add_n")
                comps = []
                ok = True

                st.caption("Build each comparison (lt/gt/eq/ne). These become the args of AND/OR.")

                for i in range(int(n)):
                    c = render_expr_builder_inline(
                        game,
                        namespace=f"{ns}_k_add_gcmp_{i}",
                        initial={"op": "lt", "args": ["$a", "$b"]},
                        title=f"Comparison {i + 1}",
                        allow_math_ops=False,
                        allow_logic_ops=True,
                        max_args=2,
                    )
                    if not c:
                        ok = False
                    else:
                        if c.get("op") not in cmp_ops or len(c.get("args", [])) != 2:
                            ok = False
                            st.warning(f"Comparison {i + 1} must be op in {cmp_ops} with exactly 2 args.")
                        comps.append(c)

                group = {"op": group_op, "args": comps} if ok else None
                if group:
                    st.code(group, language="json")

                if st.button("Add group", key=f"{ns}_k_add_group_btn", type="primary", disabled=(group is None)):
                    existing.append(group)
                    st.rerun()

            with st.expander("Advanced (paste constraint JSON)", expanded=False):
                raw = st.text_area(
                    "constraint JSON",
                    value='{"op":"lt","args":["$a","$b"]}',
                    height=110,
                    key=f"{ns}_k_add_raw",
                )
                if st.button("Add from JSON", key=f"{ns}_k_add_raw_btn"):
                    try:
                        existing.append(json.loads(raw))
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        st.divider()

        # -------------------------
        # Edit existing constraint
        # -------------------------
        if not existing:
            st.info("No constraints yet.")
        else:
            idx = st.selectbox(
                "Edit existing constraint",
                list(range(len(existing))),
                format_func=lambda i: f"{i + 1}: {existing[i].get('op', '?')}",
                key=f"{ns}_k_edit_pick",
            )

            cur = existing[idx]

            if _is_group_constraint(cur):
                st.subheader("Edit group constraint")
                group_op = st.selectbox(
                    "Group operator",
                    ["and", "or"],
                    index=0 if cur.get("op") == "and" else 1,
                    key=f"{ns}_k_edit_gop",
                )

                args = cur.get("args", []) or []
                n = st.number_input(
                    "Comparisons in group",
                    2,
                    6,
                    value=max(2, min(6, len(args) or 2)),
                    step=1,
                    key=f"{ns}_k_edit_gn",
                )

                # Normalize length
                norm = list(args)[: int(n)]
                while len(norm) < int(n):
                    norm.append({"op": "lt", "args": ["$a", "$b"]})

                new_comps = []
                ok = True
                for i in range(int(n)):
                    c = render_expr_builder_inline(
                        game,
                        namespace=f"{ns}_k_edit_gcmp_{i}",
                        initial=norm[i],
                        title=f"Comparison {i + 1}",
                        allow_math_ops=False,
                        allow_logic_ops=True,
                        max_args=2,
                    )
                    if not c:
                        ok = False
                    else:
                        if c.get("op") not in cmp_ops or len(c.get("args", [])) != 2:
                            ok = False
                            st.warning(f"Comparison {i + 1} must be op in {cmp_ops} with exactly 2 args.")
                        new_comps.append(c)

                new_group = {"op": group_op, "args": new_comps} if ok else None
                if new_group:
                    st.code(new_group, language="json")

                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    if st.button("Save group", key=f"{ns}_k_save_group", type="primary", disabled=(new_group is None)):
                        existing[idx] = new_group
                        st.rerun()
                with c2:
                    if st.button("Delete", key=f"{ns}_k_del_{idx}", type="secondary"):
                        existing.pop(idx)
                        st.rerun()
                with c3:
                    st.caption("")

            else:
                st.subheader("Edit comparison constraint")
                expr2 = render_expr_builder_inline(
                    game,
                    namespace=f"{ns}_k_edit_cmp",
                    initial=cur if isinstance(cur, dict) else {"op": "lt", "args": ["$a", "$b"]},
                    title="Comparison",
                    allow_math_ops=False,
                    allow_logic_ops=True,
                    max_args=2,
                )

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Save", key=f"{ns}_k_save_cmp", type="primary"):
                        if not expr2:
                            st.error("Fix the expression first.")
                        elif expr2.get("op") not in cmp_ops or len(expr2.get("args", [])) != 2:
                            st.error(f"Comparison must be op in {cmp_ops} with exactly 2 args.")
                        else:
                            existing[idx] = expr2
                            st.rerun()
                with c2:
                    if st.button("Delete", key=f"{ns}_k_del_cmp", type="secondary"):
                        existing.pop(idx)
                        st.rerun()

        st.divider()
        if existing and st.button("🗑️ Clear all constraints", key=f"{ns}_k_clear_all"):
            qdef["constraints"] = []
            st.rerun()

    # ---------------- PREVIEW ----------------
    with tabs[6]:
        ok, msg = _validate_qdef(game, qdef)
        st.success("Valid") if ok else st.error(msg)

        if st.button("Generate 3 samples", key=f"{ns}_prev_btn"):
            for _ in range(3):
                b = build_question(game, qdef)
                st.write(b.prompt, b.choices)

@st.dialog("Preview Question")
def dlg_preview_question(game: dict, qdef: dict):
    qid = qdef["id"]
    ns = f"prev_{qid}"
    st.subheader(qdef.get("title", qid))

    for k in range(5):
        b = build_question(game, qdef)
        st.markdown(f"**{b.prompt}**")
        st.write(b.choices)
        st.caption(f"Correct index: {b.correct_index} • Explain: {b.explain}")
        st.divider()

    if st.button("Close", key=f"{ns}_close"):
        st.rerun()

def render_expr_builder_inline(
    game: dict,
    *,
    namespace: str,
    initial: dict | None = None,
    title: str = "Expression Builder",
    allow_logic_ops: bool = True,
    allow_math_ops: bool = True,
    max_args: int = 6,
) -> dict | None:
    """
    Inline (non-dialog) expression builder.

    Produces an expression dict:
        {"op": "<allowed op>", "args": [<arg1>, <arg2>, ...]}

    Args:
      game: numeracy game dict (expects game["limits"]["math_ops"] and ["logic_ops"])
      namespace: unique string to prevent Streamlit key collisions
      initial: existing expr dict to preload (or None)
      title: header text
      allow_logic_ops / allow_math_ops: restrict operator groups
      max_args: max args to allow

    Returns:
      expr dict if current UI state is valid and an operator is selected, else None.
      (No side effects: caller decides when to "Save" into qdef.)
    """
    import streamlit as st

    limits = (game or {}).get("limits", {}) or {}
    math_ops = list(limits.get("math_ops", [])) if allow_math_ops else []
    logic_ops = list(limits.get("logic_ops", [])) if allow_logic_ops else []

    # Optional extras present in your DSL
    extras = ["ne"]  # DSL supports ne as special-case
    allowed_ops = ["(pick)"] + math_ops + logic_ops + extras

    st.subheader(title)

    init_op = (initial or {}).get("op", "(pick)")
    init_args = (initial or {}).get("args", [])

    op = st.selectbox(
        "Operator",
        allowed_ops,
        index=allowed_ops.index(init_op) if init_op in allowed_ops else 0,
        key=f"{namespace}_op",
    )

    st.caption("Args can be variables ($a) or literals (e.g., 5, -2, 0.25, true).")

    # Suggest arg count
    default_n = 1 if op in ("not",) else 2
    start_n = max(default_n, len(init_args) or default_n)
    n_args = st.number_input(
        "Number of args",
        1,
        max_args,
        value=int(start_n),
        step=1,
        key=f"{namespace}_nargs",
    )

    # Build variable list from current qdef context (caller should set game["_all_var_names"])
    var_names = [""] + sorted(list((game.get("_all_var_names") or [])))

    args = []
    args_ok = True

    for i in range(int(n_args)):
        col1, col2, col3 = st.columns([1.2, 1.4, 1.6])

        # Preload defaults per-arg
        init_is_var = i < len(init_args) and isinstance(init_args[i], str) and str(init_args[i]).startswith("$")
        init_var = ""
        init_lit = ""
        if i < len(init_args):
            if init_is_var:
                init_var = str(init_args[i])[1:]
            else:
                init_lit = str(init_args[i])

        with col1:
            mode = st.radio(
                f"Arg {i+1} type",
                ["var", "literal"],
                horizontal=True,
                index=0 if init_is_var else 1,
                key=f"{namespace}_mode_{i}",
            )
        with col2:
            var_name = st.selectbox(
                f"Var {i+1}",
                var_names,
                index=var_names.index(init_var) if init_var in var_names else 0,
                key=f"{namespace}_var_{i}",
                disabled=(mode != "var"),
            )
        with col3:
            lit = st.text_input(
                f"Literal {i+1}",
                value=init_lit,
                key=f"{namespace}_lit_{i}",
                disabled=(mode != "literal"),
            )

        try:
            args.append(_arg_from_mode(mode, var_name, lit))
        except Exception as e:
            args_ok = False
            st.warning(f"Arg {i+1}: {e}")

    expr = None
    if op != "(pick)" and args_ok:
        expr = {"op": op, "args": args}
        st.code(expr, language="json")
    elif op == "(pick)":
        st.info("Pick an operator to build an expression.")
    else:
        st.info("Fix arg warnings to build a valid expression.")

    return expr

def numeracy_admin_app():
    st.title("🛠️ Numeracy Question Builder")

    # ---- Load once ----
    if "nr_game" not in ss:
        ss.nr_game = load_game(GAME_PATH)
        ss.nr_dirty = False
        ss.nr_edit_qid = None
        ss.nr_edit_open = False
        ss.nr_preview_qid = None
        ss.nr_preview_open = False

        # one-time migration
        if _ensure_titles(ss.nr_game):
            save_game(GAME_PATH, ss.nr_game)

    game = ss.nr_game
    qlist = game.setdefault("questions", [])

    # ---- Header actions ----
    h1, h2 = st.columns([3, 1], vertical_alignment="center")
    with h2:
        if st.button("➕ New", key="nr_new", use_container_width=True):
            qid = _new_qid()
            q = _default_qdef(qid)
            qlist.append(q)

            ss.nr_edit_qid = qid
            ss.nr_edit_open = True
            ss.nr_preview_open = False
            st.rerun()

    if ss.get("nr_dirty"):
        st.warning("Unsaved changes (save via Save & Exit in the editor).", icon="⚠️")

    st.divider()

    # ---- Question list ----
    for q in sorted(qlist, key=lambda x: (x.get("title", ""), x.get("id", ""))):
        qid = q["id"]
        title = q.get("title", qid)

        with st.container(border=True):
            left, right = st.columns([4, 2], vertical_alignment="center")
            with left:
                st.markdown(f"**{title}**")
                st.caption(f"id: {qid}")

            with right:
                c1, c2, c3, c4 = st.columns(4)

                # IMPORTANT: do NOT call dialogs here. Set flags + rerun.
                with c1:
                    if st.button("✏️", key=f"edit_{qid}", help="Edit"):
                        ss.nr_edit_qid = qid
                        ss.nr_edit_open = True
                        ss.nr_preview_open = False
                        st.rerun()

                with c2:
                    if st.button("👁️", key=f"prev_{qid}", help="Preview"):
                        ss.nr_preview_qid = qid
                        ss.nr_preview_open = True
                        ss.nr_edit_open = False
                        st.rerun()

                with c3:
                    if st.button("⧉", key=f"dup_{qid}", help="Duplicate"):
                        q2 = copy.deepcopy(q)
                        q2["id"] = _new_qid()
                        q2["title"] = f"Copy of {title}"
                        qlist.append(q2)

                        ss.nr_edit_qid = q2["id"]
                        ss.nr_edit_open = True
                        ss.nr_preview_open = False
                        ss.nr_dirty = True
                        st.rerun()

                with c4:
                    if st.button("🗑️", key=f"del_{qid}", help="Delete"):
                        game["questions"] = [qq for qq in qlist if qq["id"] != qid]
                        save_game(GAME_PATH, game)
                        st.rerun()

    # ---- Open exactly ONE dialog per run (prevents DuplicateElementId) ----
    if ss.get("nr_edit_open") and ss.get("nr_edit_qid"):
        q = next((qq for qq in ss.nr_game.get("questions", []) if qq.get("id") == ss.nr_edit_qid), None)
        if q is not None:
            dlg_edit_question(ss.nr_game, q)

    elif ss.get("nr_preview_open") and ss.get("nr_preview_qid"):
        q = next((qq for qq in ss.nr_game.get("questions", []) if qq.get("id") == ss.nr_preview_qid), None)
        if q is not None:
            dlg_preview_question(ss.nr_game, q)
