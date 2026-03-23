from __future__ import annotations

import string
import plotly.graph_objects as go
import streamlit as st
from streamlit import session_state as ss

from shared.vector_tools import (
    Vector2D,
    apply_operation,
    from_bearing,
    from_cardinal,
    from_components,
    from_magnitude_angle,
    parse_component_text,
    parse_ij_text,
    vector_summary,
    vector_to_latex,
)


# =========================================================
# label helpers
# =========================================================

def _next_default_label(n: int) -> str:
    letters = string.ascii_uppercase
    if n < 26:
        return letters[n]
    q, r = divmod(n, 26)
    return f"{letters[r]}{q}"


def _active_label_widget_key(key_prefix: str) -> str:
    version = ss.get(f"{key_prefix}_label_version", 0)
    return f"{key_prefix}_label_input_{version}"


def _current_label_value(key_prefix: str) -> str:
    wkey = _active_label_widget_key(key_prefix)
    return ss.get(wkey, ss.get(f"{key_prefix}_label_value", "A")).strip()


# =========================================================
# state init
# =========================================================

def _init_vector_state(key_prefix: str = "vec"):
    ss.setdefault(f"{key_prefix}_result", Vector2D(0.0, 0.0, label="result"))
    ss.setdefault(f"{key_prefix}_history", [])
    ss.setdefault(f"{key_prefix}_format", "Components")
    ss.setdefault(f"{key_prefix}_plot_mode", "Head-to-tail")

    # label state not tied directly to one persistent widget key
    ss.setdefault(f"{key_prefix}_label_version", 0)
    ss.setdefault(f"{key_prefix}_label_value", "A")

    # seed first widget instance
    first_key = _active_label_widget_key(key_prefix)
    ss.setdefault(first_key, "A")


# =========================================================
# label management
# =========================================================

def _advance_label_widget(key_prefix: str):
    """
    Move to a fresh text_input widget instance with the next default label.
    This avoids mutating an already-instantiated widget key.
    """
    history = ss.get(f"{key_prefix}_history", [])
    new_default = _next_default_label(len(history))
    ss[f"{key_prefix}_label_value"] = new_default
    ss[f"{key_prefix}_label_version"] = ss.get(f"{key_prefix}_label_version", 0) + 1
    new_key = _active_label_widget_key(key_prefix)
    ss[new_key] = new_default


def _sync_label_from_widget(key_prefix: str):
    ss[f"{key_prefix}_label_value"] = _current_label_value(key_prefix)


# =========================================================
# vector build/apply/reset/undo
# =========================================================

def _build_vector_from_ui(key_prefix: str) -> Vector2D:
    fmt = ss[f"{key_prefix}_format"]
    label = _current_label_value(key_prefix)

    if fmt == "Components":
        x = ss[f"{key_prefix}_x"]
        y = ss[f"{key_prefix}_y"]
        return from_components(x, y, label=label)

    if fmt == "Magnitude + Angle":
        mag = ss[f"{key_prefix}_mag"]
        ang = ss[f"{key_prefix}_ang"]
        return from_magnitude_angle(mag, ang, label=label)

    if fmt == "Bearing":
        mag = ss[f"{key_prefix}_bmag"]
        p = ss[f"{key_prefix}_primary"]
        ang = ss[f"{key_prefix}_bang"]
        s = ss[f"{key_prefix}_secondary"]
        return from_bearing(mag, p, ang, s, label=label)

    if fmt == "Cardinal":
        mag = ss[f"{key_prefix}_cmag"]
        d = ss[f"{key_prefix}_cdir"]
        return from_cardinal(mag, d, label=label)

    if fmt == "Typed Components":
        text = ss[f"{key_prefix}_typed_components"]
        return parse_component_text(text, label=label)

    if fmt == "i/j Notation":
        text = ss[f"{key_prefix}_typed_ij"]
        return parse_ij_text(text, label=label)

    raise ValueError("Unsupported vector format.")


def _reset_all_vectors(key_prefix: str):
    ss[f"{key_prefix}_result"] = Vector2D(0.0, 0.0, label="result")
    ss[f"{key_prefix}_history"] = []
    ss[f"{key_prefix}_label_version"] = ss.get(f"{key_prefix}_label_version", 0) + 1
    ss[f"{key_prefix}_label_value"] = "A"
    ss[_active_label_widget_key(key_prefix)] = "A"


def _undo_last_vector(key_prefix: str):
    hist = ss.get(f"{key_prefix}_history", [])
    if hist:
        last = hist.pop()
        ss[f"{key_prefix}_result"] = last["before"]
    _advance_label_widget(key_prefix)

def _apply_vector(key_prefix: str, operation: str):
    incoming = _build_vector_from_ui(key_prefix)
    old_result = ss[f"{key_prefix}_result"]
    new_result = apply_operation(old_result, incoming, operation)

    ss[f"{key_prefix}_history"].append(
        {
            "operation": operation,
            "incoming": incoming,
            "before": old_result,
            "after": new_result,
        }
    )
    ss[f"{key_prefix}_result"] = new_result
    _advance_label_widget(key_prefix)

# =========================================================
# plotting
# =========================================================
def _plot_vectors_2d(
    result: Vector2D,
    history: list[dict],
    plot_mode: str = "Head-to-tail",
    title: str = "2D Vector Plot",
):
    import math

    mode = (plot_mode or "Head-to-tail").strip().lower()

    # segments: (x0, y0, dx, dy, label, is_result, operation)
    segments = []
    points_x = [0.0, result.x]
    points_y = [0.0, result.y]

    if mode == "from origin":
        for item in history:
            v = item["incoming"]
            op = item.get("operation", "add")
            segments.append((0.0, 0.0, v.x, v.y, v.label or "v", False, op))
            points_x.extend([0.0, v.x])
            points_y.extend([0.0, v.y])

        segments.append((0.0, 0.0, result.x, result.y, "Result", True, "result"))
        points_x.extend([0.0, result.x])
        points_y.extend([0.0, result.y])

    else:
        cx, cy = 0.0, 0.0
        points_x.append(cx)
        points_y.append(cy)

        for item in history:
            v = item["incoming"]
            op = item.get("operation", "add")

            # For head-to-tail plotting, show the ACTUAL applied vector.
            if op == "subtract":
                plot_dx = -v.x
                plot_dy = -v.y
                plot_label = f"-{v.label}" if v.label else "−v"
            else:
                plot_dx = v.x
                plot_dy = v.y
                plot_label = v.label or "v"

            segments.append((cx, cy, plot_dx, plot_dy, plot_label, False, op))

            nx, ny = cx + plot_dx, cy + plot_dy
            points_x.extend([cx, nx])
            points_y.extend([cy, ny])
            cx, cy = nx, ny

        segments.append((0.0, 0.0, result.x, result.y, "Result", True, "result"))
        points_x.extend([0.0, result.x])
        points_y.extend([0.0, result.y])

    max_abs = max([1.0] + [abs(v) for v in points_x] + [abs(v) for v in points_y])
    lim = max_abs * 1.25

    fig = go.Figure()

    # axes
    fig.add_hline(y=0, line_width=1, line_color="rgba(0,120,255,0.8)")
    fig.add_vline(x=0, line_width=1, line_color="rgba(0,120,255,0.8)")

    placed_labels = []

    def _tip_label_position(x0, y0, dx, dy):
        tip_x = x0 + dx
        tip_y = y0 + dy

        mag = math.hypot(dx, dy)
        if mag < 1e-9:
            return tip_x, tip_y

        ux = dx / mag
        uy = dy / mag
        px = -uy
        py = ux

        forward = 0.03 * lim
        side = 0.04 * lim

        lx = tip_x + forward * ux + side * px
        ly = tip_y + forward * uy + side * py

        for ex, ey in placed_labels:
            dist = math.hypot(lx - ex, ly - ey)
            if dist < 0.12 * lim:
                lx += 0.05 * lim * px
                ly += 0.05 * lim * py

        placed_labels.append((lx, ly))
        return lx, ly

    def _result_label_position(x0, y0, dx, dy):
        mag = math.hypot(dx, dy)
        if mag < 1e-9:
            return x0 + 0.08 * lim, y0 - 0.08 * lim

        ux = dx / mag
        uy = dy / mag
        px = -uy
        py = ux

        # opposite side from the regular tip labels
        lx = x0 + 0.52 * dx - 0.07 * lim * px
        ly = y0 + 0.52 * dy - 0.07 * lim * py
        return lx, ly

    for x0, y0, dx, dy, lbl, is_result, operation in segments:
        x1 = x0 + dx
        y1 = y0 + dy
        mag = math.hypot(dx, dy)
        ang = (math.degrees(math.atan2(dy, dx)) + 360) % 360 if mag > 1e-9 else 0.0

        if is_result:
            color = "green"
            width = 4
            dash = "solid"
        elif operation == "subtract":
            color = "red"
            width = 2
            dash = "dash"
        elif operation == "set":
            color = "black"
            width = 2.5
            dash = "solid"
        else:  # add
            color = "black"
            width = 2
            dash = "solid"

        # main line
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                hovertemplate=(
                    f"<b>{lbl}</b><br>"
                    f"Operation: {operation.title()}<br>"
                    f"Start: ({x0:.3f}, {y0:.3f})<br>"
                    f"End: ({x1:.3f}, {y1:.3f})<br>"
                    f"Components: ({dx:.3f}, {dy:.3f})<br>"
                    f"Magnitude: {mag:.3f}<br>"
                    f"Direction: {ang:.3f}°"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

        # arrowhead as marker at tip
        angle_deg = math.degrees(math.atan2(dy, dx)) if mag > 1e-9 else 0.0
        fig.add_trace(
            go.Scatter(
                x=[x1],
                y=[y1],
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=14 if is_result else 12,
                    color=color,
                    angle=90 - angle_deg,
                ),
                hovertemplate=(
                    f"<b>{lbl}</b><br>"
                    f"Operation: {operation.title()}<br>"
                    f"Components: ({dx:.3f}, {dy:.3f})<br>"
                    f"Magnitude: {mag:.3f}<br>"
                    f"Direction: {ang:.3f}°"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

        # label
        if is_result:
            lx, ly = _result_label_position(x0, y0, dx, dy)
            label_text = "Result"
            label_color = "green"
            label_size = 13
        else:
            lx, ly = _tip_label_position(x0, y0, dx, dy)
            label_text = lbl
            label_color = "red" if operation == "subtract" else "black"
            label_size = 14

        fig.add_trace(
            go.Scatter(
                x=[lx],
                y=[ly],
                mode="text",
                text=[label_text],
                textfont=dict(color=label_color, size=label_size),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_xaxes(
        range=[-lim, lim],
        title="x",
        showgrid=True,
        zeroline=False,
        scaleanchor="y",
        scaleratio=1,
    )
    fig.update_yaxes(
        range=[-lim, lim],
        title="y",
        showgrid=True,
        zeroline=False,
    )

    fig.update_layout(
        title=title,
        height=700,
        margin=dict(l=30, r=30, t=60, b=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig

# =========================================================
# ui fragments
# =========================================================

def _render_current_result_tab(key_prefix: str):
    result: Vector2D = ss[f"{key_prefix}_result"]
    history: list[dict] = ss[f"{key_prefix}_history"]

    st.markdown("#### Current Result")
    st.latex(vector_to_latex(result))

    rs = vector_summary(result)
    m1, m2 = st.columns(2)
    m1.metric("Magnitude", rs["magnitude"])
    m2.metric("Direction", f"{rs['angle_deg']}°")
    st.caption(f"Bearing: {rs['bearing']}")

    b1, b2 = st.columns(2)
    with b1:
        st.button(
            "Reset Result",
            key=f"{key_prefix}_reset_btn",
            on_click=_reset_all_vectors,
            args=(key_prefix,),
            width="stretch",
        )
    with b2:
        st.button(
            "Undo Last",
            key=f"{key_prefix}_undo_btn",
            on_click=_undo_last_vector,
            args=(key_prefix,),
            width="stretch",
            disabled=(len(history) == 0),
        )


def _render_steps_tab(key_prefix: str):
    history: list[dict] = ss[f"{key_prefix}_history"]

    if not history:
        st.caption("No vectors applied yet.")
        return

    for idx, item in enumerate(history, start=1):
        inc = item["incoming"]
        aft = item["after"]
        inc_s = vector_summary(inc)
        aft_s = vector_summary(aft)

        with st.container(border=True):
            st.markdown(f"**Step {idx}:** {item['operation'].title()} {inc.label or 'vector'}")
            st.write(
                f"Incoming: ({inc_s['x']}, {inc_s['y']})  |  "
                f"Magnitude: {inc_s['magnitude']}  |  "
                f"Direction: {inc_s['angle_deg']}°"
            )
            st.write(
                f"Result: ({aft_s['x']}, {aft_s['y']})  |  "
                f"Magnitude: {aft_s['magnitude']}  |  "
                f"Direction: {aft_s['angle_deg']}°"
            )

def _render_action_button(icon: str, help_text: str, key: str, operation: str, key_prefix: str):
    if st.button(icon, key=key, help=help_text, width="stretch"):
        try:
            _apply_vector(key_prefix, operation)
            st.rerun()
        except Exception as e:
            st.error(str(e))

# =========================================================
# main render
# =========================================================

def render_vector_workbench(title: str = "Vector Workbench", key_prefix: str = "vec"):
    _init_vector_state(key_prefix)

    result: Vector2D = ss[f"{key_prefix}_result"]
    history: list[dict] = ss[f"{key_prefix}_history"]

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0.65rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"### {title}")

    left, right = st.columns([1.18, 1.62], gap="large")

    with left:
        tabs = st.tabs(["Current Result", "Vector Steps"])
        with tabs[0]:
            _render_current_result_tab(key_prefix)
        with tabs[1]:
            _render_steps_tab(key_prefix)

    with right:
        with st.container(border=True):
            # ----- row 1 -----
            top1, top2 = st.columns([1.45, 1.0], gap="medium")

            with top1:
                st.selectbox(
                    "Input Format",
                    [
                        "Components",
                        "Magnitude + Angle",
                        "Bearing",
                        "Cardinal",
                        "Typed Components",
                        "i/j Notation",
                    ],
                    key=f"{key_prefix}_format",
                    width="stretch",
                )

            with top2:
                st.selectbox(
                    "Plot Mode",
                    ["Head-to-tail", "From origin"],
                    key=f"{key_prefix}_plot_mode",
                    width="stretch",
                )

            # ----- row 2 -----
            fmt = ss[f"{key_prefix}_format"]

            if fmt == "Components":
                row2a, row2b, row2c = st.columns([0.8, 1.0, 1.0], gap="medium")
                with row2a:
                    st.text_input(
                        "Label",
                        key=_active_label_widget_key(key_prefix),
                        on_change=_sync_label_from_widget,
                        args=(key_prefix,),
                        width="stretch",
                    )
                with row2b:
                    st.number_input("x", key=f"{key_prefix}_x", value=0.0, width="stretch")
                with row2c:
                    st.number_input("y", key=f"{key_prefix}_y", value=0.0, width="stretch")

            elif fmt == "Magnitude + Angle":
                row2a, row2b, row2c = st.columns([0.8, 1.0, 1.0], gap="medium")
                with row2a:
                    st.text_input(
                        "Label",
                        key=_active_label_widget_key(key_prefix),
                        on_change=_sync_label_from_widget,
                        args=(key_prefix,),
                        width="stretch",
                    )
                with row2b:
                    st.number_input("Magnitude", key=f"{key_prefix}_mag", value=0.0, width="stretch")
                with row2c:
                    st.number_input("Angle (degrees)", key=f"{key_prefix}_ang", value=0.0, width="stretch")

            elif fmt == "Bearing":
                row2a, row2b, row2c, row2d, row2e = st.columns([0.75, 1.0, 0.8, 0.9, 0.9], gap="medium")
                with row2a:
                    st.text_input(
                        "Label",
                        key=_active_label_widget_key(key_prefix),
                        on_change=_sync_label_from_widget,
                        args=(key_prefix,),
                        width="stretch",
                    )
                with row2b:
                    st.number_input("Magnitude", key=f"{key_prefix}_bmag", value=0.0, width="stretch")
                with row2c:
                    st.selectbox("Start", ["N", "S", "E", "W"], key=f"{key_prefix}_primary", width="stretch")
                with row2d:
                    st.number_input(
                        "Angle",
                        key=f"{key_prefix}_bang",
                        value=0.0,
                        min_value=0.0,
                        max_value=90.0,
                        width="stretch",
                    )
                with row2e:
                    st.selectbox("Toward", ["E", "W", "N", "S"], key=f"{key_prefix}_secondary", width="stretch")

            elif fmt == "Cardinal":
                row2a, row2b, row2c = st.columns([0.8, 1.0, 1.0], gap="medium")
                with row2a:
                    st.text_input(
                        "Label",
                        key=_active_label_widget_key(key_prefix),
                        on_change=_sync_label_from_widget,
                        args=(key_prefix,),
                        width="stretch",
                    )
                with row2b:
                    st.number_input("Magnitude", key=f"{key_prefix}_cmag", value=0.0, width="stretch")
                with row2c:
                    st.selectbox("Direction", ["east", "west", "north", "south"], key=f"{key_prefix}_cdir",
                                 width="stretch")

            elif fmt == "Typed Components":
                row2a, row2b = st.columns([0.8, 2.2], gap="medium")
                with row2a:
                    st.text_input(
                        "Label",
                        key=_active_label_widget_key(key_prefix),
                        on_change=_sync_label_from_widget,
                        args=(key_prefix,),
                        width="stretch",
                    )
                with row2b:
                    st.text_input(
                        "Enter vector",
                        key=f"{key_prefix}_typed_components",
                        placeholder="Examples: (3,4), <3,4>, 3,4",
                        width="stretch",
                    )

            elif fmt == "i/j Notation":
                row2a, row2b = st.columns([0.8, 2.2], gap="medium")
                with row2a:
                    st.text_input(
                        "Label",
                        key=_active_label_widget_key(key_prefix),
                        on_change=_sync_label_from_widget,
                        args=(key_prefix,),
                        width="stretch",
                    )
                with row2b:
                    st.text_input(
                        "Enter vector",
                        key=f"{key_prefix}_typed_ij",
                        placeholder="Examples: 3i + 4j, -2i - 5j",
                        width="stretch",
                    )

            # ----- row 3 -----
            st.caption("Choose an action to apply the entered vector")

            a1, a2, a3 = st.columns([1, 1, 1], gap="medium")

            with a1:
                _render_action_button(
                    "📌",
                    "Set the current result equal to this vector",
                    key=f"{key_prefix}_set_btn",
                    operation="set",
                    key_prefix=key_prefix,
                )

            with a2:
                _render_action_button(
                    "➕",
                    "Add this vector to the current result",
                    key=f"{key_prefix}_add_btn",
                    operation="add",
                    key_prefix=key_prefix,
                )

            with a3:
                _render_action_button(
                    "➖",
                    "Subtract this vector from the current result",
                    key=f"{key_prefix}_sub_btn",
                    operation="subtract",
                    key_prefix=key_prefix,
                )

    st.markdown("#### Vector Plot")
    fig = _plot_vectors_2d(
        result=result,
        history=history,
        plot_mode=ss[f"{key_prefix}_plot_mode"],
        title="2D Vector Plot",
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
