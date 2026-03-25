from __future__ import annotations

import string
import math
import plotly.graph_objects as go
import streamlit as st
from streamlit import session_state as ss

from shared.vector_tools import (
    Vector2D,
    Vector3D,
    apply_operation,
    from_bearing,
    from_cardinal,
    from_components,
    from_magnitude_angle,
    from_magnitude_azimuth_elevation,
    math_angle_to_bearing,
    parse_component_text,
    parse_ijk_text,
    vector_summary,
    vector_to_latex,
    round_clean,
    from_direction_with_t,
    from_direction_with_t_3d,
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
    ss.setdefault(f"{key_prefix}_result", Vector3D(0.0, 0.0, 0.0, label="result"))
    ss.setdefault(f"{key_prefix}_dimension", "2D")
    ss.setdefault(f"{key_prefix}_history", [])
    ss.setdefault(f"{key_prefix}_format", "Components")
    ss.setdefault(f"{key_prefix}_plot_mode", "Head-to-tail")

    # left-panel mode ("tab")
    ss.setdefault(f"{key_prefix}_panel_mode", "Current Result")

    # selected vectors for dot/cross
    ss.setdefault(f"{key_prefix}_selected_steps", [])

    # label state not tied directly to one persistent widget key
    ss.setdefault(f"{key_prefix}_label_version", 0)
    ss.setdefault(f"{key_prefix}_label_value", "A")

    # seed first widget instance
    first_key = _active_label_widget_key(key_prefix)
    ss.setdefault(first_key, "A")

    ss.setdefault(f"{key_prefix}_calc_mode", None)
    ss.setdefault(f"{key_prefix}_calc_result_text", "")
    ss.setdefault(f"{key_prefix}_calc_result_kind", "")
    ss.setdefault(f"{key_prefix}_calc_v1", "")
    ss.setdefault(f"{key_prefix}_calc_v2", "")
    ss.setdefault(f"{key_prefix}_calc_plot_vector", None)
    ss.setdefault(f"{key_prefix}_calc_plot_label", "")
    ss.setdefault(f"{key_prefix}_calc_plot_enabled", False)
    ss.setdefault(f"{key_prefix}_plot_meta", None)

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
def _build_vector_from_ui(key_prefix: str) -> Vector3D:
    fmt = ss[f"{key_prefix}_format"]
    dimension = ss[f"{key_prefix}_dimension"]
    label = _current_label_value(key_prefix)

    if fmt == "Components":
        x = ss[f"{key_prefix}_x"]
        y = ss[f"{key_prefix}_y"]
        z = ss.get(f"{key_prefix}_z", 0.0) if dimension == "3D" else 0.0
        return from_components(x, y, z, label=label)

    if fmt == "Typed Components":
        text = ss[f"{key_prefix}_typed_components"]
        return parse_component_text(text, label=label, dimension=dimension)

    if fmt == "i/j Notation":
        text = ss[f"{key_prefix}_typed_ij"]
        return parse_ijk_text(text, label=label, dimension="2D")

    if fmt == "i/j/k Notation":
        text = ss[f"{key_prefix}_typed_ijk"]
        return parse_ijk_text(text, label=label, dimension="3D")

    if fmt == "Magnitude + Angle":
        mag = ss[f"{key_prefix}_mag"]
        ang = ss[f"{key_prefix}_ang"]
        return from_magnitude_angle(mag, ang, label=label)

    if fmt == "Magnitude + Azimuth + Elevation":
        mag = ss[f"{key_prefix}_mag3"]
        az = ss[f"{key_prefix}_az3"]
        el = ss[f"{key_prefix}_el3"]
        return from_magnitude_azimuth_elevation(mag, az, el, label=label)

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

    if fmt == "Start + t·Direction":
        t = ss.get(f"{key_prefix}_t_param", 1.0)

        if dimension == "3D":
            sx = ss.get(f"{key_prefix}_sx", 0.0)
            sy = ss.get(f"{key_prefix}_sy", 0.0)
            sz = ss.get(f"{key_prefix}_sz", 0.0)
            dx = ss.get(f"{key_prefix}_dx", 0.0)
            dy = ss.get(f"{key_prefix}_dy", 0.0)
            dz = ss.get(f"{key_prefix}_dz", 0.0)

            ss[f"{key_prefix}_plot_meta"] = {
                "plot_start": (sx, sy, sz),
                "plot_mode_override": "anchored",
            }

            return from_direction_with_t_3d(
                dx, dy, dz, t=t, label=label
            )

        sx = ss.get(f"{key_prefix}_sx", 0.0)
        sy = ss.get(f"{key_prefix}_sy", 0.0)
        dx = ss.get(f"{key_prefix}_dx", 0.0)
        dy = ss.get(f"{key_prefix}_dy", 0.0)

        ss[f"{key_prefix}_plot_meta"] = {
            "plot_start": (sx, sy, 0.0),
            "plot_mode_override": "anchored",
        }

        return from_direction_with_t(
            dx, dy, t=t, label=label
        )

    raise ValueError("Unsupported vector format.")

def _reset_all_vectors(key_prefix: str):
    ss[f"{key_prefix}_result"] = Vector3D(0.0, 0.0, 0.0, label="")
    ss[f"{key_prefix}_history"] = []

    ss[f"{key_prefix}_label_version"] = ss.get(f"{key_prefix}_label_version", 0) + 1
    ss[f"{key_prefix}_label_value"] = "A"
    ss[_active_label_widget_key(key_prefix)] = "A"

    ss[f"{key_prefix}_selected_steps"] = []

    ss[f"{key_prefix}_plot_meta"] = None

    ss[f"{key_prefix}_calc_mode"] = None
    ss[f"{key_prefix}_calc_result_text"] = ""
    ss[f"{key_prefix}_calc_result_kind"] = ""
    ss[f"{key_prefix}_calc_v1"] = ""
    ss[f"{key_prefix}_calc_v2"] = ""

    ss[f"{key_prefix}_calc_plot_vector"] = None
    ss[f"{key_prefix}_calc_plot_label"] = ""
    ss[f"{key_prefix}_calc_plot_enabled"] = False

    ss[f"{key_prefix}_t_param"] = 1.0

def _undo_last_vector(key_prefix: str):
    hist = ss.get(f"{key_prefix}_history", [])
    if hist:
        last = hist.pop()
        ss[f"{key_prefix}_result"] = last["before"]

    # trim invalid selections after undo
    valid_steps = set(range(1, len(hist) + 1))
    current = ss.get(f"{key_prefix}_selected_steps", [])
    ss[f"{key_prefix}_selected_steps"] = [i for i in current if i in valid_steps]

    _advance_label_widget(key_prefix)

def _apply_vector(key_prefix: str, operation: str):
    incoming = _build_vector_from_ui(key_prefix)
    old_result = ss[f"{key_prefix}_result"]
    new_result = apply_operation(old_result, incoming, operation)

    plot_meta = ss.get(f"{key_prefix}_plot_meta", None)

    entry = {
        "operation": operation,
        "incoming": incoming,
        "before": old_result,
        "after": new_result,
    }

    if plot_meta:
        entry.update(plot_meta)

    ss[f"{key_prefix}_history"].append(entry)
    ss[f"{key_prefix}_plot_meta"] = None
    ss[f"{key_prefix}_result"] = new_result

    _advance_label_widget(key_prefix)

    ss[f"{key_prefix}_calc_plot_vector"] = None
    ss[f"{key_prefix}_calc_plot_label"] = ""
    ss[f"{key_prefix}_calc_plot_enabled"] = False

def _available_vectors(key_prefix: str):
    """
    Returns a list of dicts:
      [{"name": "A", "vector": v, "step": 1}, ...]
    Uses the stored incoming vectors, not the running resultant.
    """
    out = []
    hist = ss.get(f"{key_prefix}_history", [])
    for idx, item in enumerate(hist, start=1):
        v = _to_vector3d(item["incoming"])
        label = (v.label or f"V{idx}").strip()
        out.append(
            {
                "name": label,
                "vector": v,
                "step": idx,
            }
        )
    return out


def _vector_display_name(item: dict) -> str:
    return f"{item['name']} (step {item['step']})"

def _toggle_step_selection(key_prefix: str, step_idx: int):
    selected = list(ss.get(f"{key_prefix}_selected_steps", []))

    if step_idx in selected:
        selected.remove(step_idx)
    else:
        if len(selected) >= 2:
            return
        selected.append(step_idx)

    selected.sort()
    ss[f"{key_prefix}_selected_steps"] = selected


def _selected_vectors(key_prefix: str):
    selected = ss.get(f"{key_prefix}_selected_steps", [])
    hist = ss.get(f"{key_prefix}_history", [])

    out = []
    for idx in selected:
        if 1 <= idx <= len(hist):
            item = hist[idx - 1]
            v = _to_vector3d(item["incoming"])
            out.append((idx, v.label or f"V{idx}", v))
    return out


def _run_selected_calc(key_prefix: str, mode: str):
    chosen = _selected_vectors(key_prefix)
    if len(chosen) != 2:
        raise ValueError("Select exactly two vectors first.")

    (_, name1, v1), (_, name2, v2) = chosen
    ss[f"{key_prefix}_calc_mode"] = mode
    _store_calc_result(key_prefix, mode, name1, v1, name2, v2)

def _store_calc_result(key_prefix: str, mode: str, name1: str, v1: Vector3D, name2: str, v2: Vector3D):

    dimension = ss[f"{key_prefix}_dimension"]

    if mode == "dot":
        dot_val = v1.dot(v2)
        mag1 = v1.magnitude
        mag2 = v2.magnitude

        if mag1 == 0 or mag2 == 0:
            angle_text = "undefined (zero vector)"
        else:
            cos_theta = dot_val / (mag1 * mag2)
            cos_theta = max(-1.0, min(1.0, cos_theta))  # clamp for floating-point safety
            theta_deg = math.degrees(math.acos(cos_theta))
            angle_text = f"{round_clean(theta_deg)}°"

        ss[f"{key_prefix}_calc_result_kind"] = "scalar"
        ss[f"{key_prefix}_calc_result_text"] = (
            f"{name1} · {name2} = {round_clean(dot_val)}"
            f"   |   angle = {angle_text}"
        )
        ss[f"{key_prefix}_calc_plot_vector"] = None
        ss[f"{key_prefix}_calc_plot_label"] = ""
        ss[f"{key_prefix}_calc_plot_enabled"] = False

        return

    if mode == "cross":
        if dimension == "2D":
            value = v1.x * v2.y - v1.y * v2.x
            ss[f"{key_prefix}_calc_result_kind"] = "scalar"
            ss[f"{key_prefix}_calc_result_text"] = (
                f"{name1} × {name2} = {round_clean(value)}  (z-component)"
            )
            ss[f"{key_prefix}_calc_plot_vector"] = None
            ss[f"{key_prefix}_calc_plot_label"] = ""
            ss[f"{key_prefix}_calc_plot_enabled"] = False
        else:
            value = v1.cross(v2)
            ss[f"{key_prefix}_calc_result_kind"] = "vector"
            ss[f"{key_prefix}_calc_result_text"] = (
                f"{name1} × {name2} = {vector_to_latex(value, dimension='3D')}"
            )

            # new: store for optional plotting
            ss[f"{key_prefix}_calc_plot_vector"] = value
            ss[f"{key_prefix}_calc_plot_label"] = f"{name1} × {name2}"
            ss[f"{key_prefix}_calc_plot_enabled"] = True
        return

def _prepare_calc(key_prefix: str, mode: str):
    vectors = _available_vectors(key_prefix)
    ss[f"{key_prefix}_calc_mode"] = mode

    if len(vectors) == 2:
        a, b = vectors[0], vectors[1]
        _store_calc_result(key_prefix, mode, a["name"], a["vector"], b["name"], b["vector"])
        return

    # more than two: seed selectors if blank
    options = [_vector_display_name(v) for v in vectors]
    if not ss.get(f"{key_prefix}_calc_v1"):
        ss[f"{key_prefix}_calc_v1"] = options[0]
    if not ss.get(f"{key_prefix}_calc_v2"):
        ss[f"{key_prefix}_calc_v2"] = options[1] if len(options) > 1 else options[0]


def _compute_selected_calc(key_prefix: str):
    mode = ss.get(f"{key_prefix}_calc_mode")
    if not mode:
        return

    vectors = _available_vectors(key_prefix)
    lookup = {_vector_display_name(v): v for v in vectors}

    pick1 = ss.get(f"{key_prefix}_calc_v1", "")
    pick2 = ss.get(f"{key_prefix}_calc_v2", "")

    if pick1 not in lookup or pick2 not in lookup:
        raise ValueError("Please select two valid vectors.")

    if pick1 == pick2:
        raise ValueError("Choose two different vectors.")

    a = lookup[pick1]
    b = lookup[pick2]
    _store_calc_result(key_prefix, mode, a["name"], a["vector"], b["name"], b["vector"])

def _to_vector3d(v):
    if isinstance(v, Vector3D):
        return v
    if isinstance(v, Vector2D):
        return Vector3D(v.x, v.y, 0.0, label=getattr(v, "label", ""))
    return Vector3D(
        float(getattr(v, "x", 0.0)),
        float(getattr(v, "y", 0.0)),
        float(getattr(v, "z", 0.0)),
        label=getattr(v, "label", ""),
    )


def _normalize_vector_state(key_prefix: str):
    ss[f"{key_prefix}_result"] = _to_vector3d(ss[f"{key_prefix}_result"])

    hist = ss.get(f"{key_prefix}_history", [])
    normalized = []
    for item in hist:
        normalized.append(
            {
                **item,
                "incoming": _to_vector3d(item["incoming"]),
                "before": _to_vector3d(item["before"]),
                "after": _to_vector3d(item["after"]),
            }
        )
    ss[f"{key_prefix}_history"] = normalized

def _should_plot_result(history: list[dict], plot_mode: str) -> bool:
    mode = (plot_mode or "Head-to-tail").strip().lower()
    if mode == "set point":
        mode = "from origin"
    # No history -> no result
    if not history:
        return False

    # A Result only makes sense once at least two vectors exist
    if len(history) < 2:
        return False

    # Only show the Result in head-to-tail mode
    if mode != "head-to-tail":
        return False

    return True

# =========================================================
# plotting
# =========================================================
def _plot_vectors_2d(
    result: Vector3D,
    history: list[dict],
    plot_mode: str = "Head-to-tail",
    title: str = "2D Vector Plot",
):
    mode = (plot_mode or "Head-to-tail").strip().lower()
    show_result = _should_plot_result(history, plot_mode)

    # segments: (x0, y0, dx, dy, label, is_result, operation)
    segments = []
    points_x = [0.0, result.x]
    points_y = [0.0, result.y]

    if mode == "from origin":
        for item in history:
            v = _to_vector3d(item["incoming"])
            op = item.get("operation", "add")
            anchored = item.get("plot_mode_override") == "anchored"

            if op == "subtract":
                dx, dy = -v.x, -v.y
                lbl = f"-{v.label}" if v.label else "−v"
            else:
                dx, dy = v.x, v.y
                lbl = v.label or "v"

            if anchored:
                ps = item.get("plot_start", (0.0, 0.0, 0.0))
                x0, y0 = ps[0], ps[1]
            else:
                x0, y0 = 0.0, 0.0

            segments.append((x0, y0, dx, dy, lbl, False, op))
            points_x.extend([x0, x0 + dx])
            points_y.extend([y0, y0 + dy])

        if show_result:
            segments.append((0.0, 0.0, result.x, result.y, "Result", True, "result"))
            points_x.extend([0.0, result.x])
            points_y.extend([0.0, result.y])

    else:
        cx, cy = 0.0, 0.0
        points_x.append(cx)
        points_y.append(cy)

        for item in history:
            v = _to_vector3d(item["incoming"])
            op = item.get("operation", "add")
            anchored = item.get("plot_mode_override") == "anchored"

            if op == "subtract":
                plot_dx = -v.x
                plot_dy = -v.y
                plot_label = f"-{v.label}" if v.label else "−v"
            else:
                plot_dx = v.x
                plot_dy = v.y
                plot_label = v.label or "v"

            if anchored:
                ps = item.get("plot_start", (0.0, 0.0, 0.0))
                x0, y0 = ps[0], ps[1]
                segments.append((x0, y0, plot_dx, plot_dy, plot_label, False, op))
                points_x.extend([x0, x0 + plot_dx])
                points_y.extend([y0, y0 + plot_dy])
            else:
                segments.append((cx, cy, plot_dx, plot_dy, plot_label, False, op))
                nx, ny = cx + plot_dx, cy + plot_dy
                points_x.extend([cx, nx])
                points_y.extend([cy, ny])
                cx, cy = nx, ny

        if show_result:
            segments.append((0.0, 0.0, result.x, result.y, "Result", True, "result"))
            points_x.extend([0.0, result.x])
            points_y.extend([0.0, result.y])

    max_abs = max([1.0] + [abs(v) for v in points_x] + [abs(v) for v in points_y])
    lim = max_abs * 1.25

    fig = go.Figure()

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

        lx = x0 + 0.52 * dx - 0.07 * lim * px
        ly = y0 + 0.52 * dy - 0.07 * lim * py
        return lx, ly

    for x0, y0, dx, dy, lbl, is_result, operation in segments:

        is_anchored = not is_result and (abs(x0) > 1e-12 or abs(y0) > 1e-12)

        if is_anchored:
            fig.add_trace(
                go.Scatter(
                    x=[x0],
                    y=[y0],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color="rgba(80,80,80,0.75)",
                        symbol="circle-open",
                        line=dict(width=2, color="rgba(80,80,80,0.75)"),
                    ),
                    hovertemplate=(
                        f"<b>Set point</b><br>"
                        f"Start: ({x0:.3f}, {y0:.3f})"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

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
        else:
            color = "black"
            width = 2
            dash = "solid"

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

def _add_3d_arrowhead(fig, x1, y1, z1, dx, dy, dz, color, sizeref):

    mag = math.sqrt(dx**2 + dy**2 + dz**2)
    if mag < 1e-9:
        return

    ux = dx / mag
    uy = dy / mag
    uz = dz / mag

    fig.add_trace(
        go.Cone(
            x=[x1],
            y=[y1],
            z=[z1],
            u=[ux],
            v=[uy],
            w=[uz],
            anchor="tip",
            showscale=False,
            sizemode="absolute",
            sizeref=sizeref,
            colorscale=[[0, color], [1, color]],
            hoverinfo="skip",
            name="",
        )
    )

def _plot_vectors_3d(
    result: Vector3D,
    history: list[dict],
    plot_mode: str = "Head-to-tail",
    title: str = "3D Vector Plot",
    calc_plot_vector: Vector3D | None = None,
    calc_plot_label: str = "",
):
    mode = (plot_mode or "Head-to-tail").strip().lower()
    show_result = _should_plot_result(history, plot_mode)

    # segments: (x0, y0, z0, dx, dy, dz, label, is_result, operation)
    segments = []
    pts_x = [0.0, result.x]
    pts_y = [0.0, result.y]
    pts_z = [0.0, result.z]

    if calc_plot_vector is not None:
        pts_x.extend([0.0, calc_plot_vector.x])
        pts_y.extend([0.0, calc_plot_vector.y])
        pts_z.extend([0.0, calc_plot_vector.z])

    if mode == "from origin":
        for item in history:
            v = _to_vector3d(item["incoming"])
            op = item.get("operation", "add")
            anchored = item.get("plot_mode_override") == "anchored"

            if op == "subtract":
                dx, dy, dz = -v.x, -v.y, -v.z
                lbl = f"-{v.label}" if v.label else "−v"
            else:
                dx, dy, dz = v.x, v.y, v.z
                lbl = v.label or "v"

            if anchored:
                ps = item.get("plot_start", (0.0, 0.0, 0.0))
                x0, y0, z0 = ps[0], ps[1], ps[2]
            else:
                x0, y0, z0 = 0.0, 0.0, 0.0

            segments.append((x0, y0, z0, dx, dy, dz, lbl, False, op))
            pts_x.extend([x0, x0 + dx])
            pts_y.extend([y0, y0 + dy])
            pts_z.extend([z0, z0 + dz])

        if show_result:
            segments.append((0.0, 0.0, 0.0, result.x, result.y, result.z, "Result", True, "result"))
            pts_x.extend([0.0, result.x])
            pts_y.extend([0.0, result.y])
            pts_z.extend([0.0, result.z])

    else:
        cx, cy, cz = 0.0, 0.0, 0.0

        for item in history:
            v = _to_vector3d(item["incoming"])
            op = item.get("operation", "add")
            anchored = item.get("plot_mode_override") == "anchored"

            if op == "subtract":
                pdx, pdy, pdz = -v.x, -v.y, -v.z
                plabel = f"-{v.label}" if v.label else "−v"
            else:
                pdx, pdy, pdz = v.x, v.y, v.z
                plabel = v.label or "v"

            if anchored:
                ps = item.get("plot_start", (0.0, 0.0, 0.0))
                x0, y0, z0 = ps[0], ps[1], ps[2]
                segments.append((x0, y0, z0, pdx, pdy, pdz, plabel, False, op))
                pts_x.extend([x0, x0 + pdx])
                pts_y.extend([y0, y0 + pdy])
                pts_z.extend([z0, z0 + pdz])
            else:
                segments.append((cx, cy, cz, pdx, pdy, pdz, plabel, False, op))
                nx, ny, nz = cx + pdx, cy + pdy, cz + pdz
                pts_x.extend([cx, nx])
                pts_y.extend([cy, ny])
                pts_z.extend([cz, nz])
                cx, cy, cz = nx, ny, nz

        if show_result:
            segments.append((0.0, 0.0, 0.0, result.x, result.y, result.z, "Result", True, "result"))
            pts_x.extend([0.0, result.x])
            pts_y.extend([0.0, result.y])
            pts_z.extend([0.0, result.z])

    lim = max(
        [1.0]
        + [abs(v) for v in pts_x]
        + [abs(v) for v in pts_y]
        + [abs(v) for v in pts_z]
    ) * 1.25

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=[-lim, lim], y=[0, 0], z=[0, 0],
        mode="lines", line=dict(color="rgba(0,120,255,0.6)", width=3),
        hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-lim, lim], z=[0, 0],
        mode="lines", line=dict(color="rgba(0,120,255,0.6)", width=3),
        hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-lim, lim],
        mode="lines", line=dict(color="rgba(0,120,255,0.6)", width=3),
        hoverinfo="skip", showlegend=False
    ))

    cone_size = max(0.6, lim * 0.06)

    for x0, y0, z0, dx, dy, dz, lbl, is_result, operation in segments:
        is_anchored = (
            not is_result and
            (abs(x0) > 1e-12 or abs(y0) > 1e-12 or abs(z0) > 1e-12)
        )

        if is_anchored:
            fig.add_trace(
                go.Scatter3d(
                    x=[x0],
                    y=[y0],
                    z=[z0],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color="rgba(80,80,80,0.8)",
                        symbol="circle-open",
                    ),
                    hovertemplate=(
                        f"<b>Set point</b><br>"
                        f"Start: ({x0:.3f}, {y0:.3f}, {z0:.3f})"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

        x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
        mag = math.sqrt(dx**2 + dy**2 + dz**2)

        if is_result:
            color = "green"
            width = 8
            label_text = "Result"
        elif operation == "subtract":
            color = "red"
            width = 5
            label_text = lbl
        else:
            color = "black"
            width = 5
            label_text = lbl

        fig.add_trace(
            go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode="lines",
                line=dict(color=color, width=width),
                hovertemplate=(
                    f"<b>{label_text}</b><br>"
                    f"Operation: {operation.title()}<br>"
                    f"Start: ({x0:.3f}, {y0:.3f}, {z0:.3f})<br>"
                    f"End: ({x1:.3f}, {y1:.3f}, {z1:.3f})<br>"
                    f"Components: ({dx:.3f}, {dy:.3f}, {dz:.3f})<br>"
                    f"Magnitude: {mag:.3f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

        _add_3d_arrowhead(fig, x1, y1, z1, dx, dy, dz, color=color, sizeref=cone_size)

        fig.add_trace(
            go.Scatter3d(
                x=[x1],
                y=[y1],
                z=[z1],
                mode="text",
                text=[label_text],
                textposition="top center",
                textfont=dict(color=color, size=12 if is_result else 11),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if calc_plot_vector is not None:
        vx, vy, vz = calc_plot_vector.x, calc_plot_vector.y, calc_plot_vector.z
        mag = math.sqrt(vx**2 + vy**2 + vz**2)

        fig.add_trace(
            go.Scatter3d(
                x=[0.0, vx],
                y=[0.0, vy],
                z=[0.0, vz],
                mode="lines",
                line=dict(color="purple", width=6),
                hovertemplate=(
                    f"<b>{calc_plot_label or 'Cross Product'}</b><br>"
                    f"Start: (0, 0, 0)<br>"
                    f"End: ({vx:.3f}, {vy:.3f}, {vz:.3f})<br>"
                    f"Components: ({vx:.3f}, {vy:.3f}, {vz:.3f})<br>"
                    f"Magnitude: {mag:.3f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

        _add_3d_arrowhead(
            fig,
            vx, vy, vz,
            vx, vy, vz,
            color="purple",
            sizeref=cone_size,
        )

        fig.add_trace(
            go.Scatter3d(
                x=[vx],
                y=[vy],
                z=[vz],
                mode="text",
                text=[calc_plot_label or "Cross"],
                textposition="top center",
                textfont=dict(color="purple", size=12),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        height=700,
        margin=dict(l=20, r=20, t=60, b=20),
        scene=dict(
            xaxis=dict(title="x", range=[-lim, lim]),
            yaxis=dict(title="y", range=[-lim, lim]),
            zaxis=dict(title="z", range=[-lim, lim]),
            aspectmode="cube",
        ),
    )

    return fig

# =========================================================
# ui fragments
# =========================================================
def _render_current_result_tab(key_prefix: str):
    result = ss[f"{key_prefix}_result"]
    history = ss[f"{key_prefix}_history"]
    dimension = ss[f"{key_prefix}_dimension"]

    plot_mode = ss.get(f"{key_prefix}_plot_mode", "Head-to-tail")
    history = ss.get(f"{key_prefix}_history", [])

    if plot_mode in {"From origin", "Set Point"} and history:
        last_item = history[-1]
        last_vec = _to_vector3d(last_item["incoming"])
        last_label = last_vec.label or "Vector"
        heading = f"#### Last Vector: {last_label}"
    elif plot_mode in {"From origin", "Set Point"}:
        heading = "#### Last Vector"
    else:
        heading = "#### Current Result"

    st.markdown(heading)

    st.latex(vector_to_latex(result, dimension=dimension))

    rs = vector_summary(result)

    if dimension == "3D":
        st.markdown(
            """
            <style>
            div[data-testid="stMetricValue"] {
                font-size: 0.95rem;
            }
            div[data-testid="stMetricLabel"] {
                font-size: 0.78rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Magnitude", rs["magnitude"])
        m2.metric("Azimuth", f"{rs['azimuth_deg']}°")
        m3.metric("Elevation", f"{rs['elevation_deg']}°")
        st.caption(f"Components: ({rs['x']}, {rs['y']}, {rs['z']})")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Magnitude", rs["magnitude"])
        m2.metric("Direction", f"{rs['azimuth_deg']}°")
        st.caption(f"Bearing: {math_angle_to_bearing(result.azimuth_deg)}")

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
    dimension = ss[f"{key_prefix}_dimension"]
    selected = ss.get(f"{key_prefix}_selected_steps", [])

    if not history:
        st.caption("No vectors applied yet.")
        return

    for idx, item in enumerate(history, start=1):
        inc = _to_vector3d(item["incoming"])
        aft = _to_vector3d(item["after"])
        inc_s = vector_summary(inc)
        aft_s = vector_summary(aft)

        label = inc.label or f"V{idx}"
        is_selected = idx in selected
        toggle_text = f"✅ {label}" if is_selected else label

        with st.container(border=True):
            top1, top2 = st.columns([4.5, 1.2], gap="small")
            with top1:
                st.markdown(f"**Step {idx}:** {item['operation'].title()}")
            with top2:
                st.button(
                    toggle_text,
                    key=f"{key_prefix}_step_toggle_{idx}",
                    width="stretch",
                    on_click=_toggle_step_selection,
                    args=(key_prefix, idx),
                    disabled=(not is_selected and len(selected) >= 2),
                    help="Select this vector for dot/cross product",
                )

            if dimension == "3D":
                st.write(
                    f"Incoming: ({inc_s['x']}, {inc_s['y']}, {inc_s['z']})  |  "
                    f"Magnitude: {inc_s['magnitude']}  |  "
                    f"Azimuth: {inc_s['azimuth_deg']}°  |  "
                    f"Elevation: {inc_s['elevation_deg']}°"
                )
                st.write(
                    f"Result: ({aft_s['x']}, {aft_s['y']}, {aft_s['z']})  |  "
                    f"Magnitude: {aft_s['magnitude']}  |  "
                    f"Azimuth: {aft_s['azimuth_deg']}°  |  "
                    f"Elevation: {aft_s['elevation_deg']}°"
                )
            else:
                st.write(
                    f"Incoming: ({inc_s['x']}, {inc_s['y']})  |  "
                    f"Magnitude: {inc_s['magnitude']}  |  "
                    f"Direction: {inc_s['azimuth_deg']}°"
                )
                st.write(
                    f"Result: ({aft_s['x']}, {aft_s['y']})  |  "
                    f"Magnitude: {aft_s['magnitude']}  |  "
                    f"Direction: {aft_s['azimuth_deg']}°"
                )

    chosen = _selected_vectors(key_prefix)
    if chosen:
        names = ", ".join(name for _, name, _ in chosen)
        st.caption(f"Selected: {names}")

def _render_action_button(icon: str, help_text: str, key: str, operation: str, key_prefix: str):
    if st.button(icon, key=key, help=help_text, width="stretch"):
        try:
            _apply_vector(key_prefix, operation)
            st.rerun()
        except Exception as e:
            st.error(str(e))

def _render_calc_button(icon: str, help_text: str, key: str, mode: str, key_prefix: str):
    if st.button(icon, key=key, help=help_text, width="stretch"):
        try:
            _prepare_calc(key_prefix, mode)
            st.rerun()
        except Exception as e:
            st.error(str(e))

def _input_format_options(dimension: str) -> list[str]:
    if dimension == "3D":
        return [
            "Components",
            "Typed Components",
            "i/j/k Notation",
            "Magnitude + Azimuth + Elevation",
            "Start + t·Direction",
        ]
    return [
        "Components",
        "Magnitude + Angle",
        "Bearing",
        "Cardinal",
        "Typed Components",
        "i/j Notation",
        "Start + t·Direction",
    ]

def _sync_plot_mode_for_input_format(key_prefix: str):
    fmt = ss.get(f"{key_prefix}_format", "")
    if fmt == "Start + t·Direction":
        ss[f"{key_prefix}_plot_mode"] = "Set Point"

# =========================================================
# main render
# =========================================================

def render_vector_workbench(title: str = "Vector Workbench", key_prefix: str = "vec"):
    _init_vector_state(key_prefix)
    _normalize_vector_state(key_prefix)

    result: Vector3D = ss[f"{key_prefix}_result"]
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
        panel = st.segmented_control(
            "View",
            ["Current Result", "Vector Steps"],
            key=f"{key_prefix}_panel_mode",
            label_visibility="collapsed",
            width="stretch",
        )

        if panel == "Vector Steps":
            _render_steps_tab(key_prefix)
        else:
            _render_current_result_tab(key_prefix)

    with right:
        with st.container(border=True):
            # ----- row 1 -----
            top0, top1, top2 = st.columns([0.6, 1.3, 1.0], gap="small")

            with top0:
                st.selectbox(
                    "Dimension",
                    ["2D", "3D"],
                    key=f"{key_prefix}_dimension",
                    width="stretch",
                )

            with top1:
                st.selectbox(
                    "Input Format",
                    _input_format_options(ss[f"{key_prefix}_dimension"]),
                    key=f"{key_prefix}_format",
                    on_change=_sync_plot_mode_for_input_format,
                    args=(key_prefix,),
                    width="stretch",
                )

            with top2:
                plot_disabled = ss.get(f"{key_prefix}_format") == "Start + t·Direction"

                st.selectbox(
                    "Plot Mode",
                    ["Head-to-tail", "From origin", "Set Point"],
                    key=f"{key_prefix}_plot_mode",
                    disabled=plot_disabled,
                    width="stretch",
                )

            # ----- row 2 -----
            fmt = ss[f"{key_prefix}_format"]

            if fmt == "Components" and ss[f"{key_prefix}_dimension"] == "2D":
                row2a, row2b, row2c = st.columns([0.8, 1.0, 1.0], gap="small")
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

            if fmt == "Components" and ss[f"{key_prefix}_dimension"] == "3D":
                row2a, row2b, row2c, row2d = st.columns([0.7, 1.0, 1.0, 1.0], gap="small")
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
                with row2d:
                    st.number_input("z", key=f"{key_prefix}_z", value=0.0, width="stretch")

            elif fmt == "Magnitude + Angle":
                row2a, row2b, row2c = st.columns([0.8, 1.0, 1.0], gap="small")
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
                row2a, row2b, row2c, row2d, row2e = st.columns([0.75, 1.0, 0.8, 0.9, 0.9], gap="small")
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
                row2a, row2b, row2c = st.columns([0.8, 1.0, 1.0], gap="small")
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

            elif fmt == "Typed Components" and ss[f"{key_prefix}_dimension"] == "2D":
                row2a, row2b = st.columns([0.8, 2.2], gap="small")
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

            elif fmt == "Typed Components" and ss[f"{key_prefix}_dimension"] == "3D":
                row2a, row2b = st.columns([0.8, 2.2], gap="small")
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
                        placeholder="Examples: (3,4,5), <3,4,5>, 3,4,5",
                        width="stretch",
                    )

            elif fmt == "i/j Notation":
                row2a, row2b = st.columns([0.8, 2.2], gap="small")
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
            elif fmt == "i/j/k Notation":
                row2a, row2b = st.columns([0.8, 2.2], gap="small")
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
                        key=f"{key_prefix}_typed_ijk",
                        placeholder="Examples: 3i + 4j + 5k, -2i + 7k",
                        width="stretch",
                    )
            elif fmt == "Magnitude + Azimuth + Elevation":
                row2a, row2b, row2c, row2d = st.columns([0.8, 1.0, 1.0, 1.0], gap="small")
                with row2a:
                    st.text_input(
                        "Label",
                        key=_active_label_widget_key(key_prefix),
                        on_change=_sync_label_from_widget,
                        args=(key_prefix,),
                        width="stretch",
                    )
                with row2b:
                    st.number_input("Magnitude", key=f"{key_prefix}_mag3", value=0.0, width="stretch")
                with row2c:
                    st.number_input("Azimuth (°)", key=f"{key_prefix}_az3", value=0.0, width="stretch")
                with row2d:
                    st.number_input("Elevation (°)", key=f"{key_prefix}_el3", value=0.0, width="stretch")

            elif fmt == "Start + t·Direction" and ss[f"{key_prefix}_dimension"] == "2D":
                row2a, row2b, row2c, row2d = st.columns([0.7, 1.2, 1.2, 0.7], gap="small")

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
                        "Start (x0,y0)",
                        key=f"{key_prefix}_start2d_text",
                        value="0,0",
                        width="stretch",
                    )

                with row2c:
                    st.text_input(
                        "Direction (dx,dy)",
                        key=f"{key_prefix}_dir2d_text",
                        value="0,0",
                        width="stretch",
                    )

                with row2d:
                    st.number_input(
                        "t",
                        key=f"{key_prefix}_t_param",
                        value=1.0,
                        width="stretch",
                    )

                try:
                    s_parts = [p.strip() for p in ss[f"{key_prefix}_start2d_text"].replace("(", "").replace(")", "").split(",")]
                    d_parts = [p.strip() for p in ss[f"{key_prefix}_dir2d_text"].replace("<", "").replace(">", "").replace("(", "").replace(")", "").split(",")]

                    if len(s_parts) == 2 and len(d_parts) == 2:
                        ss[f"{key_prefix}_sx"] = float(s_parts[0])
                        ss[f"{key_prefix}_sy"] = float(s_parts[1])
                        ss[f"{key_prefix}_dx"] = float(d_parts[0])
                        ss[f"{key_prefix}_dy"] = float(d_parts[1])
                except Exception:
                    pass

            elif fmt == "Start + t·Direction" and ss[f"{key_prefix}_dimension"] == "3D":
                row2a, row2b, row2c, row2d = st.columns([0.7, 1.35, 1.35, 0.6], gap="small")

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
                        "Start (x0,y0,z0)",
                        key=f"{key_prefix}_start3d_text",
                        value="0,0,0",
                        width="stretch",
                    )

                with row2c:
                    st.text_input(
                        "Direction (dx,dy,dz)",
                        key=f"{key_prefix}_dir3d_text",
                        value="0,0,0",
                        width="stretch",
                    )

                with row2d:
                    st.number_input(
                        "t",
                        key=f"{key_prefix}_t_param",
                        value=1.0,
                        width="stretch",
                    )

                try:
                    s_parts = [p.strip() for p in ss[f"{key_prefix}_start3d_text"].replace("(", "").replace(")", "").split(",")]
                    d_parts = [p.strip() for p in ss[f"{key_prefix}_dir3d_text"].replace("<", "").replace(">", "").replace("(", "").replace(")", "").split(",")]

                    if len(s_parts) == 3 and len(d_parts) == 3:
                        ss[f"{key_prefix}_sx"] = float(s_parts[0])
                        ss[f"{key_prefix}_sy"] = float(s_parts[1])
                        ss[f"{key_prefix}_sz"] = float(s_parts[2])
                        ss[f"{key_prefix}_dx"] = float(d_parts[0])
                        ss[f"{key_prefix}_dy"] = float(d_parts[1])
                        ss[f"{key_prefix}_dz"] = float(d_parts[2])
                except Exception:
                    pass

            # ----- row 3 -----
            panel_mode = ss.get(f"{key_prefix}_panel_mode", "Current Result")
            selected = ss.get(f"{key_prefix}_selected_steps", [])
            selected_ready = len(selected) == 2

            st.caption("Choose an action")

            a1, a2, a3, a4, a5 = st.columns([1, 1, 1, 1, 1], gap="small")

            with a1:
                if st.button(
                    "📌",
                    key=f"{key_prefix}_set_btn",
                    help="Set the current result equal to this vector",
                    width="stretch",
                    disabled=(panel_mode != "Current Result"),
                ):
                    try:
                        _apply_vector(key_prefix, "set")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            with a2:
                if st.button(
                    "➕",
                    key=f"{key_prefix}_add_btn",
                    help="Add this vector to the current result",
                    width="stretch",
                    disabled=(panel_mode != "Current Result"),
                ):
                    try:
                        _apply_vector(key_prefix, "add")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            with a3:
                if st.button(
                    "➖",
                    key=f"{key_prefix}_sub_btn",
                    help="Subtract this vector from the current result",
                    width="stretch",
                    disabled=(panel_mode != "Current Result"),
                ):
                    try:
                        _apply_vector(key_prefix, "subtract")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            with a4:
                if st.button(
                    "•",
                    key=f"{key_prefix}_dot_btn",
                    help="Dot product of the two selected vectors",
                    width="stretch",
                    disabled=(panel_mode != "Vector Steps" or not selected_ready),
                ):
                    try:
                        _run_selected_calc(key_prefix, "dot")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            with a5:
                if st.button(
                    "✖",
                    key=f"{key_prefix}_cross_btn",
                    help="Cross product of the two selected vectors",
                    width="stretch",
                    disabled=(panel_mode != "Vector Steps" or not selected_ready),
                ):
                    try:
                        _run_selected_calc(key_prefix, "cross")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    calc_text = ss.get(f"{key_prefix}_calc_result_text", "").strip()
    if calc_text:
        st.markdown("#### Calculator Result")
        with st.container(border=True):
            if ss.get(f"{key_prefix}_calc_result_kind") == "vector":
                st.latex(calc_text.split("=", 1)[1].strip())
                st.caption(calc_text.split("=", 1)[0].strip())
            else:
                st.markdown(calc_text)

            if (
                    ss.get(f"{key_prefix}_dimension") == "3D"
                    and ss.get(f"{key_prefix}_calc_plot_vector") is not None
            ):
                st.checkbox(
                    "Plot cross product",
                    key=f"{key_prefix}_calc_plot_enabled",
                )

    st.markdown("#### Vector Plot")

    if ss[f"{key_prefix}_dimension"] == "3D":

        calc_plot_vector = None
        calc_plot_label = ""

        if (
                ss[f"{key_prefix}_dimension"] == "3D"
                and ss.get(f"{key_prefix}_calc_plot_enabled")
                and ss.get(f"{key_prefix}_calc_plot_vector") is not None
        ):
            calc_plot_vector = ss[f"{key_prefix}_calc_plot_vector"]
            calc_plot_label = ss.get(f"{key_prefix}_calc_plot_label", "")

        fig = _plot_vectors_3d(
            result=result,
            history=history,
            plot_mode=ss[f"{key_prefix}_plot_mode"],
            title="3D Vector Plot",
            calc_plot_vector=calc_plot_vector,
            calc_plot_label=calc_plot_label,
        )

    else:
        fig = _plot_vectors_2d(
            result=result,
            history=history,
            plot_mode=ss[f"{key_prefix}_plot_mode"],
            title="2D Vector Plot",
        )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

