from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin
import re


@dataclass
class Vector2D:
    x: float = 0.0
    y: float = 0.0
    label: str = ""

    @property
    def magnitude(self) -> float:
        return hypot(self.x, self.y)

    @property
    def angle_deg(self) -> float:
        """Standard math angle measured CCW from +x axis."""
        ang = degrees(atan2(self.y, self.x))
        return ang if ang >= 0 else ang + 360

    def as_tuple(self) -> tuple[float, float]:
        return self.x, self.y

    def unit(self) -> "Vector2D":
        mag = self.magnitude
        if mag == 0:
            return Vector2D(0.0, 0.0, label="unit(0)")
        return Vector2D(self.x / mag, self.y / mag, label=f"unit({self.label})")

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def scale(self, k: float) -> "Vector2D":
        return Vector2D(self.x * k, self.y * k)

    def dot(self, other: "Vector2D") -> float:
        return self.x * other.x + self.y * other.y

    def cross_z(self, other: "Vector2D") -> float:
        """2D cross product returns z-component."""
        return self.x * other.y - self.y * other.x


def round_clean(value: float, places: int = 3) -> str:
    s = f"{value:.{places}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def vector_summary(v: Vector2D, places: int = 3) -> dict:
    return {
        "x": round_clean(v.x, places),
        "y": round_clean(v.y, places),
        "magnitude": round_clean(v.magnitude, places),
        "angle_deg": round_clean(v.angle_deg, places),
        "bearing": math_angle_to_bearing(v.angle_deg),
    }


def vector_to_latex(v: Vector2D, places: int = 3) -> str:
    x = round_clean(v.x, places)
    y = round_clean(v.y, places)
    return rf"\langle {x},\ {y} \rangle"


def from_components(x: float, y: float, label: str = "") -> Vector2D:
    return Vector2D(float(x), float(y), label=label)


def from_magnitude_angle(magnitude: float, angle_deg: float, label: str = "") -> Vector2D:
    theta = radians(angle_deg)
    x = magnitude * cos(theta)
    y = magnitude * sin(theta)
    return Vector2D(x, y, label=label)


def from_cardinal(distance: float, direction: str, label: str = "") -> Vector2D:
    d = direction.strip().lower()
    if d == "east":
        return Vector2D(distance, 0, label=label)
    if d == "west":
        return Vector2D(-distance, 0, label=label)
    if d == "north":
        return Vector2D(0, distance, label=label)
    if d == "south":
        return Vector2D(0, -distance, label=label)
    raise ValueError("Direction must be one of east, west, north, south.")


def from_bearing(magnitude: float, primary: str, angle_deg: float, secondary: str, label: str = "") -> Vector2D:
    """
    Examples:
      N 30 E
      S 20 W
      E 15 N
      W 40 S
    """
    p = primary.strip().upper()
    s = secondary.strip().upper()
    a = float(angle_deg)

    if p not in {"N", "S", "E", "W"} or s not in {"N", "S", "E", "W"}:
        raise ValueError("Bearing directions must be N, S, E, or W.")

    if p in {"N", "S"} and s in {"N", "S"}:
        raise ValueError("Secondary direction must differ by axis from primary.")
    if p in {"E", "W"} and s in {"E", "W"}:
        raise ValueError("Secondary direction must differ by axis from primary.")

    theta = bearing_to_math_angle(p, a, s)
    return from_magnitude_angle(magnitude, theta, label=label)


def bearing_to_math_angle(primary: str, angle_deg: float, secondary: str) -> float:
    p, s, a = primary.upper(), secondary.upper(), float(angle_deg)

    if p == "N" and s == "E":
        return 90 - a
    if p == "N" and s == "W":
        return 90 + a
    if p == "S" and s == "E":
        return 270 + a
    if p == "S" and s == "W":
        return 270 - a
    if p == "E" and s == "N":
        return a
    if p == "E" and s == "S":
        return 360 - a
    if p == "W" and s == "N":
        return 180 - a
    if p == "W" and s == "S":
        return 180 + a

    raise ValueError("Invalid bearing combination.")


def math_angle_to_bearing(angle_deg: float) -> str:
    a = angle_deg % 360

    if 0 <= a <= 90:
        return f"E {round_clean(a)}° N"
    if 90 < a <= 180:
        return f"N {round_clean(180 - a)}° W"
    if 180 < a <= 270:
        return f"W {round_clean(a - 180)}° S"
    return f"S {round_clean(360 - a)}° E"


def apply_operation(current: Vector2D, incoming: Vector2D, operation: str) -> Vector2D:
    op = operation.strip().lower()
    if op == "set":
        return Vector2D(incoming.x, incoming.y, label="result")
    if op == "add":
        return current + incoming
    if op == "subtract":
        return current - incoming
    raise ValueError(f"Unsupported operation: {operation}")


def parse_component_text(text: str, label: str = "") -> Vector2D:
    """
    Accepts:
      (3, 4)
      <3, 4>
      3, 4
    """
    s = text.strip().replace("<", "").replace(">", "").replace("(", "").replace(")", "")
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError("Enter components as x, y")
    return from_components(float(parts[0]), float(parts[1]), label=label)


def parse_ij_text(text: str, label: str = "") -> Vector2D:
    """
    Accepts simple forms like:
      3i + 4j
      -2i - 5j
      7j
      -3i
    """
    s = text.lower().replace(" ", "")
    s = s.replace("-", "+-")
    terms = [t for t in s.split("+") if t]

    x = 0.0
    y = 0.0

    for t in terms:
        if t.endswith("i"):
            coef = t[:-1]
            if coef in {"", "+"}:
                coef = "1"
            elif coef == "-":
                coef = "-1"
            x += float(coef)
        elif t.endswith("j"):
            coef = t[:-1]
            if coef in {"", "+"}:
                coef = "1"
            elif coef == "-":
                coef = "-1"
            y += float(coef)
        else:
            raise ValueError("Invalid i/j vector format.")
    return Vector2D(x, y, label=label)
