from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin, sqrt
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

@dataclass
class Vector3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    label: str = ""

    @property
    def magnitude(self) -> float:
        return sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def azimuth_deg(self) -> float:
        """
        Angle in the xy-plane measured CCW from +x.
        """
        if abs(self.x) < 1e-12 and abs(self.y) < 1e-12:
            return 0.0
        ang = degrees(atan2(self.y, self.x))
        return ang if ang >= 0 else ang + 360

    @property
    def elevation_deg(self) -> float:
        """
        Angle above the xy-plane.
        """
        horiz = hypot(self.x, self.y)
        if abs(horiz) < 1e-12 and abs(self.z) < 1e-12:
            return 0.0
        return degrees(atan2(self.z, horiz))

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def __add__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, k: float) -> "Vector3D":
        return Vector3D(self.x * k, self.y * k, self.z * k)

    def unit(self) -> "Vector3D":
        mag = self.magnitude
        if mag == 0:
            return Vector3D(0.0, 0.0, 0.0, label="unit(0)")
        return Vector3D(self.x / mag, self.y / mag, self.z / mag, label=f"unit({self.label})")

    def dot(self, other: "Vector3D") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def projection_onto(self, other: "Vector3D") -> "Vector3D":
        denom = other.dot(other)
        if abs(denom) < 1e-12:
            return Vector3D(0.0, 0.0, 0.0, label="proj(0)")
        scale = self.dot(other) / denom
        return other.scale(scale)

    def scalar_projection_onto(self, other: "Vector3D") -> float:
        mag = other.magnitude
        if mag < 1e-12:
            return 0.0
        return self.dot(other) / mag


def round_clean(value: float, places: int = 3) -> str:
    s = f"{value:.{places}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def from_components(x: float, y: float, z: float = 0.0, label: str = "") -> Vector3D:
    return Vector3D(float(x), float(y), float(z), label=label)


def from_magnitude_angle(magnitude: float, angle_deg: float, label: str = "") -> Vector3D:
    # 2D convenience constructor
    theta = radians(angle_deg)
    return Vector3D(magnitude * cos(theta), magnitude * sin(theta), 0.0, label=label)


def from_magnitude_azimuth_elevation(
    magnitude: float,
    azimuth_deg: float,
    elevation_deg: float,
    label: str = "",
) -> Vector3D:
    az = radians(azimuth_deg)
    el = radians(elevation_deg)
    x = magnitude * cos(el) * cos(az)
    y = magnitude * cos(el) * sin(az)
    z = magnitude * sin(el)
    return Vector3D(x, y, z, label=label)


def apply_operation(current: Vector3D, incoming: Vector3D, operation: str) -> Vector3D:
    op = operation.strip().lower()
    if op == "set":
        return Vector3D(incoming.x, incoming.y, incoming.z, label="result")
    if op == "add":
        return current + incoming
    if op == "subtract":
        return current - incoming
    raise ValueError(f"Unsupported operation: {operation}")


def from_cardinal(distance: float, direction: str, label: str = "") -> Vector3D:
    d = direction.strip().lower()
    if d == "east":
        return from_components(distance, 0, 0, label=label)
    if d == "west":
        return from_components(-distance, 0, 0, label=label)
    if d == "north":
        return from_components(0, distance, 0, label=label)
    if d == "south":
        return from_components(0, -distance, 0, label=label)
    raise ValueError("Direction must be one of east, west, north, south.")

def from_bearing(magnitude: float, primary: str, angle_deg: float, secondary: str, label: str = "") -> Vector3D:
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

def from_direction_with_t(
    dir_x: float,
    dir_y: float,
    t: float = 1.0,
    label: str = "",
) -> Vector3D:
    return Vector3D(
        float(t) * float(dir_x),
        float(t) * float(dir_y),
        0.0,
        label=label,
    )


def from_direction_with_t_3d(
    dir_x: float,
    dir_y: float,
    dir_z: float,
    t: float = 1.0,
    label: str = "",
) -> Vector3D:
    return Vector3D(
        float(t) * float(dir_x),
        float(t) * float(dir_y),
        float(t) * float(dir_z),
        label=label,
    )

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


def parse_component_text(text: str, label: str = "", dimension: str = "2D") -> Vector3D:
    s = text.strip().replace("<", "").replace(">", "").replace("(", "").replace(")", "")
    parts = [p.strip() for p in s.split(",")]

    if dimension == "3D":
        if len(parts) != 3:
            raise ValueError("Enter components as x, y, z")
        return from_components(float(parts[0]), float(parts[1]), float(parts[2]), label=label)

    if len(parts) != 2:
        raise ValueError("Enter components as x, y")
    return from_components(float(parts[0]), float(parts[1]), 0.0, label=label)


def parse_ijk_text(text: str, label: str = "", dimension: str = "2D") -> Vector3D:
    s = text.lower().replace(" ", "")
    s = s.replace("-", "+-")
    terms = [t for t in s.split("+") if t]

    x = 0.0
    y = 0.0
    z = 0.0

    for t in terms:
        if t.endswith("i"):
            coef = t[:-1]
            coef = "1" if coef in {"", "+"} else "-1" if coef == "-" else coef
            x += float(coef)
        elif t.endswith("j"):
            coef = t[:-1]
            coef = "1" if coef in {"", "+"} else "-1" if coef == "-" else coef
            y += float(coef)
        elif t.endswith("k"):
            coef = t[:-1]
            coef = "1" if coef in {"", "+"} else "-1" if coef == "-" else coef
            z += float(coef)
        else:
            raise ValueError("Invalid i/j/k vector format.")

    if dimension == "2D" and abs(z) > 1e-12:
        raise ValueError("k terms are only allowed in 3D mode.")

    return Vector3D(x, y, z, label=label)


def parse_ij_text(text: str, label: str = "") -> Vector3D:
    return parse_ijk_text(text, label=label, dimension="2D")


def vector_summary(v: Vector3D, places: int = 3) -> dict:
    return {
        "x": round_clean(v.x, places),
        "y": round_clean(v.y, places),
        "z": round_clean(v.z, places),
        "magnitude": round_clean(v.magnitude, places),
        "azimuth_deg": round_clean(v.azimuth_deg, places),
        "elevation_deg": round_clean(v.elevation_deg, places),
    }


def vector_to_latex(v: Vector3D, places: int = 3, dimension: str = "2D") -> str:
    x = round_clean(v.x, places)
    y = round_clean(v.y, places)
    z = round_clean(v.z, places)
    if dimension == "3D":
        return rf"\langle {x},\ {y},\ {z} \rangle"
    return rf"\langle {x},\ {y} \rangle"