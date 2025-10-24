import re
import math
import struct
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# -------------------------------
# Types
# -------------------------------
Vec3 = Tuple[float, float, float]
Plane = Tuple[float, float, float, float]  # (nx, ny, nz, d)  keep n·x <= d
Polygon3D = List[Vec3]
Polyhedron = List[Polygon3D]

# -------------------------------
# Public API
# -------------------------------

def parse_map_to_mesh_brushes(map_path: str) -> List[Tuple[List[Plane], List[Dict[str, Any]]]]:
    """
    Parse TrenchBroom 'Generic / Standard' .map (also works for classic Quake/Hammer)
    and return [(planes, face_infos), ...] suitable for planes_to_mesh_with_uvs.

    - Supports nested entities with '{ ... }'
    - Detects brush blocks (a '{' inside an entity), ignores comments and key/values
    - Face lines are classic: (p1)(p2)(p3) tex xoff yoff rot scaleX scaleY
    """
    text = Path(map_path).read_text(encoding='utf-8', errors='ignore')
    lines = text.splitlines()

    # Regex for a classic face line (matches TB Generic)
    num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    pt  = rf"\(\s*({num})\s+({num})\s+({num})\s*\)"
    tex = r"([^\s]+)"
    rx_face = re.compile(
        rf"^{pt}\s*{pt}\s*{pt}\s*{tex}\s+({num})\s+({num})\s+({num})\s+({num})\s+({num})\s*$"
    )

    brushes: List[Tuple[List[Plane], List[Dict[str, Any]]]] = []

    depth = 0
    inside_brush = False
    cur_planes: List[Plane] = []
    cur_faces: List[Dict[str, Any]] = []

    def strip_comment(s: str) -> str:
        # Remove whole-line and end-of-line '//' comments without disturbing negatives
        if '//' in s:
            # Only treat as comment if '//' is not inside parentheses or brackets
            # (TB uses clean tokens, so this is safe.)
            idx = s.find('//')
            return s[:idx].rstrip()
        return s

    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1

        line = strip_comment(raw).strip()
        if not line:
            continue

        if line == "{":
            depth += 1
            # Entering a brush when we’re at entity depth >= 1 and not already in a brush
            if not inside_brush and depth >= 2:
                inside_brush = True
                cur_planes = []
                cur_faces = []
            continue

        if line == "}":
            # Closing a brush block?
            if inside_brush and depth >= 2:
                if cur_planes:
                    brushes.append((cur_planes, cur_faces))
                inside_brush = False
                cur_planes = []
                cur_faces = []
            depth = max(0, depth - 1)
            continue

        if not inside_brush:
            # key/value lines like: "classname" "worldspawn" — skip
            continue

        # Face line (classic)
        m = rx_face.match(line)
        if not m:
            # Non-face stuff inside brush (shouldn’t happen in TB Generic) — skip
            continue

        # Parse the three points
        p1 = tuple(map(float, m.group(1,2,3)))   # type: ignore
        p2 = tuple(map(float, m.group(4,5,6)))
        p3 = tuple(map(float, m.group(7,8,9)))
        texname = m.group(10)

        xoff = float(m.group(11))
        yoff = float(m.group(12))
        rot  = float(m.group(13))
        sU   = float(m.group(14)) if float(m.group(14)) != 0 else 1.0
        sV   = float(m.group(15)) if float(m.group(15)) != 0 else 1.0

        # Plane from points. In TB Generic, (p1,p2,p3) are ordered so that the normal
        # points OUTWARD from the brush; we keep half-space n·x <= d.
        n, d = _plane_from_points(p1, p2, p3)
        cur_planes.append((n[0], n[1], n[2], d))

        cur_faces.append({
            "is_valve220": False,   # TB Generic uses classic mapping
            "uaxis": None,          # derived later from plane normal + rot
            "ushift": xoff,
            "vaxis": None,
            "vshift": yoff,
            "rot":   rot,
            "scaleU": sU,
            "scaleV": sV,
            "tex":   texname
        })

    return brushes


def planes_to_mesh_with_uvs(planes: List[Plane], face_infos: List[Dict[str, Any]], eps: float = 1e-6):
    """
    Convert a brush defined by planes + face_infos into a mesh with UVs and tangent space.

    Returns:
        (verts, idx, uv, nrm, tan, bsign)
        - verts: (N,3) float32
        - idx:   (M,3) uint32 (triangles)
        - uv:    (N,2) float32
        - nrm:   (N,3) float32
        - tan:   (N,3) float32
        - bsign: (N,1) float32  (handedness)
    or None if degenerate/invalid.
    """
    # Build convex polyhedron by clipping a huge box by the planes
    poly = clip_brush_by_planes(planes, eps=eps)
    if not poly:
        return None

    # For each resulting face, match it back to its source plane to get UV mapping
    # We match by comparing face plane (normal,d) to input planes (parallel + close d).
    face_to_src: List[int] = []
    src_used = [False] * len(planes)

    face_polys: List[Polygon3D] = []
    for face in poly:
        if len(face) < 3:
            continue
        # Compute its plane (unoriented polygon normal ~ outward already)
        fn = _normalize(_polygon_normal(face))
        # Compute d as average dot over vertices
        d = float(np.mean([_dot(fn, v) for v in face]))
        # Find best matching input plane by angle and d-distance
        best = -1
        best_err = 1e9
        for j, (nx,ny,nz,dj) in enumerate(planes):
            n_in = _normalize((nx,ny,nz))
            angle = 1.0 - abs(_dot(fn, n_in))     # 0 means parallel/same or opposite
            derr = abs(d - dj)
            score = angle*1000.0 + derr
            if score < best_err:
                best_err = score
                best = j
        if best < 0 or best_err > 1e-3:
            # couldn't match; skip this face
            continue
        # Ensure the face normal points outward (same as plane normal)
        n_in = _normalize(planes[best][:3])
        if _dot(_polygon_normal(face), n_in) < 0:
            face = list(reversed(face))
        face_polys.append(face)
        face_to_src.append(best)
        src_used[best] = True

    if not face_polys:
        return None

    # Build per-face UV basis (U, V, shift, scale)
    per_face_params = []
    for idx_src in face_to_src:
        finfo = face_infos[idx_src]
        if finfo["is_valve220"]:
            U = _normalize(finfo["uaxis"])
            V = _normalize(finfo["vaxis"])
            # Make V orthonormal to U (robust)
            V = _normalize(_sub(V, _mul(U, _dot(U, V))))
            per_face_params.append((
                U, V, float(finfo["ushift"]), float(finfo["vshift"]),
                float(finfo["scaleU"]), float(finfo["scaleV"])
            ))
        else:
            # Classic: derive base axes from plane normal + rotation
            n = _normalize(planes[idx_src][:3])
            U0, V0 = _quake_base_axes(n)
            # Apply rotation around normal
            rot = math.radians(float(finfo["rot"]))
            U = _normalize(_rot_around_axis(U0, n, rot))
            V = _normalize(_rot_around_axis(V0, n, rot))
            per_face_params.append((
                U, V, float(finfo["ushift"]), float(finfo["vshift"]),
                float(finfo["scaleU"]), float(finfo["scaleV"])
            ))

    # Stitch all faces into one mesh (shared vertices per-face; flat shading)
    positions: List[Vec3] = []
    uvs: List[Tuple[float,float]] = []
    normals: List[Vec3] = []
    indices: List[Tuple[int,int,int]] = []

    for face, params, src_idx in zip(face_polys, per_face_params, face_to_src):
        U, V, ushift, vshift, sU, sV = params
        n = _normalize(planes[src_idx][:3])
        # UVs per vertex
        base_index = len(positions)
        for P in face:
            u = (_dot(P, U) / sU) + ushift
            v = (_dot(P, V) / sV) + vshift
            positions.append(P)
            uvs.append((u, v))
            normals.append(n)
        # Triangulate fan
        for k in range(1, len(face)-1):
            indices.append((base_index, base_index+k, base_index+k+1))

    if not indices:
        return None

    # Convert to numpy arrays
    verts = np.array(positions, dtype=np.float32)
    idx = np.array(indices, dtype=np.uint32)
    uv = np.array(uvs, dtype=np.float32)
    nrm = np.array(normals, dtype=np.float32)

    # Compute tangents + bitangent sign
    tan, bsign = _compute_tangent_space(verts, uv, nrm, idx)

    return verts, idx, uv, nrm, tan, bsign


# -------------------------------
# User-provided writer (kept verbatim)
# -------------------------------

def write_mesh(mesh_brushes, out_path):
    import numpy as np
    import struct
    valid_brushes = []
    for planes, face_infos in mesh_brushes:
        result = planes_to_mesh_with_uvs(planes, face_infos)
        if result is not None:
            valid_brushes.append(result)
        else:
            print("  [INFO] Skipped one invalid brush")

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<I', len(valid_brushes)))
        for verts, idx, uv, nrm, tan, bsign in valid_brushes:
            vert_data = np.column_stack([
                verts, uv, nrm,
                np.column_stack([tan, bsign])
            ]).astype(np.float32).ravel()
            idx_data = idx.astype(np.uint32).ravel()

            vcount = int(len(vert_data)/12)
            f.write(struct.pack('<I', vcount))
            f.write(struct.pack('<I', len(idx_data)))
            f.write(struct.pack(f'<{len(vert_data)}f', *vert_data))
            f.write(struct.pack(f'<{len(idx_data)}I', *idx_data))
            print(f'mesh: {len(verts)} verts, {len(idx)} tris')
    print(f"Wrote {len(valid_brushes)} valid meshes to {out_path}")


# -------------------------------
# HL/Quake brush clipping (3D Sutherland–Hodgman over half-spaces)
# -------------------------------

def clip_brush_by_planes(
    planes: List[Plane],
    bounds_min: Vec3 = (-1e6, -1e6, -1e6),
    bounds_max: Vec3 = ( 1e6,  1e6,  1e6),
    eps: float = 1e-6,
) -> Polyhedron:
    poly = _make_box_faces(bounds_min, bounds_max)
    for plane in planes:
        poly = _clip_polyhedron_against_plane(poly, plane, eps)
        if not poly:
            break
    return poly

def _clip_polyhedron_against_plane(poly: Polyhedron, plane: Plane, eps: float) -> Polyhedron:
    n = plane[:3]
    d = plane[3]
    new_faces: Polyhedron = []
    cut_ring_points: List[Vec3] = []

    for face in poly:
        if not face:
            continue
        clipped, cut_pts = _clip_polygon_against_plane(face, n, d, eps)
        if len(clipped) >= 3:
            new_faces.append(clipped)
        cut_ring_points.extend(cut_pts)

    # Create the new cut face from intersection ring
    cut_face = _assemble_cut_face_on_plane(cut_ring_points, n, d, eps)
    if len(cut_face) >= 3:
        # orient outward
        face_n = _polygon_normal(cut_face)
        if _dot(face_n, n) < 0:
            cut_face.reverse()
        new_faces.append(cut_face)

    return new_faces

def _clip_polygon_against_plane(
    poly: Polygon3D, n: Vec3, d: float, eps: float
) -> Tuple[Polygon3D, List[Vec3]]:
    out: Polygon3D = []
    ring_points: List[Vec3] = []
    S = poly[-1]
    Sd = _dot(n, S) - d
    S_inside = Sd <= eps

    for E in poly:
        Ed = _dot(n, E) - d
        E_inside = Ed <= eps

        if E_inside:
            if not S_inside:
                P = _segment_plane_intersection(S, E, n, d, eps)
                if P is not None:
                    out.append(P); ring_points.append(P)
            out.append(E)
        else:
            if S_inside:
                P = _segment_plane_intersection(S, E, n, d, eps)
                if P is not None:
                    out.append(P); ring_points.append(P)
        S, Sd, S_inside = E, Ed, E_inside

    out = _dedupe_polygon3d(out, eps)
    return out, ring_points

def _assemble_cut_face_on_plane(points: List[Vec3], n: Vec3, d: float, eps: float) -> Polygon3D:
    pts = _dedupe_points(points, eps)
    if len(pts) < 3:
        return []
    nh = _normalize(n)
    # Build 2D basis on plane
    u = _any_perp_unit(nh)
    v = _normalize(_cross(nh, u))
    # Project to plane and sort by angle around centroid
    proj = []
    for p in pts:
        # Project onto plane: p' = p - nh * (n·p - d)/||n||
        alpha = ( _dot(n, p) - d ) / (_norm(n) + 1e-20)
        p_on = _sub(p, _mul(nh, alpha))
        proj.append(p_on)
    c = _centroid(proj)
    uv = [(_dot(_sub(p, c), u), _dot(_sub(p, c), v), p) for p in proj]
    uv.sort(key=lambda t: math.atan2(t[1], t[0]))
    loop = [p for _,_,p in uv]
    loop = _dedupe_polygon3d(loop, eps)
    return loop

def _segment_plane_intersection(S: Vec3, E: Vec3, n: Vec3, d: float, eps: float) -> Optional[Vec3]:
    dirv = _sub(E, S)
    denom = _dot(n, dirv)
    if abs(denom) < eps:
        return None
    t = (d - _dot(n, S)) / denom
    t = max(0.0, min(1.0, t))
    return (S[0] + t*dirv[0], S[1] + t*dirv[1], S[2] + t*dirv[2])

# -------------------------------
# Tangent space (MikkTSpace-style accumulation)
# -------------------------------

def _compute_tangent_space(verts: np.ndarray, uv: np.ndarray, nrm: np.ndarray, idx: np.ndarray):
    N = verts.shape[0]
    tan1 = np.zeros((N, 3), dtype=np.float64)
    tan2 = np.zeros((N, 3), dtype=np.float64)

    for i0, i1, i2 in idx:
        p0, p1, p2 = verts[i0], verts[i1], verts[i2]
        uv0, uv1, uv2 = uv[i0], uv[i1], uv[i2]

        edge1 = p1 - p0
        edge2 = p2 - p0
        duv1 = uv1 - uv0
        duv2 = uv2 - uv0

        denom = duv1[0]*duv2[1] - duv2[0]*duv1[1]
        if abs(denom) < 1e-20:
            # Degenerate UV mapping; skip this tri
            continue
        r = 1.0 / denom
        sdir = (edge1 * duv2[1] - edge2 * duv1[1]) * r
        tdir = (edge2 * duv1[0] - edge1 * duv2[0]) * r

        tan1[i0] += sdir; tan1[i1] += sdir; tan1[i2] += sdir
        tan2[i0] += tdir; tan2[i1] += tdir; tan2[i2] += tdir

    # Orthonormalize & sign
    tan = np.zeros_like(verts, dtype=np.float32)
    bsign = np.ones((N, 1), dtype=np.float32)

    for i in range(N):
        n = nrm[i].astype(np.float64)
        t = tan1[i]
        # Gram–Schmidt orthonormalize
        t = t - n * np.dot(n, t)
        tn = np.linalg.norm(t)
        if tn < 1e-20:
            # Fallback: pick any perpendicular
            t = _any_perp(n)
            tn = np.linalg.norm(t)
        t = t / max(tn, 1e-20)

        # Handedness
        b = np.cross(n, t)
        s = 1.0 if np.dot(b, tan2[i]) >= 0.0 else -1.0

        tan[i] = t.astype(np.float32)
        bsign[i, 0] = np.float32(s)

    return tan, bsign

# -------------------------------
# Classic Quake UV base axes + rotation
# -------------------------------

def _quake_base_axes(n: Vec3) -> Tuple[Vec3, Vec3]:
    # Based on largest component of the normal; these mimic common tools.
    ax = abs(n[0]); ay = abs(n[1]); az = abs(n[2])
    if az >= ax and az >= ay:
        # floor/ceiling
        U = (1.0, 0.0, 0.0)
        V = (0.0, -1.0 if n[2] > 0 else 1.0, 0.0)
    elif ax >= ay and ax >= az:
        # wall facing +/-X
        U = (0.0, 1.0, 0.0)
        V = (0.0, 0.0, -1.0 if n[0] > 0 else 1.0)
    else:
        # wall facing +/-Y
        U = (1.0, 0.0, 0.0)
        V = (0.0, 0.0, -1.0 if n[1] < 0 else 1.0)
    # Orthonormalize just in case
    U = _normalize(U)
    V = _normalize(_sub(V, _mul(U, _dot(U, V))))
    return U, V

def _rot_around_axis(v: Vec3, axis: Vec3, angle: float) -> Vec3:
    # Rodrigues' rotation formula (axis must be unit)
    a = _normalize(axis)
    c = math.cos(angle); s = math.sin(angle)
    vx, vy, vz = v
    axx, ayy, azz = a
    cross = _cross(a, v)
    dot = _dot(a, v)
    out = (
        vx*c + cross[0]*s + axx*dot*(1-c),
        vy*c + cross[1]*s + ayy*dot*(1-c),
        vz*c + cross[2]*s + azz*dot*(1-c),
    )
    return out

# -------------------------------
# Geometry helpers
# -------------------------------

def _plane_from_points(p1: Vec3, p2: Vec3, p3: Vec3) -> Tuple[Vec3, float]:
    v1 = _sub(p2, p1)
    v2 = _sub(p3, p1)
    n = _normalize(_cross(v1, v2))
    d = _dot(n, p1)
    return n, d

def _make_box_faces(mi: Vec3, ma: Vec3) -> Polyhedron:
    x0,y0,z0 = mi
    x1,y1,z1 = ma
    return [
        [(x0,y0,z0), (x0,y0,z1), (x0,y1,z1), (x0,y1,z0)],  # -X
        [(x1,y0,z0), (x1,y1,z0), (x1,y1,z1), (x1,y0,z1)],  # +X
        [(x0,y0,z0), (x1,y0,z0), (x1,y0,z1), (x0,y0,z1)],  # -Y
        [(x0,y1,z0), (x0,y1,z1), (x1,y1,z1), (x1,y1,z0)],  # +Y
        [(x0,y0,z0), (x0,y1,z0), (x1,y1,z0), (x1,y0,z0)],  # -Z
        [(x0,y0,z1), (x1,y0,z1), (x1,y1,z1), (x0,y1,z1)],  # +Z
    ]

def _polygon_normal(poly: Polygon3D) -> Vec3:
    # Newell's method
    nx = ny = nz = 0.0
    n = len(poly)
    for i in range(n):
        x1,y1,z1 = poly[i]
        x2,y2,z2 = poly[(i+1)%n]
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    return (nx, ny, nz)

def _dedupe_points(points: List[Vec3], eps: float) -> List[Vec3]:
    out: List[Vec3] = []
    for p in points:
        if not any(_dist2(p, q) <= (eps*eps)*100 for q in out):
            out.append(p)
    return out

def _dedupe_polygon3d(poly: Polygon3D, eps: float) -> Polygon3D:
    if len(poly) <= 2:
        return poly[:]
    cleaned: Polygon3D = []
    for p in poly:
        if not cleaned or _dist2(p, cleaned[-1]) > (eps*eps)*100:
            cleaned.append(p)
    if cleaned and _dist2(cleaned[0], cleaned[-1]) <= (eps*eps)*100:
        cleaned.pop()
    # remove collinear middles
    if len(cleaned) <= 2:
        return cleaned
    out: Polygon3D = []
    def area_near_zero(a: Vec3, b: Vec3, c: Vec3, tol=1e-12) -> bool:
        ab = _sub(b, a); bc = _sub(c, b)
        cr = _cross(ab, bc)
        return _norm(cr) <= tol
    for p in cleaned:
        out.append(p)
        while len(out) >= 3 and area_near_zero(out[-3], out[-2], out[-1]):
            out.pop(-2)
    return out

def _centroid(pts: List[Vec3]) -> Vec3:
    sx = sy = sz = 0.0
    for x,y,z in pts:
        sx += x; sy += y; sz += z
    inv = 1.0/max(1, len(pts))
    return (sx*inv, sy*inv, sz*inv)

def _dot(a: Vec3, b: Vec3) -> float:
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def _mul(a: Vec3, s: float) -> Vec3:
    return (a[0]*s, a[1]*s, a[2]*s)

def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def _norm(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))

def _normalize(a: Vec3) -> Vec3:
    n = _norm(a)
    if n == 0.0: return (0.0,0.0,0.0)
    return (a[0]/n, a[1]/n, a[2]/n)

def _any_perp_unit(n: Vec3) -> Vec3:
    nx, ny, nz = map(abs, n)
    if nx < ny and nx < nz:
        v = (1.0, 0.0, 0.0)
    elif ny < nz:
        v = (0.0, 1.0, 0.0)
    else:
        v = (0.0, 0.0, 1.0)
    u = _cross(n, v)
    un = _norm(u)
    if un == 0.0:
        # pick something else
        u = _cross(n, (0.0,1.0,0.0))
        un = _norm(u) or 1.0
    return (u[0]/un, u[1]/un, u[2]/un)

def _any_perp(n: np.ndarray) -> np.ndarray:
    n = n.astype(np.float64)
    a = np.array([1.0,0.0,0.0]) if abs(n[0]) < 0.9 else np.array([0.0,1.0,0.0])
    v = np.cross(n, a)
    vn = np.linalg.norm(v) or 1.0
    return v / vn

def _dist2(p: Vec3, q: Vec3) -> float:
    return (p[0]-q[0])**2 + (p[1]-q[1])**2


# -------------------------------
# Convenience: parse + write end-to-end
# -------------------------------

def parse_and_write_map(map_path: str, out_path: str):
    brushes = parse_map_to_mesh_brushes(map_path)
    print(f"Parsed {len(brushes)} brushes")
    write_mesh(brushes, out_path)

if __name__ == "__main__":
	parse_and_write_map(f'{sys.argv[1]}.map', f'../build/levels/{sys.argv[1]}.map_geo')

