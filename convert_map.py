import re
from pathlib import Path

# Regular expressions for parsing
PLANE_RE = re.compile(
    r'\(\s*([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s*\)\s*'
    r'\(\s*([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s*\)\s*'
    r'\(\s*([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s*\)\s*'
    r'(\S+)\s*'  # texture name
    r'\[\s*([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s*\]\s*'
    r'\[\s*([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s*\]\s*'
    r'([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)'
)

def parse_map(filepath):
    text = Path(filepath).read_text(encoding='utf-8')

    entities = []
    lines = iter(text.splitlines())

    current_entity = None
    current_brush = None

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("//"):
            continue

        if line.startswith("{"):
            if current_brush is not None:
                # nested brace => new brush inside entity
                continue
            elif current_entity is not None:
                # brush start
                current_brush = {"planes": []}
            else:
                # new entity
                current_entity = {"keys": {}, "brushes": []}

        elif line.startswith("}"):
            if current_brush is not None:
                # brush end
                current_entity["brushes"].append(current_brush)
                current_brush = None
            elif current_entity is not None:
                # entity end
                entities.append(current_entity)
                current_entity = None

        elif current_brush is not None:
            # inside brush: parse plane
            m = PLANE_RE.match(line)
            if m:
                p = m.groups()
                plane = {
                    "points": [
                        (float(p[0]), float(p[1]), float(p[2])),
                        (float(p[3]), float(p[4]), float(p[5])),
                        (float(p[6]), float(p[7]), float(p[8])),
                    ],
                    "texture": p[9],
                    "uaxis": [float(p[10]), float(p[11]), float(p[12]), float(p[13])],
                    "vaxis": [float(p[14]), float(p[15]), float(p[16]), float(p[17])],
                    "rotation": float(p[18]),
                    "scale": (float(p[19]), float(p[20])),
                }
                current_brush["planes"].append(plane)
            else:
                print("⚠️ Could not parse plane:", line)

        elif current_entity is not None:
            # key/value pair
            if line.startswith('"'):
                parts = re.findall(r'"([^"]*)"', line)
                if len(parts) == 2:
                    key, val = parts
                    current_entity["keys"][key] = val

    return entities

import sys
if __name__ == "__main__":
    in_path = Path(f"source_data/{sys.argv[1]}.map")
    out_path = Path(f"build/levels/{sys.argv[1]}.map")
    entities = parse_map(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ent in entities:
            classname = ent["keys"].get("classname", "<unknown>")
            f.write(f"entity {classname}\n")
            for k, v in ent["keys"].items():
                if k == "classname":
                    continue
                f.write(f"  {k} {v}\n")
            for i, b in enumerate(ent["brushes"]):
                for plane in b["planes"]:
                    pts = plane["points"]
                    u = plane["uaxis"]
                    v = plane["vaxis"]
                    f.write(
                        f"  plane ({pts[0][0]:.3f},{pts[0][1]:.3f},{pts[0][2]:.3f}) "
                        f"({pts[1][0]:.3f},{pts[1][1]:.3f},{pts[1][2]:.3f}) "
                        f"({pts[2][0]:.3f},{pts[2][1]:.3f},{pts[2][2]:.3f}) "
                        f"({u[0]},{u[1]},{u[2]},{u[3]}) "
                        f"({v[0]},{v[1]},{v[2]},{v[3]}) "
                        f"{plane['texture']} {plane['rotation']/360.0} ({plane['scale'][0]},{plane['scale'][1]})\n"
                    )
    print(f"Parsed and wrote {len(entities)} entities to {out_path}")