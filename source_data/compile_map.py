import struct
import math
import re
from itertools import combinations
import sys
EPSILON = 1e-5
class Vec3:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = float(x), float(y), float(z)
    def __add__(self, o): return Vec3(self.x+o.x, self.y+o.y, self.z+o.z)
    def __sub__(self, o): return Vec3(self.x-o.x, self.y-o.y, self.z-o.z)
    def __mul__(self, s): return Vec3(self.x*s, self.y*s, self.z*s)
    def __truediv__(self, s): return Vec3(self.x/s, self.y/s, self.z/s)
    def dot(self,o): return self.x*o.x + self.y*o.y + self.z*o.z
    def cross(self,o): return Vec3(self.y*o.z - self.z*o.y, self.z*o.x - self.x*o.z, self.x*o.y - self.y*o.x)
    def length(self): return math.sqrt(self.dot(self))
    def normalize(self):
        l = self.length()
        if l<EPSILON: return Vec3(0,0,0)
        return self/l
    def tuple(self): return (self.x, self.y, self.z)
    def __repr__(self): return f"({self.x},{self.y},{self.z})"

class Plane:
    def __init__(self,n,d): self.n, self.d = n,d
    def dist(self,p): return self.n.dot(p) - self.d
def plane_from_points(a,b,c):
    n = (b - a).cross(c - a).normalize()
    d = n.dot(a)
    return Plane(n,d)
def intersect_planes(p1,p2,p3):
    denom = p1.n.dot(p2.n.cross(p3.n))
    if abs(denom)<EPSILON: return None
    p = (p2.n.cross(p3.n)*p1.d + p3.n.cross(p1.n)*p2.d + p1.n.cross(p2.n)*p3.d) * (1.0/denom)
    return p
def inside_all(p,planes):
    return all(pl.dist(p)>=-EPSILON for pl in planes)
def unique_vertex(v,verts):
    for u in verts:
        if (v-u).length()<0.01: return False
    return True
def build_brush_vertices(planes):
    verts=[]
    for p1,p2,p3 in combinations(planes,3):
        p=intersect_planes(p1,p2,p3)
        if p and inside_all(p,planes) and unique_vertex(p,verts):
            verts.append(p)
    return verts
def parse_map(path):
    with open(path,'r') as f: text=f.read()
    brush_blocks = re.findall(r'\{\s*(?:\(.+?\)\s*\(.+?\)\s*\(.+?\).+?\n)+?\}', text, re.DOTALL)
    all_brushes=[]
    plane_re = re.compile(r'\(\s*([-0-9\.]+)\s+([-0-9\.]+)\s+([-0-9\.]+)\s*\)\s*'
                          r'\(\s*([-0-9\.]+)\s+([-0-9\.]+)\s+([-0-9\.]+)\s*\)\s*'
                          r'\(\s*([-0-9\.]+)\s+([-0-9\.]+)\s+([-0-9\.]+)\s*\)')
    for b in brush_blocks:
        planes=[]
        for match in plane_re.finditer(b):
            pts=[Vec3(*map(float,match.groups()[i:i+3])) for i in (0,3,6)]
            planes.append(plane_from_points(pts[0],pts[1],pts[2]))
        if len(planes)>=4:
            verts = build_brush_vertices(planes)
            if verts:
                all_brushes.append((planes, verts))
    return all_brushes
def sort_face_vertices(face_vertices, normal):
    centroid = Vec3(0,0,0)
    for v in face_vertices: centroid += v
    centroid /= len(face_vertices)
    u = Vec3(1,0,0) if abs(normal.x)<0.9 else Vec3(0,1,0)
    u = (u - normal*(normal.dot(u))).normalize()
    v_vec = normal.cross(u)
    def angle(p):
        d = p - centroid
        return math.atan2(d.dot(v_vec), d.dot(u))
    return sorted(face_vertices, key=angle)
def triangulate_face(face_vertices):
    tris=[]
    if len(face_vertices)<3: return []
    for i in range(1,len(face_vertices)-1):
        tris.append([face_vertices[0], face_vertices[i+1], face_vertices[i]])
    return tris
def write_mesh(mesh_brushes, output_path):
    all_vertices = []
    vert_counts = []
    for planes, verts in mesh_brushes:
        brush_vertices = []
        for pl in planes:
            face_vertices = [v for v in verts if abs(pl.dist(v)) < EPSILON*10]
            if len(face_vertices) < 3:
                continue
            face_vertices = sort_face_vertices(face_vertices, pl.n)
            tris = triangulate_face(face_vertices)
            for tri in tris:
                for v in tri:
                    brush_vertices.append((v.x, v.y, v.z))
        vert_counts.append(len(brush_vertices))
        all_vertices.extend(brush_vertices)
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', len(mesh_brushes)))
        for count in vert_counts:
            f.write(struct.pack('<I', count))
        for v in all_vertices:
            f.write(struct.pack('<3f', *v))
    print(f"Wrote {len(mesh_brushes)} meshes with {sum(vert_counts)} vertices to {output_path}")
if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: compile_map.py <name>")
        sys.exit(0)
    brushes=parse_map(f'{sys.argv[1]}.map')
    write_mesh(brushes,f'../build/levels/{sys.argv[1]}.map_geo')
