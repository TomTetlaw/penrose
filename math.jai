
DegreesFromTurns :: 360.0;
TurnsFromDegrees :: 1.0 / 360.0;
RadiansFromTurns :: TAU;
TurnsFromRadians :: 1.0 / TAU;
sin_turns :: (t: float) -> float {return sin(t * RadiansFromTurns);}
cos_turns :: (t: float) -> float {return cos(t * RadiansFromTurns);}
tan_turns :: (t: float) -> float {return tan(t * RadiansFromTurns);}

approach :: (current: float, goal: float, duration: float, delta: float) -> float
{
  decay :: 4.6;
  return goal + (current - goal) * exp(-decay * delta * (1.0 / duration));
}

Plane :: struct
{
  normal: V3;
  d: float;
}

plane_from_points :: (p0: V3, p1: V3, p2: V3) -> Plane
{
  result: Plane;
  result.normal = normalize(cross(p1 - p0, p2 - p0));
  result.d = -dot(result.normal, p0);
  return result;
}

point_inside_planes :: (p: V3, planes: [] Plane) -> bool
{
  eps := .00001;
  for planes
  {
    if dot(it.normal, p) + it.d > eps
    {
      return false;
    }
  }
  return true;
}

intersect_planes :: (p0: Plane, p1: Plane, p2: Plane) -> bool, V3
{
  n0, n1, n2 := p0.normal, p1.normal, p2.normal;
  d0, d1, d2 := -p0.d, -p1.d, -p2.d;
  denom := dot(n0, cross(n1, n2));
  if abs(denom) < 0.00001
  {
    return false, v3(0.0);
  }
  point := (cross(n1, n2) * d0 + cross(n2, n0) * d1 + cross(n0, n1) * d2) / denom;
  return true, point;
}

Ray :: struct
{
  origin: V3;
  dir: V3;
}

ray_from_mouse :: (mouse_pos: V2, screen_size: V2, view: Mat4, projection: Mat4) -> Ray
{
  ndc_x := (mouse_pos.x / screen_size.x) * 2.0 - 1.0;
  ndc_y := 1.0 - (mouse_pos.y / screen_size.y) * 2.0;
  inv := mat_inverse(projection * view);
  near := inv * v4(ndc_x, ndc_y, -1.0, 1.0);
  far := inv * v4(ndc_x, ndc_y, 1.0, 1.0);
  near_world := near.xyz / near.w;
  far_world := far.xyz / far.w;
  result: Ray;
  result.origin = near_world;
  result.dir = normalize(far_world - near_world);
  return result;
}

intersect_ray_plane :: (ray: Ray, plane: Plane) -> bool, V3
{
  denom := dot(plane.normal, ray.dir);
  if abs(denom) < 0.000001
  {
    return false, v3(0.0);
  }
  t := -(dot(plane.normal, ray.origin) + plane.d) / denom;
  if t < 0
  {
    return false, v3(0.0);
  }
  hit_point := ray.origin + ray.dir * t;
  return true, hit_point;
}

Edge :: struct
{
  a: V3;
  b: V3;
}

intersect_edge_plane :: (edge: Edge, plane: Plane) -> bool, V3
{
  ray := Ray.{
    edge.a, normalize(edge.b - edge.a)
  };
  hit, p := intersect_ray_plane(ray, plane);
  return hit, p;
}

intersection_ray_bounds :: (ray: Ray, bounds_min: V3, bounds_max: V3) -> bool, float
{
  inv_dir := v3(1.0) / ray.dir;
  t1 := (bounds_min.x - ray.origin.x) * inv_dir.x;
  t2 := (bounds_max.x - ray.origin.x) * inv_dir.x;
  tmin := min(t1, t2);
  tmax := max(t1, t2);
  t1 = (bounds_min.y - ray.origin.y) * inv_dir.y;
  t2 = (bounds_max.y - ray.origin.y) * inv_dir.y;
  tmin = max(tmin, min(t1, t2));
  tmax = min(tmax, max(t1, t2));
  t1 = (bounds_min.z - ray.origin.z) * inv_dir.z;
  t2 = (bounds_max.z - ray.origin.z) * inv_dir.z;
  tmin = max(tmin, min(t1, t2));
  tmax = min(tmax, max(t1, t2));
  if tmax >= max(tmin, 0.0)
      return true, max(tmin, 0.0);
  return false, 0.0;
}

V2 :: struct
{
  x: float;
  y: float;
  #place x; f: [2] float;
}

V3 :: struct
{
  x: float;
  y: float;
  z: float;
  #place x; f: [3] float;
  #place x; xy: V2;
}

V4 :: struct
{
  x: float;
  y: float;
  z: float;
  w: float;
  #place x; f: [4] float;
  #place x; xy: V2;
  #place z; zw: V2;
  #place x; xyz: V3;
}

v2 :: (x: float) -> V2 { return .{x, x}; }
v2 :: (x: float, y: float) -> V2 { return .{x, y}; }

v3 :: (x: float) -> V3 { return .{x, x, x}; }
v3 :: (x: float, y: float, z: float) -> V3 { return .{x, y, z}; }
v3 :: (x: V2, z: float) -> V3 { return .{x.x, x.y, z}; }

v4 :: (x: float) -> V4 { return .{x, x, x, x}; }
v4 :: (x: float, y: float, z: float, w: float) -> V4 { return .{x, y, z, w}; }
v4 :: (x: V3, w: float) -> V4 { return .{x.x, x.y, x.z, w}; }

operator *[] :: (v: *V2, i: int) -> *float {return *v.f[i];}
operator *[] :: (v: *V3, i: int) -> *float {return *v.f[i];}
operator *[] :: (v: *V4, i: int) -> *float {return *v.f[i];}
operator - :: (v: V2) -> V2 {return V2.{-v.x, -v.y};}
operator - :: (v: V3) -> V3 {return V3.{-v.x, -v.y, -v.z};}
operator - :: (v: V4) -> V4 {return V4.{-v.x, -v.y, -v.z, -v.w};}
operator + :: (a:  V2, b: V2) -> V2 #symmetric {return V2.{a.x + b.x, a.y + b.y};}
operator - :: (a:  V2, b: V2) -> V2 #symmetric {return V2.{a.x - b.x, a.y - b.y};}
operator * :: (a:  V2, b: V2) -> V2 #symmetric {return V2.{a.x * b.x, a.y * b.y};}
operator / :: (a:  V2, b: V2) -> V2 {return V2.{a.x / b.x, a.y / b.y};}
operator + :: (a:  V2, b: float) -> V2 #symmetric {return V2.{a.x + b, a.y + b};}
operator - :: (a:  V2, b: float) -> V2 #symmetric {return V2.{a.x - b, a.y - b};}
operator * :: (a:  V2, b: float) -> V2 #symmetric {return V2.{a.x * b, a.y * b};}
operator / :: (a:  V2, b: float) -> V2 {return V2.{a.x / b, a.y / b};}
operator + :: (a:  V3, b: V3) -> V3 #symmetric {return V3.{a.x + b.x, a.y + b.y, a.z + b.z};}
operator - :: (a:  V3, b: V3) -> V3 #symmetric {return V3.{a.x - b.x, a.y - b.y, a.z - b.z};}
operator * :: (a:  V3, b: V3) -> V3 #symmetric {return V3.{a.x * b.x, a.y * b.y, a.z * b.z};}
operator / :: (a:  V3, b: V3) -> V3 {return V3.{a.x / b.x, a.y / b.y, a.z / b.z};}
operator + :: (a:  V3, b: float) -> V3 #symmetric {return V3.{a.x + b, a.y + b, a.z + b};}
operator - :: (a:  V3, b: float) -> V3 #symmetric {return V3.{a.x - b, a.y - b, a.z - b};}
operator * :: (a:  V3, b: float) -> V3 #symmetric {return V3.{a.x * b, a.y * b, a.z * b};}
operator / :: (a:  V3, b: float) -> V3 {return V3.{a.x / b, a.y / b, a.z / b};}
operator + :: (a:  V4, b: V4) -> V4 #symmetric {return V4.{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};}
operator - :: (a:  V4, b: V4) -> V4 #symmetric {return V4.{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};}
operator * :: (a:  V4, b: V4) -> V4 #symmetric {return V4.{a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};}
operator / :: (a:  V4, b: V4) -> V4 {return V4.{a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};}
operator + :: (a:  V4, b: float) -> V4 #symmetric {return V4.{a.x + b, a.y + b, a.z + b, a.w + b};}
operator - :: (a:  V4, b: float) -> V4 #symmetric {return V4.{a.x - b, a.y - b, a.z - b, a.w - b};}
operator * :: (a:  V4, b: float) -> V4 #symmetric {return V4.{a.x * b, a.y * b, a.z * b, a.w * b};}
operator / :: (a:  V4, b: float) -> V4 {return V4.{a.x / b, a.y / b, a.z / b, a.w / b};}

snap :: (v: V3, step: float) -> V3
{
  result: V3;
  result.x = floor(v.x / step) * step;
  result.y = floor(v.y / step) * step;
  result.z = floor(v.z / step) * step;
  return result;
}

snap :: (v: *V3, step: float)
{
  v.x = floor(v.x / step) * step;
  v.y = floor(v.y / step) * step;
  v.z = floor(v.z / step) * step;
}

clamp :: (a: V2, min: V2, max: V2) -> V2 { return v2(clamp(a.x, min.x, max.x), clamp(a.y, min.y, max.y)); }
clamp :: (a: V3, min: V3, max: V3) -> V3 { return v3(clamp(a.x, min.x, max.x), clamp(a.y, min.y, max.y), clamp(a.z, min.z, max.z)); }
clamp :: (a: V4, min: V4, max: V4) -> V4 { return v4(clamp(a.x, min.x, max.x), clamp(a.y, min.y, max.y), clamp(a.z, min.z, max.z), clamp(a.w, min.w, max.w)); }

min :: (a: V2, b: V2) -> V2 { return v2(min(a.x, b.x), min(a.y, b.y)); }
min :: (a: V3, b: V3) -> V3 { return v3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
min :: (a: V4, b: V4) -> V4 { return v4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w)); }

max :: (a: V2, b: V2) -> V2 { return v2(max(a.x, b.x), max(a.y, b.y)); }
max :: (a: V3, b: V3) -> V3 { return v3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
max :: (a: V4, b: V4) -> V4 { return v4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w)); }

any :: (v: V2) -> bool { return v.x > 0.0 || v.y > 0.0; }
any :: (v: V3) -> bool { return v.x > 0.0 || v.y > 0.0 || v.z > 0.0; }
any :: (v: V4) -> bool { return v.x > 0.0 || v.y > 0.0 || v.z > 0.0 || v.w > 0.0; }

length :: (v: V2) -> float { return sqrt(v.x*v.x + v.y*v.y); }
length :: (v: V3) -> float { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
length :: (v: V4) -> float { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w); }

dot :: (a: V2, b: V2) -> float { return a.x*b.x + a.y*b.y; }
dot :: (a: V3, b: V3) -> float { return a.x*b.x + a.y*b.y + a.z*b.z; }
dot :: (a: V4, b: V4) -> float { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }

cross :: (a: V3, b: V3) -> V3
{
  result: V3;
  result.x = a.y*b.z - a.z*b.y;
  result.y = a.z*b.x - a.x*b.z;
  result.z = a.x*b.y - a.y*b.x;
  return result;
}

normalize :: (v: V2) -> V2 { return v * (1.0 / length(v)); }
normalize :: (v: V3) -> V3 { return v * (1.0 / length(v)); }
normalize :: (v: V4) -> V4 { return v * (1.0 / length(v)); }

rotate :: (v: V3, r: V3) -> V3
{
  m := mat_rotate(v3(1, 0, 0), r.x) *
    mat_rotate(v3(0, 1, 0), r.y) *
    mat_rotate(v3(0, 0, 1), r.z);
  result := m * v4(v, 1.0);
  return result.xyz;
}

rotate :: (m: Mat4, v: V3) -> V3
{
  result := m * v4(v, 1.0);
  return result.xyz;
}

dir_from_rotation :: (angles: V3) -> V3
{
  rot := mat_rotate(v3(1, 0, 0), angles.x) *
    mat_rotate(v3(0, 1, 0), angles.y) *
    mat_rotate(v3(0, 0, 1), angles.z);
  result := rot * v4(1, 0, 0, 1);
  return result.xyz;
}

lerp :: (a: V2, b: V2, t: float) -> V2
{
  return v2(lerp(a.x, b.x, t), lerp(a.y, b.y, t));
}

lerp :: (a: V3, b: V3, t: float) -> V3
{
  return v3
  (
    lerp(a.x, b.x, t),
    lerp(a.y, b.y, t),
    lerp(a.z, b.z, t)
  );
}

lerp :: (a: V4, b: V4, t: float) -> V4
{
  return v4
  (
    lerp(a.x, b.x, t),
    lerp(a.y, b.y, t),
    lerp(a.z, b.z, t),
    lerp(a.w, b.w, t)
  );
}

Mat4 :: struct
{
  _00, _01, _02, _03: float;
  _10, _11, _12, _13: float;
  _20, _21, _22, _23: float;
  _30, _31, _32, _33: float;
  #place _00; m: [4][4] float;
}

mat_transform_position :: (p: V3, m: Mat4) -> V3
{
  result := mul_mat_V4(m, v4(p, 1));
  return result.xyz;
}

operator * :: (a: Mat4, b: Mat4) -> Mat4 { return mul_mat(a, b); }
mul_mat :: (a: Mat4, b: Mat4) -> Mat4
{
  result: Mat4;
  for 0..3
  {
    row := v4(a.m[it][0], a.m[it][1], a.m[it][2], a.m[it][3]);
    result.m[it][0] = dot(row, v4(b._00, b._10, b._20, b._30));
    result.m[it][1] = dot(row, v4(b._01, b._11, b._21, b._31));
    result.m[it][2] = dot(row, v4(b._02, b._12, b._22, b._32));
    result.m[it][3] = dot(row, v4(b._03, b._13, b._23, b._33));
  }
  return result;
}

operator * :: (m: Mat4, v: V4) -> V4 { return mul_mat_V4(m, v); }
mul_mat_V4 :: (m: Mat4, v: V4) -> V4
{
  result: V4;
  result.x = v.x*m._00 + v.y*m._01 + v.z*m._02 + v.w*m._03;
  result.y = v.x*m._10 + v.y*m._11 + v.z*m._12 + v.w*m._13;
  result.z = v.x*m._20 + v.y*m._21 + v.z*m._22 + v.w*m._23;
  result.w = v.x*m._30 + v.y*m._31 + v.z*m._32 + v.w*m._33;
  return result;
}

mat_identity :: () -> Mat4
{
  result: Mat4;
  result._00 = 1.0;
  result._11 = 1.0;
  result._22 = 1.0;
  result._33 = 1.0;
  return result;
}

mat_scale :: (scale: V3) -> Mat4
{
  result: Mat4;
  result._00 = scale.x;
  result._11 = scale.y;
  result._22 = scale.z;
  result._33 = 1.0;
  return result;
}

mat_translate :: (translate: V3) -> Mat4
{
  result := mat_identity();
  result._03 = translate.x;
  result._13 = translate.y;
  result._23 = translate.z;
  return result;
}

mat_rotate :: (a: V3, turns: float) -> Mat4
{
  result: Mat4;
  axis := normalize(a);
  sin_theta := sin_turns(turns);
  cos_theta := cos_turns(turns);
  cos_value := 1.0 - cos_theta;
  result._00 = (axis.x * axis.x * cos_value) + cos_theta;
  result._01 = (axis.x * axis.y * cos_value) + (axis.z * sin_theta);
  result._02 = (axis.x * axis.z * cos_value) - (axis.y * sin_theta);
  result._10 = (axis.y * axis.x * cos_value) - (axis.z * sin_theta);
  result._11 = (axis.y * axis.y * cos_value) + cos_theta;
  result._12 = (axis.y * axis.z * cos_value) + (axis.x * sin_theta);
  result._20 = (axis.z * axis.x * cos_value) + (axis.y * sin_theta);
  result._21 = (axis.z * axis.y * cos_value) - (axis.x * sin_theta);
  result._22 = (axis.z * axis.z * cos_value) + cos_theta;
  result._33 = 1.0;
  return result;
}

mat_transform :: (p: V3, r: V3, s: V3) -> Mat4
{
  result :=
    mat_translate(p) *
    mat_rotate(v3(1.0, 0.0, 0.0), r.x) *
    mat_rotate(v3(0.0, 1.0, 0.0), r.y) *
    mat_rotate(v3(0.0, 0.0, 1.0), r.z) *
    mat_scale(s);
  return result;
}

mat_transpose :: (m: *Mat4)
{
  for row: 0..3
  {
    for col: 0..3
    {
      m.m[row][col], m.m[col][row] = m.m[col][row], m.m[row][col];
    }
  }
}

mat_perspective :: (fov_turns: float, aspect: float, near: float, far: float) -> Mat4
{
  tan_theta := tan_turns(fov_turns * .5);
  cot_theta := 1.0 / tan_theta;
  denom := 1.0 / (near - far);

  result: Mat4;
  result._00 = cot_theta / aspect;
  result._11 = cot_theta;
  result._22 = near * denom;
  result._23 = far * near * denom;
  result._32 = 1.0;

  fixup: Mat4;
  fixup._01 = -1.0;
  fixup._12 = 1.0;
  fixup._20 = 1.0;
  fixup._33 = -1.0;

  return result * fixup;
}

mat_ortho :: (left: float, right: float, bottom: float, top: float, near: float, far: float) -> Mat4
{
  result: Mat4;
  result._00 = 2.0 / (right - left);
  result._11 = 2.0 / (top - bottom);
  result._22 = 2.0 / (far - near);
  result._30 = (left + right) / (left - right);
  result._31 = (bottom + top) / (bottom - top);
  result._32 = (near + far) / (near - far);
  result._33 = 1.0;
  return result;
}

mat_inverse :: (m: Mat4) -> Mat4, bool
{
  a := v3(m._00, m._10, m._20);
  b := v3(m._01, m._11, m._21);
  c := v3(m._02, m._12, m._22);
  d := v3(m._03, m._13, m._23);
  x := m._30;
  y := m._31;
  z := m._32;
  w := m._33;
  s := cross(a, b);
  t := cross(c, d);
  u := a * y - b * x;
  v := c * w - d * z;
  det := dot(s, v) + dot(t, u);
  inv_det := 1.0 / det;
  s *= inv_det;
  t *= inv_det;
  u *= inv_det;
  v *= inv_det;
  r0 := cross(b, v) + t * y;
  r1 := cross(v, a) - t * x;
  r2 := cross(d, u) + s * w;
  r3 := cross(u, c) - s * z;
  result := Mat4.{
    r0.x, r0.y, r0.z, -dot(b, t),
    r1.x, r1.y, r1.z,  dot(a, t),
    r2.x, r2.y, r2.z, -dot(d, s),
    r3.x, r3.y, r3.z,  dot(c, s)
  };
  epsilon := 0.0001;
  return result, abs(det) < epsilon;
}