
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

PlaneF32 :: struct
{
  origin: Vec3F32;
  normal: Vec3F32;
}

RayF32 :: struct
{
  origin: Vec3F32;
  dir: Vec3F32;
}

rayf32_from_mouse :: (mouse_pos: Vec2F32, screen_size: Vec2F32, view: Mat4x4F32, projection: Mat4x4F32) -> RayF32
{
  ndc_x := (mouse_pos.x / screen_size.x) * 2.0 - 1.0;
  ndc_y := 1.0 - (mouse_pos.y / screen_size.y) * 2.0;
  inv := mat4x4f32_inverse(projection * view);
  near := inv * v4f32(ndc_x, ndc_y, -1.0, 1.0);
  far := inv * v4f32(ndc_x, ndc_y, 1.0, 1.0);
  near_world := near.xyz / near.w;
  far_world := far.xyz / far.w;
  result: RayF32;
  result.origin = near_world;
  result.dir = normalize_v3(far_world - near_world);
  return result;
}

intersect_rayf32_planef32 :: (ray: RayF32, plane: PlaneF32) -> B8, Vec3F32
{
  denom := dot_v3(plane.normal, ray.dir);
  if abs(denom) < .000001
  {
    return false, v3f32(0.0);
  }
  t := dot_v3(plane.origin - ray.origin, plane.normal) / denom;
  if t < 0
  {
    return false, v3f32(0.0);
  }
  hit_point := ray.origin + ray.dir * t;
  return true, hit_point;
}

intersection_rayf32_boundsf32 :: (ray: RayF32, bounds_min: Vec3F32, bounds_max: Vec3F32) -> B8, F32
{
  inv_dir := v3f32(1.0) / ray.dir;
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

V2 :: struct($T: Type)
{
  x: T;
  y: T;
  #place x; f: [2] T;
}

V3 :: struct($T: Type)
{
  x: T;
  y: T;
  z: T;
  #place x; f: [3] T;
  #place x; xy: V2(T);
}

V4 :: struct($T: Type)
{
  x: T;
  y: T;
  z: T;
  w: T;
  #place x; f: [4] T;
  #place x; xy: V2(T);
  #place z; zw: V2(T);
  #place x; xyz: V3(T);
}

v2 :: (x: $T, y: T) -> V2(T) { return .{x, y}; }
v3 :: (x: $T, y: T, z: T) -> V3(T) { return .{x, y, z}; }
v4 :: (x: $T, y: T, z: T, w: T) -> V4(T) { return .{x, y, z, w}; }

Vec2F32 :: V2(float32);
Vec3F32 :: V3(float32);
Vec4F32 :: V4(float32);
Vec2F64 :: V2(float64);
Vec3F64 :: V3(float64);
Vec4F64 :: V4(float64);
Vec2S32 :: V2(s32);
Vec3S32 :: V3(s32);
Vec4S32 :: V4(s32);
Vec2U32 :: V2(u32);
Vec3U32 :: V3(u32);
Vec4U32 :: V4(u32);
Vec2S64 :: V2(s64);
Vec3S64 :: V3(s64);
Vec4S64 :: V4(s64);
Vec2U64 :: V2(u64);
Vec3U64 :: V3(u64);
Vec4U64 :: V4(u64);

v2f32 :: (x: float32, y: float32) -> Vec2F32 { return .{x, y}; }
v2f32 :: (x: float32) -> Vec2F32 { return .{x, x}; }
v2s32 :: (x: s32, y: s32) -> Vec2S32 { return .{x, y}; }
v2s32 :: (x: s32) -> Vec2S32 { return .{x, x}; }
v2s64 :: (x: s64, y: s64) -> Vec2S64 { return .{x, y}; }
v2s64 :: (x: s64) -> Vec2S64 { return .{x, x}; }
v2u32 :: (x: u32, y: u32) -> Vec2U32 { return .{x, y}; }
v2u32 :: (x: u32) -> Vec2U32 { return .{x, x}; }
v2u64 :: (x: u64, y: u64) -> Vec2U64 { return .{x, y}; }
v2u64 :: (x: u64) -> Vec2U64 { return .{x, x}; }
v2f64 :: (x: float64, y: float64) -> Vec2F64 { return .{x, y}; }
v2f64 :: (x: float64) -> Vec2F64 { return .{x, x}; }

v3f32 :: (x: float32, y: float32, z: float32) -> Vec3F32 { return .{x, y, z}; }
v3f32 :: (x: float32) -> Vec3F32 { return .{x, x, x}; }
v3f64 :: (x: float64, y: float64, z: float64) -> Vec3F64 { return .{x, y, z}; }
v3f64 :: (x: float64) -> Vec3F64 { return .{x, x, x}; }
v3s32 :: (x: s32, y: s32, z: s32) -> Vec3S32 { return .{x, y, z}; }
v3s32 :: (x: s32) -> Vec3S32 { return .{x, x, x}; }
v3u32 :: (x: u32, y: u32, z: u32) -> Vec3U32 { return .{x, y, z}; }
v3u32 :: (x: u32) -> Vec3U32 { return .{x, x, x}; }
v3s64 :: (x: s64, y: s64, z: s64) -> Vec3S64 { return .{x, y, z}; }
v3s64 :: (x: s64) -> Vec3S64 { return .{x, x, x}; }
v3u64 :: (x: u64, y: u64, z: u64) -> Vec3U64 { return .{x, y, z}; }
v3u64 :: (x: u64) -> Vec3U64 { return .{x, x, x}; }
v3f32 :: (xy: Vec2F32, z: float32) -> Vec3F32 { return .{xy.x, xy.y, z}; }
v3f64 :: (xy: Vec2F64, z: float64) -> Vec3F64 { return .{xy.x, xy.y, z}; }
v3s32 :: (xy: Vec2S32, z: s32) -> Vec3S32 { return .{xy.x, xy.y, z}; }
v3s64 :: (xy: Vec2S64, z: s64) -> Vec3S64 { return .{xy.x, xy.y, z}; }
v3u32 :: (xy: Vec2U32, z: u32) -> Vec3U32 { return .{xy.x, xy.y, z}; }
v3u64 :: (xy: Vec2U64, z: u64) -> Vec3U64 { return .{xy.x, xy.y, z}; }

v4f64 :: (x: float64, y: float64, z: float64, w: float64) -> Vec4F64 { return .{x, y, z, w}; }
v4f64 :: (x: float64) -> Vec4F64 { return .{x, x, x, x}; }
v4f32 :: (x: float32, y: float32, z: float32, w: float32) -> Vec4F32 { return .{x, y, z, w}; }
v4f32 :: (x: float32) -> Vec4F32 { return .{x, x, x, x}; }
v4s32 :: (x: s32, y: s32, z: s32, w: s32) -> Vec4S32 { return .{x, y, z, w}; }
v4s32 :: (x: s32) -> Vec4S32 { return .{x, x, x, x}; }
v4s64 :: (x: s64, y: s64, z: s64, w: s64) -> Vec4S64 { return .{x, y, z, w}; }
v4s64 :: (x: s64) -> Vec4S64 { return .{x, x, x, x}; }
v4u32 :: (x: u32, y: u32, z: u32, w: u32) -> Vec4U32 { return .{x, y, z, w}; }
v4u32 :: (x: u32) -> Vec4U32 { return .{x, x, x, x}; }
v4u64 :: (x: u64, y: u64, z: u64, w: u64) -> Vec4U64 { return .{x, y, z, w}; }
v4u64 :: (x: u64) -> Vec4U64 { return .{x, x, x, x}; }
v4f32 :: (xyz: Vec3F32, w: float32) -> Vec4F32 { return .{xyz.x, xyz.y, xyz.z, w}; }
v4f32 :: (xy: Vec2F32, zw: Vec2F32) -> Vec4F32 { return .{xy.x, xy.y, zw.x, zw.y}; }
v4f32 :: (xy: Vec2F32, z: F32, w: F32) -> Vec4F32 { return .{xy.x, xy.y, z, z}; }
v4f64 :: (xyz: Vec3F64, w: float64) -> Vec4F64 { return .{xyz.x, xyz.y, xyz.z, w}; }
v4f64 :: (xy: Vec2F64, zw: Vec2F64) -> Vec4F64 { return .{xy.x, xy.y, zw.x, zw.y}; }
v4s32 :: (xyz: Vec3S32, w: s32) -> Vec4S32 { return .{xyz.x, xyz.y, xyz.z, w}; }
v4s32 :: (xy: Vec2S32, zw: Vec2S32) -> Vec4S32 { return .{xy.x, xy.y, zw.x, zw.y}; }
v4s64 :: (xyz: Vec3S64, w: s64) -> Vec4S64 { return .{xyz.x, xyz.y, xyz.z, w}; }
v4s64 :: (xy: Vec2S64, zw: Vec2S64) -> Vec4S64 { return .{xy.x, xy.y, zw.x, zw.y}; }
v4u32 :: (xyz: Vec3U32, w: u32) -> Vec4U32 { return .{xyz.x, xyz.y, xyz.z, w}; }
v4u32 :: (xy: Vec2U32, zw: Vec2U32) -> Vec4U32 { return .{xy.x, xy.y, zw.x, zw.y}; }
v4u64 :: (xyz: Vec3U64, w: u64) -> Vec4U64 { return .{xyz.x, xyz.y, xyz.z, w}; }
v4u64 :: (xy: Vec2U64, zw: Vec2U64) -> Vec4U64 { return .{xy.x, xy.y, zw.x, zw.y}; }

operator *[] :: (v: *V2($T), i: int)   -> *T                {return *v.f[i];}
operator *[] :: (v: *V3($T), i: int)   -> *T                {return *v.f[i];}
operator *[] :: (v: *V4($T), i: int)   -> *T                {return *v.f[i];}
operator +   :: (a:  V2($T), b: V2(T)) ->  V2(T) #symmetric {return V2(T).{a.x + b.x, a.y + b.y};}
operator -   :: (a:  V2($T), b: V2(T)) ->  V2(T) #symmetric {return V2(T).{a.x - b.x, a.y - b.y};}
operator *   :: (a:  V2($T), b: V2(T)) ->  V2(T) #symmetric {return V2(T).{a.x * b.x, a.y * b.y};}
operator /   :: (a:  V2($T), b: V2(T)) ->  V2(T)            {return V2(T).{a.x / b.x, a.y / b.y};}
operator +   :: (a:  V2($T), b: T)     ->  V2(T) #symmetric {return V2(T).{a.x + b,   a.y + b  };}
operator -   :: (a:  V2($T), b: T)     ->  V2(T) #symmetric {return V2(T).{a.x - b,   a.y - b  };}
operator *   :: (a:  V2($T), b: T)     ->  V2(T) #symmetric {return V2(T).{a.x * b,   a.y * b  };}
operator /   :: (a:  V2($T), b: T)     ->  V2(T)            {return V2(T).{a.x / b,   a.y / b  };}
operator +   :: (a:  V3($T), b: V3(T)) ->  V3(T) #symmetric {return V3(T).{a.x + b.x, a.y + b.y, a.z + b.z};}
operator -   :: (a:  V3($T), b: V3(T)) ->  V3(T) #symmetric {return V3(T).{a.x - b.x, a.y - b.y, a.z - b.z};}
operator *   :: (a:  V3($T), b: V3(T)) ->  V3(T) #symmetric {return V3(T).{a.x * b.x, a.y * b.y, a.z * b.z};}
operator /   :: (a:  V3($T), b: V3(T)) ->  V3(T)            {return V3(T).{a.x / b.x, a.y / b.y, a.z / b.z};}
operator +   :: (a:  V3($T), b: T)     ->  V3(T) #symmetric {return V3(T).{a.x + b,   a.y + b  , a.z + b  };}
operator -   :: (a:  V3($T), b: T)     ->  V3(T) #symmetric {return V3(T).{a.x - b,   a.y - b  , a.z - b  };}
operator *   :: (a:  V3($T), b: T)     ->  V3(T) #symmetric {return V3(T).{a.x * b,   a.y * b  , a.z * b  };}
operator /   :: (a:  V3($T), b: T)     ->  V3(T)            {return V3(T).{a.x / b,   a.y / b  , a.z / b  };}
operator +   :: (a:  V4($T), b: V4(T)) ->  V4(T) #symmetric {return V4(T).{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};}
operator -   :: (a:  V4($T), b: V4(T)) ->  V4(T) #symmetric {return V4(T).{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};}
operator *   :: (a:  V4($T), b: V4(T)) ->  V4(T) #symmetric {return V4(T).{a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};}
operator /   :: (a:  V4($T), b: V4(T)) ->  V4(T)            {return V4(T).{a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};}
operator +   :: (a:  V4($T), b: T)     ->  V4(T) #symmetric {return V4(T).{a.x + b,   a.y + b  , a.z + b  , a.w + b  };}
operator -   :: (a:  V4($T), b: T)     ->  V4(T) #symmetric {return V4(T).{a.x - b,   a.y - b  , a.z - b  , a.w - b  };}
operator *   :: (a:  V4($T), b: T)     ->  V4(T) #symmetric {return V4(T).{a.x * b,   a.y * b  , a.z * b  , a.w * b  };}
operator /   :: (a:  V4($T), b: T)     ->  V4(T)            {return V4(T).{a.x / b,   a.y / b  , a.z / b  , a.w / b  };}
operator -   :: (v: V2($T))            ->  V2(T)            {return V2(T).{-v.x, -v.y};}
operator -   :: (v: V3($T))            ->  V3(T)            {return V3(T).{-v.x, -v.y, -v.z};}
operator -   :: (v: V4($T))            ->  V4(T)            {return V4(T).{-v.x, -v.y, -v.z, -v.w};}

snap_v3 :: (v: V3($T), step: T) -> V3(T)
{
  result: V3(T);
  result.x = floor(v.x / step) * step;
  result.y = floor(v.y / step) * step;
  result.z = floor(v.z / step) * step;
  return result;
}

snap_v3 :: (v: *V3($T), step: T)
{
  v.x = floor(v.x / step) * step;
  v.y = floor(v.y / step) * step;
  v.z = floor(v.z / step) * step;
}

clamp_v2 :: (a: V2($T), min: V2(T), max: V2(T)) -> V2(T) { return v2(clamp(a.x, min.x, max.x), clamp(a.y, min.y, max.y)); }
clamp_v3 :: (a: V3($T), min: V3(T), max: V3(T)) -> V3(T) { return v3(clamp(a.x, min.x, max.x), clamp(a.y, min.y, max.y), clamp(a.z, min.z, max.z)); }
clamp_v4 :: (a: V4($T), min: V4(T), max: V4(T)) -> V4(T) { return v4(clamp(a.x, min.x, max.x), clamp(a.y, min.y, max.y), clamp(a.z, min.z, max.z), clamp(a.w, min.w, max.w)); }

min_v2 :: (a: V2($T), b: V2(T)) -> V2(T) { return v2(min(a.x, b.x), min(a.y, b.y)); }
min_v3 :: (a: V3($T), b: V3(T)) -> V3(T) { return v3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
min_v4 :: (a: V4($T), b: V4(T)) -> V4(T) { return v4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w)); }

max_v2 :: (a: V2($T), b: V2(T)) -> V2(T) { return v2(max(a.x, b.x), max(a.y, b.y)); }
max_v3 :: (a: V3($T), b: V3(T)) -> V3(T) { return v3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
max_v4 :: (a: V4($T), b: V4(T)) -> V4(T) { return v4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w)); }

any_v2 :: (v: V2($T)) -> B8 { return v.x > 0.0 || v.y > 0.0; }
any_v3 :: (v: V3($T)) -> B8 { return v.x > 0.0 || v.y > 0.0 || v.z > 0.0; }
any_v3 :: (v: V4($T)) -> B8 { return v.x > 0.0 || v.y > 0.0 || v.z > 0.0 || v.w > 0.0; }

length_v2 :: (v: V2($T)) -> T { return sqrt(v.x*v.x + v.y*v.y); }
length_v3 :: (v: V3($T)) -> T { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
length_v4 :: (v: V4($T)) -> T { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w); }

dot_v2 :: (a: V2($T), b: V2(T)) -> T { return a.x*b.x + a.y*b.y; }
dot_v3 :: (a: V3($T), b: V3(T)) -> T { return a.x*b.x + a.y*b.y + a.z*b.z; }
dot_v4 :: (a: V4($T), b: V4(T)) -> T { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }

cross_v3 :: (a: V3($T), b: V3(T)) -> V3(T)
{
  result: V3(T);
  result.x = a.y*b.z - a.z*b.y;
  result.y = a.z*b.x - a.x*b.z;
  result.z = a.x*b.y - a.y*b.x;
  return result;
}

normalize_v2 :: (v: V2($T)) -> V2(T) { return v * (1.0 / length_v2(v)); }
normalize_v3 :: (v: V3($T)) -> V3(T) { return v * (1.0 / length_v3(v)); }
normalize_v4 :: (v: V4($T)) -> V4(T) { return v * (1.0 / length_v4(v)); }

rotate_v3 :: (v: Vec3F32, r: Vec3F32) -> Vec3F32
{
  m := mat4x4f32_rotate(v3f32(1, 0, 0), r.x) *
    mat4x4f32_rotate(v3f32(0, 1, 0), r.y) *
    mat4x4f32_rotate(v3f32(0, 0, 1), r.z);
  result := m * v4f32(v, 1.0);
  return result.xyz;
}

rotate_v3 :: (m: Mat4x4F32, v: Vec3F32) -> Vec3F32
{
  result := m * v4f32(v, 1.0);
  return result.xyz;
}

dir_from_rotation :: (angles: Vec3F32) -> Vec3F32
{
  rot := mat4x4f32_rotate(v3f32(1, 0, 0), angles.x) *
    mat4x4f32_rotate(v3f32(0, 1, 0), angles.y) *
    mat4x4f32_rotate(v3f32(0, 0, 1), angles.z);
  result := rot * v4f32(1, 0, 0, 1);
  return result.xyz;
}

lerp_v2 :: (a: V2($T), b: V2(T), t: T) -> V2(T)
{
  return v2f32(lerp(a.x, b.x, t), lerp(a.y, b.y, t));
}

lerp_v3 :: (a: V3($T), b: V3(T), t: T) -> V3(T)
{
  return v3f32
  (
    lerp(a.x, b.x, t),
    lerp(a.y, b.y, t),
    lerp(a.z, b.z, t)
  );
}

lerp_v4 :: (a: V4($T), b: V4(T), t: T) -> V4(T)
{
  return v4f32
  (
    lerp(a.x, b.x, t),
    lerp(a.y, b.y, t),
    lerp(a.z, b.z, t),
    lerp(a.w, b.w, t)
  );
}

Mat4x4F32 :: struct
{
  _00, _01, _02, _03: float;
  _10, _11, _12, _13: float;
  _20, _21, _22, _23: float;
  _30, _31, _32, _33: float;
  #place _00; m: [4][4] float;
}

mat4x4f32_transform_position :: (p: Vec3F32, m: Mat4x4F32) -> Vec3F32
{
  result := mul_mat4x4f32_vec4f32(m, v4f32(p, 1));
  return result.xyz;
}

operator * :: (a: Mat4x4F32, b: Mat4x4F32) -> Mat4x4F32 { return mul_mat4x4f32(a, b); }
mul_mat4x4f32 :: (a: Mat4x4F32, b: Mat4x4F32) -> Mat4x4F32
{
  result: Mat4x4F32;
  for 0..3
  {
    row := v4f32(a.m[it][0], a.m[it][1], a.m[it][2], a.m[it][3]);
    result.m[it][0] = dot_v4(row, v4f32(b._00, b._10, b._20, b._30));
    result.m[it][1] = dot_v4(row, v4f32(b._01, b._11, b._21, b._31));
    result.m[it][2] = dot_v4(row, v4f32(b._02, b._12, b._22, b._32));
    result.m[it][3] = dot_v4(row, v4f32(b._03, b._13, b._23, b._33));
  }
  return result;
}

operator * :: (m: Mat4x4F32, v: Vec4F32) -> Vec4F32 { return mul_mat4x4f32_vec4f32(m, v); }
mul_mat4x4f32_vec4f32 :: (m: Mat4x4F32, v: Vec4F32) -> Vec4F32
{
  result: Vec4F32;
  result.x = v.x*m._00 + v.y*m._01 + v.z*m._02 + v.w*m._03;
  result.y = v.x*m._10 + v.y*m._11 + v.z*m._12 + v.w*m._13;
  result.z = v.x*m._20 + v.y*m._21 + v.z*m._22 + v.w*m._23;
  result.w = v.x*m._30 + v.y*m._31 + v.z*m._32 + v.w*m._33;
  return result;
}

mat4x4f32_identity :: () -> Mat4x4F32
{
  result: Mat4x4F32;
  result._00 = 1.0;
  result._11 = 1.0;
  result._22 = 1.0;
  result._33 = 1.0;
  return result;
}

mat4x4f32_scale :: (scale: Vec3F32) -> Mat4x4F32
{
  result: Mat4x4F32;
  result._00 = scale.x;
  result._11 = scale.y;
  result._22 = scale.z;
  result._33 = 1.0;
  return result;
}

mat4x4f32_translate :: (translate: Vec3F32) -> Mat4x4F32
{
  result := mat4x4f32_identity();
  result._03 = translate.x;
  result._13 = translate.y;
  result._23 = translate.z;
  return result;
}

mat4x4f32_rotate :: (a: Vec3F32, turns: float) -> Mat4x4F32
{
  result: Mat4x4F32;
  axis := normalize_v3(a);
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

mat4x4f32_transform :: (p: Vec3F32, r: Vec3F32, s: Vec3F32) -> Mat4x4F32
{
  result :=
    mat4x4f32_translate(p) *
    mat4x4f32_rotate(v3f32(1.0, 0.0, 0.0), r.x) *
    mat4x4f32_rotate(v3f32(0.0, 1.0, 0.0), r.y) *
    mat4x4f32_rotate(v3f32(0.0, 0.0, 1.0), r.z) *
    mat4x4f32_scale(s);
  return result;
}

mat4x4f32_transpose :: (m: *Mat4x4F32)
{
  for row: 0..3
  {
    for col: 0..3
    {
      m.m[row][col], m.m[col][row] = m.m[col][row], m.m[row][col];
    }
  }
}

mat4x4f32_perspective :: (fov_turns: float, aspect: float, near: float, far: float) -> Mat4x4F32
{
  tan_theta := tan_turns(fov_turns * .5);
  cot_theta := 1.0 / tan_theta;
  denom := 1.0 / (near - far);

  result: Mat4x4F32;
  result._00 = cot_theta / aspect;
  result._11 = cot_theta;
  result._22 = near * denom;
  result._23 = far * near * denom;
  result._32 = 1.0;

  fixup: Mat4x4F32;
  fixup._01 = -1.0;
  fixup._12 = 1.0;
  fixup._20 = 1.0;
  fixup._33 = -1.0;

  return result * fixup;
}

mat4x4f32_ortho :: (left: float, right: float, bottom: float, top: float, near: float, far: float) -> Mat4x4F32
{
  result: Mat4x4F32;
  result._00 = 2.0 / (right - left);
  result._11 = 2.0 / (top - bottom);
  result._22 = 2.0 / (far - near);
  result._30 = (left + right) / (left - right);
  result._31 = (bottom + top) / (bottom - top);
  result._32 = (near + far) / (near - far);
  result._33 = 1.0;
  return result;
}

mat4x4f32_inverse :: (m: Mat4x4F32) -> Mat4x4F32, bool
{
  a := v3f32(m._00, m._10, m._20);
  b := v3f32(m._01, m._11, m._21);
  c := v3f32(m._02, m._12, m._22);
  d := v3f32(m._03, m._13, m._23);
  x := m._30;
  y := m._31;
  z := m._32;
  w := m._33;
  s := cross_v3(a, b);
  t := cross_v3(c, d);
  u := a * y - b * x;
  v := c * w - d * z;
  det := dot_v3(s, v) + dot_v3(t, u);
  inv_det := 1.0 / det;
  s *= inv_det;
  t *= inv_det;
  u *= inv_det;
  v *= inv_det;
  r0 := cross_v3(b, v) + t * y;
  r1 := cross_v3(v, a) - t * x;
  r2 := cross_v3(d, u) + s * w;
  r3 := cross_v3(u, c) - s * z;
  result := Mat4x4F32.{
    r0.x, r0.y, r0.z, -dot_v3(b, t),
    r1.x, r1.y, r1.z,  dot_v3(a, t),
    r2.x, r2.y, r2.z, -dot_v3(d, s),
    r3.x, r3.y, r3.z,  dot_v3(c, s)
  };
  epsilon := 0.0001;
  return result, abs(det) < epsilon;
}