
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
  #place x; xyz: V3(T);
}

Vec2F32 :: V2(float32);
Vec3F32 :: V3(float32);
Vec4F32 :: V4(float32);
Vec2F64 :: V2(float64);
Vec3F64 :: V3(float64);
Vec4F64 :: V4(float64);

Vec2S32 :: V2(s32);
Vec3S32 :: V3(s32);
Vec4S32 :: V4(s32);
Vec2S64 :: V2(s64);
Vec3S64 :: V3(s64);
Vec4S64 :: V4(s64);

Vec2U32 :: V2(u32);
Vec3U32 :: V3(u32);
Vec4U32 :: V4(u32);
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

Mat4x4F32 :: struct
{
  _00, _01, _02, _03: float;
  _10, _11, _12, _13: float;
  _20, _21, _22, _23: float;
  _30, _31, _32, _33: float;
  #place _00; m: [4][4] float;
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
  result._30 = translate.x;
  result._31 = translate.y;
  result._32 = translate.z;
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
  denom := 1.0 / (far - near);
  result: Mat4x4F32;
  result._00 = -cot_theta / aspect;
  result._11 = cot_theta;
  result._22 = far * denom;
  result._23 = -near * far * denom;
  result._32 = 1.0;

  fixup: Mat4x4F32;
  fixup._01 = -1;
  fixup._12 = 1;
  fixup._20 = 1;
  fixup._33 = 1;

  return fixup * result;
}

#run
{
  m := mat4x4f32_perspective(0.125, 4.0 / 3.0, 0.1, 1000.0);
  print("m[0] = %\n", m.m[0]);
  print("m[1] = %\n", m.m[1]);
  print("m[2] = %\n", m.m[2]);
  print("m[3] = %\n", m.m[3]);

  v := Vec3F32.[
    v3f32(15.0, -.5, -.5),
    v3f32(15.0,  .5, -.5),
    v3f32(15.0, -.5,  .5),
    v3f32(15.0,  .5,  .5),
  ];

  for v
  {
    print("%\n", m * v4f32(it, 1));
  }
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
