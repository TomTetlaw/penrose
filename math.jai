
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
  _11, _12, _13, _14: float;
  _21, _22, _23, _24: float;
  _31, _32, _33, _34: float;
  _41, _42, _43, _44: float;
  #place _11; m: [4][4] float;
}

operator * :: (a: Mat4x4F32, b: Mat4x4F32) -> Mat4x4F32 { return mul_mat4x4f32(a, b); }
mul_mat4x4f32 :: (a: Mat4x4F32, b: Mat4x4F32) -> Mat4x4F32
{
  result: Mat4x4F32 = ---;
  result._11 = a._11*b._11 + a._12*b._21 + a._13*b._31 + a._14*b._41;
  result._21 = a._21*b._11 + a._22*b._21 + a._23*b._31 + a._24*b._41;
  result._31 = a._31*b._11 + a._32*b._21 + a._33*b._31 + a._34*b._41;
  result._41 = a._41*b._11 + a._42*b._21 + a._43*b._31 + a._44*b._41;
  result._12 = a._11*b._12 + a._12*b._22 + a._13*b._32 + a._14*b._42;
  result._22 = a._21*b._12 + a._22*b._22 + a._23*b._32 + a._24*b._42;
  result._32 = a._31*b._12 + a._32*b._22 + a._33*b._32 + a._34*b._42;
  result._42 = a._41*b._12 + a._42*b._22 + a._43*b._32 + a._44*b._42;
  result._13 = a._11*b._13 + a._12*b._23 + a._13*b._33 + a._14*b._43;
  result._23 = a._21*b._13 + a._22*b._23 + a._23*b._33 + a._24*b._43;
  result._33 = a._31*b._13 + a._32*b._23 + a._33*b._33 + a._34*b._43;
  result._43 = a._41*b._13 + a._42*b._23 + a._43*b._33 + a._44*b._43;
  result._14 = a._11*b._14 + a._12*b._24 + a._13*b._34 + a._14*b._44;
  result._24 = a._21*b._14 + a._22*b._24 + a._23*b._34 + a._24*b._44;
  result._34 = a._31*b._14 + a._32*b._24 + a._33*b._34 + a._34*b._44;
  result._44 = a._41*b._14 + a._42*b._24 + a._43*b._34 + a._44*b._44;
  return result;
}

operator * :: (m: Mat4x4F32, v: Vec4F32) -> Vec4F32 { return mul_mat4x4f32_vec4f32(m, v); }
mul_mat4x4f32_vec4f32 :: (m: Mat4x4F32, v: Vec4F32) -> Vec4F32
{
  result: Vec4F32;
  result.x = v.x*m._11 + v.y*m._12 + v.z*m._13 + v.w*m._14;
  result.y = v.x*m._21 + v.y*m._22 + v.z*m._23 + v.w*m._24;
  result.z = v.x*m._31 + v.y*m._32 + v.z*m._33 + v.w*m._34;
  result.w = v.x*m._41 + v.y*m._42 + v.z*m._43 + v.w*m._44;
  return result;
}

mat4x4f32_identity :: () -> Mat4x4F32
{
  result: Mat4x4F32;
  result.m[0][0] = 1.0;
  result.m[1][1] = 1.0;
  result.m[2][2] = 1.0;
  result.m[3][3] = 1.0;
  return result;
}

mat4x4f32_scale :: (scale: Vec3F32) -> Mat4x4F32
{
  result: Mat4x4F32;
  result.m[0][0] = scale.x;
  result.m[1][1] = scale.y;
  result.m[2][2] = scale.z;
  result.m[3][3] = 1.0;
  return result;
}

mat4x4f32_translate :: (translate: Vec3F32) -> Mat4x4F32
{
  result := mat4x4f32_identity();
  result.m[3][0] = translate.x;
  result.m[3][1] = translate.y;
  result.m[3][2] = translate.z;
  return result;
}

mat4x4f32_rotate :: (a: Vec3F32, turns: float) -> Mat4x4F32
{
  result: Mat4x4F32;
  axis := normalize_v3(a);
  sin_theta := sin_turns(turns);
  cos_theta := cos_turns(turns);
  cos_value := 1.0 - cos_theta;
  result.m[0][0] = (axis.x * axis.x * cos_value) + cos_theta;
  result.m[0][1] = (axis.x * axis.y * cos_value) + (axis.z * sin_theta);
  result.m[0][2] = (axis.x * axis.z * cos_value) - (axis.y * sin_theta);
  result.m[1][0] = (axis.y * axis.x * cos_value) - (axis.z * sin_theta);
  result.m[1][1] = (axis.y * axis.y * cos_value) + cos_theta;
  result.m[1][2] = (axis.y * axis.z * cos_value) + (axis.x * sin_theta);
  result.m[2][0] = (axis.z * axis.x * cos_value) + (axis.y * sin_theta);
  result.m[2][1] = (axis.z * axis.y * cos_value) - (axis.x * sin_theta);
  result.m[2][2] = (axis.z * axis.z * cos_value) + cos_theta;
  result.m[3][3] = 1.0;
  return result;
}

// NOTE(tt): converts from our coordinate system to Vulkan's NDC coordinate system
mat4x4f32_axis_fixup :: () -> Mat4x4F32
{
  result: Mat4x4F32;
  result._12 = -1.0;
  result._23 = 1.0;
  result._31 = -1.0;
  result._44 = 1.0;
  return result;
}

mat4x4f32_perspective :: (fov_turns: float, aspect: float, near: float, far: float) -> Mat4x4F32
{
  tan_theta := tan_turns(fov_turns * .5);
  cot_theta := 1.0 / tan_theta;
  denom := 1.0 / (far - near);
  result: Mat4x4F32;
  result._11 = cot_theta / aspect;
  result._22 = cot_theta;
  result._33 = far * denom;
  result._34 = -near * far * denom;
  result._43 = -1.0;
  return result * mat4x4f32_axis_fixup();
}

mat4x4f32_ortho :: (left: float, right: float, bottom: float, top: float, near: float, far: float) -> Mat4x4F32
{
  result: Mat4x4F32;
  result.m[0][0] = 2.0 / (right - left);
  result.m[1][1] = 2.0 / (top - bottom);
  result.m[2][2] = 2.0 / (far - near);
  result.m[3][0] = (left + right) / (left - right);
  result.m[3][1] = (bottom + top) / (bottom - top);
  result.m[3][2] = (near + far) / (near - far);
  result.m[3][3] = 1.0;
  return mat4x4f32_axis_fixup() * result;
}
