
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

length :: (v: V2($T)) -> T { return sqrt(v.x*v.x + v.y*v.y); }
length :: (v: V3($T)) -> T { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
length :: (v: V4($T)) -> T { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w); }

dot :: (a: V2($T), b: V2(T)) -> T { return a.x*b.x + a.y*b.y; }
dot :: (a: V3($T), b: V3(T)) -> T { return a.x*b.x + a.y*b.y + a.z*b.z; }
dot :: (a: V4($T), b: V4(T)) -> T { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }

normalize :: (v: V2($T)) -> V2(T) { return v * (1.0 / length(v)); }
normalize :: (v: V3($T)) -> V3(T) { return v * (1.0 / length(v)); }
normalize :: (v: V4($T)) -> V4(T) { return v * (1.0 / length(v)); }

Mat4x4F32 :: struct
{
  m: [4][4] float;
}

mat4x4f32_axis_fixup :: Mat4x4F32.{
  m[0][2] = -1.0, // NOTE(tt): X -> NDC -Z
  m[1][0] = -1.0, // NOTE(tt): Y -> NDC -X
  m[2][1] = -1.0, // NOTE(tt): X -> NDC -Y
  m[3][3] = 1.0
};

operator * :: (a: Mat4x4F32, b: Mat4x4F32) -> Mat4x4F32 { return mul_mat4x4f32(a, b); }
mul_mat4x4f32 :: (a: Mat4x4F32, b: Mat4x4F32) -> Mat4x4F32
{
  result: Mat4x4F32;
  for row: 0..3
  {
    for col: 0..3
    {
      result.m[col][row] =
        a.m[0][row] * b.m[col][0] +
        a.m[1][row] * b.m[col][1] +
        a.m[2][row] * b.m[col][2] +
        a.m[3][row] * b.m[col][3];
    }
  }
  return result;
}

operator * :: (m: Mat4x4F32, v: Vec4F32) -> Vec4F32 { return mul_mat4x4f32_vec4f32(m, v); }
mul_mat4x4f32_vec4f32 :: (m: Mat4x4F32, v: Vec4F32) -> Vec4F32
{
  result: Vec4F32;
  for row: 0..3
  {
    result[row] =
      m.m[0][row] * v.x +
      m.m[1][row] * v.y +
      m.m[2][row] * v.z +
      m.m[3][row] * v.w;
  }
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
  result: Mat4x4F32;
  result.m[3][0] = translate.x;
  result.m[3][1] = translate.y;
  result.m[3][2] = translate.z;
  result.m[3][3] = 1.0;
  return result;
}

mat4x4f32_rotate :: (a: Vec3F32, turns: float) -> Mat4x4F32
{
  result: Mat4x4F32;
  axis := normalize(a);
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

mat4x4f32_perspective :: (fov_turns: float, aspect: float, near: float, far: float) -> Mat4x4F32
{
  tan_theta := tan_turns(fov_turns * .5);
  cot_theta := 1.0 / tan_theta;
  denom := 1.0 / (near - far);
  result: Mat4x4F32;
  result.m[0][0] = cot_theta;
  result.m[1][1] = aspect * cot_theta;
  result.m[2][2] = -(near + far) * denom;
  result.m[2][3] = 1.0;
  result.m[3][2] = 2.0 * near * far * denom;
  return mat4x4f32_axis_fixup * result;
}

mat4x4f32_ortho :: (left: float, right: float, bottom: float, top: float, near: float, far: float) -> Mat4x4F32
{
  result: Mat4x4F32;
  result.m[0][0] = 2.0 / (right - left);
  result.m[1][1] = 2.0 / (top - bottom);
  result.m[2][2] = 2.0 / (far - near);
  result.m[3][3] = 1.0;
  result.m[3][0] = (left + right) / (left - right);
  result.m[3][1] = (bottom + top) / (bottom - top);
  result.m[3][2] = (near + far) / (near - far);
  return mat4x4f32_axis_fixup * result;
}
