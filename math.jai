
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

Vec2F32 :: V2(float);
Vec3F32 :: V3(float);
Vec4F32 :: V4(float);
Vec2F64 :: V2(float64);
Vec3F64 :: V3(float64);
Vec4F64 :: V4(float64);

Vec2S32 :: V2(s32);
Vec3S32 :: V3(s32);
Vec4S32 :: V4(s32);
Vec2S64 :: V2(int);
Vec3S64 :: V3(int);
Vec4S64 :: V4(int);

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