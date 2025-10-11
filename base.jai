
F32 :: float32;
F64 :: float64;
B8 :: bool;
U32 :: u32;
U64 :: u64;
S32 :: s32;
S64 :: s64;

MinF32 :: 0h00800000;
MaxF32 :: 0h7F7FFFFF;
MinF64 :: 0h00100000_00000000;
MaxF64 :: 0h7FEFFFFF_FFFFFFFF;

first_true :: (b: [] B8) -> B8, int
{
  for b
  {
    if it
    {
      return true, it_index;
    }
  }
  return false, -1;
}

arena_temp: *Arena;

base_init :: ()
{
  arena_temp = mem_create_arena();
}

enum_cycle :: (e: *$T)
{
  value := cast(int) e.*;
  count := enum_highest_value(T) + 1;
  e.* = cast(T) ((value + 1) % count);
}

cstr_from_str :: (arena: *Arena, str: string) -> *u8
{
  data := arena_push(arena, xx (str.count + 1));
  copy(data, str.data, xx str.count);
  return data;
}

str_from_cstr :: (arena: *Arena, cstr: *u8) -> string
{
  length := c_style_strlen(cstr);
  str := arena_push_string(arena, xx length);
  copy(str.data, cstr, xx length);
  return str;
}

str_copy :: (arena: *Arena, source: string) -> string
{
  str := arena_push_string(arena, xx source.count);
  copy(str.data, source.data, xx source.count);
  return str;
}

str_format :: (arena: *Arena, format: string, args: ..Any) -> string
{
  builder: String_Builder;
  allocator := arena_allocator(arena);
  print_to_builder(*builder, format, ..args,, allocator);
  result := builder_to_string(*builder,, allocator);
  return result;
}

offset_of :: ($T: Type, $member: string) -> u32
{
  for type_info(T).members
  {
    if it.name == member return xx it.offset_in_bytes;
  }
  assert(false);
  return 0;
}

GB :: (n: u64) -> u64 #expand
{
  return n * 1024 * 1024 * 1024;
}

MB :: ($$n: u64) -> u64 #expand
{
  return n * 1024 * 1024;
}

KB :: ($$n: u64) -> u64 #expand
{
  return n * 1024;
}

align_pow2 :: (x: $T, a: T) -> T
{
  return (x + a - 1) & ~(a - 1);
}

copy :: (dest: *void, source: *void, size: u64)
{
  memcpy(dest, source, xx size);
}

zero :: (memory: *void, size: u64)
{
  memset(memory, 0, xx size);
}

stack_reset :: (first: **$T)
{
  first.* = null;
}

stack_push :: (first: **$T, value: *T, $next := "next") #expand
{
  using,only(tprint) Basic :: #import "Basic";

  #insert,scope() -> string
  {
    code :: #string HERE
    value.% = first.*;
    first.* = value;
    HERE
    return tprint(code, next);
  }
}

stack_pop :: (first: **$T, $next := "next") -> *T #expand
{
  using,only(tprint) Basic :: #import "Basic";

  #insert,scope() -> string
  {
    code :: #string HERE
    temp := first.*;
    first.* = first.*.%;
    return temp;
    HERE
    return tprint(code, next);
  }
}

queue_reset :: (first: **$T, last: **T)
{
  first.* = null;
  last.* = null;
}

queue_push :: (first: **$T, last: **T, value: *T, $next := "next") #expand
{
  using,only(tprint) Basic :: #import "Basic";

  #insert,scope() -> string
  {
    code :: #string HERE
    if first.*
    {
      last.*.% = value;
      last.* = value;
    }
    else
    {
      first.* = value;
      last.* = value;
    }
    HERE
    return tprint(code, next);
  }
}

queue_pop :: (first: **$T, last: **T, $next := "next") -> *T #expand
{
  using,only(tprint) Basic :: #import "Basic";

  #insert,scope() -> string
  {
    code :: #string HERE
    temp := first.*;
    first.* = first.*.%;
    if !first.*
    {
      last.* = null;
    }
    return temp;
    HERE
    return tprint(code, next);
  }
}

list_add :: (first: **$T, last: **T, value: *T, $next := "next", $prev := "prev")
{
  using,only(tprint) Basic :: #import "Basic";
  if !first.*
  {
    first.* = value;
    last.* = value;
  }
  else
  {
    #insert,scope() -> string
    {
      code ::#string HERE
      last.*.%1 = value;
      value.%2 = last.*;
      last.* = value;
      HERE
      return tprint(code, next, prev);
    }
  }
}

list_remove :: (first: **$T, last: **T, value: *T, $next := "next", $prev := "prev")
{
  using,only(tprint) Basic :: #import "Basic";
  #insert,scope() -> string
  {
    code ::#string HERE
    if value.%1 value.%1.%2 = value.%2;
    if value.%2 value.%2.%1 = value.%1;
    if value == first.* first.* = null;
    if value == last.* last.* = null;
    value.%1 = null;
    value.%2 = null;
    HERE
    return tprint(code, next, prev);
  }
}

LinkedListIterator :: struct($T: Type, $next := "next")
{
  first: *T;
}

ll_get_next :: (iter: LinkedListIterator($T, $next), value: *T) -> *T #expand
{
  using,only(tprint) Basic :: #import "Basic";

  #insert,scope() -> string
  {
    code :: #string HERE
    return value.%;
    HERE
    return tprint(code, next);
  }
}

each_element :: (first: *$T, $next := "next") -> LinkedListIterator(T, next)
{
  return LinkedListIterator(T, next).{first};
}

for_expansion :: (iter: LinkedListIterator($T, $next), body: Code, flags: For_Flags) #expand
{
  value := iter.first;
  index := 0;
  while value != null
  {
    `it := value;
    `it_index := index;
    #insert body;
    value = ll_get_next(iter, value);
    index += 1;
  }
}

file_write_line :: (file: *File, str: string)
{
  file_write(file, str);
  file_write(file, "\n");
}

file_print_line :: (file: *File, format: string, args: ..Any)
{
  allocator := arena_allocator(arena_temp);
  builder: String_Builder;
  print_to_builder(*builder, format, ..args,, allocator);
  append(*builder, "\n");
  str := builder_to_string(*builder,, allocator);
  file_write(file, str);
}

file_print_time :: (file: *File, name: string, t: F32)
{
  file_write(file, str_format(arena_temp, "%: %\n", t - game_time));
}

file_print_v2 :: (file: *File, name: string, v: Vec2F32)
{
  file_write(file, str_format(arena_temp, "%: [%, %]\n", name, v.x, v.y));
}

file_print_v3 :: (file: *File, name: string, v: Vec3F32)
{
  file_write(file, str_format(arena_temp, "%: [%, %, %]\n", name, v.x, v.y, v.z));
}

file_print_v4 :: (file: *File, name: string, v: Vec4F32)
{
  file_write(file, str_format(arena_temp, "%: [%, %, %, %]\n", name, v.x, v.y, v.z, v.w));
}

parse_v2 :: (str: string) -> Vec2F32
{
  x, y: F32;
  scan2(str, "[%, %]", *x, *y);
  result: Vec2F32;
  result.x = x;
  result.y = y;
  return result;
}

parse_v3 :: (str: string) -> Vec3F32
{
  x, y, z: F32;
  scan2(str, "[%, %, %]", *x, *y, *z);
  result: Vec3F32;
  result.x = x;
  result.y = y;
  result.z = z;
  return result;
}

parse_v4 :: (str: string) -> Vec4F32
{
  x, y, z, w: F32;
  scan2(str, "[%, %, %, %]", *x, *y, *z, *w);
  result: Vec4F32;
  result.x = x;
  result.y = y;
  result.z = z;
  result.w = w;
  return result;
}

parse_time :: (str: string) -> F32
{
  return game_time + parse_float(*str);
}