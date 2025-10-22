
temp_arena: *Arena;

base_init :: ()
{
  temp_arena = create_arena();
}

enum_cycle :: (e: *$T)
{
  value := cast(int) e.*;
  count := enum_highest_value(T) + 1;
  e.* = cast(T) ((value + 1) % count);
}

cstr_from_str :: (arena: *Arena, str: string) -> *u8
{
  data := push(arena, xx (str.count + 1));
  copy(data, str.data, xx str.count);
  return data;
}

str_from_cstr :: (arena: *Arena, cstr: *u8) -> string
{
  length := c_style_strlen(cstr);
  str := push_string(arena, length);
  copy(str.data, cstr, xx length);
  return str;
}

str_copy :: (arena: *Arena, source: string) -> string
{
  str := push_string(arena, source.count);
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

str_temp_format :: (format: string, args: ..Any) -> string
{
  return str_format(temp_arena, format, ..args);
}

offset_of :: ($T: Type, $member: string) -> u32
{
  for type_info(T).members
  {
    if it.name == member return xx it.offset_in_bytes;
  }
  assert(false, "member not found");
  return 0;
}

GB :: (n: int) -> int #expand
{
  return n * 1024 * 1024 * 1024;
}

MB :: ($$n: int) -> int #expand
{
  return n * 1024 * 1024;
}

KB :: ($$n: int) -> int #expand
{
  return n * 1024;
}

align_pow2 :: (x: $T, a: T) -> T
{
  return (x + a - 1) & ~(a - 1);
}

copy :: (dest: *void, source: *void, size: int)
{
  memcpy(dest, source, xx size);
}

zero :: (memory: *void, size: int)
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
  allocator := arena_allocator(temp_arena);
  builder: String_Builder;
  print_to_builder(*builder, format, ..args,, allocator);
  append(*builder, "\n");
  str := builder_to_string(*builder,, allocator);
  file_write(file, str);
}

file_print_time :: (file: *File, name: string, t: float)
{
  file_write(file, str_format(temp_arena, "%: %\n", t - game_time));
}

file_print :: (file: *File, name: string, v: V2)
{
  file_write(file, str_format(temp_arena, "%: [%, %]\n", name, v.x, v.y));
}

file_print :: (file: *File, name: string, v: V3)
{
  file_write(file, str_format(temp_arena, "%: [%, %, %]\n", name, v.x, v.y, v.z));
}

file_print :: (file: *File, name: string, v: V4)
{
  file_write(file, str_format(temp_arena, "%: [%, %, %, %]\n", name, v.x, v.y, v.z, v.w));
}

parse_v2 :: (str: string) -> V2
{
  x, y: float;
  scan2(str, "[%, %]", *x, *y);
  result: V2;
  result.x = x;
  result.y = y;
  return result;
}

parse_v3 :: (str: string) -> V3
{
  x, y, z: float;
  scan2(str, "[%, %, %]", *x, *y, *z);
  result: V3;
  result.x = x;
  result.y = y;
  result.z = z;
  return result;
}

parse_v4 :: (str: string) -> V4
{
  x, y, z, w: float;
  scan2(str, "[%, %, %, %]", *x, *y, *z, *w);
  result: V4;
  result.x = x;
  result.y = y;
  result.z = z;
  result.w = w;
  return result;
}

parse_time :: (str: string) -> float
{
  return game_time + parse_float(*str);
}