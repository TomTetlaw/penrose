
#program_export "main"
__system_entry_point :: (argc: s32, argv: **u8) -> s32 #c_call
{
  push_context
  {
    entry_point(argc, argv);
    return 0;
  }
}

entry_point :: (argc: s32, argv: **u8)
{
  list: String8List;
  arena := arena_create();
  str8_list_add(arena, *list, str8_from_lit("The spice alotment"));
  str8_list_add(arena, *list, str8_from_lit("passes over you"));
  str8_list_add(arena, *list, str8_from_lit("for this decade."));
  join: StringJoin;
  join.pre = str8_from_lit("--->");
  join.sep = str8_from_lit("... ");
  join.post = str8_from_lit("<---\n");
  joined := str8_list_join(arena, list, join);
  os_write_console(joined);
}

B8  :: bool;
I8  :: u8;
I16 :: u16;
I32 :: u32;
I64 :: u64;
F32 :: float;
F64 :: float64;

sign_bit_i8  :: cast(I8)  (1 << 7);
sign_bit_i16 :: cast(I16) (1 << 15);
sign_bit_i32 :: cast(I32) (1 << 31);
sign_bit_i64 :: cast(I64) (1 << 63);

set_ftz_daz :: ()
{
  ftz_bit :: 0b1000000000000000;
  daz_bit :: 0b0000000001000000;
  flags: I32;
  #asm
  {
    stmxcsr [flags];
    mov a:, flags;
    or a, ftz_bit;
    or a, daz_bit;
    ldmxcsr [flags];
  }
}

String8 :: struct
{
  data: *I8;
  length: I64;
}

str8_from_lit :: ($$str: string) -> String8
{
  result: String8;
  result.data = str.data;
  result.length = cast(I64) str.count;
  return result;
}

str8_copy :: (arena: *Arena, source: String8) -> String8
{
  result := arena_push_string(arena, source.length);
  copy(result.data, source.data, source.length);
  return result;
}

str8_find_char_left :: (str: String8, cursor: I64, c: I8) -> I64
{
  assert(cursor < str.length);
  for cursor..str.length-1
  {
    if str.data[it] == c
    {
      return it;
    }
  }
  return sign_bit_i64;
}

str8_split :: (arena: *Arena, str: String8, c: I8) -> String8List
{
  cursor: I64 = 0;
  list: String8List;
  while (cursor < str.length)
  {
    pos := str8_find_char_left(str, cursor, c);
    if (pos == sign_bit_i64)
    {
      break;
    }
    result: String8;
    result.data = str.data + cursor;
    result.length = pos - 1;
    str8_list_add(arena, *list, result);
    cursor += pos;
  }
  return list;
}

str8_from_int :: (arena: *Arena, value: Any, radix: I64) -> String8
{
  value_i64: I64;
  if value.type.runtime_size ==
  {
    case 1; value_i64 = (cast(*u8)  value.value_pointer).*;
    case 2; value_i64 = (cast(*u16) value.value_pointer).*;
    case 4; value_i64 = (cast(*u32) value.value_pointer).*;
    case 8; value_i64 = (cast(*u64) value.value_pointer).*;
  }
  buffer: [66] I8;
  index: I64 = 65;
  base: I64;
  if radix ==
  {
    case 16; base = 16;
    case 8; base = 8;
    case 2; base = 2;
  }
  while value_i64 > 0
  {
    digit := value_i64 % base;
    value_i64 /= base;
    if digit < 10
    {
      buffer[index] = cast(I8) (#char "0" + digit);
    }
    else
    {
      buffer[index] = cast(I8) (#char "a" + (digit - 10));
    }
    index -= 1;
  }
  if radix ==
  {
    case 16;
      buffer[index - 0] = #char "x";
      buffer[index - 1] = #char "0";
      index -= 2;
    case 8;
      buffer[index - 0] = #char "o";
      buffer[index - 1] = #char "0";
      index -= 2;
    case 2;
      buffer[index - 0] = #char "b";
      buffer[index - 1] = #char "0";
      index -= 2;
  }
  length := 66 - index;
  result := arena_push_string(arena, length);
  copy(result.data, buffer.data + index, length);
  return result;
}

str8_from_float :: (arena: *Arena, value: Any) -> String8
{
  return str8_from_lit("");
}

str8_from_bool :: (arena: *Arena, value: Any) -> String8
{
  return str8_from_lit("");
}

str8_from_enum :: (arena: *Arena, value: Any) -> String8
{
  return str8_from_lit("");
}

str8_format :: (arena: *Arena, format: String8, args: ..Any) -> String8
{
  content_strings := str8_split(arena, format, #char "%");
  if content_strings.count > 0
  {
    arg_strings: String8List;
    for args
    {
      if it.type.type ==
      {
        case .INTEGER;
          str8_list_add(arena, *list, str8_from_int(arena, it, 0));
        case .FLOAT;
          str8_list_add(arena, *list, str8_from_float(arena, it));
        case .BOOL;
          str8_list_add(arena, *list, str8_from_bool(arena, it));
        case .STRUCT;
          info := cast(*Type_Info_Struct) it.type;
          if info.name == "String8"
          {
            str := cast(*String8) it.value_pointer;
            str8_list_add(arena, *list, str.*);
          }
        case .POINTER;
          address := cast(I64) it.value_pointer;
          str8_list_add(arena, *list, str8_from_int(arena, address, 8));
        case .ENUM;
          str8_list_add(arena, *list, str8_from_enum(arena, it));
      }
    }
    String8Node *content = content_strings.first;
    String8Node *arg = arg_strings.first;
    while (content && arg)
    {
      arg = arg.next;
      content = content.next;
    }
  }
  else
  {
    return format;
  }
}

String8Node :: struct
{
  next: *String8Node;
  value: String8;
}

String8List :: struct
{
  first: *String8Node;
  last: *String8Node;
  count: I64;
}

StringJoin :: struct
{
  pre: String8;
  sep: String8;
  post: String8;
}

str8_list_add_node :: (list: *String8List, node: *String8Node)
{
  queue_push(*list.first, *list.last, node);
  list.count += 1;
}

str8_list_add :: (arena: *Arena, list: *String8List, value: String8)
{
  node := arena_push(arena, String8Node);
  node.value = str8_copy(arena, value);
  str8_list_add_node(list, node);
}

str8_list_push_node :: (arena: *Arena, list: *String8List) -> *String8Node
{
  node := arena_push(arena, String8Node);
  str8_list_add_node(list, node);
  list.count += 1;
  return node;
}

str8_list_push :: (arena: *Arena, list: *String8List, length: I64) -> *String8
{
  node := str8_list_push_node(arena, list);
  node.value = arena_push_string(arena, length);
  return *node.value;
}

str8_list_concat :: (arena: *Arena, dest: *String8List, source: String8List)
{
  for each_element(source.first)
  {
    str8_list_add_node(dest, it);
  }
}

str8_list_join :: (arena: *Arena, list: String8List, join: StringJoin) -> String8
{
  total_length: I64;
  for each_element(list.first)
  {
    total_length += it.value.length;
  }
  length := total_length + (join.sep.length * list.count) + join.pre.length + join.post.length;
  result := arena_push_string(arena, length);
  cursor: I64;
  dest := result.data;
  copy_advance(dest, *cursor, join.pre.data, join.pre.length);
  for each_element(list.first)
  {
    copy_advance(dest, *cursor, it.value.data, it.value.length);
    copy_advance(dest, *cursor, join.sep.data, join.sep.length);
  }
  copy_advance(dest, *cursor, join.post.data, join.post.length);
  return result;
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

V2F32 :: V2(F32);
V3F32 :: V3(F32);
V4F32 :: V4(F32);
V2F64 :: V2(F64);
V3F64 :: V3(F64);
V4F64 :: V4(F64);

V2U32 :: V2(I32);
V3U32 :: V3(I32);
V4U32 :: V4(I32);
V2I64 :: V2(I64);
V3I64 :: V3(I64);
V4I64 :: V4(I64);

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

// NOTE(tt): misc math stuff

align_pow2 :: (x: $T, a: T) -> T
{
  return (x + a - 1) & ~(a - 1);
}

enum_count :: ($T: Type) -> int #compile_time
{
  info := type_info(T);
  assert(info.type == .ENUM);
  return info.values.count;
}

// NOTE(tt): units

GB :: (n: I64) -> I64 #expand
{
  return n * 1024 * 1024 * 1024;
}

MB :: ($$n: I64) -> I64 #expand
{
  return n * 1024 * 1024;
}

KB :: ($$n: I64) -> I64 #expand
{
  return n * 1024;
}

// NOTE(tt): misc memory stuff

copy :: (dest: *void, source: *void, size: I64)
{
  s := cast(*I8) source;
  d := cast(*I8) dest;
  for 0..size-1
  {
    d[it] = s[it];
  }
}

copy_advance :: (dest: *void, cursor: *I64, source: *void, size: I64)
{
  copy(dest + cursor.*, source, size);
  cursor.* += size;
}

zero :: (memory: *void, size: I64)
{
  base := cast(*u8) memory;
  for 0..size-1
  {
    base[it] = 0;
  }
}

// NOTE(tt): arena allocator

Arena :: struct
{
  next_free: *Arena;
  memory: *void;
  base: *void;
  used: I64;
  committed: I64;
  reserved: I64;
  alignment: I64;
}
arena_header_size: I64: 256;
#assert(size_of(Arena) < arena_header_size);

arena_default_reserve:I64: #run GB(16);
arena_default_commit :I64: #run KB(16);
arena_default_align  :I64: 32;

arena_free_list: *Arena;

arena_create :: (reserve := arena_default_reserve, commit := arena_default_commit, alignment := arena_default_align) -> *Arena
{
  if arena_free_list
  {
    arena := stack_pop(*arena_free_list, "next_free");
    arena_reset(arena);
    return arena;
  }
  page_size := os_page_size();
  commit += arena_header_size + alignment;
  commit = align_pow2(commit, page_size);
  reserve = align_pow2(reserve, page_size);
  memory := os_reserve(reserve);
  os_commit(memory, commit);
  base := cast(*void) align_pow2(cast(I64) memory + arena_header_size, alignment);
  arena := cast(*Arena) memory;
  arena.memory = memory;
  arena.base = base;
  arena.committed = commit - arena_header_size;
  arena.reserved = reserve;
  arena.alignment = alignment;
  zero(base, arena.committed);
  return arena;
}

arena_scratch_begin :: () -> *Arena
{
  return arena_create();
}

arena_scratch_end :: (arena: *Arena)
{
  arena_free(arena);
}

arena_free :: (arena: *Arena)
{
  stack_push(*arena_free_list, arena, "next_free");
}

arena_alloc :: (arena: *Arena, size: I64) -> *void
{
  if arena.used + size >= arena.committed
  {
    page_size := os_page_size();
    aligned := align_pow2(size, page_size);
    commit := max(aligned, arena_default_commit);
    os_commit(arena.base + arena.committed, commit);
    arena.committed += commit;
  }
  size = align_pow2(size, arena.alignment);
  memory := arena.base + arena.used;
  arena.used += size;
  zero(memory, size);
  return memory;
}

arena_push :: (arena: *Arena, $T: Type, count: I64 = 1) -> *T
{
  result := arena_alloc(arena, size_of(T));
  init_proc :: initializer_of(T);
  #if init_proc init_proc(result);
  return result;
}

arena_push_string :: (arena: *Arena, length: I64) -> String8
{
  data := arena_alloc(arena, cast(I64) length);
  result: String8;
  result.data = data;
  result.length = length;
  return result;
}

arena_push_view :: (arena: *Arena, $T: Type, count: I64) -> [] T
{
  memory := cast(*T) arena_alloc(arena, size_of(T) * count);
  result := []T.{cast(int) count, memory};
  init_proc :: initializer_of(T);
  #if init_proc
  {
    for * result
    {
      init_proc(it);
    }
  }
  return result;
}

arena_push_mark :: (arena: *Arena) -> I64
{
  return arena.used;
}

arena_pop_mark :: (arena: *Arena, mark: I64)
{
  zero(arena.base + mark, mark - arena.used);
  arena.used = mark;
}

arena_reset :: (arena: *Arena)
{
  zero(arena.base, arena.used);
  arena.used = 0;
}

// NOTE(tt): linked lists

stack_reset :: (first: **$T)
{
  first.* = null;
}

stack_push :: (first: **$T, value: *T, $next := "next") #expand
{
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

LinkedListIterator :: struct($T: Type, $next := "next")
{
  first: *T;
}

ll_get_next :: (iter: LinkedListIterator($T, $next), value: *T) -> *T #expand
{
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

for_expansion
:: (iter: LinkedListIterator($T, $next), body: Code, flags: For_Flags) #expand
{
  assert(!(flags & .REVERSE));
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

// NOTE(tt): OS specific stuff

#if OS == .WINDOWS
{
  using,only
  (
    VirtualAlloc,
    VirtualFree,
    GetSystemInfo,
    MEM_RESERVE, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, PAGE_READONLY, PAGE_READWRITE,
    SYSTEM_INFO
  ) Windows :: #import "Windows";

  os_reserve :: (size: I64) -> *void
  {
    return VirtualAlloc(null, size, MEM_RESERVE, PAGE_READWRITE);
  }

  os_commit :: (memory: *void, size: I64)
  {
    VirtualAlloc(memory, size, MEM_COMMIT, PAGE_READWRITE);
  }

  os_free :: (memory: *void, size: I64)
  {
    VirtualFree(memory, size, MEM_DECOMMIT | MEM_RELEASE);
  }

  os_page_size :: () -> I64
  {
    info: SYSTEM_INFO;
    GetSystemInfo(*info);
    return info.dwPageSize;
  }

  kernel32 :: #library,system "kernel32";
  HANDLE :: *void;
  DWORD :: u32;
  BOOL :: s32;
  STD_OUTPUT_HANDLE :: -11;
  GetStdHandle :: (handle: s32) -> HANDLE #foreign kernel32;
  WriteFile :: (handle: HANDLE, buffer: *void, buffer_length: DWORD, written_result: *DWORD, overlapped: *void) -> BOOL #foreign kernel32;
  os_write_console :: (message: String8)
  {
    standard_out := GetStdHandle(STD_OUTPUT_HANDLE);
    if (standard_out)
    {
      written: DWORD;
      WriteFile(standard_out, message.data, cast(u32) message.length, *written, null);
    }
  }
}
else #if OS == .LINUX
{
  using,only
  (
    mmap,
    munmap,
    madvise,
    getpagesize,
    PROT_READ, PROT_WRITE, MAP_PRIVATE, MAP_ANONYMOUS, MAP_NORESERVE,
    MADV_WILLNEED
  ) Linux :: #import "POSIX";

  os_reserve :: (size: I64) -> *void
  {
    memory := mmap(null, xx size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, 0, 0);
    if (cast(int) memory == -1)
    {
      return null;
    }
    return memory;
  }

  os_commit :: (memory: *void, size: I64)
  {
    madvise(memory, xx size, MADV_WILLNEED);
  }

  os_free :: (memory: *void, size: I64)
  {
    munmap(memory, xx size);
  }

  os_page_size :: () -> I64
  {
    return xx getpagesize();
  }
}

raddbg_attached_data :: #run -> [] I8
{
	#import "Basic";
	#import "String";

	segment, ok := add_data_segment(".raddbg", .READ | .WRITE);
	if !ok
	{
		print("RADDBG section will not be created, sorry!\n");
		return .[0];
	}

	attached_data := add_global_data(.[0], .USER_SEGMENT, segment);

	add_global_data(xx "entry_point: \"main\"\0", .USER_SEGMENT, segment);

	add_view_rule :: (segment: *Data_Segment, type_string: string, rule: string)
	{
		rule_string := tprint("type_view: {type: ```%```, expr: ```%```}\0", type_string, rule);
		add_global_data(xx rule_string, .USER_SEGMENT, segment);
	}

	add_view_rule :: (segment: *Data_Segment, $T: Type, rule: string)
	{
		struct_info := cast (*Type_Info_Struct) type_info(T);
		rule_string := tprint("type_view: {type: ```%```, expr: ```%```}\0", struct_info.name, rule);
		replace_chars(rule_string, "\n\t", #char " ");
		add_global_data(xx rule_string, .USER_SEGMENT, segment);
	}

	add_view_rule(segment, "? (?)", "disasm($)");
	add_view_rule(segment, "[]??", "rows($, count, slice($))");
	add_view_rule(segment, "[..]??", "rows($, count, slice($))");
	add_view_rule(segment, "string", "text($.data)");
	add_view_rule(segment, V2F32, "rows($, x, y)");
	add_view_rule(segment, V3F32, "rows($, x, y, z)");
	add_view_rule(segment, V4F32, "rows($, x, y, z, w)");
	add_view_rule(segment, V2F64, "rows($, x, y)");
	add_view_rule(segment, V3F64, "rows($, x, y, z)");
	add_view_rule(segment, V4F64, "rows($, x, y, z, w)");

	return attached_data;
}

raddbg_attached_data_pointer :: #run raddbg_attached_data.data;

raddbg_is_attached :: () -> bool
{
	return raddbg_attached_data_pointer.* == 1;
}

raw :: (values: ..Any) #expand {}
bin :: (values: ..Any) #expand {}
oct :: (values: ..Any) #expand {}
dec :: (values: ..Any) #expand {}
hex :: (values: ..Any) #expand {}
digits :: (values: ..Any) #expand {}
no_string :: (values: ..Any) #expand {}
no_char :: (values: ..Any) #expand {}
no_addr :: (values: ..Any) #expand {}
sequence :: (values: ..Any) #expand {}
rows :: (values: ..Any) #expand {}
omit :: (values: ..Any) #expand {}
array :: (values: ..Any) #expand {}
slice :: (values: ..Any) #expand {}
columns :: (values: ..Any) #expand {}
text :: (values: ..Any) #expand {}
disasm :: (values: ..Any) #expand {}
memory :: (values: ..Any) #expand {}
bitmap :: (values: ..Any) #expand {}
color :: (values: ..Any) #expand {}
range1 :: (values: ..Any) #expand {}
raddbg_pin :: (values: ..Any) #expand {}

#scope_file

#import "Compiler";