
BaseState :: struct
{
  arena: *Arena;
  frame_arena: *Arena;
}

base: *BaseState;

base_init :: ()
{
  arena := arena_create();
  base = arena_push(arena, BaseState);
  base.arena = arena;
  base.frame_arena = arena_create();
}

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

GB :: ($$n: u64) -> u64 #expand
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

// NOTE(tt): misc memory stuff

copy :: (dest: *void, source: *void, size: u64)
{
  s := cast(*u8) source;
  d := cast(*u8) dest;
  for 0..size-1
  {
    d[it] = s[it];
  }
}

zero :: (mem: *void, size: u64)
{
  base := cast(*u8) mem;
  for 0..size-1
  {
    base[it] = 0;
  }
}

// NOTE(tt): logging and error handling

fatal_error :: (format: string, args: ..Any)
{
  builder: String_Builder;
  print_to_builder(*builder, format, args);
  error := builder_to_string(*builder);
  SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Error", temp_cstr(error), null);
  exit(1);
}

temp_cstr :: (s: string) -> *u8
{
  data := talloc(xx (s.count + 1));
  copy(data, s.data, xx s.count);
  return data;
}

// NOTE(tt): arena allocator

Arena :: struct
{
  next_free: *Arena;
  memory: *void;
  base: *void;
  used: u64;
  committed: u64;
  reserved: u64;
  alignment: u64;
}
arena_header_size :: 256;
#assert(size_of(Arena) - arena_header_size);

arena_default_reserve :: #run GB(16);
arena_default_commit  :: #run KB(16);
arena_default_align   :u64: 32;

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
  base := memory + arena_header_size;
  base = cast(*void) align_pow2(cast(u64) base, alignment);
  arena := cast(*Arena) memory;
  arena.memory = memory;
  arena.base = base;
  arena.committed = commit - arena_header_size;
  arena.reserved = reserve;
  arena.alignment = alignment;
  return arena;
}

arena_free :: (arena: *Arena)
{
  stack_push(*arena_free_list, arena, "next_free");
}

arena_alloc :: (arena: *Arena, size: u64) -> *void
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

arena_push :: (arena: *Arena, $T: Type) -> *T
{
  result := arena_alloc(arena, size_of(T));
  init_proc :: initializer_of(T);
  #if init_proc init_proc(result);
  return result;
}

arena_push_n :: (arena: *Arena, $T: Type, count: u64) -> *T
{
  result := cast(*T) arena_alloc(arena, size_of(T) * count);
  init_proc :: initializer_of(T);
  #if init_proc
  {
    for 0..count-1
    {
      init_proc(result[it]);
    }
  }
  return result;
}

arena_push_view :: (arena: *Arena, $T: Type, count: u64) -> [] T
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

arena_push_mark :: (arena: *Arena) -> u64
{
  return arena.used;
}

arena_pop_mark :: (arena: *Arena, mark: u64)
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

  os_reserve :: (size: u64) -> *void
  {
    return VirtualAlloc(null, size, MEM_RESERVE, PAGE_READWRITE);
  }

  os_commit :: (memory: *void, size: u64)
  {
    VirtualAlloc(memory, size, MEM_COMMIT, PAGE_READWRITE);
  }

  os_free :: (memory: *void, size: u64)
  {
    VirtualFree(memory, size, MEM_DECOMMIT | MEM_RELEASE);
  }

  os_page_size :: () -> u64
  {
    info: SYSTEM_INFO;
    GetSystemInfo(*info);
    return info.dwPageSize;
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

  os_reserve :: (size: u64) -> *void
  {
    memory := mmap(null, xx size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, 0, 0);
    if (cast(int) memory == -1)
    {
      return null;
    }
    return memory;
  }

  os_commit :: (memory: *void, size: u64)
  {
    madvise(memory, xx size, MADV_WILLNEED);
  }

  os_free :: (memory: *void, size: u64)
  {
    munmap(memory, xx size);
  }

  os_page_size :: () -> u64
  {
    return xx getpagesize();
  }
}