
Arena :: struct
{
  next_free: *Arena;
  memory: *void;
  base: *void;
  used: int;
  committed: int;
  reserved: int;
  alignment: int;
}
arena_header_size: int: 256;
#assert(size_of(Arena) < arena_header_size);

arena_default_reserve :: #run GB(16);
arena_default_commit  :: #run KB(16);
arena_default_align   :: 32;

ArenaOptions :: struct
{
  reserve := arena_default_reserve;
  commit := arena_default_commit;
  alignment := arena_default_align;
}

create_arena :: (opts: ArenaOptions = .{}, $L := #caller_location) -> *Arena
{
  if arena_free_list
  {
    arena := stack_pop(*arena_free_list, "next_free");
    arena_reset(arena);
    return arena;
  }
  page_size := os_page_size();
  commit := align_pow2(opts.commit + arena_header_size, page_size);
  reserve := align_pow2(opts.reserve, page_size);
  memory := os_reserve(reserve);
  os_commit(memory, commit);
  arena := cast(*Arena) memory;
  arena.memory = memory;
  arena.base = cast(*void) align_pow2(cast(int) memory + arena_header_size, opts.alignment);;
  arena.committed = commit - arena_header_size;
  arena.reserved = reserve;
  arena.alignment = opts.alignment;
  raddbg_annotate_vaddr_range(memory, xx reserve, #run str_from_caller_location(L));
  return arena;
}

scratch_begin :: () -> *Arena
{
  return create_arena();
}

scratch_end :: (arena: *Arena)
{
  arena_free(arena);
}

arena_free :: (arena: *Arena)
{
  stack_push(*arena_free_list, arena, "next_free");
}

push :: (arena: *Arena, size: int) -> *void
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

push_copy :: (arena: *Arena, data: *void, size: int) -> *void
{
  result := push(arena, size);
  copy(result, data, size);
  return result;
}

push_copy_view :: (arena: *Arena, view: [] $T) -> [] T
{
  data := push_copy(arena, view.data, xx (view.count * size_of(T)));
  return .{view.count, data};
}

push_struct :: (arena: *Arena, $T: Type, count: int = 1) -> *T
{
  result := push(arena, size_of(T));
  init_proc :: initializer_of(T);
  #if init_proc init_proc(result);
  return result;
}

push_string :: (arena: *Arena, length: int) -> string
{
  data := push(arena, cast(int) length);
  result: string;
  result.data = data;
  result.count = xx length;
  return result;
}

push_view :: (arena: *Arena, $T: Type, count: int) -> [] T
{
  memory := cast(*T) push(arena, size_of(T) * count);
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

push_mark :: (arena: *Arena) -> int
{
  return arena.used;
}

pop_mark :: (arena: *Arena, mark: int)
{
  zero(arena.base + mark, mark - arena.used);
  arena.used = mark;
}

arena_reset :: (arena: *Arena)
{
  zero(arena.base, arena.used);
  arena.used = 0;
}

arena_allocator :: (arena: *Arena) -> Allocator
{
  return .{arena_allocator_proc, arena};
}

arena_allocator_proc :: (mode: Allocator_Mode, requested_size: s64, old_size: s64, old_memory: *void, allocator_data: *void) -> *void
{
  arena := cast(*Arena) allocator_data;
  if mode ==
  {
    case .RESIZE;
    result := push(arena, xx requested_size);
    memcpy(result, old_memory, old_size);
    return result;
    case .ALLOCATE;
    return push(arena, xx requested_size);
  }
  return null;
}

#scope_file

arena_free_list: *Arena;

str_from_caller_location :: ($L: Source_Code_Location) -> string
{
  using,only(tprint) Basic :: #import "Basic";
  return tprint("file: %, line: %", L.fully_pathed_filename, L.line_number);
}