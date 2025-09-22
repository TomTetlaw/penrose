
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
arena_header_size: u64: 256;
#assert(size_of(Arena) < arena_header_size);

arena_default_reserve:u64: #run GB(16);
arena_default_commit :u64: #run KB(16);
arena_default_align  :u64: 32;

arena_free_list: *Arena;

ArenaOptions :: struct
{
  reserve := arena_default_reserve;
  commit := arena_default_commit;
  alignment := arena_default_align;
}

mem_create_arena :: (opts: ArenaOptions = .{}, $L := #caller_location) -> *Arena
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
  arena.base = cast(*void) align_pow2(cast(u64) memory + arena_header_size, opts.alignment);;
  arena.committed = commit - arena_header_size;
  arena.reserved = reserve;
  arena.alignment = opts.alignment;
  raddbg_annotate_vaddr_range(memory, reserve, #run str_from_caller_location(L));
  return arena;
}

arena_scratch_begin :: () -> *Arena
{
  return mem_create_arena();
}

arena_scratch_end :: (arena: *Arena)
{
  arena_free(arena);
}

arena_free :: (arena: *Arena)
{
  stack_push(*arena_free_list, arena, "next_free");
}

arena_push :: (arena: *Arena, size: u64) -> *void
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

arena_push_struct :: (arena: *Arena, $T: Type, count: u64 = 1) -> *T
{
  result := arena_push(arena, size_of(T));
  init_proc :: initializer_of(T);
  #if init_proc init_proc(result);
  return result;
}

arena_push_string :: (arena: *Arena, length: u64) -> string
{
  data := arena_push(arena, cast(u64) length);
  result: string;
  result.data = data;
  result.count = xx length;
  return result;
}

arena_push_view :: (arena: *Arena, $T: Type, count: u64) -> [] T
{
  memory := cast(*T) arena_push(arena, size_of(T) * count);
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
    result := arena_push(arena, xx requested_size);
    memcpy(result, old_memory, old_size);
    return result;
    case .ALLOCATE;
    return arena_push(arena, xx requested_size);
  }
  return null;
}

#scope_file

str_from_caller_location :: ($L: Source_Code_Location) -> string
{
  using,only(tprint) Basic :: #import "Basic";
  return tprint("file: %, line: %", L.fully_pathed_filename, L.line_number);
}