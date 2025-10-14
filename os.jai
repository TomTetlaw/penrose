
#if OS == .WINDOWS
{
  os_reserve :: (size: u64) -> *void
  {
    return VirtualAlloc(null, size, MEM_RESERVE, PAGE_READWRITE);
  }

  os_commit :: (memory: *void, size: u64)
  {
    VirtualAlloc(memory, size, MEM_COMMIT, PAGE_READWRITE);
    ___tracy_emit_memory_alloc(memory, size, 0);
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