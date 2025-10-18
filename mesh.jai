
mesh_load :: (arena: *Arena, path: string) -> *DR_Mesh
{
  file, ok := file_open(path);
  defer file_close(*file);
  if ok
  {
    vert_count: u32;
    file_read(*file, *vert_count, size_of(u32));
    verts := arena_push_view(arena, DR_MeshVertex, vert_count);
    file_read(*file, verts.data, vert_count * size_of(DR_MeshVertex));
    indices := arena_push_view(arena, u32, vert_count);
    for 0..vert_count-1
    {
      indices[it] = it;
    }
    path, base, ext := path_decomp(path);
    mesh := dr_create_mesh(base, verts, indices);
    return mesh;
  }
  return null;
}