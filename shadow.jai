
SH_ShadowVolume :: struct
{
  // NOTE(tt): data for base mesh
  base_positions: [] V3;
  base_indices: [] u32;
  // NOTE(tt): precomputed edges
  edges: [] ShadowEdge;
}

sh_generate_base_mesh :: (arena: *Arena, mesh: *DR_Mesh) -> *SH_ShadowVolume
{
  positions := push_view(arena, V3, xx mesh.positions.count);
  indices := push_view(arena, u32, xx mesh.index_count);
  table: Table(u32, u32);
  table.allocator = arena_allocator(temp_arena);
  index: u32 = 0;
  for mesh.index
  {
    p := mesh.positions[it];
    key := get_hash(p.x, get_hash(p.y, get_hash(p.z)));
    if !table_contains(*table, key)
    {
      positions[index] = p;
      indices[it_index] = index;
      table_add(*table, key, index);
      index += 1;
    }
    else
    {
      _, i := table_find_new(*table, key);
      indices[it_index] = i;
    }
  }
  positions.count = index;
  volume := push_struct(arena, SH_ShadowVolume);
  volume.base_positions = positions;
  volume.base_indices = indices;
  volume.edges = sh_generate_mesh_adjacency(arena, volume);
  return volume;
}

sh_generate_mesh_adjacency :: (arena: *Arena, volume: SH_ShadowVolume) -> [] ShadowEdge
{
  tri_count := volume.base_indices.count / 3;
  edge_count := 3 * tri_count;
  edges := push_view(arena, ShadowEdge, xx edge_count);
  for 0..tri_count-1
  {
    i0 := volume.base_indices[it*3 + 0];
    i1 := volume.base_indices[it*3 + 1];
    i2 := volume.base_indices[it*3 + 2];
    edges[it*3 + 0] = .{min(i0, i1), max(i0, i1), it, -1};
    edges[it*3 + 1] = .{min(i1, i2), max(i1, i2), it, -1};
    edges[it*3 + 2] = .{min(i2, i0), max(i2, i0), it, -1};
  }
  quick_sort(edges, (a, b) =>
    ifx a.i0 != b.i0
      then cast(int) a.i0 - cast(int) b.i0
      else cast(int) a.i1 - cast(int) b.i1
  );
  count := 0;
  result := push_view(arena, ShadowEdge, xx edge_count);
  for 0..edge_count-2
  {
    e0 := *edges[it];
    e1 := *edges[it + 1];
    if e0.i0 == e1.i0 && e0.i1 == e1.i1
    {
      e0.t1 = e1.t0;
      result[count] = e0;
      count += 1;
    }
  }
  result.count = count;
  return result;
}

sh_build_shadow_mesh :: (arena: *Arena, volume: *SH_ShadowVolume, light: DR_LightInfo, to_world: Mat4) -> [] V3
{
  tri_count := volume.base_indices.count / 3;
  facing := push_view(arena, bool, xx tri_count);
  for 0..tri_count-1
  {
    p0 := mat_transform_position(volume.base_positions[volume.base_indices[it*3 + 0]], to_world);
    p1 := mat_transform_position(volume.base_positions[volume.base_indices[it*3 + 1]], to_world);
    p2 := mat_transform_position(volume.base_positions[volume.base_indices[it*3 + 2]], to_world);
    normal := normalize(cross(p1 - p0, p2 - p0));
    if light.type ==
    {
    case .Point;
      center := (p0 + p1 + p2) / 3.0;
      light_dir := normalize(center - light.position);
      facing[it] = dot(normal, light_dir) < 0;
    }
  }
  edges := push_view(arena, ShadowEdge, xx volume.edges.count);
  edge_count := 0;
  for volume.edges
  {
    if facing[it.t0] != facing[it.t1]
    {
      edges[edge_count] = it;
      edge_count += 1;
    }
  }
  edges.count = edge_count;
  max_vertex_count := edges.count*6 + tri_count*2*3;
  positions := push_view(arena, V3, xx max_vertex_count);
  extrude := -1000.0;
  index := 0;
  // NOTE(tt): side quads
  for edges
  {
    t := ifx facing[it.t0] then it.t0 else it.t1;
    a := volume.base_indices[t*3 + 0];
    b := volume.base_indices[t*3 + 1];
    c := volume.base_indices[t*3 + 2];
    i0, i1 := -1;
    if (it.i0 == a && it.i1 == b) || (it.i1 == a && it.i0 == b)
    {
      i0 = a;
      i1 = b;
    }
    else if (it.i0 == b && it.i1 == c) || (it.i1 == b && it.i0 == c)
    {
      i0 = b;
      i1 = c;
    }
    else if (it.i0 == c && it.i1 == a) || (it.i1 == c && it.i0 == a)
    {
      i0 = c;
      i1 = a;
    }
    v0 := mat_transform_position(volume.base_positions[i0], to_world);
    v1 := mat_transform_position(volume.base_positions[i1], to_world);
    v0_ex, v1_ex: V3;
    if light.type ==
    {
    case .Point;
      v0_dir := normalize(light.position - v0);
      v1_dir := normalize(light.position - v1);
      v0_ex = v0 + v0_dir * extrude;
      v1_ex = v1 + v1_dir * extrude;
    }
    // NOTE(tt): first tri
    positions[index + 0] = v0;
    positions[index + 1] = v1;
    positions[index + 2] = v1_ex;
    index += 3;
    // NOTE(tt): second tri
    positions[index + 0] = v0;
    positions[index + 1] = v1_ex;
    positions[index + 2] = v0_ex;
    index += 3;
  }
  // NOTE(tt): front cap (facing toward light)
  for 0..tri_count-1
  {
    if facing[it]
    {
      i0 := volume.base_indices[it*3 + 0];
      i1 := volume.base_indices[it*3 + 1];
      i2 := volume.base_indices[it*3 + 2];
      p0 := mat_transform_position(volume.base_positions[i0], to_world);
      p1 := mat_transform_position(volume.base_positions[i1], to_world);
      p2 := mat_transform_position(volume.base_positions[i2], to_world);
      positions[index + 0] = p0;
      positions[index + 1] = p2;
      positions[index + 2] = p1;
      index += 3;
    }
  }
  // NOTE(tt): back cap (facing away from light)
  for 0..tri_count-1
  {
    if !facing[it]
    {
      i0 := volume.base_indices[it*3 + 0];
      i1 := volume.base_indices[it*3 + 1];
      i2 := volume.base_indices[it*3 + 2];
      p0 := mat_transform_position(volume.base_positions[i0], to_world);
      p1 := mat_transform_position(volume.base_positions[i1], to_world);
      p2 := mat_transform_position(volume.base_positions[i2], to_world);
      if light.type ==
      {
      case .Point;
        p0 = p0 + normalize(light.position - p0) * extrude;
        p1 = p1 + normalize(light.position - p1) * extrude;
        p2 = p2 + normalize(light.position - p2) * extrude;
      }
      positions[index + 0] = p0;
      positions[index + 1] = p2;
      positions[index + 2] = p1;
      index += 3;
    }
  }
  positions.count = index;
  return positions;
}

#scope_file

ShadowEdge :: struct
{
  i0, i1: u32;
  t0, t1: int;
}