
tex_load :: (path: string) -> *DR_Texture
{
  data, ok := read_entire_file(path);
  if ok
  {
    w, h, c: s32;
    source_pixels := stbi_load_from_memory(data.data, xx data.count, *w, *h, *c, 4);
    defer stbi_image_free(source_pixels);
    if source_pixels
    {
      result := dr_create_draw_texture(path, w, h, .R8G8B8A8_UNORM, source_pixels);
      return result;
    }
  }
  return null;
}

tex_load_material :: (colour_path: string, normal_path: string, spec_path: string) -> *DR_Texture, *DR_Texture
{
  TexInfo :: struct
  {
    pixels: *u8;
    w, h: int;
  }
  read_tex :: (path: string) -> TexInfo
  {
    data, ok := read_entire_file(path);
    if ok
    {
      w, h, c: s32;
      pixels := stbi_load_from_memory(data.data, xx data.count, *w, *h, *c, 4);
      return .{pixels, w, h};
    }
    return .{};
  }
  colour := read_tex(colour_path);
  normal := read_tex(normal_path);
  spec := read_tex(spec_path);
  for y: 0..colour.h-1
  {
    for x: 0..colour.w-1
    {
      colour.pixels[(x + y*colour.w)*4 + 3] = spec.pixels[(x + y*colour.w)*4 + 0];
    }
  }
  colour_texture := dr_create_draw_texture(colour_path, colour.w, colour.h, .R8G8B8A8_UNORM, colour.pixels);
  normal_dest := cast(*u8) arena_push(arena_temp, xx (normal.w * normal.h * 2));
  for y: 0..normal.h-1
  {
    for x: 0..normal.w-1
    {
      normal_dest[(x + y*normal.w)*2 + 0] = normal.pixels[(x + y*normal.w)*4 + 0];
      normal_dest[(x + y*normal.w)*2 + 1] = normal.pixels[(x + y*normal.w)*4 + 1];
    }
  }
  normal_texture := dr_create_draw_texture(normal_path, normal.w, normal.h, .R8G8_UNORM, normal_dest);
  return colour_texture, normal_texture;
}