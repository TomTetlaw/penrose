
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