
tex_load :: (path: string) -> *DR_Texture
{
  data, ok := read_entire_file(path);
  if ok
  {
    w, h, c: s32;
    pixels := stbi_load_from_memory(data.data, xx data.count, *w, *h, *c, 4);
    if pixels
    {
      result := dr_create_draw_texture(path, w, h, pixels);
      stbi_image_free(pixels);
      return result;
    }
  }
  return null;
}