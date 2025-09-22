
stb_image_path :: "modules/stb_image";
sdl3_path :: "modules/sdl3";

build_stb_image :: ()
{
  // NOTE(tt): build binaries
  build_cpp_static_lib(tprint("%/lib/stb_image", stb_image_path), tprint("%/source/stb_image.cpp", stb_image_path));
  build_cpp_dynamic_lib(tprint("%/lib/stb_image", stb_image_path), tprint("%/source/stb_image.cpp", stb_image_path));
  // NOTE(tt): copy binaries
  copy_file(tprint("%/lib/stb_image.dll", stb_image_path), "build/stb_image.dll");
  // NOTE(tt): generate bindings
  options: Generate_Bindings_Options;
  options.footer = "FILE :: *void;";
  array_add(*options.include_paths, tprint("%/include", stb_image_path));
  array_add(*options.library_search_paths, tprint("%/lib", stb_image_path));
  array_add(*options.libraries, .{filename = "stb_image"});
  array_add(*options.source_files, "stb_image.h");
  generate_bindings(options, tprint("%/module.jai", stb_image_path));
}

sdl3_strip_decls :: (decl: *Declaration, parent_decl: *Declaration) -> Declaration_Visit_Result
{
  if decl.name == "MSG"
  {
    decl.decl_flags |= .OMIT_FROM_OUTPUT;
  }
  return .RECURSE;
}

build_sdl3 :: ()
{
  // NOTE(tt): copy binaries
  copy_file(tprint("%/lib/SDL3.dll", sdl3_path), "build/SDL3.dll");
  copy_file(tprint("%/lib/SDL3.pdb", sdl3_path), "build/SDL3.pdb");
  copy_file(tprint("%/lib/SDL3_shadercross.dll", sdl3_path), "build/SDL3_shadercross.dll");
  // NOTE(tt): generate bindings
  options: Generate_Bindings_Options;
  array_add(*options.include_paths, tprint("%/include", sdl3_path));
  array_add(*options.include_paths, tprint("%/include/SDL3", sdl3_path));
  array_add(*options.library_search_paths, tprint("%/lib", sdl3_path));
  array_add(*options.libraries, .{filename = "sdl3"});
  array_add(*options.libraries, .{filename = "SDL3_shadercross"});
  array_add(*options.source_files, "SDL.h");
  array_add(*options.source_files, "SDL_shadercross.h");
  options.visitor = sdl3_strip_decls;
  options.generate_library_declarations = false;
  options.footer = #string HERE
    MSG :: struct{};
    sdl3 :: #library "/lib/sdl3";
    sdl3_shadercross :: #library "/lib/SDL3_shadercross";
  HERE
  generate_bindings(options, tprint("%/module.jai", sdl3_path));
}

build :: ()
{
  project_dir := get_working_directory();
  workspace := compiler_create_workspace("Penrose build");
  release_build := false;
  options := get_build_options();
  args := options.compile_time_command_line;
  if array_find(args, "release")
  {
    release_build = true;
  }
  #if false
  {
    build_stb_image();
    build_sdl3();
  }
  options.output_executable_name = "penrose";
  options.output_path = "build";
  options.text_output_flags = 0;
  options.backend = .LLVM;
  if (release_build)
  {
    set_optimization(*options, .VERY_OPTIMIZED);
  }
  else
  {
    options.debug_for_expansions = true;
    set_optimization(*options, .VERY_DEBUG);
  }
  set_build_options(options, workspace);
  make_directory_if_it_does_not_exist("./build/shaders");
  files := file_list("./");
  for files
  {
    if contains(it, ".jai") && !contains(it, "build.jai")
    {
      add_build_file(it, workspace);
    }
    if contains(it, ".hlsl")
    {
      path, base, ext := path_decomp(it);
      copy_file(tprint("%.hlsl", base), tprint("build/shaders/%.hlsl", base));
    }
  }
  set_build_options_dc(.{do_output=false});
}

#run build();

#scope_file

#import "Basic";
#import "File";
#import "File_Utilities";
#import "String";
#import "Compiler";
#import "Bindings_Generator";
#import "Process";
#import "BuildCpp";