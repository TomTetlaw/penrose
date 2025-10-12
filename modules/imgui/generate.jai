
generate :: ()
{
  set_build_options_dc(.{do_output = false});

  source_files := string.[
    "src/imgui_widgets.cpp",
    "src/imgui_draw.cpp",
    "src/imgui_tables.cpp",
    "src/imgui_demo.cpp",
    "src/imgui.cpp",
    "src/imgui_impl_sdl3.cpp",
    "src/imgui_impl_sdlgpu3.cpp"
  ];
  extra := string.[
    "-Iinclude",
    "-Iinclude/backends",
    "-I../sdl3/include"
  ];
  build_cpp_static_lib("lib/imgui", ..source_files, extra = extra, debug = true);
  build_cpp_dynamic_lib("bin/imgui", ..source_files, extra = extra, debug = true, library_files = .["../sdl3/lib/sdl3.lib", "user32.lib", "imm32.lib"]);

  options: Generate_Bindings_Options;
  array_add(*options.library_search_paths, "lib");
  array_add(*options.library_search_paths, "bin");
  array_add(*options.libraries, .{filename = "user32"}, .{filename = "imgui"});
  array_add(*options.include_paths, "include/backends");
  array_add(*options.include_paths, "include");
  array_add(*options.source_files, "imgui.h", "imgui_impl_sdl3.h", "imgui_impl_sdlgpu3.h");
  array_add(*options.system_include_paths, "sdl3/include");
  array_add(*options.system_include_paths, "sdl3");
  array_add(*options.extra_clang_arguments, "-I../sdl3/include/", "-x", "c++");
  options.visitor = visitor;
  options.footer = #string HERE
    FLOAT32_MIN :: 0h00800000;
    FLOAT32_MAX :: 0h7F7FFFFF;
    #import "sdl3";
  HERE;
  generate_bindings(options, "module.jai");
}

visitor :: (decl: *Declaration, parent_decl: *Declaration) -> Declaration_Visit_Result
{
  found, path := table_find_new(*context.generator.filenames, decl.location.file);
  if found
  {
    if contains(path, "sdl3")
    {
      found = true;
      decl.decl_flags = .OMIT_FROM_OUTPUT;
      return .STOP;
    }
    found = false;
  }
  names := string.[
    "HAVE_WINAPIFAMILY_H",
    "SDL_PLATFORM_WINDOWS",
    "SDL_PLATFORM_WIN32",
    "SDL_Window",
    "SDL_Renderer",
    "SDL_Gamepad",
    "SDL_Event"
  ];
  if array_find(names, decl.name)
  {
    decl.decl_flags = .OMIT_FROM_OUTPUT;
  }
  return .RECURSE;
}

#run generate();

#import "Bindings_Generator";
#import "Basic";
#import "Compiler";
#import "BuildCpp";
#import "Hash_Table";
#import "String";