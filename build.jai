
build_sdl3 :: (working_dir: string, release_build: bool, workspace: Workspace)
{
  set_working_directory(working_dir);

  if OS == .WINDOWS {
    copy_file("lib/SDL3.dll", "../../build/SDL3.dll");
    copy_file("lib/SDL3_shadercross.dll", "../../build/SDL3_shadercross.dll");
  } else if OS == .LINUX {
    copy_file("lib/libSDL3.so", "../../build/libSDL3.so.0");
    copy_file("lib/libSDL3_shadercross.so", "../../build/libSDL3_shadercross.so.0");
    copy_file("lib/libspirv-cross-c-shared.so.0", "../../build/libspirv-cross-c-shared.so.0");
    copy_file("lib/libdxcompiler.so", "../../build/libdxcompiler.so");
  }

  options: Generate_Bindings_Options;
  array_add(*options.include_paths, "include");
  array_add(*options.include_paths, "include/SDL3");
  array_add(*options.library_search_paths, "lib");

  if OS == .WINDOWS {
    array_add(*options.libraries, Library.{filename="sdl3"});
    array_add(*options.libraries, Library.{filename="SDL3_shadercross"});
  } else if OS == .LINUX {
    array_add(*options.libraries, Library.{filename="libSDL3"});
    array_add(*options.libraries, Library.{filename="libSDL3_shadercross"});
  }

  array_add(*options.source_files, "SDL.h");
  array_add(*options.source_files, "SDL_shadercross.h");

  generate_bindings(options, "module.jai");
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

  #if 0
  {
    build_sdl3(tprint("%/modules/sdl3", project_dir), release_build, workspace);
  }

  set_working_directory(project_dir);

  options.output_executable_name = "penrose";
  options.output_path = "./build";
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

  imports := string.["Basic", "String", "sdl3", "Hash", "File", "Math", "File_Utilities"];
  for imports
  {
    add_build_string(tprint("#import \"%\";\n", it), workspace);
  }

  files := file_list("./");
  for files
  {
    if contains(it, "build.jai") continue;
    if !contains(it, ".jai") continue;
    add_build_file(it, workspace);
  }

  set_build_options_dc(.{do_output=false});
}

#run build();

#import "Basic";
#import "File_Utilities";
#import "String";
#import "Compiler";
#import "Bindings_Generator";