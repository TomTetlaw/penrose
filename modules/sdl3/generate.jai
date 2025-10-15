
generate :: ()
{
  set_build_options_dc(.{do_output = false});
  options: Generate_Bindings_Options;
  array_add(*options.include_paths, "include");
  array_add(*options.include_paths, "include/SDL3");
  array_add(*options.source_files, "SDL.h");
  array_add(*options.source_files, "SDL_shadercross.h");
  array_add(*options.library_search_paths, "lib");
  array_add(*options.libraries, .{filename = "SDL3"});
  array_add(*options.libraries, .{filename = "SDL3_shadercross.lib"});
  generate_bindings(options, "module.jai");
}

#run generate();

#import "Basic";
#import "Bindings_Generator";
#import "Compiler";