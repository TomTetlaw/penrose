
generate :: ()
{
  set_build_options_dc(.{do_output = false});
  build_cpp_static_lib("lib/stb_image", "source/stb_image.cpp");
  build_cpp_dynamic_lib("lib/stb_image", "source/stb_image.cpp");
  options: Generate_Bindings_Options;
  array_add(*options.include_paths, "include");
  array_add(*options.library_search_paths, "lib");
  array_add(*options.library_search_paths, "bin");
  array_add(*options.libraries, .{filename = "stb_image"});
  array_add(*options.source_files, "stb_image.h");
  generate_bindings(options, "module.jai");
}

#run generate();

#import "Bindings_Generator";
#import "Basic";
#import "Compiler";
#import "BuildCpp";