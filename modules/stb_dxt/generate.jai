
generate :: ()
{
  set_build_options_dc(.{do_output = false});
  build_cpp_static_lib("lib/stb_dxt", "source/stb_dxt.cpp");
  options: Generate_Bindings_Options;
  array_add(*options.include_paths, "source");
  array_add(*options.library_search_paths, "lib");
  array_add(*options.libraries, .{filename = "stb_dxt"});
  array_add(*options.source_files, "stb_dxt.h");
  generate_bindings(options, "module.jai");
}

#run generate();

#import "Bindings_Generator";
#import "Basic";
#import "Compiler";
#import "BuildCpp";