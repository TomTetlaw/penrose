// Workspace: Penrose build

//
// String added via add_build_string() from F:/penrose/build.jai:74.
//
#import "Basic";

//
// String added via add_build_string() from F:/penrose/build.jai:74.
//
#import "String";

//
// String added via add_build_string() from F:/penrose/build.jai:74.
//
#import "sdl3";

//
// String added via add_build_string() from F:/penrose/build.jai:74.
//
#import "Hash";

//
// String added via add_build_string() from F:/penrose/build.jai:74.
//
#import "File";

//
// String added via add_build_string() from F:/penrose/build.jai:74.
//
#import "Math";

//
// String added via add_build_string() from F:/penrose/build.jai:74.
//
#import "File_Utilities";

//
// #insert text. Generated from F:/penrose/.//base.jai:231.
//
    temp := first.*;
    first.* = first.*.next_free;
    return temp;

//
// #insert text. Generated from F:/penrose/.//base.jai:292.
//
    return value.next;

//
// #insert text. Generated from F:/penrose/.//base.jai:219.
//
    value.next = first.*;
    first.* = value;

//
// #insert text. Generated from F:/penrose/.//base.jai:219.
//
    value.next_free = first.*;
    first.* = value;

//
// #insert text. Generated from F:/penrose/.//base.jai:250.
//
    if first.*
    {
      last.*.next = value;
      last.* = value;
    }
    else
    {
      first.* = value;
      last.* = value;
    }

//
// #insert text. Generated from F:/penrose/.//base.jai:250.
//
    if first.*
    {
      last.*.next = value;
      last.* = value;
    }
    else
    {
      first.* = value;
      last.* = value;
    }

//
// #insert text. Generated from F:/penrose/.//base.jai:292.
//
    return value.next;

//
// #insert text. Generated from F:/penrose/.//base.jai:292.
//
    return value.next;
