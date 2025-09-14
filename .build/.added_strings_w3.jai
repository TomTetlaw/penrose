// Workspace: Penrose build

//
// #insert text. Generated from F:/penrose/.//entry_point.jai:569.
//
    temp := first.*;
    first.* = first.*.next_free;
    return temp;

//
// #insert text. Generated from F:/penrose/.//entry_point.jai:630.
//
    return value.next;

//
// #insert text. Generated from F:/penrose/.//entry_point.jai:630.
//
    return value.next;

//
// #insert text. Generated from F:/penrose/.//entry_point.jai:588.
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
// #insert text. Generated from F:/penrose/.//entry_point.jai:630.
//
    return value.next;

//
// #insert text. Generated from F:/penrose/.//entry_point.jai:557.
//
    value.next_free = first.*;
    first.* = value;
