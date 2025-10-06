//
// This file was auto-generated using the following command:
//
// jai generate.jai
//



IMGUI_VERSION :: "1.92.4 WIP";
IMGUI_VERSION_NUM :: 19234;

ImTextureID_Invalid :: cast(ImTextureID) 0;

IMGUI_PAYLOAD_TYPE_COLOR_3F :: "_COL3F";
IMGUI_PAYLOAD_TYPE_COLOR_4F :: "_COL4F";

IM_UNICODE_CODEPOINT_INVALID :: 0xFFFD;

IM_UNICODE_CODEPOINT_MAX :: 0xFFFF;

IM_COL32_R_SHIFT :: 0;
IM_COL32_G_SHIFT :: 8;
IM_COL32_B_SHIFT :: 16;
IM_COL32_A_SHIFT :: 24;
IM_COL32_A_MASK :: 0xFF000000;

IM_DRAWLIST_TEX_LINES_WIDTH_MAX :: 32;

ImDrawCallback_ResetRenderState :: cast(ImDrawCallback) (-8);

ImFontAtlasRectId_Invalid :: -1;

// Scalar data types
ImGuiID :: u32;
ImS8 :: s8;
ImU8 :: u8;
ImS16 :: s16;
ImU16 :: u16;
ImS32 :: s32;
ImU32 :: u32;
ImS64 :: s64;
ImU64 :: u64;

ImDrawListSharedData :: struct {}

ImFontAtlasBuilder :: struct {}

ImFontLoader :: struct {}

// Forward declarations: ImGui layer
ImGuiContext :: struct {}

// Enumerations
// - We don't use strongly typed enums much because they add constraints (can't extend in private code, can't store typed in bit fields, extra casting on iteration)
// - Tip: Use your programming IDE navigation facilities on the names in the _central column_ below to find the actual flags/enum lists!
//   - In Visual Studio: CTRL+comma ("Edit.GoToAll") can follow symbols inside comments, whereas CTRL+F12 ("Edit.GoToImplementation") cannot.
//   - In Visual Studio w/ Visual Assist installed: ALT+G ("VAssistX.GoToImplementation") can also follow symbols inside comments.
//   - In VS Code, CLion, etc.: CTRL+click can follow symbols inside comments.
ImGuiDir :: enum s32 {
    None  :: -1;
    Left  :: 0;
    Right :: 1;
    Up    :: 2;
    Down  :: 3;
    COUNT :: 4;

    ImGuiDir_None  :: None;
    ImGuiDir_Left  :: Left;
    ImGuiDir_Right :: Right;
    ImGuiDir_Up    :: Up;
    ImGuiDir_Down  :: Down;
    ImGuiDir_COUNT :: COUNT;
}

// A key identifier (ImGuiKey_XXX or ImGuiMod_XXX value): can represent Keyboard, Mouse and Gamepad values.
// All our named keys are >= 512. Keys value 0 to 511 are left unused and were legacy native/opaque key values (< 1.87).
// Support for legacy keys was completely removed in 1.91.5.
// Read details about the 1.87+ transition : https://github.com/ocornut/imgui/issues/4921
// Note that "Keys" related to physical keys and are not the same concept as input "Characters", the later are submitted via io.AddInputCharacter().
// The keyboard key enum values are named after the keys on a standard US keyboard, and on other keyboard types the keys reported may not match the keycaps.
ImGuiKey :: enum s32 {
    None                :: 0;
    NamedKey_BEGIN      :: 512;

    Tab                 :: 512;
    LeftArrow           :: 513;
    RightArrow          :: 514;
    UpArrow             :: 515;
    DownArrow           :: 516;
    PageUp              :: 517;
    PageDown            :: 518;
    Home                :: 519;
    End                 :: 520;
    Insert              :: 521;
    Delete              :: 522;
    Backspace           :: 523;
    Space               :: 524;
    Enter               :: 525;
    Escape              :: 526;
    LeftCtrl            :: 527;
    LeftShift           :: 528;
    LeftAlt             :: 529;
    LeftSuper           :: 530;
    RightCtrl           :: 531;
    RightShift          :: 532;
    RightAlt            :: 533;
    RightSuper          :: 534;
    Menu                :: 535;
    _0                  :: 536;
    _1                  :: 537;
    _2                  :: 538;
    _3                  :: 539;
    _4                  :: 540;
    _5                  :: 541;
    _6                  :: 542;
    _7                  :: 543;
    _8                  :: 544;
    _9                  :: 545;
    A                   :: 546;
    B                   :: 547;
    C                   :: 548;
    D                   :: 549;
    E                   :: 550;
    F                   :: 551;
    G                   :: 552;
    H                   :: 553;
    I                   :: 554;
    J                   :: 555;
    K                   :: 556;
    L                   :: 557;
    M                   :: 558;
    N                   :: 559;
    O                   :: 560;
    P                   :: 561;
    Q                   :: 562;
    R                   :: 563;
    S                   :: 564;
    T                   :: 565;
    U                   :: 566;
    V                   :: 567;
    W                   :: 568;
    X                   :: 569;
    Y                   :: 570;
    Z                   :: 571;
    F1                  :: 572;
    F2                  :: 573;
    F3                  :: 574;
    F4                  :: 575;
    F5                  :: 576;
    F6                  :: 577;
    F7                  :: 578;
    F8                  :: 579;
    F9                  :: 580;
    F10                 :: 581;
    F11                 :: 582;
    F12                 :: 583;
    F13                 :: 584;
    F14                 :: 585;
    F15                 :: 586;
    F16                 :: 587;
    F17                 :: 588;
    F18                 :: 589;
    F19                 :: 590;
    F20                 :: 591;
    F21                 :: 592;
    F22                 :: 593;
    F23                 :: 594;
    F24                 :: 595;
    Apostrophe          :: 596;
    Comma               :: 597;
    Minus               :: 598;
    Period              :: 599;
    Slash               :: 600;
    Semicolon           :: 601;
    Equal               :: 602;
    LeftBracket         :: 603;
    Backslash           :: 604;
    RightBracket        :: 605;
    GraveAccent         :: 606;
    CapsLock            :: 607;
    ScrollLock          :: 608;
    NumLock             :: 609;
    PrintScreen         :: 610;
    Pause               :: 611;
    Keypad0             :: 612;
    Keypad1             :: 613;
    Keypad2             :: 614;
    Keypad3             :: 615;
    Keypad4             :: 616;
    Keypad5             :: 617;
    Keypad6             :: 618;
    Keypad7             :: 619;
    Keypad8             :: 620;
    Keypad9             :: 621;
    KeypadDecimal       :: 622;
    KeypadDivide        :: 623;
    KeypadMultiply      :: 624;
    KeypadSubtract      :: 625;
    KeypadAdd           :: 626;
    KeypadEnter         :: 627;
    KeypadEqual         :: 628;
    AppBack             :: 629;
    AppForward          :: 630;
    Oem102              :: 631;

    GamepadStart        :: 632;
    GamepadBack         :: 633;
    GamepadFaceLeft     :: 634;
    GamepadFaceRight    :: 635;
    GamepadFaceUp       :: 636;
    GamepadFaceDown     :: 637;
    GamepadDpadLeft     :: 638;
    GamepadDpadRight    :: 639;
    GamepadDpadUp       :: 640;
    GamepadDpadDown     :: 641;
    GamepadL1           :: 642;
    GamepadR1           :: 643;
    GamepadL2           :: 644;
    GamepadR2           :: 645;
    GamepadL3           :: 646;
    GamepadR3           :: 647;
    GamepadLStickLeft   :: 648;
    GamepadLStickRight  :: 649;
    GamepadLStickUp     :: 650;
    GamepadLStickDown   :: 651;
    GamepadRStickLeft   :: 652;
    GamepadRStickRight  :: 653;
    GamepadRStickUp     :: 654;
    GamepadRStickDown   :: 655;

    MouseLeft           :: 656;
    MouseRight          :: 657;
    MouseMiddle         :: 658;
    MouseX1             :: 659;
    MouseX2             :: 660;
    MouseWheelX         :: 661;
    MouseWheelY         :: 662;

    ReservedForModCtrl  :: 663;
    ReservedForModShift :: 664;
    ReservedForModAlt   :: 665;
    ReservedForModSuper :: 666;

    NamedKey_END        :: 667;
    NamedKey_COUNT      :: 155;

    ImGuiMod_None       :: 0;
    ImGuiMod_Ctrl       :: 4096;
    ImGuiMod_Shift      :: 8192;
    ImGuiMod_Alt        :: 16384;
    ImGuiMod_Super      :: 32768;
    ImGuiMod_Mask_      :: 61440;

    COUNT               :: 667;
    ImGuiMod_Shortcut   :: 4096;
    ModCtrl             :: 4096;
    ModShift            :: 8192;
    ModAlt              :: 16384;
    ModSuper            :: 32768;

    ImGuiKey_None                :: None;
    ImGuiKey_NamedKey_BEGIN      :: NamedKey_BEGIN;

    ImGuiKey_Tab                 :: Tab;
    ImGuiKey_LeftArrow           :: LeftArrow;
    ImGuiKey_RightArrow          :: RightArrow;
    ImGuiKey_UpArrow             :: UpArrow;
    ImGuiKey_DownArrow           :: DownArrow;
    ImGuiKey_PageUp              :: PageUp;
    ImGuiKey_PageDown            :: PageDown;
    ImGuiKey_Home                :: Home;
    ImGuiKey_End                 :: End;
    ImGuiKey_Insert              :: Insert;
    ImGuiKey_Delete              :: Delete;
    ImGuiKey_Backspace           :: Backspace;
    ImGuiKey_Space               :: Space;
    ImGuiKey_Enter               :: Enter;
    ImGuiKey_Escape              :: Escape;
    ImGuiKey_LeftCtrl            :: LeftCtrl;
    ImGuiKey_LeftShift           :: LeftShift;
    ImGuiKey_LeftAlt             :: LeftAlt;
    ImGuiKey_LeftSuper           :: LeftSuper;
    ImGuiKey_RightCtrl           :: RightCtrl;
    ImGuiKey_RightShift          :: RightShift;
    ImGuiKey_RightAlt            :: RightAlt;
    ImGuiKey_RightSuper          :: RightSuper;
    ImGuiKey_Menu                :: Menu;
    ImGuiKey_0                   :: _0;
    ImGuiKey_1                   :: _1;
    ImGuiKey_2                   :: _2;
    ImGuiKey_3                   :: _3;
    ImGuiKey_4                   :: _4;
    ImGuiKey_5                   :: _5;
    ImGuiKey_6                   :: _6;
    ImGuiKey_7                   :: _7;
    ImGuiKey_8                   :: _8;
    ImGuiKey_9                   :: _9;
    ImGuiKey_A                   :: A;
    ImGuiKey_B                   :: B;
    ImGuiKey_C                   :: C;
    ImGuiKey_D                   :: D;
    ImGuiKey_E                   :: E;
    ImGuiKey_F                   :: F;
    ImGuiKey_G                   :: G;
    ImGuiKey_H                   :: H;
    ImGuiKey_I                   :: I;
    ImGuiKey_J                   :: J;
    ImGuiKey_K                   :: K;
    ImGuiKey_L                   :: L;
    ImGuiKey_M                   :: M;
    ImGuiKey_N                   :: N;
    ImGuiKey_O                   :: O;
    ImGuiKey_P                   :: P;
    ImGuiKey_Q                   :: Q;
    ImGuiKey_R                   :: R;
    ImGuiKey_S                   :: S;
    ImGuiKey_T                   :: T;
    ImGuiKey_U                   :: U;
    ImGuiKey_V                   :: V;
    ImGuiKey_W                   :: W;
    ImGuiKey_X                   :: X;
    ImGuiKey_Y                   :: Y;
    ImGuiKey_Z                   :: Z;
    ImGuiKey_F1                  :: F1;
    ImGuiKey_F2                  :: F2;
    ImGuiKey_F3                  :: F3;
    ImGuiKey_F4                  :: F4;
    ImGuiKey_F5                  :: F5;
    ImGuiKey_F6                  :: F6;
    ImGuiKey_F7                  :: F7;
    ImGuiKey_F8                  :: F8;
    ImGuiKey_F9                  :: F9;
    ImGuiKey_F10                 :: F10;
    ImGuiKey_F11                 :: F11;
    ImGuiKey_F12                 :: F12;
    ImGuiKey_F13                 :: F13;
    ImGuiKey_F14                 :: F14;
    ImGuiKey_F15                 :: F15;
    ImGuiKey_F16                 :: F16;
    ImGuiKey_F17                 :: F17;
    ImGuiKey_F18                 :: F18;
    ImGuiKey_F19                 :: F19;
    ImGuiKey_F20                 :: F20;
    ImGuiKey_F21                 :: F21;
    ImGuiKey_F22                 :: F22;
    ImGuiKey_F23                 :: F23;
    ImGuiKey_F24                 :: F24;
    ImGuiKey_Apostrophe          :: Apostrophe;
    ImGuiKey_Comma               :: Comma;
    ImGuiKey_Minus               :: Minus;
    ImGuiKey_Period              :: Period;
    ImGuiKey_Slash               :: Slash;
    ImGuiKey_Semicolon           :: Semicolon;
    ImGuiKey_Equal               :: Equal;
    ImGuiKey_LeftBracket         :: LeftBracket;
    ImGuiKey_Backslash           :: Backslash;
    ImGuiKey_RightBracket        :: RightBracket;
    ImGuiKey_GraveAccent         :: GraveAccent;
    ImGuiKey_CapsLock            :: CapsLock;
    ImGuiKey_ScrollLock          :: ScrollLock;
    ImGuiKey_NumLock             :: NumLock;
    ImGuiKey_PrintScreen         :: PrintScreen;
    ImGuiKey_Pause               :: Pause;
    ImGuiKey_Keypad0             :: Keypad0;
    ImGuiKey_Keypad1             :: Keypad1;
    ImGuiKey_Keypad2             :: Keypad2;
    ImGuiKey_Keypad3             :: Keypad3;
    ImGuiKey_Keypad4             :: Keypad4;
    ImGuiKey_Keypad5             :: Keypad5;
    ImGuiKey_Keypad6             :: Keypad6;
    ImGuiKey_Keypad7             :: Keypad7;
    ImGuiKey_Keypad8             :: Keypad8;
    ImGuiKey_Keypad9             :: Keypad9;
    ImGuiKey_KeypadDecimal       :: KeypadDecimal;
    ImGuiKey_KeypadDivide        :: KeypadDivide;
    ImGuiKey_KeypadMultiply      :: KeypadMultiply;
    ImGuiKey_KeypadSubtract      :: KeypadSubtract;
    ImGuiKey_KeypadAdd           :: KeypadAdd;
    ImGuiKey_KeypadEnter         :: KeypadEnter;
    ImGuiKey_KeypadEqual         :: KeypadEqual;
    ImGuiKey_AppBack             :: AppBack;
    ImGuiKey_AppForward          :: AppForward;
    ImGuiKey_Oem102              :: Oem102;

    ImGuiKey_GamepadStart        :: GamepadStart;
    ImGuiKey_GamepadBack         :: GamepadBack;
    ImGuiKey_GamepadFaceLeft     :: GamepadFaceLeft;
    ImGuiKey_GamepadFaceRight    :: GamepadFaceRight;
    ImGuiKey_GamepadFaceUp       :: GamepadFaceUp;
    ImGuiKey_GamepadFaceDown     :: GamepadFaceDown;
    ImGuiKey_GamepadDpadLeft     :: GamepadDpadLeft;
    ImGuiKey_GamepadDpadRight    :: GamepadDpadRight;
    ImGuiKey_GamepadDpadUp       :: GamepadDpadUp;
    ImGuiKey_GamepadDpadDown     :: GamepadDpadDown;
    ImGuiKey_GamepadL1           :: GamepadL1;
    ImGuiKey_GamepadR1           :: GamepadR1;
    ImGuiKey_GamepadL2           :: GamepadL2;
    ImGuiKey_GamepadR2           :: GamepadR2;
    ImGuiKey_GamepadL3           :: GamepadL3;
    ImGuiKey_GamepadR3           :: GamepadR3;
    ImGuiKey_GamepadLStickLeft   :: GamepadLStickLeft;
    ImGuiKey_GamepadLStickRight  :: GamepadLStickRight;
    ImGuiKey_GamepadLStickUp     :: GamepadLStickUp;
    ImGuiKey_GamepadLStickDown   :: GamepadLStickDown;
    ImGuiKey_GamepadRStickLeft   :: GamepadRStickLeft;
    ImGuiKey_GamepadRStickRight  :: GamepadRStickRight;
    ImGuiKey_GamepadRStickUp     :: GamepadRStickUp;
    ImGuiKey_GamepadRStickDown   :: GamepadRStickDown;

    ImGuiKey_MouseLeft           :: MouseLeft;
    ImGuiKey_MouseRight          :: MouseRight;
    ImGuiKey_MouseMiddle         :: MouseMiddle;
    ImGuiKey_MouseX1             :: MouseX1;
    ImGuiKey_MouseX2             :: MouseX2;
    ImGuiKey_MouseWheelX         :: MouseWheelX;
    ImGuiKey_MouseWheelY         :: MouseWheelY;

    ImGuiKey_ReservedForModCtrl  :: ReservedForModCtrl;
    ImGuiKey_ReservedForModShift :: ReservedForModShift;
    ImGuiKey_ReservedForModAlt   :: ReservedForModAlt;
    ImGuiKey_ReservedForModSuper :: ReservedForModSuper;

    ImGuiKey_NamedKey_END        :: NamedKey_END;
    ImGuiKey_NamedKey_COUNT      :: NamedKey_COUNT;

    ImGuiKey_COUNT               :: COUNT;

    ImGuiKey_ModCtrl             :: ModCtrl;
    ImGuiKey_ModShift            :: ModShift;
    ImGuiKey_ModAlt              :: ModAlt;
    ImGuiKey_ModSuper            :: ModSuper;
}

// Enumeration for AddMouseSourceEvent() actual source of Mouse Input data.
// Historically we use "Mouse" terminology everywhere to indicate pointer data, e.g. MousePos, IsMousePressed(), io.AddMousePosEvent()
// But that "Mouse" data can come from different source which occasionally may be useful for application to know about.
// You can submit a change of pointer type using io.AddMouseSourceEvent().
ImGuiMouseSource :: enum s32 {
    Mouse       :: 0;
    TouchScreen :: 1;
    Pen         :: 2;
    COUNT       :: 3;

    ImGuiMouseSource_Mouse       :: Mouse;
    ImGuiMouseSource_TouchScreen :: TouchScreen;
    ImGuiMouseSource_Pen         :: Pen;
    ImGuiMouseSource_COUNT       :: COUNT;
}

// A sorting direction
ImGuiSortDirection :: enum ImU8 {
    None       :: 0;
    Ascending  :: 1;
    Descending :: 2;

    ImGuiSortDirection_None       :: None;
    ImGuiSortDirection_Ascending  :: Ascending;
    ImGuiSortDirection_Descending :: Descending;
}

ImGuiCol :: s32;
ImGuiCond :: s32;
ImGuiDataType :: s32;
ImGuiMouseButton :: s32;
ImGuiMouseCursor :: s32;
ImGuiStyleVar :: s32;
ImGuiTableBgTarget :: s32;

// Flags (declared as int to allow using as flags without overhead, and to not pollute the top of this file)
// - Tip: Use your programming IDE navigation facilities on the names in the _central column_ below to find the actual flags/enum lists!
//   - In Visual Studio: CTRL+comma ("Edit.GoToAll") can follow symbols inside comments, whereas CTRL+F12 ("Edit.GoToImplementation") cannot.
//   - In Visual Studio w/ Visual Assist installed: ALT+G ("VAssistX.GoToImplementation") can also follow symbols inside comments.
//   - In VS Code, CLion, etc.: CTRL+click can follow symbols inside comments.
ImDrawFlags :: s32;
ImDrawListFlags :: s32;
ImDrawTextFlags :: s32;
ImFontFlags :: s32;
ImFontAtlasFlags :: s32;
ImGuiBackendFlags :: s32;
ImGuiButtonFlags :: s32;
ImGuiChildFlags :: s32;
ImGuiColorEditFlags :: s32;
ImGuiConfigFlags :: s32;
ImGuiComboFlags :: s32;
ImGuiDragDropFlags :: s32;
ImGuiFocusedFlags :: s32;
ImGuiHoveredFlags :: s32;
ImGuiInputFlags :: s32;
ImGuiInputTextFlags :: s32;
ImGuiItemFlags :: s32;
ImGuiKeyChord :: s32;
ImGuiListClipperFlags :: s32;
ImGuiPopupFlags :: s32;
ImGuiMultiSelectFlags :: s32;
ImGuiSelectableFlags :: s32;
ImGuiSliderFlags :: s32;
ImGuiTabBarFlags :: s32;
ImGuiTabItemFlags :: s32;
ImGuiTableFlags :: s32;
ImGuiTableColumnFlags :: s32;
ImGuiTableRowFlags :: s32;
ImGuiTreeNodeFlags :: s32;
ImGuiViewportFlags :: s32;
ImGuiWindowFlags :: s32;

// Character types
// (we generally use UTF-8 encoded string in the API. This is storage specifically for a decoded character used for keyboard input and display)
ImWchar32 :: u32;
ImWchar16 :: u16;

ImWchar :: ImWchar16;

// Multi-Selection item index or identifier when using BeginMultiSelect()
// - Used by SetNextItemSelectionUserData() + and inside ImGuiMultiSelectIO structure.
// - Most users are likely to use this store an item INDEX but this may be used to store a POINTER/ID as well. Read comments near ImGuiMultiSelectIO for details.
ImGuiSelectionUserData :: ImS64;

// Callback and functions types
ImGuiInputTextCallback :: #type (data: *ImGuiInputTextCallbackData) -> s32 #c_call;
ImGuiSizeCallback :: #type (data: *ImGuiSizeCallbackData) -> void #c_call;
ImGuiMemAllocFunc :: #type (sz: u64, user_data: *void) -> *void #c_call;
ImGuiMemFreeFunc :: #type (ptr: *void, user_data: *void) -> void #c_call;

// ImVec2: 2D vector used to store positions, sizes etc. [Compile-time configurable type]
// - This is a frequently used type in the API. Consider using IM_VEC2_CLASS_EXTRA to create implicit cast from/to our preferred type.
// - Add '#define IMGUI_DEFINE_MATH_OPERATORS' before including this file (or in imconfig.h) to access courtesy maths operators for ImVec2 and ImVec4.
ImVec2 :: struct {
    x: float;
    y: float;
    Constructor :: (this: *ImVec2) -> void #cpp_method #foreign imgui "??0ImVec2@@QEAA@XZ";
    Constructor :: (this: *ImVec2, _x: float, _y: float) -> void #cpp_method #foreign imgui "??0ImVec2@@QEAA@MM@Z";
}
operator[] :: (this: *ImVec2, idx: u64) -> *float #cpp_method #foreign imgui "??AImVec2@@QEAAAEAM_K@Z";
operator[] :: (this: ImVec2, idx: u64) -> *float #no_context {
    return operator[](*this, idx);
}
operator_brackets_1 :: (this: /*const*/ *ImVec2, idx: u64) -> float #cpp_method #foreign imgui "??AImVec2@@QEBAM_K@Z";
operator_brackets_1 :: (this: ImVec2, idx: u64) -> float #no_context {
    return operator_brackets_1(*this, idx);
}

// ImVec4: 4D vector used to store clipping rectangles, colors etc. [Compile-time configurable type]
ImVec4 :: struct {
    x: float;
    y: float;
    z: float;
    w: float;
    Constructor :: (this: *ImVec4) -> void #cpp_method #foreign imgui "??0ImVec4@@QEAA@XZ";
    Constructor :: (this: *ImVec4, _x: float, _y: float, _z: float, _w: float) -> void #cpp_method #foreign imgui "??0ImVec4@@QEAA@MMMM@Z";
}

ImTextureID :: ImU64;

// ImTextureRef = higher-level identifier for a texture. Store a ImTextureID _or_ a ImTextureData*.
// The identifier is valid even before the texture has been uploaded to the GPU/graphics system.
// This is what gets passed to functions such as `ImGui::Image()`, `ImDrawList::AddImage()`.
// This is what gets stored in draw commands (`ImDrawCmd`) to identify a texture during rendering.
// - When a texture is created by user code (e.g. custom images), we directly stores the low-level ImTextureID.
//   Because of this, when displaying your own texture you are likely to ever only manage ImTextureID values on your side.
// - When a texture is created by the backend, we stores a ImTextureData* which becomes an indirection
//   to extract the ImTextureID value during rendering, after texture upload has happened.
// - To create a ImTextureRef from a ImTextureData you can use ImTextureData::GetTexRef().
//   We intentionally do not provide an ImTextureRef constructor for this: we don't expect this
//   to be frequently useful to the end-user, and it would be erroneously called by many legacy code.
// - If you want to bind the current atlas when using custom rectangle, you can use io.Fonts->TexRef.
// - Binding generators for languages such as C (which don't have constructors), should provide a helper, e.g.
//      inline ImTextureRef ImTextureRefFromID(ImTextureID tex_id) { ImTextureRef tex_ref = { ._TexData = NULL, .TexID = tex_id }; return tex_ref; }
// In 1.92 we changed most drawing functions using ImTextureID to use ImTextureRef.
// We intentionally do not provide an implicit ImTextureRef -> ImTextureID cast operator because it is technically lossy to convert ImTextureRef to ImTextureID before rendering.
ImTextureRef :: struct {
    Constructor :: (this: *ImTextureRef) -> void #cpp_method #foreign imgui "??0ImTextureRef@@QEAA@XZ";

    // This is provided for consistency (but we don't actually use this)
    GetTexID :: (this: *ImTextureRef) -> ImTextureID #cpp_method #foreign imgui "?GetTexID@ImTextureRef@@QEBA_KXZ";

    _TexData: *ImTextureData; //      A texture, generally owned by a ImFontAtlas. Will convert to ImTextureID during render loop, after texture has been uploaded.
    _TexID:   ImTextureID; // _OR_ Low-level backend texture identifier, if already uploaded or created by user/app. Generally provided to e.g. ImGui::Image() calls.
}

//-----------------------------------------------------------------------------
// [SECTION] Dear ImGui end-user API functions
// (Note that ImGui:: being a namespace, you can add extra ImGui:: functions in your own separate file. Please don't modify imgui source files!)
//-----------------------------------------------------------------------------
ImGui :: struct {
    // Context creation and access
    // - Each context create its own ImFontAtlas by default. You may instance one yourself and pass it to CreateContext() to share a font atlas between contexts.
    // - DLL users: heaps and globals are not shared across DLL boundaries! You will need to call SetCurrentContext() + SetAllocatorFunctions()
    //   for each static/DLL boundary you are calling from. Read "Context and Memory Allocators" section of imgui.cpp for details.
    CreateContext :: (shared_font_atlas: *ImFontAtlas = null) -> *ImGuiContext #foreign imgui "?CreateContext@ImGui@@YAPEAUImGuiContext@@PEAUImFontAtlas@@@Z";
    DestroyContext :: (ctx: *ImGuiContext = null) -> void #foreign imgui "?DestroyContext@ImGui@@YAXPEAUImGuiContext@@@Z";
    GetCurrentContext :: () -> *ImGuiContext #foreign imgui "?GetCurrentContext@ImGui@@YAPEAUImGuiContext@@XZ";
    SetCurrentContext :: (ctx: *ImGuiContext) -> void #foreign imgui "?SetCurrentContext@ImGui@@YAXPEAUImGuiContext@@@Z";

    // Main
    GetIO :: () -> *ImGuiIO #foreign imgui "?GetIO@ImGui@@YAAEAUImGuiIO@@XZ";
    GetPlatformIO :: () -> *ImGuiPlatformIO #foreign imgui "?GetPlatformIO@ImGui@@YAAEAUImGuiPlatformIO@@XZ";
    GetStyle :: () -> *ImGuiStyle #foreign imgui "?GetStyle@ImGui@@YAAEAUImGuiStyle@@XZ";
    NewFrame :: () -> void #foreign imgui "?NewFrame@ImGui@@YAXXZ";
    EndFrame :: () -> void #foreign imgui "?EndFrame@ImGui@@YAXXZ";
    Render :: () -> void #foreign imgui "?Render@ImGui@@YAXXZ";
    GetDrawData :: () -> *ImDrawData #foreign imgui "?GetDrawData@ImGui@@YAPEAUImDrawData@@XZ";

    // Demo, Debug, Information
    ShowDemoWindow :: (p_open: *bool = null) -> void #foreign imgui "?ShowDemoWindow@ImGui@@YAXPEA_N@Z";
    ShowMetricsWindow :: (p_open: *bool = null) -> void #foreign imgui "?ShowMetricsWindow@ImGui@@YAXPEA_N@Z";
    ShowDebugLogWindow :: (p_open: *bool = null) -> void #foreign imgui "?ShowDebugLogWindow@ImGui@@YAXPEA_N@Z";
    ShowIDStackToolWindow :: (p_open: *bool = null) -> void #foreign imgui "?ShowIDStackToolWindow@ImGui@@YAXPEA_N@Z";
    ShowAboutWindow :: (p_open: *bool = null) -> void #foreign imgui "?ShowAboutWindow@ImGui@@YAXPEA_N@Z";
    ShowStyleEditor :: (ref: *ImGuiStyle = null) -> void #foreign imgui "?ShowStyleEditor@ImGui@@YAXPEAUImGuiStyle@@@Z";
    ShowStyleSelector :: (label: *u8) -> bool #foreign imgui "?ShowStyleSelector@ImGui@@YA_NPEBD@Z";
    ShowFontSelector :: (label: *u8) -> void #foreign imgui "?ShowFontSelector@ImGui@@YAXPEBD@Z";
    ShowUserGuide :: () -> void #foreign imgui "?ShowUserGuide@ImGui@@YAXXZ";
    GetVersion :: () -> *u8 #foreign imgui "?GetVersion@ImGui@@YAPEBDXZ";

    // Styles
    StyleColorsDark :: (dst: *ImGuiStyle = null) -> void #foreign imgui "?StyleColorsDark@ImGui@@YAXPEAUImGuiStyle@@@Z";
    StyleColorsLight :: (dst: *ImGuiStyle = null) -> void #foreign imgui "?StyleColorsLight@ImGui@@YAXPEAUImGuiStyle@@@Z";
    StyleColorsClassic :: (dst: *ImGuiStyle = null) -> void #foreign imgui "?StyleColorsClassic@ImGui@@YAXPEAUImGuiStyle@@@Z";

    // Windows
    // - Begin() = push window to the stack and start appending to it. End() = pop window from the stack.
    // - Passing 'bool* p_open != NULL' shows a window-closing widget in the upper-right corner of the window,
    //   which clicking will set the boolean to false when clicked.
    // - You may append multiple times to the same window during the same frame by calling Begin()/End() pairs multiple times.
    //   Some information such as 'flags' or 'p_open' will only be considered by the first call to Begin().
    // - Begin() return false to indicate the window is collapsed or fully clipped, so you may early out and omit submitting
    //   anything to the window. Always call a matching End() for each Begin() call, regardless of its return value!
    //   [Important: due to legacy reason, Begin/End and BeginChild/EndChild are inconsistent with all other functions
    //    such as BeginMenu/EndMenu, BeginPopup/EndPopup, etc. where the EndXXX call should only be called if the corresponding
    //    BeginXXX function returned true. Begin and BeginChild are the only odd ones out. Will be fixed in a future update.]
    // - Note that the bottom of window stack always contains a window called "Debug".
    Begin :: (name: *u8, p_open: *bool = null, flags: ImGuiWindowFlags = 0) -> bool #foreign imgui "?Begin@ImGui@@YA_NPEBDPEA_NH@Z";
    End :: () -> void #foreign imgui "?End@ImGui@@YAXXZ";

    // Child Windows
    // - Use child windows to begin into a self-contained independent scrolling/clipping regions within a host window. Child windows can embed their own child.
    // - Before 1.90 (November 2023), the "ImGuiChildFlags child_flags = 0" parameter was "bool border = false".
    //   This API is backward compatible with old code, as we guarantee that ImGuiChildFlags_Borders == true.
    //   Consider updating your old code:
    //      BeginChild("Name", size, false)   -> Begin("Name", size, 0); or Begin("Name", size, ImGuiChildFlags_None);
    //      BeginChild("Name", size, true)    -> Begin("Name", size, ImGuiChildFlags_Borders);
    // - Manual sizing (each axis can use a different setting e.g. ImVec2(0.0f, 400.0f)):
    //     == 0.0f: use remaining parent window size for this axis.
    //      > 0.0f: use specified size for this axis.
    //      < 0.0f: right/bottom-align to specified distance from available content boundaries.
    // - Specifying ImGuiChildFlags_AutoResizeX or ImGuiChildFlags_AutoResizeY makes the sizing automatic based on child contents.
    //   Combining both ImGuiChildFlags_AutoResizeX _and_ ImGuiChildFlags_AutoResizeY defeats purpose of a scrolling region and is NOT recommended.
    // - BeginChild() returns false to indicate the window is collapsed or fully clipped, so you may early out and omit submitting
    //   anything to the window. Always call a matching EndChild() for each BeginChild() call, regardless of its return value.
    //   [Important: due to legacy reason, Begin/End and BeginChild/EndChild are inconsistent with all other functions
    //    such as BeginMenu/EndMenu, BeginPopup/EndPopup, etc. where the EndXXX call should only be called if the corresponding
    //    BeginXXX function returned true. Begin and BeginChild are the only odd ones out. Will be fixed in a future update.]
    BeginChild :: (str_id: *u8, size: *ImVec2, child_flags: ImGuiChildFlags = 0, window_flags: ImGuiWindowFlags = 0) -> bool #foreign imgui "?BeginChild@ImGui@@YA_NPEBDAEBUImVec2@@HH@Z";
    BeginChild :: (str_id: *u8, size: ImVec2 = ImVec2.{0, 0}, child_flags: ImGuiChildFlags = 0, window_flags: ImGuiWindowFlags = 0) -> bool #no_context {
        return BeginChild(str_id, *size, child_flags, window_flags);
    }
    BeginChild :: (id: ImGuiID, size: *ImVec2, child_flags: ImGuiChildFlags = 0, window_flags: ImGuiWindowFlags = 0) -> bool #foreign imgui "?BeginChild@ImGui@@YA_NIAEBUImVec2@@HH@Z";
    BeginChild :: (id: ImGuiID, size: ImVec2 = ImVec2.{0, 0}, child_flags: ImGuiChildFlags = 0, window_flags: ImGuiWindowFlags = 0) -> bool #no_context {
        return BeginChild(id, *size, child_flags, window_flags);
    }
    EndChild :: () -> void #foreign imgui "?EndChild@ImGui@@YAXXZ";

    // Windows Utilities
    // - 'current window' = the window we are appending into while inside a Begin()/End() block. 'next window' = next window we will Begin() into.
    IsWindowAppearing :: () -> bool #foreign imgui "?IsWindowAppearing@ImGui@@YA_NXZ";
    IsWindowCollapsed :: () -> bool #foreign imgui "?IsWindowCollapsed@ImGui@@YA_NXZ";
    IsWindowFocused :: (flags: ImGuiFocusedFlags = 0) -> bool #foreign imgui "?IsWindowFocused@ImGui@@YA_NH@Z";
    IsWindowHovered :: (flags: ImGuiHoveredFlags = 0) -> bool #foreign imgui "?IsWindowHovered@ImGui@@YA_NH@Z";
    GetWindowDrawList :: () -> *ImDrawList #foreign imgui "?GetWindowDrawList@ImGui@@YAPEAUImDrawList@@XZ";
    GetWindowPos :: () -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetWindowPos@ImGui@@YA?AUImVec2@@XZ";
    GetWindowSize :: () -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetWindowSize@ImGui@@YA?AUImVec2@@XZ";
    GetWindowWidth :: () -> float #foreign imgui "?GetWindowWidth@ImGui@@YAMXZ";
    GetWindowHeight :: () -> float #foreign imgui "?GetWindowHeight@ImGui@@YAMXZ";

    // Window manipulation
    // - Prefer using SetNextXXX functions (before Begin) rather that SetXXX functions (after Begin).
    SetNextWindowPos :: (pos: *ImVec2, cond: ImGuiCond = 0, pivot: *ImVec2) -> void #foreign imgui "?SetNextWindowPos@ImGui@@YAXAEBUImVec2@@H0@Z";
    SetNextWindowPos :: (pos: ImVec2, cond: ImGuiCond = 0, pivot: ImVec2 = ImVec2.{0, 0}) #no_context {
        SetNextWindowPos(*pos, cond, *pivot);
    }
    SetNextWindowSize :: (size: *ImVec2, cond: ImGuiCond = 0) -> void #foreign imgui "?SetNextWindowSize@ImGui@@YAXAEBUImVec2@@H@Z";
    SetNextWindowSize :: (size: ImVec2, cond: ImGuiCond = 0) #no_context {
        SetNextWindowSize(*size, cond);
    }
    SetNextWindowSizeConstraints :: (size_min: *ImVec2, size_max: *ImVec2, custom_callback: ImGuiSizeCallback = null, custom_callback_data: *void = null) -> void #foreign imgui "?SetNextWindowSizeConstraints@ImGui@@YAXAEBUImVec2@@0P6AXPEAUImGuiSizeCallbackData@@@ZPEAX@Z";
    SetNextWindowSizeConstraints :: (size_min: ImVec2, size_max: ImVec2, custom_callback: ImGuiSizeCallback = null, custom_callback_data: *void = null) #no_context {
        SetNextWindowSizeConstraints(*size_min, *size_max, custom_callback, custom_callback_data);
    }
    SetNextWindowContentSize :: (size: *ImVec2) -> void #foreign imgui "?SetNextWindowContentSize@ImGui@@YAXAEBUImVec2@@@Z";
    SetNextWindowContentSize :: (size: ImVec2) #no_context {
        SetNextWindowContentSize(*size);
    }
    SetNextWindowCollapsed :: (collapsed: bool, cond: ImGuiCond = 0) -> void #foreign imgui "?SetNextWindowCollapsed@ImGui@@YAX_NH@Z";
    SetNextWindowFocus :: () -> void #foreign imgui "?SetNextWindowFocus@ImGui@@YAXXZ";
    SetNextWindowScroll :: (scroll: *ImVec2) -> void #foreign imgui "?SetNextWindowScroll@ImGui@@YAXAEBUImVec2@@@Z";
    SetNextWindowScroll :: (scroll: ImVec2) #no_context {
        SetNextWindowScroll(*scroll);
    }
    SetNextWindowBgAlpha :: (alpha: float) -> void #foreign imgui "?SetNextWindowBgAlpha@ImGui@@YAXM@Z";
    SetWindowPos :: (pos: *ImVec2, cond: ImGuiCond = 0) -> void #foreign imgui "?SetWindowPos@ImGui@@YAXAEBUImVec2@@H@Z";
    SetWindowPos :: (pos: ImVec2, cond: ImGuiCond = 0) #no_context {
        SetWindowPos(*pos, cond);
    }
    SetWindowSize :: (size: *ImVec2, cond: ImGuiCond = 0) -> void #foreign imgui "?SetWindowSize@ImGui@@YAXAEBUImVec2@@H@Z";
    SetWindowSize :: (size: ImVec2, cond: ImGuiCond = 0) #no_context {
        SetWindowSize(*size, cond);
    }
    SetWindowCollapsed :: (collapsed: bool, cond: ImGuiCond = 0) -> void #foreign imgui "?SetWindowCollapsed@ImGui@@YAX_NH@Z";
    SetWindowFocus :: () -> void #foreign imgui "?SetWindowFocus@ImGui@@YAXXZ";
    SetWindowPos :: (name: *u8, pos: *ImVec2, cond: ImGuiCond = 0) -> void #foreign imgui "?SetWindowPos@ImGui@@YAXPEBDAEBUImVec2@@H@Z";
    SetWindowPos :: (name: *u8, pos: ImVec2, cond: ImGuiCond = 0) #no_context {
        SetWindowPos(name, *pos, cond);
    }
    SetWindowSize :: (name: *u8, size: *ImVec2, cond: ImGuiCond = 0) -> void #foreign imgui "?SetWindowSize@ImGui@@YAXPEBDAEBUImVec2@@H@Z";
    SetWindowSize :: (name: *u8, size: ImVec2, cond: ImGuiCond = 0) #no_context {
        SetWindowSize(name, *size, cond);
    }
    SetWindowCollapsed :: (name: *u8, collapsed: bool, cond: ImGuiCond = 0) -> void #foreign imgui "?SetWindowCollapsed@ImGui@@YAXPEBD_NH@Z";
    SetWindowFocus :: (name: *u8) -> void #foreign imgui "?SetWindowFocus@ImGui@@YAXPEBD@Z";

    // Windows Scrolling
    // - Any change of Scroll will be applied at the beginning of next frame in the first call to Begin().
    // - You may instead use SetNextWindowScroll() prior to calling Begin() to avoid this delay, as an alternative to using SetScrollX()/SetScrollY().
    GetScrollX :: () -> float #foreign imgui "?GetScrollX@ImGui@@YAMXZ";
    GetScrollY :: () -> float #foreign imgui "?GetScrollY@ImGui@@YAMXZ";
    SetScrollX :: (scroll_x: float) -> void #foreign imgui "?SetScrollX@ImGui@@YAXM@Z";
    SetScrollY :: (scroll_y: float) -> void #foreign imgui "?SetScrollY@ImGui@@YAXM@Z";
    GetScrollMaxX :: () -> float #foreign imgui "?GetScrollMaxX@ImGui@@YAMXZ";
    GetScrollMaxY :: () -> float #foreign imgui "?GetScrollMaxY@ImGui@@YAMXZ";
    SetScrollHereX :: (center_x_ratio: float = 0.5) -> void #foreign imgui "?SetScrollHereX@ImGui@@YAXM@Z";
    SetScrollHereY :: (center_y_ratio: float = 0.5) -> void #foreign imgui "?SetScrollHereY@ImGui@@YAXM@Z";
    SetScrollFromPosX :: (local_x: float, center_x_ratio: float = 0.5) -> void #foreign imgui "?SetScrollFromPosX@ImGui@@YAXMM@Z";
    SetScrollFromPosY :: (local_y: float, center_y_ratio: float = 0.5) -> void #foreign imgui "?SetScrollFromPosY@ImGui@@YAXMM@Z";

    // Parameters stacks (font)
    //  - PushFont(font, 0.0f)                       // Change font and keep current size
    //  - PushFont(NULL, 20.0f)                      // Keep font and change current size
    //  - PushFont(font, 20.0f)                      // Change font and set size to 20.0f
    //  - PushFont(font, style.FontSizeBase * 2.0f)  // Change font and set size to be twice bigger than current size.
    //  - PushFont(font, font->LegacySize)           // Change font and set size to size passed to AddFontXXX() function. Same as pre-1.92 behavior.
    // *IMPORTANT* before 1.92, fonts had a single size. They can now be dynamically be adjusted.
    //  - In 1.92 we have REMOVED the single parameter version of PushFont() because it seems like the easiest way to provide an error-proof transition.
    //  - PushFont(font) before 1.92 = PushFont(font, font->LegacySize) after 1.92          // Use default font size as passed to AddFontXXX() function.
    // *IMPORTANT* global scale factors are applied over the provided size.
    //  - Global scale factors are: 'style.FontScaleMain', 'style.FontScaleDpi' and maybe more.
    // -  If you want to apply a factor to the _current_ font size:
    //  - CORRECT:   PushFont(NULL, style.FontSizeBase)         // use current unscaled size    == does nothing
    //  - CORRECT:   PushFont(NULL, style.FontSizeBase * 2.0f)  // use current unscaled size x2 == make text twice bigger
    //  - INCORRECT: PushFont(NULL, GetFontSize())              // INCORRECT! using size after global factors already applied == GLOBAL SCALING FACTORS WILL APPLY TWICE!
    //  - INCORRECT: PushFont(NULL, GetFontSize() * 2.0f)       // INCORRECT! using size after global factors already applied == GLOBAL SCALING FACTORS WILL APPLY TWICE!
    PushFont :: (font: *ImFont, font_size_base_unscaled: float) -> void #foreign imgui "?PushFont@ImGui@@YAXPEAUImFont@@M@Z";
    PopFont :: () -> void #foreign imgui "?PopFont@ImGui@@YAXXZ";
    GetFont :: () -> *ImFont #foreign imgui "?GetFont@ImGui@@YAPEAUImFont@@XZ";
    GetFontSize :: () -> float #foreign imgui "?GetFontSize@ImGui@@YAMXZ";
    GetFontBaked :: () -> *ImFontBaked #foreign imgui "?GetFontBaked@ImGui@@YAPEAUImFontBaked@@XZ";

    // Parameters stacks (shared)
    PushStyleColor :: (idx: ImGuiCol, col: ImU32) -> void #foreign imgui "?PushStyleColor@ImGui@@YAXHI@Z";
    PushStyleColor :: (idx: ImGuiCol, col: *ImVec4) -> void #foreign imgui "?PushStyleColor@ImGui@@YAXHAEBUImVec4@@@Z";
    PushStyleColor :: (idx: ImGuiCol, col: ImVec4) #no_context {
        PushStyleColor(idx, *col);
    }
    PopStyleColor :: (count: s32 = 1) -> void #foreign imgui "?PopStyleColor@ImGui@@YAXH@Z";
    PushStyleVar :: (idx: ImGuiStyleVar, val: float) -> void #foreign imgui "?PushStyleVar@ImGui@@YAXHM@Z";
    PushStyleVar :: (idx: ImGuiStyleVar, val: *ImVec2) -> void #foreign imgui "?PushStyleVar@ImGui@@YAXHAEBUImVec2@@@Z";
    PushStyleVar :: (idx: ImGuiStyleVar, val: ImVec2) #no_context {
        PushStyleVar(idx, *val);
    }
    PushStyleVarX :: (idx: ImGuiStyleVar, val_x: float) -> void #foreign imgui "?PushStyleVarX@ImGui@@YAXHM@Z";
    PushStyleVarY :: (idx: ImGuiStyleVar, val_y: float) -> void #foreign imgui "?PushStyleVarY@ImGui@@YAXHM@Z";
    PopStyleVar :: (count: s32 = 1) -> void #foreign imgui "?PopStyleVar@ImGui@@YAXH@Z";
    PushItemFlag :: (option: ImGuiItemFlags, enabled: bool) -> void #foreign imgui "?PushItemFlag@ImGui@@YAXH_N@Z";
    PopItemFlag :: () -> void #foreign imgui "?PopItemFlag@ImGui@@YAXXZ";

    // Parameters stacks (current window)
    PushItemWidth :: (item_width: float) -> void #foreign imgui "?PushItemWidth@ImGui@@YAXM@Z";
    PopItemWidth :: () -> void #foreign imgui "?PopItemWidth@ImGui@@YAXXZ";
    SetNextItemWidth :: (item_width: float) -> void #foreign imgui "?SetNextItemWidth@ImGui@@YAXM@Z";
    CalcItemWidth :: () -> float #foreign imgui "?CalcItemWidth@ImGui@@YAMXZ";
    PushTextWrapPos :: (wrap_local_pos_x: float = 0.0) -> void #foreign imgui "?PushTextWrapPos@ImGui@@YAXM@Z";
    PopTextWrapPos :: () -> void #foreign imgui "?PopTextWrapPos@ImGui@@YAXXZ";

    // Style read access
    // - Use the ShowStyleEditor() function to interactively see/edit the colors.
    GetFontTexUvWhitePixel :: () -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetFontTexUvWhitePixel@ImGui@@YA?AUImVec2@@XZ";
    GetColorU32 :: (idx: ImGuiCol, alpha_mul: float = 1.0) -> ImU32 #foreign imgui "?GetColorU32@ImGui@@YAIHM@Z";
    GetColorU32 :: (col: *ImVec4) -> ImU32 #foreign imgui "?GetColorU32@ImGui@@YAIAEBUImVec4@@@Z";
    GetColorU32 :: (col: ImVec4) -> ImU32 #no_context {
        return GetColorU32(*col);
    }
    GetColorU32 :: (col: ImU32, alpha_mul: float = 1.0) -> ImU32 #foreign imgui "?GetColorU32@ImGui@@YAIIM@Z";
    GetStyleColorVec4 :: (idx: ImGuiCol) -> *ImVec4 #foreign imgui "?GetStyleColorVec4@ImGui@@YAAEBUImVec4@@H@Z";

    // Layout cursor positioning
    // - By "cursor" we mean the current output position.
    // - The typical widget behavior is to output themselves at the current cursor position, then move the cursor one line down.
    // - You can call SameLine() between widgets to undo the last carriage return and output at the right of the preceding widget.
    // - YOU CAN DO 99% OF WHAT YOU NEED WITH ONLY GetCursorScreenPos() and GetContentRegionAvail().
    // - Attention! We currently have inconsistencies between window-local and absolute positions we will aim to fix with future API:
    //    - Absolute coordinate:        GetCursorScreenPos(), SetCursorScreenPos(), all ImDrawList:: functions. -> this is the preferred way forward.
    //    - Window-local coordinates:   SameLine(offset), GetCursorPos(), SetCursorPos(), GetCursorStartPos(), PushTextWrapPos()
    //    - Window-local coordinates:   GetContentRegionMax(), GetWindowContentRegionMin(), GetWindowContentRegionMax() --> all obsoleted. YOU DON'T NEED THEM.
    // - GetCursorScreenPos() = GetCursorPos() + GetWindowPos(). GetWindowPos() is almost only ever useful to convert from window-local to absolute coordinates. Try not to use it.
    GetCursorScreenPos :: () -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetCursorScreenPos@ImGui@@YA?AUImVec2@@XZ";
    SetCursorScreenPos :: (pos: *ImVec2) -> void #foreign imgui "?SetCursorScreenPos@ImGui@@YAXAEBUImVec2@@@Z";
    SetCursorScreenPos :: (pos: ImVec2) #no_context {
        SetCursorScreenPos(*pos);
    }
    GetContentRegionAvail :: () -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetContentRegionAvail@ImGui@@YA?AUImVec2@@XZ";
    GetCursorPos :: () -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetCursorPos@ImGui@@YA?AUImVec2@@XZ";
    GetCursorPosX :: () -> float #foreign imgui "?GetCursorPosX@ImGui@@YAMXZ";
    GetCursorPosY :: () -> float #foreign imgui "?GetCursorPosY@ImGui@@YAMXZ";
    SetCursorPos :: (local_pos: *ImVec2) -> void #foreign imgui "?SetCursorPos@ImGui@@YAXAEBUImVec2@@@Z";
    SetCursorPos :: (local_pos: ImVec2) #no_context {
        SetCursorPos(*local_pos);
    }
    SetCursorPosX :: (local_x: float) -> void #foreign imgui "?SetCursorPosX@ImGui@@YAXM@Z";
    SetCursorPosY :: (local_y: float) -> void #foreign imgui "?SetCursorPosY@ImGui@@YAXM@Z";
    GetCursorStartPos :: () -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetCursorStartPos@ImGui@@YA?AUImVec2@@XZ";

    // Other layout functions
    Separator :: () -> void #foreign imgui "?Separator@ImGui@@YAXXZ";
    SameLine :: (offset_from_start_x: float = 0.0, spacing: float = -1.0) -> void #foreign imgui "?SameLine@ImGui@@YAXMM@Z";
    NewLine :: () -> void #foreign imgui "?NewLine@ImGui@@YAXXZ";
    Spacing :: () -> void #foreign imgui "?Spacing@ImGui@@YAXXZ";
    Dummy :: (size: *ImVec2) -> void #foreign imgui "?Dummy@ImGui@@YAXAEBUImVec2@@@Z";
    Dummy :: (size: ImVec2) #no_context {
        Dummy(*size);
    }
    Indent :: (indent_w: float = 0.0) -> void #foreign imgui "?Indent@ImGui@@YAXM@Z";
    Unindent :: (indent_w: float = 0.0) -> void #foreign imgui "?Unindent@ImGui@@YAXM@Z";
    BeginGroup :: () -> void #foreign imgui "?BeginGroup@ImGui@@YAXXZ";
    EndGroup :: () -> void #foreign imgui "?EndGroup@ImGui@@YAXXZ";
    AlignTextToFramePadding :: () -> void #foreign imgui "?AlignTextToFramePadding@ImGui@@YAXXZ";
    GetTextLineHeight :: () -> float #foreign imgui "?GetTextLineHeight@ImGui@@YAMXZ";
    GetTextLineHeightWithSpacing :: () -> float #foreign imgui "?GetTextLineHeightWithSpacing@ImGui@@YAMXZ";
    GetFrameHeight :: () -> float #foreign imgui "?GetFrameHeight@ImGui@@YAMXZ";
    GetFrameHeightWithSpacing :: () -> float #foreign imgui "?GetFrameHeightWithSpacing@ImGui@@YAMXZ";

    // ID stack/scopes
    // Read the FAQ (docs/FAQ.md or http://dearimgui.com/faq) for more details about how ID are handled in dear imgui.
    // - Those questions are answered and impacted by understanding of the ID stack system:
    //   - "Q: Why is my widget not reacting when I click on it?"
    //   - "Q: How can I have widgets with an empty label?"
    //   - "Q: How can I have multiple widgets with the same label?"
    // - Short version: ID are hashes of the entire ID stack. If you are creating widgets in a loop you most likely
    //   want to push a unique identifier (e.g. object pointer, loop index) to uniquely differentiate them.
    // - You can also use the "Label##foobar" syntax within widget label to distinguish them from each others.
    // - In this header file we use the "label"/"name" terminology to denote a string that will be displayed + used as an ID,
    //   whereas "str_id" denote a string that is only used as an ID and not normally displayed.
    PushID :: (str_id: *u8) -> void #foreign imgui "?PushID@ImGui@@YAXPEBD@Z";
    PushID :: (str_id_begin: *u8, str_id_end: *u8) -> void #foreign imgui "?PushID@ImGui@@YAXPEBD0@Z";
    PushID :: (ptr_id: *void) -> void #foreign imgui "?PushID@ImGui@@YAXPEBX@Z";
    PushID :: (int_id: s32) -> void #foreign imgui "?PushID@ImGui@@YAXH@Z";
    PopID :: () -> void #foreign imgui "?PopID@ImGui@@YAXXZ";
    GetID :: (str_id: *u8) -> ImGuiID #foreign imgui "?GetID@ImGui@@YAIPEBD@Z";
    GetID :: (str_id_begin: *u8, str_id_end: *u8) -> ImGuiID #foreign imgui "?GetID@ImGui@@YAIPEBD0@Z";
    GetID :: (ptr_id: *void) -> ImGuiID #foreign imgui "?GetID@ImGui@@YAIPEBX@Z";
    GetID :: (int_id: s32) -> ImGuiID #foreign imgui "?GetID@ImGui@@YAIH@Z";

    // Widgets: Text
    TextUnformatted :: (text: *u8, text_end: *u8 = null) -> void #foreign imgui "?TextUnformatted@ImGui@@YAXPEBD0@Z";
    Text_CFormat :: (fmt: *u8, __args: ..Any) -> void #foreign imgui "?Text@ImGui@@YAXPEBDZZ";
    Text :: (fmt: string, __args: ..Any) {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, fmt, ..__args);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        Text_CFormat("%s", formatted_text.data);
    } @PrintLike

    TextColored_CFormat :: (col: *ImVec4, fmt: *u8, __args: ..Any) -> void #foreign imgui "?TextColored@ImGui@@YAXAEBUImVec4@@PEBDZZ";
    TextColored :: (col: ImVec4, fmt: string, __args: ..Any) {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, fmt, ..__args);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        TextColored_CFormat(*col, "%s", formatted_text.data);
    } @PrintLike

    TextDisabled_CFormat :: (fmt: *u8, __args: ..Any) -> void #foreign imgui "?TextDisabled@ImGui@@YAXPEBDZZ";
    TextDisabled :: (fmt: string, __args: ..Any) {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, fmt, ..__args);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        TextDisabled_CFormat("%s", formatted_text.data);
    } @PrintLike

    TextWrapped_CFormat :: (fmt: *u8, __args: ..Any) -> void #foreign imgui "?TextWrapped@ImGui@@YAXPEBDZZ";
    TextWrapped :: (fmt: string, __args: ..Any) {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, fmt, ..__args);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        TextWrapped_CFormat("%s", formatted_text.data);
    } @PrintLike

    LabelText_CFormat :: (label: *u8, fmt: *u8, __args: ..Any) -> void #foreign imgui "?LabelText@ImGui@@YAXPEBD0ZZ";
    LabelText :: (label: *u8, fmt: string, __args: ..Any) {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, fmt, ..__args);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        LabelText_CFormat(label, "%s", formatted_text.data);
    } @PrintLike

    BulletText_CFormat :: (fmt: *u8, __args: ..Any) -> void #foreign imgui "?BulletText@ImGui@@YAXPEBDZZ";
    BulletText :: (fmt: string, __args: ..Any) {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, fmt, ..__args);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        BulletText_CFormat("%s", formatted_text.data);
    } @PrintLike

    SeparatorText :: (label: *u8) -> void #foreign imgui "?SeparatorText@ImGui@@YAXPEBD@Z";

    // Widgets: Main
    // - Most widgets return true when the value has been changed or when pressed/selected
    // - You may also use one of the many IsItemXXX functions (e.g. IsItemActive, IsItemHovered, etc.) to query widget state.
    Button :: (label: *u8, size: *ImVec2) -> bool #foreign imgui "?Button@ImGui@@YA_NPEBDAEBUImVec2@@@Z";
    Button :: (label: *u8, size: ImVec2 = ImVec2.{0, 0}) -> bool #no_context {
        return Button(label, *size);
    }
    SmallButton :: (label: *u8) -> bool #foreign imgui "?SmallButton@ImGui@@YA_NPEBD@Z";
    InvisibleButton :: (str_id: *u8, size: *ImVec2, flags: ImGuiButtonFlags = 0) -> bool #foreign imgui "?InvisibleButton@ImGui@@YA_NPEBDAEBUImVec2@@H@Z";
    InvisibleButton :: (str_id: *u8, size: ImVec2, flags: ImGuiButtonFlags = 0) -> bool #no_context {
        return InvisibleButton(str_id, *size, flags);
    }
    ArrowButton :: (str_id: *u8, dir: ImGuiDir) -> bool #foreign imgui "?ArrowButton@ImGui@@YA_NPEBDW4ImGuiDir@@@Z";
    Checkbox :: (label: *u8, v: *bool) -> bool #foreign imgui "?Checkbox@ImGui@@YA_NPEBDPEA_N@Z";
    CheckboxFlags :: (label: *u8, flags: *s32, flags_value: s32) -> bool #foreign imgui "?CheckboxFlags@ImGui@@YA_NPEBDPEAHH@Z";
    CheckboxFlags :: (label: *u8, flags: *u32, flags_value: u32) -> bool #foreign imgui "?CheckboxFlags@ImGui@@YA_NPEBDPEAII@Z";
    RadioButton :: (label: *u8, active: bool) -> bool #foreign imgui "?RadioButton@ImGui@@YA_NPEBD_N@Z";
    RadioButton :: (label: *u8, v: *s32, v_button: s32) -> bool #foreign imgui "?RadioButton@ImGui@@YA_NPEBDPEAHH@Z";
    ProgressBar :: (fraction: float, size_arg: *ImVec2, overlay: *u8 = null) -> void #foreign imgui "?ProgressBar@ImGui@@YAXMAEBUImVec2@@PEBD@Z";
    ProgressBar :: (fraction: float, size_arg: ImVec2 = ImVec2.{-FLOAT32_MIN, 0}, overlay: *u8 = null) #no_context {
        ProgressBar(fraction, *size_arg, overlay);
    }
    Bullet :: () -> void #foreign imgui "?Bullet@ImGui@@YAXXZ";
    TextLink :: (label: *u8) -> bool #foreign imgui "?TextLink@ImGui@@YA_NPEBD@Z";
    TextLinkOpenURL :: (label: *u8, url: *u8 = null) -> bool #foreign imgui "?TextLinkOpenURL@ImGui@@YA_NPEBD0@Z";

    // Widgets: Images
    // - Read about ImTextureID/ImTextureRef  here: https://github.com/ocornut/imgui/wiki/Image-Loading-and-Displaying-Examples
    // - 'uv0' and 'uv1' are texture coordinates. Read about them from the same link above.
    // - Image() pads adds style.ImageBorderSize on each side, ImageButton() adds style.FramePadding on each side.
    // - ImageButton() draws a background based on regular Button() color + optionally an inner background if specified.
    // - An obsolete version of Image(), before 1.91.9 (March 2025), had a 'tint_col' parameter which is now supported by the ImageWithBg() function.
    Image :: (tex_ref: ImTextureRef, image_size: *ImVec2, uv0: *ImVec2, uv1: *ImVec2) -> void #foreign imgui "?Image@ImGui@@YAXUImTextureRef@@AEBUImVec2@@11@Z";
    Image :: (tex_ref: ImTextureRef, image_size: ImVec2, uv0: ImVec2 = ImVec2.{0, 0}, uv1: ImVec2 = ImVec2.{1, 1}) #no_context {
        Image(tex_ref, *image_size, *uv0, *uv1);
    }
    ImageWithBg :: (tex_ref: ImTextureRef, image_size: *ImVec2, uv0: *ImVec2, uv1: *ImVec2, bg_col: *ImVec4, tint_col: *ImVec4) -> void #foreign imgui "?ImageWithBg@ImGui@@YAXUImTextureRef@@AEBUImVec2@@11AEBUImVec4@@2@Z";
    ImageWithBg :: (tex_ref: ImTextureRef, image_size: ImVec2, uv0: ImVec2 = ImVec2.{0, 0}, uv1: ImVec2 = ImVec2.{1, 1}, bg_col: ImVec4 = ImVec4.{0, 0, 0, 0}, tint_col: ImVec4 = ImVec4.{1, 1, 1, 1}) #no_context {
        ImageWithBg(tex_ref, *image_size, *uv0, *uv1, *bg_col, *tint_col);
    }
    ImageButton :: (str_id: *u8, tex_ref: ImTextureRef, image_size: *ImVec2, uv0: *ImVec2, uv1: *ImVec2, bg_col: *ImVec4, tint_col: *ImVec4) -> bool #foreign imgui "?ImageButton@ImGui@@YA_NPEBDUImTextureRef@@AEBUImVec2@@22AEBUImVec4@@3@Z";
    ImageButton :: (str_id: *u8, tex_ref: ImTextureRef, image_size: ImVec2, uv0: ImVec2 = ImVec2.{0, 0}, uv1: ImVec2 = ImVec2.{1, 1}, bg_col: ImVec4 = ImVec4.{0, 0, 0, 0}, tint_col: ImVec4 = ImVec4.{1, 1, 1, 1}) -> bool #no_context {
        return ImageButton(str_id, tex_ref, *image_size, *uv0, *uv1, *bg_col, *tint_col);
    }

    // Widgets: Combo Box (Dropdown)
    // - The BeginCombo()/EndCombo() api allows you to manage your contents and selection state however you want it, by creating e.g. Selectable() items.
    // - The old Combo() api are helpers over BeginCombo()/EndCombo() which are kept available for convenience purpose. This is analogous to how ListBox are created.
    BeginCombo :: (label: *u8, preview_value: *u8, flags: ImGuiComboFlags = 0) -> bool #foreign imgui "?BeginCombo@ImGui@@YA_NPEBD0H@Z";
    EndCombo :: () -> void #foreign imgui "?EndCombo@ImGui@@YAXXZ";
    Combo :: (label: *u8, current_item: *s32, items: **u8, items_count: s32, popup_max_height_in_items: s32 = -1) -> bool #foreign imgui "?Combo@ImGui@@YA_NPEBDPEAHQEBQEBDHH@Z";
    Combo :: (label: *u8, current_item: *s32, items_separated_by_zeros: *u8, popup_max_height_in_items: s32 = -1) -> bool #foreign imgui "?Combo@ImGui@@YA_NPEBDPEAH0H@Z";
    Combo :: (label: *u8, current_item: *s32, getter: #type (user_data: *void, idx: s32) -> *u8 #c_call, user_data: *void, items_count: s32, popup_max_height_in_items: s32 = -1) -> bool #foreign imgui "?Combo@ImGui@@YA_NPEBDPEAHP6APEBDPEAXH@Z2HH@Z";

    // Widgets: Drag Sliders
    // - CTRL+Click on any drag box to turn them into an input box. Manually input values aren't clamped by default and can go off-bounds. Use ImGuiSliderFlags_AlwaysClamp to always clamp.
    // - For all the Float2/Float3/Float4/Int2/Int3/Int4 versions of every function, note that a 'float v[X]' function argument is the same as 'float* v',
    //   the array syntax is just a way to document the number of elements that are expected to be accessible. You can pass address of your first element out of a contiguous set, e.g. &myvector.x
    // - Adjust format string to decorate the value with a prefix, a suffix, or adapt the editing and display precision e.g. "%.3f" -> 1.234; "%5.2f secs" -> 01.23 secs; "Biscuit: %.0f" -> Biscuit: 1; etc.
    // - Format string may also be set to NULL or use the default format ("%f" or "%d").
    // - Speed are per-pixel of mouse movement (v_speed=0.2f: mouse needs to move by 5 pixels to increase value by 1). For keyboard/gamepad navigation, minimum speed is Max(v_speed, minimum_step_at_given_precision).
    // - Use v_min < v_max to clamp edits to given limits. Note that CTRL+Click manual input can override those limits if ImGuiSliderFlags_AlwaysClamp is not used.
    // - Use v_max = FLT_MAX / INT_MAX etc to avoid clamping to a maximum, same with v_min = -FLT_MAX / INT_MIN to avoid clamping to a minimum.
    // - We use the same sets of flags for DragXXX() and SliderXXX() functions as the features are the same and it makes it easier to swap them.
    // - Legacy: Pre-1.78 there are DragXXX() function signatures that take a final `float power=1.0f' argument instead of the `ImGuiSliderFlags flags=0' argument.
    //   If you get a warning converting a float to ImGuiSliderFlags, read https://github.com/ocornut/imgui/issues/3361
    DragFloat :: (label: *u8, v: *float, v_speed: float = 1.0, v_min: float = 0.0, v_max: float = 0.0, format: *u8 = "%.3f", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?DragFloat@ImGui@@YA_NPEBDPEAMMMM0H@Z";
    DragFloat2 :: (label: *u8, v: *[2] float, v_speed: float = 1.0, v_min: float = 0.0, v_max: float = 0.0, format: *u8 = "%.3f", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?DragFloat2@ImGui@@YA_NPEBDQEAMMMM0H@Z";
    DragFloat3 :: (label: *u8, v: *[3] float, v_speed: float = 1.0, v_min: float = 0.0, v_max: float = 0.0, format: *u8 = "%.3f", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?DragFloat3@ImGui@@YA_NPEBDQEAMMMM0H@Z";
    DragFloat4 :: (label: *u8, v: *[4] float, v_speed: float = 1.0, v_min: float = 0.0, v_max: float = 0.0, format: *u8 = "%.3f", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?DragFloat4@ImGui@@YA_NPEBDQEAMMMM0H@Z";
    DragFloatRange2 :: (label: *u8, v_current_min: *float, v_current_max: *float, v_speed: float = 1.0, v_min: float = 0.0, v_max: float = 0.0, format: *u8 = "%.3f", format_max: *u8 = null, flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?DragFloatRange2@ImGui@@YA_NPEBDPEAM1MMM00H@Z";
    DragInt :: (label: *u8, v: *s32, v_speed: float = 1.0, v_min: s32 = 0, v_max: s32 = 0, format: *u8 = "%d", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?DragInt@ImGui@@YA_NPEBDPEAHMHH0H@Z";
    DragInt2 :: (label: *u8, v: *[2] s32, v_speed: float = 1.0, v_min: s32 = 0, v_max: s32 = 0, format: *u8 = "%d", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?DragInt2@ImGui@@YA_NPEBDQEAHMHH0H@Z";
    DragInt3 :: (label: *u8, v: *[3] s32, v_speed: float = 1.0, v_min: s32 = 0, v_max: s32 = 0, format: *u8 = "%d", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?DragInt3@ImGui@@YA_NPEBDQEAHMHH0H@Z";
    DragInt4 :: (label: *u8, v: *[4] s32, v_speed: float = 1.0, v_min: s32 = 0, v_max: s32 = 0, format: *u8 = "%d", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?DragInt4@ImGui@@YA_NPEBDQEAHMHH0H@Z";
    DragIntRange2 :: (label: *u8, v_current_min: *s32, v_current_max: *s32, v_speed: float = 1.0, v_min: s32 = 0, v_max: s32 = 0, format: *u8 = "%d", format_max: *u8 = null, flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?DragIntRange2@ImGui@@YA_NPEBDPEAH1MHH00H@Z";
    DragScalar :: (label: *u8, data_type: ImGuiDataType, p_data: *void, v_speed: float = 1.0, p_min: *void = null, p_max: *void = null, format: *u8 = null, flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?DragScalar@ImGui@@YA_NPEBDHPEAXMPEBX20H@Z";
    DragScalarN :: (label: *u8, data_type: ImGuiDataType, p_data: *void, components: s32, v_speed: float = 1.0, p_min: *void = null, p_max: *void = null, format: *u8 = null, flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?DragScalarN@ImGui@@YA_NPEBDHPEAXHMPEBX20H@Z";

    // Widgets: Regular Sliders
    // - CTRL+Click on any slider to turn them into an input box. Manually input values aren't clamped by default and can go off-bounds. Use ImGuiSliderFlags_AlwaysClamp to always clamp.
    // - Adjust format string to decorate the value with a prefix, a suffix, or adapt the editing and display precision e.g. "%.3f" -> 1.234; "%5.2f secs" -> 01.23 secs; "Biscuit: %.0f" -> Biscuit: 1; etc.
    // - Format string may also be set to NULL or use the default format ("%f" or "%d").
    // - Legacy: Pre-1.78 there are SliderXXX() function signatures that take a final `float power=1.0f' argument instead of the `ImGuiSliderFlags flags=0' argument.
    //   If you get a warning converting a float to ImGuiSliderFlags, read https://github.com/ocornut/imgui/issues/3361
    SliderFloat :: (label: *u8, v: *float, v_min: float, v_max: float, format: *u8 = "%.3f", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?SliderFloat@ImGui@@YA_NPEBDPEAMMM0H@Z";
    SliderFloat2 :: (label: *u8, v: *[2] float, v_min: float, v_max: float, format: *u8 = "%.3f", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?SliderFloat2@ImGui@@YA_NPEBDQEAMMM0H@Z";
    SliderFloat3 :: (label: *u8, v: *[3] float, v_min: float, v_max: float, format: *u8 = "%.3f", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?SliderFloat3@ImGui@@YA_NPEBDQEAMMM0H@Z";
    SliderFloat4 :: (label: *u8, v: *[4] float, v_min: float, v_max: float, format: *u8 = "%.3f", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?SliderFloat4@ImGui@@YA_NPEBDQEAMMM0H@Z";
    SliderAngle :: (label: *u8, v_rad: *float, v_degrees_min: float = -360.0, v_degrees_max: float = +360.0, format: *u8 = "%.0f deg", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?SliderAngle@ImGui@@YA_NPEBDPEAMMM0H@Z";
    SliderInt :: (label: *u8, v: *s32, v_min: s32, v_max: s32, format: *u8 = "%d", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?SliderInt@ImGui@@YA_NPEBDPEAHHH0H@Z";
    SliderInt2 :: (label: *u8, v: *[2] s32, v_min: s32, v_max: s32, format: *u8 = "%d", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?SliderInt2@ImGui@@YA_NPEBDQEAHHH0H@Z";
    SliderInt3 :: (label: *u8, v: *[3] s32, v_min: s32, v_max: s32, format: *u8 = "%d", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?SliderInt3@ImGui@@YA_NPEBDQEAHHH0H@Z";
    SliderInt4 :: (label: *u8, v: *[4] s32, v_min: s32, v_max: s32, format: *u8 = "%d", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?SliderInt4@ImGui@@YA_NPEBDQEAHHH0H@Z";
    SliderScalar :: (label: *u8, data_type: ImGuiDataType, p_data: *void, p_min: *void, p_max: *void, format: *u8 = null, flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?SliderScalar@ImGui@@YA_NPEBDHPEAXPEBX20H@Z";
    SliderScalarN :: (label: *u8, data_type: ImGuiDataType, p_data: *void, components: s32, p_min: *void, p_max: *void, format: *u8 = null, flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?SliderScalarN@ImGui@@YA_NPEBDHPEAXHPEBX20H@Z";
    VSliderFloat :: (label: *u8, size: *ImVec2, v: *float, v_min: float, v_max: float, format: *u8 = "%.3f", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?VSliderFloat@ImGui@@YA_NPEBDAEBUImVec2@@PEAMMM0H@Z";
    VSliderFloat :: (label: *u8, size: ImVec2, v: *float, v_min: float, v_max: float, format: *u8 = "%.3f", flags: ImGuiSliderFlags = 0) -> bool #no_context {
        return VSliderFloat(label, *size, v, v_min, v_max, format, flags);
    }
    VSliderInt :: (label: *u8, size: *ImVec2, v: *s32, v_min: s32, v_max: s32, format: *u8 = "%d", flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?VSliderInt@ImGui@@YA_NPEBDAEBUImVec2@@PEAHHH0H@Z";
    VSliderInt :: (label: *u8, size: ImVec2, v: *s32, v_min: s32, v_max: s32, format: *u8 = "%d", flags: ImGuiSliderFlags = 0) -> bool #no_context {
        return VSliderInt(label, *size, v, v_min, v_max, format, flags);
    }
    VSliderScalar :: (label: *u8, size: *ImVec2, data_type: ImGuiDataType, p_data: *void, p_min: *void, p_max: *void, format: *u8 = null, flags: ImGuiSliderFlags = 0) -> bool #foreign imgui "?VSliderScalar@ImGui@@YA_NPEBDAEBUImVec2@@HPEAXPEBX30H@Z";
    VSliderScalar :: (label: *u8, size: ImVec2, data_type: ImGuiDataType, p_data: *void, p_min: *void, p_max: *void, format: *u8 = null, flags: ImGuiSliderFlags = 0) -> bool #no_context {
        return VSliderScalar(label, *size, data_type, p_data, p_min, p_max, format, flags);
    }

    // Widgets: Input with Keyboard
    // - If you want to use InputText() with std::string or any custom dynamic string type, see misc/cpp/imgui_stdlib.h and comments in imgui_demo.cpp.
    // - Most of the ImGuiInputTextFlags flags are only useful for InputText() and not for InputFloatX, InputIntX, InputDouble etc.
    InputText :: (label: *u8, buf: *u8, buf_size: u64, flags: ImGuiInputTextFlags = 0, callback: ImGuiInputTextCallback = null, user_data: *void = null) -> bool #foreign imgui "?InputText@ImGui@@YA_NPEBDPEAD_KHP6AHPEAUImGuiInputTextCallbackData@@@ZPEAX@Z";
    InputTextMultiline :: (label: *u8, buf: *u8, buf_size: u64, size: *ImVec2, flags: ImGuiInputTextFlags = 0, callback: ImGuiInputTextCallback = null, user_data: *void = null) -> bool #foreign imgui "?InputTextMultiline@ImGui@@YA_NPEBDPEAD_KAEBUImVec2@@HP6AHPEAUImGuiInputTextCallbackData@@@ZPEAX@Z";
    InputTextMultiline :: (label: *u8, buf: *u8, buf_size: u64, size: ImVec2 = ImVec2.{0, 0}, flags: ImGuiInputTextFlags = 0, callback: ImGuiInputTextCallback = null, user_data: *void = null) -> bool #no_context {
        return InputTextMultiline(label, buf, buf_size, *size, flags, callback, user_data);
    }
    InputTextWithHint :: (label: *u8, hint: *u8, buf: *u8, buf_size: u64, flags: ImGuiInputTextFlags = 0, callback: ImGuiInputTextCallback = null, user_data: *void = null) -> bool #foreign imgui "?InputTextWithHint@ImGui@@YA_NPEBD0PEAD_KHP6AHPEAUImGuiInputTextCallbackData@@@ZPEAX@Z";
    InputFloat :: (label: *u8, v: *float, step: float = 0.0, step_fast: float = 0.0, format: *u8 = "%.3f", flags: ImGuiInputTextFlags = 0) -> bool #foreign imgui "?InputFloat@ImGui@@YA_NPEBDPEAMMM0H@Z";
    InputFloat2 :: (label: *u8, v: *[2] float, format: *u8 = "%.3f", flags: ImGuiInputTextFlags = 0) -> bool #foreign imgui "?InputFloat2@ImGui@@YA_NPEBDQEAM0H@Z";
    InputFloat3 :: (label: *u8, v: *[3] float, format: *u8 = "%.3f", flags: ImGuiInputTextFlags = 0) -> bool #foreign imgui "?InputFloat3@ImGui@@YA_NPEBDQEAM0H@Z";
    InputFloat4 :: (label: *u8, v: *[4] float, format: *u8 = "%.3f", flags: ImGuiInputTextFlags = 0) -> bool #foreign imgui "?InputFloat4@ImGui@@YA_NPEBDQEAM0H@Z";
    InputInt :: (label: *u8, v: *s32, step: s32 = 1, step_fast: s32 = 100, flags: ImGuiInputTextFlags = 0) -> bool #foreign imgui "?InputInt@ImGui@@YA_NPEBDPEAHHHH@Z";
    InputInt2 :: (label: *u8, v: *[2] s32, flags: ImGuiInputTextFlags = 0) -> bool #foreign imgui "?InputInt2@ImGui@@YA_NPEBDQEAHH@Z";
    InputInt3 :: (label: *u8, v: *[3] s32, flags: ImGuiInputTextFlags = 0) -> bool #foreign imgui "?InputInt3@ImGui@@YA_NPEBDQEAHH@Z";
    InputInt4 :: (label: *u8, v: *[4] s32, flags: ImGuiInputTextFlags = 0) -> bool #foreign imgui "?InputInt4@ImGui@@YA_NPEBDQEAHH@Z";
    InputDouble :: (label: *u8, v: *float64, step: float64 = 0.0, step_fast: float64 = 0.0, format: *u8 = "%.6f", flags: ImGuiInputTextFlags = 0) -> bool #foreign imgui "?InputDouble@ImGui@@YA_NPEBDPEANNN0H@Z";
    InputScalar :: (label: *u8, data_type: ImGuiDataType, p_data: *void, p_step: *void = null, p_step_fast: *void = null, format: *u8 = null, flags: ImGuiInputTextFlags = 0) -> bool #foreign imgui "?InputScalar@ImGui@@YA_NPEBDHPEAXPEBX20H@Z";
    InputScalarN :: (label: *u8, data_type: ImGuiDataType, p_data: *void, components: s32, p_step: *void = null, p_step_fast: *void = null, format: *u8 = null, flags: ImGuiInputTextFlags = 0) -> bool #foreign imgui "?InputScalarN@ImGui@@YA_NPEBDHPEAXHPEBX20H@Z";

    // Widgets: Color Editor/Picker (tip: the ColorEdit* functions have a little color square that can be left-clicked to open a picker, and right-clicked to open an option menu.)
    // - Note that in C++ a 'float v[X]' function argument is the _same_ as 'float* v', the array syntax is just a way to document the number of elements that are expected to be accessible.
    // - You can pass the address of a first float element out of a contiguous structure, e.g. &myvector.x
    ColorEdit3 :: (label: *u8, col: *[3] float, flags: ImGuiColorEditFlags = 0) -> bool #foreign imgui "?ColorEdit3@ImGui@@YA_NPEBDQEAMH@Z";
    ColorEdit4 :: (label: *u8, col: *[4] float, flags: ImGuiColorEditFlags = 0) -> bool #foreign imgui "?ColorEdit4@ImGui@@YA_NPEBDQEAMH@Z";
    ColorPicker3 :: (label: *u8, col: *[3] float, flags: ImGuiColorEditFlags = 0) -> bool #foreign imgui "?ColorPicker3@ImGui@@YA_NPEBDQEAMH@Z";
    ColorPicker4 :: (label: *u8, col: *[4] float, flags: ImGuiColorEditFlags = 0, ref_col: *float = null) -> bool #foreign imgui "?ColorPicker4@ImGui@@YA_NPEBDQEAMHPEBM@Z";
    ColorButton :: (desc_id: *u8, col: *ImVec4, flags: ImGuiColorEditFlags = 0, size: *ImVec2) -> bool #foreign imgui "?ColorButton@ImGui@@YA_NPEBDAEBUImVec4@@HAEBUImVec2@@@Z";
    ColorButton :: (desc_id: *u8, col: ImVec4, flags: ImGuiColorEditFlags = 0, size: ImVec2 = ImVec2.{0, 0}) -> bool #no_context {
        return ColorButton(desc_id, *col, flags, *size);
    }
    SetColorEditOptions :: (flags: ImGuiColorEditFlags) -> void #foreign imgui "?SetColorEditOptions@ImGui@@YAXH@Z";

    // Widgets: Trees
    // - TreeNode functions return true when the node is open, in which case you need to also call TreePop() when you are finished displaying the tree node contents.
    TreeNode :: (label: *u8) -> bool #foreign imgui "?TreeNode@ImGui@@YA_NPEBD@Z";
    TreeNode_CFormat :: (str_id: *u8, fmt: *u8, __args: ..Any) -> bool #foreign imgui "?TreeNode@ImGui@@YA_NPEBD0ZZ";
    TreeNode :: (str_id: *u8, fmt: string, __args: ..Any) -> bool {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, fmt, ..__args);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        return TreeNode_CFormat(str_id, "%s", formatted_text.data);
    } @PrintLike
    TreeNode_CFormat :: (ptr_id: *void, fmt: *u8, __args: ..Any) -> bool #foreign imgui "?TreeNode@ImGui@@YA_NPEBXPEBDZZ";
    TreeNode :: (ptr_id: *void, fmt: string, __args: ..Any) -> bool {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, fmt, ..__args);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        return TreeNode_CFormat(ptr_id, "%s", formatted_text.data);
    } @PrintLike

    TreeNodeEx :: (label: *u8, flags: ImGuiTreeNodeFlags = 0) -> bool #foreign imgui "?TreeNodeEx@ImGui@@YA_NPEBDH@Z";
    TreeNodeEx_CFormat :: (str_id: *u8, flags: ImGuiTreeNodeFlags, fmt: *u8, __args: ..Any) -> bool #foreign imgui "?TreeNodeEx@ImGui@@YA_NPEBDH0ZZ";
    TreeNodeEx :: (str_id: *u8, flags: ImGuiTreeNodeFlags, fmt: string, __args: ..Any) -> bool {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, fmt, ..__args);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        return TreeNodeEx_CFormat(str_id, flags, "%s", formatted_text.data);
    } @PrintLike
    TreeNodeEx_CFormat :: (ptr_id: *void, flags: ImGuiTreeNodeFlags, fmt: *u8, __args: ..Any) -> bool #foreign imgui "?TreeNodeEx@ImGui@@YA_NPEBXHPEBDZZ";
    TreeNodeEx :: (ptr_id: *void, flags: ImGuiTreeNodeFlags, fmt: string, __args: ..Any) -> bool {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, fmt, ..__args);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        return TreeNodeEx_CFormat(ptr_id, flags, "%s", formatted_text.data);
    } @PrintLike

    TreePush :: (str_id: *u8) -> void #foreign imgui "?TreePush@ImGui@@YAXPEBD@Z";
    TreePush :: (ptr_id: *void) -> void #foreign imgui "?TreePush@ImGui@@YAXPEBX@Z";
    TreePop :: () -> void #foreign imgui "?TreePop@ImGui@@YAXXZ";
    GetTreeNodeToLabelSpacing :: () -> float #foreign imgui "?GetTreeNodeToLabelSpacing@ImGui@@YAMXZ";
    CollapsingHeader :: (label: *u8, flags: ImGuiTreeNodeFlags = 0) -> bool #foreign imgui "?CollapsingHeader@ImGui@@YA_NPEBDH@Z";
    CollapsingHeader :: (label: *u8, p_visible: *bool, flags: ImGuiTreeNodeFlags = 0) -> bool #foreign imgui "?CollapsingHeader@ImGui@@YA_NPEBDPEA_NH@Z";
    SetNextItemOpen :: (is_open: bool, cond: ImGuiCond = 0) -> void #foreign imgui "?SetNextItemOpen@ImGui@@YAX_NH@Z";
    SetNextItemStorageID :: (storage_id: ImGuiID) -> void #foreign imgui "?SetNextItemStorageID@ImGui@@YAXI@Z";

    // Widgets: Selectables
    // - A selectable highlights when hovered, and can display another color when selected.
    // - Neighbors selectable extend their highlight bounds in order to leave no gap between them. This is so a series of selected Selectable appear contiguous.
    Selectable :: (label: *u8, selected := false, flags: ImGuiSelectableFlags = 0, size: *ImVec2) -> bool #foreign imgui "?Selectable@ImGui@@YA_NPEBD_NHAEBUImVec2@@@Z";
    Selectable :: (label: *u8, selected := false, flags: ImGuiSelectableFlags = 0, size: ImVec2 = ImVec2.{0, 0}) -> bool #no_context {
        return Selectable(label, selected, flags, *size);
    }
    Selectable :: (label: *u8, p_selected: *bool, flags: ImGuiSelectableFlags = 0, size: *ImVec2) -> bool #foreign imgui "?Selectable@ImGui@@YA_NPEBDPEA_NHAEBUImVec2@@@Z";
    Selectable :: (label: *u8, p_selected: *bool, flags: ImGuiSelectableFlags = 0, size: ImVec2 = ImVec2.{0, 0}) -> bool #no_context {
        return Selectable(label, p_selected, flags, *size);
    }

    // Multi-selection system for Selectable(), Checkbox(), TreeNode() functions [BETA]
    // - This enables standard multi-selection/range-selection idioms (CTRL+Mouse/Keyboard, SHIFT+Mouse/Keyboard, etc.) in a way that also allow a clipper to be used.
    // - ImGuiSelectionUserData is often used to store your item index within the current view (but may store something else).
    // - Read comments near ImGuiMultiSelectIO for instructions/details and see 'Demo->Widgets->Selection State & Multi-Select' for demo.
    // - TreeNode() is technically supported but... using this correctly is more complicated. You need some sort of linear/random access to your tree,
    //   which is suited to advanced trees setups already implementing filters and clipper. We will work simplifying the current demo.
    // - 'selection_size' and 'items_count' parameters are optional and used by a few features. If they are costly for you to compute, you may avoid them.
    BeginMultiSelect :: (flags: ImGuiMultiSelectFlags, selection_size: s32 = -1, items_count: s32 = -1) -> *ImGuiMultiSelectIO #foreign imgui "?BeginMultiSelect@ImGui@@YAPEAUImGuiMultiSelectIO@@HHH@Z";
    EndMultiSelect :: () -> *ImGuiMultiSelectIO #foreign imgui "?EndMultiSelect@ImGui@@YAPEAUImGuiMultiSelectIO@@XZ";
    SetNextItemSelectionUserData :: (selection_user_data: ImGuiSelectionUserData) -> void #foreign imgui "?SetNextItemSelectionUserData@ImGui@@YAX_J@Z";
    IsItemToggledSelection :: () -> bool #foreign imgui "?IsItemToggledSelection@ImGui@@YA_NXZ";

    // Widgets: List Boxes
    // - This is essentially a thin wrapper to using BeginChild/EndChild with the ImGuiChildFlags_FrameStyle flag for stylistic changes + displaying a label.
    // - If you don't need a label you can probably simply use BeginChild() with the ImGuiChildFlags_FrameStyle flag for the same result.
    // - You can submit contents and manage your selection state however you want it, by creating e.g. Selectable() or any other items.
    // - The simplified/old ListBox() api are helpers over BeginListBox()/EndListBox() which are kept available for convenience purpose. This is analogous to how Combos are created.
    // - Choose frame width:   size.x > 0.0f: custom  /  size.x < 0.0f or -FLT_MIN: right-align   /  size.x = 0.0f (default): use current ItemWidth
    // - Choose frame height:  size.y > 0.0f: custom  /  size.y < 0.0f or -FLT_MIN: bottom-align  /  size.y = 0.0f (default): arbitrary default height which can fit ~7 items
    BeginListBox :: (label: *u8, size: *ImVec2) -> bool #foreign imgui "?BeginListBox@ImGui@@YA_NPEBDAEBUImVec2@@@Z";
    BeginListBox :: (label: *u8, size: ImVec2 = ImVec2.{0, 0}) -> bool #no_context {
        return BeginListBox(label, *size);
    }
    EndListBox :: () -> void #foreign imgui "?EndListBox@ImGui@@YAXXZ";
    ListBox :: (label: *u8, current_item: *s32, items: **u8, items_count: s32, height_in_items: s32 = -1) -> bool #foreign imgui "?ListBox@ImGui@@YA_NPEBDPEAHQEBQEBDHH@Z";
    ListBox :: (label: *u8, current_item: *s32, getter: #type (user_data: *void, idx: s32) -> *u8 #c_call, user_data: *void, items_count: s32, height_in_items: s32 = -1) -> bool #foreign imgui "?ListBox@ImGui@@YA_NPEBDPEAHP6APEBDPEAXH@Z2HH@Z";

    // Widgets: Data Plotting
    // - Consider using ImPlot (https://github.com/epezent/implot) which is much better!
    PlotLines :: (label: *u8, values: *float, values_count: s32, values_offset: s32 = 0, overlay_text: *u8 = null, scale_min: float = FLOAT32_MAX, scale_max: float = FLOAT32_MAX, graph_size: ImVec2 = ImVec2.{0, 0}, stride: s32 = 4) -> void #foreign imgui "?PlotLines@ImGui@@YAXPEBDPEBMHH0MMUImVec2@@H@Z";
    PlotLines :: (label: *u8, values_getter: #type (data: *void, idx: s32) -> float #c_call, data: *void, values_count: s32, values_offset: s32 = 0, overlay_text: *u8 = null, scale_min: float = FLOAT32_MAX, scale_max: float = FLOAT32_MAX, graph_size: ImVec2 = ImVec2.{0, 0}) -> void #foreign imgui "?PlotLines@ImGui@@YAXPEBDP6AMPEAXH@Z1HH0MMUImVec2@@@Z";
    PlotHistogram :: (label: *u8, values: *float, values_count: s32, values_offset: s32 = 0, overlay_text: *u8 = null, scale_min: float = FLOAT32_MAX, scale_max: float = FLOAT32_MAX, graph_size: ImVec2 = ImVec2.{0, 0}, stride: s32 = 4) -> void #foreign imgui "?PlotHistogram@ImGui@@YAXPEBDPEBMHH0MMUImVec2@@H@Z";
    PlotHistogram :: (label: *u8, values_getter: #type (data: *void, idx: s32) -> float #c_call, data: *void, values_count: s32, values_offset: s32 = 0, overlay_text: *u8 = null, scale_min: float = FLOAT32_MAX, scale_max: float = FLOAT32_MAX, graph_size: ImVec2 = ImVec2.{0, 0}) -> void #foreign imgui "?PlotHistogram@ImGui@@YAXPEBDP6AMPEAXH@Z1HH0MMUImVec2@@@Z";

    // Widgets: Value() Helpers.
    // - Those are merely shortcut to calling Text() with a format string. Output single value in "name: value" format (tip: freely declare more in your code to handle your types. you can add functions to the ImGui namespace)
    Value :: (prefix: *u8, b: bool) -> void #foreign imgui "?Value@ImGui@@YAXPEBD_N@Z";
    Value :: (prefix: *u8, v: s32) -> void #foreign imgui "?Value@ImGui@@YAXPEBDH@Z";
    Value :: (prefix: *u8, v: u32) -> void #foreign imgui "?Value@ImGui@@YAXPEBDI@Z";
    Value :: (prefix: *u8, v: float, float_format: *u8 = null) -> void #foreign imgui "?Value@ImGui@@YAXPEBDM0@Z";

    // Widgets: Menus
    // - Use BeginMenuBar() on a window ImGuiWindowFlags_MenuBar to append to its menu bar.
    // - Use BeginMainMenuBar() to create a menu bar at the top of the screen and append to it.
    // - Use BeginMenu() to create a menu. You can call BeginMenu() multiple time with the same identifier to append more items to it.
    // - Not that MenuItem() keyboardshortcuts are displayed as a convenience but _not processed_ by Dear ImGui at the moment.
    BeginMenuBar :: () -> bool #foreign imgui "?BeginMenuBar@ImGui@@YA_NXZ";
    EndMenuBar :: () -> void #foreign imgui "?EndMenuBar@ImGui@@YAXXZ";
    BeginMainMenuBar :: () -> bool #foreign imgui "?BeginMainMenuBar@ImGui@@YA_NXZ";
    EndMainMenuBar :: () -> void #foreign imgui "?EndMainMenuBar@ImGui@@YAXXZ";
    BeginMenu :: (label: *u8, enabled := true) -> bool #foreign imgui "?BeginMenu@ImGui@@YA_NPEBD_N@Z";
    EndMenu :: () -> void #foreign imgui "?EndMenu@ImGui@@YAXXZ";
    MenuItem :: (label: *u8, shortcut: *u8 = null, selected := false, enabled := true) -> bool #foreign imgui "?MenuItem@ImGui@@YA_NPEBD0_N1@Z";
    MenuItem :: (label: *u8, shortcut: *u8, p_selected: *bool, enabled := true) -> bool #foreign imgui "?MenuItem@ImGui@@YA_NPEBD0PEA_N_N@Z";

    // Tooltips
    // - Tooltips are windows following the mouse. They do not take focus away.
    // - A tooltip window can contain items of any types.
    // - SetTooltip() is more or less a shortcut for the 'if (BeginTooltip()) { Text(...); EndTooltip(); }' idiom (with a subtlety that it discard any previously submitted tooltip)
    BeginTooltip :: () -> bool #foreign imgui "?BeginTooltip@ImGui@@YA_NXZ";
    EndTooltip :: () -> void #foreign imgui "?EndTooltip@ImGui@@YAXXZ";
    SetTooltip_CFormat :: (fmt: *u8, __args: ..Any) -> void #foreign imgui "?SetTooltip@ImGui@@YAXPEBDZZ";
    SetTooltip :: (fmt: string, __args: ..Any) {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, fmt, ..__args);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        SetTooltip_CFormat("%s", formatted_text.data);
    } @PrintLike

    // Tooltips: helpers for showing a tooltip when hovering an item
    // - BeginItemTooltip() is a shortcut for the 'if (IsItemHovered(ImGuiHoveredFlags_ForTooltip) && BeginTooltip())' idiom.
    // - SetItemTooltip() is a shortcut for the 'if (IsItemHovered(ImGuiHoveredFlags_ForTooltip)) { SetTooltip(...); }' idiom.
    // - Where 'ImGuiHoveredFlags_ForTooltip' itself is a shortcut to use 'style.HoverFlagsForTooltipMouse' or 'style.HoverFlagsForTooltipNav' depending on active input type. For mouse it defaults to 'ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_DelayShort'.
    BeginItemTooltip :: () -> bool #foreign imgui "?BeginItemTooltip@ImGui@@YA_NXZ";
    SetItemTooltip_CFormat :: (fmt: *u8, __args: ..Any) -> void #foreign imgui "?SetItemTooltip@ImGui@@YAXPEBDZZ";
    SetItemTooltip :: (fmt: string, __args: ..Any) {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, fmt, ..__args);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        SetItemTooltip_CFormat("%s", formatted_text.data);
    } @PrintLike

    // Popups, Modals
    //  - They block normal mouse hovering detection (and therefore most mouse interactions) behind them.
    //  - If not modal: they can be closed by clicking anywhere outside them, or by pressing ESCAPE.
    //  - Their visibility state (~bool) is held internally instead of being held by the programmer as we are used to with regular Begin*() calls.
    //  - The 3 properties above are related: we need to retain popup visibility state in the library because popups may be closed as any time.
    //  - You can bypass the hovering restriction by using ImGuiHoveredFlags_AllowWhenBlockedByPopup when calling IsItemHovered() or IsWindowHovered().
    //  - IMPORTANT: Popup identifiers are relative to the current ID stack, so OpenPopup and BeginPopup generally needs to be at the same level of the stack.
    //    This is sometimes leading to confusing mistakes. May rework this in the future.
    //  - BeginPopup(): query popup state, if open start appending into the window. Call EndPopup() afterwards if returned true. ImGuiWindowFlags are forwarded to the window.
    //  - BeginPopupModal(): block every interaction behind the window, cannot be closed by user, add a dimming background, has a title bar.
    BeginPopup :: (str_id: *u8, flags: ImGuiWindowFlags = 0) -> bool #foreign imgui "?BeginPopup@ImGui@@YA_NPEBDH@Z";
    BeginPopupModal :: (name: *u8, p_open: *bool = null, flags: ImGuiWindowFlags = 0) -> bool #foreign imgui "?BeginPopupModal@ImGui@@YA_NPEBDPEA_NH@Z";
    EndPopup :: () -> void #foreign imgui "?EndPopup@ImGui@@YAXXZ";

    // Popups: open/close functions
    //  - OpenPopup(): set popup state to open. ImGuiPopupFlags are available for opening options.
    //  - If not modal: they can be closed by clicking anywhere outside them, or by pressing ESCAPE.
    //  - CloseCurrentPopup(): use inside the BeginPopup()/EndPopup() scope to close manually.
    //  - CloseCurrentPopup() is called by default by Selectable()/MenuItem() when activated (FIXME: need some options).
    //  - Use ImGuiPopupFlags_NoOpenOverExistingPopup to avoid opening a popup if there's already one at the same level. This is equivalent to e.g. testing for !IsAnyPopupOpen() prior to OpenPopup().
    //  - Use IsWindowAppearing() after BeginPopup() to tell if a window just opened.
    //  - IMPORTANT: Notice that for OpenPopupOnItemClick() we exceptionally default flags to 1 (== ImGuiPopupFlags_MouseButtonRight) for backward compatibility with older API taking 'int mouse_button = 1' parameter
    OpenPopup :: (str_id: *u8, popup_flags: ImGuiPopupFlags = 0) -> void #foreign imgui "?OpenPopup@ImGui@@YAXPEBDH@Z";
    OpenPopup :: (id: ImGuiID, popup_flags: ImGuiPopupFlags = 0) -> void #foreign imgui "?OpenPopup@ImGui@@YAXIH@Z";
    OpenPopupOnItemClick :: (str_id: *u8 = null, popup_flags: ImGuiPopupFlags = 1) -> void #foreign imgui "?OpenPopupOnItemClick@ImGui@@YAXPEBDH@Z";
    CloseCurrentPopup :: () -> void #foreign imgui "?CloseCurrentPopup@ImGui@@YAXXZ";

    // Popups: open+begin combined functions helpers
    //  - Helpers to do OpenPopup+BeginPopup where the Open action is triggered by e.g. hovering an item and right-clicking.
    //  - They are convenient to easily create context menus, hence the name.
    //  - IMPORTANT: Notice that BeginPopupContextXXX takes ImGuiPopupFlags just like OpenPopup() and unlike BeginPopup(). For full consistency, we may add ImGuiWindowFlags to the BeginPopupContextXXX functions in the future.
    //  - IMPORTANT: Notice that we exceptionally default their flags to 1 (== ImGuiPopupFlags_MouseButtonRight) for backward compatibility with older API taking 'int mouse_button = 1' parameter, so if you add other flags remember to re-add the ImGuiPopupFlags_MouseButtonRight.
    BeginPopupContextItem :: (str_id: *u8 = null, popup_flags: ImGuiPopupFlags = 1) -> bool #foreign imgui "?BeginPopupContextItem@ImGui@@YA_NPEBDH@Z";
    BeginPopupContextWindow :: (str_id: *u8 = null, popup_flags: ImGuiPopupFlags = 1) -> bool #foreign imgui "?BeginPopupContextWindow@ImGui@@YA_NPEBDH@Z";
    BeginPopupContextVoid :: (str_id: *u8 = null, popup_flags: ImGuiPopupFlags = 1) -> bool #foreign imgui "?BeginPopupContextVoid@ImGui@@YA_NPEBDH@Z";

    // Popups: query functions
    //  - IsPopupOpen(): return true if the popup is open at the current BeginPopup() level of the popup stack.
    //  - IsPopupOpen() with ImGuiPopupFlags_AnyPopupId: return true if any popup is open at the current BeginPopup() level of the popup stack.
    //  - IsPopupOpen() with ImGuiPopupFlags_AnyPopupId + ImGuiPopupFlags_AnyPopupLevel: return true if any popup is open.
    IsPopupOpen :: (str_id: *u8, flags: ImGuiPopupFlags = 0) -> bool #foreign imgui "?IsPopupOpen@ImGui@@YA_NPEBDH@Z";

    // Tables
    // - Full-featured replacement for old Columns API.
    // - See Demo->Tables for demo code. See top of imgui_tables.cpp for general commentary.
    // - See ImGuiTableFlags_ and ImGuiTableColumnFlags_ enums for a description of available flags.
    // The typical call flow is:
    // - 1. Call BeginTable(), early out if returning false.
    // - 2. Optionally call TableSetupColumn() to submit column name/flags/defaults.
    // - 3. Optionally call TableSetupScrollFreeze() to request scroll freezing of columns/rows.
    // - 4. Optionally call TableHeadersRow() to submit a header row. Names are pulled from TableSetupColumn() data.
    // - 5. Populate contents:
    //    - In most situations you can use TableNextRow() + TableSetColumnIndex(N) to start appending into a column.
    //    - If you are using tables as a sort of grid, where every column is holding the same type of contents,
    //      you may prefer using TableNextColumn() instead of TableNextRow() + TableSetColumnIndex().
    //      TableNextColumn() will automatically wrap-around into the next row if needed.
    //    - IMPORTANT: Comparatively to the old Columns() API, we need to call TableNextColumn() for the first column!
    //    - Summary of possible call flow:
    //        - TableNextRow() -> TableSetColumnIndex(0) -> Text("Hello 0") -> TableSetColumnIndex(1) -> Text("Hello 1")  // OK
    //        - TableNextRow() -> TableNextColumn()      -> Text("Hello 0") -> TableNextColumn()      -> Text("Hello 1")  // OK
    //        -                   TableNextColumn()      -> Text("Hello 0") -> TableNextColumn()      -> Text("Hello 1")  // OK: TableNextColumn() automatically gets to next row!
    //        - TableNextRow()                           -> Text("Hello 0")                                               // Not OK! Missing TableSetColumnIndex() or TableNextColumn()! Text will not appear!
    // - 5. Call EndTable()
    BeginTable :: (str_id: *u8, columns: s32, flags: ImGuiTableFlags = 0, outer_size: *ImVec2, inner_width: float = 0.0) -> bool #foreign imgui "?BeginTable@ImGui@@YA_NPEBDHHAEBUImVec2@@M@Z";
    BeginTable :: (str_id: *u8, columns: s32, flags: ImGuiTableFlags = 0, outer_size: ImVec2 = ImVec2.{0.0, 0.0}, inner_width: float = 0.0) -> bool #no_context {
        return BeginTable(str_id, columns, flags, *outer_size, inner_width);
    }
    EndTable :: () -> void #foreign imgui "?EndTable@ImGui@@YAXXZ";
    TableNextRow :: (row_flags: ImGuiTableRowFlags = 0, min_row_height: float = 0.0) -> void #foreign imgui "?TableNextRow@ImGui@@YAXHM@Z";
    TableNextColumn :: () -> bool #foreign imgui "?TableNextColumn@ImGui@@YA_NXZ";
    TableSetColumnIndex :: (column_n: s32) -> bool #foreign imgui "?TableSetColumnIndex@ImGui@@YA_NH@Z";

    // Tables: Headers & Columns declaration
    // - Use TableSetupColumn() to specify label, resizing policy, default width/weight, id, various other flags etc.
    // - Use TableHeadersRow() to create a header row and automatically submit a TableHeader() for each column.
    //   Headers are required to perform: reordering, sorting, and opening the context menu.
    //   The context menu can also be made available in columns body using ImGuiTableFlags_ContextMenuInBody.
    // - You may manually submit headers using TableNextRow() + TableHeader() calls, but this is only useful in
    //   some advanced use cases (e.g. adding custom widgets in header row).
    // - Use TableSetupScrollFreeze() to lock columns/rows so they stay visible when scrolled.
    TableSetupColumn :: (label: *u8, flags: ImGuiTableColumnFlags = 0, init_width_or_weight: float = 0.0, user_id: ImGuiID = 0) -> void #foreign imgui "?TableSetupColumn@ImGui@@YAXPEBDHMI@Z";
    TableSetupScrollFreeze :: (cols: s32, rows: s32) -> void #foreign imgui "?TableSetupScrollFreeze@ImGui@@YAXHH@Z";
    TableHeader :: (label: *u8) -> void #foreign imgui "?TableHeader@ImGui@@YAXPEBD@Z";
    TableHeadersRow :: () -> void #foreign imgui "?TableHeadersRow@ImGui@@YAXXZ";
    TableAngledHeadersRow :: () -> void #foreign imgui "?TableAngledHeadersRow@ImGui@@YAXXZ";

    // Tables: Sorting & Miscellaneous functions
    // - Sorting: call TableGetSortSpecs() to retrieve latest sort specs for the table. NULL when not sorting.
    //   When 'sort_specs->SpecsDirty == true' you should sort your data. It will be true when sorting specs have
    //   changed since last call, or the first time. Make sure to set 'SpecsDirty = false' after sorting,
    //   else you may wastefully sort your data every frame!
    // - Functions args 'int column_n' treat the default value of -1 as the same as passing the current column index.
    TableGetSortSpecs :: () -> *ImGuiTableSortSpecs #foreign imgui "?TableGetSortSpecs@ImGui@@YAPEAUImGuiTableSortSpecs@@XZ";
    TableGetColumnCount :: () -> s32 #foreign imgui "?TableGetColumnCount@ImGui@@YAHXZ";
    TableGetColumnIndex :: () -> s32 #foreign imgui "?TableGetColumnIndex@ImGui@@YAHXZ";
    TableGetRowIndex :: () -> s32 #foreign imgui "?TableGetRowIndex@ImGui@@YAHXZ";
    TableGetColumnName :: (column_n: s32 = -1) -> *u8 #foreign imgui "?TableGetColumnName@ImGui@@YAPEBDH@Z";
    TableGetColumnFlags :: (column_n: s32 = -1) -> ImGuiTableColumnFlags #foreign imgui "?TableGetColumnFlags@ImGui@@YAHH@Z";
    TableSetColumnEnabled :: (column_n: s32, v: bool) -> void #foreign imgui "?TableSetColumnEnabled@ImGui@@YAXH_N@Z";
    TableGetHoveredColumn :: () -> s32 #foreign imgui "?TableGetHoveredColumn@ImGui@@YAHXZ";
    TableSetBgColor :: (target: ImGuiTableBgTarget, color: ImU32, column_n: s32 = -1) -> void #foreign imgui "?TableSetBgColor@ImGui@@YAXHIH@Z";

    // Legacy Columns API (prefer using Tables!)
    // - You can also use SameLine(pos_x) to mimic simplified columns.
    Columns :: (count: s32 = 1, id: *u8 = null, borders := true) -> void #foreign imgui "?Columns@ImGui@@YAXHPEBD_N@Z";
    NextColumn :: () -> void #foreign imgui "?NextColumn@ImGui@@YAXXZ";
    GetColumnIndex :: () -> s32 #foreign imgui "?GetColumnIndex@ImGui@@YAHXZ";
    GetColumnWidth :: (column_index: s32 = -1) -> float #foreign imgui "?GetColumnWidth@ImGui@@YAMH@Z";
    SetColumnWidth :: (column_index: s32, width: float) -> void #foreign imgui "?SetColumnWidth@ImGui@@YAXHM@Z";
    GetColumnOffset :: (column_index: s32 = -1) -> float #foreign imgui "?GetColumnOffset@ImGui@@YAMH@Z";
    SetColumnOffset :: (column_index: s32, offset_x: float) -> void #foreign imgui "?SetColumnOffset@ImGui@@YAXHM@Z";
    GetColumnsCount :: () -> s32 #foreign imgui "?GetColumnsCount@ImGui@@YAHXZ";

    // Tab Bars, Tabs
    // - Note: Tabs are automatically created by the docking system (when in 'docking' branch). Use this to create tab bars/tabs yourself.
    BeginTabBar :: (str_id: *u8, flags: ImGuiTabBarFlags = 0) -> bool #foreign imgui "?BeginTabBar@ImGui@@YA_NPEBDH@Z";
    EndTabBar :: () -> void #foreign imgui "?EndTabBar@ImGui@@YAXXZ";
    BeginTabItem :: (label: *u8, p_open: *bool = null, flags: ImGuiTabItemFlags = 0) -> bool #foreign imgui "?BeginTabItem@ImGui@@YA_NPEBDPEA_NH@Z";
    EndTabItem :: () -> void #foreign imgui "?EndTabItem@ImGui@@YAXXZ";
    TabItemButton :: (label: *u8, flags: ImGuiTabItemFlags = 0) -> bool #foreign imgui "?TabItemButton@ImGui@@YA_NPEBDH@Z";
    SetTabItemClosed :: (tab_or_docked_window_label: *u8) -> void #foreign imgui "?SetTabItemClosed@ImGui@@YAXPEBD@Z";

    // Logging/Capture
    // - All text output from the interface can be captured into tty/file/clipboard. By default, tree nodes are automatically opened during logging.
    LogToTTY :: (auto_open_depth: s32 = -1) -> void #foreign imgui "?LogToTTY@ImGui@@YAXH@Z";
    LogToFile :: (auto_open_depth: s32 = -1, filename: *u8 = null) -> void #foreign imgui "?LogToFile@ImGui@@YAXHPEBD@Z";
    LogToClipboard :: (auto_open_depth: s32 = -1) -> void #foreign imgui "?LogToClipboard@ImGui@@YAXH@Z";
    LogFinish :: () -> void #foreign imgui "?LogFinish@ImGui@@YAXXZ";
    LogButtons :: () -> void #foreign imgui "?LogButtons@ImGui@@YAXXZ";
    LogText_CFormat :: (fmt: *u8, __args: ..Any) -> void #foreign imgui "?LogText@ImGui@@YAXPEBDZZ";
    LogText :: (fmt: string, __args: ..Any) {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, fmt, ..__args);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        LogText_CFormat("%s", formatted_text.data);
    } @PrintLike

    // Drag and Drop
    // - On source items, call BeginDragDropSource(), if it returns true also call SetDragDropPayload() + EndDragDropSource().
    // - On target candidates, call BeginDragDropTarget(), if it returns true also call AcceptDragDropPayload() + EndDragDropTarget().
    // - If you stop calling BeginDragDropSource() the payload is preserved however it won't have a preview tooltip (we currently display a fallback "..." tooltip, see #1725)
    // - An item can be both drag source and drop target.
    BeginDragDropSource :: (flags: ImGuiDragDropFlags = 0) -> bool #foreign imgui "?BeginDragDropSource@ImGui@@YA_NH@Z";
    SetDragDropPayload :: (type: *u8, data: *void, sz: u64, cond: ImGuiCond = 0) -> bool #foreign imgui "?SetDragDropPayload@ImGui@@YA_NPEBDPEBX_KH@Z";
    EndDragDropSource :: () -> void #foreign imgui "?EndDragDropSource@ImGui@@YAXXZ";
    BeginDragDropTarget :: () -> bool #foreign imgui "?BeginDragDropTarget@ImGui@@YA_NXZ";
    AcceptDragDropPayload :: (type: *u8, flags: ImGuiDragDropFlags = 0) -> *ImGuiPayload #foreign imgui "?AcceptDragDropPayload@ImGui@@YAPEBUImGuiPayload@@PEBDH@Z";
    EndDragDropTarget :: () -> void #foreign imgui "?EndDragDropTarget@ImGui@@YAXXZ";
    GetDragDropPayload :: () -> *ImGuiPayload #foreign imgui "?GetDragDropPayload@ImGui@@YAPEBUImGuiPayload@@XZ";

    // Disabling [BETA API]
    // - Disable all user interactions and dim items visuals (applying style.DisabledAlpha over current colors)
    // - Those can be nested but it cannot be used to enable an already disabled section (a single BeginDisabled(true) in the stack is enough to keep everything disabled)
    // - Tooltips windows by exception are opted out of disabling.
    // - BeginDisabled(false)/EndDisabled() essentially does nothing but is provided to facilitate use of boolean expressions (as a micro-optimization: if you have tens of thousands of BeginDisabled(false)/EndDisabled() pairs, you might want to reformulate your code to avoid making those calls)
    BeginDisabled :: (disabled := true) -> void #foreign imgui "?BeginDisabled@ImGui@@YAX_N@Z";
    EndDisabled :: () -> void #foreign imgui "?EndDisabled@ImGui@@YAXXZ";

    // Clipping
    // - Mouse hovering is affected by ImGui::PushClipRect() calls, unlike direct calls to ImDrawList::PushClipRect() which are render only.
    PushClipRect :: (clip_rect_min: *ImVec2, clip_rect_max: *ImVec2, intersect_with_current_clip_rect: bool) -> void #foreign imgui "?PushClipRect@ImGui@@YAXAEBUImVec2@@0_N@Z";
    PushClipRect :: (clip_rect_min: ImVec2, clip_rect_max: ImVec2, intersect_with_current_clip_rect: bool) #no_context {
        PushClipRect(*clip_rect_min, *clip_rect_max, intersect_with_current_clip_rect);
    }
    PopClipRect :: () -> void #foreign imgui "?PopClipRect@ImGui@@YAXXZ";

    // Focus, Activation
    SetItemDefaultFocus :: () -> void #foreign imgui "?SetItemDefaultFocus@ImGui@@YAXXZ";
    SetKeyboardFocusHere :: (offset: s32 = 0) -> void #foreign imgui "?SetKeyboardFocusHere@ImGui@@YAXH@Z";

    // Keyboard/Gamepad Navigation
    SetNavCursorVisible :: (visible: bool) -> void #foreign imgui "?SetNavCursorVisible@ImGui@@YAX_N@Z";

    // Overlapping mode
    SetNextItemAllowOverlap :: () -> void #foreign imgui "?SetNextItemAllowOverlap@ImGui@@YAXXZ";

    // Item/Widgets Utilities and Query Functions
    // - Most of the functions are referring to the previous Item that has been submitted.
    // - See Demo Window under "Widgets->Querying Status" for an interactive visualization of most of those functions.
    IsItemHovered :: (flags: ImGuiHoveredFlags = 0) -> bool #foreign imgui "?IsItemHovered@ImGui@@YA_NH@Z";
    IsItemActive :: () -> bool #foreign imgui "?IsItemActive@ImGui@@YA_NXZ";
    IsItemFocused :: () -> bool #foreign imgui "?IsItemFocused@ImGui@@YA_NXZ";
    IsItemClicked :: (mouse_button: ImGuiMouseButton = 0) -> bool #foreign imgui "?IsItemClicked@ImGui@@YA_NH@Z";
    IsItemVisible :: () -> bool #foreign imgui "?IsItemVisible@ImGui@@YA_NXZ";
    IsItemEdited :: () -> bool #foreign imgui "?IsItemEdited@ImGui@@YA_NXZ";
    IsItemActivated :: () -> bool #foreign imgui "?IsItemActivated@ImGui@@YA_NXZ";
    IsItemDeactivated :: () -> bool #foreign imgui "?IsItemDeactivated@ImGui@@YA_NXZ";
    IsItemDeactivatedAfterEdit :: () -> bool #foreign imgui "?IsItemDeactivatedAfterEdit@ImGui@@YA_NXZ";
    IsItemToggledOpen :: () -> bool #foreign imgui "?IsItemToggledOpen@ImGui@@YA_NXZ";
    IsAnyItemHovered :: () -> bool #foreign imgui "?IsAnyItemHovered@ImGui@@YA_NXZ";
    IsAnyItemActive :: () -> bool #foreign imgui "?IsAnyItemActive@ImGui@@YA_NXZ";
    IsAnyItemFocused :: () -> bool #foreign imgui "?IsAnyItemFocused@ImGui@@YA_NXZ";
    GetItemID :: () -> ImGuiID #foreign imgui "?GetItemID@ImGui@@YAIXZ";
    GetItemRectMin :: () -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetItemRectMin@ImGui@@YA?AUImVec2@@XZ";
    GetItemRectMax :: () -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetItemRectMax@ImGui@@YA?AUImVec2@@XZ";
    GetItemRectSize :: () -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetItemRectSize@ImGui@@YA?AUImVec2@@XZ";

    // Viewports
    // - Currently represents the Platform Window created by the application which is hosting our Dear ImGui windows.
    // - In 'docking' branch with multi-viewport enabled, we extend this concept to have multiple active viewports.
    // - In the future we will extend this concept further to also represent Platform Monitor and support a "no main platform window" operation mode.
    GetMainViewport :: () -> *ImGuiViewport #foreign imgui "?GetMainViewport@ImGui@@YAPEAUImGuiViewport@@XZ";

    // Background/Foreground Draw Lists
    GetBackgroundDrawList :: () -> *ImDrawList #foreign imgui "?GetBackgroundDrawList@ImGui@@YAPEAUImDrawList@@XZ";
    GetForegroundDrawList :: () -> *ImDrawList #foreign imgui "?GetForegroundDrawList@ImGui@@YAPEAUImDrawList@@XZ";

    // Miscellaneous Utilities
    IsRectVisible :: (size: *ImVec2) -> bool #foreign imgui "?IsRectVisible@ImGui@@YA_NAEBUImVec2@@@Z";
    IsRectVisible :: (size: ImVec2) -> bool #no_context {
        return IsRectVisible(*size);
    }
    IsRectVisible :: (rect_min: *ImVec2, rect_max: *ImVec2) -> bool #foreign imgui "?IsRectVisible@ImGui@@YA_NAEBUImVec2@@0@Z";
    IsRectVisible :: (rect_min: ImVec2, rect_max: ImVec2) -> bool #no_context {
        return IsRectVisible(*rect_min, *rect_max);
    }
    GetTime :: () -> float64 #foreign imgui "?GetTime@ImGui@@YANXZ";
    GetFrameCount :: () -> s32 #foreign imgui "?GetFrameCount@ImGui@@YAHXZ";
    GetDrawListSharedData :: () -> *ImDrawListSharedData #foreign imgui "?GetDrawListSharedData@ImGui@@YAPEAUImDrawListSharedData@@XZ";
    GetStyleColorName :: (idx: ImGuiCol) -> *u8 #foreign imgui "?GetStyleColorName@ImGui@@YAPEBDH@Z";
    SetStateStorage :: (storage: *ImGuiStorage) -> void #foreign imgui "?SetStateStorage@ImGui@@YAXPEAUImGuiStorage@@@Z";
    GetStateStorage :: () -> *ImGuiStorage #foreign imgui "?GetStateStorage@ImGui@@YAPEAUImGuiStorage@@XZ";

    // Text Utilities
    CalcTextSize :: (text: *u8, text_end: *u8 = null, hide_text_after_double_hash := false, wrap_width: float = -1.0) -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?CalcTextSize@ImGui@@YA?AUImVec2@@PEBD0_NM@Z";

    // Color Utilities
    ColorConvertU32ToFloat4 :: (in: ImU32) -> ImVec4 #cpp_return_type_is_non_pod #foreign imgui "?ColorConvertU32ToFloat4@ImGui@@YA?AUImVec4@@I@Z";
    ColorConvertFloat4ToU32 :: (in: *ImVec4) -> ImU32 #foreign imgui "?ColorConvertFloat4ToU32@ImGui@@YAIAEBUImVec4@@@Z";
    ColorConvertFloat4ToU32 :: (in: ImVec4) -> ImU32 #no_context {
        return ColorConvertFloat4ToU32(*in);
    }
    ColorConvertRGBtoHSV :: (r: float, g: float, b: float, out_h: *float, out_s: *float, out_v: *float) -> void #foreign imgui "?ColorConvertRGBtoHSV@ImGui@@YAXMMMAEAM00@Z";
    ColorConvertHSVtoRGB :: (h: float, s: float, v: float, out_r: *float, out_g: *float, out_b: *float) -> void #foreign imgui "?ColorConvertHSVtoRGB@ImGui@@YAXMMMAEAM00@Z";

    // Inputs Utilities: Keyboard/Mouse/Gamepad
    // - the ImGuiKey enum contains all possible keyboard, mouse and gamepad inputs (e.g. ImGuiKey_A, ImGuiKey_MouseLeft, ImGuiKey_GamepadDpadUp...).
    // - (legacy: before v1.87, we used ImGuiKey to carry native/user indices as defined by each backends. This was obsoleted in 1.87 (2022-02) and completely removed in 1.91.5 (2024-11). See https://github.com/ocornut/imgui/issues/4921)
    // - (legacy: any use of ImGuiKey will assert when key < 512 to detect passing legacy native/user indices)
    IsKeyDown :: (key: ImGuiKey) -> bool #foreign imgui "?IsKeyDown@ImGui@@YA_NW4ImGuiKey@@@Z";
    IsKeyPressed :: (key: ImGuiKey, repeat := true) -> bool #foreign imgui "?IsKeyPressed@ImGui@@YA_NW4ImGuiKey@@_N@Z";
    IsKeyReleased :: (key: ImGuiKey) -> bool #foreign imgui "?IsKeyReleased@ImGui@@YA_NW4ImGuiKey@@@Z";
    IsKeyChordPressed :: (key_chord: ImGuiKeyChord) -> bool #foreign imgui "?IsKeyChordPressed@ImGui@@YA_NH@Z";
    GetKeyPressedAmount :: (key: ImGuiKey, repeat_delay: float, rate: float) -> s32 #foreign imgui "?GetKeyPressedAmount@ImGui@@YAHW4ImGuiKey@@MM@Z";
    GetKeyName :: (key: ImGuiKey) -> *u8 #foreign imgui "?GetKeyName@ImGui@@YAPEBDW4ImGuiKey@@@Z";
    SetNextFrameWantCaptureKeyboard :: (want_capture_keyboard: bool) -> void #foreign imgui "?SetNextFrameWantCaptureKeyboard@ImGui@@YAX_N@Z";

    // Inputs Utilities: Shortcut Testing & Routing [BETA]
    // - ImGuiKeyChord = a ImGuiKey + optional ImGuiMod_Alt/ImGuiMod_Ctrl/ImGuiMod_Shift/ImGuiMod_Super.
    //       ImGuiKey_C                          // Accepted by functions taking ImGuiKey or ImGuiKeyChord arguments
    //       ImGuiMod_Ctrl | ImGuiKey_C          // Accepted by functions taking ImGuiKeyChord arguments
    //   only ImGuiMod_XXX values are legal to combine with an ImGuiKey. You CANNOT combine two ImGuiKey values.
    // - The general idea is that several callers may register interest in a shortcut, and only one owner gets it.
    //      Parent   -> call Shortcut(Ctrl+S)    // When Parent is focused, Parent gets the shortcut.
    //        Child1 -> call Shortcut(Ctrl+S)    // When Child1 is focused, Child1 gets the shortcut (Child1 overrides Parent shortcuts)
    //        Child2 -> no call                  // When Child2 is focused, Parent gets the shortcut.
    //   The whole system is order independent, so if Child1 makes its calls before Parent, results will be identical.
    //   This is an important property as it facilitate working with foreign code or larger codebase.
    // - To understand the difference:
    //   - IsKeyChordPressed() compares mods and call IsKeyPressed() -> function has no side-effect.
    //   - Shortcut() submits a route, routes are resolved, if it currently can be routed it calls IsKeyChordPressed() -> function has (desirable) side-effects as it can prevents another call from getting the route.
    // - Visualize registered routes in 'Metrics/Debugger->Inputs'.
    Shortcut :: (key_chord: ImGuiKeyChord, flags: ImGuiInputFlags = 0) -> bool #foreign imgui "?Shortcut@ImGui@@YA_NHH@Z";
    SetNextItemShortcut :: (key_chord: ImGuiKeyChord, flags: ImGuiInputFlags = 0) -> void #foreign imgui "?SetNextItemShortcut@ImGui@@YAXHH@Z";

    // Inputs Utilities: Key/Input Ownership [BETA]
    // - One common use case would be to allow your items to disable standard inputs behaviors such
    //   as Tab or Alt key handling, Mouse Wheel scrolling, etc.
    //   e.g. Button(...); SetItemKeyOwner(ImGuiKey_MouseWheelY); to make hovering/activating a button disable wheel for scrolling.
    // - Reminder ImGuiKey enum include access to mouse buttons and gamepad, so key ownership can apply to them.
    // - Many related features are still in imgui_internal.h. For instance, most IsKeyXXX()/IsMouseXXX() functions have an owner-id-aware version.
    SetItemKeyOwner :: (key: ImGuiKey) -> void #foreign imgui "?SetItemKeyOwner@ImGui@@YAXW4ImGuiKey@@@Z";

    // Inputs Utilities: Mouse
    // - To refer to a mouse button, you may use named enums in your code e.g. ImGuiMouseButton_Left, ImGuiMouseButton_Right.
    // - You can also use regular integer: it is forever guaranteed that 0=Left, 1=Right, 2=Middle.
    // - Dragging operations are only reported after mouse has moved a certain distance away from the initial clicking position (see 'lock_threshold' and 'io.MouseDraggingThreshold')
    IsMouseDown :: (button: ImGuiMouseButton) -> bool #foreign imgui "?IsMouseDown@ImGui@@YA_NH@Z";
    IsMouseClicked :: (button: ImGuiMouseButton, repeat := false) -> bool #foreign imgui "?IsMouseClicked@ImGui@@YA_NH_N@Z";
    IsMouseReleased :: (button: ImGuiMouseButton) -> bool #foreign imgui "?IsMouseReleased@ImGui@@YA_NH@Z";
    IsMouseDoubleClicked :: (button: ImGuiMouseButton) -> bool #foreign imgui "?IsMouseDoubleClicked@ImGui@@YA_NH@Z";
    IsMouseReleasedWithDelay :: (button: ImGuiMouseButton, delay: float) -> bool #foreign imgui "?IsMouseReleasedWithDelay@ImGui@@YA_NHM@Z";
    GetMouseClickedCount :: (button: ImGuiMouseButton) -> s32 #foreign imgui "?GetMouseClickedCount@ImGui@@YAHH@Z";
    IsMouseHoveringRect :: (r_min: *ImVec2, r_max: *ImVec2, clip := true) -> bool #foreign imgui "?IsMouseHoveringRect@ImGui@@YA_NAEBUImVec2@@0_N@Z";
    IsMouseHoveringRect :: (r_min: ImVec2, r_max: ImVec2, clip := true) -> bool #no_context {
        return IsMouseHoveringRect(*r_min, *r_max, clip);
    }
    IsMousePosValid :: (mouse_pos: *ImVec2 = null) -> bool #foreign imgui "?IsMousePosValid@ImGui@@YA_NPEBUImVec2@@@Z";
    IsAnyMouseDown :: () -> bool #foreign imgui "?IsAnyMouseDown@ImGui@@YA_NXZ";
    GetMousePos :: () -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetMousePos@ImGui@@YA?AUImVec2@@XZ";
    GetMousePosOnOpeningCurrentPopup :: () -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetMousePosOnOpeningCurrentPopup@ImGui@@YA?AUImVec2@@XZ";
    IsMouseDragging :: (button: ImGuiMouseButton, lock_threshold: float = -1.0) -> bool #foreign imgui "?IsMouseDragging@ImGui@@YA_NHM@Z";
    GetMouseDragDelta :: (button: ImGuiMouseButton = 0, lock_threshold: float = -1.0) -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetMouseDragDelta@ImGui@@YA?AUImVec2@@HM@Z";
    ResetMouseDragDelta :: (button: ImGuiMouseButton = 0) -> void #foreign imgui "?ResetMouseDragDelta@ImGui@@YAXH@Z";
    GetMouseCursor :: () -> ImGuiMouseCursor #foreign imgui "?GetMouseCursor@ImGui@@YAHXZ";
    SetMouseCursor :: (cursor_type: ImGuiMouseCursor) -> void #foreign imgui "?SetMouseCursor@ImGui@@YAXH@Z";
    SetNextFrameWantCaptureMouse :: (want_capture_mouse: bool) -> void #foreign imgui "?SetNextFrameWantCaptureMouse@ImGui@@YAX_N@Z";

    // Clipboard Utilities
    // - Also see the LogToClipboard() function to capture GUI into clipboard, or easily output text data to the clipboard.
    GetClipboardText :: () -> *u8 #foreign imgui "?GetClipboardText@ImGui@@YAPEBDXZ";
    SetClipboardText :: (text: *u8) -> void #foreign imgui "?SetClipboardText@ImGui@@YAXPEBD@Z";

    // Settings/.Ini Utilities
    // - The disk functions are automatically called if io.IniFilename != NULL (default is "imgui.ini").
    // - Set io.IniFilename to NULL to load/save manually. Read io.WantSaveIniSettings description about handling .ini saving manually.
    // - Important: default value "imgui.ini" is relative to current working dir! Most apps will want to lock this to an absolute path (e.g. same path as executables).
    LoadIniSettingsFromDisk :: (ini_filename: *u8) -> void #foreign imgui "?LoadIniSettingsFromDisk@ImGui@@YAXPEBD@Z";
    LoadIniSettingsFromMemory :: (ini_data: *u8, ini_size: u64 = 0) -> void #foreign imgui "?LoadIniSettingsFromMemory@ImGui@@YAXPEBD_K@Z";
    SaveIniSettingsToDisk :: (ini_filename: *u8) -> void #foreign imgui "?SaveIniSettingsToDisk@ImGui@@YAXPEBD@Z";
    SaveIniSettingsToMemory :: (out_ini_size: *u64 = null) -> *u8 #foreign imgui "?SaveIniSettingsToMemory@ImGui@@YAPEBDPEA_K@Z";

    // Debug Utilities
    // - Your main debugging friend is the ShowMetricsWindow() function, which is also accessible from Demo->Tools->Metrics Debugger
    DebugTextEncoding :: (text: *u8) -> void #foreign imgui "?DebugTextEncoding@ImGui@@YAXPEBD@Z";
    DebugFlashStyleColor :: (idx: ImGuiCol) -> void #foreign imgui "?DebugFlashStyleColor@ImGui@@YAXH@Z";
    DebugStartItemPicker :: () -> void #foreign imgui "?DebugStartItemPicker@ImGui@@YAXXZ";
    DebugCheckVersionAndDataLayout :: (version_str: *u8, sz_io: u64, sz_style: u64, sz_vec2: u64, sz_vec4: u64, sz_drawvert: u64, sz_drawidx: u64) -> bool #foreign imgui "?DebugCheckVersionAndDataLayout@ImGui@@YA_NPEBD_K11111@Z";

    DebugLog_CFormat :: (fmt: *u8, __args: ..Any) -> void #foreign imgui "?DebugLog@ImGui@@YAXPEBDZZ";
    DebugLog :: (fmt: string, __args: ..Any) {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, fmt, ..__args);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        DebugLog_CFormat("%s", formatted_text.data);
    } @PrintLike

    // Memory Allocators
    // - Those functions are not reliant on the current context.
    // - DLL users: heaps and globals are not shared across DLL boundaries! You will need to call SetCurrentContext() + SetAllocatorFunctions()
    //   for each static/DLL boundary you are calling from. Read "Context and Memory Allocators" section of imgui.cpp for more details.
    SetAllocatorFunctions :: (alloc_func: ImGuiMemAllocFunc, free_func: ImGuiMemFreeFunc, user_data: *void = null) -> void #foreign imgui "?SetAllocatorFunctions@ImGui@@YAXP6APEAX_KPEAX@ZP6AX11@Z1@Z";
    GetAllocatorFunctions :: (p_alloc_func: *ImGuiMemAllocFunc, p_free_func: *ImGuiMemFreeFunc, p_user_data: **void) -> void #foreign imgui "?GetAllocatorFunctions@ImGui@@YAXPEAP6APEAX_KPEAX@ZPEAP6AX11@ZPEAPEAX@Z";
    MemAlloc :: (size: u64) -> *void #foreign imgui "?MemAlloc@ImGui@@YAPEAX_K@Z";
    MemFree :: (ptr: *void) -> void #foreign imgui "?MemFree@ImGui@@YAXPEAX@Z";

    SetWindowFontScale :: (scale: float) -> void #foreign imgui "?SetWindowFontScale@ImGui@@YAXM@Z";

    // OBSOLETED in 1.91.9 (from February 2025)
    Image :: (tex_ref: ImTextureRef, image_size: *ImVec2, uv0: *ImVec2, uv1: *ImVec2, tint_col: *ImVec4, border_col: *ImVec4) -> void #foreign imgui "?Image@ImGui@@YAXUImTextureRef@@AEBUImVec2@@11AEBUImVec4@@2@Z";
    Image :: (tex_ref: ImTextureRef, image_size: ImVec2, uv0: ImVec2, uv1: ImVec2, tint_col: ImVec4, border_col: ImVec4) #no_context {
        Image(tex_ref, *image_size, *uv0, *uv1, *tint_col, *border_col);
    }

    GetContentRegionMax :: () -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetContentRegionMax@ImGui@@YA?AUImVec2@@XZ";
    GetWindowContentRegionMin :: () -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetWindowContentRegionMin@ImGui@@YA?AUImVec2@@XZ";
    GetWindowContentRegionMax :: () -> ImVec2 #cpp_return_type_is_non_pod #foreign imgui "?GetWindowContentRegionMax@ImGui@@YA?AUImVec2@@XZ";

    Combo :: (label: *u8, current_item: *s32, old_callback: #type (user_data: *void, idx: s32, out_text: **u8) -> bool #c_call, user_data: *void, items_count: s32, popup_max_height_in_items: s32 = -1) -> bool #foreign imgui "?Combo@ImGui@@YA_NPEBDPEAHP6A_NPEAXHPEAPEBD@Z2HH@Z";
    ListBox :: (label: *u8, current_item: *s32, old_callback: #type (user_data: *void, idx: s32, out_text: **u8) -> bool #c_call, user_data: *void, items_count: s32, height_in_items: s32 = -1) -> bool #foreign imgui "?ListBox@ImGui@@YA_NPEBDPEAHP6A_NPEAXHPEAPEBD@Z2HH@Z";

    // OBSOLETED in 1.89.7 (from June 2023)
    SetItemAllowOverlap :: () -> void #foreign imgui "?SetItemAllowOverlap@ImGui@@YAXXZ";
}

// Flags for ImGui::Begin()
// (Those are per-window flags. There are shared flags in ImGuiIO: io.ConfigWindowsResizeFromEdges and io.ConfigWindowsMoveFromTitleBarOnly)
ImGuiWindowFlags_ :: enum s32 {
    None                      :: 0;
    NoTitleBar                :: 1;
    NoResize                  :: 2;
    NoMove                    :: 4;
    NoScrollbar               :: 8;
    NoScrollWithMouse         :: 16;
    NoCollapse                :: 32;
    AlwaysAutoResize          :: 64;
    NoBackground              :: 128;
    NoSavedSettings           :: 256;
    NoMouseInputs             :: 512;
    MenuBar                   :: 1024;
    HorizontalScrollbar       :: 2048;
    NoFocusOnAppearing        :: 4096;
    NoBringToFrontOnFocus     :: 8192;
    AlwaysVerticalScrollbar   :: 16384;
    AlwaysHorizontalScrollbar :: 32768;
    NoNavInputs               :: 65536;
    NoNavFocus                :: 131072;
    UnsavedDocument           :: 262144;
    NoNav                     :: 196608;
    NoDecoration              :: 43;
    NoInputs                  :: 197120;

    ChildWindow               :: 16777216;
    Tooltip                   :: 33554432;
    Popup                     :: 67108864;
    Modal                     :: 134217728;
    ChildMenu                 :: 268435456;

    NavFlattened              :: 536870912;
    AlwaysUseWindowPadding    :: 1073741824;

    ImGuiWindowFlags_None                      :: None;
    ImGuiWindowFlags_NoTitleBar                :: NoTitleBar;
    ImGuiWindowFlags_NoResize                  :: NoResize;
    ImGuiWindowFlags_NoMove                    :: NoMove;
    ImGuiWindowFlags_NoScrollbar               :: NoScrollbar;
    ImGuiWindowFlags_NoScrollWithMouse         :: NoScrollWithMouse;
    ImGuiWindowFlags_NoCollapse                :: NoCollapse;
    ImGuiWindowFlags_AlwaysAutoResize          :: AlwaysAutoResize;
    ImGuiWindowFlags_NoBackground              :: NoBackground;
    ImGuiWindowFlags_NoSavedSettings           :: NoSavedSettings;
    ImGuiWindowFlags_NoMouseInputs             :: NoMouseInputs;
    ImGuiWindowFlags_MenuBar                   :: MenuBar;
    ImGuiWindowFlags_HorizontalScrollbar       :: HorizontalScrollbar;
    ImGuiWindowFlags_NoFocusOnAppearing        :: NoFocusOnAppearing;
    ImGuiWindowFlags_NoBringToFrontOnFocus     :: NoBringToFrontOnFocus;
    ImGuiWindowFlags_AlwaysVerticalScrollbar   :: AlwaysVerticalScrollbar;
    ImGuiWindowFlags_AlwaysHorizontalScrollbar :: AlwaysHorizontalScrollbar;
    ImGuiWindowFlags_NoNavInputs               :: NoNavInputs;
    ImGuiWindowFlags_NoNavFocus                :: NoNavFocus;
    ImGuiWindowFlags_UnsavedDocument           :: UnsavedDocument;
    ImGuiWindowFlags_NoNav                     :: NoNav;
    ImGuiWindowFlags_NoDecoration              :: NoDecoration;
    ImGuiWindowFlags_NoInputs                  :: NoInputs;

    ImGuiWindowFlags_ChildWindow               :: ChildWindow;
    ImGuiWindowFlags_Tooltip                   :: Tooltip;
    ImGuiWindowFlags_Popup                     :: Popup;
    ImGuiWindowFlags_Modal                     :: Modal;
    ImGuiWindowFlags_ChildMenu                 :: ChildMenu;

    ImGuiWindowFlags_NavFlattened              :: NavFlattened;
    ImGuiWindowFlags_AlwaysUseWindowPadding    :: AlwaysUseWindowPadding;
}

// Flags for ImGui::BeginChild()
// (Legacy: bit 0 must always correspond to ImGuiChildFlags_Borders to be backward compatible with old API using 'bool border = false'.)
// About using AutoResizeX/AutoResizeY flags:
// - May be combined with SetNextWindowSizeConstraints() to set a min/max size for each axis (see "Demo->Child->Auto-resize with Constraints").
// - Size measurement for a given axis is only performed when the child window is within visible boundaries, or is just appearing.
//   - This allows BeginChild() to return false when not within boundaries (e.g. when scrolling), which is more optimal. BUT it won't update its auto-size while clipped.
//     While not perfect, it is a better default behavior as the always-on performance gain is more valuable than the occasional "resizing after becoming visible again" glitch.
//   - You may also use ImGuiChildFlags_AlwaysAutoResize to force an update even when child window is not in view.
//     HOWEVER PLEASE UNDERSTAND THAT DOING SO WILL PREVENT BeginChild() FROM EVER RETURNING FALSE, disabling benefits of coarse clipping.
ImGuiChildFlags_ :: enum s32 {
    None                   :: 0;
    Borders                :: 1;
    AlwaysUseWindowPadding :: 2;
    ResizeX                :: 4;
    ResizeY                :: 8;
    AutoResizeX            :: 16;
    AutoResizeY            :: 32;
    AlwaysAutoResize       :: 64;
    FrameStyle             :: 128;
    NavFlattened           :: 256;

    Border                 :: 1;

    ImGuiChildFlags_None                   :: None;
    ImGuiChildFlags_Borders                :: Borders;
    ImGuiChildFlags_AlwaysUseWindowPadding :: AlwaysUseWindowPadding;
    ImGuiChildFlags_ResizeX                :: ResizeX;
    ImGuiChildFlags_ResizeY                :: ResizeY;
    ImGuiChildFlags_AutoResizeX            :: AutoResizeX;
    ImGuiChildFlags_AutoResizeY            :: AutoResizeY;
    ImGuiChildFlags_AlwaysAutoResize       :: AlwaysAutoResize;
    ImGuiChildFlags_FrameStyle             :: FrameStyle;
    ImGuiChildFlags_NavFlattened           :: NavFlattened;

    ImGuiChildFlags_Border                 :: Border;
}

// Flags for ImGui::PushItemFlag()
// (Those are shared by all items)
ImGuiItemFlags_ :: enum s32 {
    None              :: 0;
    NoTabStop         :: 1;
    NoNav             :: 2;
    NoNavDefaultFocus :: 4;
    ButtonRepeat      :: 8;
    AutoClosePopups   :: 16;
    AllowDuplicateId  :: 32;

    ImGuiItemFlags_None              :: None;
    ImGuiItemFlags_NoTabStop         :: NoTabStop;
    ImGuiItemFlags_NoNav             :: NoNav;
    ImGuiItemFlags_NoNavDefaultFocus :: NoNavDefaultFocus;
    ImGuiItemFlags_ButtonRepeat      :: ButtonRepeat;
    ImGuiItemFlags_AutoClosePopups   :: AutoClosePopups;
    ImGuiItemFlags_AllowDuplicateId  :: AllowDuplicateId;
}

// Flags for ImGui::InputText()
// (Those are per-item flags. There are shared flags in ImGuiIO: io.ConfigInputTextCursorBlink and io.ConfigInputTextEnterKeepActive)
ImGuiInputTextFlags_ :: enum s32 {
    None                :: 0;
    CharsDecimal        :: 1;
    CharsHexadecimal    :: 2;
    CharsScientific     :: 4;
    CharsUppercase      :: 8;
    CharsNoBlank        :: 16;

    AllowTabInput       :: 32;
    EnterReturnsTrue    :: 64;
    EscapeClearsAll     :: 128;
    CtrlEnterForNewLine :: 256;

    ReadOnly            :: 512;
    Password            :: 1024;
    AlwaysOverwrite     :: 2048;
    AutoSelectAll       :: 4096;
    ParseEmptyRefVal    :: 8192;
    DisplayEmptyRefVal  :: 16384;
    NoHorizontalScroll  :: 32768;
    NoUndoRedo          :: 65536;

    ElideLeft           :: 131072;

    CallbackCompletion  :: 262144;
    CallbackHistory     :: 524288;
    CallbackAlways      :: 1048576;
    CallbackCharFilter  :: 2097152;
    CallbackResize      :: 4194304;
    CallbackEdit        :: 8388608;

    WordWrap            :: 16777216;

    ImGuiInputTextFlags_None                :: None;
    ImGuiInputTextFlags_CharsDecimal        :: CharsDecimal;
    ImGuiInputTextFlags_CharsHexadecimal    :: CharsHexadecimal;
    ImGuiInputTextFlags_CharsScientific     :: CharsScientific;
    ImGuiInputTextFlags_CharsUppercase      :: CharsUppercase;
    ImGuiInputTextFlags_CharsNoBlank        :: CharsNoBlank;

    ImGuiInputTextFlags_AllowTabInput       :: AllowTabInput;
    ImGuiInputTextFlags_EnterReturnsTrue    :: EnterReturnsTrue;
    ImGuiInputTextFlags_EscapeClearsAll     :: EscapeClearsAll;
    ImGuiInputTextFlags_CtrlEnterForNewLine :: CtrlEnterForNewLine;

    ImGuiInputTextFlags_ReadOnly            :: ReadOnly;
    ImGuiInputTextFlags_Password            :: Password;
    ImGuiInputTextFlags_AlwaysOverwrite     :: AlwaysOverwrite;
    ImGuiInputTextFlags_AutoSelectAll       :: AutoSelectAll;
    ImGuiInputTextFlags_ParseEmptyRefVal    :: ParseEmptyRefVal;
    ImGuiInputTextFlags_DisplayEmptyRefVal  :: DisplayEmptyRefVal;
    ImGuiInputTextFlags_NoHorizontalScroll  :: NoHorizontalScroll;
    ImGuiInputTextFlags_NoUndoRedo          :: NoUndoRedo;

    ImGuiInputTextFlags_ElideLeft           :: ElideLeft;

    ImGuiInputTextFlags_CallbackCompletion  :: CallbackCompletion;
    ImGuiInputTextFlags_CallbackHistory     :: CallbackHistory;
    ImGuiInputTextFlags_CallbackAlways      :: CallbackAlways;
    ImGuiInputTextFlags_CallbackCharFilter  :: CallbackCharFilter;
    ImGuiInputTextFlags_CallbackResize      :: CallbackResize;
    ImGuiInputTextFlags_CallbackEdit        :: CallbackEdit;

    ImGuiInputTextFlags_WordWrap            :: WordWrap;
}

// Flags for ImGui::TreeNodeEx(), ImGui::CollapsingHeader*()
ImGuiTreeNodeFlags_ :: enum s32 {
    None                 :: 0;
    Selected             :: 1;
    Framed               :: 2;
    AllowOverlap         :: 4;
    NoTreePushOnOpen     :: 8;
    NoAutoOpenOnLog      :: 16;
    DefaultOpen          :: 32;
    OpenOnDoubleClick    :: 64;
    OpenOnArrow          :: 128;
    Leaf                 :: 256;
    Bullet               :: 512;
    FramePadding         :: 1024;
    SpanAvailWidth       :: 2048;
    SpanFullWidth        :: 4096;
    SpanLabelWidth       :: 8192;
    SpanAllColumns       :: 16384;
    LabelSpanAllColumns  :: 32768;

    NavLeftJumpsToParent :: 131072;
    CollapsingHeader     :: 26;

    DrawLinesNone        :: 262144;
    DrawLinesFull        :: 524288;
    DrawLinesToNodes     :: 1048576;

    NavLeftJumpsBackHere :: 131072;
    SpanTextWidth        :: 8192;
    AllowItemOverlap     :: 4;

    ImGuiTreeNodeFlags_None                 :: None;
    ImGuiTreeNodeFlags_Selected             :: Selected;
    ImGuiTreeNodeFlags_Framed               :: Framed;
    ImGuiTreeNodeFlags_AllowOverlap         :: AllowOverlap;
    ImGuiTreeNodeFlags_NoTreePushOnOpen     :: NoTreePushOnOpen;
    ImGuiTreeNodeFlags_NoAutoOpenOnLog      :: NoAutoOpenOnLog;
    ImGuiTreeNodeFlags_DefaultOpen          :: DefaultOpen;
    ImGuiTreeNodeFlags_OpenOnDoubleClick    :: OpenOnDoubleClick;
    ImGuiTreeNodeFlags_OpenOnArrow          :: OpenOnArrow;
    ImGuiTreeNodeFlags_Leaf                 :: Leaf;
    ImGuiTreeNodeFlags_Bullet               :: Bullet;
    ImGuiTreeNodeFlags_FramePadding         :: FramePadding;
    ImGuiTreeNodeFlags_SpanAvailWidth       :: SpanAvailWidth;
    ImGuiTreeNodeFlags_SpanFullWidth        :: SpanFullWidth;
    ImGuiTreeNodeFlags_SpanLabelWidth       :: SpanLabelWidth;
    ImGuiTreeNodeFlags_SpanAllColumns       :: SpanAllColumns;
    ImGuiTreeNodeFlags_LabelSpanAllColumns  :: LabelSpanAllColumns;

    ImGuiTreeNodeFlags_NavLeftJumpsToParent :: NavLeftJumpsToParent;
    ImGuiTreeNodeFlags_CollapsingHeader     :: CollapsingHeader;

    ImGuiTreeNodeFlags_DrawLinesNone        :: DrawLinesNone;
    ImGuiTreeNodeFlags_DrawLinesFull        :: DrawLinesFull;
    ImGuiTreeNodeFlags_DrawLinesToNodes     :: DrawLinesToNodes;

    ImGuiTreeNodeFlags_NavLeftJumpsBackHere :: NavLeftJumpsBackHere;
    ImGuiTreeNodeFlags_SpanTextWidth        :: SpanTextWidth;
    ImGuiTreeNodeFlags_AllowItemOverlap     :: AllowItemOverlap;
}

// Flags for OpenPopup*(), BeginPopupContext*(), IsPopupOpen() functions.
// - To be backward compatible with older API which took an 'int mouse_button = 1' argument instead of 'ImGuiPopupFlags flags',
//   we need to treat small flags values as a mouse button index, so we encode the mouse button in the first few bits of the flags.
//   It is therefore guaranteed to be legal to pass a mouse button index in ImGuiPopupFlags.
// - For the same reason, we exceptionally default the ImGuiPopupFlags argument of BeginPopupContextXXX functions to 1 instead of 0.
//   IMPORTANT: because the default parameter is 1 (==ImGuiPopupFlags_MouseButtonRight), if you rely on the default parameter
//   and want to use another flag, you need to pass in the ImGuiPopupFlags_MouseButtonRight flag explicitly.
// - Multiple buttons currently cannot be combined/or-ed in those functions (we could allow it later).
ImGuiPopupFlags_ :: enum s32 {
    None                    :: 0;
    MouseButtonLeft         :: 0;
    MouseButtonRight        :: 1;
    MouseButtonMiddle       :: 2;
    MouseButtonMask_        :: 31;
    MouseButtonDefault_     :: 1;
    NoReopen                :: 32;

    NoOpenOverExistingPopup :: 128;
    NoOpenOverItems         :: 256;
    AnyPopupId              :: 1024;
    AnyPopupLevel           :: 2048;
    AnyPopup                :: 3072;

    ImGuiPopupFlags_None                    :: None;
    ImGuiPopupFlags_MouseButtonLeft         :: MouseButtonLeft;
    ImGuiPopupFlags_MouseButtonRight        :: MouseButtonRight;
    ImGuiPopupFlags_MouseButtonMiddle       :: MouseButtonMiddle;
    ImGuiPopupFlags_MouseButtonMask_        :: MouseButtonMask_;
    ImGuiPopupFlags_MouseButtonDefault_     :: MouseButtonDefault_;
    ImGuiPopupFlags_NoReopen                :: NoReopen;

    ImGuiPopupFlags_NoOpenOverExistingPopup :: NoOpenOverExistingPopup;
    ImGuiPopupFlags_NoOpenOverItems         :: NoOpenOverItems;
    ImGuiPopupFlags_AnyPopupId              :: AnyPopupId;
    ImGuiPopupFlags_AnyPopupLevel           :: AnyPopupLevel;
    ImGuiPopupFlags_AnyPopup                :: AnyPopup;
}

// Flags for ImGui::Selectable()
ImGuiSelectableFlags_ :: enum s32 {
    None              :: 0;
    NoAutoClosePopups :: 1;
    SpanAllColumns    :: 2;
    AllowDoubleClick  :: 4;
    Disabled          :: 8;
    AllowOverlap      :: 16;
    Highlight         :: 32;
    SelectOnNav       :: 64;

    DontClosePopups   :: 1;
    AllowItemOverlap  :: 16;

    ImGuiSelectableFlags_None              :: None;
    ImGuiSelectableFlags_NoAutoClosePopups :: NoAutoClosePopups;
    ImGuiSelectableFlags_SpanAllColumns    :: SpanAllColumns;
    ImGuiSelectableFlags_AllowDoubleClick  :: AllowDoubleClick;
    ImGuiSelectableFlags_Disabled          :: Disabled;
    ImGuiSelectableFlags_AllowOverlap      :: AllowOverlap;
    ImGuiSelectableFlags_Highlight         :: Highlight;
    ImGuiSelectableFlags_SelectOnNav       :: SelectOnNav;

    ImGuiSelectableFlags_DontClosePopups   :: DontClosePopups;
    ImGuiSelectableFlags_AllowItemOverlap  :: AllowItemOverlap;
}

// Flags for ImGui::BeginCombo()
ImGuiComboFlags_ :: enum s32 {
    None            :: 0;
    PopupAlignLeft  :: 1;
    HeightSmall     :: 2;
    HeightRegular   :: 4;
    HeightLarge     :: 8;
    HeightLargest   :: 16;
    NoArrowButton   :: 32;
    NoPreview       :: 64;
    WidthFitPreview :: 128;
    HeightMask_     :: 30;

    ImGuiComboFlags_None            :: None;
    ImGuiComboFlags_PopupAlignLeft  :: PopupAlignLeft;
    ImGuiComboFlags_HeightSmall     :: HeightSmall;
    ImGuiComboFlags_HeightRegular   :: HeightRegular;
    ImGuiComboFlags_HeightLarge     :: HeightLarge;
    ImGuiComboFlags_HeightLargest   :: HeightLargest;
    ImGuiComboFlags_NoArrowButton   :: NoArrowButton;
    ImGuiComboFlags_NoPreview       :: NoPreview;
    ImGuiComboFlags_WidthFitPreview :: WidthFitPreview;
    ImGuiComboFlags_HeightMask_     :: HeightMask_;
}

// Flags for ImGui::BeginTabBar()
ImGuiTabBarFlags_ :: enum s32 {
    None                         :: 0;
    Reorderable                  :: 1;
    AutoSelectNewTabs            :: 2;
    TabListPopupButton           :: 4;
    NoCloseWithMiddleMouseButton :: 8;
    NoTabListScrollingButtons    :: 16;
    NoTooltip                    :: 32;
    DrawSelectedOverline         :: 64;

    FittingPolicyMixed           :: 128;
    FittingPolicyShrink          :: 256;
    FittingPolicyScroll          :: 512;
    FittingPolicyMask_           :: 896;
    FittingPolicyDefault_        :: 128;

    FittingPolicyResizeDown      :: 256;

    ImGuiTabBarFlags_None                         :: None;
    ImGuiTabBarFlags_Reorderable                  :: Reorderable;
    ImGuiTabBarFlags_AutoSelectNewTabs            :: AutoSelectNewTabs;
    ImGuiTabBarFlags_TabListPopupButton           :: TabListPopupButton;
    ImGuiTabBarFlags_NoCloseWithMiddleMouseButton :: NoCloseWithMiddleMouseButton;
    ImGuiTabBarFlags_NoTabListScrollingButtons    :: NoTabListScrollingButtons;
    ImGuiTabBarFlags_NoTooltip                    :: NoTooltip;
    ImGuiTabBarFlags_DrawSelectedOverline         :: DrawSelectedOverline;

    ImGuiTabBarFlags_FittingPolicyMixed           :: FittingPolicyMixed;
    ImGuiTabBarFlags_FittingPolicyShrink          :: FittingPolicyShrink;
    ImGuiTabBarFlags_FittingPolicyScroll          :: FittingPolicyScroll;
    ImGuiTabBarFlags_FittingPolicyMask_           :: FittingPolicyMask_;
    ImGuiTabBarFlags_FittingPolicyDefault_        :: FittingPolicyDefault_;

    ImGuiTabBarFlags_FittingPolicyResizeDown      :: FittingPolicyResizeDown;
}

// Flags for ImGui::BeginTabItem()
ImGuiTabItemFlags_ :: enum s32 {
    None                         :: 0;
    UnsavedDocument              :: 1;
    SetSelected                  :: 2;
    NoCloseWithMiddleMouseButton :: 4;
    NoPushId                     :: 8;
    NoTooltip                    :: 16;
    NoReorder                    :: 32;
    Leading                      :: 64;
    Trailing                     :: 128;
    NoAssumedClosure             :: 256;

    ImGuiTabItemFlags_None                         :: None;
    ImGuiTabItemFlags_UnsavedDocument              :: UnsavedDocument;
    ImGuiTabItemFlags_SetSelected                  :: SetSelected;
    ImGuiTabItemFlags_NoCloseWithMiddleMouseButton :: NoCloseWithMiddleMouseButton;
    ImGuiTabItemFlags_NoPushId                     :: NoPushId;
    ImGuiTabItemFlags_NoTooltip                    :: NoTooltip;
    ImGuiTabItemFlags_NoReorder                    :: NoReorder;
    ImGuiTabItemFlags_Leading                      :: Leading;
    ImGuiTabItemFlags_Trailing                     :: Trailing;
    ImGuiTabItemFlags_NoAssumedClosure             :: NoAssumedClosure;
}

// Flags for ImGui::IsWindowFocused()
ImGuiFocusedFlags_ :: enum s32 {
    None                :: 0;
    ChildWindows        :: 1;
    RootWindow          :: 2;
    AnyWindow           :: 4;
    NoPopupHierarchy    :: 8;

    RootAndChildWindows :: 3;

    ImGuiFocusedFlags_None                :: None;
    ImGuiFocusedFlags_ChildWindows        :: ChildWindows;
    ImGuiFocusedFlags_RootWindow          :: RootWindow;
    ImGuiFocusedFlags_AnyWindow           :: AnyWindow;
    ImGuiFocusedFlags_NoPopupHierarchy    :: NoPopupHierarchy;

    ImGuiFocusedFlags_RootAndChildWindows :: RootAndChildWindows;
}

// Flags for ImGui::IsItemHovered(), ImGui::IsWindowHovered()
// Note: if you are trying to check whether your mouse should be dispatched to Dear ImGui or to your app, you should use 'io.WantCaptureMouse' instead! Please read the FAQ!
// Note: windows with the ImGuiWindowFlags_NoInputs flag are ignored by IsWindowHovered() calls.
ImGuiHoveredFlags_ :: enum s32 {
    None                         :: 0;
    ChildWindows                 :: 1;
    RootWindow                   :: 2;
    AnyWindow                    :: 4;
    NoPopupHierarchy             :: 8;

    AllowWhenBlockedByPopup      :: 32;

    AllowWhenBlockedByActiveItem :: 128;
    AllowWhenOverlappedByItem    :: 256;
    AllowWhenOverlappedByWindow  :: 512;
    AllowWhenDisabled            :: 1024;
    NoNavOverride                :: 2048;
    AllowWhenOverlapped          :: 768;
    RectOnly                     :: 928;
    RootAndChildWindows          :: 3;

    ForTooltip                   :: 4096;

    Stationary                   :: 8192;
    DelayNone                    :: 16384;
    DelayShort                   :: 32768;
    DelayNormal                  :: 65536;
    NoSharedDelay                :: 131072;

    ImGuiHoveredFlags_None                         :: None;
    ImGuiHoveredFlags_ChildWindows                 :: ChildWindows;
    ImGuiHoveredFlags_RootWindow                   :: RootWindow;
    ImGuiHoveredFlags_AnyWindow                    :: AnyWindow;
    ImGuiHoveredFlags_NoPopupHierarchy             :: NoPopupHierarchy;

    ImGuiHoveredFlags_AllowWhenBlockedByPopup      :: AllowWhenBlockedByPopup;

    ImGuiHoveredFlags_AllowWhenBlockedByActiveItem :: AllowWhenBlockedByActiveItem;
    ImGuiHoveredFlags_AllowWhenOverlappedByItem    :: AllowWhenOverlappedByItem;
    ImGuiHoveredFlags_AllowWhenOverlappedByWindow  :: AllowWhenOverlappedByWindow;
    ImGuiHoveredFlags_AllowWhenDisabled            :: AllowWhenDisabled;
    ImGuiHoveredFlags_NoNavOverride                :: NoNavOverride;
    ImGuiHoveredFlags_AllowWhenOverlapped          :: AllowWhenOverlapped;
    ImGuiHoveredFlags_RectOnly                     :: RectOnly;
    ImGuiHoveredFlags_RootAndChildWindows          :: RootAndChildWindows;

    ImGuiHoveredFlags_ForTooltip                   :: ForTooltip;

    ImGuiHoveredFlags_Stationary                   :: Stationary;
    ImGuiHoveredFlags_DelayNone                    :: DelayNone;
    ImGuiHoveredFlags_DelayShort                   :: DelayShort;
    ImGuiHoveredFlags_DelayNormal                  :: DelayNormal;
    ImGuiHoveredFlags_NoSharedDelay                :: NoSharedDelay;
}

// Flags for ImGui::BeginDragDropSource(), ImGui::AcceptDragDropPayload()
ImGuiDragDropFlags_ :: enum s32 {
    None                     :: 0;

    SourceNoPreviewTooltip   :: 1;
    SourceNoDisableHover     :: 2;
    SourceNoHoldToOpenOthers :: 4;
    SourceAllowNullID        :: 8;
    SourceExtern             :: 16;
    PayloadAutoExpire        :: 32;
    PayloadNoCrossContext    :: 64;
    PayloadNoCrossProcess    :: 128;

    AcceptBeforeDelivery     :: 1024;
    AcceptNoDrawDefaultRect  :: 2048;
    AcceptNoPreviewTooltip   :: 4096;
    AcceptPeekOnly           :: 3072;

    SourceAutoExpirePayload  :: 32;

    ImGuiDragDropFlags_None                     :: None;

    ImGuiDragDropFlags_SourceNoPreviewTooltip   :: SourceNoPreviewTooltip;
    ImGuiDragDropFlags_SourceNoDisableHover     :: SourceNoDisableHover;
    ImGuiDragDropFlags_SourceNoHoldToOpenOthers :: SourceNoHoldToOpenOthers;
    ImGuiDragDropFlags_SourceAllowNullID        :: SourceAllowNullID;
    ImGuiDragDropFlags_SourceExtern             :: SourceExtern;
    ImGuiDragDropFlags_PayloadAutoExpire        :: PayloadAutoExpire;
    ImGuiDragDropFlags_PayloadNoCrossContext    :: PayloadNoCrossContext;
    ImGuiDragDropFlags_PayloadNoCrossProcess    :: PayloadNoCrossProcess;

    ImGuiDragDropFlags_AcceptBeforeDelivery     :: AcceptBeforeDelivery;
    ImGuiDragDropFlags_AcceptNoDrawDefaultRect  :: AcceptNoDrawDefaultRect;
    ImGuiDragDropFlags_AcceptNoPreviewTooltip   :: AcceptNoPreviewTooltip;
    ImGuiDragDropFlags_AcceptPeekOnly           :: AcceptPeekOnly;

    ImGuiDragDropFlags_SourceAutoExpirePayload  :: SourceAutoExpirePayload;
}

// A primary data type
ImGuiDataType_ :: enum s32 {
    S8     :: 0;
    U8     :: 1;
    S16    :: 2;
    U16    :: 3;
    S32    :: 4;
    U32    :: 5;
    S64    :: 6;
    U64    :: 7;
    Float  :: 8;
    Double :: 9;
    Bool   :: 10;
    String :: 11;
    COUNT  :: 12;

    ImGuiDataType_S8     :: S8;
    ImGuiDataType_U8     :: U8;
    ImGuiDataType_S16    :: S16;
    ImGuiDataType_U16    :: U16;
    ImGuiDataType_S32    :: S32;
    ImGuiDataType_U32    :: U32;
    ImGuiDataType_S64    :: S64;
    ImGuiDataType_U64    :: U64;
    ImGuiDataType_Float  :: Float;
    ImGuiDataType_Double :: Double;
    ImGuiDataType_Bool   :: Bool;
    ImGuiDataType_String :: String;
    ImGuiDataType_COUNT  :: COUNT;
}

// Flags for Shortcut(), SetNextItemShortcut(),
// (and for upcoming extended versions of IsKeyPressed(), IsMouseClicked(), Shortcut(), SetKeyOwner(), SetItemKeyOwner() that are still in imgui_internal.h)
// Don't mistake with ImGuiInputTextFlags! (which is for ImGui::InputText() function)
ImGuiInputFlags_ :: enum s32 {
    None                 :: 0;
    Repeat               :: 1;

    RouteActive          :: 1024;
    RouteFocused         :: 2048;
    RouteGlobal          :: 4096;
    RouteAlways          :: 8192;

    RouteOverFocused     :: 16384;
    RouteOverActive      :: 32768;
    RouteUnlessBgFocused :: 65536;
    RouteFromRootWindow  :: 131072;

    Tooltip              :: 262144;

    ImGuiInputFlags_None                 :: None;
    ImGuiInputFlags_Repeat               :: Repeat;

    ImGuiInputFlags_RouteActive          :: RouteActive;
    ImGuiInputFlags_RouteFocused         :: RouteFocused;
    ImGuiInputFlags_RouteGlobal          :: RouteGlobal;
    ImGuiInputFlags_RouteAlways          :: RouteAlways;

    ImGuiInputFlags_RouteOverFocused     :: RouteOverFocused;
    ImGuiInputFlags_RouteOverActive      :: RouteOverActive;
    ImGuiInputFlags_RouteUnlessBgFocused :: RouteUnlessBgFocused;
    ImGuiInputFlags_RouteFromRootWindow  :: RouteFromRootWindow;

    ImGuiInputFlags_Tooltip              :: Tooltip;
}

// Configuration flags stored in io.ConfigFlags. Set by user/application.
ImGuiConfigFlags_ :: enum s32 {
    None                 :: 0;
    NavEnableKeyboard    :: 1;
    NavEnableGamepad     :: 2;
    NoMouse              :: 16;
    NoMouseCursorChange  :: 32;
    NoKeyboard           :: 64;

    IsSRGB               :: 1048576;
    IsTouchScreen        :: 2097152;

    NavEnableSetMousePos :: 4;
    NavNoCaptureKeyboard :: 8;

    ImGuiConfigFlags_None                 :: None;
    ImGuiConfigFlags_NavEnableKeyboard    :: NavEnableKeyboard;
    ImGuiConfigFlags_NavEnableGamepad     :: NavEnableGamepad;
    ImGuiConfigFlags_NoMouse              :: NoMouse;
    ImGuiConfigFlags_NoMouseCursorChange  :: NoMouseCursorChange;
    ImGuiConfigFlags_NoKeyboard           :: NoKeyboard;

    ImGuiConfigFlags_IsSRGB               :: IsSRGB;
    ImGuiConfigFlags_IsTouchScreen        :: IsTouchScreen;

    ImGuiConfigFlags_NavEnableSetMousePos :: NavEnableSetMousePos;
    ImGuiConfigFlags_NavNoCaptureKeyboard :: NavNoCaptureKeyboard;
}

// Backend capabilities flags stored in io.BackendFlags. Set by imgui_impl_xxx or custom backend.
ImGuiBackendFlags_ :: enum s32 {
    None                 :: 0;
    HasGamepad           :: 1;
    HasMouseCursors      :: 2;
    HasSetMousePos       :: 4;
    RendererHasVtxOffset :: 8;
    RendererHasTextures  :: 16;

    ImGuiBackendFlags_None                 :: None;
    ImGuiBackendFlags_HasGamepad           :: HasGamepad;
    ImGuiBackendFlags_HasMouseCursors      :: HasMouseCursors;
    ImGuiBackendFlags_HasSetMousePos       :: HasSetMousePos;
    ImGuiBackendFlags_RendererHasVtxOffset :: RendererHasVtxOffset;
    ImGuiBackendFlags_RendererHasTextures  :: RendererHasTextures;
}

// Enumeration for PushStyleColor() / PopStyleColor()
ImGuiCol_ :: enum s32 {
    Text                      :: 0;
    TextDisabled              :: 1;
    WindowBg                  :: 2;
    ChildBg                   :: 3;
    PopupBg                   :: 4;
    Border                    :: 5;
    BorderShadow              :: 6;
    FrameBg                   :: 7;
    FrameBgHovered            :: 8;
    FrameBgActive             :: 9;
    TitleBg                   :: 10;
    TitleBgActive             :: 11;
    TitleBgCollapsed          :: 12;
    MenuBarBg                 :: 13;
    ScrollbarBg               :: 14;
    ScrollbarGrab             :: 15;
    ScrollbarGrabHovered      :: 16;
    ScrollbarGrabActive       :: 17;
    CheckMark                 :: 18;
    SliderGrab                :: 19;
    SliderGrabActive          :: 20;
    Button                    :: 21;
    ButtonHovered             :: 22;
    ButtonActive              :: 23;
    Header                    :: 24;
    HeaderHovered             :: 25;
    HeaderActive              :: 26;
    Separator                 :: 27;
    SeparatorHovered          :: 28;
    SeparatorActive           :: 29;
    ResizeGrip                :: 30;
    ResizeGripHovered         :: 31;
    ResizeGripActive          :: 32;
    InputTextCursor           :: 33;
    TabHovered                :: 34;
    Tab                       :: 35;
    TabSelected               :: 36;
    TabSelectedOverline       :: 37;
    TabDimmed                 :: 38;
    TabDimmedSelected         :: 39;
    TabDimmedSelectedOverline :: 40;
    PlotLines                 :: 41;
    PlotLinesHovered          :: 42;
    PlotHistogram             :: 43;
    PlotHistogramHovered      :: 44;
    TableHeaderBg             :: 45;
    TableBorderStrong         :: 46;
    TableBorderLight          :: 47;
    TableRowBg                :: 48;
    TableRowBgAlt             :: 49;
    TextLink                  :: 50;
    TextSelectedBg            :: 51;
    TreeLines                 :: 52;
    DragDropTarget            :: 53;
    NavCursor                 :: 54;
    NavWindowingHighlight     :: 55;
    NavWindowingDimBg         :: 56;
    ModalWindowDimBg          :: 57;
    COUNT                     :: 58;

    TabActive                 :: 36;
    TabUnfocused              :: 38;
    TabUnfocusedActive        :: 39;
    NavHighlight              :: 54;

    ImGuiCol_Text                      :: Text;
    ImGuiCol_TextDisabled              :: TextDisabled;
    ImGuiCol_WindowBg                  :: WindowBg;
    ImGuiCol_ChildBg                   :: ChildBg;
    ImGuiCol_PopupBg                   :: PopupBg;
    ImGuiCol_Border                    :: Border;
    ImGuiCol_BorderShadow              :: BorderShadow;
    ImGuiCol_FrameBg                   :: FrameBg;
    ImGuiCol_FrameBgHovered            :: FrameBgHovered;
    ImGuiCol_FrameBgActive             :: FrameBgActive;
    ImGuiCol_TitleBg                   :: TitleBg;
    ImGuiCol_TitleBgActive             :: TitleBgActive;
    ImGuiCol_TitleBgCollapsed          :: TitleBgCollapsed;
    ImGuiCol_MenuBarBg                 :: MenuBarBg;
    ImGuiCol_ScrollbarBg               :: ScrollbarBg;
    ImGuiCol_ScrollbarGrab             :: ScrollbarGrab;
    ImGuiCol_ScrollbarGrabHovered      :: ScrollbarGrabHovered;
    ImGuiCol_ScrollbarGrabActive       :: ScrollbarGrabActive;
    ImGuiCol_CheckMark                 :: CheckMark;
    ImGuiCol_SliderGrab                :: SliderGrab;
    ImGuiCol_SliderGrabActive          :: SliderGrabActive;
    ImGuiCol_Button                    :: Button;
    ImGuiCol_ButtonHovered             :: ButtonHovered;
    ImGuiCol_ButtonActive              :: ButtonActive;
    ImGuiCol_Header                    :: Header;
    ImGuiCol_HeaderHovered             :: HeaderHovered;
    ImGuiCol_HeaderActive              :: HeaderActive;
    ImGuiCol_Separator                 :: Separator;
    ImGuiCol_SeparatorHovered          :: SeparatorHovered;
    ImGuiCol_SeparatorActive           :: SeparatorActive;
    ImGuiCol_ResizeGrip                :: ResizeGrip;
    ImGuiCol_ResizeGripHovered         :: ResizeGripHovered;
    ImGuiCol_ResizeGripActive          :: ResizeGripActive;
    ImGuiCol_InputTextCursor           :: InputTextCursor;
    ImGuiCol_TabHovered                :: TabHovered;
    ImGuiCol_Tab                       :: Tab;
    ImGuiCol_TabSelected               :: TabSelected;
    ImGuiCol_TabSelectedOverline       :: TabSelectedOverline;
    ImGuiCol_TabDimmed                 :: TabDimmed;
    ImGuiCol_TabDimmedSelected         :: TabDimmedSelected;
    ImGuiCol_TabDimmedSelectedOverline :: TabDimmedSelectedOverline;
    ImGuiCol_PlotLines                 :: PlotLines;
    ImGuiCol_PlotLinesHovered          :: PlotLinesHovered;
    ImGuiCol_PlotHistogram             :: PlotHistogram;
    ImGuiCol_PlotHistogramHovered      :: PlotHistogramHovered;
    ImGuiCol_TableHeaderBg             :: TableHeaderBg;
    ImGuiCol_TableBorderStrong         :: TableBorderStrong;
    ImGuiCol_TableBorderLight          :: TableBorderLight;
    ImGuiCol_TableRowBg                :: TableRowBg;
    ImGuiCol_TableRowBgAlt             :: TableRowBgAlt;
    ImGuiCol_TextLink                  :: TextLink;
    ImGuiCol_TextSelectedBg            :: TextSelectedBg;
    ImGuiCol_TreeLines                 :: TreeLines;
    ImGuiCol_DragDropTarget            :: DragDropTarget;
    ImGuiCol_NavCursor                 :: NavCursor;
    ImGuiCol_NavWindowingHighlight     :: NavWindowingHighlight;
    ImGuiCol_NavWindowingDimBg         :: NavWindowingDimBg;
    ImGuiCol_ModalWindowDimBg          :: ModalWindowDimBg;
    ImGuiCol_COUNT                     :: COUNT;

    ImGuiCol_TabActive                 :: TabActive;
    ImGuiCol_TabUnfocused              :: TabUnfocused;
    ImGuiCol_TabUnfocusedActive        :: TabUnfocusedActive;
    ImGuiCol_NavHighlight              :: NavHighlight;
}

// Enumeration for PushStyleVar() / PopStyleVar() to temporarily modify the ImGuiStyle structure.
// - The enum only refers to fields of ImGuiStyle which makes sense to be pushed/popped inside UI code.
//   During initialization or between frames, feel free to just poke into ImGuiStyle directly.
// - Tip: Use your programming IDE navigation facilities on the names in the _second column_ below to find the actual members and their description.
//   - In Visual Studio: CTRL+comma ("Edit.GoToAll") can follow symbols inside comments, whereas CTRL+F12 ("Edit.GoToImplementation") cannot.
//   - In Visual Studio w/ Visual Assist installed: ALT+G ("VAssistX.GoToImplementation") can also follow symbols inside comments.
//   - In VS Code, CLion, etc.: CTRL+click can follow symbols inside comments.
// - When changing this enum, you need to update the associated internal table GStyleVarInfo[] accordingly. This is where we link enum values to members offset/type.
ImGuiStyleVar_ :: enum s32 {
    Alpha                       :: 0;
    DisabledAlpha               :: 1;
    WindowPadding               :: 2;
    WindowRounding              :: 3;
    WindowBorderSize            :: 4;
    WindowMinSize               :: 5;
    WindowTitleAlign            :: 6;
    ChildRounding               :: 7;
    ChildBorderSize             :: 8;
    PopupRounding               :: 9;
    PopupBorderSize             :: 10;
    FramePadding                :: 11;
    FrameRounding               :: 12;
    FrameBorderSize             :: 13;
    ItemSpacing                 :: 14;
    ItemInnerSpacing            :: 15;
    IndentSpacing               :: 16;
    CellPadding                 :: 17;
    ScrollbarSize               :: 18;
    ScrollbarRounding           :: 19;
    ScrollbarPadding            :: 20;
    GrabMinSize                 :: 21;
    GrabRounding                :: 22;
    ImageBorderSize             :: 23;
    TabRounding                 :: 24;
    TabBorderSize               :: 25;
    TabMinWidthBase             :: 26;
    TabMinWidthShrink           :: 27;
    TabBarBorderSize            :: 28;
    TabBarOverlineSize          :: 29;
    TableAngledHeadersAngle     :: 30;
    TableAngledHeadersTextAlign :: 31;
    TreeLinesSize               :: 32;
    TreeLinesRounding           :: 33;
    ButtonTextAlign             :: 34;
    SelectableTextAlign         :: 35;
    SeparatorTextBorderSize     :: 36;
    SeparatorTextAlign          :: 37;
    SeparatorTextPadding        :: 38;
    COUNT                       :: 39;

    ImGuiStyleVar_Alpha                       :: Alpha;
    ImGuiStyleVar_DisabledAlpha               :: DisabledAlpha;
    ImGuiStyleVar_WindowPadding               :: WindowPadding;
    ImGuiStyleVar_WindowRounding              :: WindowRounding;
    ImGuiStyleVar_WindowBorderSize            :: WindowBorderSize;
    ImGuiStyleVar_WindowMinSize               :: WindowMinSize;
    ImGuiStyleVar_WindowTitleAlign            :: WindowTitleAlign;
    ImGuiStyleVar_ChildRounding               :: ChildRounding;
    ImGuiStyleVar_ChildBorderSize             :: ChildBorderSize;
    ImGuiStyleVar_PopupRounding               :: PopupRounding;
    ImGuiStyleVar_PopupBorderSize             :: PopupBorderSize;
    ImGuiStyleVar_FramePadding                :: FramePadding;
    ImGuiStyleVar_FrameRounding               :: FrameRounding;
    ImGuiStyleVar_FrameBorderSize             :: FrameBorderSize;
    ImGuiStyleVar_ItemSpacing                 :: ItemSpacing;
    ImGuiStyleVar_ItemInnerSpacing            :: ItemInnerSpacing;
    ImGuiStyleVar_IndentSpacing               :: IndentSpacing;
    ImGuiStyleVar_CellPadding                 :: CellPadding;
    ImGuiStyleVar_ScrollbarSize               :: ScrollbarSize;
    ImGuiStyleVar_ScrollbarRounding           :: ScrollbarRounding;
    ImGuiStyleVar_ScrollbarPadding            :: ScrollbarPadding;
    ImGuiStyleVar_GrabMinSize                 :: GrabMinSize;
    ImGuiStyleVar_GrabRounding                :: GrabRounding;
    ImGuiStyleVar_ImageBorderSize             :: ImageBorderSize;
    ImGuiStyleVar_TabRounding                 :: TabRounding;
    ImGuiStyleVar_TabBorderSize               :: TabBorderSize;
    ImGuiStyleVar_TabMinWidthBase             :: TabMinWidthBase;
    ImGuiStyleVar_TabMinWidthShrink           :: TabMinWidthShrink;
    ImGuiStyleVar_TabBarBorderSize            :: TabBarBorderSize;
    ImGuiStyleVar_TabBarOverlineSize          :: TabBarOverlineSize;
    ImGuiStyleVar_TableAngledHeadersAngle     :: TableAngledHeadersAngle;
    ImGuiStyleVar_TableAngledHeadersTextAlign :: TableAngledHeadersTextAlign;
    ImGuiStyleVar_TreeLinesSize               :: TreeLinesSize;
    ImGuiStyleVar_TreeLinesRounding           :: TreeLinesRounding;
    ImGuiStyleVar_ButtonTextAlign             :: ButtonTextAlign;
    ImGuiStyleVar_SelectableTextAlign         :: SelectableTextAlign;
    ImGuiStyleVar_SeparatorTextBorderSize     :: SeparatorTextBorderSize;
    ImGuiStyleVar_SeparatorTextAlign          :: SeparatorTextAlign;
    ImGuiStyleVar_SeparatorTextPadding        :: SeparatorTextPadding;
    ImGuiStyleVar_COUNT                       :: COUNT;
}

// Flags for InvisibleButton() [extended in imgui_internal.h]
ImGuiButtonFlags_ :: enum s32 {
    None              :: 0;
    MouseButtonLeft   :: 1;
    MouseButtonRight  :: 2;
    MouseButtonMiddle :: 4;
    MouseButtonMask_  :: 7;
    EnableNav         :: 8;

    ImGuiButtonFlags_None              :: None;
    ImGuiButtonFlags_MouseButtonLeft   :: MouseButtonLeft;
    ImGuiButtonFlags_MouseButtonRight  :: MouseButtonRight;
    ImGuiButtonFlags_MouseButtonMiddle :: MouseButtonMiddle;
    ImGuiButtonFlags_MouseButtonMask_  :: MouseButtonMask_;
    ImGuiButtonFlags_EnableNav         :: EnableNav;
}

// Flags for ColorEdit3() / ColorEdit4() / ColorPicker3() / ColorPicker4() / ColorButton()
ImGuiColorEditFlags_ :: enum s32 {
    None             :: 0;
    NoAlpha          :: 2;
    NoPicker         :: 4;
    NoOptions        :: 8;
    NoSmallPreview   :: 16;
    NoInputs         :: 32;
    NoTooltip        :: 64;
    NoLabel          :: 128;
    NoSidePreview    :: 256;
    NoDragDrop       :: 512;
    NoBorder         :: 1024;

    AlphaOpaque      :: 2048;
    AlphaNoBg        :: 4096;
    AlphaPreviewHalf :: 8192;

    AlphaBar         :: 65536;
    HDR              :: 524288;
    DisplayRGB       :: 1048576;
    DisplayHSV       :: 2097152;
    DisplayHex       :: 4194304;
    Uint8            :: 8388608;
    Float            :: 16777216;
    PickerHueBar     :: 33554432;
    PickerHueWheel   :: 67108864;
    InputRGB         :: 134217728;
    InputHSV         :: 268435456;

    DefaultOptions_  :: 177209344;

    AlphaMask_       :: 14338;
    DisplayMask_     :: 7340032;
    DataTypeMask_    :: 25165824;
    PickerMask_      :: 100663296;
    InputMask_       :: 402653184;

    AlphaPreview     :: 0;

    ImGuiColorEditFlags_None             :: None;
    ImGuiColorEditFlags_NoAlpha          :: NoAlpha;
    ImGuiColorEditFlags_NoPicker         :: NoPicker;
    ImGuiColorEditFlags_NoOptions        :: NoOptions;
    ImGuiColorEditFlags_NoSmallPreview   :: NoSmallPreview;
    ImGuiColorEditFlags_NoInputs         :: NoInputs;
    ImGuiColorEditFlags_NoTooltip        :: NoTooltip;
    ImGuiColorEditFlags_NoLabel          :: NoLabel;
    ImGuiColorEditFlags_NoSidePreview    :: NoSidePreview;
    ImGuiColorEditFlags_NoDragDrop       :: NoDragDrop;
    ImGuiColorEditFlags_NoBorder         :: NoBorder;

    ImGuiColorEditFlags_AlphaOpaque      :: AlphaOpaque;
    ImGuiColorEditFlags_AlphaNoBg        :: AlphaNoBg;
    ImGuiColorEditFlags_AlphaPreviewHalf :: AlphaPreviewHalf;

    ImGuiColorEditFlags_AlphaBar         :: AlphaBar;
    ImGuiColorEditFlags_HDR              :: HDR;
    ImGuiColorEditFlags_DisplayRGB       :: DisplayRGB;
    ImGuiColorEditFlags_DisplayHSV       :: DisplayHSV;
    ImGuiColorEditFlags_DisplayHex       :: DisplayHex;
    ImGuiColorEditFlags_Uint8            :: Uint8;
    ImGuiColorEditFlags_Float            :: Float;
    ImGuiColorEditFlags_PickerHueBar     :: PickerHueBar;
    ImGuiColorEditFlags_PickerHueWheel   :: PickerHueWheel;
    ImGuiColorEditFlags_InputRGB         :: InputRGB;
    ImGuiColorEditFlags_InputHSV         :: InputHSV;

    ImGuiColorEditFlags_DefaultOptions_  :: DefaultOptions_;

    ImGuiColorEditFlags_AlphaMask_       :: AlphaMask_;
    ImGuiColorEditFlags_DisplayMask_     :: DisplayMask_;
    ImGuiColorEditFlags_DataTypeMask_    :: DataTypeMask_;
    ImGuiColorEditFlags_PickerMask_      :: PickerMask_;
    ImGuiColorEditFlags_InputMask_       :: InputMask_;

    ImGuiColorEditFlags_AlphaPreview     :: AlphaPreview;
}

// Flags for DragFloat(), DragInt(), SliderFloat(), SliderInt() etc.
// We use the same sets of flags for DragXXX() and SliderXXX() functions as the features are the same and it makes it easier to swap them.
// (Those are per-item flags. There is shared behavior flag too: ImGuiIO: io.ConfigDragClickToInputText)
ImGuiSliderFlags_ :: enum s32 {
    None            :: 0;
    Logarithmic     :: 32;
    NoRoundToFormat :: 64;
    NoInput         :: 128;
    WrapAround      :: 256;
    ClampOnInput    :: 512;
    ClampZeroRange  :: 1024;
    NoSpeedTweaks   :: 2048;
    AlwaysClamp     :: 1536;
    InvalidMask_    :: 1879048207;

    ImGuiSliderFlags_None            :: None;
    ImGuiSliderFlags_Logarithmic     :: Logarithmic;
    ImGuiSliderFlags_NoRoundToFormat :: NoRoundToFormat;
    ImGuiSliderFlags_NoInput         :: NoInput;
    ImGuiSliderFlags_WrapAround      :: WrapAround;
    ImGuiSliderFlags_ClampOnInput    :: ClampOnInput;
    ImGuiSliderFlags_ClampZeroRange  :: ClampZeroRange;
    ImGuiSliderFlags_NoSpeedTweaks   :: NoSpeedTweaks;
    ImGuiSliderFlags_AlwaysClamp     :: AlwaysClamp;
    ImGuiSliderFlags_InvalidMask_    :: InvalidMask_;
}

// Identify a mouse button.
// Those values are guaranteed to be stable and we frequently use 0/1 directly. Named enums provided for convenience.
ImGuiMouseButton_ :: enum s32 {
    Left   :: 0;
    Right  :: 1;
    Middle :: 2;
    COUNT  :: 5;

    ImGuiMouseButton_Left   :: Left;
    ImGuiMouseButton_Right  :: Right;
    ImGuiMouseButton_Middle :: Middle;
    ImGuiMouseButton_COUNT  :: COUNT;
}

// Enumeration for GetMouseCursor()
// User code may request backend to display given cursor by calling SetMouseCursor(), which is why we have some cursors that are marked unused here
ImGuiMouseCursor_ :: enum s32 {
    None       :: -1;
    Arrow      :: 0;
    TextInput  :: 1;
    ResizeAll  :: 2;
    ResizeNS   :: 3;
    ResizeEW   :: 4;
    ResizeNESW :: 5;
    ResizeNWSE :: 6;
    Hand       :: 7;
    Wait       :: 8;
    Progress   :: 9;
    NotAllowed :: 10;
    COUNT      :: 11;

    ImGuiMouseCursor_None       :: None;
    ImGuiMouseCursor_Arrow      :: Arrow;
    ImGuiMouseCursor_TextInput  :: TextInput;
    ImGuiMouseCursor_ResizeAll  :: ResizeAll;
    ImGuiMouseCursor_ResizeNS   :: ResizeNS;
    ImGuiMouseCursor_ResizeEW   :: ResizeEW;
    ImGuiMouseCursor_ResizeNESW :: ResizeNESW;
    ImGuiMouseCursor_ResizeNWSE :: ResizeNWSE;
    ImGuiMouseCursor_Hand       :: Hand;
    ImGuiMouseCursor_Wait       :: Wait;
    ImGuiMouseCursor_Progress   :: Progress;
    ImGuiMouseCursor_NotAllowed :: NotAllowed;
    ImGuiMouseCursor_COUNT      :: COUNT;
}

// Enumeration for ImGui::SetNextWindow***(), SetWindow***(), SetNextItem***() functions
// Represent a condition.
// Important: Treat as a regular enum! Do NOT combine multiple values using binary operators! All the functions above treat 0 as a shortcut to ImGuiCond_Always.
ImGuiCond_ :: enum s32 {
    None         :: 0;
    Always       :: 1;
    Once         :: 2;
    FirstUseEver :: 4;
    Appearing    :: 8;

    ImGuiCond_None         :: None;
    ImGuiCond_Always       :: Always;
    ImGuiCond_Once         :: Once;
    ImGuiCond_FirstUseEver :: FirstUseEver;
    ImGuiCond_Appearing    :: Appearing;
}

// Flags for ImGui::BeginTable()
// - Important! Sizing policies have complex and subtle side effects, much more so than you would expect.
//   Read comments/demos carefully + experiment with live demos to get acquainted with them.
// - The DEFAULT sizing policies are:
//    - Default to ImGuiTableFlags_SizingFixedFit    if ScrollX is on, or if host window has ImGuiWindowFlags_AlwaysAutoResize.
//    - Default to ImGuiTableFlags_SizingStretchSame if ScrollX is off.
// - When ScrollX is off:
//    - Table defaults to ImGuiTableFlags_SizingStretchSame -> all Columns defaults to ImGuiTableColumnFlags_WidthStretch with same weight.
//    - Columns sizing policy allowed: Stretch (default), Fixed/Auto.
//    - Fixed Columns (if any) will generally obtain their requested width (unless the table cannot fit them all).
//    - Stretch Columns will share the remaining width according to their respective weight.
//    - Mixed Fixed/Stretch columns is possible but has various side-effects on resizing behaviors.
//      The typical use of mixing sizing policies is: any number of LEADING Fixed columns, followed by one or two TRAILING Stretch columns.
//      (this is because the visible order of columns have subtle but necessary effects on how they react to manual resizing).
// - When ScrollX is on:
//    - Table defaults to ImGuiTableFlags_SizingFixedFit -> all Columns defaults to ImGuiTableColumnFlags_WidthFixed
//    - Columns sizing policy allowed: Fixed/Auto mostly.
//    - Fixed Columns can be enlarged as needed. Table will show a horizontal scrollbar if needed.
//    - When using auto-resizing (non-resizable) fixed columns, querying the content width to use item right-alignment e.g. SetNextItemWidth(-FLT_MIN) doesn't make sense, would create a feedback loop.
//    - Using Stretch columns OFTEN DOES NOT MAKE SENSE if ScrollX is on, UNLESS you have specified a value for 'inner_width' in BeginTable().
//      If you specify a value for 'inner_width' then effectively the scrolling space is known and Stretch or mixed Fixed/Stretch columns become meaningful again.
// - Read on documentation at the top of imgui_tables.cpp for details.
ImGuiTableFlags_ :: enum s32 {
    None                       :: 0;
    Resizable                  :: 1;
    Reorderable                :: 2;
    Hideable                   :: 4;
    Sortable                   :: 8;
    NoSavedSettings            :: 16;
    ContextMenuInBody          :: 32;

    RowBg                      :: 64;
    BordersInnerH              :: 128;
    BordersOuterH              :: 256;
    BordersInnerV              :: 512;
    BordersOuterV              :: 1024;
    BordersH                   :: 384;
    BordersV                   :: 1536;
    BordersInner               :: 640;
    BordersOuter               :: 1280;
    Borders                    :: 1920;
    NoBordersInBody            :: 2048;
    NoBordersInBodyUntilResize :: 4096;

    SizingFixedFit             :: 8192;
    SizingFixedSame            :: 16384;
    SizingStretchProp          :: 24576;
    SizingStretchSame          :: 32768;

    NoHostExtendX              :: 65536;
    NoHostExtendY              :: 131072;
    NoKeepColumnsVisible       :: 262144;
    PreciseWidths              :: 524288;

    NoClip                     :: 1048576;

    PadOuterX                  :: 2097152;
    NoPadOuterX                :: 4194304;
    NoPadInnerX                :: 8388608;

    ScrollX                    :: 16777216;
    ScrollY                    :: 33554432;

    SortMulti                  :: 67108864;
    SortTristate               :: 134217728;

    HighlightHoveredColumn     :: 268435456;

    SizingMask_                :: 57344;

    ImGuiTableFlags_None                       :: None;
    ImGuiTableFlags_Resizable                  :: Resizable;
    ImGuiTableFlags_Reorderable                :: Reorderable;
    ImGuiTableFlags_Hideable                   :: Hideable;
    ImGuiTableFlags_Sortable                   :: Sortable;
    ImGuiTableFlags_NoSavedSettings            :: NoSavedSettings;
    ImGuiTableFlags_ContextMenuInBody          :: ContextMenuInBody;

    ImGuiTableFlags_RowBg                      :: RowBg;
    ImGuiTableFlags_BordersInnerH              :: BordersInnerH;
    ImGuiTableFlags_BordersOuterH              :: BordersOuterH;
    ImGuiTableFlags_BordersInnerV              :: BordersInnerV;
    ImGuiTableFlags_BordersOuterV              :: BordersOuterV;
    ImGuiTableFlags_BordersH                   :: BordersH;
    ImGuiTableFlags_BordersV                   :: BordersV;
    ImGuiTableFlags_BordersInner               :: BordersInner;
    ImGuiTableFlags_BordersOuter               :: BordersOuter;
    ImGuiTableFlags_Borders                    :: Borders;
    ImGuiTableFlags_NoBordersInBody            :: NoBordersInBody;
    ImGuiTableFlags_NoBordersInBodyUntilResize :: NoBordersInBodyUntilResize;

    ImGuiTableFlags_SizingFixedFit             :: SizingFixedFit;
    ImGuiTableFlags_SizingFixedSame            :: SizingFixedSame;
    ImGuiTableFlags_SizingStretchProp          :: SizingStretchProp;
    ImGuiTableFlags_SizingStretchSame          :: SizingStretchSame;

    ImGuiTableFlags_NoHostExtendX              :: NoHostExtendX;
    ImGuiTableFlags_NoHostExtendY              :: NoHostExtendY;
    ImGuiTableFlags_NoKeepColumnsVisible       :: NoKeepColumnsVisible;
    ImGuiTableFlags_PreciseWidths              :: PreciseWidths;

    ImGuiTableFlags_NoClip                     :: NoClip;

    ImGuiTableFlags_PadOuterX                  :: PadOuterX;
    ImGuiTableFlags_NoPadOuterX                :: NoPadOuterX;
    ImGuiTableFlags_NoPadInnerX                :: NoPadInnerX;

    ImGuiTableFlags_ScrollX                    :: ScrollX;
    ImGuiTableFlags_ScrollY                    :: ScrollY;

    ImGuiTableFlags_SortMulti                  :: SortMulti;
    ImGuiTableFlags_SortTristate               :: SortTristate;

    ImGuiTableFlags_HighlightHoveredColumn     :: HighlightHoveredColumn;

    ImGuiTableFlags_SizingMask_                :: SizingMask_;
}

// Flags for ImGui::TableSetupColumn()
ImGuiTableColumnFlags_ :: enum s32 {
    None                 :: 0;
    Disabled             :: 1;
    DefaultHide          :: 2;
    DefaultSort          :: 4;
    WidthStretch         :: 8;
    WidthFixed           :: 16;
    NoResize             :: 32;
    NoReorder            :: 64;
    NoHide               :: 128;
    NoClip               :: 256;
    NoSort               :: 512;
    NoSortAscending      :: 1024;
    NoSortDescending     :: 2048;
    NoHeaderLabel        :: 4096;
    NoHeaderWidth        :: 8192;
    PreferSortAscending  :: 16384;
    PreferSortDescending :: 32768;
    IndentEnable         :: 65536;
    IndentDisable        :: 131072;
    AngledHeader         :: 262144;

    IsEnabled            :: 16777216;
    IsVisible            :: 33554432;
    IsSorted             :: 67108864;
    IsHovered            :: 134217728;

    WidthMask_           :: 24;
    IndentMask_          :: 196608;
    StatusMask_          :: 251658240;
    NoDirectResize_      :: 1073741824;

    ImGuiTableColumnFlags_None                 :: None;
    ImGuiTableColumnFlags_Disabled             :: Disabled;
    ImGuiTableColumnFlags_DefaultHide          :: DefaultHide;
    ImGuiTableColumnFlags_DefaultSort          :: DefaultSort;
    ImGuiTableColumnFlags_WidthStretch         :: WidthStretch;
    ImGuiTableColumnFlags_WidthFixed           :: WidthFixed;
    ImGuiTableColumnFlags_NoResize             :: NoResize;
    ImGuiTableColumnFlags_NoReorder            :: NoReorder;
    ImGuiTableColumnFlags_NoHide               :: NoHide;
    ImGuiTableColumnFlags_NoClip               :: NoClip;
    ImGuiTableColumnFlags_NoSort               :: NoSort;
    ImGuiTableColumnFlags_NoSortAscending      :: NoSortAscending;
    ImGuiTableColumnFlags_NoSortDescending     :: NoSortDescending;
    ImGuiTableColumnFlags_NoHeaderLabel        :: NoHeaderLabel;
    ImGuiTableColumnFlags_NoHeaderWidth        :: NoHeaderWidth;
    ImGuiTableColumnFlags_PreferSortAscending  :: PreferSortAscending;
    ImGuiTableColumnFlags_PreferSortDescending :: PreferSortDescending;
    ImGuiTableColumnFlags_IndentEnable         :: IndentEnable;
    ImGuiTableColumnFlags_IndentDisable        :: IndentDisable;
    ImGuiTableColumnFlags_AngledHeader         :: AngledHeader;

    ImGuiTableColumnFlags_IsEnabled            :: IsEnabled;
    ImGuiTableColumnFlags_IsVisible            :: IsVisible;
    ImGuiTableColumnFlags_IsSorted             :: IsSorted;
    ImGuiTableColumnFlags_IsHovered            :: IsHovered;

    ImGuiTableColumnFlags_WidthMask_           :: WidthMask_;
    ImGuiTableColumnFlags_IndentMask_          :: IndentMask_;
    ImGuiTableColumnFlags_StatusMask_          :: StatusMask_;
    ImGuiTableColumnFlags_NoDirectResize_      :: NoDirectResize_;
}

// Flags for ImGui::TableNextRow()
ImGuiTableRowFlags_ :: enum s32 {
    None    :: 0;
    Headers :: 1;

    ImGuiTableRowFlags_None    :: None;
    ImGuiTableRowFlags_Headers :: Headers;
}

// Enum for ImGui::TableSetBgColor()
// Background colors are rendering in 3 layers:
//  - Layer 0: draw with RowBg0 color if set, otherwise draw with ColumnBg0 if set.
//  - Layer 1: draw with RowBg1 color if set, otherwise draw with ColumnBg1 if set.
//  - Layer 2: draw with CellBg color if set.
// The purpose of the two row/columns layers is to let you decide if a background color change should override or blend with the existing color.
// When using ImGuiTableFlags_RowBg on the table, each row has the RowBg0 color automatically set for odd/even rows.
// If you set the color of RowBg0 target, your color will override the existing RowBg0 color.
// If you set the color of RowBg1 or ColumnBg1 target, your color will blend over the RowBg0 color.
ImGuiTableBgTarget_ :: enum s32 {
    None   :: 0;
    RowBg0 :: 1;
    RowBg1 :: 2;
    CellBg :: 3;

    ImGuiTableBgTarget_None   :: None;
    ImGuiTableBgTarget_RowBg0 :: RowBg0;
    ImGuiTableBgTarget_RowBg1 :: RowBg1;
    ImGuiTableBgTarget_CellBg :: CellBg;
}

// Sorting specifications for a table (often handling sort specs for a single column, occasionally more)
// Obtained by calling TableGetSortSpecs().
// When 'SpecsDirty == true' you can sort your data. It will be true with sorting specs have changed since last call, or the first time.
// Make sure to set 'SpecsDirty = false' after sorting, else you may wastefully sort your data every frame!
ImGuiTableSortSpecs :: struct {
    Specs:      *ImGuiTableColumnSortSpecs; // Pointer to sort spec array.
    SpecsCount: s32; // Sort spec count. Most often 1. May be > 1 when ImGuiTableFlags_SortMulti is enabled. May be == 0 when ImGuiTableFlags_SortTristate is enabled.
    SpecsDirty: bool; // Set to true when specs have changed since last time! Use this to sort again, then clear the flag.

    Constructor :: (this: *ImGuiTableSortSpecs) -> void #cpp_method #foreign imgui "??0ImGuiTableSortSpecs@@QEAA@XZ";
}

// Sorting specification for one column of a table (sizeof == 12 bytes)
ImGuiTableColumnSortSpecs :: struct {
    ColumnUserID:  ImGuiID; // User id of the column (if specified by a TableSetupColumn() call)
    ColumnIndex:   ImS16; // Index of the column
    SortOrder:     ImS16; // Index within parent ImGuiTableSortSpecs (always stored in order starting from 0, tables sorted on a single criteria will always have a 0 here)
    SortDirection: ImGuiSortDirection; // ImGuiSortDirection_Ascending or ImGuiSortDirection_Descending

    Constructor :: (this: *ImGuiTableColumnSortSpecs) -> void #cpp_method #foreign imgui "??0ImGuiTableColumnSortSpecs@@QEAA@XZ";
}

//-----------------------------------------------------------------------------
// IM_MALLOC(), IM_FREE(), IM_NEW(), IM_PLACEMENT_NEW(), IM_DELETE()
// We call C++ constructor on own allocated memory via the placement "new(ptr) Type()" syntax.
// Defining a custom placement new() with a custom parameter allows us to bypass including <new> which on some platforms complains when user has disabled exceptions.
//-----------------------------------------------------------------------------
ImNewWrapper :: struct {
    using imnewwrapper__raw: ImNewWrapper__RAW;
    __empty_struct_padding:  u8; // C++ makes empty structs have length 1
}
// ImNewWrapper without empty struct padding. Needed to generate a C++-equivalent struct layout when "inheriting" from ImNewWrapper
ImNewWrapper__RAW :: struct {}
operatornew :: (unknown0: u64, unknown1: ImNewWrapper, ptr: *void) -> *void #foreign imgui "??2@YAPEAX_KUImNewWrapper@@PEAX@Z";

//-----------------------------------------------------------------------------
// ImVector<>
// Lightweight std::vector<>-like class to avoid dragging dependencies (also, some implementations of STL with debug enabled are absurdly slow, we bypass it so our code runs fast in debug).
//-----------------------------------------------------------------------------
// - You generally do NOT need to care or use this ever. But we need to make it available in imgui.h because some of our public structures are relying on it.
// - We use std-like naming convention here, which is a little unusual for this codebase.
// - Important: clear() frees memory, resize(0) keep the allocated buffer. We use resize(0) a lot to intentionally recycle allocated buffers across frames and amortize our costs.
// - Important: our implementation does NOT call C++ constructors/destructors, we treat everything as raw data! This is intentional but be extra mindful of that,
//   Do NOT use this class as a std::vector replacement in your own code! Many of the structures used by dear imgui can be safely initialized by a zero-memset.
//-----------------------------------------------------------------------------
ImVector :: struct(T: Type) {
    Size:           s32;
    Capacity:       s32;
    Data:           *T;

    // Provide standard typedefs but we don't use them ourselves.
    value_type :: T;
    iterator :: *value_type;
    const_iterator :: *value_type;
}

//-----------------------------------------------------------------------------
// [SECTION] ImGuiStyle
//-----------------------------------------------------------------------------
// You may modify the ImGui::GetStyle() main instance during initialization and before NewFrame().
// During the frame, use ImGui::PushStyleVar(ImGuiStyleVar_XXXX)/PopStyleVar() to alter the main style values,
// and ImGui::PushStyleColor(ImGuiCol_XXX)/PopStyleColor() for colors.
//-----------------------------------------------------------------------------
ImGuiStyle :: struct {
    FontSizeBase:                     float; // Current base font size before external global factors are applied. Use PushFont(NULL, size) to modify. Use ImGui::GetFontSize() to obtain scaled value.
    FontScaleMain:                    float; // Main global scale factor. May be set by application once, or exposed to end-user.
    FontScaleDpi:                     float; // Additional global scale factor from viewport/monitor contents scale. When io.ConfigDpiScaleFonts is enabled, this is automatically overwritten when changing monitor DPI.

    Alpha:                            float; // Global alpha applies to everything in Dear ImGui.
    DisabledAlpha:                    float; // Additional alpha multiplier applied by BeginDisabled(). Multiply over current value of Alpha.
    WindowPadding:                    ImVec2; // Padding within a window.
    WindowRounding:                   float; // Radius of window corners rounding. Set to 0.0f to have rectangular windows. Large values tend to lead to variety of artifacts and are not recommended.
    WindowBorderSize:                 float; // Thickness of border around windows. Generally set to 0.0f or 1.0f. (Other values are not well tested and more CPU/GPU costly).
    WindowBorderHoverPadding:         float; // Hit-testing extent outside/inside resizing border. Also extend determination of hovered window. Generally meaningfully larger than WindowBorderSize to make it easy to reach borders.
    WindowMinSize:                    ImVec2; // Minimum window size. This is a global setting. If you want to constrain individual windows, use SetNextWindowSizeConstraints().
    WindowTitleAlign:                 ImVec2; // Alignment for title bar text. Defaults to (0.0f,0.5f) for left-aligned,vertically centered.
    WindowMenuButtonPosition:         ImGuiDir; // Side of the collapsing/docking button in the title bar (None/Left/Right). Defaults to ImGuiDir_Left.
    ChildRounding:                    float; // Radius of child window corners rounding. Set to 0.0f to have rectangular windows.
    ChildBorderSize:                  float; // Thickness of border around child windows. Generally set to 0.0f or 1.0f. (Other values are not well tested and more CPU/GPU costly).
    PopupRounding:                    float; // Radius of popup window corners rounding. (Note that tooltip windows use WindowRounding)
    PopupBorderSize:                  float; // Thickness of border around popup/tooltip windows. Generally set to 0.0f or 1.0f. (Other values are not well tested and more CPU/GPU costly).
    FramePadding:                     ImVec2; // Padding within a framed rectangle (used by most widgets).
    FrameRounding:                    float; // Radius of frame corners rounding. Set to 0.0f to have rectangular frame (used by most widgets).
    FrameBorderSize:                  float; // Thickness of border around frames. Generally set to 0.0f or 1.0f. (Other values are not well tested and more CPU/GPU costly).
    ItemSpacing:                      ImVec2; // Horizontal and vertical spacing between widgets/lines.
    ItemInnerSpacing:                 ImVec2; // Horizontal and vertical spacing between within elements of a composed widget (e.g. a slider and its label).
    CellPadding:                      ImVec2; // Padding within a table cell. Cellpadding.x is locked for entire table. CellPadding.y may be altered between different rows.
    TouchExtraPadding:                ImVec2; // Expand reactive bounding box for touch-based system where touch position is not accurate enough. Unfortunately we don't sort widgets so priority on overlap will always be given to the first widget. So don't grow this too much!
    IndentSpacing:                    float; // Horizontal indentation when e.g. entering a tree node. Generally == (FontSize + FramePadding.x*2).
    ColumnsMinSpacing:                float; // Minimum horizontal spacing between two columns. Preferably > (FramePadding.x + 1).
    ScrollbarSize:                    float; // Width of the vertical scrollbar, Height of the horizontal scrollbar.
    ScrollbarRounding:                float; // Radius of grab corners for scrollbar.
    ScrollbarPadding:                 float; // Padding of scrollbar grab within its frame (same for both axises).
    GrabMinSize:                      float; // Minimum width/height of a grab box for slider/scrollbar.
    GrabRounding:                     float; // Radius of grabs corners rounding. Set to 0.0f to have rectangular slider grabs.
    LogSliderDeadzone:                float; // The size in pixels of the dead-zone around zero on logarithmic sliders that cross zero.
    ImageBorderSize:                  float; // Thickness of border around Image() calls.
    TabRounding:                      float; // Radius of upper corners of a tab. Set to 0.0f to have rectangular tabs.
    TabBorderSize:                    float; // Thickness of border around tabs.
    TabMinWidthBase:                  float; // Minimum tab width, to make tabs larger than their contents. TabBar buttons are not affected.
    TabMinWidthShrink:                float; // Minimum tab width after shrinking, when using ImGuiTabBarFlags_FittingPolicyMixed policy.
    TabCloseButtonMinWidthSelected:   float; // -1: always visible. 0.0f: visible when hovered. >0.0f: visible when hovered if minimum width.
    TabCloseButtonMinWidthUnselected: float; // -1: always visible. 0.0f: visible when hovered. >0.0f: visible when hovered if minimum width. FLT_MAX: never show close button when unselected.
    TabBarBorderSize:                 float; // Thickness of tab-bar separator, which takes on the tab active color to denote focus.
    TabBarOverlineSize:               float; // Thickness of tab-bar overline, which highlights the selected tab-bar.
    TableAngledHeadersAngle:          float; // Angle of angled headers (supported values range from -50.0f degrees to +50.0f degrees).
    TableAngledHeadersTextAlign:      ImVec2; // Alignment of angled headers within the cell
    TreeLinesFlags:                   ImGuiTreeNodeFlags; // Default way to draw lines connecting TreeNode hierarchy. ImGuiTreeNodeFlags_DrawLinesNone or ImGuiTreeNodeFlags_DrawLinesFull or ImGuiTreeNodeFlags_DrawLinesToNodes.
    TreeLinesSize:                    float; // Thickness of outlines when using ImGuiTreeNodeFlags_DrawLines.
    TreeLinesRounding:                float; // Radius of lines connecting child nodes to the vertical line.
    ColorButtonPosition:              ImGuiDir; // Side of the color button in the ColorEdit4 widget (left/right). Defaults to ImGuiDir_Right.
    ButtonTextAlign:                  ImVec2; // Alignment of button text when button is larger than text. Defaults to (0.5f, 0.5f) (centered).
    SelectableTextAlign:              ImVec2; // Alignment of selectable text. Defaults to (0.0f, 0.0f) (top-left aligned). It's generally important to keep this left-aligned if you want to lay multiple items on a same line.
    SeparatorTextBorderSize:          float; // Thickness of border in SeparatorText()
    SeparatorTextAlign:               ImVec2; // Alignment of text within the separator. Defaults to (0.0f, 0.5f) (left aligned, center).
    SeparatorTextPadding:             ImVec2; // Horizontal offset of text from each edge of the separator + spacing on other axis. Generally small values. .y is recommended to be == FramePadding.y.
    DisplayWindowPadding:             ImVec2; // Apply to regular windows: amount which we enforce to keep visible when moving near edges of your screen.
    DisplaySafeAreaPadding:           ImVec2; // Apply to every windows, menus, popups, tooltips: amount where we avoid displaying contents. Adjust if you cannot see the edges of your screen (e.g. on a TV where scaling has not been configured).
    MouseCursorScale:                 float; // Scale software rendered mouse cursor (when io.MouseDrawCursor is enabled). We apply per-monitor DPI scaling over this scale. May be removed later.
    AntiAliasedLines:                 bool; // Enable anti-aliased lines/borders. Disable if you are really tight on CPU/GPU. Latched at the beginning of the frame (copied to ImDrawList).
    AntiAliasedLinesUseTex:           bool; // Enable anti-aliased lines/borders using textures where possible. Require backend to render with bilinear filtering (NOT point/nearest filtering). Latched at the beginning of the frame (copied to ImDrawList).
    AntiAliasedFill:                  bool; // Enable anti-aliased edges around filled shapes (rounded rectangles, circles, etc.). Disable if you are really tight on CPU/GPU. Latched at the beginning of the frame (copied to ImDrawList).
    CurveTessellationTol:             float; // Tessellation tolerance when using PathBezierCurveTo() without a specific number of segments. Decrease for highly tessellated curves (higher quality, more polygons), increase to reduce quality.
    CircleTessellationMaxError:       float; // Maximum error (in pixels) allowed when using AddCircle()/AddCircleFilled() or drawing rounded corner rectangles with no explicit segment count specified. Decrease for higher quality but more geometry.

    // Colors
    Colors:                           [58] ImVec4;

    HoverStationaryDelay:             float; // Delay for IsItemHovered(ImGuiHoveredFlags_Stationary). Time required to consider mouse stationary.
    HoverDelayShort:                  float; // Delay for IsItemHovered(ImGuiHoveredFlags_DelayShort). Usually used along with HoverStationaryDelay.
    HoverDelayNormal:                 float; // Delay for IsItemHovered(ImGuiHoveredFlags_DelayNormal). "
    HoverFlagsForTooltipMouse:        ImGuiHoveredFlags; // Default flags when using IsItemHovered(ImGuiHoveredFlags_ForTooltip) or BeginItemTooltip()/SetItemTooltip() while using mouse.
    HoverFlagsForTooltipNav:          ImGuiHoveredFlags; // Default flags when using IsItemHovered(ImGuiHoveredFlags_ForTooltip) or BeginItemTooltip()/SetItemTooltip() while using keyboard/gamepad.

    _MainScale:                       float; // FIXME-WIP: Reference scale, as applied by ScaleAllSizes().
    _NextFrameFontSizeBase:           float; // FIXME: Temporary hack until we finish remaining work.

    // Functions
    Constructor :: (this: *ImGuiStyle) -> void #cpp_method #foreign imgui "??0ImGuiStyle@@QEAA@XZ";
    ScaleAllSizes :: (this: *ImGuiStyle, scale_factor: float) -> void #cpp_method #foreign imgui "?ScaleAllSizes@ImGuiStyle@@QEAAXM@Z";
}

// [Internal] Storage used by IsKeyDown(), IsKeyPressed() etc functions.
// If prior to 1.87 you used io.KeysDownDuration[] (which was marked as internal), you should use GetKeyData(key)->DownDuration and *NOT* io.KeysData[key]->DownDuration.
ImGuiKeyData :: struct {
    Down:             bool; // True for if key is down
    DownDuration:     float; // Duration the key has been down (<0.0f: not pressed, 0.0f: just pressed, >0.0f: time held)
    DownDurationPrev: float; // Last frame duration the key has been down
    AnalogValue:      float; // 0.0f..1.0f for gamepad values
}

ImGuiIO :: struct {
    ConfigFlags:                                   ImGuiConfigFlags; // = 0              // See ImGuiConfigFlags_ enum. Set by user/application. Keyboard/Gamepad navigation options, etc.
    BackendFlags:                                  ImGuiBackendFlags; // = 0              // See ImGuiBackendFlags_ enum. Set by backend (imgui_impl_xxx files or custom backend) to communicate features supported by the backend.
    DisplaySize:                                   ImVec2; // <unset>          // Main display size, in pixels (== GetMainViewport()->Size). May change every frame.
    DisplayFramebufferScale:                       ImVec2; // = (1, 1)         // Main display density. For retina display where window coordinates are different from framebuffer coordinates. This will affect font density + will end up in ImDrawData::FramebufferScale.
    DeltaTime:                                     float; // = 1.0f/60.0f     // Time elapsed since last frame, in seconds. May change every frame.
    IniSavingRate:                                 float; // = 5.0f           // Minimum time between saving positions/sizes to .ini file, in seconds.
    IniFilename:                                   *u8; // = "imgui.ini"    // Path to .ini file (important: default "imgui.ini" is relative to current working dir!). Set NULL to disable automatic .ini loading/saving or if you want to manually call LoadIniSettingsXXX() / SaveIniSettingsXXX() functions.
    LogFilename:                                   *u8; // = "imgui_log.txt"// Path to .log file (default parameter to ImGui::LogToFile when no file is specified).
    UserData:                                      *void; // = NULL           // Store your own data.

    Fonts:                                         *ImFontAtlas; // <auto>           // Font atlas: load, rasterize and pack one or more fonts into a single texture.
    FontDefault:                                   *ImFont; // = NULL           // Font to use on NewFrame(). Use NULL to uses Fonts->Fonts[0].
    FontAllowUserScaling:                          bool; // = false          // [OBSOLETE] Allow user scaling text of individual window with CTRL+Wheel.

    ConfigNavSwapGamepadButtons:                   bool; // = false          // Swap Activate<>Cancel (A<>B) buttons, matching typical "Nintendo/Japanese style" gamepad layout.
    ConfigNavMoveSetMousePos:                      bool; // = false          // Directional/tabbing navigation teleports the mouse cursor. May be useful on TV/console systems where moving a virtual mouse is difficult. Will update io.MousePos and set io.WantSetMousePos=true.
    ConfigNavCaptureKeyboard:                      bool; // = true           // Sets io.WantCaptureKeyboard when io.NavActive is set.
    ConfigNavEscapeClearFocusItem:                 bool; // = true           // Pressing Escape can clear focused item + navigation id/highlight. Set to false if you want to always keep highlight on.
    ConfigNavEscapeClearFocusWindow:               bool; // = false          // Pressing Escape can clear focused window as well (super set of io.ConfigNavEscapeClearFocusItem).
    ConfigNavCursorVisibleAuto:                    bool; // = true           // Using directional navigation key makes the cursor visible. Mouse click hides the cursor.
    ConfigNavCursorVisibleAlways:                  bool; // = false          // Navigation cursor is always visible.

    MouseDrawCursor:                               bool; // = false          // Request ImGui to draw a mouse cursor for you (if you are on a platform without a mouse cursor). Cannot be easily renamed to 'io.ConfigXXX' because this is frequently used by backend implementations.
    ConfigMacOSXBehaviors:                         bool; // = defined(__APPLE__) // Swap Cmd<>Ctrl keys + OS X style text editing cursor movement using Alt instead of Ctrl, Shortcuts using Cmd/Super instead of Ctrl, Line/Text Start and End using Cmd+Arrows instead of Home/End, Double click selects by word instead of selecting whole text, Multi-selection in lists uses Cmd/Super instead of Ctrl.
    ConfigInputTrickleEventQueue:                  bool; // = true           // Enable input queue trickling: some types of events submitted during the same frame (e.g. button down + up) will be spread over multiple frames, improving interactions with low framerates.
    ConfigInputTextCursorBlink:                    bool; // = true           // Enable blinking cursor (optional as some users consider it to be distracting).
    ConfigInputTextEnterKeepActive:                bool; // = false          // [BETA] Pressing Enter will keep item active and select contents (single-line only).
    ConfigDragClickToInputText:                    bool; // = false          // [BETA] Enable turning DragXXX widgets into text input with a simple mouse click-release (without moving). Not desirable on devices without a keyboard.
    ConfigWindowsResizeFromEdges:                  bool; // = true           // Enable resizing of windows from their edges and from the lower-left corner. This requires ImGuiBackendFlags_HasMouseCursors for better mouse cursor feedback. (This used to be a per-window ImGuiWindowFlags_ResizeFromAnySide flag)
    ConfigWindowsMoveFromTitleBarOnly:             bool; // = false      // Enable allowing to move windows only when clicking on their title bar. Does not apply to windows without a title bar.
    ConfigWindowsCopyContentsWithCtrlC:            bool; // = false      // [EXPERIMENTAL] CTRL+C copy the contents of focused window into the clipboard. Experimental because: (1) has known issues with nested Begin/End pairs (2) text output quality varies (3) text output is in submission order rather than spatial order.
    ConfigScrollbarScrollByPage:                   bool; // = true           // Enable scrolling page by page when clicking outside the scrollbar grab. When disabled, always scroll to clicked location. When enabled, Shift+Click scrolls to clicked location.
    ConfigMemoryCompactTimer:                      float; // = 60.0f          // Timer (in seconds) to free transient windows/tables memory buffers when unused. Set to -1.0f to disable.

    MouseDoubleClickTime:                          float; // = 0.30f          // Time for a double-click, in seconds.
    MouseDoubleClickMaxDist:                       float; // = 6.0f           // Distance threshold to stay in to validate a double-click, in pixels.
    MouseDragThreshold:                            float; // = 6.0f           // Distance threshold before considering we are dragging.
    KeyRepeatDelay:                                float; // = 0.275f         // When holding a key/button, time before it starts repeating, in seconds (for buttons in Repeat mode, etc.).
    KeyRepeatRate:                                 float; // = 0.050f         // When holding a key/button, rate at which it repeats, in seconds.

    ConfigErrorRecovery:                           bool; // = true       // Enable error recovery support. Some errors won't be detected and lead to direct crashes if recovery is disabled.
    ConfigErrorRecoveryEnableAssert:               bool; // = true       // Enable asserts on recoverable error. By default call IM_ASSERT() when returning from a failing IM_ASSERT_USER_ERROR()
    ConfigErrorRecoveryEnableDebugLog:             bool; // = true       // Enable debug log output on recoverable errors.
    ConfigErrorRecoveryEnableTooltip:              bool; // = true       // Enable tooltip on recoverable errors. The tooltip include a way to enable asserts if they were disabled.

    ConfigDebugIsDebuggerPresent:                  bool; // = false          // Enable various tools calling IM_DEBUG_BREAK().

    ConfigDebugHighlightIdConflicts:               bool; // = true           // Highlight and show an error message popup when multiple items have conflicting identifiers.
    ConfigDebugHighlightIdConflictsShowItemPicker: bool; //=true // Show "Item Picker" button in aforementioned popup.

    ConfigDebugBeginReturnValueOnce:               bool; // = false          // First-time calls to Begin()/BeginChild() will return false. NEEDS TO BE SET AT APPLICATION BOOT TIME if you don't want to miss windows.
    ConfigDebugBeginReturnValueLoop:               bool; // = false          // Some calls to Begin()/BeginChild() will return false. Will cycle through window depths then repeat. Suggested use: add "io.ConfigDebugBeginReturnValue = io.KeyShift" in your main loop then occasionally press SHIFT. Windows should be flickering while running.

    ConfigDebugIgnoreFocusLoss:                    bool; // = false          // Ignore io.AddFocusEvent(false), consequently not calling io.ClearInputKeys()/io.ClearInputMouse() in input processing.

    ConfigDebugIniSettings:                        bool; // = false          // Save .ini data with extra comments (particularly helpful for Docking, but makes saving slower)

    BackendPlatformName:                           *u8; // = NULL
    BackendRendererName:                           *u8; // = NULL
    BackendPlatformUserData:                       *void; // = NULL           // User data for platform backend
    BackendRendererUserData:                       *void; // = NULL           // User data for renderer backend
    BackendLanguageUserData:                       *void; // = NULL           // User data for non C++ programming language backend

    // Input Functions
    AddKeyEvent :: (this: *ImGuiIO, key: ImGuiKey, down: bool) -> void #cpp_method #foreign imgui "?AddKeyEvent@ImGuiIO@@QEAAXW4ImGuiKey@@_N@Z";
    AddKeyAnalogEvent :: (this: *ImGuiIO, key: ImGuiKey, down: bool, v: float) -> void #cpp_method #foreign imgui "?AddKeyAnalogEvent@ImGuiIO@@QEAAXW4ImGuiKey@@_NM@Z";
    AddMousePosEvent :: (this: *ImGuiIO, x: float, y: float) -> void #cpp_method #foreign imgui "?AddMousePosEvent@ImGuiIO@@QEAAXMM@Z";
    AddMouseButtonEvent :: (this: *ImGuiIO, button: s32, down: bool) -> void #cpp_method #foreign imgui "?AddMouseButtonEvent@ImGuiIO@@QEAAXH_N@Z";
    AddMouseWheelEvent :: (this: *ImGuiIO, wheel_x: float, wheel_y: float) -> void #cpp_method #foreign imgui "?AddMouseWheelEvent@ImGuiIO@@QEAAXMM@Z";
    AddMouseSourceEvent :: (this: *ImGuiIO, source: ImGuiMouseSource) -> void #cpp_method #foreign imgui "?AddMouseSourceEvent@ImGuiIO@@QEAAXW4ImGuiMouseSource@@@Z";
    AddFocusEvent :: (this: *ImGuiIO, focused: bool) -> void #cpp_method #foreign imgui "?AddFocusEvent@ImGuiIO@@QEAAX_N@Z";
    AddInputCharacter :: (this: *ImGuiIO, c: u32) -> void #cpp_method #foreign imgui "?AddInputCharacter@ImGuiIO@@QEAAXI@Z";
    AddInputCharacterUTF16 :: (this: *ImGuiIO, c: ImWchar16) -> void #cpp_method #foreign imgui "?AddInputCharacterUTF16@ImGuiIO@@QEAAXG@Z";
    AddInputCharactersUTF8 :: (this: *ImGuiIO, str: *u8) -> void #cpp_method #foreign imgui "?AddInputCharactersUTF8@ImGuiIO@@QEAAXPEBD@Z";

    SetKeyEventNativeData :: (this: *ImGuiIO, key: ImGuiKey, native_keycode: s32, native_scancode: s32, native_legacy_index: s32 = -1) -> void #cpp_method #foreign imgui "?SetKeyEventNativeData@ImGuiIO@@QEAAXW4ImGuiKey@@HHH@Z";
    SetAppAcceptingEvents :: (this: *ImGuiIO, accepting_events: bool) -> void #cpp_method #foreign imgui "?SetAppAcceptingEvents@ImGuiIO@@QEAAX_N@Z";
    ClearEventsQueue :: (this: *ImGuiIO) -> void #cpp_method #foreign imgui "?ClearEventsQueue@ImGuiIO@@QEAAXXZ";
    ClearInputKeys :: (this: *ImGuiIO) -> void #cpp_method #foreign imgui "?ClearInputKeys@ImGuiIO@@QEAAXXZ";
    ClearInputMouse :: (this: *ImGuiIO) -> void #cpp_method #foreign imgui "?ClearInputMouse@ImGuiIO@@QEAAXXZ";

    ClearInputCharacters :: (this: *ImGuiIO) -> void #cpp_method #foreign imgui "?ClearInputCharacters@ImGuiIO@@QEAAXXZ";

    WantCaptureMouse:                              bool; // Set when Dear ImGui will use mouse inputs, in this case do not dispatch them to your main game/application (either way, always pass on mouse inputs to imgui). (e.g. unclicked mouse is hovering over an imgui window, widget is active, mouse was clicked over an imgui window, etc.).
    WantCaptureKeyboard:                           bool; // Set when Dear ImGui will use keyboard inputs, in this case do not dispatch them to your main game/application (either way, always pass keyboard inputs to imgui). (e.g. InputText active, or an imgui window is focused and navigation is enabled, etc.).
    WantTextInput:                                 bool; // Mobile/console: when set, you may display an on-screen keyboard. This is set by Dear ImGui when it wants textual keyboard input to happen (e.g. when a InputText widget is active).
    WantSetMousePos:                               bool; // MousePos has been altered, backend should reposition mouse on next frame. Rarely used! Set only when io.ConfigNavMoveSetMousePos is enabled.
    WantSaveIniSettings:                           bool; // When manual .ini load/save is active (io.IniFilename == NULL), this will be set to notify your application that you can call SaveIniSettingsToMemory() and save yourself. Important: clear io.WantSaveIniSettings yourself after saving!
    NavActive:                                     bool; // Keyboard/Gamepad navigation is currently allowed (will handle ImGuiKey_NavXXX events) = a window is focused and it doesn't use the ImGuiWindowFlags_NoNavInputs flag.
    NavVisible:                                    bool; // Keyboard/Gamepad navigation highlight is visible and allowed (will handle ImGuiKey_NavXXX events).
    Framerate:                                     float; // Estimate of application framerate (rolling average over 60 frames, based on io.DeltaTime), in frame per second. Solely for convenience. Slow applications may not want to use a moving average or may want to reset underlying buffers occasionally.
    MetricsRenderVertices:                         s32; // Vertices output during last call to Render()
    MetricsRenderIndices:                          s32; // Indices output during last call to Render() = number of triangles * 3
    MetricsRenderWindows:                          s32; // Number of visible windows
    MetricsActiveWindows:                          s32; // Number of active windows
    MouseDelta:                                    ImVec2; // Mouse delta. Note that this is zero if either current or previous position are invalid (-FLT_MAX,-FLT_MAX), so a disappearing/reappearing mouse won't have a huge delta.

    Ctx:                                           *ImGuiContext; // Parent UI context (needs to be set explicitly by parent).

    MousePos:                                      ImVec2; // Mouse position, in pixels. Set to ImVec2(-FLT_MAX, -FLT_MAX) if mouse is unavailable (on another screen, etc.)
    MouseDown:                                     [5] bool; // Mouse buttons: 0=left, 1=right, 2=middle + extras (ImGuiMouseButton_COUNT == 5). Dear ImGui mostly uses left and right buttons. Other buttons allow us to track if the mouse is being used by your application + available to user as a convenience via IsMouse** API.
    MouseWheel:                                    float; // Mouse wheel Vertical: 1 unit scrolls about 5 lines text. >0 scrolls Up, <0 scrolls Down. Hold SHIFT to turn vertical scroll into horizontal scroll.
    MouseWheelH:                                   float; // Mouse wheel Horizontal. >0 scrolls Left, <0 scrolls Right. Most users don't have a mouse with a horizontal wheel, may not be filled by all backends.
    MouseSource:                                   ImGuiMouseSource; // Mouse actual input peripheral (Mouse/TouchScreen/Pen).
    KeyCtrl:                                       bool; // Keyboard modifier down: Ctrl (non-macOS), Cmd (macOS)
    KeyShift:                                      bool; // Keyboard modifier down: Shift
    KeyAlt:                                        bool; // Keyboard modifier down: Alt
    KeySuper:                                      bool; // Keyboard modifier down: Windows/Super (non-macOS), Ctrl (macOS)

    KeyMods:                                       ImGuiKeyChord; // Key mods flags (any of ImGuiMod_Ctrl/ImGuiMod_Shift/ImGuiMod_Alt/ImGuiMod_Super flags, same as io.KeyCtrl/KeyShift/KeyAlt/KeySuper but merged into flags). Read-only, updated by NewFrame()
    KeysData:                                      [155] ImGuiKeyData; // Key state for all known keys. MUST use 'key - ImGuiKey_NamedKey_BEGIN' as index. Use IsKeyXXX() functions to access this.
    WantCaptureMouseUnlessPopupClose:              bool; // Alternative to WantCaptureMouse: (WantCaptureMouse == true && WantCaptureMouseUnlessPopupClose == false) when a click over void is expected to close a popup.
    MousePosPrev:                                  ImVec2; // Previous mouse position (note that MouseDelta is not necessary == MousePos-MousePosPrev, in case either position is invalid)
    MouseClickedPos:                               [5] ImVec2; // Position at time of clicking
    MouseClickedTime:                              [5] float64; // Time of last click (used to figure out double-click)
    MouseClicked:                                  [5] bool; // Mouse button went from !Down to Down (same as MouseClickedCount[x] != 0)
    MouseDoubleClicked:                            [5] bool; // Has mouse button been double-clicked? (same as MouseClickedCount[x] == 2)
    MouseClickedCount:                             [5] ImU16; // == 0 (not clicked), == 1 (same as MouseClicked[]), == 2 (double-clicked), == 3 (triple-clicked) etc. when going from !Down to Down
    MouseClickedLastCount:                         [5] ImU16; // Count successive number of clicks. Stays valid after mouse release. Reset after another click is done.
    MouseReleased:                                 [5] bool; // Mouse button went from Down to !Down
    MouseReleasedTime:                             [5] float64; // Time of last released (rarely used! but useful to handle delayed single-click when trying to disambiguate them from double-click).
    MouseDownOwned:                                [5] bool; // Track if button was clicked inside a dear imgui window or over void blocked by a popup. We don't request mouse capture from the application if click started outside ImGui bounds.
    MouseDownOwnedUnlessPopupClose:                [5] bool; // Track if button was clicked inside a dear imgui window.
    MouseWheelRequestAxisSwap:                     bool; // On a non-Mac system, holding SHIFT requests WheelY to perform the equivalent of a WheelX event. On a Mac system this is already enforced by the system.
    MouseCtrlLeftAsRightClick:                     bool; // (OSX) Set to true when the current click was a Ctrl+click that spawned a simulated right click
    MouseDownDuration:                             [5] float; // Duration the mouse button has been down (0.0f == just clicked)
    MouseDownDurationPrev:                         [5] float; // Previous time the mouse button has been down
    MouseDragMaxDistanceSqr:                       [5] float; // Squared maximum distance of how much mouse has traveled from the clicking point (used for moving thresholds)
    PenPressure:                                   float; // Touch/Pen pressure (0.0f to 1.0f, should be >0.0f only when MouseDown[0] == true). Helper storage currently unused by Dear ImGui.
    AppFocusLost:                                  bool; // Only modify via AddFocusEvent()
    AppAcceptingEvents:                            bool; // Only modify via SetAppAcceptingEvents()
    InputQueueSurrogate:                           ImWchar16; // For AddInputCharacterUTF16()
    InputQueueCharacters:                          ImVector(ImWchar); // Queue of _characters_ input (obtained by platform backend). Fill using AddInputCharacter() helper.

    FontGlobalScale:                               float; // Moved io.FontGlobalScale to style.FontScaleMain in 1.92 (June 2025)

    // Legacy: before 1.91.1, clipboard functions were stored in ImGuiIO instead of ImGuiPlatformIO.
    // As this is will affect all users of custom engines/backends, we are providing proper legacy redirection (will obsolete).
    GetClipboardTextFn:                            #type (user_data: *void) -> *u8 #c_call;
    SetClipboardTextFn:                            #type (user_data: *void, text: *u8) -> void #c_call;
    ClipboardUserData:                             *void;

    Constructor :: (this: *ImGuiIO) -> void #cpp_method #foreign imgui "??0ImGuiIO@@QEAA@XZ";
}

// Shared state of InputText(), passed as an argument to your callback when a ImGuiInputTextFlags_Callback* flag is used.
// The callback function should return 0 by default.
// Callbacks (follow a flag name and see comments in ImGuiInputTextFlags_ declarations for more details)
// - ImGuiInputTextFlags_CallbackEdit:        Callback on buffer edit. Note that InputText() already returns true on edit + you can always use IsItemEdited(). The callback is useful to manipulate the underlying buffer while focus is active.
// - ImGuiInputTextFlags_CallbackAlways:      Callback on each iteration
// - ImGuiInputTextFlags_CallbackCompletion:  Callback on pressing TAB
// - ImGuiInputTextFlags_CallbackHistory:     Callback on pressing Up/Down arrows
// - ImGuiInputTextFlags_CallbackCharFilter:  Callback on character inputs to replace or discard them. Modify 'EventChar' to replace or discard, or return 1 in callback to discard.
// - ImGuiInputTextFlags_CallbackResize:      Callback on buffer capacity changes request (beyond 'buf_size' parameter value), allowing the string to grow.
ImGuiInputTextCallbackData :: struct {
    Ctx:            *ImGuiContext; // Parent UI context
    EventFlag:      ImGuiInputTextFlags; // One ImGuiInputTextFlags_Callback*    // Read-only
    Flags:          ImGuiInputTextFlags; // What user passed to InputText()      // Read-only
    UserData:       *void; // What user passed to InputText()      // Read-only

    EventChar:      ImWchar; // Character input                      // Read-write   // [CharFilter] Replace character with another one, or set to zero to drop. return 1 is equivalent to setting EventChar=0;
    EventKey:       ImGuiKey; // Key pressed (Up/Down/TAB)            // Read-only    // [Completion,History]
    Buf:            *u8; // Text buffer                          // Read-write   // [Resize] Can replace pointer / [Completion,History,Always] Only write to pointed data, don't replace the actual pointer!
    BufTextLen:     s32; // Text length (in bytes)               // Read-write   // [Resize,Completion,History,Always] Exclude zero-terminator storage. In C land: == strlen(some_text), in C++ land: string.length()
    BufSize:        s32; // Buffer size (in bytes) = capacity+1  // Read-only    // [Resize,Completion,History,Always] Include zero-terminator storage. In C land: == ARRAYSIZE(my_char_array), in C++ land: string.capacity()+1
    BufDirty:       bool; // Set if you modify Buf/BufTextLen!    // Write        // [Completion,History,Always]
    CursorPos:      s32; //                                      // Read-write   // [Completion,History,Always]
    SelectionStart: s32; //                                      // Read-write   // [Completion,History,Always] == to SelectionEnd when no selection
    SelectionEnd:   s32; //                                      // Read-write   // [Completion,History,Always]

    // Helper functions for text manipulation.
    // Use those function to benefit from the CallbackResize behaviors. Calling those function reset the selection.
    Constructor :: (this: *ImGuiInputTextCallbackData) -> void #cpp_method #foreign imgui "??0ImGuiInputTextCallbackData@@QEAA@XZ";
    DeleteChars :: (this: *ImGuiInputTextCallbackData, pos: s32, bytes_count: s32) -> void #cpp_method #foreign imgui "?DeleteChars@ImGuiInputTextCallbackData@@QEAAXHH@Z";
    InsertChars :: (this: *ImGuiInputTextCallbackData, pos: s32, text: *u8, text_end: *u8 = null) -> void #cpp_method #foreign imgui "?InsertChars@ImGuiInputTextCallbackData@@QEAAXHPEBD0@Z";
    SelectAll :: (this: *ImGuiInputTextCallbackData) -> void #cpp_method #foreign imgui "?SelectAll@ImGuiInputTextCallbackData@@QEAAXXZ";
}

// Resizing callback data to apply custom constraint. As enabled by SetNextWindowSizeConstraints(). Callback is called during the next Begin().
// NB: For basic min/max size constraint on each axis you don't need to use the callback! The SetNextWindowSizeConstraints() parameters are enough.
ImGuiSizeCallbackData :: struct {
    UserData:    *void; // Read-only.   What user passed to SetNextWindowSizeConstraints(). Generally store an integer or float in here (need reinterpret_cast<>).
    Pos:         ImVec2; // Read-only.   Window position, for reference.
    CurrentSize: ImVec2; // Read-only.   Current window size.
    DesiredSize: ImVec2; // Read-write.  Desired size, based on user's mouse position. Write to this field to restrain resizing.
}

// Data payload for Drag and Drop operations: AcceptDragDropPayload(), GetDragDropPayload()
ImGuiPayload :: struct {
    Data:           *void; // Data (copied and owned by dear imgui)
    DataSize:       s32; // Data size

    SourceId:       ImGuiID; // Source item id
    SourceParentId: ImGuiID; // Source parent id (if available)
    DataFrameCount: s32; // Data timestamp
    DataType:       [33] u8; // Data type tag (short user-supplied string, 32 characters max)
    Preview:        bool; // Set when AcceptDragDropPayload() was called and mouse has been hovering the target item (nb: handle overlapping drag targets)
    Delivery:       bool; // Set when AcceptDragDropPayload() was called and mouse button is released over the target item.

    Constructor :: (this: *ImGuiPayload) -> void #cpp_method #foreign imgui "??0ImGuiPayload@@QEAA@XZ";
    Clear :: (this: *ImGuiPayload) -> void #cpp_method #foreign imgui "?Clear@ImGuiPayload@@QEAAXXZ";
    IsDataType :: (this: *ImGuiPayload, type: *u8) -> bool #cpp_method #foreign imgui "?IsDataType@ImGuiPayload@@QEBA_NPEBD@Z";
}

// Helper: Execute a block of code at maximum once a frame. Convenient if you want to quickly create a UI within deep-nested code that runs multiple times every frame.
// Usage: static ImGuiOnceUponAFrame oaf; if (oaf) ImGui::Text("This will be called only once per frame");
ImGuiOnceUponAFrame :: struct {
    RefFrame: s32;
}

// Helper: Parse and apply text filters. In format "aaaaa[,bbbb][,ccccc]"
ImGuiTextFilter :: struct {
    Constructor :: (this: *ImGuiTextFilter, default_filter: *u8 = "") -> void #cpp_method #foreign imgui "??0ImGuiTextFilter@@QEAA@PEBD@Z";
    Draw :: (this: *ImGuiTextFilter, label: *u8 = "Filter (inc,-exc)", width: float = 0.0) -> bool #cpp_method #foreign imgui "?Draw@ImGuiTextFilter@@QEAA_NPEBDM@Z";
    PassFilter :: (this: *ImGuiTextFilter, text: *u8, text_end: *u8 = null) -> bool #cpp_method #foreign imgui "?PassFilter@ImGuiTextFilter@@QEBA_NPEBD0@Z";
    Build :: (this: *ImGuiTextFilter) -> void #cpp_method #foreign imgui "?Build@ImGuiTextFilter@@QEAAXXZ";
    Clear :: (this: *ImGuiTextFilter) -> void #cpp_method #foreign imgui "?Clear@ImGuiTextFilter@@QEAAXXZ";
    IsActive :: (this: *ImGuiTextFilter) -> bool #cpp_method #foreign imgui "?IsActive@ImGuiTextFilter@@QEBA_NXZ";

    // [Internal]
    ImGuiTextRange :: struct {
        b: *u8;
        e: *u8;

        Constructor :: (this: *ImGuiTextRange, _b: *u8, _e: *u8) -> void #cpp_method #foreign imgui "??0ImGuiTextRange@ImGuiTextFilter@@QEAA@PEBD0@Z";
        empty :: (this: *ImGuiTextRange) -> bool #cpp_method #foreign imgui "?empty@ImGuiTextRange@ImGuiTextFilter@@QEBA_NXZ";
        split :: (this: *ImGuiTextRange, separator: u8, out: *ImVector(ImGuiTextRange)) -> void #cpp_method #foreign imgui "?split@ImGuiTextRange@ImGuiTextFilter@@QEBAXDPEAU?$ImVector@UImGuiTextRange@ImGuiTextFilter@@@@@Z";
    }

    InputBuf:  [256] u8;
    Filters:   ImVector(ImGuiTextRange);
    CountGrep: s32;
}

// Helper: Growable text buffer for logging/accumulating text
// (this could be called 'ImGuiTextBuilder' / 'ImGuiStringBuilder')
ImGuiTextBuffer :: struct {
    Buf: ImVector(u8);

    Constructor :: (this: *ImGuiTextBuffer) -> void #cpp_method #foreign imgui "??0ImGuiTextBuffer@@QEAA@XZ";

    begin :: (this: *ImGuiTextBuffer) -> *u8 #cpp_method #foreign imgui "?begin@ImGuiTextBuffer@@QEBAPEBDXZ";
    end :: (this: *ImGuiTextBuffer) -> *u8 #cpp_method #foreign imgui "?end@ImGuiTextBuffer@@QEBAPEBDXZ";
    size :: (this: *ImGuiTextBuffer) -> s32 #cpp_method #foreign imgui "?size@ImGuiTextBuffer@@QEBAHXZ";
    empty :: (this: *ImGuiTextBuffer) -> bool #cpp_method #foreign imgui "?empty@ImGuiTextBuffer@@QEBA_NXZ";
    clear :: (this: *ImGuiTextBuffer) -> void #cpp_method #foreign imgui "?clear@ImGuiTextBuffer@@QEAAXXZ";
    resize :: (this: *ImGuiTextBuffer, size: s32) -> void #cpp_method #foreign imgui "?resize@ImGuiTextBuffer@@QEAAXH@Z";
    reserve :: (this: *ImGuiTextBuffer, capacity: s32) -> void #cpp_method #foreign imgui "?reserve@ImGuiTextBuffer@@QEAAXH@Z";
    c_str :: (this: *ImGuiTextBuffer) -> *u8 #cpp_method #foreign imgui "?c_str@ImGuiTextBuffer@@QEBAPEBDXZ";
    append :: (this: *ImGuiTextBuffer, str: *u8, str_end: *u8 = null) -> void #cpp_method #foreign imgui "?append@ImGuiTextBuffer@@QEAAXPEBD0@Z";
    appendf_CFormat :: (this: *ImGuiTextBuffer, fmt: *u8, __args: ..Any) -> void #cpp_method #foreign imgui "?appendf@ImGuiTextBuffer@@QEAAXPEBDZZ";
    appendf :: (this: *ImGuiTextBuffer, fmt: string, __args: ..Any) {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, fmt, ..__args);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        appendf_CFormat(this, "%s", formatted_text.data);
    } @PrintLike
}

operator[] :: (this: *ImGuiTextBuffer, i: s32) -> u8 #cpp_method #foreign imgui "??AImGuiTextBuffer@@QEBADH@Z";
operator[] :: (this: ImGuiTextBuffer, i: s32) -> u8 #no_context {
    return operator[](*this, i);
}

// [Internal] Key+Value for ImGuiStorage
ImGuiStoragePair :: struct {
    key: ImGuiID;
    union {
        val_i: s32;
        val_f: float;
        val_p: *void;
    }
    Constructor :: (this: *ImGuiStoragePair, _key: ImGuiID, _val: s32) -> void #cpp_method #foreign imgui "??0ImGuiStoragePair@@QEAA@IH@Z";
    Constructor :: (this: *ImGuiStoragePair, _key: ImGuiID, _val: float) -> void #cpp_method #foreign imgui "??0ImGuiStoragePair@@QEAA@IM@Z";
    Constructor :: (this: *ImGuiStoragePair, _key: ImGuiID, _val: *void) -> void #cpp_method #foreign imgui "??0ImGuiStoragePair@@QEAA@IPEAX@Z";
}

// Helper: Key->Value storage
// Typically you don't have to worry about this since a storage is held within each Window.
// We use it to e.g. store collapse state for a tree (Int 0/1)
// This is optimized for efficient lookup (dichotomy into a contiguous buffer) and rare insertion (typically tied to user interactions aka max once a frame)
// You can use it as custom user storage for temporary values. Declare your own storage if, for example:
// - You want to manipulate the open/close state of a particular sub-tree in your interface (tree node uses Int 0/1 to store their state).
// - You want to store custom debug data easily without adding or editing structures in your code (probably not efficient, but convenient)
// Types are NOT stored, so it is up to you to make sure your Key don't collide with different types.
ImGuiStorage :: struct {
    // [Internal]
    Data: ImVector(ImGuiStoragePair);

    // - Get***() functions find pair, never add/allocate. Pairs are sorted so a query is O(log N)
    // - Set***() functions find pair, insertion on demand if missing.
    // - Sorted insertion is costly, paid once. A typical frame shouldn't need to insert any new pair.
    Clear :: (this: *ImGuiStorage) -> void #cpp_method #foreign imgui "?Clear@ImGuiStorage@@QEAAXXZ";
    GetInt :: (this: *ImGuiStorage, key: ImGuiID, default_val: s32 = 0) -> s32 #cpp_method #foreign imgui "?GetInt@ImGuiStorage@@QEBAHIH@Z";
    SetInt :: (this: *ImGuiStorage, key: ImGuiID, val: s32) -> void #cpp_method #foreign imgui "?SetInt@ImGuiStorage@@QEAAXIH@Z";
    GetBool :: (this: *ImGuiStorage, key: ImGuiID, default_val := false) -> bool #cpp_method #foreign imgui "?GetBool@ImGuiStorage@@QEBA_NI_N@Z";
    SetBool :: (this: *ImGuiStorage, key: ImGuiID, val: bool) -> void #cpp_method #foreign imgui "?SetBool@ImGuiStorage@@QEAAXI_N@Z";
    GetFloat :: (this: *ImGuiStorage, key: ImGuiID, default_val: float = 0.0) -> float #cpp_method #foreign imgui "?GetFloat@ImGuiStorage@@QEBAMIM@Z";
    SetFloat :: (this: *ImGuiStorage, key: ImGuiID, val: float) -> void #cpp_method #foreign imgui "?SetFloat@ImGuiStorage@@QEAAXIM@Z";
    GetVoidPtr :: (this: *ImGuiStorage, key: ImGuiID) -> *void #cpp_method #foreign imgui "?GetVoidPtr@ImGuiStorage@@QEBAPEAXI@Z";
    SetVoidPtr :: (this: *ImGuiStorage, key: ImGuiID, val: *void) -> void #cpp_method #foreign imgui "?SetVoidPtr@ImGuiStorage@@QEAAXIPEAX@Z";

    // - Get***Ref() functions finds pair, insert on demand if missing, return pointer. Useful if you intend to do Get+Set.
    // - References are only valid until a new value is added to the storage. Calling a Set***() function or a Get***Ref() function invalidates the pointer.
    // - A typical use case where this is convenient for quick hacking (e.g. add storage during a live Edit&Continue session if you can't modify existing struct)
    //      float* pvar = ImGui::GetFloatRef(key); ImGui::SliderFloat("var", pvar, 0, 100.0f); some_var += *pvar;
    GetIntRef :: (this: *ImGuiStorage, key: ImGuiID, default_val: s32 = 0) -> *s32 #cpp_method #foreign imgui "?GetIntRef@ImGuiStorage@@QEAAPEAHIH@Z";
    GetBoolRef :: (this: *ImGuiStorage, key: ImGuiID, default_val := false) -> *bool #cpp_method #foreign imgui "?GetBoolRef@ImGuiStorage@@QEAAPEA_NI_N@Z";
    GetFloatRef :: (this: *ImGuiStorage, key: ImGuiID, default_val: float = 0.0) -> *float #cpp_method #foreign imgui "?GetFloatRef@ImGuiStorage@@QEAAPEAMIM@Z";
    GetVoidPtrRef :: (this: *ImGuiStorage, key: ImGuiID, default_val: *void = null) -> **void #cpp_method #foreign imgui "?GetVoidPtrRef@ImGuiStorage@@QEAAPEAPEAXIPEAX@Z";

    // Advanced: for quicker full rebuild of a storage (instead of an incremental one), you may add all your contents and then sort once.
    BuildSortByKey :: (this: *ImGuiStorage) -> void #cpp_method #foreign imgui "?BuildSortByKey@ImGuiStorage@@QEAAXXZ";

    // Obsolete: use on your own storage if you know only integer are being stored (open/close all tree nodes)
    SetAllInt :: (this: *ImGuiStorage, val: s32) -> void #cpp_method #foreign imgui "?SetAllInt@ImGuiStorage@@QEAAXH@Z";
}

// Flags for ImGuiListClipper (currently not fully exposed in function calls: a future refactor will likely add this to ImGuiListClipper::Begin function equivalent)
ImGuiListClipperFlags_ :: enum s32 {
    None                  :: 0;
    NoSetTableRowCounters :: 1;

    ImGuiListClipperFlags_None                  :: None;
    ImGuiListClipperFlags_NoSetTableRowCounters :: NoSetTableRowCounters;
}

// Helper: Manually clip large list of items.
// If you have lots evenly spaced items and you have random access to the list, you can perform coarse
// clipping based on visibility to only submit items that are in view.
// The clipper calculates the range of visible items and advance the cursor to compensate for the non-visible items we have skipped.
// (Dear ImGui already clip items based on their bounds but: it needs to first layout the item to do so, and generally
//  fetching/submitting your own data incurs additional cost. Coarse clipping using ImGuiListClipper allows you to easily
//  scale using lists with tens of thousands of items without a problem)
// Usage:
//   ImGuiListClipper clipper;
//   clipper.Begin(1000);         // We have 1000 elements, evenly spaced.
//   while (clipper.Step())
//       for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++)
//           ImGui::Text("line number %d", i);
// Generally what happens is:
// - Clipper lets you process the first element (DisplayStart = 0, DisplayEnd = 1) regardless of it being visible or not.
// - User code submit that one element.
// - Clipper can measure the height of the first element
// - Clipper calculate the actual range of elements to display based on the current clipping rectangle, position the cursor before the first visible element.
// - User code submit visible elements.
// - The clipper also handles various subtleties related to keyboard/gamepad navigation, wrapping etc.
ImGuiListClipper :: struct {
    Ctx:              *ImGuiContext; // Parent UI context
    DisplayStart:     s32; // First item to display, updated by each call to Step()
    DisplayEnd:       s32; // End of items to display (exclusive)
    ItemsCount:       s32; // [Internal] Number of items
    ItemsHeight:      float; // [Internal] Height of item after a first step and item submission can calculate it
    StartPosY:        float64; // [Internal] Cursor position at the time of Begin() or after table frozen rows are all processed
    StartSeekOffsetY: float64; // [Internal] Account for frozen rows in a table and initial loss of precision in very large windows.
    TempData:         *void; // [Internal] Internal data
    Flags:            ImGuiListClipperFlags; // [Internal] Flags, currently not yet well exposed.

    // items_count: Use INT_MAX if you don't know how many items you have (in which case the cursor won't be advanced in the final step, and you can call SeekCursorForItem() manually if you need)
    // items_height: Use -1.0f to be calculated automatically on first step. Otherwise pass in the distance between your items, typically GetTextLineHeightWithSpacing() or GetFrameHeightWithSpacing().
    Constructor :: (this: *ImGuiListClipper) -> void #cpp_method #foreign imgui "??0ImGuiListClipper@@QEAA@XZ";
    Destructor :: (this: *ImGuiListClipper) -> void #cpp_method #foreign imgui "??1ImGuiListClipper@@QEAA@XZ";
    Begin :: (this: *ImGuiListClipper, items_count: s32, items_height: float = -1.0) -> void #cpp_method #foreign imgui "?Begin@ImGuiListClipper@@QEAAXHM@Z";
    End :: (this: *ImGuiListClipper) -> void #cpp_method #foreign imgui "?End@ImGuiListClipper@@QEAAXXZ";
    Step :: (this: *ImGuiListClipper) -> bool #cpp_method #foreign imgui "?Step@ImGuiListClipper@@QEAA_NXZ";

    // Call IncludeItemByIndex() or IncludeItemsByIndex() *BEFORE* first call to Step() if you need a range of items to not be clipped, regardless of their visibility.
    // (Due to alignment / padding of certain items it is possible that an extra item may be included on either end of the display range).
    IncludeItemByIndex :: (this: *ImGuiListClipper, item_index: s32) -> void #cpp_method #foreign imgui "?IncludeItemByIndex@ImGuiListClipper@@QEAAXH@Z";
    IncludeItemsByIndex :: (this: *ImGuiListClipper, item_begin: s32, item_end: s32) -> void #cpp_method #foreign imgui "?IncludeItemsByIndex@ImGuiListClipper@@QEAAXHH@Z";

    // Seek cursor toward given item. This is automatically called while stepping.
    // - The only reason to call this is: you can use ImGuiListClipper::Begin(INT_MAX) if you don't know item count ahead of time.
    // - In this case, after all steps are done, you'll want to call SeekCursorForItem(item_count).
    SeekCursorForItem :: (this: *ImGuiListClipper, item_index: s32) -> void #cpp_method #foreign imgui "?SeekCursorForItem@ImGuiListClipper@@QEAAXH@Z";
}

// Helper: ImColor() implicitly converts colors to either ImU32 (packed 4x1 byte) or ImVec4 (4x1 float)
// Prefer using IM_COL32() macros if you want a guaranteed compile-time ImU32 for usage with ImDrawList API.
// **Avoid storing ImColor! Store either u32 of ImVec4. This is not a full-featured color class. MAY OBSOLETE.
// **None of the ImGui API are using ImColor directly but you can use it as a convenience to pass colors in either ImU32 or ImVec4 formats. Explicitly cast to ImU32 or ImVec4 if needed.
ImColor :: struct {
    Value: ImVec4;

    Constructor :: (this: *ImColor, r: float, g: float, b: float, a: float = 1.0) -> void #cpp_method #foreign imgui "??0ImColor@@QEAA@MMMM@Z";
    Constructor :: (this: *ImColor, col: *ImVec4) -> void #cpp_method #foreign imgui "??0ImColor@@QEAA@AEBUImVec4@@@Z";
    Constructor :: (this: *ImColor, col: ImVec4) #no_context {
        Constructor(this, *col);
    }

    Constructor :: (this: *ImColor, rgba: ImU32) -> void #cpp_method #foreign imgui "??0ImColor@@QEAA@I@Z";

    HSV :: (h: float, s: float, v: float, a: float = 1.0) -> ImColor #cpp_return_type_is_non_pod #foreign imgui "?HSV@ImColor@@SA?AU1@MMMM@Z";
}

// Flags for BeginMultiSelect()
ImGuiMultiSelectFlags_ :: enum s32 {
    None                  :: 0;
    SingleSelect          :: 1;
    NoSelectAll           :: 2;
    NoRangeSelect         :: 4;
    NoAutoSelect          :: 8;
    NoAutoClear           :: 16;
    NoAutoClearOnReselect :: 32;
    BoxSelect1d           :: 64;
    BoxSelect2d           :: 128;
    BoxSelectNoScroll     :: 256;
    ClearOnEscape         :: 512;
    ClearOnClickVoid      :: 1024;
    ScopeWindow           :: 2048;
    ScopeRect             :: 4096;
    SelectOnClick         :: 8192;
    SelectOnClickRelease  :: 16384;

    NavWrapX              :: 65536;

    ImGuiMultiSelectFlags_None                  :: None;
    ImGuiMultiSelectFlags_SingleSelect          :: SingleSelect;
    ImGuiMultiSelectFlags_NoSelectAll           :: NoSelectAll;
    ImGuiMultiSelectFlags_NoRangeSelect         :: NoRangeSelect;
    ImGuiMultiSelectFlags_NoAutoSelect          :: NoAutoSelect;
    ImGuiMultiSelectFlags_NoAutoClear           :: NoAutoClear;
    ImGuiMultiSelectFlags_NoAutoClearOnReselect :: NoAutoClearOnReselect;
    ImGuiMultiSelectFlags_BoxSelect1d           :: BoxSelect1d;
    ImGuiMultiSelectFlags_BoxSelect2d           :: BoxSelect2d;
    ImGuiMultiSelectFlags_BoxSelectNoScroll     :: BoxSelectNoScroll;
    ImGuiMultiSelectFlags_ClearOnEscape         :: ClearOnEscape;
    ImGuiMultiSelectFlags_ClearOnClickVoid      :: ClearOnClickVoid;
    ImGuiMultiSelectFlags_ScopeWindow           :: ScopeWindow;
    ImGuiMultiSelectFlags_ScopeRect             :: ScopeRect;
    ImGuiMultiSelectFlags_SelectOnClick         :: SelectOnClick;
    ImGuiMultiSelectFlags_SelectOnClickRelease  :: SelectOnClickRelease;

    ImGuiMultiSelectFlags_NavWrapX              :: NavWrapX;
}

// Main IO structure returned by BeginMultiSelect()/EndMultiSelect().
// This mainly contains a list of selection requests.
// - Use 'Demo->Tools->Debug Log->Selection' to see requests as they happen.
// - Some fields are only useful if your list is dynamic and allows deletion (getting post-deletion focus/state right is shown in the demo)
// - Below: who reads/writes each fields? 'r'=read, 'w'=write, 'ms'=multi-select code, 'app'=application/user code.
ImGuiMultiSelectIO :: struct {
    Requests:      ImVector(ImGuiSelectionRequest); //  ms:w, app:r     /  ms:w  app:r   // Requests to apply to your selection data.
    RangeSrcItem:  ImGuiSelectionUserData; //  ms:w  app:r     /                // (If using clipper) Begin: Source item (often the first selected item) must never be clipped: use clipper.IncludeItemByIndex() to ensure it is submitted.
    NavIdItem:     ImGuiSelectionUserData; //  ms:w, app:r     /                // (If using deletion) Last known SetNextItemSelectionUserData() value for NavId (if part of submitted items).
    NavIdSelected: bool; //  ms:w, app:r     /        app:r   // (If using deletion) Last known selection state for NavId (if part of submitted items).
    RangeSrcReset: bool; //        app:w     /  ms:r          // (If using deletion) Set before EndMultiSelect() to reset ResetSrcItem (e.g. if deleted selection).
    ItemsCount:    s32; //  ms:w, app:r     /        app:r   // 'int items_count' parameter to BeginMultiSelect() is copied here for convenience, allowing simpler calls to your ApplyRequests handler. Not used internally.
}

// Selection request type
ImGuiSelectionRequestType :: enum s32 {
    None     :: 0;
    SetAll   :: 1;
    SetRange :: 2;

    ImGuiSelectionRequestType_None     :: None;
    ImGuiSelectionRequestType_SetAll   :: SetAll;
    ImGuiSelectionRequestType_SetRange :: SetRange;
}

// Selection request item
ImGuiSelectionRequest :: struct {
    Type:           ImGuiSelectionRequestType; //  ms:w, app:r     /  ms:w, app:r   // Request type. You'll most often receive 1 Clear + 1 SetRange with a single-item range.
    Selected:       bool; //  ms:w, app:r     /  ms:w, app:r   // Parameter for SetAll/SetRange requests (true = select, false = unselect)
    RangeDirection: ImS8; //                  /  ms:w  app:r   // Parameter for SetRange request: +1 when RangeFirstItem comes before RangeLastItem, -1 otherwise. Useful if you want to preserve selection order on a backward Shift+Click.
    RangeFirstItem: ImGuiSelectionUserData; //                  /  ms:w, app:r   // Parameter for SetRange request (this is generally == RangeSrcItem when shift selecting from top to bottom).
    RangeLastItem:  ImGuiSelectionUserData; //                  /  ms:w, app:r   // Parameter for SetRange request (this is generally == RangeSrcItem when shift selecting from bottom to top). Inclusive!
}

// Optional helper to store multi-selection state + apply multi-selection requests.
// - Used by our demos and provided as a convenience to easily implement basic multi-selection.
// - Iterate selection with 'void* it = NULL; ImGuiID id; while (selection.GetNextSelectedItem(&it, &id)) { ... }'
//   Or you can check 'if (Contains(id)) { ... }' for each possible object if their number is not too high to iterate.
// - USING THIS IS NOT MANDATORY. This is only a helper and not a required API.
// To store a multi-selection, in your application you could:
// - Use this helper as a convenience. We use our simple key->value ImGuiStorage as a std::set<ImGuiID> replacement.
// - Use your own external storage: e.g. std::set<MyObjectId>, std::vector<MyObjectId>, interval trees, intrusively stored selection etc.
// In ImGuiSelectionBasicStorage we:
// - always use indices in the multi-selection API (passed to SetNextItemSelectionUserData(), retrieved in ImGuiMultiSelectIO)
// - use the AdapterIndexToStorageId() indirection layer to abstract how persistent selection data is derived from an index.
// - use decently optimized logic to allow queries and insertion of very large selection sets.
// - do not preserve selection order.
// Many combinations are possible depending on how you prefer to store your items and how you prefer to store your selection.
// Large applications are likely to eventually want to get rid of this indirection layer and do their own thing.
// See https://github.com/ocornut/imgui/wiki/Multi-Select for details and pseudo-code using this helper.
ImGuiSelectionBasicStorage :: struct {
    Size:                    s32; //          // Number of selected items, maintained by this helper.
    PreserveOrder:           bool; // = false  // GetNextSelectedItem() will return ordered selection (currently implemented by two additional sorts of selection. Could be improved)
    UserData:                *void; // = NULL   // User data for use by adapter function        // e.g. selection.UserData = (void*)my_items;
    AdapterIndexToStorageId: #type (self: *ImGuiSelectionBasicStorage, idx: s32) -> ImGuiID #c_call; // e.g. selection.AdapterIndexToStorageId = [](ImGuiSelectionBasicStorage* self, int idx) { return ((MyItems**)self->UserData)[idx]->ID; };
    _SelectionOrder:         s32; // [Internal] Increasing counter to store selection order
    _Storage:                ImGuiStorage; // [Internal] Selection set. Think of this as similar to e.g. std::set<ImGuiID>. Prefer not accessing directly: iterate with GetNextSelectedItem().

    // Methods
    Constructor :: (this: *ImGuiSelectionBasicStorage) -> void #cpp_method #foreign imgui "??0ImGuiSelectionBasicStorage@@QEAA@XZ";
    ApplyRequests :: (this: *ImGuiSelectionBasicStorage, ms_io: *ImGuiMultiSelectIO) -> void #cpp_method #foreign imgui "?ApplyRequests@ImGuiSelectionBasicStorage@@QEAAXPEAUImGuiMultiSelectIO@@@Z";
    Contains :: (this: *ImGuiSelectionBasicStorage, id: ImGuiID) -> bool #cpp_method #foreign imgui "?Contains@ImGuiSelectionBasicStorage@@QEBA_NI@Z";
    Clear :: (this: *ImGuiSelectionBasicStorage) -> void #cpp_method #foreign imgui "?Clear@ImGuiSelectionBasicStorage@@QEAAXXZ";
    Swap :: (this: *ImGuiSelectionBasicStorage, r: *ImGuiSelectionBasicStorage) -> void #cpp_method #foreign imgui "?Swap@ImGuiSelectionBasicStorage@@QEAAXAEAU1@@Z";
    SetItemSelected :: (this: *ImGuiSelectionBasicStorage, id: ImGuiID, selected: bool) -> void #cpp_method #foreign imgui "?SetItemSelected@ImGuiSelectionBasicStorage@@QEAAXI_N@Z";
    GetNextSelectedItem :: (this: *ImGuiSelectionBasicStorage, opaque_it: **void, out_id: *ImGuiID) -> bool #cpp_method #foreign imgui "?GetNextSelectedItem@ImGuiSelectionBasicStorage@@QEAA_NPEAPEAXPEAI@Z";
    GetStorageIdFromIndex :: (this: *ImGuiSelectionBasicStorage, idx: s32) -> ImGuiID #cpp_method #foreign imgui "?GetStorageIdFromIndex@ImGuiSelectionBasicStorage@@QEAAIH@Z";
}

// Optional helper to apply multi-selection requests to existing randomly accessible storage.
// Convenient if you want to quickly wire multi-select API on e.g. an array of bool or items storing their own selection state.
ImGuiSelectionExternalStorage :: struct {
    UserData:               *void; // User data for use by adapter function                                // e.g. selection.UserData = (void*)my_items;
    AdapterSetItemSelected: #type (self: *ImGuiSelectionExternalStorage, idx: s32, selected: bool) -> void #c_call; // e.g. AdapterSetItemSelected = [](ImGuiSelectionExternalStorage* self, int idx, bool selected) { ((MyItems**)self->UserData)[idx]->Selected = selected; }

    // Methods
    Constructor :: (this: *ImGuiSelectionExternalStorage) -> void #cpp_method #foreign imgui "??0ImGuiSelectionExternalStorage@@QEAA@XZ";
    ApplyRequests :: (this: *ImGuiSelectionExternalStorage, ms_io: *ImGuiMultiSelectIO) -> void #cpp_method #foreign imgui "?ApplyRequests@ImGuiSelectionExternalStorage@@QEAAXPEAUImGuiMultiSelectIO@@@Z";
}

ImDrawIdx :: u16;

ImDrawCallback :: #type (parent_list: *ImDrawList, cmd: *ImDrawCmd) -> void #c_call;

// Typically, 1 command = 1 GPU draw call (unless command is a callback)
// - VtxOffset: When 'io.BackendFlags & ImGuiBackendFlags_RendererHasVtxOffset' is enabled,
//   this fields allow us to render meshes larger than 64K vertices while keeping 16-bit indices.
//   Backends made for <1.71. will typically ignore the VtxOffset fields.
// - The ClipRect/TexRef/VtxOffset fields must be contiguous as we memcmp() them together (this is asserted for).
ImDrawCmd :: struct {
    ClipRect:               ImVec4; // 4*4  // Clipping rectangle (x1, y1, x2, y2). Subtract ImDrawData->DisplayPos to get clipping rectangle in "viewport" coordinates
    TexRef:                 ImTextureRef; // 16   // Reference to a font/texture atlas (where backend called ImTextureData::SetTexID()) or to a user-provided texture ID (via e.g. ImGui::Image() calls). Both will lead to a ImTextureID value.
    VtxOffset:              u32; // 4    // Start offset in vertex buffer. ImGuiBackendFlags_RendererHasVtxOffset: always 0, otherwise may be >0 to support meshes larger than 64K vertices with 16-bit indices.
    IdxOffset:              u32; // 4    // Start offset in index buffer.
    ElemCount:              u32; // 4    // Number of indices (multiple of 3) to be rendered as triangles. Vertices are stored in the callee ImDrawList's vtx_buffer[] array, indices in idx_buffer[].
    UserCallback:           ImDrawCallback; // 4-8  // If != NULL, call the function instead of rendering the vertices. clip_rect and texture_id will be set normally.
    UserCallbackData:       *void; // 4-8  // Callback user data (when UserCallback != NULL). If called AddCallback() with size == 0, this is a copy of the AddCallback() argument. If called AddCallback() with size > 0, this is pointing to a buffer where data is stored.
    UserCallbackDataSize:   s32; // 4 // Size of callback user data when using storage, otherwise 0.
    UserCallbackDataOffset: s32; // 4 // [Internal] Offset of callback user data when using storage, otherwise -1.

    Constructor :: (this: *ImDrawCmd) -> void #cpp_method #foreign imgui "??0ImDrawCmd@@QEAA@XZ";

    // Since 1.83: returns ImTextureID associated with this draw call. Warning: DO NOT assume this is always same as 'TextureId' (we will change this function for an upcoming feature)
    // Since 1.92: removed ImDrawCmd::TextureId field, the getter function must be used!
    GetTexID :: (this: *ImDrawCmd) -> ImTextureID #cpp_method #foreign imgui "?GetTexID@ImDrawCmd@@QEBA_KXZ";
}

ImDrawVert :: struct {
    pos: ImVec2;
    uv:  ImVec2;
    col: ImU32;
}

// [Internal] For use by ImDrawList
ImDrawCmdHeader :: struct {
    ClipRect:  ImVec4;
    TexRef:    ImTextureRef;
    VtxOffset: u32;
}

// Forward declarations: ImDrawList, ImFontAtlas layer
ImDrawChannel :: struct {
    _CmdBuffer: ImVector(ImDrawCmd);
    _IdxBuffer: ImVector(ImDrawIdx);
}

// Split/Merge functions are used to split the draw list into different layers which can be drawn into out of order.
// This is used by the Columns/Tables API, so items of each column can be batched together in a same draw call.
ImDrawListSplitter :: struct {
    _Current:  s32; // Current channel number (0)
    _Count:    s32; // Number of active channels (1+)
    _Channels: ImVector(ImDrawChannel); // Draw channels (not resized down so _Count might be < Channels.Size)

    Constructor :: (this: *ImDrawListSplitter) -> void #cpp_method #foreign imgui "??0ImDrawListSplitter@@QEAA@XZ";
    Destructor :: (this: *ImDrawListSplitter) -> void #cpp_method #foreign imgui "??1ImDrawListSplitter@@QEAA@XZ";
    Clear :: (this: *ImDrawListSplitter) -> void #cpp_method #foreign imgui "?Clear@ImDrawListSplitter@@QEAAXXZ";
    ClearFreeMemory :: (this: *ImDrawListSplitter) -> void #cpp_method #foreign imgui "?ClearFreeMemory@ImDrawListSplitter@@QEAAXXZ";
    Split :: (this: *ImDrawListSplitter, draw_list: *ImDrawList, count: s32) -> void #cpp_method #foreign imgui "?Split@ImDrawListSplitter@@QEAAXPEAUImDrawList@@H@Z";
    Merge :: (this: *ImDrawListSplitter, draw_list: *ImDrawList) -> void #cpp_method #foreign imgui "?Merge@ImDrawListSplitter@@QEAAXPEAUImDrawList@@@Z";
    SetCurrentChannel :: (this: *ImDrawListSplitter, draw_list: *ImDrawList, channel_idx: s32) -> void #cpp_method #foreign imgui "?SetCurrentChannel@ImDrawListSplitter@@QEAAXPEAUImDrawList@@H@Z";
}

// Flags for ImDrawList functions
// (Legacy: bit 0 must always correspond to ImDrawFlags_Closed to be backward compatible with old API using a bool. Bits 1..3 must be unused)
ImDrawFlags_ :: enum s32 {
    None                    :: 0;
    Closed                  :: 1;
    RoundCornersTopLeft     :: 16;
    RoundCornersTopRight    :: 32;
    RoundCornersBottomLeft  :: 64;
    RoundCornersBottomRight :: 128;
    RoundCornersNone        :: 256;
    RoundCornersTop         :: 48;
    RoundCornersBottom      :: 192;
    RoundCornersLeft        :: 80;
    RoundCornersRight       :: 160;
    RoundCornersAll         :: 240;
    RoundCornersDefault_    :: 240;
    RoundCornersMask_       :: 496;

    ImDrawFlags_None                    :: None;
    ImDrawFlags_Closed                  :: Closed;
    ImDrawFlags_RoundCornersTopLeft     :: RoundCornersTopLeft;
    ImDrawFlags_RoundCornersTopRight    :: RoundCornersTopRight;
    ImDrawFlags_RoundCornersBottomLeft  :: RoundCornersBottomLeft;
    ImDrawFlags_RoundCornersBottomRight :: RoundCornersBottomRight;
    ImDrawFlags_RoundCornersNone        :: RoundCornersNone;
    ImDrawFlags_RoundCornersTop         :: RoundCornersTop;
    ImDrawFlags_RoundCornersBottom      :: RoundCornersBottom;
    ImDrawFlags_RoundCornersLeft        :: RoundCornersLeft;
    ImDrawFlags_RoundCornersRight       :: RoundCornersRight;
    ImDrawFlags_RoundCornersAll         :: RoundCornersAll;
    ImDrawFlags_RoundCornersDefault_    :: RoundCornersDefault_;
    ImDrawFlags_RoundCornersMask_       :: RoundCornersMask_;
}

// Flags for ImDrawList instance. Those are set automatically by ImGui:: functions from ImGuiIO settings, and generally not manipulated directly.
// It is however possible to temporarily alter flags between calls to ImDrawList:: functions.
ImDrawListFlags_ :: enum s32 {
    None                   :: 0;
    AntiAliasedLines       :: 1;
    AntiAliasedLinesUseTex :: 2;
    AntiAliasedFill        :: 4;
    AllowVtxOffset         :: 8;

    ImDrawListFlags_None                   :: None;
    ImDrawListFlags_AntiAliasedLines       :: AntiAliasedLines;
    ImDrawListFlags_AntiAliasedLinesUseTex :: AntiAliasedLinesUseTex;
    ImDrawListFlags_AntiAliasedFill        :: AntiAliasedFill;
    ImDrawListFlags_AllowVtxOffset         :: AllowVtxOffset;
}

// Draw command list
// This is the low-level list of polygons that ImGui:: functions are filling. At the end of the frame,
// all command lists are passed to your ImGuiIO::RenderDrawListFn function for rendering.
// Each dear imgui window contains its own ImDrawList. You can use ImGui::GetWindowDrawList() to
// access the current window draw list and draw custom primitives.
// You can interleave normal ImGui:: calls and adding primitives to the current draw list.
// In single viewport mode, top-left is == GetMainViewport()->Pos (generally 0,0), bottom-right is == GetMainViewport()->Pos+Size (generally io.DisplaySize).
// You are totally free to apply whatever transformation matrix you want to the data (depending on the use of the transformation you may want to apply it to ClipRect as well!)
// Important: Primitives are always added to the list and not culled (culling is done at higher-level by ImGui:: functions), if you use this API a lot consider coarse culling your drawn objects.
ImDrawList :: struct {
    CmdBuffer:         ImVector(ImDrawCmd); // Draw commands. Typically 1 command = 1 GPU draw call, unless the command is a callback.
    IdxBuffer:         ImVector(ImDrawIdx); // Index buffer. Each command consume ImDrawCmd::ElemCount of those
    VtxBuffer:         ImVector(ImDrawVert); // Vertex buffer.
    Flags:             ImDrawListFlags; // Flags, you may poke into these to adjust anti-aliasing settings per-primitive.

    _VtxCurrentIdx:    u32; // [Internal] generally == VtxBuffer.Size unless we are past 64K vertices, in which case this gets reset to 0.
    _Data:             *ImDrawListSharedData; // Pointer to shared draw data (you can use ImGui::GetDrawListSharedData() to get the one from current ImGui context)
    _VtxWritePtr:      *ImDrawVert; // [Internal] point within VtxBuffer.Data after each add command (to avoid using the ImVector<> operators too much)
    _IdxWritePtr:      *ImDrawIdx; // [Internal] point within IdxBuffer.Data after each add command (to avoid using the ImVector<> operators too much)
    _Path:             ImVector(ImVec2); // [Internal] current path building
    _CmdHeader:        ImDrawCmdHeader; // [Internal] template of active commands. Fields should match those of CmdBuffer.back().
    _Splitter:         ImDrawListSplitter; // [Internal] for channels api (note: prefer using your own persistent instance of ImDrawListSplitter!)
    _ClipRectStack:    ImVector(ImVec4); // [Internal]
    _TextureStack:     ImVector(ImTextureRef); // [Internal]
    _CallbacksDataBuf: ImVector(ImU8); // [Internal]
    _FringeScale:      float; // [Internal] anti-alias fringe is scaled by this value, this helps to keep things sharp while zooming at vertex buffer content
    _OwnerName:        *u8; // Pointer to owner window's name for debugging

    // If you want to create ImDrawList instances, pass them ImGui::GetDrawListSharedData().
    // (advanced: you may create and use your own ImDrawListSharedData so you can use ImDrawList without ImGui, but that's more involved)
    Constructor :: (this: *ImDrawList, shared_data: *ImDrawListSharedData) -> void #cpp_method #foreign imgui "??0ImDrawList@@QEAA@PEAUImDrawListSharedData@@@Z";
    Destructor :: (this: *ImDrawList) -> void #cpp_method #foreign imgui "??1ImDrawList@@QEAA@XZ";

    PushClipRect :: (this: *ImDrawList, clip_rect_min: *ImVec2, clip_rect_max: *ImVec2, intersect_with_current_clip_rect := false) -> void #cpp_method #foreign imgui "?PushClipRect@ImDrawList@@QEAAXAEBUImVec2@@0_N@Z";
    PushClipRect :: (this: *ImDrawList, clip_rect_min: ImVec2, clip_rect_max: ImVec2, intersect_with_current_clip_rect := false) #no_context {
        PushClipRect(this, *clip_rect_min, *clip_rect_max, intersect_with_current_clip_rect);
    }
    PushClipRectFullScreen :: (this: *ImDrawList) -> void #cpp_method #foreign imgui "?PushClipRectFullScreen@ImDrawList@@QEAAXXZ";
    PopClipRect :: (this: *ImDrawList) -> void #cpp_method #foreign imgui "?PopClipRect@ImDrawList@@QEAAXXZ";
    PushTexture :: (this: *ImDrawList, tex_ref: ImTextureRef) -> void #cpp_method #foreign imgui "?PushTexture@ImDrawList@@QEAAXUImTextureRef@@@Z";
    PopTexture :: (this: *ImDrawList) -> void #cpp_method #foreign imgui "?PopTexture@ImDrawList@@QEAAXXZ";

    // Primitives
    // - Filled shapes must always use clockwise winding order. The anti-aliasing fringe depends on it. Counter-clockwise shapes will have "inward" anti-aliasing.
    // - For rectangular primitives, "p_min" and "p_max" represent the upper-left and lower-right corners.
    // - For circle primitives, use "num_segments == 0" to automatically calculate tessellation (preferred).
    //   In older versions (until Dear ImGui 1.77) the AddCircle functions defaulted to num_segments == 12.
    //   In future versions we will use textures to provide cheaper and higher-quality circles.
    //   Use AddNgon() and AddNgonFilled() functions if you need to guarantee a specific number of sides.
    AddLine :: (this: *ImDrawList, p1: *ImVec2, p2: *ImVec2, col: ImU32, thickness: float = 1.0) -> void #cpp_method #foreign imgui "?AddLine@ImDrawList@@QEAAXAEBUImVec2@@0IM@Z";
    AddLine :: (this: *ImDrawList, p1: ImVec2, p2: ImVec2, col: ImU32, thickness: float = 1.0) #no_context {
        AddLine(this, *p1, *p2, col, thickness);
    }
    AddRect :: (this: *ImDrawList, p_min: *ImVec2, p_max: *ImVec2, col: ImU32, rounding: float = 0.0, flags: ImDrawFlags = 0, thickness: float = 1.0) -> void #cpp_method #foreign imgui "?AddRect@ImDrawList@@QEAAXAEBUImVec2@@0IMHM@Z";
    AddRect :: (this: *ImDrawList, p_min: ImVec2, p_max: ImVec2, col: ImU32, rounding: float = 0.0, flags: ImDrawFlags = 0, thickness: float = 1.0) #no_context {
        AddRect(this, *p_min, *p_max, col, rounding, flags, thickness);
    }
    AddRectFilled :: (this: *ImDrawList, p_min: *ImVec2, p_max: *ImVec2, col: ImU32, rounding: float = 0.0, flags: ImDrawFlags = 0) -> void #cpp_method #foreign imgui "?AddRectFilled@ImDrawList@@QEAAXAEBUImVec2@@0IMH@Z";
    AddRectFilled :: (this: *ImDrawList, p_min: ImVec2, p_max: ImVec2, col: ImU32, rounding: float = 0.0, flags: ImDrawFlags = 0) #no_context {
        AddRectFilled(this, *p_min, *p_max, col, rounding, flags);
    }
    AddRectFilledMultiColor :: (this: *ImDrawList, p_min: *ImVec2, p_max: *ImVec2, col_upr_left: ImU32, col_upr_right: ImU32, col_bot_right: ImU32, col_bot_left: ImU32) -> void #cpp_method #foreign imgui "?AddRectFilledMultiColor@ImDrawList@@QEAAXAEBUImVec2@@0IIII@Z";
    AddRectFilledMultiColor :: (this: *ImDrawList, p_min: ImVec2, p_max: ImVec2, col_upr_left: ImU32, col_upr_right: ImU32, col_bot_right: ImU32, col_bot_left: ImU32) #no_context {
        AddRectFilledMultiColor(this, *p_min, *p_max, col_upr_left, col_upr_right, col_bot_right, col_bot_left);
    }
    AddQuad :: (this: *ImDrawList, p1: *ImVec2, p2: *ImVec2, p3: *ImVec2, p4: *ImVec2, col: ImU32, thickness: float = 1.0) -> void #cpp_method #foreign imgui "?AddQuad@ImDrawList@@QEAAXAEBUImVec2@@000IM@Z";
    AddQuad :: (this: *ImDrawList, p1: ImVec2, p2: ImVec2, p3: ImVec2, p4: ImVec2, col: ImU32, thickness: float = 1.0) #no_context {
        AddQuad(this, *p1, *p2, *p3, *p4, col, thickness);
    }
    AddQuadFilled :: (this: *ImDrawList, p1: *ImVec2, p2: *ImVec2, p3: *ImVec2, p4: *ImVec2, col: ImU32) -> void #cpp_method #foreign imgui "?AddQuadFilled@ImDrawList@@QEAAXAEBUImVec2@@000I@Z";
    AddQuadFilled :: (this: *ImDrawList, p1: ImVec2, p2: ImVec2, p3: ImVec2, p4: ImVec2, col: ImU32) #no_context {
        AddQuadFilled(this, *p1, *p2, *p3, *p4, col);
    }
    AddTriangle :: (this: *ImDrawList, p1: *ImVec2, p2: *ImVec2, p3: *ImVec2, col: ImU32, thickness: float = 1.0) -> void #cpp_method #foreign imgui "?AddTriangle@ImDrawList@@QEAAXAEBUImVec2@@00IM@Z";
    AddTriangle :: (this: *ImDrawList, p1: ImVec2, p2: ImVec2, p3: ImVec2, col: ImU32, thickness: float = 1.0) #no_context {
        AddTriangle(this, *p1, *p2, *p3, col, thickness);
    }
    AddTriangleFilled :: (this: *ImDrawList, p1: *ImVec2, p2: *ImVec2, p3: *ImVec2, col: ImU32) -> void #cpp_method #foreign imgui "?AddTriangleFilled@ImDrawList@@QEAAXAEBUImVec2@@00I@Z";
    AddTriangleFilled :: (this: *ImDrawList, p1: ImVec2, p2: ImVec2, p3: ImVec2, col: ImU32) #no_context {
        AddTriangleFilled(this, *p1, *p2, *p3, col);
    }
    AddCircle :: (this: *ImDrawList, center: *ImVec2, radius: float, col: ImU32, num_segments: s32 = 0, thickness: float = 1.0) -> void #cpp_method #foreign imgui "?AddCircle@ImDrawList@@QEAAXAEBUImVec2@@MIHM@Z";
    AddCircle :: (this: *ImDrawList, center: ImVec2, radius: float, col: ImU32, num_segments: s32 = 0, thickness: float = 1.0) #no_context {
        AddCircle(this, *center, radius, col, num_segments, thickness);
    }
    AddCircleFilled :: (this: *ImDrawList, center: *ImVec2, radius: float, col: ImU32, num_segments: s32 = 0) -> void #cpp_method #foreign imgui "?AddCircleFilled@ImDrawList@@QEAAXAEBUImVec2@@MIH@Z";
    AddCircleFilled :: (this: *ImDrawList, center: ImVec2, radius: float, col: ImU32, num_segments: s32 = 0) #no_context {
        AddCircleFilled(this, *center, radius, col, num_segments);
    }
    AddNgon :: (this: *ImDrawList, center: *ImVec2, radius: float, col: ImU32, num_segments: s32, thickness: float = 1.0) -> void #cpp_method #foreign imgui "?AddNgon@ImDrawList@@QEAAXAEBUImVec2@@MIHM@Z";
    AddNgon :: (this: *ImDrawList, center: ImVec2, radius: float, col: ImU32, num_segments: s32, thickness: float = 1.0) #no_context {
        AddNgon(this, *center, radius, col, num_segments, thickness);
    }
    AddNgonFilled :: (this: *ImDrawList, center: *ImVec2, radius: float, col: ImU32, num_segments: s32) -> void #cpp_method #foreign imgui "?AddNgonFilled@ImDrawList@@QEAAXAEBUImVec2@@MIH@Z";
    AddNgonFilled :: (this: *ImDrawList, center: ImVec2, radius: float, col: ImU32, num_segments: s32) #no_context {
        AddNgonFilled(this, *center, radius, col, num_segments);
    }
    AddEllipse :: (this: *ImDrawList, center: *ImVec2, radius: *ImVec2, col: ImU32, rot: float = 0.0, num_segments: s32 = 0, thickness: float = 1.0) -> void #cpp_method #foreign imgui "?AddEllipse@ImDrawList@@QEAAXAEBUImVec2@@0IMHM@Z";
    AddEllipse :: (this: *ImDrawList, center: ImVec2, radius: ImVec2, col: ImU32, rot: float = 0.0, num_segments: s32 = 0, thickness: float = 1.0) #no_context {
        AddEllipse(this, *center, *radius, col, rot, num_segments, thickness);
    }
    AddEllipseFilled :: (this: *ImDrawList, center: *ImVec2, radius: *ImVec2, col: ImU32, rot: float = 0.0, num_segments: s32 = 0) -> void #cpp_method #foreign imgui "?AddEllipseFilled@ImDrawList@@QEAAXAEBUImVec2@@0IMH@Z";
    AddEllipseFilled :: (this: *ImDrawList, center: ImVec2, radius: ImVec2, col: ImU32, rot: float = 0.0, num_segments: s32 = 0) #no_context {
        AddEllipseFilled(this, *center, *radius, col, rot, num_segments);
    }
    AddText :: (this: *ImDrawList, pos: *ImVec2, col: ImU32, text_begin: *u8, text_end: *u8 = null) -> void #cpp_method #foreign imgui "?AddText@ImDrawList@@QEAAXAEBUImVec2@@IPEBD1@Z";
    AddText :: (this: *ImDrawList, pos: ImVec2, col: ImU32, text_begin: *u8, text_end: *u8 = null) #no_context {
        AddText(this, *pos, col, text_begin, text_end);
    }
    AddText :: (this: *ImDrawList, font: *ImFont, font_size: float, pos: *ImVec2, col: ImU32, text_begin: *u8, text_end: *u8 = null, wrap_width: float = 0.0, cpu_fine_clip_rect: *ImVec4 = null) -> void #cpp_method #foreign imgui "?AddText@ImDrawList@@QEAAXPEAUImFont@@MAEBUImVec2@@IPEBD2MPEBUImVec4@@@Z";
    AddText :: (this: *ImDrawList, font: *ImFont, font_size: float, pos: ImVec2, col: ImU32, text_begin: *u8, text_end: *u8 = null, wrap_width: float = 0.0, cpu_fine_clip_rect: *ImVec4 = null) #no_context {
        AddText(this, font, font_size, *pos, col, text_begin, text_end, wrap_width, cpu_fine_clip_rect);
    }
    AddBezierCubic :: (this: *ImDrawList, p1: *ImVec2, p2: *ImVec2, p3: *ImVec2, p4: *ImVec2, col: ImU32, thickness: float, num_segments: s32 = 0) -> void #cpp_method #foreign imgui "?AddBezierCubic@ImDrawList@@QEAAXAEBUImVec2@@000IMH@Z";
    AddBezierCubic :: (this: *ImDrawList, p1: ImVec2, p2: ImVec2, p3: ImVec2, p4: ImVec2, col: ImU32, thickness: float, num_segments: s32 = 0) #no_context {
        AddBezierCubic(this, *p1, *p2, *p3, *p4, col, thickness, num_segments);
    }
    AddBezierQuadratic :: (this: *ImDrawList, p1: *ImVec2, p2: *ImVec2, p3: *ImVec2, col: ImU32, thickness: float, num_segments: s32 = 0) -> void #cpp_method #foreign imgui "?AddBezierQuadratic@ImDrawList@@QEAAXAEBUImVec2@@00IMH@Z";
    AddBezierQuadratic :: (this: *ImDrawList, p1: ImVec2, p2: ImVec2, p3: ImVec2, col: ImU32, thickness: float, num_segments: s32 = 0) #no_context {
        AddBezierQuadratic(this, *p1, *p2, *p3, col, thickness, num_segments);
    }

    // General polygon
    // - Only simple polygons are supported by filling functions (no self-intersections, no holes).
    // - Concave polygon fill is more expensive than convex one: it has O(N^2) complexity. Provided as a convenience for the user but not used by the main library.
    AddPolyline :: (this: *ImDrawList, points: *ImVec2, num_points: s32, col: ImU32, flags: ImDrawFlags, thickness: float) -> void #cpp_method #foreign imgui "?AddPolyline@ImDrawList@@QEAAXPEBUImVec2@@HIHM@Z";
    AddConvexPolyFilled :: (this: *ImDrawList, points: *ImVec2, num_points: s32, col: ImU32) -> void #cpp_method #foreign imgui "?AddConvexPolyFilled@ImDrawList@@QEAAXPEBUImVec2@@HI@Z";
    AddConcavePolyFilled :: (this: *ImDrawList, points: *ImVec2, num_points: s32, col: ImU32) -> void #cpp_method #foreign imgui "?AddConcavePolyFilled@ImDrawList@@QEAAXPEBUImVec2@@HI@Z";

    // Image primitives
    // - Read FAQ to understand what ImTextureID/ImTextureRef are.
    // - "p_min" and "p_max" represent the upper-left and lower-right corners of the rectangle.
    // - "uv_min" and "uv_max" represent the normalized texture coordinates to use for those corners. Using (0,0)->(1,1) texture coordinates will generally display the entire texture.
    AddImage :: (this: *ImDrawList, tex_ref: ImTextureRef, p_min: *ImVec2, p_max: *ImVec2, uv_min: *ImVec2, uv_max: *ImVec2, col: ImU32 = 4294967295) -> void #cpp_method #foreign imgui "?AddImage@ImDrawList@@QEAAXUImTextureRef@@AEBUImVec2@@111I@Z";
    AddImage :: (this: *ImDrawList, tex_ref: ImTextureRef, p_min: ImVec2, p_max: ImVec2, uv_min: ImVec2 = ImVec2.{0, 0}, uv_max: ImVec2 = ImVec2.{1, 1}, col: ImU32 = 4294967295) #no_context {
        AddImage(this, tex_ref, *p_min, *p_max, *uv_min, *uv_max, col);
    }
    AddImageQuad :: (this: *ImDrawList, tex_ref: ImTextureRef, p1: *ImVec2, p2: *ImVec2, p3: *ImVec2, p4: *ImVec2, uv1: *ImVec2, uv2: *ImVec2, uv3: *ImVec2, uv4: *ImVec2, col: ImU32 = 4294967295) -> void #cpp_method #foreign imgui "?AddImageQuad@ImDrawList@@QEAAXUImTextureRef@@AEBUImVec2@@1111111I@Z";
    AddImageQuad :: (this: *ImDrawList, tex_ref: ImTextureRef, p1: ImVec2, p2: ImVec2, p3: ImVec2, p4: ImVec2, uv1: ImVec2 = ImVec2.{0, 0}, uv2: ImVec2 = ImVec2.{1, 0}, uv3: ImVec2 = ImVec2.{1, 1}, uv4: ImVec2 = ImVec2.{0, 1}, col: ImU32 = 4294967295) #no_context {
        AddImageQuad(this, tex_ref, *p1, *p2, *p3, *p4, *uv1, *uv2, *uv3, *uv4, col);
    }
    AddImageRounded :: (this: *ImDrawList, tex_ref: ImTextureRef, p_min: *ImVec2, p_max: *ImVec2, uv_min: *ImVec2, uv_max: *ImVec2, col: ImU32, rounding: float, flags: ImDrawFlags = 0) -> void #cpp_method #foreign imgui "?AddImageRounded@ImDrawList@@QEAAXUImTextureRef@@AEBUImVec2@@111IMH@Z";
    AddImageRounded :: (this: *ImDrawList, tex_ref: ImTextureRef, p_min: ImVec2, p_max: ImVec2, uv_min: ImVec2, uv_max: ImVec2, col: ImU32, rounding: float, flags: ImDrawFlags = 0) #no_context {
        AddImageRounded(this, tex_ref, *p_min, *p_max, *uv_min, *uv_max, col, rounding, flags);
    }

    PathLineTo :: (this: *ImDrawList, pos: *ImVec2) -> void #cpp_method #foreign imgui "?PathLineTo@ImDrawList@@QEAAXAEBUImVec2@@@Z";
    PathLineTo :: (this: *ImDrawList, pos: ImVec2) #no_context {
        PathLineTo(this, *pos);
    }

    PathFillConvex :: (this: *ImDrawList, col: ImU32) -> void #cpp_method #foreign imgui "?PathFillConvex@ImDrawList@@QEAAXI@Z";
    PathFillConcave :: (this: *ImDrawList, col: ImU32) -> void #cpp_method #foreign imgui "?PathFillConcave@ImDrawList@@QEAAXI@Z";
    PathStroke :: (this: *ImDrawList, col: ImU32, flags: ImDrawFlags = 0, thickness: float = 1.0) -> void #cpp_method #foreign imgui "?PathStroke@ImDrawList@@QEAAXIHM@Z";
    PathArcTo :: (this: *ImDrawList, center: *ImVec2, radius: float, a_min: float, a_max: float, num_segments: s32 = 0) -> void #cpp_method #foreign imgui "?PathArcTo@ImDrawList@@QEAAXAEBUImVec2@@MMMH@Z";
    PathArcTo :: (this: *ImDrawList, center: ImVec2, radius: float, a_min: float, a_max: float, num_segments: s32 = 0) #no_context {
        PathArcTo(this, *center, radius, a_min, a_max, num_segments);
    }
    PathArcToFast :: (this: *ImDrawList, center: *ImVec2, radius: float, a_min_of_12: s32, a_max_of_12: s32) -> void #cpp_method #foreign imgui "?PathArcToFast@ImDrawList@@QEAAXAEBUImVec2@@MHH@Z";
    PathArcToFast :: (this: *ImDrawList, center: ImVec2, radius: float, a_min_of_12: s32, a_max_of_12: s32) #no_context {
        PathArcToFast(this, *center, radius, a_min_of_12, a_max_of_12);
    }
    PathEllipticalArcTo :: (this: *ImDrawList, center: *ImVec2, radius: *ImVec2, rot: float, a_min: float, a_max: float, num_segments: s32 = 0) -> void #cpp_method #foreign imgui "?PathEllipticalArcTo@ImDrawList@@QEAAXAEBUImVec2@@0MMMH@Z";
    PathEllipticalArcTo :: (this: *ImDrawList, center: ImVec2, radius: ImVec2, rot: float, a_min: float, a_max: float, num_segments: s32 = 0) #no_context {
        PathEllipticalArcTo(this, *center, *radius, rot, a_min, a_max, num_segments);
    }
    PathBezierCubicCurveTo :: (this: *ImDrawList, p2: *ImVec2, p3: *ImVec2, p4: *ImVec2, num_segments: s32 = 0) -> void #cpp_method #foreign imgui "?PathBezierCubicCurveTo@ImDrawList@@QEAAXAEBUImVec2@@00H@Z";
    PathBezierCubicCurveTo :: (this: *ImDrawList, p2: ImVec2, p3: ImVec2, p4: ImVec2, num_segments: s32 = 0) #no_context {
        PathBezierCubicCurveTo(this, *p2, *p3, *p4, num_segments);
    }
    PathBezierQuadraticCurveTo :: (this: *ImDrawList, p2: *ImVec2, p3: *ImVec2, num_segments: s32 = 0) -> void #cpp_method #foreign imgui "?PathBezierQuadraticCurveTo@ImDrawList@@QEAAXAEBUImVec2@@0H@Z";
    PathBezierQuadraticCurveTo :: (this: *ImDrawList, p2: ImVec2, p3: ImVec2, num_segments: s32 = 0) #no_context {
        PathBezierQuadraticCurveTo(this, *p2, *p3, num_segments);
    }
    PathRect :: (this: *ImDrawList, rect_min: *ImVec2, rect_max: *ImVec2, rounding: float = 0.0, flags: ImDrawFlags = 0) -> void #cpp_method #foreign imgui "?PathRect@ImDrawList@@QEAAXAEBUImVec2@@0MH@Z";
    PathRect :: (this: *ImDrawList, rect_min: ImVec2, rect_max: ImVec2, rounding: float = 0.0, flags: ImDrawFlags = 0) #no_context {
        PathRect(this, *rect_min, *rect_max, rounding, flags);
    }

    // Advanced: Draw Callbacks
    // - May be used to alter render state (change sampler, blending, current shader). May be used to emit custom rendering commands (difficult to do correctly, but possible).
    // - Use special ImDrawCallback_ResetRenderState callback to instruct backend to reset its render state to the default.
    // - Your rendering loop must check for 'UserCallback' in ImDrawCmd and call the function instead of rendering triangles. All standard backends are honoring this.
    // - For some backends, the callback may access selected render-states exposed by the backend in a ImGui_ImplXXXX_RenderState structure pointed to by platform_io.Renderer_RenderState.
    // - IMPORTANT: please be mindful of the different level of indirection between using size==0 (copying argument) and using size>0 (copying pointed data into a buffer).
    //   - If userdata_size == 0: we copy/store the 'userdata' argument as-is. It will be available unmodified in ImDrawCmd::UserCallbackData during render.
    //   - If userdata_size > 0,  we copy/store 'userdata_size' bytes pointed to by 'userdata'. We store them in a buffer stored inside the drawlist. ImDrawCmd::UserCallbackData will point inside that buffer so you have to retrieve data from there. Your callback may need to use ImDrawCmd::UserCallbackDataSize if you expect dynamically-sized data.
    //   - Support for userdata_size > 0 was added in v1.91.4, October 2024. So earlier code always only allowed to copy/store a simple void*.
    AddCallback :: (this: *ImDrawList, callback: ImDrawCallback, userdata: *void, userdata_size: u64 = 0) -> void #cpp_method #foreign imgui "?AddCallback@ImDrawList@@QEAAXP6AXPEBU1@PEBUImDrawCmd@@@ZPEAX_K@Z";

    // Advanced: Miscellaneous
    AddDrawCmd :: (this: *ImDrawList) -> void #cpp_method #foreign imgui "?AddDrawCmd@ImDrawList@@QEAAXXZ";
    CloneOutput :: (this: *ImDrawList) -> *ImDrawList #cpp_method #foreign imgui "?CloneOutput@ImDrawList@@QEBAPEAU1@XZ";

    // Advanced: Channels
    // - Use to split render into layers. By switching channels to can render out-of-order (e.g. submit FG primitives before BG primitives)
    // - Use to minimize draw calls (e.g. if going back-and-forth between multiple clipping rectangles, prefer to append into separate channels then merge at the end)
    // - This API shouldn't have been in ImDrawList in the first place!
    //   Prefer using your own persistent instance of ImDrawListSplitter as you can stack them.
    //   Using the ImDrawList::ChannelsXXXX you cannot stack a split over another.
    ChannelsSplit :: (this: *ImDrawList, count: s32) -> void #cpp_method #foreign imgui "?ChannelsSplit@ImDrawList@@QEAAXH@Z";
    ChannelsMerge :: (this: *ImDrawList) -> void #cpp_method #foreign imgui "?ChannelsMerge@ImDrawList@@QEAAXXZ";
    ChannelsSetCurrent :: (this: *ImDrawList, n: s32) -> void #cpp_method #foreign imgui "?ChannelsSetCurrent@ImDrawList@@QEAAXH@Z";

    // Advanced: Primitives allocations
    // - We render triangles (three vertices)
    // - All primitives needs to be reserved via PrimReserve() beforehand.
    PrimReserve :: (this: *ImDrawList, idx_count: s32, vtx_count: s32) -> void #cpp_method #foreign imgui "?PrimReserve@ImDrawList@@QEAAXHH@Z";
    PrimUnreserve :: (this: *ImDrawList, idx_count: s32, vtx_count: s32) -> void #cpp_method #foreign imgui "?PrimUnreserve@ImDrawList@@QEAAXHH@Z";
    PrimRect :: (this: *ImDrawList, a: *ImVec2, b: *ImVec2, col: ImU32) -> void #cpp_method #foreign imgui "?PrimRect@ImDrawList@@QEAAXAEBUImVec2@@0I@Z";
    PrimRect :: (this: *ImDrawList, a: ImVec2, b: ImVec2, col: ImU32) #no_context {
        PrimRect(this, *a, *b, col);
    }
    PrimRectUV :: (this: *ImDrawList, a: *ImVec2, b: *ImVec2, uv_a: *ImVec2, uv_b: *ImVec2, col: ImU32) -> void #cpp_method #foreign imgui "?PrimRectUV@ImDrawList@@QEAAXAEBUImVec2@@000I@Z";
    PrimRectUV :: (this: *ImDrawList, a: ImVec2, b: ImVec2, uv_a: ImVec2, uv_b: ImVec2, col: ImU32) #no_context {
        PrimRectUV(this, *a, *b, *uv_a, *uv_b, col);
    }
    PrimQuadUV :: (this: *ImDrawList, a: *ImVec2, b: *ImVec2, c: *ImVec2, d: *ImVec2, uv_a: *ImVec2, uv_b: *ImVec2, uv_c: *ImVec2, uv_d: *ImVec2, col: ImU32) -> void #cpp_method #foreign imgui "?PrimQuadUV@ImDrawList@@QEAAXAEBUImVec2@@0000000I@Z";
    PrimQuadUV :: (this: *ImDrawList, a: ImVec2, b: ImVec2, c: ImVec2, d: ImVec2, uv_a: ImVec2, uv_b: ImVec2, uv_c: ImVec2, uv_d: ImVec2, col: ImU32) #no_context {
        PrimQuadUV(this, *a, *b, *c, *d, *uv_a, *uv_b, *uv_c, *uv_d, col);
    }
    PrimWriteVtx :: (this: *ImDrawList, pos: *ImVec2, uv: *ImVec2, col: ImU32) -> void #cpp_method #foreign imgui "?PrimWriteVtx@ImDrawList@@QEAAXAEBUImVec2@@0I@Z";
    PrimWriteVtx :: (this: *ImDrawList, pos: ImVec2, uv: ImVec2, col: ImU32) #no_context {
        PrimWriteVtx(this, *pos, *uv, col);
    }
    PrimWriteIdx :: (this: *ImDrawList, idx: ImDrawIdx) -> void #cpp_method #foreign imgui "?PrimWriteIdx@ImDrawList@@QEAAXG@Z";
    PrimVtx :: (this: *ImDrawList, pos: *ImVec2, uv: *ImVec2, col: ImU32) -> void #cpp_method #foreign imgui "?PrimVtx@ImDrawList@@QEAAXAEBUImVec2@@0I@Z";
    PrimVtx :: (this: *ImDrawList, pos: ImVec2, uv: ImVec2, col: ImU32) #no_context {
        PrimVtx(this, *pos, *uv, col);
    }

    // [Internal helpers]
    _SetDrawListSharedData :: (this: *ImDrawList, data: *ImDrawListSharedData) -> void #cpp_method #foreign imgui "?_SetDrawListSharedData@ImDrawList@@QEAAXPEAUImDrawListSharedData@@@Z";
    _ResetForNewFrame :: (this: *ImDrawList) -> void #cpp_method #foreign imgui "?_ResetForNewFrame@ImDrawList@@QEAAXXZ";
    _ClearFreeMemory :: (this: *ImDrawList) -> void #cpp_method #foreign imgui "?_ClearFreeMemory@ImDrawList@@QEAAXXZ";
    _PopUnusedDrawCmd :: (this: *ImDrawList) -> void #cpp_method #foreign imgui "?_PopUnusedDrawCmd@ImDrawList@@QEAAXXZ";
    _TryMergeDrawCmds :: (this: *ImDrawList) -> void #cpp_method #foreign imgui "?_TryMergeDrawCmds@ImDrawList@@QEAAXXZ";
    _OnChangedClipRect :: (this: *ImDrawList) -> void #cpp_method #foreign imgui "?_OnChangedClipRect@ImDrawList@@QEAAXXZ";
    _OnChangedTexture :: (this: *ImDrawList) -> void #cpp_method #foreign imgui "?_OnChangedTexture@ImDrawList@@QEAAXXZ";
    _OnChangedVtxOffset :: (this: *ImDrawList) -> void #cpp_method #foreign imgui "?_OnChangedVtxOffset@ImDrawList@@QEAAXXZ";
    _SetTexture :: (this: *ImDrawList, tex_ref: ImTextureRef) -> void #cpp_method #foreign imgui "?_SetTexture@ImDrawList@@QEAAXUImTextureRef@@@Z";
    _CalcCircleAutoSegmentCount :: (this: *ImDrawList, radius: float) -> s32 #cpp_method #foreign imgui "?_CalcCircleAutoSegmentCount@ImDrawList@@QEBAHM@Z";
    _PathArcToFastEx :: (this: *ImDrawList, center: *ImVec2, radius: float, a_min_sample: s32, a_max_sample: s32, a_step: s32) -> void #cpp_method #foreign imgui "?_PathArcToFastEx@ImDrawList@@QEAAXAEBUImVec2@@MHHH@Z";
    _PathArcToFastEx :: (this: *ImDrawList, center: ImVec2, radius: float, a_min_sample: s32, a_max_sample: s32, a_step: s32) #no_context {
        _PathArcToFastEx(this, *center, radius, a_min_sample, a_max_sample, a_step);
    }
    _PathArcToN :: (this: *ImDrawList, center: *ImVec2, radius: float, a_min: float, a_max: float, num_segments: s32) -> void #cpp_method #foreign imgui "?_PathArcToN@ImDrawList@@QEAAXAEBUImVec2@@MMMH@Z";
    _PathArcToN :: (this: *ImDrawList, center: ImVec2, radius: float, a_min: float, a_max: float, num_segments: s32) #no_context {
        _PathArcToN(this, *center, radius, a_min, a_max, num_segments);
    }
}

// All draw data to render a Dear ImGui frame
// (NB: the style and the naming convention here is a little inconsistent, we currently preserve them for backward compatibility purpose,
// as this is one of the oldest structure exposed by the library! Basically, ImDrawList == CmdList)
ImDrawData :: struct {
    Valid:            bool; // Only valid after Render() is called and before the next NewFrame() is called.
    CmdListsCount:    s32; // == CmdLists.Size. (OBSOLETE: exists for legacy reasons). Number of ImDrawList* to render.
    TotalIdxCount:    s32; // For convenience, sum of all ImDrawList's IdxBuffer.Size
    TotalVtxCount:    s32; // For convenience, sum of all ImDrawList's VtxBuffer.Size
    CmdLists:         ImVector(*ImDrawList); // Array of ImDrawList* to render. The ImDrawLists are owned by ImGuiContext and only pointed to from here.
    DisplayPos:       ImVec2; // Top-left position of the viewport to render (== top-left of the orthogonal projection matrix to use) (== GetMainViewport()->Pos for the main viewport, == (0.0) in most single-viewport applications)
    DisplaySize:      ImVec2; // Size of the viewport to render (== GetMainViewport()->Size for the main viewport, == io.DisplaySize in most single-viewport applications)
    FramebufferScale: ImVec2; // Amount of pixels for each unit of DisplaySize. Copied from viewport->FramebufferScale (== io.DisplayFramebufferScale for main viewport). Generally (1,1) on normal display, (2,2) on OSX with Retina display.
    OwnerViewport:    *ImGuiViewport; // Viewport carrying the ImDrawData instance, might be of use to the renderer (generally not).
    Textures:         *ImVector(*ImTextureData); // List of textures to update. Most of the times the list is shared by all ImDrawData, has only 1 texture and it doesn't need any update. This almost always points to ImGui::GetPlatformIO().Textures[]. May be overriden or set to NULL if you want to manually update textures.

    // Functions
    Constructor :: (this: *ImDrawData) -> void #cpp_method #foreign imgui "??0ImDrawData@@QEAA@XZ";
    Clear :: (this: *ImDrawData) -> void #cpp_method #foreign imgui "?Clear@ImDrawData@@QEAAXXZ";
    AddDrawList :: (this: *ImDrawData, draw_list: *ImDrawList) -> void #cpp_method #foreign imgui "?AddDrawList@ImDrawData@@QEAAXPEAUImDrawList@@@Z";
    DeIndexAllBuffers :: (this: *ImDrawData) -> void #cpp_method #foreign imgui "?DeIndexAllBuffers@ImDrawData@@QEAAXXZ";
    ScaleClipRects :: (this: *ImDrawData, fb_scale: *ImVec2) -> void #cpp_method #foreign imgui "?ScaleClipRects@ImDrawData@@QEAAXAEBUImVec2@@@Z";
    ScaleClipRects :: (this: *ImDrawData, fb_scale: ImVec2) #no_context {
        ScaleClipRects(this, *fb_scale);
    }
}

// We intentionally support a limited amount of texture formats to limit burden on CPU-side code and extension.
// Most standard backends only support RGBA32 but we provide a single channel option for low-resource/embedded systems.
ImTextureFormat :: enum s32 {
    RGBA32 :: 0;
    Alpha8 :: 1;

    ImTextureFormat_RGBA32 :: RGBA32;
    ImTextureFormat_Alpha8 :: Alpha8;
}

// Status of a texture to communicate with Renderer Backend.
ImTextureStatus :: enum s32 {
    OK          :: 0;
    Destroyed   :: 1;
    WantCreate  :: 2;
    WantUpdates :: 3;
    WantDestroy :: 4;

    ImTextureStatus_OK          :: OK;
    ImTextureStatus_Destroyed   :: Destroyed;
    ImTextureStatus_WantCreate  :: WantCreate;
    ImTextureStatus_WantUpdates :: WantUpdates;
    ImTextureStatus_WantDestroy :: WantDestroy;
}

// Coordinates of a rectangle within a texture.
// When a texture is in ImTextureStatus_WantUpdates state, we provide a list of individual rectangles to copy to the graphics system.
// You may use ImTextureData::Updates[] for the list, or ImTextureData::UpdateBox for a single bounding box.
ImTextureRect :: struct {
    x: u16; // Upper-left coordinates of rectangle to update
    y: u16; // Upper-left coordinates of rectangle to update
    w: u16; // Size of rectangle to update (in pixels)
    h: u16; // Size of rectangle to update (in pixels)
}

// Specs and pixel storage for a texture used by Dear ImGui.
// This is only useful for (1) core library and (2) backends. End-user/applications do not need to care about this.
// Renderer Backends will create a GPU-side version of this.
// Why does we store two identifiers: TexID and BackendUserData?
// - ImTextureID    TexID           = lower-level identifier stored in ImDrawCmd. ImDrawCmd can refer to textures not created by the backend, and for which there's no ImTextureData.
// - void*          BackendUserData = higher-level opaque storage for backend own book-keeping. Some backends may have enough with TexID and not need both.
// In columns below: who reads/writes each fields? 'r'=read, 'w'=write, 'core'=main library, 'backend'=renderer backend
ImTextureData :: struct {
    UniqueID:             s32; // w    -   // [DEBUG] Sequential index to facilitate identifying a texture when debugging/printing. Unique per atlas.
    Status:               ImTextureStatus; // rw   rw  // ImTextureStatus_OK/_WantCreate/_WantUpdates/_WantDestroy. Always use SetStatus() to modify!
    BackendUserData:      *void; // -    rw  // Convenience storage for backend. Some backends may have enough with TexID.
    TexID:                ImTextureID; // r    w   // Backend-specific texture identifier. Always use SetTexID() to modify! The identifier will stored in ImDrawCmd::GetTexID() and passed to backend's RenderDrawData function.
    Format:               ImTextureFormat; // w    r   // ImTextureFormat_RGBA32 (default) or ImTextureFormat_Alpha8
    Width:                s32; // w    r   // Texture width
    Height:               s32; // w    r   // Texture height
    BytesPerPixel:        s32; // w    r   // 4 or 1
    Pixels:               *u8; // w    r   // Pointer to buffer holding 'Width*Height' pixels and 'Width*Height*BytesPerPixels' bytes.
    UsedRect:             ImTextureRect; // w    r   // Bounding box encompassing all past and queued Updates[].
    UpdateRect:           ImTextureRect; // w    r   // Bounding box encompassing all queued Updates[].
    Updates:              ImVector(ImTextureRect); // w    r   // Array of individual updates.
    UnusedFrames:         s32; // w    r   // In order to facilitate handling Status==WantDestroy in some backend: this is a count successive frames where the texture was not used. Always >0 when Status==WantDestroy.
    RefCount:             u16; // w    r   // Number of contexts using this texture. Used during backend shutdown.
    UseColors:            bool; // w    r   // Tell whether our texture data is known to use colors (rather than just white + alpha).
    WantDestroyNextFrame: bool; // rw   -   // [Internal] Queued to set ImTextureStatus_WantDestroy next frame. May still be used in the current frame.

    // Functions
    Constructor :: (this: *ImTextureData) -> void #cpp_method #foreign imgui "??0ImTextureData@@QEAA@XZ";
    Destructor :: (this: *ImTextureData) -> void #cpp_method #foreign imgui "??1ImTextureData@@QEAA@XZ";
    Create :: (this: *ImTextureData, format: ImTextureFormat, w: s32, h: s32) -> void #cpp_method #foreign imgui "?Create@ImTextureData@@QEAAXW4ImTextureFormat@@HH@Z";
    DestroyPixels :: (this: *ImTextureData) -> void #cpp_method #foreign imgui "?DestroyPixels@ImTextureData@@QEAAXXZ";

    GetPixelsAt :: (this: *ImTextureData, x: s32, y: s32) -> *void #cpp_method #foreign imgui "?GetPixelsAt@ImTextureData@@QEAAPEAXHH@Z";

    GetPitch :: (this: *ImTextureData) -> s32 #cpp_method #foreign imgui "?GetPitch@ImTextureData@@QEBAHXZ";
    GetTexRef :: (this: *ImTextureData) -> ImTextureRef #cpp_method #cpp_return_type_is_non_pod #foreign imgui "?GetTexRef@ImTextureData@@QEAA?AUImTextureRef@@XZ";
    GetTexID :: (this: *ImTextureData) -> ImTextureID #cpp_method #foreign imgui "?GetTexID@ImTextureData@@QEBA_KXZ";

    // Called by Renderer backend
    // - Call SetTexID() and SetStatus() after honoring texture requests. Never modify TexID and Status directly!
    // - A backend may decide to destroy a texture that we did not request to destroy, which is fine (e.g. freeing resources), but we immediately set the texture back in _WantCreate mode.
    SetTexID :: (this: *ImTextureData, tex_id: ImTextureID) -> void #cpp_method #foreign imgui "?SetTexID@ImTextureData@@QEAAX_K@Z";
    SetStatus :: (this: *ImTextureData, status: ImTextureStatus) -> void #cpp_method #foreign imgui "?SetStatus@ImTextureData@@QEAAXW4ImTextureStatus@@@Z";
}

// A font input/source (we may rename this to ImFontSource in the future)
ImFontConfig :: struct {
    Name:                 [40] u8; // <auto>   // Name (strictly to ease debugging, hence limited size buffer)
    FontData:             *void; //          // TTF/OTF data
    FontDataSize:         s32; //          // TTF/OTF data size
    FontDataOwnedByAtlas: bool; // true     // TTF/OTF data ownership taken by the container ImFontAtlas (will delete memory itself).

    MergeMode:            bool; // false    // Merge into previous ImFont, so you can combine multiple inputs font into one ImFont (e.g. ASCII font + icons + Japanese glyphs). You may want to use GlyphOffset.y when merge font of different heights.
    PixelSnapH:           bool; // false    // Align every glyph AdvanceX to pixel boundaries. Useful e.g. if you are merging a non-pixel aligned font with the default font. If enabled, you can set OversampleH/V to 1.
    PixelSnapV:           bool; // true     // Align Scaled GlyphOffset.y to pixel boundaries.
    OversampleH:          ImS8; // 0 (2)    // Rasterize at higher quality for sub-pixel positioning. 0 == auto == 1 or 2 depending on size. Note the difference between 2 and 3 is minimal. You can reduce this to 1 for large glyphs save memory. Read https://github.com/nothings/stb/blob/master/tests/oversample/README.md for details.
    OversampleV:          ImS8; // 0 (1)    // Rasterize at higher quality for sub-pixel positioning. 0 == auto == 1. This is not really useful as we don't use sub-pixel positions on the Y axis.
    EllipsisChar:         ImWchar; // 0        // Explicitly specify Unicode codepoint of ellipsis character. When fonts are being merged first specified ellipsis will be used.
    SizePixels:           float; //          // Size in pixels for rasterizer (more or less maps to the resulting font height).
    GlyphRanges:          *ImWchar; // NULL     // *LEGACY* THE ARRAY DATA NEEDS TO PERSIST AS LONG AS THE FONT IS ALIVE. Pointer to a user-provided list of Unicode range (2 value per range, values are inclusive, zero-terminated list).
    GlyphExcludeRanges:   *ImWchar; // NULL     // Pointer to a small user-provided list of Unicode ranges (2 value per range, values are inclusive, zero-terminated list). This is very close to GlyphRanges[] but designed to exclude ranges from a font source, when merging fonts with overlapping glyphs. Use "Input Glyphs Overlap Detection Tool" to find about your overlapping ranges.

    GlyphOffset:          ImVec2; // 0, 0     // Offset (in pixels) all glyphs from this font input. Absolute value for default size, other sizes will scale this value.
    GlyphMinAdvanceX:     float; // 0        // Minimum AdvanceX for glyphs, set Min to align font icons, set both Min/Max to enforce mono-space font. Absolute value for default size, other sizes will scale this value.
    GlyphMaxAdvanceX:     float; // FLT_MAX  // Maximum AdvanceX for glyphs
    GlyphExtraAdvanceX:   float; // 0        // Extra spacing (in pixels) between glyphs. Please contact us if you are using this. // FIXME-NEWATLAS: Intentionally unscaled
    FontNo:               ImU32; // 0        // Index of font within TTF/OTF file
    FontLoaderFlags:      u32; // 0        // Settings for custom font builder. THIS IS BUILDER IMPLEMENTATION DEPENDENT. Leave as zero if unsure.

    RasterizerMultiply:   float; // 1.0f     // Linearly brighten (>1.0f) or darken (<1.0f) font output. Brightening small fonts may be a good workaround to make them more readable. This is a silly thing we may remove in the future.
    RasterizerDensity:    float; // 1.0f     // [LEGACY: this only makes sense when ImGuiBackendFlags_RendererHasTextures is not supported] DPI scale multiplier for rasterization. Not altering other font metrics: makes it easy to swap between e.g. a 100% and a 400% fonts for a zooming display, or handle Retina screen. IMPORTANT: If you change this it is expected that you increase/decrease font scale roughly to the inverse of this, otherwise quality may look lowered.

    Flags:                ImFontFlags; // Font flags (don't use just yet, will be exposed in upcoming 1.92.X updates)
    DstFont:              *ImFont; // Target font (as we merging fonts, multiple ImFontConfig may target the same font)
    FontLoader:           *ImFontLoader; // Custom font backend for this source (default source is the one stored in ImFontAtlas)
    FontLoaderData:       *void; // Font loader opaque storage (per font config)

    Constructor :: (this: *ImFontConfig) -> void #cpp_method #foreign imgui "??0ImFontConfig@@QEAA@XZ";
}

// Hold rendering data for one glyph.
// (Note: some language parsers may fail to convert the bitfield members, in this case maybe drop store a single u32 or we can rework this)
ImFontGlyph :: struct {
    __bitfield: u32;
    /* 
        Colored: u32; /* 1 bits */ // Flag to indicate glyph is colored and should generally ignore tinting (make it usable with no shift on little-endian as this is used in loops)
        Visible: u32; /* 1 bits */ // Flag to indicate glyph has no visible pixels (e.g. space). Allow early out when rendering.
        SourceIdx: u32; /* 4 bits */ // Index of source in parent font
        Codepoint: u32; /* 26 bits */ // 0x0000..0x10FFFF;
    */

    AdvanceX:   float; // Horizontal distance to advance cursor/layout position.
    X0:         float; // Glyph corners. Offsets from current cursor/layout position.
    Y0:         float; // Glyph corners. Offsets from current cursor/layout position.
    X1:         float; // Glyph corners. Offsets from current cursor/layout position.
    Y1:         float; // Glyph corners. Offsets from current cursor/layout position.
    U0:         float; // Texture coordinates for the current value of ImFontAtlas->TexRef. Cached equivalent of calling GetCustomRect() with PackId.
    V0:         float; // Texture coordinates for the current value of ImFontAtlas->TexRef. Cached equivalent of calling GetCustomRect() with PackId.
    U1:         float; // Texture coordinates for the current value of ImFontAtlas->TexRef. Cached equivalent of calling GetCustomRect() with PackId.
    V1:         float; // Texture coordinates for the current value of ImFontAtlas->TexRef. Cached equivalent of calling GetCustomRect() with PackId.
    PackId:     s32; // [Internal] ImFontAtlasRectId value (FIXME: Cold data, could be moved elsewhere?)

    Constructor :: (this: *ImFontGlyph) -> void #cpp_method #foreign imgui "??0ImFontGlyph@@QEAA@XZ";
}

// Helper to build glyph ranges from text/string data. Feed your application strings/characters to it then call BuildRanges().
// This is essentially a tightly packed of vector of 64k booleans = 8KB storage.
ImFontGlyphRangesBuilder :: struct {
    UsedChars: ImVector(ImU32); // Store 1-bit per Unicode code point (0=unused, 1=used)

    GetBit :: (this: *ImFontGlyphRangesBuilder, n: u64) -> bool #cpp_method #foreign imgui "?GetBit@ImFontGlyphRangesBuilder@@QEBA_N_K@Z";
    SetBit :: (this: *ImFontGlyphRangesBuilder, n: u64) -> void #cpp_method #foreign imgui "?SetBit@ImFontGlyphRangesBuilder@@QEAAX_K@Z";
    AddChar :: (this: *ImFontGlyphRangesBuilder, c: ImWchar) -> void #cpp_method #foreign imgui "?AddChar@ImFontGlyphRangesBuilder@@QEAAXG@Z";
    AddText :: (this: *ImFontGlyphRangesBuilder, text: *u8, text_end: *u8 = null) -> void #cpp_method #foreign imgui "?AddText@ImFontGlyphRangesBuilder@@QEAAXPEBD0@Z";
    AddRanges :: (this: *ImFontGlyphRangesBuilder, ranges: *ImWchar) -> void #cpp_method #foreign imgui "?AddRanges@ImFontGlyphRangesBuilder@@QEAAXPEBG@Z";
    BuildRanges :: (this: *ImFontGlyphRangesBuilder, out_ranges: *ImVector(ImWchar)) -> void #cpp_method #foreign imgui "?BuildRanges@ImFontGlyphRangesBuilder@@QEAAXPEAU?$ImVector@G@@@Z";
}

// An opaque identifier to a rectangle in the atlas. -1 when invalid.
// The rectangle may move and UV may be invalidated, use GetCustomRect() to retrieve it.
ImFontAtlasRectId :: s32;

// Output of ImFontAtlas::GetCustomRect() when using custom rectangles.
// Those values may not be cached/stored as they are only valid for the current value of atlas->TexRef
// (this is in theory derived from ImTextureRect but we use separate structures for reasons)
ImFontAtlasRect :: struct {
    x:   u16; // Position (in current texture)
    y:   u16; // Position (in current texture)
    w:   u16; // Size
    h:   u16; // Size
    uv0: ImVec2; // UV coordinates (in current texture)
    uv1: ImVec2; // UV coordinates (in current texture)

    Constructor :: (this: *ImFontAtlasRect) -> void #cpp_method #foreign imgui "??0ImFontAtlasRect@@QEAA@XZ";
}

// Flags for ImFontAtlas build
ImFontAtlasFlags_ :: enum s32 {
    None               :: 0;
    NoPowerOfTwoHeight :: 1;
    NoMouseCursors     :: 2;
    NoBakedLines       :: 4;

    ImFontAtlasFlags_None               :: None;
    ImFontAtlasFlags_NoPowerOfTwoHeight :: NoPowerOfTwoHeight;
    ImFontAtlasFlags_NoMouseCursors     :: NoMouseCursors;
    ImFontAtlasFlags_NoBakedLines       :: NoBakedLines;
}

// Load and rasterize multiple TTF/OTF fonts into a same texture. The font atlas will build a single texture holding:
//  - One or more fonts.
//  - Custom graphics data needed to render the shapes needed by Dear ImGui.
//  - Mouse cursor shapes for software cursor rendering (unless setting 'Flags |= ImFontAtlasFlags_NoMouseCursors' in the font atlas).
//  - If you don't call any AddFont*** functions, the default font embedded in the code will be loaded for you.
// It is the rendering backend responsibility to upload texture into your graphics API:
//  - ImGui_ImplXXXX_RenderDrawData() functions generally iterate platform_io->Textures[] to create/update/destroy each ImTextureData instance.
//  - Backend then set ImTextureData's TexID and BackendUserData.
//  - Texture id are passed back to you during rendering to identify the texture. Read FAQ entry about ImTextureID/ImTextureRef for more details.
// Legacy path:
//  - Call Build() + GetTexDataAsAlpha8() or GetTexDataAsRGBA32() to build and retrieve pixels data.
//  - Call SetTexID(my_tex_id); and pass the pointer/identifier to your texture in a format natural to your graphics API.
// Common pitfalls:
// - If you pass a 'glyph_ranges' array to AddFont*** functions, you need to make sure that your array persist up until the
//   atlas is build (when calling GetTexData*** or Build()). We only copy the pointer, not the data.
// - Important: By default, AddFontFromMemoryTTF() takes ownership of the data. Even though we are not writing to it, we will free the pointer on destruction.
//   You can set font_cfg->FontDataOwnedByAtlas=false to keep ownership of your data and it won't be freed,
// - Even though many functions are suffixed with "TTF", OTF data is supported just as well.
// - This is an old API and it is currently awkward for those and various other reasons! We will address them in the future!
ImFontAtlas :: struct {
    Constructor :: (this: *ImFontAtlas) -> void #cpp_method #foreign imgui "??0ImFontAtlas@@QEAA@XZ";
    Destructor :: (this: *ImFontAtlas) -> void #cpp_method #foreign imgui "??1ImFontAtlas@@QEAA@XZ";
    AddFont :: (this: *ImFontAtlas, font_cfg: *ImFontConfig) -> *ImFont #cpp_method #foreign imgui "?AddFont@ImFontAtlas@@QEAAPEAUImFont@@PEBUImFontConfig@@@Z";
    AddFontDefault :: (this: *ImFontAtlas, font_cfg: *ImFontConfig = null) -> *ImFont #cpp_method #foreign imgui "?AddFontDefault@ImFontAtlas@@QEAAPEAUImFont@@PEBUImFontConfig@@@Z";
    AddFontFromFileTTF :: (this: *ImFontAtlas, filename: *u8, size_pixels: float = 0.0, font_cfg: *ImFontConfig = null, glyph_ranges: *ImWchar = null) -> *ImFont #cpp_method #foreign imgui "?AddFontFromFileTTF@ImFontAtlas@@QEAAPEAUImFont@@PEBDMPEBUImFontConfig@@PEBG@Z";
    AddFontFromMemoryTTF :: (this: *ImFontAtlas, font_data: *void, font_data_size: s32, size_pixels: float = 0.0, font_cfg: *ImFontConfig = null, glyph_ranges: *ImWchar = null) -> *ImFont #cpp_method #foreign imgui "?AddFontFromMemoryTTF@ImFontAtlas@@QEAAPEAUImFont@@PEAXHMPEBUImFontConfig@@PEBG@Z";
    AddFontFromMemoryCompressedTTF :: (this: *ImFontAtlas, compressed_font_data: *void, compressed_font_data_size: s32, size_pixels: float = 0.0, font_cfg: *ImFontConfig = null, glyph_ranges: *ImWchar = null) -> *ImFont #cpp_method #foreign imgui "?AddFontFromMemoryCompressedTTF@ImFontAtlas@@QEAAPEAUImFont@@PEBXHMPEBUImFontConfig@@PEBG@Z";
    AddFontFromMemoryCompressedBase85TTF :: (this: *ImFontAtlas, compressed_font_data_base85: *u8, size_pixels: float = 0.0, font_cfg: *ImFontConfig = null, glyph_ranges: *ImWchar = null) -> *ImFont #cpp_method #foreign imgui "?AddFontFromMemoryCompressedBase85TTF@ImFontAtlas@@QEAAPEAUImFont@@PEBDMPEBUImFontConfig@@PEBG@Z";
    RemoveFont :: (this: *ImFontAtlas, font: *ImFont) -> void #cpp_method #foreign imgui "?RemoveFont@ImFontAtlas@@QEAAXPEAUImFont@@@Z";

    Clear :: (this: *ImFontAtlas) -> void #cpp_method #foreign imgui "?Clear@ImFontAtlas@@QEAAXXZ";
    CompactCache :: (this: *ImFontAtlas) -> void #cpp_method #foreign imgui "?CompactCache@ImFontAtlas@@QEAAXXZ";
    SetFontLoader :: (this: *ImFontAtlas, font_loader: *ImFontLoader) -> void #cpp_method #foreign imgui "?SetFontLoader@ImFontAtlas@@QEAAXPEBUImFontLoader@@@Z";

    // As we are transitioning toward a new font system, we expect to obsolete those soon:
    ClearInputData :: (this: *ImFontAtlas) -> void #cpp_method #foreign imgui "?ClearInputData@ImFontAtlas@@QEAAXXZ";
    ClearFonts :: (this: *ImFontAtlas) -> void #cpp_method #foreign imgui "?ClearFonts@ImFontAtlas@@QEAAXXZ";
    ClearTexData :: (this: *ImFontAtlas) -> void #cpp_method #foreign imgui "?ClearTexData@ImFontAtlas@@QEAAXXZ";

    // Legacy path for build atlas + retrieving pixel data.
    // - User is in charge of copying the pixels into graphics memory (e.g. create a texture with your engine). Then store your texture handle with SetTexID().
    // - The pitch is always = Width * BytesPerPixels (1 or 4)
    // - Building in RGBA32 format is provided for convenience and compatibility, but note that unless you manually manipulate or copy color data into
    //   the texture (e.g. when using the AddCustomRect*** api), then the RGB pixels emitted will always be white (~75% of memory/bandwidth waste).
    // - From 1.92 with backends supporting ImGuiBackendFlags_RendererHasTextures:
    //   - Calling Build(), GetTexDataAsAlpha8(), GetTexDataAsRGBA32() is not needed.
    //   - In backend: replace calls to ImFontAtlas::SetTexID() with calls to ImTextureData::SetTexID() after honoring texture creation.
    Build :: (this: *ImFontAtlas) -> bool #cpp_method #foreign imgui "?Build@ImFontAtlas@@QEAA_NXZ";
    GetTexDataAsAlpha8 :: (this: *ImFontAtlas, out_pixels: **u8, out_width: *s32, out_height: *s32, out_bytes_per_pixel: *s32 = null) -> void #cpp_method #foreign imgui "?GetTexDataAsAlpha8@ImFontAtlas@@QEAAXPEAPEAEPEAH11@Z";
    GetTexDataAsRGBA32 :: (this: *ImFontAtlas, out_pixels: **u8, out_width: *s32, out_height: *s32, out_bytes_per_pixel: *s32 = null) -> void #cpp_method #foreign imgui "?GetTexDataAsRGBA32@ImFontAtlas@@QEAAXPEAPEAEPEAH11@Z";

    // Since 1.92: specifying glyph ranges is only useful/necessary if your backend doesn't support ImGuiBackendFlags_RendererHasTextures!
    GetGlyphRangesDefault :: (this: *ImFontAtlas) -> *ImWchar #cpp_method #foreign imgui "?GetGlyphRangesDefault@ImFontAtlas@@QEAAPEBGXZ";

    // Helpers to retrieve list of common Unicode ranges (2 value per range, values are inclusive, zero-terminated list)
    // NB: Make sure that your string are UTF-8 and NOT in your local code page.
    // Read https://github.com/ocornut/imgui/blob/master/docs/FONTS.md/#about-utf-8-encoding for details.
    // NB: Consider using ImFontGlyphRangesBuilder to build glyph ranges from textual data.
    GetGlyphRangesGreek :: (this: *ImFontAtlas) -> *ImWchar #cpp_method #foreign imgui "?GetGlyphRangesGreek@ImFontAtlas@@QEAAPEBGXZ";
    GetGlyphRangesKorean :: (this: *ImFontAtlas) -> *ImWchar #cpp_method #foreign imgui "?GetGlyphRangesKorean@ImFontAtlas@@QEAAPEBGXZ";
    GetGlyphRangesJapanese :: (this: *ImFontAtlas) -> *ImWchar #cpp_method #foreign imgui "?GetGlyphRangesJapanese@ImFontAtlas@@QEAAPEBGXZ";
    GetGlyphRangesChineseFull :: (this: *ImFontAtlas) -> *ImWchar #cpp_method #foreign imgui "?GetGlyphRangesChineseFull@ImFontAtlas@@QEAAPEBGXZ";
    GetGlyphRangesChineseSimplifiedCommon :: (this: *ImFontAtlas) -> *ImWchar #cpp_method #foreign imgui "?GetGlyphRangesChineseSimplifiedCommon@ImFontAtlas@@QEAAPEBGXZ";
    GetGlyphRangesCyrillic :: (this: *ImFontAtlas) -> *ImWchar #cpp_method #foreign imgui "?GetGlyphRangesCyrillic@ImFontAtlas@@QEAAPEBGXZ";
    GetGlyphRangesThai :: (this: *ImFontAtlas) -> *ImWchar #cpp_method #foreign imgui "?GetGlyphRangesThai@ImFontAtlas@@QEAAPEBGXZ";
    GetGlyphRangesVietnamese :: (this: *ImFontAtlas) -> *ImWchar #cpp_method #foreign imgui "?GetGlyphRangesVietnamese@ImFontAtlas@@QEAAPEBGXZ";

    // Register and retrieve custom rectangles
    // - You can request arbitrary rectangles to be packed into the atlas, for your own purpose.
    // - Since 1.92.X, packing is done immediately in the function call (previously packing was done during the Build call)
    // - You can render your pixels into the texture right after calling the AddCustomRect() functions.
    // - VERY IMPORTANT:
    //   - Texture may be created/resized at any time when calling ImGui or ImFontAtlas functions.
    //   - IT WILL INVALIDATE RECTANGLE DATA SUCH AS UV COORDINATES. Always use latest values from GetCustomRect().
    //   - UV coordinates are associated to the current texture identifier aka 'atlas->TexRef'. Both TexRef and UV coordinates are typically changed at the same time.
    // - If you render colored output into your custom rectangles: set 'atlas->TexPixelsUseColors = true' as this may help some backends decide of preferred texture format.
    // - Read docs/FONTS.md for more details about using colorful icons.
    // - Note: this API may be reworked further in order to facilitate supporting e.g. multi-monitor, varying DPI settings.
    // - (Pre-1.92 names) ------------> (1.92 names)
    //   - GetCustomRectByIndex()   --> Use GetCustomRect()
    //   - CalcCustomRectUV()       --> Use GetCustomRect() and read uv0, uv1 fields.
    //   - AddCustomRectRegular()   --> Renamed to AddCustomRect()
    //   - AddCustomRectFontGlyph() --> Prefer using custom ImFontLoader inside ImFontConfig
    //   - ImFontAtlasCustomRect    --> Renamed to ImFontAtlasRect
    AddCustomRect :: (this: *ImFontAtlas, width: s32, height: s32, out_r: *ImFontAtlasRect = null) -> ImFontAtlasRectId #cpp_method #foreign imgui "?AddCustomRect@ImFontAtlas@@QEAAHHHPEAUImFontAtlasRect@@@Z";
    RemoveCustomRect :: (this: *ImFontAtlas, id: ImFontAtlasRectId) -> void #cpp_method #foreign imgui "?RemoveCustomRect@ImFontAtlas@@QEAAXH@Z";
    GetCustomRect :: (this: *ImFontAtlas, id: ImFontAtlasRectId, out_r: *ImFontAtlasRect) -> bool #cpp_method #foreign imgui "?GetCustomRect@ImFontAtlas@@QEBA_NHPEAUImFontAtlasRect@@@Z";

    Flags:               ImFontAtlasFlags; // Build flags (see ImFontAtlasFlags_)
    TexDesiredFormat:    ImTextureFormat; // Desired texture format (default to ImTextureFormat_RGBA32 but may be changed to ImTextureFormat_Alpha8).
    TexGlyphPadding:     s32; // FIXME: Should be called "TexPackPadding". Padding between glyphs within texture in pixels. Defaults to 1. If your rendering method doesn't rely on bilinear filtering you may set this to 0 (will also need to set AntiAliasedLinesUseTex = false).
    TexMinWidth:         s32; // Minimum desired texture width. Must be a power of two. Default to 512.
    TexMinHeight:        s32; // Minimum desired texture height. Must be a power of two. Default to 128.
    TexMaxWidth:         s32; // Maximum desired texture width. Must be a power of two. Default to 8192.
    TexMaxHeight:        s32; // Maximum desired texture height. Must be a power of two. Default to 8192.
    UserData:            *void; // Store your own atlas related user-data (if e.g. you have multiple font atlas).

    union {
        TexRef: ImTextureRef; // Latest texture identifier == TexData->GetTexRef(). // RENAMED TexID to TexRef in 1.92.x
        TexID:  ImTextureRef; // Latest texture identifier == TexData->GetTexRef(). // RENAMED TexID to TexRef in 1.92.x
    }

    TexData:             *ImTextureData; // Latest texture.

    TexList:             ImVector(*ImTextureData); // Texture list (most often TexList.Size == 1). TexData is always == TexList.back(). DO NOT USE DIRECTLY, USE GetDrawData().Textures[]/GetPlatformIO().Textures[] instead!
    Locked:              bool; // Marked as locked during ImGui::NewFrame()..EndFrame() scope if TexUpdates are not supported. Any attempt to modify the atlas will assert.
    RendererHasTextures: bool; // Copy of (BackendFlags & ImGuiBackendFlags_RendererHasTextures) from supporting context.
    TexIsBuilt:          bool; // Set when texture was built matching current font input. Mostly useful for legacy IsBuilt() call.
    TexPixelsUseColors:  bool; // Tell whether our texture data is known to use colors (rather than just alpha channel), in order to help backend select a format or conversion process.
    TexUvScale:          ImVec2; // = (1.0f/TexData->TexWidth, 1.0f/TexData->TexHeight). May change as new texture gets created.
    TexUvWhitePixel:     ImVec2; // Texture coordinates to a white pixel. May change as new texture gets created.
    Fonts:               ImVector(*ImFont); // Hold all the fonts returned by AddFont*. Fonts[0] is the default font upon calling ImGui::NewFrame(), use ImGui::PushFont()/PopFont() to change the current font.
    Sources:             ImVector(ImFontConfig); // Source/configuration data
    TexUvLines:          [33] ImVec4; // UVs for baked anti-aliased lines
    TexNextUniqueID:     s32; // Next value to be stored in TexData->UniqueID
    FontNextUniqueID:    s32; // Next value to be stored in ImFont->FontID
    DrawListSharedDatas: ImVector(*ImDrawListSharedData); // List of users for this atlas. Typically one per Dear ImGui context.
    Builder:             *ImFontAtlasBuilder; // Opaque interface to our data that doesn't need to be public and may be discarded when rebuilding.
    FontLoader:          *ImFontLoader; // Font loader opaque interface (default to use FreeType when IMGUI_ENABLE_FREETYPE is defined, otherwise default to use stb_truetype). Use SetFontLoader() to change this at runtime.
    FontLoaderName:      *u8; // Font loader name (for display e.g. in About box) == FontLoader->Name
    FontLoaderData:      *void; // Font backend opaque storage
    FontLoaderFlags:     u32; // Shared flags (for all fonts) for font loader. THIS IS BUILD IMPLEMENTATION DEPENDENT (e.g. Per-font override is also available in ImFontConfig).
    RefCount:            s32; // Number of contexts using this atlas
    OwnerContext:        *ImGuiContext; // Context which own the atlas will be in charge of updating and destroying it.

    TempRect:            ImFontAtlasRect; // For old GetCustomRectByIndex() API

    AddCustomRectFontGlyph :: (this: *ImFontAtlas, font: *ImFont, codepoint: ImWchar, w: s32, h: s32, advance_x: float, offset: *ImVec2) -> ImFontAtlasRectId #cpp_method #foreign imgui "?AddCustomRectFontGlyph@ImFontAtlas@@QEAAHPEAUImFont@@GHHMAEBUImVec2@@@Z";
    AddCustomRectFontGlyph :: (this: *ImFontAtlas, font: *ImFont, codepoint: ImWchar, w: s32, h: s32, advance_x: float, offset: ImVec2 = ImVec2.{0, 0}) -> ImFontAtlasRectId #no_context {
        return AddCustomRectFontGlyph(this, font, codepoint, w, h, advance_x, *offset);
    }
    AddCustomRectFontGlyphForSize :: (this: *ImFontAtlas, font: *ImFont, font_size: float, codepoint: ImWchar, w: s32, h: s32, advance_x: float, offset: *ImVec2) -> ImFontAtlasRectId #cpp_method #foreign imgui "?AddCustomRectFontGlyphForSize@ImFontAtlas@@QEAAHPEAUImFont@@MGHHMAEBUImVec2@@@Z";
    AddCustomRectFontGlyphForSize :: (this: *ImFontAtlas, font: *ImFont, font_size: float, codepoint: ImWchar, w: s32, h: s32, advance_x: float, offset: ImVec2 = ImVec2.{0, 0}) -> ImFontAtlasRectId #no_context {
        return AddCustomRectFontGlyphForSize(this, font, font_size, codepoint, w, h, advance_x, *offset);
    }
}

// Font runtime data for a given size
// Important: pointers to ImFontBaked are only valid for the current frame.
ImFontBaked :: struct {
    IndexAdvanceX:      ImVector(float); // 12-16 // out // Sparse. Glyphs->AdvanceX in a directly indexable way (cache-friendly for CalcTextSize functions which only this info, and are often bottleneck in large UI).
    FallbackAdvanceX:   float; // 4     // out // FindGlyph(FallbackChar)->AdvanceX
    Size:               float; // 4     // in  // Height of characters/line, set during loading (doesn't change after loading)
    RasterizerDensity:  float; // 4     // in  // Density this is baked at

    IndexLookup:        ImVector(ImU16); // 12-16 // out // Sparse. Index glyphs by Unicode code-point.
    Glyphs:             ImVector(ImFontGlyph); // 12-16 // out // All glyphs.
    FallbackGlyphIndex: s32; // 4     // out // Index of FontFallbackChar

    Ascent:             float; // 4+4   // out // Ascent: distance from top to bottom of e.g. 'A' [0..FontSize] (unscaled)
    Descent:            float; // 4+4   // out // Ascent: distance from top to bottom of e.g. 'A' [0..FontSize] (unscaled)
    __bitfield:         u32;
    /* 
        MetricsTotalSurface: u32; /* 26 bits */ // 3  // out // Total surface in pixels to get an idea of the font rasterization/texture cost (not exact, we approximate the cost of padding between glyphs)
        WantDestroy: u32; /* 1 bits */ // 0  //     // Queued for destroy
        LoadNoFallback: u32; /* 1 bits */ // 0  //     // Disable loading fallback in lower-level calls.
        LoadNoRenderOnLayout: u32; /* 1 bits */ // 0  //     // Enable a two-steps mode where CalcTextSize() calls will load AdvanceX *without* rendering/packing glyphs. Only advantagous if you know that the glyph is unlikely to actually be rendered, otherwise it is slower because we'd do one query on the first CalcTextSize and one query on the first Draw.;
    */

    LastUsedFrame:      s32; // 4  //     // Record of that time this was bounds
    BakedId:            ImGuiID; // 4     //     // Unique ID for this baked storage
    ContainerFont:      *ImFont; // 4-8   // in  // Parent font
    FontLoaderDatas:    *void; // 4-8   //     // Font loader opaque storage (per baked font * sources): single contiguous buffer allocated by imgui, passed to loader.

    // Functions
    Constructor :: (this: *ImFontBaked) -> void #cpp_method #foreign imgui "??0ImFontBaked@@QEAA@XZ";
    ClearOutputData :: (this: *ImFontBaked) -> void #cpp_method #foreign imgui "?ClearOutputData@ImFontBaked@@QEAAXXZ";
    FindGlyph :: (this: *ImFontBaked, c: ImWchar) -> *ImFontGlyph #cpp_method #foreign imgui "?FindGlyph@ImFontBaked@@QEAAPEAUImFontGlyph@@G@Z";
    FindGlyphNoFallback :: (this: *ImFontBaked, c: ImWchar) -> *ImFontGlyph #cpp_method #foreign imgui "?FindGlyphNoFallback@ImFontBaked@@QEAAPEAUImFontGlyph@@G@Z";
    GetCharAdvance :: (this: *ImFontBaked, c: ImWchar) -> float #cpp_method #foreign imgui "?GetCharAdvance@ImFontBaked@@QEAAMG@Z";
    IsGlyphLoaded :: (this: *ImFontBaked, c: ImWchar) -> bool #cpp_method #foreign imgui "?IsGlyphLoaded@ImFontBaked@@QEAA_NG@Z";
}

// Font flags
// (in future versions as we redesign font loading API, this will become more important and better documented. for now please consider this as internal/advanced use)
ImFontFlags_ :: enum s32 {
    None           :: 0;
    NoLoadError    :: 2;
    NoLoadGlyphs   :: 4;
    LockBakedSizes :: 8;

    ImFontFlags_None           :: None;
    ImFontFlags_NoLoadError    :: NoLoadError;
    ImFontFlags_NoLoadGlyphs   :: NoLoadGlyphs;
    ImFontFlags_LockBakedSizes :: LockBakedSizes;
}

// Font runtime data and rendering
// - ImFontAtlas automatically loads a default embedded font for you if you didn't load one manually.
// - Since 1.92.X a font may be rendered as any size! Therefore a font doesn't have one specific size.
// - Use 'font->GetFontBaked(size)' to retrieve the ImFontBaked* corresponding to a given size.
// - If you used g.Font + g.FontSize (which is frequent from the ImGui layer), you can use g.FontBaked as a shortcut, as g.FontBaked == g.Font->GetFontBaked(g.FontSize).
ImFont :: struct {
    LastBaked:                *ImFontBaked; // 4-8   // Cache last bound baked. NEVER USE DIRECTLY. Use GetFontBaked().
    ContainerAtlas:           *ImFontAtlas; // 4-8   // What we have been loaded into.
    Flags:                    ImFontFlags; // 4     // Font flags.
    CurrentRasterizerDensity: float; // Current rasterizer density. This is a varying state of the font.

    FontId:                   ImGuiID; // Unique identifier for the font
    LegacySize:               float; // 4     // in  // Font size passed to AddFont(). Use for old code calling PushFont() expecting to use that size. (use ImGui::GetFontBaked() to get font baked at current bound size).
    Sources:                  ImVector(*ImFontConfig); // 16    // in  // List of sources. Pointers within ContainerAtlas->Sources[]
    EllipsisChar:             ImWchar; // 2-4   // out // Character used for ellipsis rendering ('...').
    FallbackChar:             ImWchar; // 2-4   // out // Character used if a glyph isn't found (U+FFFD, '?')
    Used8kPagesMap:           [1] ImU8; // 1 bytes if ImWchar=ImWchar16, 16 bytes if ImWchar==ImWchar32. Store 1-bit for each block of 4K codepoints that has one active glyph. This is mainly used to facilitate iterations across all used codepoints.
    EllipsisAutoBake:         bool; // 1     //     // Mark when the "..." glyph needs to be generated.
    RemapPairs:               ImGuiStorage; // 16    //     // Remapping pairs when using AddRemapChar(), otherwise empty.

    Scale:                    float; // 4     // in  // Legacy base font scale (~1.0f), multiplied by the per-window font scale which you can adjust with SetWindowFontScale()

    // Methods
    Constructor :: (this: *ImFont) -> void #cpp_method #foreign imgui "??0ImFont@@QEAA@XZ";
    Destructor :: (this: *ImFont) -> void #cpp_method #foreign imgui "??1ImFont@@QEAA@XZ";
    IsGlyphInFont :: (this: *ImFont, c: ImWchar) -> bool #cpp_method #foreign imgui "?IsGlyphInFont@ImFont@@QEAA_NG@Z";
    IsLoaded :: (this: *ImFont) -> bool #cpp_method #foreign imgui "?IsLoaded@ImFont@@QEBA_NXZ";
    GetDebugName :: (this: *ImFont) -> *u8 #cpp_method #foreign imgui "?GetDebugName@ImFont@@QEBAPEBDXZ";

    // [Internal] Don't use!
    // 'max_width' stops rendering after a certain width (could be turned into a 2d size). FLT_MAX to disable.
    // 'wrap_width' enable automatic word-wrapping across multiple lines to fit into given width. 0.0f to disable.
    GetFontBaked :: (this: *ImFont, font_size: float, density: float = -1.0) -> *ImFontBaked #cpp_method #foreign imgui "?GetFontBaked@ImFont@@QEAAPEAUImFontBaked@@MM@Z";
    CalcTextSizeA :: (this: *ImFont, size: float, max_width: float, wrap_width: float, text_begin: *u8, text_end: *u8 = null, out_remaining: **u8 = null) -> ImVec2 #cpp_method #cpp_return_type_is_non_pod #foreign imgui "?CalcTextSizeA@ImFont@@QEAA?AUImVec2@@MMMPEBD0PEAPEBD@Z";
    CalcWordWrapPosition :: (this: *ImFont, size: float, text: *u8, text_end: *u8, wrap_width: float) -> *u8 #cpp_method #foreign imgui "?CalcWordWrapPosition@ImFont@@QEAAPEBDMPEBD0M@Z";
    RenderChar :: (this: *ImFont, draw_list: *ImDrawList, size: float, pos: *ImVec2, col: ImU32, c: ImWchar, cpu_fine_clip: *ImVec4 = null) -> void #cpp_method #foreign imgui "?RenderChar@ImFont@@QEAAXPEAUImDrawList@@MAEBUImVec2@@IGPEBUImVec4@@@Z";
    RenderChar :: (this: *ImFont, draw_list: *ImDrawList, size: float, pos: ImVec2, col: ImU32, c: ImWchar, cpu_fine_clip: *ImVec4 = null) #no_context {
        RenderChar(this, draw_list, size, *pos, col, c, cpu_fine_clip);
    }
    RenderText :: (this: *ImFont, draw_list: *ImDrawList, size: float, pos: *ImVec2, col: ImU32, clip_rect: *ImVec4, text_begin: *u8, text_end: *u8, wrap_width: float = 0.0, flags: ImDrawTextFlags = 0) -> void #cpp_method #foreign imgui "?RenderText@ImFont@@QEAAXPEAUImDrawList@@MAEBUImVec2@@IAEBUImVec4@@PEBD3MH@Z";
    RenderText :: (this: *ImFont, draw_list: *ImDrawList, size: float, pos: ImVec2, col: ImU32, clip_rect: ImVec4, text_begin: *u8, text_end: *u8, wrap_width: float = 0.0, flags: ImDrawTextFlags = 0) #no_context {
        RenderText(this, draw_list, size, *pos, col, *clip_rect, text_begin, text_end, wrap_width, flags);
    }

    // [Internal] Don't use!
    ClearOutputData :: (this: *ImFont) -> void #cpp_method #foreign imgui "?ClearOutputData@ImFont@@QEAAXXZ";
    AddRemapChar :: (this: *ImFont, from_codepoint: ImWchar, to_codepoint: ImWchar) -> void #cpp_method #foreign imgui "?AddRemapChar@ImFont@@QEAAXGG@Z";
    IsGlyphRangeUnused :: (this: *ImFont, c_begin: u32, c_last: u32) -> bool #cpp_method #foreign imgui "?IsGlyphRangeUnused@ImFont@@QEAA_NII@Z";
}

// Flags stored in ImGuiViewport::Flags, giving indications to the platform backends.
ImGuiViewportFlags_ :: enum s32 {
    None              :: 0;
    IsPlatformWindow  :: 1;
    IsPlatformMonitor :: 2;
    OwnedByApp        :: 4;

    ImGuiViewportFlags_None              :: None;
    ImGuiViewportFlags_IsPlatformWindow  :: IsPlatformWindow;
    ImGuiViewportFlags_IsPlatformMonitor :: IsPlatformMonitor;
    ImGuiViewportFlags_OwnedByApp        :: OwnedByApp;
}

// - Currently represents the Platform Window created by the application which is hosting our Dear ImGui windows.
// - In 'docking' branch with multi-viewport enabled, we extend this concept to have multiple active viewports.
// - In the future we will extend this concept further to also represent Platform Monitor and support a "no main platform window" operation mode.
// - About Main Area vs Work Area:
//   - Main Area = entire viewport.
//   - Work Area = entire viewport minus sections used by main menu bars (for platform windows), or by task bar (for platform monitor).
//   - Windows are generally trying to stay within the Work Area of their host viewport.
ImGuiViewport :: struct {
    ID:                ImGuiID; // Unique identifier for the viewport
    Flags:             ImGuiViewportFlags; // See ImGuiViewportFlags_
    Pos:               ImVec2; // Main Area: Position of the viewport (Dear ImGui coordinates are the same as OS desktop/native coordinates)
    Size:              ImVec2; // Main Area: Size of the viewport.
    FramebufferScale:  ImVec2; // Density of the viewport for Retina display (always 1,1 on Windows, may be 2,2 etc on macOS/iOS). This will affect font rasterizer density.
    WorkPos:           ImVec2; // Work Area: Position of the viewport minus task bars, menus bars, status bars (>= Pos)
    WorkSize:          ImVec2; // Work Area: Size of the viewport minus task bars, menu bars, status bars (<= Size)

    PlatformHandle:    *void; // void* to hold higher-level, platform window handle (e.g. HWND, GLFWWindow*, SDL_Window*)
    PlatformHandleRaw: *void; // void* to hold lower-level, platform-native window handle (under Win32 this is expected to be a HWND, unused for other platforms)

    Constructor :: (this: *ImGuiViewport) -> void #cpp_method #foreign imgui "??0ImGuiViewport@@QEAA@XZ";

    // Helpers
    GetCenter :: (this: *ImGuiViewport) -> ImVec2 #cpp_method #cpp_return_type_is_non_pod #foreign imgui "?GetCenter@ImGuiViewport@@QEBA?AUImVec2@@XZ";
}

// Access via ImGui::GetPlatformIO()
ImGuiPlatformIO :: struct {
    Constructor :: (this: *ImGuiPlatformIO) -> void #cpp_method #foreign imgui "??0ImGuiPlatformIO@@QEAA@XZ";

    // Optional: Access OS clipboard
    // (default to use native Win32 clipboard on Windows, otherwise uses a private clipboard. Override to access OS clipboard on other architectures)
    Platform_GetClipboardTextFn:  #type (ctx: *ImGuiContext) -> *u8 #c_call;
    Platform_SetClipboardTextFn:  #type (ctx: *ImGuiContext, text: *u8) -> void #c_call;
    Platform_ClipboardUserData:   *void;

    // Optional: Open link/folder/file in OS Shell
    // (default to use ShellExecuteW() on Windows, system() on Linux/Mac)
    Platform_OpenInShellFn:       #type (ctx: *ImGuiContext, path: *u8) -> bool #c_call;
    Platform_OpenInShellUserData: *void;

    // Optional: Notify OS Input Method Editor of the screen position of your cursor for text input position (e.g. when using Japanese/Chinese IME on Windows)
    // (default to use native imm32 api on Windows)
    Platform_SetImeDataFn:        #type (ctx: *ImGuiContext, viewport: *ImGuiViewport, data: *ImGuiPlatformImeData) -> void #c_call;
    Platform_ImeUserData:         *void;

    Platform_LocaleDecimalPoint:  ImWchar; // '.'

    // Optional: Maximum texture size supported by renderer (used to adjust how we size textures). 0 if not known.
    Renderer_TextureMaxWidth:     s32;
    Renderer_TextureMaxHeight:    s32;

    // Written by some backends during ImGui_ImplXXXX_RenderDrawData() call to point backend_specific ImGui_ImplXXXX_RenderState* structure.
    Renderer_RenderState:         *void;

    Textures:                     ImVector(*ImTextureData); // List of textures used by Dear ImGui (most often 1) + contents of external texture list is automatically appended into this.

    //------------------------------------------------------------------
    // Functions
    //------------------------------------------------------------------
    ClearPlatformHandlers :: (this: *ImGuiPlatformIO) -> void #cpp_method #foreign imgui "?ClearPlatformHandlers@ImGuiPlatformIO@@QEAAXXZ";
    ClearRendererHandlers :: (this: *ImGuiPlatformIO) -> void #cpp_method #foreign imgui "?ClearRendererHandlers@ImGuiPlatformIO@@QEAAXXZ";
}

// (Optional) Support for IME (Input Method Editor) via the platform_io.Platform_SetImeDataFn() function. Handler is called during EndFrame().
ImGuiPlatformImeData :: struct {
    WantVisible:     bool; // A widget wants the IME to be visible.
    WantTextInput:   bool; // A widget wants text input, not necessarily IME to be visible. This is automatically set to the upcoming value of io.WantTextInput.
    InputPos:        ImVec2; // Position of input cursor (for IME).
    InputLineHeight: float; // Line height (for IME).
    ViewportId:      ImGuiID; // ID of platform window/viewport.

    Constructor :: (this: *ImGuiPlatformImeData) -> void #cpp_method #foreign imgui "??0ImGuiPlatformImeData@@QEAA@XZ";
}

//-- OBSOLETED in 1.92.x: ImFontAtlasCustomRect becomes ImTextureRect
// - ImFontAtlasCustomRect::X,Y          --> ImTextureRect::x,y
// - ImFontAtlasCustomRect::Width,Height --> ImTextureRect::w,h
// - ImFontAtlasCustomRect::GlyphColored --> if you need to write to this, instead you can write to 'font->Glyphs.back()->Colored' after calling AddCustomRectFontGlyph()
// We could make ImTextureRect an union to use old names, but 1) this would be confusing 2) the fix is easy 3) ImFontAtlasCustomRect was always a rather esoteric api.
ImFontAtlasCustomRect :: ImFontAtlasRect;

// Follow "Getting Started" link and check examples/ folder to learn about using backends!
ImGui_ImplSDL3_InitForOpenGL :: (window: *SDL_Window, sdl_gl_context: *void) -> bool #foreign imgui "?ImGui_ImplSDL3_InitForOpenGL@@YA_NPEAUSDL_Window@@PEAX@Z";
ImGui_ImplSDL3_InitForVulkan :: (window: *SDL_Window) -> bool #foreign imgui "?ImGui_ImplSDL3_InitForVulkan@@YA_NPEAUSDL_Window@@@Z";
ImGui_ImplSDL3_InitForD3D :: (window: *SDL_Window) -> bool #foreign imgui "?ImGui_ImplSDL3_InitForD3D@@YA_NPEAUSDL_Window@@@Z";
ImGui_ImplSDL3_InitForMetal :: (window: *SDL_Window) -> bool #foreign imgui "?ImGui_ImplSDL3_InitForMetal@@YA_NPEAUSDL_Window@@@Z";
ImGui_ImplSDL3_InitForSDLRenderer :: (window: *SDL_Window, renderer: *SDL_Renderer) -> bool #foreign imgui "?ImGui_ImplSDL3_InitForSDLRenderer@@YA_NPEAUSDL_Window@@PEAUSDL_Renderer@@@Z";
ImGui_ImplSDL3_InitForSDLGPU :: (window: *SDL_Window) -> bool #foreign imgui "?ImGui_ImplSDL3_InitForSDLGPU@@YA_NPEAUSDL_Window@@@Z";
ImGui_ImplSDL3_InitForOther :: (window: *SDL_Window) -> bool #foreign imgui "?ImGui_ImplSDL3_InitForOther@@YA_NPEAUSDL_Window@@@Z";
ImGui_ImplSDL3_Shutdown :: () -> void #foreign imgui "?ImGui_ImplSDL3_Shutdown@@YAXXZ";
ImGui_ImplSDL3_NewFrame :: () -> void #foreign imgui "?ImGui_ImplSDL3_NewFrame@@YAXXZ";
ImGui_ImplSDL3_ProcessEvent :: (event: *SDL_Event) -> bool #foreign imgui "?ImGui_ImplSDL3_ProcessEvent@@YA_NPEBTSDL_Event@@@Z";

// Gamepad selection automatically starts in AutoFirst mode, picking first available SDL_Gamepad. You may override this.
// When using manual mode, caller is responsible for opening/closing gamepad.
ImGui_ImplSDL3_GamepadMode :: enum s32 {
    AutoFirst :: 0;
    AutoAll   :: 1;
    Manual    :: 2;
    ImGui_ImplSDL3_GamepadMode_AutoFirst :: AutoFirst;
    ImGui_ImplSDL3_GamepadMode_AutoAll   :: AutoAll;
    ImGui_ImplSDL3_GamepadMode_Manual    :: Manual;
}
ImGui_ImplSDL3_SetGamepadMode :: (mode: ImGui_ImplSDL3_GamepadMode, manual_gamepads_array: **SDL_Gamepad = null, manual_gamepads_count: s32 = -1) -> void #foreign imgui "?ImGui_ImplSDL3_SetGamepadMode@@YAXW4ImGui_ImplSDL3_GamepadMode@@PEAPEAUSDL_Gamepad@@H@Z";

// Initialization data, for ImGui_ImplSDLGPU_Init()
// - Remember to set ColorTargetFormat to the correct format. If you're rendering to the swapchain, call SDL_GetGPUSwapchainTextureFormat() to query the right value
ImGui_ImplSDLGPU3_InitInfo :: struct {
    Device:               *SDL_GPUDevice = null;
    ColorTargetFormat:    SDL_GPUTextureFormat = .INVALID;
    MSAASamples:          SDL_GPUSampleCount = ._1;
    SwapchainComposition: SDL_GPUSwapchainComposition = .SDR; // Only used in multi-viewports mode.
    PresentMode:          SDL_GPUPresentMode = .VSYNC; // Only used in multi-viewports mode.
}

// Follow "Getting Started" link and check examples/ folder to learn about using backends!
ImGui_ImplSDLGPU3_Init :: (info: *ImGui_ImplSDLGPU3_InitInfo) -> bool #foreign imgui "?ImGui_ImplSDLGPU3_Init@@YA_NPEAUImGui_ImplSDLGPU3_InitInfo@@@Z";
ImGui_ImplSDLGPU3_Shutdown :: () -> void #foreign imgui "?ImGui_ImplSDLGPU3_Shutdown@@YAXXZ";
ImGui_ImplSDLGPU3_NewFrame :: () -> void #foreign imgui "?ImGui_ImplSDLGPU3_NewFrame@@YAXXZ";
ImGui_ImplSDLGPU3_PrepareDrawData :: (draw_data: *ImDrawData, command_buffer: *SDL_GPUCommandBuffer) -> void #foreign imgui "?ImGui_ImplSDLGPU3_PrepareDrawData@@YAXPEAUImDrawData@@PEAUSDL_GPUCommandBuffer@@@Z";
ImGui_ImplSDLGPU3_RenderDrawData :: (draw_data: *ImDrawData, command_buffer: *SDL_GPUCommandBuffer, render_pass: *SDL_GPURenderPass, pipeline: *SDL_GPUGraphicsPipeline = null) -> void #foreign imgui "?ImGui_ImplSDLGPU3_RenderDrawData@@YAXPEAUImDrawData@@PEAUSDL_GPUCommandBuffer@@PEAUSDL_GPURenderPass@@PEAUSDL_GPUGraphicsPipeline@@@Z";

// Use if you want to reset your rendering device without losing Dear ImGui state.
ImGui_ImplSDLGPU3_CreateDeviceObjects :: () -> void #foreign imgui "?ImGui_ImplSDLGPU3_CreateDeviceObjects@@YAXXZ";
ImGui_ImplSDLGPU3_DestroyDeviceObjects :: () -> void #foreign imgui "?ImGui_ImplSDLGPU3_DestroyDeviceObjects@@YAXXZ";

// (Advanced) Use e.g. if you need to precisely control the timing of texture updates (e.g. for staged rendering), by setting ImDrawData::Textures = NULL to handle this manually.
ImGui_ImplSDLGPU3_UpdateTexture :: (tex: *ImTextureData) -> void #foreign imgui "?ImGui_ImplSDLGPU3_UpdateTexture@@YAXPEAUImTextureData@@@Z";

// [BETA] Selected render state data shared with callbacks.
// This is temporarily stored in GetPlatformIO().Renderer_RenderState during the ImGui_ImplSDLGPU3_RenderDrawData() call.
// (Please open an issue if you feel you need access to more data)
ImGui_ImplSDLGPU3_RenderState :: struct {
    Device:         *SDL_GPUDevice;
    SamplerDefault: *SDL_GPUSampler; // Default sampler (bilinear filtering)
    SamplerCurrent: *SDL_GPUSampler; // Current sampler (may be changed by callback)
}

#scope_file

#import "Basic"; // For assert, push_context

user32 :: #library,system "user32";
imgui :: #library,no_dll "lib/imgui";

#run {
    {
        info := type_info(ImVec2);
        for info.members {
            if it.name == {
                case "x";
                    assert(it.offset_in_bytes == 0, "ImVec2.x has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImVec2.x has unexpected size % instead of 4", it.type.runtime_size);
                case "y";
                    assert(it.offset_in_bytes == 4, "ImVec2.y has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImVec2.y has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImVec2) == 8, "ImVec2 has size % instead of 8", size_of(ImVec2));
    }

    {
        info := type_info(ImVec4);
        for info.members {
            if it.name == {
                case "x";
                    assert(it.offset_in_bytes == 0, "ImVec4.x has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImVec4.x has unexpected size % instead of 4", it.type.runtime_size);
                case "y";
                    assert(it.offset_in_bytes == 4, "ImVec4.y has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImVec4.y has unexpected size % instead of 4", it.type.runtime_size);
                case "z";
                    assert(it.offset_in_bytes == 8, "ImVec4.z has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImVec4.z has unexpected size % instead of 4", it.type.runtime_size);
                case "w";
                    assert(it.offset_in_bytes == 12, "ImVec4.w has unexpected offset % instead of 12", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImVec4.w has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImVec4) == 16, "ImVec4 has size % instead of 16", size_of(ImVec4));
    }

    {
        info := type_info(ImTextureRef);
        for info.members {
            if it.name == {
                case "_TexData";
                    assert(it.offset_in_bytes == 0, "ImTextureRef._TexData has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImTextureRef._TexData has unexpected size % instead of 8", it.type.runtime_size);
                case "_TexID";
                    assert(it.offset_in_bytes == 8, "ImTextureRef._TexID has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImTextureRef._TexID has unexpected size % instead of 8", it.type.runtime_size);
            }
        }
        assert(size_of(ImTextureRef) == 16, "ImTextureRef has size % instead of 16", size_of(ImTextureRef));
    }

    {
        info := type_info(ImGuiTableSortSpecs);
        for info.members {
            if it.name == {
                case "Specs";
                    assert(it.offset_in_bytes == 0, "ImGuiTableSortSpecs.Specs has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiTableSortSpecs.Specs has unexpected size % instead of 8", it.type.runtime_size);
                case "SpecsCount";
                    assert(it.offset_in_bytes == 8, "ImGuiTableSortSpecs.SpecsCount has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiTableSortSpecs.SpecsCount has unexpected size % instead of 4", it.type.runtime_size);
                case "SpecsDirty";
                    assert(it.offset_in_bytes == 12, "ImGuiTableSortSpecs.SpecsDirty has unexpected offset % instead of 12", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiTableSortSpecs.SpecsDirty has unexpected size % instead of 1", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiTableSortSpecs) == 16, "ImGuiTableSortSpecs has size % instead of 16", size_of(ImGuiTableSortSpecs));
    }

    {
        info := type_info(ImGuiTableColumnSortSpecs);
        for info.members {
            if it.name == {
                case "ColumnUserID";
                    assert(it.offset_in_bytes == 0, "ImGuiTableColumnSortSpecs.ColumnUserID has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiTableColumnSortSpecs.ColumnUserID has unexpected size % instead of 4", it.type.runtime_size);
                case "ColumnIndex";
                    assert(it.offset_in_bytes == 4, "ImGuiTableColumnSortSpecs.ColumnIndex has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImGuiTableColumnSortSpecs.ColumnIndex has unexpected size % instead of 2", it.type.runtime_size);
                case "SortOrder";
                    assert(it.offset_in_bytes == 6, "ImGuiTableColumnSortSpecs.SortOrder has unexpected offset % instead of 6", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImGuiTableColumnSortSpecs.SortOrder has unexpected size % instead of 2", it.type.runtime_size);
                case "SortDirection";
                    assert(it.offset_in_bytes == 8, "ImGuiTableColumnSortSpecs.SortDirection has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiTableColumnSortSpecs.SortDirection has unexpected size % instead of 1", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiTableColumnSortSpecs) == 12, "ImGuiTableColumnSortSpecs has size % instead of 12", size_of(ImGuiTableColumnSortSpecs));
    }

    assert(size_of(ImNewWrapper) == 1, "ImNewWrapper has size % instead of 1", size_of(ImNewWrapper));

    {
        info := type_info(ImGuiStyle);
        for info.members {
            if it.name == {
                case "FontSizeBase";
                    assert(it.offset_in_bytes == 0, "ImGuiStyle.FontSizeBase has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.FontSizeBase has unexpected size % instead of 4", it.type.runtime_size);
                case "FontScaleMain";
                    assert(it.offset_in_bytes == 4, "ImGuiStyle.FontScaleMain has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.FontScaleMain has unexpected size % instead of 4", it.type.runtime_size);
                case "FontScaleDpi";
                    assert(it.offset_in_bytes == 8, "ImGuiStyle.FontScaleDpi has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.FontScaleDpi has unexpected size % instead of 4", it.type.runtime_size);
                case "Alpha";
                    assert(it.offset_in_bytes == 12, "ImGuiStyle.Alpha has unexpected offset % instead of 12", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.Alpha has unexpected size % instead of 4", it.type.runtime_size);
                case "DisabledAlpha";
                    assert(it.offset_in_bytes == 16, "ImGuiStyle.DisabledAlpha has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.DisabledAlpha has unexpected size % instead of 4", it.type.runtime_size);
                case "WindowPadding";
                    assert(it.offset_in_bytes == 20, "ImGuiStyle.WindowPadding has unexpected offset % instead of 20", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiStyle.WindowPadding has unexpected size % instead of 8", it.type.runtime_size);
                case "WindowRounding";
                    assert(it.offset_in_bytes == 28, "ImGuiStyle.WindowRounding has unexpected offset % instead of 28", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.WindowRounding has unexpected size % instead of 4", it.type.runtime_size);
                case "WindowBorderSize";
                    assert(it.offset_in_bytes == 32, "ImGuiStyle.WindowBorderSize has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.WindowBorderSize has unexpected size % instead of 4", it.type.runtime_size);
                case "WindowBorderHoverPadding";
                    assert(it.offset_in_bytes == 36, "ImGuiStyle.WindowBorderHoverPadding has unexpected offset % instead of 36", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.WindowBorderHoverPadding has unexpected size % instead of 4", it.type.runtime_size);
                case "WindowMinSize";
                    assert(it.offset_in_bytes == 40, "ImGuiStyle.WindowMinSize has unexpected offset % instead of 40", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiStyle.WindowMinSize has unexpected size % instead of 8", it.type.runtime_size);
                case "WindowTitleAlign";
                    assert(it.offset_in_bytes == 48, "ImGuiStyle.WindowTitleAlign has unexpected offset % instead of 48", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiStyle.WindowTitleAlign has unexpected size % instead of 8", it.type.runtime_size);
                case "WindowMenuButtonPosition";
                    assert(it.offset_in_bytes == 56, "ImGuiStyle.WindowMenuButtonPosition has unexpected offset % instead of 56", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.WindowMenuButtonPosition has unexpected size % instead of 4", it.type.runtime_size);
                case "ChildRounding";
                    assert(it.offset_in_bytes == 60, "ImGuiStyle.ChildRounding has unexpected offset % instead of 60", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.ChildRounding has unexpected size % instead of 4", it.type.runtime_size);
                case "ChildBorderSize";
                    assert(it.offset_in_bytes == 64, "ImGuiStyle.ChildBorderSize has unexpected offset % instead of 64", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.ChildBorderSize has unexpected size % instead of 4", it.type.runtime_size);
                case "PopupRounding";
                    assert(it.offset_in_bytes == 68, "ImGuiStyle.PopupRounding has unexpected offset % instead of 68", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.PopupRounding has unexpected size % instead of 4", it.type.runtime_size);
                case "PopupBorderSize";
                    assert(it.offset_in_bytes == 72, "ImGuiStyle.PopupBorderSize has unexpected offset % instead of 72", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.PopupBorderSize has unexpected size % instead of 4", it.type.runtime_size);
                case "FramePadding";
                    assert(it.offset_in_bytes == 76, "ImGuiStyle.FramePadding has unexpected offset % instead of 76", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiStyle.FramePadding has unexpected size % instead of 8", it.type.runtime_size);
                case "FrameRounding";
                    assert(it.offset_in_bytes == 84, "ImGuiStyle.FrameRounding has unexpected offset % instead of 84", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.FrameRounding has unexpected size % instead of 4", it.type.runtime_size);
                case "FrameBorderSize";
                    assert(it.offset_in_bytes == 88, "ImGuiStyle.FrameBorderSize has unexpected offset % instead of 88", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.FrameBorderSize has unexpected size % instead of 4", it.type.runtime_size);
                case "ItemSpacing";
                    assert(it.offset_in_bytes == 92, "ImGuiStyle.ItemSpacing has unexpected offset % instead of 92", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiStyle.ItemSpacing has unexpected size % instead of 8", it.type.runtime_size);
                case "ItemInnerSpacing";
                    assert(it.offset_in_bytes == 100, "ImGuiStyle.ItemInnerSpacing has unexpected offset % instead of 100", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiStyle.ItemInnerSpacing has unexpected size % instead of 8", it.type.runtime_size);
                case "CellPadding";
                    assert(it.offset_in_bytes == 108, "ImGuiStyle.CellPadding has unexpected offset % instead of 108", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiStyle.CellPadding has unexpected size % instead of 8", it.type.runtime_size);
                case "TouchExtraPadding";
                    assert(it.offset_in_bytes == 116, "ImGuiStyle.TouchExtraPadding has unexpected offset % instead of 116", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiStyle.TouchExtraPadding has unexpected size % instead of 8", it.type.runtime_size);
                case "IndentSpacing";
                    assert(it.offset_in_bytes == 124, "ImGuiStyle.IndentSpacing has unexpected offset % instead of 124", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.IndentSpacing has unexpected size % instead of 4", it.type.runtime_size);
                case "ColumnsMinSpacing";
                    assert(it.offset_in_bytes == 128, "ImGuiStyle.ColumnsMinSpacing has unexpected offset % instead of 128", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.ColumnsMinSpacing has unexpected size % instead of 4", it.type.runtime_size);
                case "ScrollbarSize";
                    assert(it.offset_in_bytes == 132, "ImGuiStyle.ScrollbarSize has unexpected offset % instead of 132", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.ScrollbarSize has unexpected size % instead of 4", it.type.runtime_size);
                case "ScrollbarRounding";
                    assert(it.offset_in_bytes == 136, "ImGuiStyle.ScrollbarRounding has unexpected offset % instead of 136", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.ScrollbarRounding has unexpected size % instead of 4", it.type.runtime_size);
                case "ScrollbarPadding";
                    assert(it.offset_in_bytes == 140, "ImGuiStyle.ScrollbarPadding has unexpected offset % instead of 140", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.ScrollbarPadding has unexpected size % instead of 4", it.type.runtime_size);
                case "GrabMinSize";
                    assert(it.offset_in_bytes == 144, "ImGuiStyle.GrabMinSize has unexpected offset % instead of 144", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.GrabMinSize has unexpected size % instead of 4", it.type.runtime_size);
                case "GrabRounding";
                    assert(it.offset_in_bytes == 148, "ImGuiStyle.GrabRounding has unexpected offset % instead of 148", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.GrabRounding has unexpected size % instead of 4", it.type.runtime_size);
                case "LogSliderDeadzone";
                    assert(it.offset_in_bytes == 152, "ImGuiStyle.LogSliderDeadzone has unexpected offset % instead of 152", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.LogSliderDeadzone has unexpected size % instead of 4", it.type.runtime_size);
                case "ImageBorderSize";
                    assert(it.offset_in_bytes == 156, "ImGuiStyle.ImageBorderSize has unexpected offset % instead of 156", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.ImageBorderSize has unexpected size % instead of 4", it.type.runtime_size);
                case "TabRounding";
                    assert(it.offset_in_bytes == 160, "ImGuiStyle.TabRounding has unexpected offset % instead of 160", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.TabRounding has unexpected size % instead of 4", it.type.runtime_size);
                case "TabBorderSize";
                    assert(it.offset_in_bytes == 164, "ImGuiStyle.TabBorderSize has unexpected offset % instead of 164", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.TabBorderSize has unexpected size % instead of 4", it.type.runtime_size);
                case "TabMinWidthBase";
                    assert(it.offset_in_bytes == 168, "ImGuiStyle.TabMinWidthBase has unexpected offset % instead of 168", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.TabMinWidthBase has unexpected size % instead of 4", it.type.runtime_size);
                case "TabMinWidthShrink";
                    assert(it.offset_in_bytes == 172, "ImGuiStyle.TabMinWidthShrink has unexpected offset % instead of 172", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.TabMinWidthShrink has unexpected size % instead of 4", it.type.runtime_size);
                case "TabCloseButtonMinWidthSelected";
                    assert(it.offset_in_bytes == 176, "ImGuiStyle.TabCloseButtonMinWidthSelected has unexpected offset % instead of 176", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.TabCloseButtonMinWidthSelected has unexpected size % instead of 4", it.type.runtime_size);
                case "TabCloseButtonMinWidthUnselected";
                    assert(it.offset_in_bytes == 180, "ImGuiStyle.TabCloseButtonMinWidthUnselected has unexpected offset % instead of 180", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.TabCloseButtonMinWidthUnselected has unexpected size % instead of 4", it.type.runtime_size);
                case "TabBarBorderSize";
                    assert(it.offset_in_bytes == 184, "ImGuiStyle.TabBarBorderSize has unexpected offset % instead of 184", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.TabBarBorderSize has unexpected size % instead of 4", it.type.runtime_size);
                case "TabBarOverlineSize";
                    assert(it.offset_in_bytes == 188, "ImGuiStyle.TabBarOverlineSize has unexpected offset % instead of 188", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.TabBarOverlineSize has unexpected size % instead of 4", it.type.runtime_size);
                case "TableAngledHeadersAngle";
                    assert(it.offset_in_bytes == 192, "ImGuiStyle.TableAngledHeadersAngle has unexpected offset % instead of 192", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.TableAngledHeadersAngle has unexpected size % instead of 4", it.type.runtime_size);
                case "TableAngledHeadersTextAlign";
                    assert(it.offset_in_bytes == 196, "ImGuiStyle.TableAngledHeadersTextAlign has unexpected offset % instead of 196", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiStyle.TableAngledHeadersTextAlign has unexpected size % instead of 8", it.type.runtime_size);
                case "TreeLinesFlags";
                    assert(it.offset_in_bytes == 204, "ImGuiStyle.TreeLinesFlags has unexpected offset % instead of 204", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.TreeLinesFlags has unexpected size % instead of 4", it.type.runtime_size);
                case "TreeLinesSize";
                    assert(it.offset_in_bytes == 208, "ImGuiStyle.TreeLinesSize has unexpected offset % instead of 208", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.TreeLinesSize has unexpected size % instead of 4", it.type.runtime_size);
                case "TreeLinesRounding";
                    assert(it.offset_in_bytes == 212, "ImGuiStyle.TreeLinesRounding has unexpected offset % instead of 212", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.TreeLinesRounding has unexpected size % instead of 4", it.type.runtime_size);
                case "ColorButtonPosition";
                    assert(it.offset_in_bytes == 216, "ImGuiStyle.ColorButtonPosition has unexpected offset % instead of 216", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.ColorButtonPosition has unexpected size % instead of 4", it.type.runtime_size);
                case "ButtonTextAlign";
                    assert(it.offset_in_bytes == 220, "ImGuiStyle.ButtonTextAlign has unexpected offset % instead of 220", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiStyle.ButtonTextAlign has unexpected size % instead of 8", it.type.runtime_size);
                case "SelectableTextAlign";
                    assert(it.offset_in_bytes == 228, "ImGuiStyle.SelectableTextAlign has unexpected offset % instead of 228", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiStyle.SelectableTextAlign has unexpected size % instead of 8", it.type.runtime_size);
                case "SeparatorTextBorderSize";
                    assert(it.offset_in_bytes == 236, "ImGuiStyle.SeparatorTextBorderSize has unexpected offset % instead of 236", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.SeparatorTextBorderSize has unexpected size % instead of 4", it.type.runtime_size);
                case "SeparatorTextAlign";
                    assert(it.offset_in_bytes == 240, "ImGuiStyle.SeparatorTextAlign has unexpected offset % instead of 240", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiStyle.SeparatorTextAlign has unexpected size % instead of 8", it.type.runtime_size);
                case "SeparatorTextPadding";
                    assert(it.offset_in_bytes == 248, "ImGuiStyle.SeparatorTextPadding has unexpected offset % instead of 248", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiStyle.SeparatorTextPadding has unexpected size % instead of 8", it.type.runtime_size);
                case "DisplayWindowPadding";
                    assert(it.offset_in_bytes == 256, "ImGuiStyle.DisplayWindowPadding has unexpected offset % instead of 256", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiStyle.DisplayWindowPadding has unexpected size % instead of 8", it.type.runtime_size);
                case "DisplaySafeAreaPadding";
                    assert(it.offset_in_bytes == 264, "ImGuiStyle.DisplaySafeAreaPadding has unexpected offset % instead of 264", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiStyle.DisplaySafeAreaPadding has unexpected size % instead of 8", it.type.runtime_size);
                case "MouseCursorScale";
                    assert(it.offset_in_bytes == 272, "ImGuiStyle.MouseCursorScale has unexpected offset % instead of 272", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.MouseCursorScale has unexpected size % instead of 4", it.type.runtime_size);
                case "AntiAliasedLines";
                    assert(it.offset_in_bytes == 276, "ImGuiStyle.AntiAliasedLines has unexpected offset % instead of 276", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiStyle.AntiAliasedLines has unexpected size % instead of 1", it.type.runtime_size);
                case "AntiAliasedLinesUseTex";
                    assert(it.offset_in_bytes == 277, "ImGuiStyle.AntiAliasedLinesUseTex has unexpected offset % instead of 277", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiStyle.AntiAliasedLinesUseTex has unexpected size % instead of 1", it.type.runtime_size);
                case "AntiAliasedFill";
                    assert(it.offset_in_bytes == 278, "ImGuiStyle.AntiAliasedFill has unexpected offset % instead of 278", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiStyle.AntiAliasedFill has unexpected size % instead of 1", it.type.runtime_size);
                case "CurveTessellationTol";
                    assert(it.offset_in_bytes == 280, "ImGuiStyle.CurveTessellationTol has unexpected offset % instead of 280", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.CurveTessellationTol has unexpected size % instead of 4", it.type.runtime_size);
                case "CircleTessellationMaxError";
                    assert(it.offset_in_bytes == 284, "ImGuiStyle.CircleTessellationMaxError has unexpected offset % instead of 284", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.CircleTessellationMaxError has unexpected size % instead of 4", it.type.runtime_size);
                case "Colors";
                    assert(it.offset_in_bytes == 288, "ImGuiStyle.Colors has unexpected offset % instead of 288", it.offset_in_bytes);
                    assert(it.type.runtime_size == 928, "ImGuiStyle.Colors has unexpected size % instead of 928", it.type.runtime_size);
                case "HoverStationaryDelay";
                    assert(it.offset_in_bytes == 1216, "ImGuiStyle.HoverStationaryDelay has unexpected offset % instead of 1216", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.HoverStationaryDelay has unexpected size % instead of 4", it.type.runtime_size);
                case "HoverDelayShort";
                    assert(it.offset_in_bytes == 1220, "ImGuiStyle.HoverDelayShort has unexpected offset % instead of 1220", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.HoverDelayShort has unexpected size % instead of 4", it.type.runtime_size);
                case "HoverDelayNormal";
                    assert(it.offset_in_bytes == 1224, "ImGuiStyle.HoverDelayNormal has unexpected offset % instead of 1224", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.HoverDelayNormal has unexpected size % instead of 4", it.type.runtime_size);
                case "HoverFlagsForTooltipMouse";
                    assert(it.offset_in_bytes == 1228, "ImGuiStyle.HoverFlagsForTooltipMouse has unexpected offset % instead of 1228", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.HoverFlagsForTooltipMouse has unexpected size % instead of 4", it.type.runtime_size);
                case "HoverFlagsForTooltipNav";
                    assert(it.offset_in_bytes == 1232, "ImGuiStyle.HoverFlagsForTooltipNav has unexpected offset % instead of 1232", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle.HoverFlagsForTooltipNav has unexpected size % instead of 4", it.type.runtime_size);
                case "_MainScale";
                    assert(it.offset_in_bytes == 1236, "ImGuiStyle._MainScale has unexpected offset % instead of 1236", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle._MainScale has unexpected size % instead of 4", it.type.runtime_size);
                case "_NextFrameFontSizeBase";
                    assert(it.offset_in_bytes == 1240, "ImGuiStyle._NextFrameFontSizeBase has unexpected offset % instead of 1240", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStyle._NextFrameFontSizeBase has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiStyle) == 1244, "ImGuiStyle has size % instead of 1244", size_of(ImGuiStyle));
    }

    {
        info := type_info(ImGuiKeyData);
        for info.members {
            if it.name == {
                case "Down";
                    assert(it.offset_in_bytes == 0, "ImGuiKeyData.Down has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiKeyData.Down has unexpected size % instead of 1", it.type.runtime_size);
                case "DownDuration";
                    assert(it.offset_in_bytes == 4, "ImGuiKeyData.DownDuration has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiKeyData.DownDuration has unexpected size % instead of 4", it.type.runtime_size);
                case "DownDurationPrev";
                    assert(it.offset_in_bytes == 8, "ImGuiKeyData.DownDurationPrev has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiKeyData.DownDurationPrev has unexpected size % instead of 4", it.type.runtime_size);
                case "AnalogValue";
                    assert(it.offset_in_bytes == 12, "ImGuiKeyData.AnalogValue has unexpected offset % instead of 12", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiKeyData.AnalogValue has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiKeyData) == 16, "ImGuiKeyData has size % instead of 16", size_of(ImGuiKeyData));
    }

    {
        info := type_info(ImGuiIO);
        for info.members {
            if it.name == {
                case "ConfigFlags";
                    assert(it.offset_in_bytes == 0, "ImGuiIO.ConfigFlags has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.ConfigFlags has unexpected size % instead of 4", it.type.runtime_size);
                case "BackendFlags";
                    assert(it.offset_in_bytes == 4, "ImGuiIO.BackendFlags has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.BackendFlags has unexpected size % instead of 4", it.type.runtime_size);
                case "DisplaySize";
                    assert(it.offset_in_bytes == 8, "ImGuiIO.DisplaySize has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.DisplaySize has unexpected size % instead of 8", it.type.runtime_size);
                case "DisplayFramebufferScale";
                    assert(it.offset_in_bytes == 16, "ImGuiIO.DisplayFramebufferScale has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.DisplayFramebufferScale has unexpected size % instead of 8", it.type.runtime_size);
                case "DeltaTime";
                    assert(it.offset_in_bytes == 24, "ImGuiIO.DeltaTime has unexpected offset % instead of 24", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.DeltaTime has unexpected size % instead of 4", it.type.runtime_size);
                case "IniSavingRate";
                    assert(it.offset_in_bytes == 28, "ImGuiIO.IniSavingRate has unexpected offset % instead of 28", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.IniSavingRate has unexpected size % instead of 4", it.type.runtime_size);
                case "IniFilename";
                    assert(it.offset_in_bytes == 32, "ImGuiIO.IniFilename has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.IniFilename has unexpected size % instead of 8", it.type.runtime_size);
                case "LogFilename";
                    assert(it.offset_in_bytes == 40, "ImGuiIO.LogFilename has unexpected offset % instead of 40", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.LogFilename has unexpected size % instead of 8", it.type.runtime_size);
                case "UserData";
                    assert(it.offset_in_bytes == 48, "ImGuiIO.UserData has unexpected offset % instead of 48", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.UserData has unexpected size % instead of 8", it.type.runtime_size);
                case "Fonts";
                    assert(it.offset_in_bytes == 56, "ImGuiIO.Fonts has unexpected offset % instead of 56", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.Fonts has unexpected size % instead of 8", it.type.runtime_size);
                case "FontDefault";
                    assert(it.offset_in_bytes == 64, "ImGuiIO.FontDefault has unexpected offset % instead of 64", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.FontDefault has unexpected size % instead of 8", it.type.runtime_size);
                case "FontAllowUserScaling";
                    assert(it.offset_in_bytes == 72, "ImGuiIO.FontAllowUserScaling has unexpected offset % instead of 72", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.FontAllowUserScaling has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigNavSwapGamepadButtons";
                    assert(it.offset_in_bytes == 73, "ImGuiIO.ConfigNavSwapGamepadButtons has unexpected offset % instead of 73", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigNavSwapGamepadButtons has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigNavMoveSetMousePos";
                    assert(it.offset_in_bytes == 74, "ImGuiIO.ConfigNavMoveSetMousePos has unexpected offset % instead of 74", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigNavMoveSetMousePos has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigNavCaptureKeyboard";
                    assert(it.offset_in_bytes == 75, "ImGuiIO.ConfigNavCaptureKeyboard has unexpected offset % instead of 75", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigNavCaptureKeyboard has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigNavEscapeClearFocusItem";
                    assert(it.offset_in_bytes == 76, "ImGuiIO.ConfigNavEscapeClearFocusItem has unexpected offset % instead of 76", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigNavEscapeClearFocusItem has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigNavEscapeClearFocusWindow";
                    assert(it.offset_in_bytes == 77, "ImGuiIO.ConfigNavEscapeClearFocusWindow has unexpected offset % instead of 77", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigNavEscapeClearFocusWindow has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigNavCursorVisibleAuto";
                    assert(it.offset_in_bytes == 78, "ImGuiIO.ConfigNavCursorVisibleAuto has unexpected offset % instead of 78", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigNavCursorVisibleAuto has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigNavCursorVisibleAlways";
                    assert(it.offset_in_bytes == 79, "ImGuiIO.ConfigNavCursorVisibleAlways has unexpected offset % instead of 79", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigNavCursorVisibleAlways has unexpected size % instead of 1", it.type.runtime_size);
                case "MouseDrawCursor";
                    assert(it.offset_in_bytes == 80, "ImGuiIO.MouseDrawCursor has unexpected offset % instead of 80", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.MouseDrawCursor has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigMacOSXBehaviors";
                    assert(it.offset_in_bytes == 81, "ImGuiIO.ConfigMacOSXBehaviors has unexpected offset % instead of 81", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigMacOSXBehaviors has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigInputTrickleEventQueue";
                    assert(it.offset_in_bytes == 82, "ImGuiIO.ConfigInputTrickleEventQueue has unexpected offset % instead of 82", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigInputTrickleEventQueue has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigInputTextCursorBlink";
                    assert(it.offset_in_bytes == 83, "ImGuiIO.ConfigInputTextCursorBlink has unexpected offset % instead of 83", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigInputTextCursorBlink has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigInputTextEnterKeepActive";
                    assert(it.offset_in_bytes == 84, "ImGuiIO.ConfigInputTextEnterKeepActive has unexpected offset % instead of 84", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigInputTextEnterKeepActive has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigDragClickToInputText";
                    assert(it.offset_in_bytes == 85, "ImGuiIO.ConfigDragClickToInputText has unexpected offset % instead of 85", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigDragClickToInputText has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigWindowsResizeFromEdges";
                    assert(it.offset_in_bytes == 86, "ImGuiIO.ConfigWindowsResizeFromEdges has unexpected offset % instead of 86", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigWindowsResizeFromEdges has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigWindowsMoveFromTitleBarOnly";
                    assert(it.offset_in_bytes == 87, "ImGuiIO.ConfigWindowsMoveFromTitleBarOnly has unexpected offset % instead of 87", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigWindowsMoveFromTitleBarOnly has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigWindowsCopyContentsWithCtrlC";
                    assert(it.offset_in_bytes == 88, "ImGuiIO.ConfigWindowsCopyContentsWithCtrlC has unexpected offset % instead of 88", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigWindowsCopyContentsWithCtrlC has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigScrollbarScrollByPage";
                    assert(it.offset_in_bytes == 89, "ImGuiIO.ConfigScrollbarScrollByPage has unexpected offset % instead of 89", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigScrollbarScrollByPage has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigMemoryCompactTimer";
                    assert(it.offset_in_bytes == 92, "ImGuiIO.ConfigMemoryCompactTimer has unexpected offset % instead of 92", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.ConfigMemoryCompactTimer has unexpected size % instead of 4", it.type.runtime_size);
                case "MouseDoubleClickTime";
                    assert(it.offset_in_bytes == 96, "ImGuiIO.MouseDoubleClickTime has unexpected offset % instead of 96", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.MouseDoubleClickTime has unexpected size % instead of 4", it.type.runtime_size);
                case "MouseDoubleClickMaxDist";
                    assert(it.offset_in_bytes == 100, "ImGuiIO.MouseDoubleClickMaxDist has unexpected offset % instead of 100", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.MouseDoubleClickMaxDist has unexpected size % instead of 4", it.type.runtime_size);
                case "MouseDragThreshold";
                    assert(it.offset_in_bytes == 104, "ImGuiIO.MouseDragThreshold has unexpected offset % instead of 104", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.MouseDragThreshold has unexpected size % instead of 4", it.type.runtime_size);
                case "KeyRepeatDelay";
                    assert(it.offset_in_bytes == 108, "ImGuiIO.KeyRepeatDelay has unexpected offset % instead of 108", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.KeyRepeatDelay has unexpected size % instead of 4", it.type.runtime_size);
                case "KeyRepeatRate";
                    assert(it.offset_in_bytes == 112, "ImGuiIO.KeyRepeatRate has unexpected offset % instead of 112", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.KeyRepeatRate has unexpected size % instead of 4", it.type.runtime_size);
                case "ConfigErrorRecovery";
                    assert(it.offset_in_bytes == 116, "ImGuiIO.ConfigErrorRecovery has unexpected offset % instead of 116", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigErrorRecovery has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigErrorRecoveryEnableAssert";
                    assert(it.offset_in_bytes == 117, "ImGuiIO.ConfigErrorRecoveryEnableAssert has unexpected offset % instead of 117", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigErrorRecoveryEnableAssert has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigErrorRecoveryEnableDebugLog";
                    assert(it.offset_in_bytes == 118, "ImGuiIO.ConfigErrorRecoveryEnableDebugLog has unexpected offset % instead of 118", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigErrorRecoveryEnableDebugLog has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigErrorRecoveryEnableTooltip";
                    assert(it.offset_in_bytes == 119, "ImGuiIO.ConfigErrorRecoveryEnableTooltip has unexpected offset % instead of 119", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigErrorRecoveryEnableTooltip has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigDebugIsDebuggerPresent";
                    assert(it.offset_in_bytes == 120, "ImGuiIO.ConfigDebugIsDebuggerPresent has unexpected offset % instead of 120", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigDebugIsDebuggerPresent has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigDebugHighlightIdConflicts";
                    assert(it.offset_in_bytes == 121, "ImGuiIO.ConfigDebugHighlightIdConflicts has unexpected offset % instead of 121", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigDebugHighlightIdConflicts has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigDebugHighlightIdConflictsShowItemPicker";
                    assert(it.offset_in_bytes == 122, "ImGuiIO.ConfigDebugHighlightIdConflictsShowItemPicker has unexpected offset % instead of 122", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigDebugHighlightIdConflictsShowItemPicker has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigDebugBeginReturnValueOnce";
                    assert(it.offset_in_bytes == 123, "ImGuiIO.ConfigDebugBeginReturnValueOnce has unexpected offset % instead of 123", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigDebugBeginReturnValueOnce has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigDebugBeginReturnValueLoop";
                    assert(it.offset_in_bytes == 124, "ImGuiIO.ConfigDebugBeginReturnValueLoop has unexpected offset % instead of 124", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigDebugBeginReturnValueLoop has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigDebugIgnoreFocusLoss";
                    assert(it.offset_in_bytes == 125, "ImGuiIO.ConfigDebugIgnoreFocusLoss has unexpected offset % instead of 125", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigDebugIgnoreFocusLoss has unexpected size % instead of 1", it.type.runtime_size);
                case "ConfigDebugIniSettings";
                    assert(it.offset_in_bytes == 126, "ImGuiIO.ConfigDebugIniSettings has unexpected offset % instead of 126", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.ConfigDebugIniSettings has unexpected size % instead of 1", it.type.runtime_size);
                case "BackendPlatformName";
                    assert(it.offset_in_bytes == 128, "ImGuiIO.BackendPlatformName has unexpected offset % instead of 128", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.BackendPlatformName has unexpected size % instead of 8", it.type.runtime_size);
                case "BackendRendererName";
                    assert(it.offset_in_bytes == 136, "ImGuiIO.BackendRendererName has unexpected offset % instead of 136", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.BackendRendererName has unexpected size % instead of 8", it.type.runtime_size);
                case "BackendPlatformUserData";
                    assert(it.offset_in_bytes == 144, "ImGuiIO.BackendPlatformUserData has unexpected offset % instead of 144", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.BackendPlatformUserData has unexpected size % instead of 8", it.type.runtime_size);
                case "BackendRendererUserData";
                    assert(it.offset_in_bytes == 152, "ImGuiIO.BackendRendererUserData has unexpected offset % instead of 152", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.BackendRendererUserData has unexpected size % instead of 8", it.type.runtime_size);
                case "BackendLanguageUserData";
                    assert(it.offset_in_bytes == 160, "ImGuiIO.BackendLanguageUserData has unexpected offset % instead of 160", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.BackendLanguageUserData has unexpected size % instead of 8", it.type.runtime_size);
                case "WantCaptureMouse";
                    assert(it.offset_in_bytes == 168, "ImGuiIO.WantCaptureMouse has unexpected offset % instead of 168", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.WantCaptureMouse has unexpected size % instead of 1", it.type.runtime_size);
                case "WantCaptureKeyboard";
                    assert(it.offset_in_bytes == 169, "ImGuiIO.WantCaptureKeyboard has unexpected offset % instead of 169", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.WantCaptureKeyboard has unexpected size % instead of 1", it.type.runtime_size);
                case "WantTextInput";
                    assert(it.offset_in_bytes == 170, "ImGuiIO.WantTextInput has unexpected offset % instead of 170", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.WantTextInput has unexpected size % instead of 1", it.type.runtime_size);
                case "WantSetMousePos";
                    assert(it.offset_in_bytes == 171, "ImGuiIO.WantSetMousePos has unexpected offset % instead of 171", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.WantSetMousePos has unexpected size % instead of 1", it.type.runtime_size);
                case "WantSaveIniSettings";
                    assert(it.offset_in_bytes == 172, "ImGuiIO.WantSaveIniSettings has unexpected offset % instead of 172", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.WantSaveIniSettings has unexpected size % instead of 1", it.type.runtime_size);
                case "NavActive";
                    assert(it.offset_in_bytes == 173, "ImGuiIO.NavActive has unexpected offset % instead of 173", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.NavActive has unexpected size % instead of 1", it.type.runtime_size);
                case "NavVisible";
                    assert(it.offset_in_bytes == 174, "ImGuiIO.NavVisible has unexpected offset % instead of 174", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.NavVisible has unexpected size % instead of 1", it.type.runtime_size);
                case "Framerate";
                    assert(it.offset_in_bytes == 176, "ImGuiIO.Framerate has unexpected offset % instead of 176", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.Framerate has unexpected size % instead of 4", it.type.runtime_size);
                case "MetricsRenderVertices";
                    assert(it.offset_in_bytes == 180, "ImGuiIO.MetricsRenderVertices has unexpected offset % instead of 180", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.MetricsRenderVertices has unexpected size % instead of 4", it.type.runtime_size);
                case "MetricsRenderIndices";
                    assert(it.offset_in_bytes == 184, "ImGuiIO.MetricsRenderIndices has unexpected offset % instead of 184", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.MetricsRenderIndices has unexpected size % instead of 4", it.type.runtime_size);
                case "MetricsRenderWindows";
                    assert(it.offset_in_bytes == 188, "ImGuiIO.MetricsRenderWindows has unexpected offset % instead of 188", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.MetricsRenderWindows has unexpected size % instead of 4", it.type.runtime_size);
                case "MetricsActiveWindows";
                    assert(it.offset_in_bytes == 192, "ImGuiIO.MetricsActiveWindows has unexpected offset % instead of 192", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.MetricsActiveWindows has unexpected size % instead of 4", it.type.runtime_size);
                case "MouseDelta";
                    assert(it.offset_in_bytes == 196, "ImGuiIO.MouseDelta has unexpected offset % instead of 196", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.MouseDelta has unexpected size % instead of 8", it.type.runtime_size);
                case "Ctx";
                    assert(it.offset_in_bytes == 208, "ImGuiIO.Ctx has unexpected offset % instead of 208", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.Ctx has unexpected size % instead of 8", it.type.runtime_size);
                case "MousePos";
                    assert(it.offset_in_bytes == 216, "ImGuiIO.MousePos has unexpected offset % instead of 216", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.MousePos has unexpected size % instead of 8", it.type.runtime_size);
                case "MouseDown";
                    assert(it.offset_in_bytes == 224, "ImGuiIO.MouseDown has unexpected offset % instead of 224", it.offset_in_bytes);
                    assert(it.type.runtime_size == 5, "ImGuiIO.MouseDown has unexpected size % instead of 5", it.type.runtime_size);
                case "MouseWheel";
                    assert(it.offset_in_bytes == 232, "ImGuiIO.MouseWheel has unexpected offset % instead of 232", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.MouseWheel has unexpected size % instead of 4", it.type.runtime_size);
                case "MouseWheelH";
                    assert(it.offset_in_bytes == 236, "ImGuiIO.MouseWheelH has unexpected offset % instead of 236", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.MouseWheelH has unexpected size % instead of 4", it.type.runtime_size);
                case "MouseSource";
                    assert(it.offset_in_bytes == 240, "ImGuiIO.MouseSource has unexpected offset % instead of 240", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.MouseSource has unexpected size % instead of 4", it.type.runtime_size);
                case "KeyCtrl";
                    assert(it.offset_in_bytes == 244, "ImGuiIO.KeyCtrl has unexpected offset % instead of 244", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.KeyCtrl has unexpected size % instead of 1", it.type.runtime_size);
                case "KeyShift";
                    assert(it.offset_in_bytes == 245, "ImGuiIO.KeyShift has unexpected offset % instead of 245", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.KeyShift has unexpected size % instead of 1", it.type.runtime_size);
                case "KeyAlt";
                    assert(it.offset_in_bytes == 246, "ImGuiIO.KeyAlt has unexpected offset % instead of 246", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.KeyAlt has unexpected size % instead of 1", it.type.runtime_size);
                case "KeySuper";
                    assert(it.offset_in_bytes == 247, "ImGuiIO.KeySuper has unexpected offset % instead of 247", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.KeySuper has unexpected size % instead of 1", it.type.runtime_size);
                case "KeyMods";
                    assert(it.offset_in_bytes == 248, "ImGuiIO.KeyMods has unexpected offset % instead of 248", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.KeyMods has unexpected size % instead of 4", it.type.runtime_size);
                case "KeysData";
                    assert(it.offset_in_bytes == 252, "ImGuiIO.KeysData has unexpected offset % instead of 252", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2480, "ImGuiIO.KeysData has unexpected size % instead of 2480", it.type.runtime_size);
                case "WantCaptureMouseUnlessPopupClose";
                    assert(it.offset_in_bytes == 2732, "ImGuiIO.WantCaptureMouseUnlessPopupClose has unexpected offset % instead of 2732", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.WantCaptureMouseUnlessPopupClose has unexpected size % instead of 1", it.type.runtime_size);
                case "MousePosPrev";
                    assert(it.offset_in_bytes == 2736, "ImGuiIO.MousePosPrev has unexpected offset % instead of 2736", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.MousePosPrev has unexpected size % instead of 8", it.type.runtime_size);
                case "MouseClickedPos";
                    assert(it.offset_in_bytes == 2744, "ImGuiIO.MouseClickedPos has unexpected offset % instead of 2744", it.offset_in_bytes);
                    assert(it.type.runtime_size == 40, "ImGuiIO.MouseClickedPos has unexpected size % instead of 40", it.type.runtime_size);
                case "MouseClickedTime";
                    assert(it.offset_in_bytes == 2784, "ImGuiIO.MouseClickedTime has unexpected offset % instead of 2784", it.offset_in_bytes);
                    assert(it.type.runtime_size == 40, "ImGuiIO.MouseClickedTime has unexpected size % instead of 40", it.type.runtime_size);
                case "MouseClicked";
                    assert(it.offset_in_bytes == 2824, "ImGuiIO.MouseClicked has unexpected offset % instead of 2824", it.offset_in_bytes);
                    assert(it.type.runtime_size == 5, "ImGuiIO.MouseClicked has unexpected size % instead of 5", it.type.runtime_size);
                case "MouseDoubleClicked";
                    assert(it.offset_in_bytes == 2829, "ImGuiIO.MouseDoubleClicked has unexpected offset % instead of 2829", it.offset_in_bytes);
                    assert(it.type.runtime_size == 5, "ImGuiIO.MouseDoubleClicked has unexpected size % instead of 5", it.type.runtime_size);
                case "MouseClickedCount";
                    assert(it.offset_in_bytes == 2834, "ImGuiIO.MouseClickedCount has unexpected offset % instead of 2834", it.offset_in_bytes);
                    assert(it.type.runtime_size == 10, "ImGuiIO.MouseClickedCount has unexpected size % instead of 10", it.type.runtime_size);
                case "MouseClickedLastCount";
                    assert(it.offset_in_bytes == 2844, "ImGuiIO.MouseClickedLastCount has unexpected offset % instead of 2844", it.offset_in_bytes);
                    assert(it.type.runtime_size == 10, "ImGuiIO.MouseClickedLastCount has unexpected size % instead of 10", it.type.runtime_size);
                case "MouseReleased";
                    assert(it.offset_in_bytes == 2854, "ImGuiIO.MouseReleased has unexpected offset % instead of 2854", it.offset_in_bytes);
                    assert(it.type.runtime_size == 5, "ImGuiIO.MouseReleased has unexpected size % instead of 5", it.type.runtime_size);
                case "MouseReleasedTime";
                    assert(it.offset_in_bytes == 2864, "ImGuiIO.MouseReleasedTime has unexpected offset % instead of 2864", it.offset_in_bytes);
                    assert(it.type.runtime_size == 40, "ImGuiIO.MouseReleasedTime has unexpected size % instead of 40", it.type.runtime_size);
                case "MouseDownOwned";
                    assert(it.offset_in_bytes == 2904, "ImGuiIO.MouseDownOwned has unexpected offset % instead of 2904", it.offset_in_bytes);
                    assert(it.type.runtime_size == 5, "ImGuiIO.MouseDownOwned has unexpected size % instead of 5", it.type.runtime_size);
                case "MouseDownOwnedUnlessPopupClose";
                    assert(it.offset_in_bytes == 2909, "ImGuiIO.MouseDownOwnedUnlessPopupClose has unexpected offset % instead of 2909", it.offset_in_bytes);
                    assert(it.type.runtime_size == 5, "ImGuiIO.MouseDownOwnedUnlessPopupClose has unexpected size % instead of 5", it.type.runtime_size);
                case "MouseWheelRequestAxisSwap";
                    assert(it.offset_in_bytes == 2914, "ImGuiIO.MouseWheelRequestAxisSwap has unexpected offset % instead of 2914", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.MouseWheelRequestAxisSwap has unexpected size % instead of 1", it.type.runtime_size);
                case "MouseCtrlLeftAsRightClick";
                    assert(it.offset_in_bytes == 2915, "ImGuiIO.MouseCtrlLeftAsRightClick has unexpected offset % instead of 2915", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.MouseCtrlLeftAsRightClick has unexpected size % instead of 1", it.type.runtime_size);
                case "MouseDownDuration";
                    assert(it.offset_in_bytes == 2916, "ImGuiIO.MouseDownDuration has unexpected offset % instead of 2916", it.offset_in_bytes);
                    assert(it.type.runtime_size == 20, "ImGuiIO.MouseDownDuration has unexpected size % instead of 20", it.type.runtime_size);
                case "MouseDownDurationPrev";
                    assert(it.offset_in_bytes == 2936, "ImGuiIO.MouseDownDurationPrev has unexpected offset % instead of 2936", it.offset_in_bytes);
                    assert(it.type.runtime_size == 20, "ImGuiIO.MouseDownDurationPrev has unexpected size % instead of 20", it.type.runtime_size);
                case "MouseDragMaxDistanceSqr";
                    assert(it.offset_in_bytes == 2956, "ImGuiIO.MouseDragMaxDistanceSqr has unexpected offset % instead of 2956", it.offset_in_bytes);
                    assert(it.type.runtime_size == 20, "ImGuiIO.MouseDragMaxDistanceSqr has unexpected size % instead of 20", it.type.runtime_size);
                case "PenPressure";
                    assert(it.offset_in_bytes == 2976, "ImGuiIO.PenPressure has unexpected offset % instead of 2976", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.PenPressure has unexpected size % instead of 4", it.type.runtime_size);
                case "AppFocusLost";
                    assert(it.offset_in_bytes == 2980, "ImGuiIO.AppFocusLost has unexpected offset % instead of 2980", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.AppFocusLost has unexpected size % instead of 1", it.type.runtime_size);
                case "AppAcceptingEvents";
                    assert(it.offset_in_bytes == 2981, "ImGuiIO.AppAcceptingEvents has unexpected offset % instead of 2981", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiIO.AppAcceptingEvents has unexpected size % instead of 1", it.type.runtime_size);
                case "InputQueueSurrogate";
                    assert(it.offset_in_bytes == 2982, "ImGuiIO.InputQueueSurrogate has unexpected offset % instead of 2982", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImGuiIO.InputQueueSurrogate has unexpected size % instead of 2", it.type.runtime_size);
                case "InputQueueCharacters";
                    assert(it.offset_in_bytes == 2984, "ImGuiIO.InputQueueCharacters has unexpected offset % instead of 2984", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImGuiIO.InputQueueCharacters has unexpected size % instead of 16", it.type.runtime_size);
                case "FontGlobalScale";
                    assert(it.offset_in_bytes == 3000, "ImGuiIO.FontGlobalScale has unexpected offset % instead of 3000", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiIO.FontGlobalScale has unexpected size % instead of 4", it.type.runtime_size);
                case "GetClipboardTextFn";
                    assert(it.offset_in_bytes == 3008, "ImGuiIO.GetClipboardTextFn has unexpected offset % instead of 3008", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.GetClipboardTextFn has unexpected size % instead of 8", it.type.runtime_size);
                case "SetClipboardTextFn";
                    assert(it.offset_in_bytes == 3016, "ImGuiIO.SetClipboardTextFn has unexpected offset % instead of 3016", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.SetClipboardTextFn has unexpected size % instead of 8", it.type.runtime_size);
                case "ClipboardUserData";
                    assert(it.offset_in_bytes == 3024, "ImGuiIO.ClipboardUserData has unexpected offset % instead of 3024", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiIO.ClipboardUserData has unexpected size % instead of 8", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiIO) == 3032, "ImGuiIO has size % instead of 3032", size_of(ImGuiIO));
    }

    {
        info := type_info(ImGuiInputTextCallbackData);
        for info.members {
            if it.name == {
                case "Ctx";
                    assert(it.offset_in_bytes == 0, "ImGuiInputTextCallbackData.Ctx has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiInputTextCallbackData.Ctx has unexpected size % instead of 8", it.type.runtime_size);
                case "EventFlag";
                    assert(it.offset_in_bytes == 8, "ImGuiInputTextCallbackData.EventFlag has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiInputTextCallbackData.EventFlag has unexpected size % instead of 4", it.type.runtime_size);
                case "Flags";
                    assert(it.offset_in_bytes == 12, "ImGuiInputTextCallbackData.Flags has unexpected offset % instead of 12", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiInputTextCallbackData.Flags has unexpected size % instead of 4", it.type.runtime_size);
                case "UserData";
                    assert(it.offset_in_bytes == 16, "ImGuiInputTextCallbackData.UserData has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiInputTextCallbackData.UserData has unexpected size % instead of 8", it.type.runtime_size);
                case "EventChar";
                    assert(it.offset_in_bytes == 24, "ImGuiInputTextCallbackData.EventChar has unexpected offset % instead of 24", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImGuiInputTextCallbackData.EventChar has unexpected size % instead of 2", it.type.runtime_size);
                case "EventKey";
                    assert(it.offset_in_bytes == 28, "ImGuiInputTextCallbackData.EventKey has unexpected offset % instead of 28", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiInputTextCallbackData.EventKey has unexpected size % instead of 4", it.type.runtime_size);
                case "Buf";
                    assert(it.offset_in_bytes == 32, "ImGuiInputTextCallbackData.Buf has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiInputTextCallbackData.Buf has unexpected size % instead of 8", it.type.runtime_size);
                case "BufTextLen";
                    assert(it.offset_in_bytes == 40, "ImGuiInputTextCallbackData.BufTextLen has unexpected offset % instead of 40", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiInputTextCallbackData.BufTextLen has unexpected size % instead of 4", it.type.runtime_size);
                case "BufSize";
                    assert(it.offset_in_bytes == 44, "ImGuiInputTextCallbackData.BufSize has unexpected offset % instead of 44", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiInputTextCallbackData.BufSize has unexpected size % instead of 4", it.type.runtime_size);
                case "BufDirty";
                    assert(it.offset_in_bytes == 48, "ImGuiInputTextCallbackData.BufDirty has unexpected offset % instead of 48", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiInputTextCallbackData.BufDirty has unexpected size % instead of 1", it.type.runtime_size);
                case "CursorPos";
                    assert(it.offset_in_bytes == 52, "ImGuiInputTextCallbackData.CursorPos has unexpected offset % instead of 52", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiInputTextCallbackData.CursorPos has unexpected size % instead of 4", it.type.runtime_size);
                case "SelectionStart";
                    assert(it.offset_in_bytes == 56, "ImGuiInputTextCallbackData.SelectionStart has unexpected offset % instead of 56", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiInputTextCallbackData.SelectionStart has unexpected size % instead of 4", it.type.runtime_size);
                case "SelectionEnd";
                    assert(it.offset_in_bytes == 60, "ImGuiInputTextCallbackData.SelectionEnd has unexpected offset % instead of 60", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiInputTextCallbackData.SelectionEnd has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiInputTextCallbackData) == 64, "ImGuiInputTextCallbackData has size % instead of 64", size_of(ImGuiInputTextCallbackData));
    }

    {
        info := type_info(ImGuiSizeCallbackData);
        for info.members {
            if it.name == {
                case "UserData";
                    assert(it.offset_in_bytes == 0, "ImGuiSizeCallbackData.UserData has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiSizeCallbackData.UserData has unexpected size % instead of 8", it.type.runtime_size);
                case "Pos";
                    assert(it.offset_in_bytes == 8, "ImGuiSizeCallbackData.Pos has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiSizeCallbackData.Pos has unexpected size % instead of 8", it.type.runtime_size);
                case "CurrentSize";
                    assert(it.offset_in_bytes == 16, "ImGuiSizeCallbackData.CurrentSize has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiSizeCallbackData.CurrentSize has unexpected size % instead of 8", it.type.runtime_size);
                case "DesiredSize";
                    assert(it.offset_in_bytes == 24, "ImGuiSizeCallbackData.DesiredSize has unexpected offset % instead of 24", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiSizeCallbackData.DesiredSize has unexpected size % instead of 8", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiSizeCallbackData) == 32, "ImGuiSizeCallbackData has size % instead of 32", size_of(ImGuiSizeCallbackData));
    }

    {
        info := type_info(ImGuiPayload);
        for info.members {
            if it.name == {
                case "Data";
                    assert(it.offset_in_bytes == 0, "ImGuiPayload.Data has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiPayload.Data has unexpected size % instead of 8", it.type.runtime_size);
                case "DataSize";
                    assert(it.offset_in_bytes == 8, "ImGuiPayload.DataSize has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiPayload.DataSize has unexpected size % instead of 4", it.type.runtime_size);
                case "SourceId";
                    assert(it.offset_in_bytes == 12, "ImGuiPayload.SourceId has unexpected offset % instead of 12", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiPayload.SourceId has unexpected size % instead of 4", it.type.runtime_size);
                case "SourceParentId";
                    assert(it.offset_in_bytes == 16, "ImGuiPayload.SourceParentId has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiPayload.SourceParentId has unexpected size % instead of 4", it.type.runtime_size);
                case "DataFrameCount";
                    assert(it.offset_in_bytes == 20, "ImGuiPayload.DataFrameCount has unexpected offset % instead of 20", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiPayload.DataFrameCount has unexpected size % instead of 4", it.type.runtime_size);
                case "DataType";
                    assert(it.offset_in_bytes == 24, "ImGuiPayload.DataType has unexpected offset % instead of 24", it.offset_in_bytes);
                    assert(it.type.runtime_size == 33, "ImGuiPayload.DataType has unexpected size % instead of 33", it.type.runtime_size);
                case "Preview";
                    assert(it.offset_in_bytes == 57, "ImGuiPayload.Preview has unexpected offset % instead of 57", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiPayload.Preview has unexpected size % instead of 1", it.type.runtime_size);
                case "Delivery";
                    assert(it.offset_in_bytes == 58, "ImGuiPayload.Delivery has unexpected offset % instead of 58", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiPayload.Delivery has unexpected size % instead of 1", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiPayload) == 64, "ImGuiPayload has size % instead of 64", size_of(ImGuiPayload));
    }

    {
        info := type_info(ImGuiOnceUponAFrame);
        for info.members {
            if it.name == {
                case "RefFrame";
                    assert(it.offset_in_bytes == 0, "ImGuiOnceUponAFrame.RefFrame has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiOnceUponAFrame.RefFrame has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiOnceUponAFrame) == 4, "ImGuiOnceUponAFrame has size % instead of 4", size_of(ImGuiOnceUponAFrame));
    }

    {
        info := type_info(ImGuiTextFilter.ImGuiTextRange);
        for info.members {
            if it.name == {
                case "b";
                    assert(it.offset_in_bytes == 0, "ImGuiTextFilter.ImGuiTextRange.b has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiTextFilter.ImGuiTextRange.b has unexpected size % instead of 8", it.type.runtime_size);
                case "e";
                    assert(it.offset_in_bytes == 8, "ImGuiTextFilter.ImGuiTextRange.e has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiTextFilter.ImGuiTextRange.e has unexpected size % instead of 8", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiTextFilter.ImGuiTextRange) == 16, "ImGuiTextFilter.ImGuiTextRange has size % instead of 16", size_of(ImGuiTextFilter.ImGuiTextRange));
    }

    {
        info := type_info(ImGuiTextFilter);
        for info.members {
            if it.name == {
                case "InputBuf";
                    assert(it.offset_in_bytes == 0, "ImGuiTextFilter.InputBuf has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 256, "ImGuiTextFilter.InputBuf has unexpected size % instead of 256", it.type.runtime_size);
                case "Filters";
                    assert(it.offset_in_bytes == 256, "ImGuiTextFilter.Filters has unexpected offset % instead of 256", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImGuiTextFilter.Filters has unexpected size % instead of 16", it.type.runtime_size);
                case "CountGrep";
                    assert(it.offset_in_bytes == 272, "ImGuiTextFilter.CountGrep has unexpected offset % instead of 272", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiTextFilter.CountGrep has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiTextFilter) == 280, "ImGuiTextFilter has size % instead of 280", size_of(ImGuiTextFilter));
    }

    {
        info := type_info(ImGuiTextBuffer);
        for info.members {
            if it.name == {
                case "Buf";
                    assert(it.offset_in_bytes == 0, "ImGuiTextBuffer.Buf has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImGuiTextBuffer.Buf has unexpected size % instead of 16", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiTextBuffer) == 16, "ImGuiTextBuffer has size % instead of 16", size_of(ImGuiTextBuffer));
    }

    {
        info := type_info(ImGuiStoragePair);
        for info.members {
            if it.name == {
                case "key";
                    assert(it.offset_in_bytes == 0, "ImGuiStoragePair.key has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiStoragePair.key has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiStoragePair) == 16, "ImGuiStoragePair has size % instead of 16", size_of(ImGuiStoragePair));
    }

    {
        info := type_info(ImGuiStorage);
        for info.members {
            if it.name == {
                case "Data";
                    assert(it.offset_in_bytes == 0, "ImGuiStorage.Data has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImGuiStorage.Data has unexpected size % instead of 16", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiStorage) == 16, "ImGuiStorage has size % instead of 16", size_of(ImGuiStorage));
    }

    {
        info := type_info(ImGuiListClipper);
        for info.members {
            if it.name == {
                case "Ctx";
                    assert(it.offset_in_bytes == 0, "ImGuiListClipper.Ctx has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiListClipper.Ctx has unexpected size % instead of 8", it.type.runtime_size);
                case "DisplayStart";
                    assert(it.offset_in_bytes == 8, "ImGuiListClipper.DisplayStart has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiListClipper.DisplayStart has unexpected size % instead of 4", it.type.runtime_size);
                case "DisplayEnd";
                    assert(it.offset_in_bytes == 12, "ImGuiListClipper.DisplayEnd has unexpected offset % instead of 12", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiListClipper.DisplayEnd has unexpected size % instead of 4", it.type.runtime_size);
                case "ItemsCount";
                    assert(it.offset_in_bytes == 16, "ImGuiListClipper.ItemsCount has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiListClipper.ItemsCount has unexpected size % instead of 4", it.type.runtime_size);
                case "ItemsHeight";
                    assert(it.offset_in_bytes == 20, "ImGuiListClipper.ItemsHeight has unexpected offset % instead of 20", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiListClipper.ItemsHeight has unexpected size % instead of 4", it.type.runtime_size);
                case "StartPosY";
                    assert(it.offset_in_bytes == 24, "ImGuiListClipper.StartPosY has unexpected offset % instead of 24", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiListClipper.StartPosY has unexpected size % instead of 8", it.type.runtime_size);
                case "StartSeekOffsetY";
                    assert(it.offset_in_bytes == 32, "ImGuiListClipper.StartSeekOffsetY has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiListClipper.StartSeekOffsetY has unexpected size % instead of 8", it.type.runtime_size);
                case "TempData";
                    assert(it.offset_in_bytes == 40, "ImGuiListClipper.TempData has unexpected offset % instead of 40", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiListClipper.TempData has unexpected size % instead of 8", it.type.runtime_size);
                case "Flags";
                    assert(it.offset_in_bytes == 48, "ImGuiListClipper.Flags has unexpected offset % instead of 48", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiListClipper.Flags has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiListClipper) == 56, "ImGuiListClipper has size % instead of 56", size_of(ImGuiListClipper));
    }

    {
        info := type_info(ImColor);
        for info.members {
            if it.name == {
                case "Value";
                    assert(it.offset_in_bytes == 0, "ImColor.Value has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImColor.Value has unexpected size % instead of 16", it.type.runtime_size);
            }
        }
        assert(size_of(ImColor) == 16, "ImColor has size % instead of 16", size_of(ImColor));
    }

    {
        info := type_info(ImGuiMultiSelectIO);
        for info.members {
            if it.name == {
                case "Requests";
                    assert(it.offset_in_bytes == 0, "ImGuiMultiSelectIO.Requests has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImGuiMultiSelectIO.Requests has unexpected size % instead of 16", it.type.runtime_size);
                case "RangeSrcItem";
                    assert(it.offset_in_bytes == 16, "ImGuiMultiSelectIO.RangeSrcItem has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiMultiSelectIO.RangeSrcItem has unexpected size % instead of 8", it.type.runtime_size);
                case "NavIdItem";
                    assert(it.offset_in_bytes == 24, "ImGuiMultiSelectIO.NavIdItem has unexpected offset % instead of 24", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiMultiSelectIO.NavIdItem has unexpected size % instead of 8", it.type.runtime_size);
                case "NavIdSelected";
                    assert(it.offset_in_bytes == 32, "ImGuiMultiSelectIO.NavIdSelected has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiMultiSelectIO.NavIdSelected has unexpected size % instead of 1", it.type.runtime_size);
                case "RangeSrcReset";
                    assert(it.offset_in_bytes == 33, "ImGuiMultiSelectIO.RangeSrcReset has unexpected offset % instead of 33", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiMultiSelectIO.RangeSrcReset has unexpected size % instead of 1", it.type.runtime_size);
                case "ItemsCount";
                    assert(it.offset_in_bytes == 36, "ImGuiMultiSelectIO.ItemsCount has unexpected offset % instead of 36", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiMultiSelectIO.ItemsCount has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiMultiSelectIO) == 40, "ImGuiMultiSelectIO has size % instead of 40", size_of(ImGuiMultiSelectIO));
    }

    {
        info := type_info(ImGuiSelectionRequest);
        for info.members {
            if it.name == {
                case "Type";
                    assert(it.offset_in_bytes == 0, "ImGuiSelectionRequest.Type has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiSelectionRequest.Type has unexpected size % instead of 4", it.type.runtime_size);
                case "Selected";
                    assert(it.offset_in_bytes == 4, "ImGuiSelectionRequest.Selected has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiSelectionRequest.Selected has unexpected size % instead of 1", it.type.runtime_size);
                case "RangeDirection";
                    assert(it.offset_in_bytes == 5, "ImGuiSelectionRequest.RangeDirection has unexpected offset % instead of 5", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiSelectionRequest.RangeDirection has unexpected size % instead of 1", it.type.runtime_size);
                case "RangeFirstItem";
                    assert(it.offset_in_bytes == 8, "ImGuiSelectionRequest.RangeFirstItem has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiSelectionRequest.RangeFirstItem has unexpected size % instead of 8", it.type.runtime_size);
                case "RangeLastItem";
                    assert(it.offset_in_bytes == 16, "ImGuiSelectionRequest.RangeLastItem has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiSelectionRequest.RangeLastItem has unexpected size % instead of 8", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiSelectionRequest) == 24, "ImGuiSelectionRequest has size % instead of 24", size_of(ImGuiSelectionRequest));
    }

    {
        info := type_info(ImGuiSelectionBasicStorage);
        for info.members {
            if it.name == {
                case "Size";
                    assert(it.offset_in_bytes == 0, "ImGuiSelectionBasicStorage.Size has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiSelectionBasicStorage.Size has unexpected size % instead of 4", it.type.runtime_size);
                case "PreserveOrder";
                    assert(it.offset_in_bytes == 4, "ImGuiSelectionBasicStorage.PreserveOrder has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiSelectionBasicStorage.PreserveOrder has unexpected size % instead of 1", it.type.runtime_size);
                case "UserData";
                    assert(it.offset_in_bytes == 8, "ImGuiSelectionBasicStorage.UserData has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiSelectionBasicStorage.UserData has unexpected size % instead of 8", it.type.runtime_size);
                case "AdapterIndexToStorageId";
                    assert(it.offset_in_bytes == 16, "ImGuiSelectionBasicStorage.AdapterIndexToStorageId has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiSelectionBasicStorage.AdapterIndexToStorageId has unexpected size % instead of 8", it.type.runtime_size);
                case "_SelectionOrder";
                    assert(it.offset_in_bytes == 24, "ImGuiSelectionBasicStorage._SelectionOrder has unexpected offset % instead of 24", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiSelectionBasicStorage._SelectionOrder has unexpected size % instead of 4", it.type.runtime_size);
                case "_Storage";
                    assert(it.offset_in_bytes == 32, "ImGuiSelectionBasicStorage._Storage has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImGuiSelectionBasicStorage._Storage has unexpected size % instead of 16", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiSelectionBasicStorage) == 48, "ImGuiSelectionBasicStorage has size % instead of 48", size_of(ImGuiSelectionBasicStorage));
    }

    {
        info := type_info(ImGuiSelectionExternalStorage);
        for info.members {
            if it.name == {
                case "UserData";
                    assert(it.offset_in_bytes == 0, "ImGuiSelectionExternalStorage.UserData has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiSelectionExternalStorage.UserData has unexpected size % instead of 8", it.type.runtime_size);
                case "AdapterSetItemSelected";
                    assert(it.offset_in_bytes == 8, "ImGuiSelectionExternalStorage.AdapterSetItemSelected has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiSelectionExternalStorage.AdapterSetItemSelected has unexpected size % instead of 8", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiSelectionExternalStorage) == 16, "ImGuiSelectionExternalStorage has size % instead of 16", size_of(ImGuiSelectionExternalStorage));
    }

    {
        info := type_info(ImDrawCmd);
        for info.members {
            if it.name == {
                case "ClipRect";
                    assert(it.offset_in_bytes == 0, "ImDrawCmd.ClipRect has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImDrawCmd.ClipRect has unexpected size % instead of 16", it.type.runtime_size);
                case "TexRef";
                    assert(it.offset_in_bytes == 16, "ImDrawCmd.TexRef has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImDrawCmd.TexRef has unexpected size % instead of 16", it.type.runtime_size);
                case "VtxOffset";
                    assert(it.offset_in_bytes == 32, "ImDrawCmd.VtxOffset has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImDrawCmd.VtxOffset has unexpected size % instead of 4", it.type.runtime_size);
                case "IdxOffset";
                    assert(it.offset_in_bytes == 36, "ImDrawCmd.IdxOffset has unexpected offset % instead of 36", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImDrawCmd.IdxOffset has unexpected size % instead of 4", it.type.runtime_size);
                case "ElemCount";
                    assert(it.offset_in_bytes == 40, "ImDrawCmd.ElemCount has unexpected offset % instead of 40", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImDrawCmd.ElemCount has unexpected size % instead of 4", it.type.runtime_size);
                case "UserCallback";
                    assert(it.offset_in_bytes == 48, "ImDrawCmd.UserCallback has unexpected offset % instead of 48", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImDrawCmd.UserCallback has unexpected size % instead of 8", it.type.runtime_size);
                case "UserCallbackData";
                    assert(it.offset_in_bytes == 56, "ImDrawCmd.UserCallbackData has unexpected offset % instead of 56", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImDrawCmd.UserCallbackData has unexpected size % instead of 8", it.type.runtime_size);
                case "UserCallbackDataSize";
                    assert(it.offset_in_bytes == 64, "ImDrawCmd.UserCallbackDataSize has unexpected offset % instead of 64", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImDrawCmd.UserCallbackDataSize has unexpected size % instead of 4", it.type.runtime_size);
                case "UserCallbackDataOffset";
                    assert(it.offset_in_bytes == 68, "ImDrawCmd.UserCallbackDataOffset has unexpected offset % instead of 68", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImDrawCmd.UserCallbackDataOffset has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImDrawCmd) == 72, "ImDrawCmd has size % instead of 72", size_of(ImDrawCmd));
    }

    {
        info := type_info(ImDrawVert);
        for info.members {
            if it.name == {
                case "pos";
                    assert(it.offset_in_bytes == 0, "ImDrawVert.pos has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImDrawVert.pos has unexpected size % instead of 8", it.type.runtime_size);
                case "uv";
                    assert(it.offset_in_bytes == 8, "ImDrawVert.uv has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImDrawVert.uv has unexpected size % instead of 8", it.type.runtime_size);
                case "col";
                    assert(it.offset_in_bytes == 16, "ImDrawVert.col has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImDrawVert.col has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImDrawVert) == 20, "ImDrawVert has size % instead of 20", size_of(ImDrawVert));
    }

    {
        info := type_info(ImDrawCmdHeader);
        for info.members {
            if it.name == {
                case "ClipRect";
                    assert(it.offset_in_bytes == 0, "ImDrawCmdHeader.ClipRect has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImDrawCmdHeader.ClipRect has unexpected size % instead of 16", it.type.runtime_size);
                case "TexRef";
                    assert(it.offset_in_bytes == 16, "ImDrawCmdHeader.TexRef has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImDrawCmdHeader.TexRef has unexpected size % instead of 16", it.type.runtime_size);
                case "VtxOffset";
                    assert(it.offset_in_bytes == 32, "ImDrawCmdHeader.VtxOffset has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImDrawCmdHeader.VtxOffset has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImDrawCmdHeader) == 40, "ImDrawCmdHeader has size % instead of 40", size_of(ImDrawCmdHeader));
    }

    {
        info := type_info(ImDrawChannel);
        for info.members {
            if it.name == {
                case "_CmdBuffer";
                    assert(it.offset_in_bytes == 0, "ImDrawChannel._CmdBuffer has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImDrawChannel._CmdBuffer has unexpected size % instead of 16", it.type.runtime_size);
                case "_IdxBuffer";
                    assert(it.offset_in_bytes == 16, "ImDrawChannel._IdxBuffer has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImDrawChannel._IdxBuffer has unexpected size % instead of 16", it.type.runtime_size);
            }
        }
        assert(size_of(ImDrawChannel) == 32, "ImDrawChannel has size % instead of 32", size_of(ImDrawChannel));
    }

    {
        info := type_info(ImDrawListSplitter);
        for info.members {
            if it.name == {
                case "_Current";
                    assert(it.offset_in_bytes == 0, "ImDrawListSplitter._Current has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImDrawListSplitter._Current has unexpected size % instead of 4", it.type.runtime_size);
                case "_Count";
                    assert(it.offset_in_bytes == 4, "ImDrawListSplitter._Count has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImDrawListSplitter._Count has unexpected size % instead of 4", it.type.runtime_size);
                case "_Channels";
                    assert(it.offset_in_bytes == 8, "ImDrawListSplitter._Channels has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImDrawListSplitter._Channels has unexpected size % instead of 16", it.type.runtime_size);
            }
        }
        assert(size_of(ImDrawListSplitter) == 24, "ImDrawListSplitter has size % instead of 24", size_of(ImDrawListSplitter));
    }

    {
        info := type_info(ImDrawList);
        for info.members {
            if it.name == {
                case "CmdBuffer";
                    assert(it.offset_in_bytes == 0, "ImDrawList.CmdBuffer has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImDrawList.CmdBuffer has unexpected size % instead of 16", it.type.runtime_size);
                case "IdxBuffer";
                    assert(it.offset_in_bytes == 16, "ImDrawList.IdxBuffer has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImDrawList.IdxBuffer has unexpected size % instead of 16", it.type.runtime_size);
                case "VtxBuffer";
                    assert(it.offset_in_bytes == 32, "ImDrawList.VtxBuffer has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImDrawList.VtxBuffer has unexpected size % instead of 16", it.type.runtime_size);
                case "Flags";
                    assert(it.offset_in_bytes == 48, "ImDrawList.Flags has unexpected offset % instead of 48", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImDrawList.Flags has unexpected size % instead of 4", it.type.runtime_size);
                case "_VtxCurrentIdx";
                    assert(it.offset_in_bytes == 52, "ImDrawList._VtxCurrentIdx has unexpected offset % instead of 52", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImDrawList._VtxCurrentIdx has unexpected size % instead of 4", it.type.runtime_size);
                case "_Data";
                    assert(it.offset_in_bytes == 56, "ImDrawList._Data has unexpected offset % instead of 56", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImDrawList._Data has unexpected size % instead of 8", it.type.runtime_size);
                case "_VtxWritePtr";
                    assert(it.offset_in_bytes == 64, "ImDrawList._VtxWritePtr has unexpected offset % instead of 64", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImDrawList._VtxWritePtr has unexpected size % instead of 8", it.type.runtime_size);
                case "_IdxWritePtr";
                    assert(it.offset_in_bytes == 72, "ImDrawList._IdxWritePtr has unexpected offset % instead of 72", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImDrawList._IdxWritePtr has unexpected size % instead of 8", it.type.runtime_size);
                case "_Path";
                    assert(it.offset_in_bytes == 80, "ImDrawList._Path has unexpected offset % instead of 80", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImDrawList._Path has unexpected size % instead of 16", it.type.runtime_size);
                case "_CmdHeader";
                    assert(it.offset_in_bytes == 96, "ImDrawList._CmdHeader has unexpected offset % instead of 96", it.offset_in_bytes);
                    assert(it.type.runtime_size == 40, "ImDrawList._CmdHeader has unexpected size % instead of 40", it.type.runtime_size);
                case "_Splitter";
                    assert(it.offset_in_bytes == 136, "ImDrawList._Splitter has unexpected offset % instead of 136", it.offset_in_bytes);
                    assert(it.type.runtime_size == 24, "ImDrawList._Splitter has unexpected size % instead of 24", it.type.runtime_size);
                case "_ClipRectStack";
                    assert(it.offset_in_bytes == 160, "ImDrawList._ClipRectStack has unexpected offset % instead of 160", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImDrawList._ClipRectStack has unexpected size % instead of 16", it.type.runtime_size);
                case "_TextureStack";
                    assert(it.offset_in_bytes == 176, "ImDrawList._TextureStack has unexpected offset % instead of 176", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImDrawList._TextureStack has unexpected size % instead of 16", it.type.runtime_size);
                case "_CallbacksDataBuf";
                    assert(it.offset_in_bytes == 192, "ImDrawList._CallbacksDataBuf has unexpected offset % instead of 192", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImDrawList._CallbacksDataBuf has unexpected size % instead of 16", it.type.runtime_size);
                case "_FringeScale";
                    assert(it.offset_in_bytes == 208, "ImDrawList._FringeScale has unexpected offset % instead of 208", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImDrawList._FringeScale has unexpected size % instead of 4", it.type.runtime_size);
                case "_OwnerName";
                    assert(it.offset_in_bytes == 216, "ImDrawList._OwnerName has unexpected offset % instead of 216", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImDrawList._OwnerName has unexpected size % instead of 8", it.type.runtime_size);
            }
        }
        assert(size_of(ImDrawList) == 224, "ImDrawList has size % instead of 224", size_of(ImDrawList));
    }

    {
        info := type_info(ImDrawData);
        for info.members {
            if it.name == {
                case "Valid";
                    assert(it.offset_in_bytes == 0, "ImDrawData.Valid has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImDrawData.Valid has unexpected size % instead of 1", it.type.runtime_size);
                case "CmdListsCount";
                    assert(it.offset_in_bytes == 4, "ImDrawData.CmdListsCount has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImDrawData.CmdListsCount has unexpected size % instead of 4", it.type.runtime_size);
                case "TotalIdxCount";
                    assert(it.offset_in_bytes == 8, "ImDrawData.TotalIdxCount has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImDrawData.TotalIdxCount has unexpected size % instead of 4", it.type.runtime_size);
                case "TotalVtxCount";
                    assert(it.offset_in_bytes == 12, "ImDrawData.TotalVtxCount has unexpected offset % instead of 12", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImDrawData.TotalVtxCount has unexpected size % instead of 4", it.type.runtime_size);
                case "CmdLists";
                    assert(it.offset_in_bytes == 16, "ImDrawData.CmdLists has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImDrawData.CmdLists has unexpected size % instead of 16", it.type.runtime_size);
                case "DisplayPos";
                    assert(it.offset_in_bytes == 32, "ImDrawData.DisplayPos has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImDrawData.DisplayPos has unexpected size % instead of 8", it.type.runtime_size);
                case "DisplaySize";
                    assert(it.offset_in_bytes == 40, "ImDrawData.DisplaySize has unexpected offset % instead of 40", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImDrawData.DisplaySize has unexpected size % instead of 8", it.type.runtime_size);
                case "FramebufferScale";
                    assert(it.offset_in_bytes == 48, "ImDrawData.FramebufferScale has unexpected offset % instead of 48", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImDrawData.FramebufferScale has unexpected size % instead of 8", it.type.runtime_size);
                case "OwnerViewport";
                    assert(it.offset_in_bytes == 56, "ImDrawData.OwnerViewport has unexpected offset % instead of 56", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImDrawData.OwnerViewport has unexpected size % instead of 8", it.type.runtime_size);
                case "Textures";
                    assert(it.offset_in_bytes == 64, "ImDrawData.Textures has unexpected offset % instead of 64", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImDrawData.Textures has unexpected size % instead of 8", it.type.runtime_size);
            }
        }
        assert(size_of(ImDrawData) == 72, "ImDrawData has size % instead of 72", size_of(ImDrawData));
    }

    {
        info := type_info(ImTextureRect);
        for info.members {
            if it.name == {
                case "x";
                    assert(it.offset_in_bytes == 0, "ImTextureRect.x has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImTextureRect.x has unexpected size % instead of 2", it.type.runtime_size);
                case "y";
                    assert(it.offset_in_bytes == 2, "ImTextureRect.y has unexpected offset % instead of 2", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImTextureRect.y has unexpected size % instead of 2", it.type.runtime_size);
                case "w";
                    assert(it.offset_in_bytes == 4, "ImTextureRect.w has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImTextureRect.w has unexpected size % instead of 2", it.type.runtime_size);
                case "h";
                    assert(it.offset_in_bytes == 6, "ImTextureRect.h has unexpected offset % instead of 6", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImTextureRect.h has unexpected size % instead of 2", it.type.runtime_size);
            }
        }
        assert(size_of(ImTextureRect) == 8, "ImTextureRect has size % instead of 8", size_of(ImTextureRect));
    }

    {
        info := type_info(ImTextureData);
        for info.members {
            if it.name == {
                case "UniqueID";
                    assert(it.offset_in_bytes == 0, "ImTextureData.UniqueID has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImTextureData.UniqueID has unexpected size % instead of 4", it.type.runtime_size);
                case "Status";
                    assert(it.offset_in_bytes == 4, "ImTextureData.Status has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImTextureData.Status has unexpected size % instead of 4", it.type.runtime_size);
                case "BackendUserData";
                    assert(it.offset_in_bytes == 8, "ImTextureData.BackendUserData has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImTextureData.BackendUserData has unexpected size % instead of 8", it.type.runtime_size);
                case "TexID";
                    assert(it.offset_in_bytes == 16, "ImTextureData.TexID has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImTextureData.TexID has unexpected size % instead of 8", it.type.runtime_size);
                case "Format";
                    assert(it.offset_in_bytes == 24, "ImTextureData.Format has unexpected offset % instead of 24", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImTextureData.Format has unexpected size % instead of 4", it.type.runtime_size);
                case "Width";
                    assert(it.offset_in_bytes == 28, "ImTextureData.Width has unexpected offset % instead of 28", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImTextureData.Width has unexpected size % instead of 4", it.type.runtime_size);
                case "Height";
                    assert(it.offset_in_bytes == 32, "ImTextureData.Height has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImTextureData.Height has unexpected size % instead of 4", it.type.runtime_size);
                case "BytesPerPixel";
                    assert(it.offset_in_bytes == 36, "ImTextureData.BytesPerPixel has unexpected offset % instead of 36", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImTextureData.BytesPerPixel has unexpected size % instead of 4", it.type.runtime_size);
                case "Pixels";
                    assert(it.offset_in_bytes == 40, "ImTextureData.Pixels has unexpected offset % instead of 40", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImTextureData.Pixels has unexpected size % instead of 8", it.type.runtime_size);
                case "UsedRect";
                    assert(it.offset_in_bytes == 48, "ImTextureData.UsedRect has unexpected offset % instead of 48", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImTextureData.UsedRect has unexpected size % instead of 8", it.type.runtime_size);
                case "UpdateRect";
                    assert(it.offset_in_bytes == 56, "ImTextureData.UpdateRect has unexpected offset % instead of 56", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImTextureData.UpdateRect has unexpected size % instead of 8", it.type.runtime_size);
                case "Updates";
                    assert(it.offset_in_bytes == 64, "ImTextureData.Updates has unexpected offset % instead of 64", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImTextureData.Updates has unexpected size % instead of 16", it.type.runtime_size);
                case "UnusedFrames";
                    assert(it.offset_in_bytes == 80, "ImTextureData.UnusedFrames has unexpected offset % instead of 80", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImTextureData.UnusedFrames has unexpected size % instead of 4", it.type.runtime_size);
                case "RefCount";
                    assert(it.offset_in_bytes == 84, "ImTextureData.RefCount has unexpected offset % instead of 84", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImTextureData.RefCount has unexpected size % instead of 2", it.type.runtime_size);
                case "UseColors";
                    assert(it.offset_in_bytes == 86, "ImTextureData.UseColors has unexpected offset % instead of 86", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImTextureData.UseColors has unexpected size % instead of 1", it.type.runtime_size);
                case "WantDestroyNextFrame";
                    assert(it.offset_in_bytes == 87, "ImTextureData.WantDestroyNextFrame has unexpected offset % instead of 87", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImTextureData.WantDestroyNextFrame has unexpected size % instead of 1", it.type.runtime_size);
            }
        }
        assert(size_of(ImTextureData) == 88, "ImTextureData has size % instead of 88", size_of(ImTextureData));
    }

    {
        info := type_info(ImFontConfig);
        for info.members {
            if it.name == {
                case "Name";
                    assert(it.offset_in_bytes == 0, "ImFontConfig.Name has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 40, "ImFontConfig.Name has unexpected size % instead of 40", it.type.runtime_size);
                case "FontData";
                    assert(it.offset_in_bytes == 40, "ImFontConfig.FontData has unexpected offset % instead of 40", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontConfig.FontData has unexpected size % instead of 8", it.type.runtime_size);
                case "FontDataSize";
                    assert(it.offset_in_bytes == 48, "ImFontConfig.FontDataSize has unexpected offset % instead of 48", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontConfig.FontDataSize has unexpected size % instead of 4", it.type.runtime_size);
                case "FontDataOwnedByAtlas";
                    assert(it.offset_in_bytes == 52, "ImFontConfig.FontDataOwnedByAtlas has unexpected offset % instead of 52", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImFontConfig.FontDataOwnedByAtlas has unexpected size % instead of 1", it.type.runtime_size);
                case "MergeMode";
                    assert(it.offset_in_bytes == 53, "ImFontConfig.MergeMode has unexpected offset % instead of 53", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImFontConfig.MergeMode has unexpected size % instead of 1", it.type.runtime_size);
                case "PixelSnapH";
                    assert(it.offset_in_bytes == 54, "ImFontConfig.PixelSnapH has unexpected offset % instead of 54", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImFontConfig.PixelSnapH has unexpected size % instead of 1", it.type.runtime_size);
                case "PixelSnapV";
                    assert(it.offset_in_bytes == 55, "ImFontConfig.PixelSnapV has unexpected offset % instead of 55", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImFontConfig.PixelSnapV has unexpected size % instead of 1", it.type.runtime_size);
                case "OversampleH";
                    assert(it.offset_in_bytes == 56, "ImFontConfig.OversampleH has unexpected offset % instead of 56", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImFontConfig.OversampleH has unexpected size % instead of 1", it.type.runtime_size);
                case "OversampleV";
                    assert(it.offset_in_bytes == 57, "ImFontConfig.OversampleV has unexpected offset % instead of 57", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImFontConfig.OversampleV has unexpected size % instead of 1", it.type.runtime_size);
                case "EllipsisChar";
                    assert(it.offset_in_bytes == 58, "ImFontConfig.EllipsisChar has unexpected offset % instead of 58", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImFontConfig.EllipsisChar has unexpected size % instead of 2", it.type.runtime_size);
                case "SizePixels";
                    assert(it.offset_in_bytes == 60, "ImFontConfig.SizePixels has unexpected offset % instead of 60", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontConfig.SizePixels has unexpected size % instead of 4", it.type.runtime_size);
                case "GlyphRanges";
                    assert(it.offset_in_bytes == 64, "ImFontConfig.GlyphRanges has unexpected offset % instead of 64", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontConfig.GlyphRanges has unexpected size % instead of 8", it.type.runtime_size);
                case "GlyphExcludeRanges";
                    assert(it.offset_in_bytes == 72, "ImFontConfig.GlyphExcludeRanges has unexpected offset % instead of 72", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontConfig.GlyphExcludeRanges has unexpected size % instead of 8", it.type.runtime_size);
                case "GlyphOffset";
                    assert(it.offset_in_bytes == 80, "ImFontConfig.GlyphOffset has unexpected offset % instead of 80", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontConfig.GlyphOffset has unexpected size % instead of 8", it.type.runtime_size);
                case "GlyphMinAdvanceX";
                    assert(it.offset_in_bytes == 88, "ImFontConfig.GlyphMinAdvanceX has unexpected offset % instead of 88", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontConfig.GlyphMinAdvanceX has unexpected size % instead of 4", it.type.runtime_size);
                case "GlyphMaxAdvanceX";
                    assert(it.offset_in_bytes == 92, "ImFontConfig.GlyphMaxAdvanceX has unexpected offset % instead of 92", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontConfig.GlyphMaxAdvanceX has unexpected size % instead of 4", it.type.runtime_size);
                case "GlyphExtraAdvanceX";
                    assert(it.offset_in_bytes == 96, "ImFontConfig.GlyphExtraAdvanceX has unexpected offset % instead of 96", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontConfig.GlyphExtraAdvanceX has unexpected size % instead of 4", it.type.runtime_size);
                case "FontNo";
                    assert(it.offset_in_bytes == 100, "ImFontConfig.FontNo has unexpected offset % instead of 100", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontConfig.FontNo has unexpected size % instead of 4", it.type.runtime_size);
                case "FontLoaderFlags";
                    assert(it.offset_in_bytes == 104, "ImFontConfig.FontLoaderFlags has unexpected offset % instead of 104", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontConfig.FontLoaderFlags has unexpected size % instead of 4", it.type.runtime_size);
                case "RasterizerMultiply";
                    assert(it.offset_in_bytes == 108, "ImFontConfig.RasterizerMultiply has unexpected offset % instead of 108", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontConfig.RasterizerMultiply has unexpected size % instead of 4", it.type.runtime_size);
                case "RasterizerDensity";
                    assert(it.offset_in_bytes == 112, "ImFontConfig.RasterizerDensity has unexpected offset % instead of 112", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontConfig.RasterizerDensity has unexpected size % instead of 4", it.type.runtime_size);
                case "Flags";
                    assert(it.offset_in_bytes == 116, "ImFontConfig.Flags has unexpected offset % instead of 116", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontConfig.Flags has unexpected size % instead of 4", it.type.runtime_size);
                case "DstFont";
                    assert(it.offset_in_bytes == 120, "ImFontConfig.DstFont has unexpected offset % instead of 120", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontConfig.DstFont has unexpected size % instead of 8", it.type.runtime_size);
                case "FontLoader";
                    assert(it.offset_in_bytes == 128, "ImFontConfig.FontLoader has unexpected offset % instead of 128", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontConfig.FontLoader has unexpected size % instead of 8", it.type.runtime_size);
                case "FontLoaderData";
                    assert(it.offset_in_bytes == 136, "ImFontConfig.FontLoaderData has unexpected offset % instead of 136", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontConfig.FontLoaderData has unexpected size % instead of 8", it.type.runtime_size);
            }
        }
        assert(size_of(ImFontConfig) == 144, "ImFontConfig has size % instead of 144", size_of(ImFontConfig));
    }

    {
        info := type_info(ImFontGlyph);
        for info.members {
            if it.name == {
                case "__bitfield";
                    assert(it.offset_in_bytes == 0, "ImFontGlyph.__bitfield has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontGlyph.__bitfield has unexpected size % instead of 4", it.type.runtime_size);
                case "AdvanceX";
                    assert(it.offset_in_bytes == 4, "ImFontGlyph.AdvanceX has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontGlyph.AdvanceX has unexpected size % instead of 4", it.type.runtime_size);
                case "X0";
                    assert(it.offset_in_bytes == 8, "ImFontGlyph.X0 has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontGlyph.X0 has unexpected size % instead of 4", it.type.runtime_size);
                case "Y0";
                    assert(it.offset_in_bytes == 12, "ImFontGlyph.Y0 has unexpected offset % instead of 12", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontGlyph.Y0 has unexpected size % instead of 4", it.type.runtime_size);
                case "X1";
                    assert(it.offset_in_bytes == 16, "ImFontGlyph.X1 has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontGlyph.X1 has unexpected size % instead of 4", it.type.runtime_size);
                case "Y1";
                    assert(it.offset_in_bytes == 20, "ImFontGlyph.Y1 has unexpected offset % instead of 20", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontGlyph.Y1 has unexpected size % instead of 4", it.type.runtime_size);
                case "U0";
                    assert(it.offset_in_bytes == 24, "ImFontGlyph.U0 has unexpected offset % instead of 24", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontGlyph.U0 has unexpected size % instead of 4", it.type.runtime_size);
                case "V0";
                    assert(it.offset_in_bytes == 28, "ImFontGlyph.V0 has unexpected offset % instead of 28", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontGlyph.V0 has unexpected size % instead of 4", it.type.runtime_size);
                case "U1";
                    assert(it.offset_in_bytes == 32, "ImFontGlyph.U1 has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontGlyph.U1 has unexpected size % instead of 4", it.type.runtime_size);
                case "V1";
                    assert(it.offset_in_bytes == 36, "ImFontGlyph.V1 has unexpected offset % instead of 36", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontGlyph.V1 has unexpected size % instead of 4", it.type.runtime_size);
                case "PackId";
                    assert(it.offset_in_bytes == 40, "ImFontGlyph.PackId has unexpected offset % instead of 40", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontGlyph.PackId has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImFontGlyph) == 44, "ImFontGlyph has size % instead of 44", size_of(ImFontGlyph));
    }

    {
        info := type_info(ImFontGlyphRangesBuilder);
        for info.members {
            if it.name == {
                case "UsedChars";
                    assert(it.offset_in_bytes == 0, "ImFontGlyphRangesBuilder.UsedChars has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImFontGlyphRangesBuilder.UsedChars has unexpected size % instead of 16", it.type.runtime_size);
            }
        }
        assert(size_of(ImFontGlyphRangesBuilder) == 16, "ImFontGlyphRangesBuilder has size % instead of 16", size_of(ImFontGlyphRangesBuilder));
    }

    {
        info := type_info(ImFontAtlasRect);
        for info.members {
            if it.name == {
                case "x";
                    assert(it.offset_in_bytes == 0, "ImFontAtlasRect.x has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImFontAtlasRect.x has unexpected size % instead of 2", it.type.runtime_size);
                case "y";
                    assert(it.offset_in_bytes == 2, "ImFontAtlasRect.y has unexpected offset % instead of 2", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImFontAtlasRect.y has unexpected size % instead of 2", it.type.runtime_size);
                case "w";
                    assert(it.offset_in_bytes == 4, "ImFontAtlasRect.w has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImFontAtlasRect.w has unexpected size % instead of 2", it.type.runtime_size);
                case "h";
                    assert(it.offset_in_bytes == 6, "ImFontAtlasRect.h has unexpected offset % instead of 6", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImFontAtlasRect.h has unexpected size % instead of 2", it.type.runtime_size);
                case "uv0";
                    assert(it.offset_in_bytes == 8, "ImFontAtlasRect.uv0 has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontAtlasRect.uv0 has unexpected size % instead of 8", it.type.runtime_size);
                case "uv1";
                    assert(it.offset_in_bytes == 16, "ImFontAtlasRect.uv1 has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontAtlasRect.uv1 has unexpected size % instead of 8", it.type.runtime_size);
            }
        }
        assert(size_of(ImFontAtlasRect) == 24, "ImFontAtlasRect has size % instead of 24", size_of(ImFontAtlasRect));
    }

    {
        info := type_info(ImFontAtlas);
        for info.members {
            if it.name == {
                case "Flags";
                    assert(it.offset_in_bytes == 0, "ImFontAtlas.Flags has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontAtlas.Flags has unexpected size % instead of 4", it.type.runtime_size);
                case "TexDesiredFormat";
                    assert(it.offset_in_bytes == 4, "ImFontAtlas.TexDesiredFormat has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontAtlas.TexDesiredFormat has unexpected size % instead of 4", it.type.runtime_size);
                case "TexGlyphPadding";
                    assert(it.offset_in_bytes == 8, "ImFontAtlas.TexGlyphPadding has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontAtlas.TexGlyphPadding has unexpected size % instead of 4", it.type.runtime_size);
                case "TexMinWidth";
                    assert(it.offset_in_bytes == 12, "ImFontAtlas.TexMinWidth has unexpected offset % instead of 12", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontAtlas.TexMinWidth has unexpected size % instead of 4", it.type.runtime_size);
                case "TexMinHeight";
                    assert(it.offset_in_bytes == 16, "ImFontAtlas.TexMinHeight has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontAtlas.TexMinHeight has unexpected size % instead of 4", it.type.runtime_size);
                case "TexMaxWidth";
                    assert(it.offset_in_bytes == 20, "ImFontAtlas.TexMaxWidth has unexpected offset % instead of 20", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontAtlas.TexMaxWidth has unexpected size % instead of 4", it.type.runtime_size);
                case "TexMaxHeight";
                    assert(it.offset_in_bytes == 24, "ImFontAtlas.TexMaxHeight has unexpected offset % instead of 24", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontAtlas.TexMaxHeight has unexpected size % instead of 4", it.type.runtime_size);
                case "UserData";
                    assert(it.offset_in_bytes == 32, "ImFontAtlas.UserData has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontAtlas.UserData has unexpected size % instead of 8", it.type.runtime_size);
                case "TexData";
                    assert(it.offset_in_bytes == 56, "ImFontAtlas.TexData has unexpected offset % instead of 56", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontAtlas.TexData has unexpected size % instead of 8", it.type.runtime_size);
                case "TexList";
                    assert(it.offset_in_bytes == 64, "ImFontAtlas.TexList has unexpected offset % instead of 64", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImFontAtlas.TexList has unexpected size % instead of 16", it.type.runtime_size);
                case "Locked";
                    assert(it.offset_in_bytes == 80, "ImFontAtlas.Locked has unexpected offset % instead of 80", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImFontAtlas.Locked has unexpected size % instead of 1", it.type.runtime_size);
                case "RendererHasTextures";
                    assert(it.offset_in_bytes == 81, "ImFontAtlas.RendererHasTextures has unexpected offset % instead of 81", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImFontAtlas.RendererHasTextures has unexpected size % instead of 1", it.type.runtime_size);
                case "TexIsBuilt";
                    assert(it.offset_in_bytes == 82, "ImFontAtlas.TexIsBuilt has unexpected offset % instead of 82", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImFontAtlas.TexIsBuilt has unexpected size % instead of 1", it.type.runtime_size);
                case "TexPixelsUseColors";
                    assert(it.offset_in_bytes == 83, "ImFontAtlas.TexPixelsUseColors has unexpected offset % instead of 83", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImFontAtlas.TexPixelsUseColors has unexpected size % instead of 1", it.type.runtime_size);
                case "TexUvScale";
                    assert(it.offset_in_bytes == 84, "ImFontAtlas.TexUvScale has unexpected offset % instead of 84", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontAtlas.TexUvScale has unexpected size % instead of 8", it.type.runtime_size);
                case "TexUvWhitePixel";
                    assert(it.offset_in_bytes == 92, "ImFontAtlas.TexUvWhitePixel has unexpected offset % instead of 92", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontAtlas.TexUvWhitePixel has unexpected size % instead of 8", it.type.runtime_size);
                case "Fonts";
                    assert(it.offset_in_bytes == 104, "ImFontAtlas.Fonts has unexpected offset % instead of 104", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImFontAtlas.Fonts has unexpected size % instead of 16", it.type.runtime_size);
                case "Sources";
                    assert(it.offset_in_bytes == 120, "ImFontAtlas.Sources has unexpected offset % instead of 120", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImFontAtlas.Sources has unexpected size % instead of 16", it.type.runtime_size);
                case "TexUvLines";
                    assert(it.offset_in_bytes == 136, "ImFontAtlas.TexUvLines has unexpected offset % instead of 136", it.offset_in_bytes);
                    assert(it.type.runtime_size == 528, "ImFontAtlas.TexUvLines has unexpected size % instead of 528", it.type.runtime_size);
                case "TexNextUniqueID";
                    assert(it.offset_in_bytes == 664, "ImFontAtlas.TexNextUniqueID has unexpected offset % instead of 664", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontAtlas.TexNextUniqueID has unexpected size % instead of 4", it.type.runtime_size);
                case "FontNextUniqueID";
                    assert(it.offset_in_bytes == 668, "ImFontAtlas.FontNextUniqueID has unexpected offset % instead of 668", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontAtlas.FontNextUniqueID has unexpected size % instead of 4", it.type.runtime_size);
                case "DrawListSharedDatas";
                    assert(it.offset_in_bytes == 672, "ImFontAtlas.DrawListSharedDatas has unexpected offset % instead of 672", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImFontAtlas.DrawListSharedDatas has unexpected size % instead of 16", it.type.runtime_size);
                case "Builder";
                    assert(it.offset_in_bytes == 688, "ImFontAtlas.Builder has unexpected offset % instead of 688", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontAtlas.Builder has unexpected size % instead of 8", it.type.runtime_size);
                case "FontLoader";
                    assert(it.offset_in_bytes == 696, "ImFontAtlas.FontLoader has unexpected offset % instead of 696", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontAtlas.FontLoader has unexpected size % instead of 8", it.type.runtime_size);
                case "FontLoaderName";
                    assert(it.offset_in_bytes == 704, "ImFontAtlas.FontLoaderName has unexpected offset % instead of 704", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontAtlas.FontLoaderName has unexpected size % instead of 8", it.type.runtime_size);
                case "FontLoaderData";
                    assert(it.offset_in_bytes == 712, "ImFontAtlas.FontLoaderData has unexpected offset % instead of 712", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontAtlas.FontLoaderData has unexpected size % instead of 8", it.type.runtime_size);
                case "FontLoaderFlags";
                    assert(it.offset_in_bytes == 720, "ImFontAtlas.FontLoaderFlags has unexpected offset % instead of 720", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontAtlas.FontLoaderFlags has unexpected size % instead of 4", it.type.runtime_size);
                case "RefCount";
                    assert(it.offset_in_bytes == 724, "ImFontAtlas.RefCount has unexpected offset % instead of 724", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontAtlas.RefCount has unexpected size % instead of 4", it.type.runtime_size);
                case "OwnerContext";
                    assert(it.offset_in_bytes == 728, "ImFontAtlas.OwnerContext has unexpected offset % instead of 728", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontAtlas.OwnerContext has unexpected size % instead of 8", it.type.runtime_size);
                case "TempRect";
                    assert(it.offset_in_bytes == 736, "ImFontAtlas.TempRect has unexpected offset % instead of 736", it.offset_in_bytes);
                    assert(it.type.runtime_size == 24, "ImFontAtlas.TempRect has unexpected size % instead of 24", it.type.runtime_size);
            }
        }
        assert(size_of(ImFontAtlas) == 760, "ImFontAtlas has size % instead of 760", size_of(ImFontAtlas));
    }

    {
        info := type_info(ImFontBaked);
        for info.members {
            if it.name == {
                case "IndexAdvanceX";
                    assert(it.offset_in_bytes == 0, "ImFontBaked.IndexAdvanceX has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImFontBaked.IndexAdvanceX has unexpected size % instead of 16", it.type.runtime_size);
                case "FallbackAdvanceX";
                    assert(it.offset_in_bytes == 16, "ImFontBaked.FallbackAdvanceX has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontBaked.FallbackAdvanceX has unexpected size % instead of 4", it.type.runtime_size);
                case "Size";
                    assert(it.offset_in_bytes == 20, "ImFontBaked.Size has unexpected offset % instead of 20", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontBaked.Size has unexpected size % instead of 4", it.type.runtime_size);
                case "RasterizerDensity";
                    assert(it.offset_in_bytes == 24, "ImFontBaked.RasterizerDensity has unexpected offset % instead of 24", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontBaked.RasterizerDensity has unexpected size % instead of 4", it.type.runtime_size);
                case "IndexLookup";
                    assert(it.offset_in_bytes == 32, "ImFontBaked.IndexLookup has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImFontBaked.IndexLookup has unexpected size % instead of 16", it.type.runtime_size);
                case "Glyphs";
                    assert(it.offset_in_bytes == 48, "ImFontBaked.Glyphs has unexpected offset % instead of 48", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImFontBaked.Glyphs has unexpected size % instead of 16", it.type.runtime_size);
                case "FallbackGlyphIndex";
                    assert(it.offset_in_bytes == 64, "ImFontBaked.FallbackGlyphIndex has unexpected offset % instead of 64", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontBaked.FallbackGlyphIndex has unexpected size % instead of 4", it.type.runtime_size);
                case "Ascent";
                    assert(it.offset_in_bytes == 68, "ImFontBaked.Ascent has unexpected offset % instead of 68", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontBaked.Ascent has unexpected size % instead of 4", it.type.runtime_size);
                case "Descent";
                    assert(it.offset_in_bytes == 72, "ImFontBaked.Descent has unexpected offset % instead of 72", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontBaked.Descent has unexpected size % instead of 4", it.type.runtime_size);
                case "__bitfield";
                    assert(it.offset_in_bytes == 76, "ImFontBaked.__bitfield has unexpected offset % instead of 76", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontBaked.__bitfield has unexpected size % instead of 4", it.type.runtime_size);
                case "LastUsedFrame";
                    assert(it.offset_in_bytes == 80, "ImFontBaked.LastUsedFrame has unexpected offset % instead of 80", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontBaked.LastUsedFrame has unexpected size % instead of 4", it.type.runtime_size);
                case "BakedId";
                    assert(it.offset_in_bytes == 84, "ImFontBaked.BakedId has unexpected offset % instead of 84", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFontBaked.BakedId has unexpected size % instead of 4", it.type.runtime_size);
                case "ContainerFont";
                    assert(it.offset_in_bytes == 88, "ImFontBaked.ContainerFont has unexpected offset % instead of 88", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontBaked.ContainerFont has unexpected size % instead of 8", it.type.runtime_size);
                case "FontLoaderDatas";
                    assert(it.offset_in_bytes == 96, "ImFontBaked.FontLoaderDatas has unexpected offset % instead of 96", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFontBaked.FontLoaderDatas has unexpected size % instead of 8", it.type.runtime_size);
            }
        }
        assert(size_of(ImFontBaked) == 104, "ImFontBaked has size % instead of 104", size_of(ImFontBaked));
    }

    {
        info := type_info(ImFont);
        for info.members {
            if it.name == {
                case "LastBaked";
                    assert(it.offset_in_bytes == 0, "ImFont.LastBaked has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFont.LastBaked has unexpected size % instead of 8", it.type.runtime_size);
                case "ContainerAtlas";
                    assert(it.offset_in_bytes == 8, "ImFont.ContainerAtlas has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImFont.ContainerAtlas has unexpected size % instead of 8", it.type.runtime_size);
                case "Flags";
                    assert(it.offset_in_bytes == 16, "ImFont.Flags has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFont.Flags has unexpected size % instead of 4", it.type.runtime_size);
                case "CurrentRasterizerDensity";
                    assert(it.offset_in_bytes == 20, "ImFont.CurrentRasterizerDensity has unexpected offset % instead of 20", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFont.CurrentRasterizerDensity has unexpected size % instead of 4", it.type.runtime_size);
                case "FontId";
                    assert(it.offset_in_bytes == 24, "ImFont.FontId has unexpected offset % instead of 24", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFont.FontId has unexpected size % instead of 4", it.type.runtime_size);
                case "LegacySize";
                    assert(it.offset_in_bytes == 28, "ImFont.LegacySize has unexpected offset % instead of 28", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFont.LegacySize has unexpected size % instead of 4", it.type.runtime_size);
                case "Sources";
                    assert(it.offset_in_bytes == 32, "ImFont.Sources has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImFont.Sources has unexpected size % instead of 16", it.type.runtime_size);
                case "EllipsisChar";
                    assert(it.offset_in_bytes == 48, "ImFont.EllipsisChar has unexpected offset % instead of 48", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImFont.EllipsisChar has unexpected size % instead of 2", it.type.runtime_size);
                case "FallbackChar";
                    assert(it.offset_in_bytes == 50, "ImFont.FallbackChar has unexpected offset % instead of 50", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImFont.FallbackChar has unexpected size % instead of 2", it.type.runtime_size);
                case "Used8kPagesMap";
                    assert(it.offset_in_bytes == 52, "ImFont.Used8kPagesMap has unexpected offset % instead of 52", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImFont.Used8kPagesMap has unexpected size % instead of 1", it.type.runtime_size);
                case "EllipsisAutoBake";
                    assert(it.offset_in_bytes == 53, "ImFont.EllipsisAutoBake has unexpected offset % instead of 53", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImFont.EllipsisAutoBake has unexpected size % instead of 1", it.type.runtime_size);
                case "RemapPairs";
                    assert(it.offset_in_bytes == 56, "ImFont.RemapPairs has unexpected offset % instead of 56", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImFont.RemapPairs has unexpected size % instead of 16", it.type.runtime_size);
                case "Scale";
                    assert(it.offset_in_bytes == 72, "ImFont.Scale has unexpected offset % instead of 72", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImFont.Scale has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImFont) == 80, "ImFont has size % instead of 80", size_of(ImFont));
    }

    {
        info := type_info(ImGuiViewport);
        for info.members {
            if it.name == {
                case "ID";
                    assert(it.offset_in_bytes == 0, "ImGuiViewport.ID has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiViewport.ID has unexpected size % instead of 4", it.type.runtime_size);
                case "Flags";
                    assert(it.offset_in_bytes == 4, "ImGuiViewport.Flags has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiViewport.Flags has unexpected size % instead of 4", it.type.runtime_size);
                case "Pos";
                    assert(it.offset_in_bytes == 8, "ImGuiViewport.Pos has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiViewport.Pos has unexpected size % instead of 8", it.type.runtime_size);
                case "Size";
                    assert(it.offset_in_bytes == 16, "ImGuiViewport.Size has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiViewport.Size has unexpected size % instead of 8", it.type.runtime_size);
                case "FramebufferScale";
                    assert(it.offset_in_bytes == 24, "ImGuiViewport.FramebufferScale has unexpected offset % instead of 24", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiViewport.FramebufferScale has unexpected size % instead of 8", it.type.runtime_size);
                case "WorkPos";
                    assert(it.offset_in_bytes == 32, "ImGuiViewport.WorkPos has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiViewport.WorkPos has unexpected size % instead of 8", it.type.runtime_size);
                case "WorkSize";
                    assert(it.offset_in_bytes == 40, "ImGuiViewport.WorkSize has unexpected offset % instead of 40", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiViewport.WorkSize has unexpected size % instead of 8", it.type.runtime_size);
                case "PlatformHandle";
                    assert(it.offset_in_bytes == 48, "ImGuiViewport.PlatformHandle has unexpected offset % instead of 48", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiViewport.PlatformHandle has unexpected size % instead of 8", it.type.runtime_size);
                case "PlatformHandleRaw";
                    assert(it.offset_in_bytes == 56, "ImGuiViewport.PlatformHandleRaw has unexpected offset % instead of 56", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiViewport.PlatformHandleRaw has unexpected size % instead of 8", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiViewport) == 64, "ImGuiViewport has size % instead of 64", size_of(ImGuiViewport));
    }

    {
        info := type_info(ImGuiPlatformIO);
        for info.members {
            if it.name == {
                case "Platform_GetClipboardTextFn";
                    assert(it.offset_in_bytes == 0, "ImGuiPlatformIO.Platform_GetClipboardTextFn has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiPlatformIO.Platform_GetClipboardTextFn has unexpected size % instead of 8", it.type.runtime_size);
                case "Platform_SetClipboardTextFn";
                    assert(it.offset_in_bytes == 8, "ImGuiPlatformIO.Platform_SetClipboardTextFn has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiPlatformIO.Platform_SetClipboardTextFn has unexpected size % instead of 8", it.type.runtime_size);
                case "Platform_ClipboardUserData";
                    assert(it.offset_in_bytes == 16, "ImGuiPlatformIO.Platform_ClipboardUserData has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiPlatformIO.Platform_ClipboardUserData has unexpected size % instead of 8", it.type.runtime_size);
                case "Platform_OpenInShellFn";
                    assert(it.offset_in_bytes == 24, "ImGuiPlatformIO.Platform_OpenInShellFn has unexpected offset % instead of 24", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiPlatformIO.Platform_OpenInShellFn has unexpected size % instead of 8", it.type.runtime_size);
                case "Platform_OpenInShellUserData";
                    assert(it.offset_in_bytes == 32, "ImGuiPlatformIO.Platform_OpenInShellUserData has unexpected offset % instead of 32", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiPlatformIO.Platform_OpenInShellUserData has unexpected size % instead of 8", it.type.runtime_size);
                case "Platform_SetImeDataFn";
                    assert(it.offset_in_bytes == 40, "ImGuiPlatformIO.Platform_SetImeDataFn has unexpected offset % instead of 40", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiPlatformIO.Platform_SetImeDataFn has unexpected size % instead of 8", it.type.runtime_size);
                case "Platform_ImeUserData";
                    assert(it.offset_in_bytes == 48, "ImGuiPlatformIO.Platform_ImeUserData has unexpected offset % instead of 48", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiPlatformIO.Platform_ImeUserData has unexpected size % instead of 8", it.type.runtime_size);
                case "Platform_LocaleDecimalPoint";
                    assert(it.offset_in_bytes == 56, "ImGuiPlatformIO.Platform_LocaleDecimalPoint has unexpected offset % instead of 56", it.offset_in_bytes);
                    assert(it.type.runtime_size == 2, "ImGuiPlatformIO.Platform_LocaleDecimalPoint has unexpected size % instead of 2", it.type.runtime_size);
                case "Renderer_TextureMaxWidth";
                    assert(it.offset_in_bytes == 60, "ImGuiPlatformIO.Renderer_TextureMaxWidth has unexpected offset % instead of 60", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiPlatformIO.Renderer_TextureMaxWidth has unexpected size % instead of 4", it.type.runtime_size);
                case "Renderer_TextureMaxHeight";
                    assert(it.offset_in_bytes == 64, "ImGuiPlatformIO.Renderer_TextureMaxHeight has unexpected offset % instead of 64", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiPlatformIO.Renderer_TextureMaxHeight has unexpected size % instead of 4", it.type.runtime_size);
                case "Renderer_RenderState";
                    assert(it.offset_in_bytes == 72, "ImGuiPlatformIO.Renderer_RenderState has unexpected offset % instead of 72", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiPlatformIO.Renderer_RenderState has unexpected size % instead of 8", it.type.runtime_size);
                case "Textures";
                    assert(it.offset_in_bytes == 80, "ImGuiPlatformIO.Textures has unexpected offset % instead of 80", it.offset_in_bytes);
                    assert(it.type.runtime_size == 16, "ImGuiPlatformIO.Textures has unexpected size % instead of 16", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiPlatformIO) == 96, "ImGuiPlatformIO has size % instead of 96", size_of(ImGuiPlatformIO));
    }

    {
        info := type_info(ImGuiPlatformImeData);
        for info.members {
            if it.name == {
                case "WantVisible";
                    assert(it.offset_in_bytes == 0, "ImGuiPlatformImeData.WantVisible has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiPlatformImeData.WantVisible has unexpected size % instead of 1", it.type.runtime_size);
                case "WantTextInput";
                    assert(it.offset_in_bytes == 1, "ImGuiPlatformImeData.WantTextInput has unexpected offset % instead of 1", it.offset_in_bytes);
                    assert(it.type.runtime_size == 1, "ImGuiPlatformImeData.WantTextInput has unexpected size % instead of 1", it.type.runtime_size);
                case "InputPos";
                    assert(it.offset_in_bytes == 4, "ImGuiPlatformImeData.InputPos has unexpected offset % instead of 4", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGuiPlatformImeData.InputPos has unexpected size % instead of 8", it.type.runtime_size);
                case "InputLineHeight";
                    assert(it.offset_in_bytes == 12, "ImGuiPlatformImeData.InputLineHeight has unexpected offset % instead of 12", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiPlatformImeData.InputLineHeight has unexpected size % instead of 4", it.type.runtime_size);
                case "ViewportId";
                    assert(it.offset_in_bytes == 16, "ImGuiPlatformImeData.ViewportId has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGuiPlatformImeData.ViewportId has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImGuiPlatformImeData) == 20, "ImGuiPlatformImeData has size % instead of 20", size_of(ImGuiPlatformImeData));
    }

    {
        info := type_info(ImGui_ImplSDLGPU3_InitInfo);
        for info.members {
            if it.name == {
                case "Device";
                    assert(it.offset_in_bytes == 0, "ImGui_ImplSDLGPU3_InitInfo.Device has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGui_ImplSDLGPU3_InitInfo.Device has unexpected size % instead of 8", it.type.runtime_size);
                case "ColorTargetFormat";
                    assert(it.offset_in_bytes == 8, "ImGui_ImplSDLGPU3_InitInfo.ColorTargetFormat has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGui_ImplSDLGPU3_InitInfo.ColorTargetFormat has unexpected size % instead of 4", it.type.runtime_size);
                case "MSAASamples";
                    assert(it.offset_in_bytes == 12, "ImGui_ImplSDLGPU3_InitInfo.MSAASamples has unexpected offset % instead of 12", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGui_ImplSDLGPU3_InitInfo.MSAASamples has unexpected size % instead of 4", it.type.runtime_size);
                case "SwapchainComposition";
                    assert(it.offset_in_bytes == 16, "ImGui_ImplSDLGPU3_InitInfo.SwapchainComposition has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGui_ImplSDLGPU3_InitInfo.SwapchainComposition has unexpected size % instead of 4", it.type.runtime_size);
                case "PresentMode";
                    assert(it.offset_in_bytes == 20, "ImGui_ImplSDLGPU3_InitInfo.PresentMode has unexpected offset % instead of 20", it.offset_in_bytes);
                    assert(it.type.runtime_size == 4, "ImGui_ImplSDLGPU3_InitInfo.PresentMode has unexpected size % instead of 4", it.type.runtime_size);
            }
        }
        assert(size_of(ImGui_ImplSDLGPU3_InitInfo) == 24, "ImGui_ImplSDLGPU3_InitInfo has size % instead of 24", size_of(ImGui_ImplSDLGPU3_InitInfo));
    }

    {
        info := type_info(ImGui_ImplSDLGPU3_RenderState);
        for info.members {
            if it.name == {
                case "Device";
                    assert(it.offset_in_bytes == 0, "ImGui_ImplSDLGPU3_RenderState.Device has unexpected offset % instead of 0", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGui_ImplSDLGPU3_RenderState.Device has unexpected size % instead of 8", it.type.runtime_size);
                case "SamplerDefault";
                    assert(it.offset_in_bytes == 8, "ImGui_ImplSDLGPU3_RenderState.SamplerDefault has unexpected offset % instead of 8", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGui_ImplSDLGPU3_RenderState.SamplerDefault has unexpected size % instead of 8", it.type.runtime_size);
                case "SamplerCurrent";
                    assert(it.offset_in_bytes == 16, "ImGui_ImplSDLGPU3_RenderState.SamplerCurrent has unexpected offset % instead of 16", it.offset_in_bytes);
                    assert(it.type.runtime_size == 8, "ImGui_ImplSDLGPU3_RenderState.SamplerCurrent has unexpected size % instead of 8", it.type.runtime_size);
            }
        }
        assert(size_of(ImGui_ImplSDLGPU3_RenderState) == 24, "ImGui_ImplSDLGPU3_RenderState has size % instead of 24", size_of(ImGui_ImplSDLGPU3_RenderState));
    }
}

    FLOAT32_MIN :: 0h00800000;
    FLOAT32_MAX :: 0h7F7FFFFF;
    #import "sdl3";
