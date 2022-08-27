import imgui

""" Convenience consts and functions to make copy-pasting C++ style configs easier """

# ==== Color identifiers for styling ====
# https://github.com/pyimgui/pyimgui/blob/1.4.0/imgui/core.pyx#L193
ImGuiCol_Text = imgui.COLOR_TEXT
ImGuiCol_TextDisabled = imgui.COLOR_TEXT_DISABLED
ImGuiCol_WindowBg = imgui.COLOR_WINDOW_BACKGROUND
ImGuiCol_ChildBg = imgui.COLOR_CHILD_BACKGROUND
ImGuiCol_PopupBg = imgui.COLOR_POPUP_BACKGROUND
ImGuiCol_Border = imgui.COLOR_BORDER
ImGuiCol_BorderShadow = imgui.COLOR_BORDER_SHADOW
ImGuiCol_FrameBg = imgui.COLOR_FRAME_BACKGROUND
ImGuiCol_FrameBgHovered = imgui.COLOR_FRAME_BACKGROUND_HOVERED
ImGuiCol_FrameBgActive = imgui.COLOR_FRAME_BACKGROUND_ACTIVE
ImGuiCol_TitleBg = imgui.COLOR_TITLE_BACKGROUND
ImGuiCol_TitleBgActive = imgui.COLOR_TITLE_BACKGROUND_ACTIVE
ImGuiCol_TitleBgCollapsed = imgui.COLOR_TITLE_BACKGROUND_COLLAPSED
ImGuiCol_MenuBarBg = imgui.COLOR_MENUBAR_BACKGROUND
ImGuiCol_ScrollbarBg = imgui.COLOR_SCROLLBAR_BACKGROUND
ImGuiCol_ScrollbarGrab = imgui.COLOR_SCROLLBAR_GRAB
ImGuiCol_ScrollbarGrabHovered = imgui.COLOR_SCROLLBAR_GRAB_HOVERED
ImGuiCol_ScrollbarGrabActive = imgui.COLOR_SCROLLBAR_GRAB_ACTIVE
ImGuiCol_CheckMark = imgui.COLOR_CHECK_MARK
ImGuiCol_SliderGrab = imgui.COLOR_SLIDER_GRAB
ImGuiCol_SliderGrabActive = imgui.COLOR_SLIDER_GRAB_ACTIVE
ImGuiCol_Button = imgui.COLOR_BUTTON
ImGuiCol_ButtonHovered = imgui.COLOR_BUTTON_HOVERED
ImGuiCol_ButtonActive = imgui.COLOR_BUTTON_ACTIVE
ImGuiCol_Header = imgui.COLOR_HEADER
ImGuiCol_HeaderHovered = imgui.COLOR_HEADER_HOVERED
ImGuiCol_HeaderActive = imgui.COLOR_HEADER_ACTIVE
ImGuiCol_Separator = imgui.COLOR_SEPARATOR
ImGuiCol_SeparatorHovered = imgui.COLOR_SEPARATOR_HOVERED
ImGuiCol_SeparatorActive = imgui.COLOR_SEPARATOR_ACTIVE
ImGuiCol_ResizeGrip = imgui.COLOR_RESIZE_GRIP
ImGuiCol_ResizeGripHovered = imgui.COLOR_RESIZE_GRIP_HOVERED
ImGuiCol_ResizeGripActive = imgui.COLOR_RESIZE_GRIP_ACTIVE
ImGuiCol_PlotLines = imgui.COLOR_PLOT_LINES
ImGuiCol_PlotLinesHovered = imgui.COLOR_PLOT_LINES_HOVERED
ImGuiCol_PlotHistogram = imgui.COLOR_PLOT_HISTOGRAM
ImGuiCol_PlotHistogramHovered = imgui.COLOR_PLOT_HISTOGRAM_HOVERED
ImGuiCol_TextSelectedBg = imgui.COLOR_TEXT_SELECTED_BACKGROUND
ImGuiCol_DragDropTarget = imgui.COLOR_DRAG_DROP_TARGET
ImGuiCol_NavHighlight = imgui.COLOR_NAV_HIGHLIGHT
ImGuiCol_NavWindowingHighlight = imgui.COLOR_NAV_WINDOWING_HIGHLIGHT
ImGuiCol_NavWindowingDimBg = imgui.COLOR_NAV_WINDOWING_DIM_BACKGROUND
ImGuiCol_ModalWindowDimBg = imgui.COLOR_MODAL_WINDOW_DIM_BACKGROUND
ImGuiCol_COUNT = imgui.COLOR_COUNT
# ImGuiCol_Tab = ''
# ImGuiCol_TabHovered = ''
# ImGuiCol_TabActive = ''
# ImGuiCol_TabUnfocused = ''
# ImGuiCol_TabUnfocusedActive = ''
# ImGuiCol_DockingPreview = ''
# ImGuiCol_DockingEmptyBg = ''
# ImGuiCol_TableHeaderBg = ''
# ImGuiCol_TableBorderStrong = ''
# ImGuiCol_TableBorderLight = ''
# ImGuiCol_TableRowBg = ''
# ImGuiCol_TableRowBgAlt = ''

# style consts in v1.4.0
Alpha = 'alpha'
AntiAliasedFill = 'anti_aliased_fill'
AntiAliasedLines = 'anti_aliased_lines'
ButtonTextAlign = 'button_text_align'
ChildBorderSize = 'child_border_size'
ChildRounding = 'child_rounding'
Color = 'color'
ColumnsMinSpacing = 'columns_min_spacing'
CurveTessellationTolerance = 'curve_tessellation_tolerance'
DisplaySafeAreaPadding = 'display_safe_area_padding'
DisplayWindowPadding = 'display_window_padding'
FrameBorderSize = 'frame_border_size'
FramePadding = 'frame_padding'
FrameRounding = 'frame_rounding'
GrabMinSize = 'grab_min_size'
GrabRounding = 'grab_rounding'
IndentSpacing = 'indent_spacing'
ItemInnerSpacing = 'item_inner_spacing'
ItemSpacing = 'item_spacing'
MouseCursorScale = 'mouse_cursor_scale'
PopupBorderSize = 'popup_border_size'
PopupRounding = 'popup_rounding'
ScrollbarRounding = 'scrollbar_rounding'
ScrollbarSize = 'scrollbar_size'
TouchExtraPadding = 'touch_extra_padding'
WindowBorderSize = 'window_border_size'
WindowMinSize = 'window_min_size'
WindowPadding = 'window_padding'
WindowRounding = 'window_rounding'
WindowTitleAlign = 'window_title_align'
# DisabledAlpha = ''
# WindowMenuButtonPosition = ''
# CellPadding = ''
# TabRounding = ''
# TabBorderSize = ''
# TabMinWidthForCloseButton = ''
# ColorButtonPosition = ''
# ButtonTextAlign = ''
# SelectableTextAlign = ''
# LogSliderDeadzone = ''

def color(hex):
    hex = hex.lstrip('#')
    rgba = (int(hex[i:i+2], 16) / 255.0 for i in (0, 2, 4, 6))
    return imgui.Vec4(*rgba)

# Photoshop style by Derydoca from ImThemes (https://github.com/Patitotective/ImThemes/releases)
def theme_ps():
    s = imgui.get_style()

    setattr(s, Alpha, 1.0)
    #setattr(s, DisabledAlpha, 0.6000000238418579)
    setattr(s, WindowPadding, imgui.Vec2(8.0, 8.0))
    setattr(s, WindowRounding, 4.0)
    setattr(s, WindowBorderSize, 1.0)
    setattr(s, WindowMinSize, imgui.Vec2(32.0, 32.0))
    setattr(s, WindowTitleAlign, imgui.Vec2(0.0, 0.5))
    #setattr(s, WindowMenuButtonPosition, imgui.DIRECTION_LEFT)
    setattr(s, ChildRounding, 4.0)
    setattr(s, ChildBorderSize, 1.0)
    setattr(s, PopupRounding, 2.0)
    setattr(s, PopupBorderSize, 1.0)
    setattr(s, FramePadding, imgui.Vec2(4.0, 3.0))
    setattr(s, FrameRounding, 2.0)
    setattr(s, FrameBorderSize, 1.0)
    setattr(s, ItemSpacing, imgui.Vec2(8.0, 4.0))
    setattr(s, ItemInnerSpacing, imgui.Vec2(4.0, 4.0))
    #setattr(s, CellPadding, imgui.Vec2(4.0, 2.0))
    setattr(s, IndentSpacing, 21.0)
    setattr(s, ColumnsMinSpacing, 6.0)
    setattr(s, ScrollbarSize, 13.0)
    setattr(s, ScrollbarRounding, 12.0)
    setattr(s, GrabMinSize, 7.0)
    setattr(s, GrabRounding, 0.0)
    #setattr(s, TabRounding, 0.0)
    #setattr(s, TabBorderSize, 1.0)
    #setattr(s, TabMinWidthForCloseButton, 0.0)
    #setattr(s, ColorButtonPosition, imgui.DIRECTION_RIGHT)
    setattr(s, ButtonTextAlign, imgui.Vec2(0.5, 0.5))
    #setattr(s, SelectableTextAlign, imgui.Vec2(0.0, 0.0))
    
    s.colors[ImGuiCol_Text] =                  imgui.Vec4(1.0, 1.0, 1.0, 1.0)
    s.colors[ImGuiCol_TextDisabled] =          imgui.Vec4(0.4980392158031464, 0.4980392158031464, 0.4980392158031464, 1.0)
    s.colors[ImGuiCol_WindowBg] =              imgui.Vec4(0.1764705926179886, 0.1764705926179886, 0.1764705926179886, 1.0)
    s.colors[ImGuiCol_ChildBg] =               imgui.Vec4(0.2784313857555389, 0.2784313857555389, 0.2784313857555389, 0.0)
    s.colors[ImGuiCol_PopupBg] =               imgui.Vec4(0.3098039329051971, 0.3098039329051971, 0.3098039329051971, 1.0)
    s.colors[ImGuiCol_Border] =                imgui.Vec4(0.2627451121807098, 0.2627451121807098, 0.2627451121807098, 1.0)
    s.colors[ImGuiCol_BorderShadow] =          imgui.Vec4(0.0, 0.0, 0.0, 0.0)
    s.colors[ImGuiCol_FrameBg] =               imgui.Vec4(0.1568627506494522, 0.1568627506494522, 0.1568627506494522, 1.0)
    s.colors[ImGuiCol_FrameBgHovered] =        imgui.Vec4(0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 1.0)
    s.colors[ImGuiCol_FrameBgActive] =         imgui.Vec4(0.2784313857555389, 0.2784313857555389, 0.2784313857555389, 1.0)
    s.colors[ImGuiCol_TitleBg] =               imgui.Vec4(0.1450980454683304, 0.1450980454683304, 0.1450980454683304, 1.0)
    s.colors[ImGuiCol_TitleBgActive] =         imgui.Vec4(0.1450980454683304, 0.1450980454683304, 0.1450980454683304, 1.0)
    s.colors[ImGuiCol_TitleBgCollapsed] =      imgui.Vec4(0.1450980454683304, 0.1450980454683304, 0.1450980454683304, 1.0)
    s.colors[ImGuiCol_MenuBarBg] =             imgui.Vec4(0.1921568661928177, 0.1921568661928177, 0.1921568661928177, 1.0)
    s.colors[ImGuiCol_ScrollbarBg] =           imgui.Vec4(0.1568627506494522, 0.1568627506494522, 0.1568627506494522, 1.0)
    s.colors[ImGuiCol_ScrollbarGrab] =         imgui.Vec4(0.2745098173618317, 0.2745098173618317, 0.2745098173618317, 1.0)
    s.colors[ImGuiCol_ScrollbarGrabHovered] =  imgui.Vec4(0.2980392277240753, 0.2980392277240753, 0.2980392277240753, 1.0)
    s.colors[ImGuiCol_ScrollbarGrabActive] =   imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_CheckMark] =             imgui.Vec4(1.0, 1.0, 1.0, 1.0)
    s.colors[ImGuiCol_SliderGrab] =            imgui.Vec4(0.3882353007793427, 0.3882353007793427, 0.3882353007793427, 1.0)
    s.colors[ImGuiCol_SliderGrabActive] =      imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_Button] =                imgui.Vec4(1.0, 1.0, 1.0, 0.0)
    s.colors[ImGuiCol_ButtonHovered] =         imgui.Vec4(1.0, 1.0, 1.0, 0.1560000032186508)
    s.colors[ImGuiCol_ButtonActive] =          imgui.Vec4(1.0, 1.0, 1.0, 0.3910000026226044)
    s.colors[ImGuiCol_Header] =                imgui.Vec4(0.3098039329051971, 0.3098039329051971, 0.3098039329051971, 1.0)
    s.colors[ImGuiCol_HeaderHovered] =         imgui.Vec4(0.4666666686534882, 0.4666666686534882, 0.4666666686534882, 1.0)
    s.colors[ImGuiCol_HeaderActive] =          imgui.Vec4(0.4666666686534882, 0.4666666686534882, 0.4666666686534882, 1.0)
    s.colors[ImGuiCol_Separator] =             imgui.Vec4(0.2627451121807098, 0.2627451121807098, 0.2627451121807098, 1.0)
    s.colors[ImGuiCol_SeparatorHovered] =      imgui.Vec4(0.3882353007793427, 0.3882353007793427, 0.3882353007793427, 1.0)
    s.colors[ImGuiCol_SeparatorActive] =       imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_ResizeGrip] =            imgui.Vec4(1.0, 1.0, 1.0, 0.25)
    s.colors[ImGuiCol_ResizeGripHovered] =     imgui.Vec4(1.0, 1.0, 1.0, 0.6700000166893005)
    s.colors[ImGuiCol_ResizeGripActive] =      imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_PlotLines] =             imgui.Vec4(0.4666666686534882, 0.4666666686534882, 0.4666666686534882, 1.0)
    s.colors[ImGuiCol_PlotLinesHovered] =      imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_PlotHistogram] =         imgui.Vec4(0.5843137502670288, 0.5843137502670288, 0.5843137502670288, 1.0)
    s.colors[ImGuiCol_PlotHistogramHovered] =  imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_TextSelectedBg] =        imgui.Vec4(1.0, 1.0, 1.0, 0.1560000032186508)
    s.colors[ImGuiCol_DragDropTarget] =        imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_NavHighlight] =          imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_NavWindowingHighlight] = imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_NavWindowingDimBg] =     imgui.Vec4(0.0, 0.0, 0.0, 0.5860000252723694)
    s.colors[ImGuiCol_ModalWindowDimBg] =      imgui.Vec4(0.0, 0.0, 0.0, 0.5860000252723694)

# https://github.com/ocornut/imgui/issues/707#issuecomment-917151020
def theme_deep_dark():
    s = imgui.get_style()

    # setattr(s, WindowPadding,     imgui.Vec2(8.00, 8.00))
    # setattr(s, FramePadding,      imgui.Vec2(5.00, 2.00))
    # #setattr(s, CellPadding,       imgui.Vec2(6.00, 6.00))
    # setattr(s, ItemSpacing,       imgui.Vec2(6.00, 6.00))
    # setattr(s, ItemInnerSpacing,  imgui.Vec2(6.00, 6.00))
    # setattr(s, TouchExtraPadding, imgui.Vec2(0.00, 0.00))
    # setattr(s, IndentSpacing,     25)
    # setattr(s, ScrollbarSize,     15)
    #setattr(s, GrabMinSize,       10)
    setattr(s, WindowBorderSize,  1)
    setattr(s, ChildBorderSize,   1)
    setattr(s, PopupBorderSize,   1)
    setattr(s, FrameBorderSize,   1)
    # #setattr(s, TabBorderSize,     1)
    # setattr(s, WindowRounding,    7)
    setattr(s, ChildRounding,     4)
    setattr(s, FrameRounding,     3)
    setattr(s, PopupRounding,     4)
    setattr(s, ScrollbarRounding, 5)
    setattr(s, GrabRounding,      3)
    # #setattr(s, LogSliderDeadzone, 4)
    # #setattr(s, TabRounding,       4)
    
    s.colors[ImGuiCol_Text]                   = imgui.Vec4(1.00, 1.00, 1.00, 1.00)
    s.colors[ImGuiCol_TextDisabled]           = imgui.Vec4(0.50, 0.50, 0.50, 1.00)
    s.colors[ImGuiCol_WindowBg]               = imgui.Vec4(0.10, 0.10, 0.10, 1.00)
    s.colors[ImGuiCol_ChildBg]                = imgui.Vec4(0.00, 0.00, 0.00, 0.00)
    s.colors[ImGuiCol_PopupBg]                = imgui.Vec4(0.19, 0.19, 0.19, 0.92)
    s.colors[ImGuiCol_Border]                 = imgui.Vec4(0.19, 0.19, 0.19, 0.29)
    s.colors[ImGuiCol_BorderShadow]           = imgui.Vec4(0.00, 0.00, 0.00, 0.24)
    s.colors[ImGuiCol_FrameBg]                = imgui.Vec4(0.05, 0.05, 0.05, 0.54)
    s.colors[ImGuiCol_FrameBgHovered]         = imgui.Vec4(0.19, 0.19, 0.19, 0.54)
    s.colors[ImGuiCol_FrameBgActive]          = imgui.Vec4(0.20, 0.22, 0.23, 1.00)
    s.colors[ImGuiCol_TitleBg]                = imgui.Vec4(0.00, 0.00, 0.00, 1.00)
    s.colors[ImGuiCol_TitleBgActive]          = imgui.Vec4(0.06, 0.06, 0.06, 1.00)
    s.colors[ImGuiCol_TitleBgCollapsed]       = imgui.Vec4(0.00, 0.00, 0.00, 1.00)
    s.colors[ImGuiCol_MenuBarBg]              = imgui.Vec4(0.14, 0.14, 0.14, 1.00)
    s.colors[ImGuiCol_ScrollbarBg]            = imgui.Vec4(0.05, 0.05, 0.05, 0.54)
    s.colors[ImGuiCol_ScrollbarGrab]          = imgui.Vec4(0.34, 0.34, 0.34, 0.54)
    s.colors[ImGuiCol_ScrollbarGrabHovered]   = imgui.Vec4(0.40, 0.40, 0.40, 0.54)
    s.colors[ImGuiCol_ScrollbarGrabActive]    = imgui.Vec4(0.56, 0.56, 0.56, 0.54)
    s.colors[ImGuiCol_CheckMark]              = imgui.Vec4(0.33, 0.67, 0.86, 1.00)
    s.colors[ImGuiCol_SliderGrab]             = imgui.Vec4(0.34, 0.34, 0.34, 0.54)
    s.colors[ImGuiCol_SliderGrabActive]       = imgui.Vec4(0.56, 0.56, 0.56, 0.54)
    s.colors[ImGuiCol_Button]                 = imgui.Vec4(0.05, 0.05, 0.05, 0.54)
    s.colors[ImGuiCol_ButtonHovered]          = imgui.Vec4(0.19, 0.19, 0.19, 0.54)
    s.colors[ImGuiCol_ButtonActive]           = imgui.Vec4(0.20, 0.22, 0.23, 1.00)
    s.colors[ImGuiCol_Header]                 = imgui.Vec4(0.00, 0.00, 0.00, 0.52)
    s.colors[ImGuiCol_HeaderHovered]          = imgui.Vec4(0.00, 0.00, 0.00, 0.36)
    s.colors[ImGuiCol_HeaderActive]           = imgui.Vec4(0.20, 0.22, 0.23, 0.33)
    s.colors[ImGuiCol_Separator]              = imgui.Vec4(0.28, 0.28, 0.28, 0.29)
    s.colors[ImGuiCol_SeparatorHovered]       = imgui.Vec4(0.44, 0.44, 0.44, 0.29)
    s.colors[ImGuiCol_SeparatorActive]        = imgui.Vec4(0.40, 0.44, 0.47, 1.00)
    s.colors[ImGuiCol_ResizeGrip]             = imgui.Vec4(0.28, 0.28, 0.28, 0.29)
    s.colors[ImGuiCol_ResizeGripHovered]      = imgui.Vec4(0.44, 0.44, 0.44, 0.29)
    s.colors[ImGuiCol_ResizeGripActive]       = imgui.Vec4(0.40, 0.44, 0.47, 1.00)
    #s.colors[ImGuiCol_Tab]                    = imgui.Vec4(0.00, 0.00, 0.00, 0.52)
    #s.colors[ImGuiCol_TabHovered]             = imgui.Vec4(0.14, 0.14, 0.14, 1.00)
    #s.colors[ImGuiCol_TabActive]              = imgui.Vec4(0.20, 0.20, 0.20, 0.36)
    #s.colors[ImGuiCol_TabUnfocused]           = imgui.Vec4(0.00, 0.00, 0.00, 0.52)
    #s.colors[ImGuiCol_TabUnfocusedActive]     = imgui.Vec4(0.14, 0.14, 0.14, 1.00)
    #s.colors[ImGuiCol_DockingPreview]         = imgui.Vec4(0.33, 0.67, 0.86, 1.00)
    #s.colors[ImGuiCol_DockingEmptyBg]         = imgui.Vec4(1.00, 0.00, 0.00, 1.00)
    s.colors[ImGuiCol_PlotLines]              = imgui.Vec4(1.00, 0.00, 0.00, 1.00)
    s.colors[ImGuiCol_PlotLinesHovered]       = imgui.Vec4(1.00, 0.00, 0.00, 1.00)
    s.colors[ImGuiCol_PlotHistogram]          = imgui.Vec4(1.00, 0.00, 0.00, 1.00)
    s.colors[ImGuiCol_PlotHistogramHovered]   = imgui.Vec4(1.00, 0.00, 0.00, 1.00)
    #s.colors[ImGuiCol_TableHeaderBg]          = imgui.Vec4(0.00, 0.00, 0.00, 0.52)
    #s.colors[ImGuiCol_TableBorderStrong]      = imgui.Vec4(0.00, 0.00, 0.00, 0.52)
    #s.colors[ImGuiCol_TableBorderLight]       = imgui.Vec4(0.28, 0.28, 0.28, 0.29)
    #s.colors[ImGuiCol_TableRowBg]             = imgui.Vec4(0.00, 0.00, 0.00, 0.00)
    #s.colors[ImGuiCol_TableRowBgAlt]          = imgui.Vec4(1.00, 1.00, 1.00, 0.06)
    s.colors[ImGuiCol_TextSelectedBg]         = imgui.Vec4(0.20, 0.22, 0.23, 1.00)
    s.colors[ImGuiCol_DragDropTarget]         = imgui.Vec4(0.33, 0.67, 0.86, 1.00)
    s.colors[ImGuiCol_NavHighlight]           = imgui.Vec4(1.00, 0.00, 0.00, 1.00)
    s.colors[ImGuiCol_NavWindowingHighlight]  = imgui.Vec4(1.00, 0.00, 0.00, 0.70)
    s.colors[ImGuiCol_NavWindowingDimBg]      = imgui.Vec4(1.00, 0.00, 0.00, 0.20)
    s.colors[ImGuiCol_ModalWindowDimBg]       = imgui.Vec4(1.00, 0.00, 0.00, 0.35)

# Deep-dark with ps-style borders
def theme_contrast():
    theme_deep_dark()
    s = imgui.get_style()
    s.colors[ImGuiCol_Border] = imgui.Vec4(0.26, 0.26, 0.26, 1.0)
    s.colors[ImGuiCol_BorderShadow] = imgui.Vec4(0.0, 0.0, 0.0, 0.0)
    setattr(s, ChildBorderSize, 1.0)
    setattr(s, PopupBorderSize, 1.0)
    setattr(s, FrameBorderSize, 1.0)

# https://github.com/ocornut/imgui/issues/707#issuecomment-678611331
def theme_dark_overshifted():
    s = imgui.get_style()
    
    s.colors[ImGuiCol_Text]                  = imgui.Vec4(1.00, 1.00, 1.00, 1.00)
    s.colors[ImGuiCol_TextDisabled]          = imgui.Vec4(0.50, 0.50, 0.50, 1.00)
    s.colors[ImGuiCol_WindowBg]              = imgui.Vec4(0.13, 0.14, 0.15, 1.00)
    s.colors[ImGuiCol_ChildBg]               = imgui.Vec4(0.13, 0.14, 0.15, 1.00)
    s.colors[ImGuiCol_PopupBg]               = imgui.Vec4(0.13, 0.14, 0.15, 1.00)
    s.colors[ImGuiCol_Border]                = imgui.Vec4(0.43, 0.43, 0.50, 0.50)
    s.colors[ImGuiCol_BorderShadow]          = imgui.Vec4(0.00, 0.00, 0.00, 0.00)
    s.colors[ImGuiCol_FrameBg]               = imgui.Vec4(0.25, 0.25, 0.25, 1.00)
    s.colors[ImGuiCol_FrameBgHovered]        = imgui.Vec4(0.38, 0.38, 0.38, 1.00)
    s.colors[ImGuiCol_FrameBgActive]         = imgui.Vec4(0.67, 0.67, 0.67, 0.39)
    s.colors[ImGuiCol_TitleBg]               = imgui.Vec4(0.08, 0.08, 0.09, 1.00)
    s.colors[ImGuiCol_TitleBgActive]         = imgui.Vec4(0.08, 0.08, 0.09, 1.00)
    s.colors[ImGuiCol_TitleBgCollapsed]      = imgui.Vec4(0.00, 0.00, 0.00, 0.51)
    s.colors[ImGuiCol_MenuBarBg]             = imgui.Vec4(0.14, 0.14, 0.14, 1.00)
    s.colors[ImGuiCol_ScrollbarBg]           = imgui.Vec4(0.02, 0.02, 0.02, 0.53)
    s.colors[ImGuiCol_ScrollbarGrab]         = imgui.Vec4(0.31, 0.31, 0.31, 1.00)
    s.colors[ImGuiCol_ScrollbarGrabHovered]  = imgui.Vec4(0.41, 0.41, 0.41, 1.00)
    s.colors[ImGuiCol_ScrollbarGrabActive]   = imgui.Vec4(0.51, 0.51, 0.51, 1.00)
    s.colors[ImGuiCol_CheckMark]             = imgui.Vec4(0.11, 0.64, 0.92, 1.00)
    s.colors[ImGuiCol_SliderGrab]            = imgui.Vec4(0.11, 0.64, 0.92, 1.00)
    s.colors[ImGuiCol_SliderGrabActive]      = imgui.Vec4(0.08, 0.50, 0.72, 1.00)
    s.colors[ImGuiCol_Button]                = imgui.Vec4(0.25, 0.25, 0.25, 1.00)
    s.colors[ImGuiCol_ButtonHovered]         = imgui.Vec4(0.38, 0.38, 0.38, 1.00)
    s.colors[ImGuiCol_ButtonActive]          = imgui.Vec4(0.67, 0.67, 0.67, 0.39)
    s.colors[ImGuiCol_Header]                = imgui.Vec4(0.22, 0.22, 0.22, 1.00)
    s.colors[ImGuiCol_HeaderHovered]         = imgui.Vec4(0.25, 0.25, 0.25, 1.00)
    s.colors[ImGuiCol_HeaderActive]          = imgui.Vec4(0.67, 0.67, 0.67, 0.39)
    s.colors[ImGuiCol_Separator]             = s.colors[ImGuiCol_Border]
    s.colors[ImGuiCol_SeparatorHovered]      = imgui.Vec4(0.41, 0.42, 0.44, 1.00)
    s.colors[ImGuiCol_SeparatorActive]       = imgui.Vec4(0.26, 0.59, 0.98, 0.95)
    s.colors[ImGuiCol_ResizeGrip]            = imgui.Vec4(0.00, 0.00, 0.00, 0.00)
    s.colors[ImGuiCol_ResizeGripHovered]     = imgui.Vec4(0.29, 0.30, 0.31, 0.67)
    s.colors[ImGuiCol_ResizeGripActive]      = imgui.Vec4(0.26, 0.59, 0.98, 0.95)
    #s.colors[ImGuiCol_Tab]                   = imgui.Vec4(0.08, 0.08, 0.09, 0.83)
    #s.colors[ImGuiCol_TabHovered]            = imgui.Vec4(0.33, 0.34, 0.36, 0.83)
    #s.colors[ImGuiCol_TabActive]             = imgui.Vec4(0.23, 0.23, 0.24, 1.00)
    #s.colors[ImGuiCol_TabUnfocused]          = imgui.Vec4(0.08, 0.08, 0.09, 1.00)
    #s.colors[ImGuiCol_TabUnfocusedActive]    = imgui.Vec4(0.13, 0.14, 0.15, 1.00)
    #s.colors[ImGuiCol_DockingPreview]        = imgui.Vec4(0.26, 0.59, 0.98, 0.70)
    #s.colors[ImGuiCol_DockingEmptyBg]        = imgui.Vec4(0.20, 0.20, 0.20, 1.00)
    s.colors[ImGuiCol_PlotLines]             = imgui.Vec4(0.61, 0.61, 0.61, 1.00)
    s.colors[ImGuiCol_PlotLinesHovered]      = imgui.Vec4(1.00, 0.43, 0.35, 1.00)
    s.colors[ImGuiCol_PlotHistogram]         = imgui.Vec4(0.90, 0.70, 0.00, 1.00)
    s.colors[ImGuiCol_PlotHistogramHovered]  = imgui.Vec4(1.00, 0.60, 0.00, 1.00)
    s.colors[ImGuiCol_TextSelectedBg]        = imgui.Vec4(0.26, 0.59, 0.98, 0.35)
    s.colors[ImGuiCol_DragDropTarget]        = imgui.Vec4(0.11, 0.64, 0.92, 1.00)
    s.colors[ImGuiCol_NavHighlight]          = imgui.Vec4(0.26, 0.59, 0.98, 1.00)
    s.colors[ImGuiCol_NavWindowingHighlight] = imgui.Vec4(1.00, 1.00, 1.00, 0.70)
    s.colors[ImGuiCol_NavWindowingDimBg]     = imgui.Vec4(0.80, 0.80, 0.80, 0.20)
    s.colors[ImGuiCol_ModalWindowDimBg]      = imgui.Vec4(0.80, 0.80, 0.80, 0.35)

# Based on style_colors_dark()
def theme_custom():
    # Values from: https://github.com/ocornut/imgui/blob/v1.65/imgui_draw.cpp#L170

    s = imgui.get_style()
    s.frame_padding         = [4, 3]
    s.window_border_size    = 1
    s.child_border_size     = 0
    s.popup_border_size     = 0
    s.frame_border_size     = 0
    s.window_rounding       = 0
    s.child_rounding        = 0
    s.popup_rounding        = 0
    s.frame_rounding        = 0
    s.scrollbar_rounding    = 0
    s.grab_rounding         = 0

    # Color preview: install vscode extension 'json-color-token', set value "jsonColorToken.languages": ["json", "jsonc", "python"]
    # Converted with matplotlib.colors.to_hex(color, keep_alpha=True)
    s.colors[ImGuiCol_Text]                  = color('#ffffffff')
    s.colors[ImGuiCol_TextDisabled]          = color('#808080ff')
    s.colors[ImGuiCol_WindowBg]              = color('#0f0f0ff0')
    s.colors[ImGuiCol_ChildBg]               = color('#ffffff00')
    s.colors[ImGuiCol_PopupBg]               = color('#141414f0')
    s.colors[ImGuiCol_Border]                = color('#6e6e8080')
    s.colors[ImGuiCol_BorderShadow]          = color('#00000000')
    s.colors[ImGuiCol_FrameBg]               = color('#294a7a8a')
    s.colors[ImGuiCol_FrameBgHovered]        = color('#4296fa66')
    s.colors[ImGuiCol_FrameBgActive]         = color('#4296faab')
    s.colors[ImGuiCol_TitleBg]               = color('#0a0a0aff')
    s.colors[ImGuiCol_TitleBgActive]         = color('#294a7aff')
    s.colors[ImGuiCol_TitleBgCollapsed]      = color('#00000082')
    s.colors[ImGuiCol_MenuBarBg]             = color('#242424ff')
    s.colors[ImGuiCol_ScrollbarBg]           = color('#05050587')
    s.colors[ImGuiCol_ScrollbarGrab]         = color('#4f4f4fff')
    s.colors[ImGuiCol_ScrollbarGrabHovered]  = color('#696969ff')
    s.colors[ImGuiCol_ScrollbarGrabActive]   = color('#828282ff')
    s.colors[ImGuiCol_CheckMark]             = color('#4296faff')
    s.colors[ImGuiCol_SliderGrab]            = color('#3d85e0ff')
    s.colors[ImGuiCol_SliderGrabActive]      = color('#4296faff')
    s.colors[ImGuiCol_Button]                = color('#4296fa66')
    s.colors[ImGuiCol_ButtonHovered]         = color('#4296faff')
    s.colors[ImGuiCol_ButtonActive]          = color('#0f87faff')
    s.colors[ImGuiCol_Header]                = color('#4296fa4f')
    s.colors[ImGuiCol_HeaderHovered]         = color('#4296facc')
    s.colors[ImGuiCol_HeaderActive]          = color('#4296faff')
    s.colors[ImGuiCol_Separator]             = s.colors[imgui.COLOR_BORDER]
    s.colors[ImGuiCol_SeparatorHovered]      = color('#1a66bfc7')
    s.colors[ImGuiCol_SeparatorActive]       = color('#1a66bfff')
    s.colors[ImGuiCol_ResizeGrip]            = color('#4296fa40')
    s.colors[ImGuiCol_ResizeGripHovered]     = color('#4296faab')
    s.colors[ImGuiCol_ResizeGripActive]      = color('#4296faf2')
    s.colors[ImGuiCol_PlotLines]             = color('#9c9c9cff')
    s.colors[ImGuiCol_PlotLinesHovered]      = color('#ff6e59ff')
    s.colors[ImGuiCol_PlotHistogram]         = color('#e6b200ff')
    s.colors[ImGuiCol_PlotHistogramHovered]  = color('#ff9900ff')
    s.colors[ImGuiCol_TextSelectedBg]        = color('#4296fa59')
    s.colors[ImGuiCol_DragDropTarget]        = color('#BABA267D')
    s.colors[ImGuiCol_NavHighlight]          = color('#4296faff')
    s.colors[ImGuiCol_NavWindowingHighlight] = color('#ffffffb2')
    s.colors[ImGuiCol_NavWindowingDimBg]     = color('#cccccc33')
    s.colors[ImGuiCol_ModalWindowDimBg]      = color('#cccccc59')