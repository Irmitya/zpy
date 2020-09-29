"""shortcuts for invoking popup windows"""
import bpy


def _draw(draw=None):
    if draw is None:
        # def draw(self, ui, context):
        def draw(self, context):
            pass
    return draw

# -------------------------------------------------------------------------
# region: Returns None

def popover(context, draw_menu=None, *args, **kargs):
    """
    Small narrow popup window, similar to the redo panel\\
    kargs:
        ui_units_x=0, keymap=None
    """

    wm = context.window_manager
    d = popup._draw(draw_menu)

    is27 = bpy.app.version < (2, 80, 0)
    is28 = not is27
    if is28: wm.popover(d, *args, **kargs)
    if is27: wm.popup_menu(d, title="popover not in 2.7!", icon='ERROR')

    return {'CANCELLED'}

def menu(context, draw_menu=None, **kargs):
    """
    Basic menu with no items (designed for single-column)\\
    kargs:
        title="", icon='NONE'
    """

    d = popup._draw(draw_menu)
    context.window_manager.popup_menu(d, **kargs)

    return {'CANCELLED'}

def pie(context, event, draw_menu=None, **kargs):
    """
    kargs:
        title="", icon='NONE'
    """
    d = popup._draw(draw_menu)
    context.window_manager.popup_menu_pie(event, d, **kargs)

    return {'CANCELLED'}

# endregion
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# region: Returns exit set

def invoke_confirm(context, operator, event):
    """Confirm Only (gives a single menu button)"""

    return context.window_manager.invoke_confirm(operator, event)

def invoke_confirm_ok(context, self, **kargs):
    """
    Does not run until clicking Ok\\
    kargs:
        width=554, height=247):
    """

    return context.window_manager.invoke_props_dialog(self, **kargs)

def invoke_popup(context, self, **kargs):
    """
    Display Only\\
    kargs:
        width=554
        # height=247  deprecated in 2.83

    Menu default width = 174
    """

    return context.window_manager.invoke_popup(self, **kargs)

def invoke_props(context, self, event):
    """No customizable scale"""

    return context.window_manager.invoke_props_popup(self, event)

def invoke_search(context, self):
    """little menu to search properties"""

    return context.window_manager.invoke_search_popup(self)

# endregion
# -------------------------------------------------------------------------


popup = type('', (), globals())
