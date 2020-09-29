import bpy

# utils
from . import cpp, Get, Is, utils

# Version check
is27 = bpy.app.version < (2, 80, 0)
is28 = not is27

# functions
from . import keyframe, New, popup, Set

# functions that are initialized()
from .register_keymaps import register_keymaps
from .load_modules import load_modules

# Version check
import bpy
is27 = bpy.app.version < (2, 80, 0)
is28 = not is27
