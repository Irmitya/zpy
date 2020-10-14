import bpy

# utils
from . import cpp, Get, Is, utils

# functions
from . import keyframe, New, popup, Set

# functions that are initialized()
from .register_keymaps import register_keymaps
from .load_modules import load_modules

bl_info = {
    "name": "Zpy Module (ignore)",
    "author": "COnLOAR",
    "description": "Module for various functions re-used across zpy addons",
    "blender": (2, 90, 0),
    "category": "System",
    'wiki_url': "https://blenderartists.org/t/zpy-various-miscellaneous-addons/1254269",
    "warning": "Does not need to be enabled",
}

register, unregister = bpy.utils.register_classes_factory(())
