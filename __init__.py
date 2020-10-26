import bpy

# utils
from .cpp import cpp
from .Get import Get
from .Is import Is
from .utils import utils

# functions
from .keyframe import keyframe
from .New import New
from .popup import popup
from .Set import Set

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

__addon_enabled__ = False
register, unregister = bpy.utils.register_classes_factory(())
