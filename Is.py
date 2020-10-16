"Find if an item 'is' a particular thing"
import bpy
from zpy import Get

# -------------------------------------------------------------------------
# region: Python types

def bool(src):
    """item is True or False"""
    bool = __builtins__['bool']
    return isinstance(src, bool)

def digit(src):
    """item is a number"""
    return (Is.float(src) or Is.int(src))

def float(src):
    """src is a decimal number"""
    float = __builtins__['float']
    if Is.string(src):
        try:
            src = Get.as_float(src)
        except:
            return False
    return isinstance(src, float)

def int(src):
    """src is a whole number"""
    int = __builtins__['int']
    if Is.string(src):
        try:
            src = Get.as_int(src)
        except:
            return False
    return isinstance(src, int) and not Is.bool(src)

def iterable(src):
    """item can be search like a list"""
    import mathutils

    return any((
        hasattr(src, '__iter__'),
        isinstance(src, dict),
        isinstance(src, list),
        isinstance(src, set),
        isinstance(src, tuple),
        isinstance(src, mathutils.Euler),
        isinstance(src, mathutils.Matrix),
        isinstance(src, mathutils.Quaternion),
        isinstance(src, mathutils.Vector),
    ))

def matrix(src):
    """item is a matrix"""
    import mathutils

    return isinstance(src, mathutils.Matrix)

def string(src):
    """item is a text string"""
    return isinstance(src, str)

# endregion
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# region: Blender type

def armature(src):
    """
    src is an armature object \\
    (TODO: check for armature data type, separately from rig object)
    """
    bool = __builtins__['bool']
    return bool(Is.object(src) and src.type in {'ARMATURE'})

def bone(src):
    """
    src is a generic bone\\
    The sub.bone of a posebone or a bone from the armature(data).bones
    """
    return isinstance(src, bpy.types.Bone)

def camera(src):
    """src is a camera object"""
    bool = __builtins__['bool']
    return bool(Is.object(src) and src.type in {'CAMERA'})

def collection(src):
    return isinstance(src, bpy.types.Collection)

def curve(src):
    """src is a curve object"""
    bool = __builtins__['bool']
    return bool(Is.object(src) and src.type in {'CURVE'})

def editbone(src):
    """src is a bone from edit mode"""
    return isinstance(src, bpy.types.EditBone)

def empty(src):
    """src is an empty object"""
    bool = __builtins__['bool']
    return bool(Is.object(src) and not src.data)

def gpencil(src):
    """src is a grease pencil object"""
    bool = __builtins__['bool']
    return bool(Is.object(src) and src.type == 'GPENCIL')

def light(src):
    """src is a light object"""
    bool = __builtins__['bool']
    return bool(Is.object(src) and src.type in {'LIGHT', 'LAMP'})

def mesh(src):
    """Item is a mesh object"""
    bool = __builtins__['bool']
    return bool(Is.object(src) and src.type in {'MESH'})

def nla_strip(src):
    """src is a nla strip"""
    return isinstance(src, bpy.types.NlaStrip)

def object(src):
    """src is an object"""
    return isinstance(src, bpy.types.Object)

def posebone(src):
    """src is a bone from pose mode"""
    return isinstance(src, bpy.types.PoseBone)

# endregion
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# region: Status

def animation_playing(context):
    """Return if the active screen is running the timeline playback"""
    scr = context.screen

    if scr:
        return (scr.is_animation_playing and not scr.is_scrubbing)

def connected(src, pseudo=True):
    """
    Returns if a bone is connected to it's parent\\
    or pseudo connected (the head is at the parent's tail)
    """

    # if not src.bone.use_connect and (not src.parent or src.bone.head_local != src.bone.parent.tail_local):
    if Is.posebone(src):
        pbone = src.bone
    elif Is.bone(src) or Is.editbone(src):
        pbone = src
        src = Get.rig(bpy.context, src).pose.bones[src.name]
    else:
        if not Is.object(src):
            assert None, ("Is.connected() checked on invalid item",
                src, type(src)
            )
        return

    if pbone.parent:
        if (src.bone.use_connect):
            return True
        elif pseudo and (src.bone.head_local == src.bone.parent.tail_local):
            if not sum(src.lock_location) and sum(src.location) >= 0.0001:
                # bone was moved and wasn't locked
                return False
            else:
                return True

    return False

def in_dopesheet(context, src):
    """
    src is visible in the animation editor (provided it has animation data)
    """

    src = getattr(src, 'id_data', None)
    space = Get.space(context, 'dopesheet')

    if not Is.object(src):
        # there's a filter for FCurves, and datablocks can be used too,
        # but just ignore them for now
        return

    poll = not any((
        # Isolation
        not (Is.visible(context, src) or space.show_hidden),
        # 'show_missing_nla': True,
        space.show_only_selected and not Is.selected(src),

        # Main Filter
        not (
            space.filter_collection and
            src.name in space.filter_collection.objects,
        ),
        # 'filter_fcurve_name': '',
        # 'filter_text': '',

        # Filters
        # 'show_armatures': True,
        # 'show_cache_files': True,
        # 'show_cameras': True,
        # 'show_curves': True,
        # 'show_datablock_filters': False,
        # 'show_expanded_summary': True,
        # 'show_gpencil': True,
        # 'show_gpencil_3d_only': False,
        # 'show_lattices': True,
        # 'show_lights': True,
        # 'show_lamps': True,  # 27
        # 'show_linestyles': True,
        # 'show_materials': True,
        # 'show_meshes': True,
        # 'show_metaballs': True,
        # 'show_modifiers': True,
        # 'show_nodes': True,
        # 'show_only_errors': False,
        # 'show_only_matching_fcurves': False,  # 27
        # 'show_particles': True,
        # 'show_scenes': True,
        # 'show_shapekeys': True,
        # 'show_speakers': True,
        # 'show_summary': False,
        # 'show_textures': True,
        # 'show_transforms': True,
        # 'show_worlds': True,
    ))

    return poll

def in_scene(context, src):
    """src is in the active scene"""
    return (src in Get.in_scene(context))

def in_view(context, src, *views):
    """src is in the active view layer"""
    filters = dict(object=Is.object(src), collection=Is.collection(src))
    return (src in Get.in_view(context, *views, **filters))

def in_visible_armature_layers(bone, arm):
    """Bone is on a visibile layer of the armature (data)"""
    if Is.posebone(bone):
        bone = bone.bone
    if Is.armature(arm):
        arm = arm.data
    return [
        i for i, j in zip(bone.layers, arm.layers)
        if i and j
    ]

def linked(src):
    """Find is something is from an external blend file"""
    if Is.posebone(src):
        obj = src.id_data
    elif Is.object(src):
        obj = src
    elif Is.bone(src):
        return bool(src.id_data.library)
    elif Is.editbone(src):
        # You can't access edit bones without edit mode
        return False
    else:
        assert None, ("Have not calculated for this data type " + repr(src))

    if obj.proxy or obj.library:
        return True
    else:
        return False

def panel_expanded(context):
    """
    Try to find if the sidebar is stretched enough to see button text\\
    Returns bool and whether 1 or 2 buttons are visible
    """
    from zpy import utils

    # Currently this just checks the width,
    # we could have different layouts as preferences too.
    region = context.region
    system = utils.prefs().system
    view2d = region.view2d
    view2d_scale = (
        view2d.region_to_view(1.0, 0.0)[0] -
        view2d.region_to_view(0.0, 0.0)[0]
    )
    width_scale = region.width * view2d_scale / \
        getattr(system, 'ui_scale', 1.0)

    show_text = (width_scale > 120.0)
    columns = (80.0 < width_scale < 120.0)

    # print(show_text, columns, width_scale)

    return (show_text, columns, width_scale)

def selected(src):
    """Item is selected"""

    if Is.object(src):
        if hasattr(src, 'select'):  # 2.7
            return src.select
        elif hasattr(src, 'select_get'):  # 2.8
            return src.select_get()
        else:
            assert None, ('Object missing attribute to find selection')
    elif Is.posebone(src):
        return getattr(src.bone, 'select', None)
    elif Is.bone(src) or Is.editbone(src):
        return src.select
    elif src is None:
        return
    else:
        assert None, ('Is.selected() has not accounted for this type', src)

def visible(context, src, viewport=False):
    """
    Object not hidden and in a visible layer/collection\\
    Bone not hidden and in a visible layer of it's armature
    """
    bool = __builtins__['bool']

    if Is.object(src):
        if viewport:
            if viewport is True:
                space = context.space_data
            else:
                space = in_viewport

            if space.type == 'VIEW_3D':
                return src.visible_in_viewport_get(space)

        return src.visible_get()

        if not (Is.in_scene(context, src) and Is.in_view(context, src)):
            # Object not in active scene and/or deleted
            return

        vl = context.view_layer
        if src in vl.objects.values():
            res = src.visible_get(view_layer=vl)
            return res
    elif Is.posebone(src):
        if not src.bone.hide:
            return bool(Get.visible_armature_layers(src, src.id_data.data))
        else:
            return False
    elif Is.bone(src) or Is.editbone(src):
        if not src.hide:
            return bool(Get.visible_armature_layers(src, src.id_data))
        else:
            return False
    elif Is.collection(src):
        if not src.hide_viewport:
            return Is.in_view(context, src)
        else:
            return False
    else:
        assert None, ("Is.visible(), has not accounted for this type",
            src, type(src)
        )

# endregion
# -------------------------------------------------------------------------


Is = type('', (), globals())
