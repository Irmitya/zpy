"Functions to apply properties or status to data"
import bpy
from zpy import Get, Is


# -------------------------------------------------------------------------
# region: Utils

def armature_display_type(armature, display_type='RANDOM', random_src=None):
    """
    Set the default bone display for rig objects\\
    display_type is enum in with the following odds:
        'OCTAHEDRAL'    * 6,
        'STICK'         * 6,
        'WIRE'          * 2,
        'BBONE'         * 4,
        'ENVELOPE'      * 1,
    random_src is the source armature to avoid when randomly selecting type
    """
    import random

    display_types = [
        *['OCTAHEDRAL'] * 6,
        *['STICK'] * 6,
        *['WIRE'] * 2,
        *['BBONE'] * 4,
        *['ENVELOPE'] * 1,
    ]
    # armature.show_in_front = True
    if Is.armature(armature):
        armature = armature.data
    if Is.armature(random_src):
        random_src = random_src.data

    def set_type(arm, type):
        if hasattr(arm, 'draw_type'):  # 2.7
            arm.draw_type = type
        elif hasattr(arm, 'display_type'):  # 2.8
            arm.display_type = type

    def get_type(arm):
        if hasattr(arm, 'draw_type'):  # 2.7
            return arm.draw_type
        elif hasattr(arm, 'display_type'):  # 2.8
            return arm.display_type

    if display_type == 'RANDOM':
        display_type = display_types.pop(
            random.randrange(len(display_types)))

        while random_src and display_types and \
        display_type == get_type(random_src):
            # Remove items from the display types until it's different
            display_type = display_types.pop(
                random.randrange(len(display_types)))

    set_type(armature, display_type)

    return display_type

def object_display_type(src, type):
    """
    type enum in ['BOUNDS', 'WIRE', 'SOLID', 'TEXTURED']
    """

    if hasattr(src, 'display_type'):  # 2.8
        src.display_type = type
    elif hasattr(src, 'draw_type'):  # 2.7
        src.draw_type = type

def bone_group(bone, group, color=None):
    """
    Assign bone to group\\
    if group is text, find the group or create it first
    """

    if Is.string(group):
        bgs = bone.id_data.pose.bone_groups
        if group in bgs:
            group = bgs[group]
        else:
            from zpy import New
            group = New.bone_group(bone, name=group)

    if Is.string(color):
        group.color_set = color
        # DEFAULT = regular bone color (not unique)
        # THEME01 - THEME15 = Builtin color palettes
        # THEME16 - THEME20 = Black for user assigned templates
        # CUSTOM = manually set
    elif color is False:
        group.color_set = 'DEFAULT'
    elif color is True:
        from random import randrange as random
        rand = f"{random(1, 15):02}"
        group.color_set = f'THEME{rand}'
    elif color:
        group.color_set = 'CUSTOM'
        gc = group.colors
        if not Is.iterable(color):
            if Is.digit(color):
                color = [[color] * 3] * 3
            else:
                color = [color] * 3
        (gc.normal, gc.select, gc.active) = color

    bone.bone_group = group

    return group

def empty_type(empty, type='PLAIN_AXES'):
    """
    Set the visual shape of an empty object
    type = enum in
        ['PLAIN_AXES', 'ARROWS', 'SINGLE_ARROW',
        'CIRCLE', 'CUBE', 'SPHERE', 'CONE', 'IMAGE']
    """

    if not Is.empty(empty):
        return

    if hasattr(empty, 'empty_draw_type'):  # 2.7
        empty.empty_draw_type = type.upper()
    if hasattr(empty, 'empty_display_type'):  # 2.8
        empty.empty_display_type = type.upper()

def empty_size(empty, size=1.0):
    """Set the scale of an empty object"""

    if not Is.empty(empty):
        return

    if hasattr(empty, 'empty_draw_size'):  # 2.7
        empty.empty_draw_size = size
    if hasattr(empty, 'empty_display_size'):  # 2.8
        empty.empty_display_size = size

def matrix(src, matrix, local=False, basis=False):
    """
    Set the visual transforms for a bone or object
    The parameters vary, so the input matrix should too:
        editbone.     matrix
        bone.         matrix, matrix_local
        posebone.     matrix, matrix_basis, matrix_channel
        object.       matrix_world, matrix_basis, matrix_local, matrix_parent_inverse
    """

    if Is.object(src):
        if basis:
            src.matrix_basis = matrix
        elif local:
            src.matrix_local = matrix
        else:
            src.matrix_world = matrix
    else:
        if basis or local:
            if Is.posebone(src): src.matrix_basis = matrix
            elif Is.bone(src): src.matrix_local = matrix
            else: src.matrix = matrix
        else:
            src.matrix = matrix

def select(target, value=True):
    """Select or Deselect an item"""

    if Is.object(target):
        target.select_set(value)
    elif Is.posebone(target):
        target.bone.select = value
    elif Is.bone(target) or Is.editbone(target):
        target.select = value
    elif target is None:
        pass
    else:
        # Give error
        assert None, (
            "Error: zpy\\Set.select() can't use the provided target \n",
            target,
        )

def xray(src, value=True):
    """Set the xray display toggle for an object"""

    if hasattr(src, 'show_in_front'):  # 2.8
        src.show_in_front = value
        # 2.8: use bone xray, to show bones
            # try:
            #     context.space_data.overlay.show_xray_bone = value
            # except:
            #     src.show_in_front = value
    elif hasattr(src, 'show_x_ray'):  # 2.7
        src.show_x_ray = value

# endregion
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# region: Contextual

def active(context, target):
    """Set target as active scene object or bone"""

    objects = Get.objects(context)

    # Remember previous active
    previous = Get.active(context)
    selected = Is.selected(target)

    # Set the active
    if Is.object(target):
        obj = target
    elif Is.posebone(target):
        obj = target.id_data
        obj.data.bones.active = obj.data.bones.get(target.name)
    elif isinstance(target, bpy.types.Armature):
        obj = Get.rig(context, target)
    elif Is.bone(target):
        obj = Get.rig(context, target)
        bones = target.id_data.bones
        bones.active = bones.get(target.name)
    elif Is.editbone(target):
        obj = Get.rig(context, target)

        if obj: in_edit = (obj.mode == 'EDIT')
        else: in_edit = (context.mode == 'EDIT_ARMATURE')

        if in_edit:
            bones = target.id_data.edit_bones
            bones.active = bones.get(target.name)
    elif target is None:
        obj = None
        # debug("Set.active() has None as the target")
    else:
        assert None, ("Set.active() can't use the provided target", target)

    if (target and Is.selected(target) != selected):
        # When setting a bone as active in a rig, it gets selected as well.
        Set.select(target, selected)
    objects.active = obj

    return previous

def active_select(context, target, isolate=True):  # , isolate_mode=False):
    """
    Set target as active scene object or bone, and select it\\
    # isolate_mode would set all other selected objects to Object mode
    """

    # def deselect(src):
        # if isolate_mode and src.id_data.mode != 'OBJECT':
        #     Set.mode(context, 'OBJECT', src)
        # Set.select(src, False)

    if isolate:
        objects = Get.selected_objects(context)
        bones = Get.selected_pose_bones(context)

        for src in [*objects, *bones]:
            # deselect(src)
            Set.select(src, False)

    Set.visible(context, target, True)
    Set.active(context, target)
    Set.select(target, True)
    if Is.posebone(target):
        Set.mode(context, 'POSE', target)

def constraint_toggle(context, srcs, constraints, influence=None, insert_key=None):
    """Disable constraint while maintaining the visual transform."""
    from zpy import keyframe

    if insert_key is None:
        insert_key = keyframe.poll_insert(context, insert_key)

    def store_matrix(mats=dict()):
        # Get the matrix in world space.
        # Save the matrix
        for src in srcs:
            mats[src] = Get.matrix(src)

        return mats

    def set_influence():
        def get_src(constraint):
            for src in srcs:
                for con in src.constraints:
                    if con == constraint:
                        return src
        for constraint in constraints:
            driver = Get.driver(constraint, 'influence')
            if driver:
                (constraint, prop) = Get.controls_from_driver(driver)[0]
                src = constraint
            else:
                prop = 'influence'
                src = get_src(constraint)

                if src is None:
                    continue

            if influence is None:
                # mode = 'apply'
                pass
            elif influence is False:
                # mode = 'invert'
                setattr(constraint, prop, not getattr(constraint, prop))
            else:
                # mode = 'set'
                setattr(constraint, prop, influence)

            # key_con = prefs.get("Keyframe Constraints Inf")
            # if key_con:
            if insert_key:
                keyframe.prop(context, constraint, prop,
                    group=keyframe.group_name(src),
                    options={'INSERTKEY_NEEDED'},
                )

    def update_constraint_matrix(mats):
        from zpy import utils

        # Set the matrix
        for con in constraints:
            if hasattr(con, 'use_offset') and con.use_offset:
                utils.update(context)
        # utils.update(context)
        for src in srcs:
            new_mat = Get.matrix_constraints(context, src, mats[src], constraints)
            mats[src] = new_mat

    def apply_constraint_matrix(mats):
        # Set the matrix
        for src in srcs:
            set_constraint_matrix(src, mats[src])

    def set_constraint_matrix(src, mat):
        # if Is.posebone(src):
        #     src.matrix = multiply_matrix(src.id_data.matrix_world.inverted(), mat)
        # else:
        #     src.matrix_world = mat
        Set.matrix(src, mat)

        keyframe.keyingset(context, insert_key=insert_key, selected=[src])
        # if insert_key:
            # keyframe.location(context, src)
            # keyframe.rotation(context, src)
            # keyframe.scale(context, src)

    matrices = store_matrix()
    set_influence()
    update_constraint_matrix(matrices)
    apply_constraint_matrix(matrices)

class cursor:
    """Set a transform value for the 3D cursor"""

    def location(context, x):
        Get.cursor(context).location = x

    def rotation_axis_angle(context, x):
        Get.cursor(context).rotation_axis_angle = x

    def rotation_euler(context, x):
        Get.cursor(context).rotation_euler = x

    def rotation_mode(context, x):
        Get.cursor(context).rotation_mode = x

    def rotation_quaternion(context, x):
        Get.cursor(context).rotation_quaternion = x

def in_scene(context, object, value=True):
    # TODO: set this function to be able to remove an object from scene
    #       currently it ONLY uses the value for set at True

    if value:
        if Is.collection(object):
            scn = context.scene
            if not Get.collection_from_scene(object, scn.collection):
                scn.collection.children.link(object)
        elif not (Is.in_scene(context, object) and Is.in_view(context, object)):
            Get.objects(context, link=True).link(object)

def mode(context, mode, target=None, keep_visiblity=True):
    """
    Set the context.mode for an object (or bone's rig)
    """
    if not target:
        bpy.ops.object.mode_set(mode=mode)
        return context.mode == mode

    target = target.id_data
        # I can't think of a situation where I care to use
        # a bone/etc instead of the object

    # objects = context.selected_objects
    # for obj in objects:
        # select(obj, False)

    if Is.linked(target) and mode not in ('OBJECT', 'POSE'):
        return False

    class active_item:
        mode = target.mode
        is_visible = Set.visible(context, target, True)

    if mode != target.mode:
        modes = dict()
        selected = list()
        objects = list()

        # Find the visible objects of the same type as the target object
        # Remember their modes and selection, then deselect them
        for obj in Get.in_view(context):
            if obj == target or obj.type != target.type or not Is.visible(context, obj):
                continue
            if obj.mode not in modes:
                modes[obj.mode] = list()
            modes[obj.mode].append(obj)

            if Is.selected(obj):
                selected.append(obj)
            Set.select(obj, False)
            objects.append(obj)

        # Remember the target's selected status
        pselect = Is.selected(target)

        # Set the mode for the target object
        previous = Set.active(context, target)
        Set.select(target, False)
        bpy.ops.object.mode_set(mode=mode)

        # Since the operator switches all objects to the specified mode
        # Go through the objects and manually set them back their previous mode
        for pmode in modes:
            for obj in modes[pmode]:
                Set.select(obj, True)
            Set.active(context, obj)
            bpy.ops.object.mode_set(mode=pmode)
            Set.active(context, None)
            for obj in modes[pmode]:
                Set.select(obj, False)

        # Re-select previously selected objects
        for obj in selected:
            Set.select(obj, True)

        # reselect target if it was selected
        Set.select(target, pselect)

        # Set the active object back to the original
        if previous is not None:
            Set.active(context, previous)
        else:
            Set.active(context, target)

    if (keep_visiblity):
        Set.visible(context, target, active_item.is_visible)

    # for obj in objects:
        # select(obj, True)

    return (target.mode == mode)

def visible(context, object, value=True, **kargs):
    """
    Set an object's (or bone's object's) visibility to the specified value
    """

    scn = context.scene

    if not Is.object(object):
        if isinstance(object, bpy.types.Collection):
            found = False

            def loop(root, tree=list()):
                nonlocal found

                if root.collection == object:
                    return True

                for child in root.children:
                    if loop(child, tree):
                        found = True
                        tree.append(child)
                        break

                if found:
                    return tree

            view_layer = kargs.get('view_layer', False)

            if not view_layer:
                object.hide_viewport = not value
            if value or view_layer:
                # Only enables the collection for the view layer once
                tree = loop(context.view_layer.layer_collection)
                for col in tree:
                    if (col.exclude == value) and (col.name == object.name):
                        # When a collection is enabled in the view layer,
                        # all of its child collections are as well.
                        col.exclude = not value
                    if value and col.collection.hide_viewport:
                        col.collection.hide_viewport = False
        elif Is.posebone(object):
            return Set.visible(context, object.id_data, value)
        elif Is.bone(object) or Is.editbone(object):
            return Set.visible(context, Get.rig(context, object), value)
        else:
            assert None, (
                "Set.visible() does not work with the specified item",
                object,
            )
        return

    Set.in_scene(context, object)

    is_visible = Is.visible(context, object)
    object_visible = not object.hide_viewport

    # if Is.visible(context, object) is value:
        # return visible

    while (Is.visible(context, object) is not value):
        "If object isn't in the desired visiblity, loop until it is"

        if (object.hide_viewport is value):
            object.hide_viewport = not value
            continue

        is_visible = object_visible
        view = None

        for collection in object.users_collection:
            view = context.view_layer.layer_collection.children.get(collection.name)
            if not view:
                # collection isn't in scene or whatever
                continue
            if view.hide_viewport is value:
                view.hide_viewport = not value
            break

        if view is None:
            assert None, (
                "Set.visible(): Object[", object,
                "] \nis hidden from viewport and I don't know how to change it"
            )
            # collection.hide_viewport = value

        break

    return is_visible

# endregion
# -------------------------------------------------------------------------


Set = type('', (), globals())
