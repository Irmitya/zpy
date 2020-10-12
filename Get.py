"Get various variables"
import bpy
from zpy import Is, utils


# -------------------------------------------------------------------------
# region: Python Types
# -------------------------------------------------------------------------

def as_digit(text):
    try:
        return int(text)
    except:
        return float(text)

def as_float(src):
    return float(src)

def as_int(src):
    return int(Get.as_float(src))

def as_list(context, attr):
    "Returns the bpy.context.%attr% item or an empty list if it's blank"

    items = getattr(context, attr, None)
    if items is None:
        items = []

    return items

# endregion
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# region: Debug
# -------------------------------------------------------------------------

def file(levels=0):
    import os
    from inspect import (currentframe, getframeinfo)
    cf = currentframe().f_back.f_back
    while levels:
        cff = cf.f_back
        if cff is None:
            break
        cf = cff
        levels -= 1

    return os.path.basename(getframeinfo(cf).filename)

def line(levels=0):
    from inspect import (currentframe, getframeinfo)

    cf = currentframe().f_back.f_back
    while levels:
        cff = cf.f_back
        if cff is None:
            break
        cf = cff
        levels -= 1

    return cf.f_lineno

def stack(level=0):
    from inspect import stack
    # 0 = current def
    # 2 is used to call from stack << debug << original file call

    return stack()[level][3]

# endregion
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# region: Utils
# -------------------------------------------------------------------------

def action_fcurves(src, actions=None, raw_action=True):
    "Find actions used by a bone/object and return it's action + fcurves"

    fcurves = list()

    if actions is None:
        rig = src.id_data
        strips = Get.strips(rig, selected=False)
        anim = rig.animation_data
        an_act = getattr(anim, 'action', None) if raw_action else None
        actions = {an_act, *(s.action for s in strips)}

    for action in actions:
        if not action:
            continue
        for fc in action.fcurves:
            if any((
                    Is.object(src),
                    Is.posebone(src) and fc.data_path.find(
                        f'bones[\"{src.name}\"]') != -1,
            )):
                fcurves.append((action, fc))

    return fcurves

def active_strip(anim):
    active_strip = None

    for track in anim.nla_tracks:
        for strip in track.strips:
            if strip.active:
                return strip
            elif anim.use_tweak_mode and (strip.action == anim.action):
                active_strip = strip
                # strip isn't marked active but it likely is the correct one

    return active_strip

def animation_datas(src):
    anims = list()
    data = getattr(src, 'data', None)
    shapes = getattr(data, 'shape_keys', None)

    for x in (src, data, shapes):
        anim = getattr(x, 'animation_data', None)
        if anim:
            anims.append(anim)

    return anims

def collection_from_scene(collection, scene):
    """Find a collection from the scene or sub collections"""

    if isinstance(scene, bpy.types.Scene):
        scene = scene.collection

    if collection in scene.children.values():
        return scene
    else:
        for sub in scene.children:
            search = Get.collection_from_scene(collection, sub)
            if search:
                return search

def controls_from_driver(driver, report=None):
    bone_props = set()

    if hasattr(driver, 'driver'):
        driver = driver.driver

    for var in driver.variables:
        for tar in var.targets:
            # Should be exactly one target (no more, no less)

            control = tar.id
            prop = tar.data_path

            if not prop:
                continue

            is_bone = prop.startswith(('pose.bones', 'bones'))
            redo_obj = False

            if not hasattr(control, prop):
                redo_obj = True

            if is_bone or redo_obj:
                if tar.data_path.endswith('"]'):
                    control, prop = tar.data_path.rsplit('["', 1)
                    prop = '["' + prop
                else:
                    control, prop = tar.data_path.rsplit('.', 1)

            try:
                if is_bone or redo_obj:
                    control = tar.id.path_resolve(control)
                bone_props.add((control, prop))
            except:
                # invalid driver?
                if report:
                    report({'ERROR'}, ("Invalid driver:" + repr(driver)))

    return list(bone_props)

def constraint_target(con):
    if hasattr(con, 'target'):
        target = con.target
        subtarget = getattr(con, 'subtarget', None)
    elif hasattr(con, 'targets'):
        targets = list()
        for subcon in con.targets:
            target = subcon.target
            subtarget = subcon.subtarget

            if target and subtarget and hasattr(target, 'pose'):
                target = target.pose.bones.get(subtarget)
            targets.append(target)

        return targets
    else:
        target = subtarget = None

    if target and subtarget and hasattr(target, 'pose'):
        target = target.pose.bones.get(subtarget)

    return target

def bone_group(src, name=""):
    """
    Returns a named group inside rig.
    if src is a posebone, will return the bone's group instead.
        If the bone doesn't have a group, will try to return the named group instead.
    """

    group = None

    if Is.posebone(src):
        group = src.bone_group

    if not group:
        group = src.id_data.pose.bone_groups.get(name)

    return group

def copy(src):
    ""

    return type("Duplicate", (), {p: getattr(src, p) for p in dir(src)})

def distance(point1, point2) -> float:
    """Calculate distance between two points in 3D."""
    from math import sqrt
    return sqrt(sum(
        (x[1] - x[0]) ** 2
        for x in zip(point1, point2)
    ))

def driver(src, prop, index=-1):
    ""

    drivers = getattr(src.id_data.animation_data, 'drivers', None)
    if drivers:
        if hasattr(src, prop):
            prop = src.path_from_id(prop)
        driver = drivers.find(prop, index=index)
    else:
        driver = None

    return driver

def event(event=None):
    "returns a fake event if not supplied with a real one"

    if isinstance(event, bpy.types.Event):
        return event

    class event:
        __module__ = 'bpy.types'
        __slots__ = ()
        # bl_rna =  <bpy_struct, Struct("Event")>
        # rna_type =  <bpy_struct, Struct("Event")>
        alt = False
        ctrl = False
        oskey = False
        shift = False
        type = 'NONE'
        value = 'NOTHING'
        ascii = ''
        unicode = ''
        is_mouse_absolute = False
        is_tablet = False
        mouse_prev_x = 0
        mouse_prev_y = 0
        mouse_region_x = -1
        mouse_region_y = -1
        mouse_x = 0
        mouse_y = 0
        pressure = 1.0
        tilt = Vector((0.0, 0.0))

    return event

def fcurve(action, path, index=-1):
    """Find if a specified path exists in an action"""
    fcs = action.fcurves
    if index == -1:
        fc = fcs.find(path)
    else:
        fc = fcs.find(path, index=index)

    # if fc and index != -1 and fc.array_index != index:
        # for fc in fcs:
            # if fc.data_path == path and fc.array_index == index:
                # break
        # else:
            # fc = None

    return fc

def icon(*args):
    """
    args:
        name_to_search, bool
        Off_Icon, On_Icon, bool(s)
    """
    if len(args) > 2:
        off, on, *var = args
        value = all(var)
        return (off, on)[value]
    else:
        name, value = args
        name = name.lower()
        value = bool(value)

    if name in {'disclosure', 'disclosure_tri_right'}:
        icon = 'DISCLOSURE_TRI_' + ('RIGHT', 'DOWN')[value]
    else:
        icon = 'BLANK1'

    return icon

def icon_from_type(src):
    "return an Icon id for the specified item's type"

    if Is.object(src):
        if src.type == 'LIGHT_PROBE':
            icon = 'LIGHTPROBE_' + src.data.type
        else:
            icon = src.type + '_DATA'
    elif Is.bone(src) or Is.editbone(src) or Is.posebone(src):
        icon = 'BONE_DATA'
    else:
        icon = 'ERROR'
        utils.debug("Can't find icon type for ", src, type(src))

    return icon

def ik_chain(bone, constraint=None):
    "return list of parent bones connected to the bone through an IK chain"

    parents = list()

    if not (Is.posebone(bone) and bone.parent):
        return parents

    maxlen = len(bone.parent_recursive)

    def scan(constraint):
        if (constraint.type in ('IK', 'SPLINE_IK')):
            count = constraint.chain_count
            if (count == 0 or count > maxlen):
                parents.extend(bone.parent_recursive)
            else:
                if getattr(constraint, 'use_tail', True):
                    # Spline IK starts count from the bone with the constraint
                    chain_range = count - 1
                else:
                    chain_range = count
                for chain in range(chain_range):
                    parents.append(bone.parent_recursive[chain])

    if constraint is None:
        for constraint in bone.constraints:
            scan(constraint)
    else:
        scan(constraint)

    parents_sorted = list()
    if parents:
        for parent in bone.parent_recursive:
            if parent in parents:
                parents_sorted.append(parent)

    return parents_sorted

def macro(*ops, poll=None, **kargs):
    """Get an operator to run a sequence of operators in succession
        if they aren't cancelled"""

    idname = kargs.get('idname', '_macro_._macro_start_')
    # For defaults, insert ops as operator or bl_idname
    # For operators with props:
    # ops = (bl_idname, dict(prop=var, ))

    class MACRO(bpy.types.Macro):
        bl_idname = idname
        bl_label = "Start Macro"
        bl_options = {'MACRO'}

        @classmethod
        def poll(self, context):
            if poll:
                return poll(self, context)
            return True
    bpy.utils.register_class(MACRO)
    # bpy.macro = MACRO

    for op in ops:
        if Is.iterable(op) and not Is.string(op):
            (op, props) = (op[0], dict(*op[1:]))
        else:
            props = dict()

        if hasattr(op, 'bl_idname'):
            op = op.bl_idname

        if Is.string(op):
            if op.startswith('bpy.ops.'):
                op = eval(op)
            else:
                op = eval('bpy.ops.' + op)
        operator = MACRO.define(op.idname())
        for prop in props:
            setattr(operator.properties, prop, props[prop])

    return eval('bpy.ops.' + idname)

def matrix(src, local=False, basis=False, tail=False, copy=True):
    # editbone.     matrix
    # bone.         matrix, matrix_local
    # posebone.     matrix, matrix_basis, matrix_channel
    # object.       matrix_world, matrix_basis, matrix_local, matrix_parent_inverse

    if Is.posebone(src):
        if copy:
            matrix = utils.multiply_matrix(
                src.id_data.matrix_world, src.matrix)
        else:
            matrix = src.matrix

        if basis:
            matrix_local = src.matrix_basis
        else:
            matrix_local = src.bone.matrix_local
    elif Is.bone(src):
        matrix = src.matrix
        matrix_local = src.matrix_basis
    elif Is.editbone(src):
        matrix = matrix_local = src.matrix
    else:
        matrix = src.matrix_world
        if basis:
            matrix_local = src.matrix_basis
        else:
            matrix_local = src.matrix_local

    if copy:
        matrix = matrix.copy()
        matrix_local = matrix_local.copy()

    # scaled = False
    # if len(set(matrix.to_scale())) == 1:
    #     matrix *= matrix.to_scale()[0]
    #     scaled = True

    if (tail and hasattr(src, 'tail')):
        # matrix.translation += (src.tail - src.head)
        matrix.translation = matrix.Translation(src.tail).translation
        matrix_local.translation = \
            matrix_local.Translation(src.tail).translation
        # for (i, t) in enumerate(matrix.translation):
        #     v = (src.tail[i] - src.head[i]) * (
        #         1 if scaled else matrix.to_scale()[i])
        #     matrix.translation[i] += v

    if (basis or local):
        return matrix_local
    else:
        return matrix

def matrix_local(src, mat=None):
    """Get and convert a matrix to local space for later re-use"""
    if mat is None:
        mat = Get.matrix(src)

    if Is.posebone(src):
        # from zpy import Set
        # Set.matrix(src, Get.matrix(src))
        # mat = Get.matrix(src, basis=True)

        # Setting the matrix updates it to the visual pose
        # Basis gets the applied pose

        # This would do that as well, without needing "apply":
        mat = src.id_data.convert_space(
            pose_bone=src, matrix=mat,
            from_space='POSE', to_space='LOCAL')
        "or so my notes said"
        # From Baker:

        # if visual keying:
        #     # Get the final transform of the bone in its own local space...
        #     matrix[name] = obj.convert_space(
        #         pose_bone=bone, matrix=bone.matrix,
        #         from_space=self.bone_matrix, to_space='LOCAL')
        # else:
        #     matrix = bone.matrix_basis.copy()
    else:
        mat = src.id_data.convert_space(
            pose_bone=None, matrix=mat,
            from_space='WORLD', to_space='LOCAL')

        # From Baker:
        # If deleting object parents, do matrix world

        # if (Global.do_parents_clear):
        #     trans['matrix'] = obj.matrix_world.copy()
        # else:
        #     parent = obj.parent
        #     matrix = obj.matrix_world
        #     trans['matrix'] = multiply_matrix(
        #         parent.matrix_world.inverted_safe(), matrix
        #         ) if parent else matrix.copy()

    return mat

def mode(object):

    if hasattr(object, 'mode'):
        return object.mode

def name(src):

    # orig = Constraint.get.original(src)
    # if orig:
    #     src = orig

    if Is.posebone(src):
        name = f"{src.id_data.name}.{src.name}"
    else:
        name = f"{src.name}"

    # if src.constraints_relative.is_new:
    #     name = name.rsplit('-', 1)[0]
    #     if name.endswith('-'):
    #         name = name.rsplit('-', 1)[0]

    return name

def nla_blend_name(anim_or_strip):

    if Is.nla_strip(anim_or_strip):
        blend_type = anim_or_strip.blend_type
    else:
        blend_type = anim_or_strip.action_blend_type

    if blend_type == 'REPLACE':
        return 'Base'
    else:
        return blend_type.replace('COMBINE', 'Layer').title()

def object_bone_from_string(target: "String or Object", subtarget=''):
    """Try to find an object an bone from text"""

    if Is.string(target):
        obj = bpy.data.objects.get(target)
    else:
        obj = target

    bone = None

    if Is.armature(obj) and subtarget:
        if (obj.mode == 'EDIT'):
            bones = obj.data.edit_bones
        else:
            bones = obj.pose.bones
        bone = bones.get(subtarget)

    return (obj, bone)

def reverse(src):
    "return a reversed list/string or opposite number"

    if Is.iterable(src):
        if type(src) is str:
            inverse = ""
        else:
            inverse = []
        for item in reversed(src):
            inverse += item
    elif type(src) is bool:
        inverse = not src
    elif type(src) in (int, float):
        inverse = src * -1
    else:
        inverse = src

    return inverse

def sorted_chains(selected):
    "Scan through selected bones to find order of bones for spline_ik chains"

    chains = []
    for bone in selected:
        if not Is.posebone(bone):
            continue
        chain = []
        end = True
        for child in bone.children_recursive:
            if child in selected:
                end = False
                continue
        if end:
            for par in reversed(bone.parent_recursive):
                if par in selected:
                    chain.append(par)
            chain.append(bone)
            chains.append(chain)

    return chains

def strip_co_frame(strip, co=None, frame=None):
    "Try to convert the frame number between scene and action keyframes"

    action_start = strip.action_frame_start
    action_end = strip.action_frame_end
    strip_start = strip.frame_start
    strip_end = strip.frame_end
    # Repeat: store repeat then reset it to get the actual strip range
    # repeat = strip.repeat; scale = strip.scale; strip.repeat = 1; strip.scale = 1; strip_end = strip.frame_end

    end = utils.scale_range(
        strip_end, strip_start, strip_end,
        0, abs(strip_start - strip_end)
    )
    end /= strip.repeat
    end /= strip.scale
    end *= strip.scale
    end += strip_start

    # Restore repeat value
    # strip.repeat = repeat; strip.scale = scale

    if (co is not None):
        # target = relative strip
        value = utils.scale_range(co, action_start, action_end, strip_start, end)
    elif (frame is not None):
        # target = absolute frame
        value = utils.scale_range(frame, strip_start, end, action_start, action_end)

    # # for looping curve keyframes
    # while (value > end):
        # value -= abs(strip_start - end)

    return value

def strip_tracks(src, items=[], selected=True, top_down=True, active_first=True):
    """Return a class pack of strips and track + owner for object"""
    anim = getattr(src, 'animation_data', None)
    if not anim:
        return []

    has_strips = False
    active = False
    strips = []
    if top_down:
        tracks = reversed(anim.nla_tracks)
    else:
        tracks = anim.nla_tracks

    for track in tracks:
        for strip in track.strips:
            if any((
                    not (strip.select or not selected),
                    # strip.type == 'TRANSITION',
            )):
                continue
            has_strips = True
            active_strip = type('strip', (), dict(
                data=src,
                track=track,
                strip=strip,
            ))

            if strip.active and active_first:
                active = True
                strips.insert(0, active_strip)
                # strips.insert(0, strip)
            else:
                strips.append(active_strip)
                # strips.append(strip)
    if not has_strips:
        return []

    if active:
        # items = [*strips, *items]
        items.insert(0, [src, strips])
    else:
        # items = [*items, *strips]
        items.append([src, strips])

    return strips

def strips(src, items=[], selected=False, top_down=True, active_first=True):
    """Return all strips available from the specified object"""
    strips = []
    for item in Get.strip_tracks(
            src, items=items, selected=selected,
            top_down=top_down, active_first=active_first):
        strips.append(item.strip)

    return strips

def valid_op(*ops):
    """Validate whether or not an operator is in bpy.ops;
        if True, return operator"""

    valid = list()

    for op in ops:
        if not Is.string(op):
            if hasattr(op, 'bl_idname'):
                op = op.bl_idname
            else:
                continue

        try:
            exec(f'bpy.ops.{op}.get_rna_type()')
            valid.append(op)
        except:
            continue

    if len(ops) > 1:
        return valid
    elif valid:
        return valid[0]
    else:
        return None

def visible_armature_layers(bone, arm):
    """List of layer indexes that is visible and the Bone is on"""

    if Is.posebone(bone):
        bone = bone.bone
    if Is.armature(arm):
        arm = arm.data

    return [i for i, j in zip(bone.layers, arm.layers) if i and j]

# endregion
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# region: Contextual:
# -------------------------------------------------------------------------

# def context(C=None):
    # import bpy

    # if C is None:
    #     context = bpy.context
    # elif isinstance(C, dict):
    #     context = type('', (Get.copy(bpy.context), bpy.types.Context), C)
    # else:
    #     context = type('', (Get.copy(C), bpy.types.Context,), {})

    # return context

def active(context, mode=None):
    "Return active object or bone, depending on the active mode"

    if mode is None:
        mode = context.mode

    # context_modes = (
        # 'OBJECT', 'SCULPT', 'POSE',
        # 'EDIT_ARMATURE', 'EDIT_MESH',
        # 'PAINT_WEIGHT', 'PAINT_TEXTURE', 'PAINT_VERTEX',
    # )
    # object_modes = (
        # 'OBJECT', 'SCULPT', 'POSE',
        # 'EDIT',
        # 'WEIGHT_PAINT', 'TEXTURE_PAINT', 'VERTEX_PAINT',
    # )

    if mode in ('OBJECT', 'EDIT_MESH', 'SCULPT', 'PAINT_WEIGHT', 'PAINT_TEXTURE', 'PAINT_VERTEX'):
        active = context.object
        if active is None:
            active = (Get.selected_objects(context) + [None])[0]
    elif mode in {'EDIT_ARMATURE', 'edit_bones'}:
        active = context.active_bone
        if active is None:
            active = (Get.selected_edit_bones(context) + [None])[0]
    elif mode in {'bones'}:
        active = context.active_bone
        if active is None:
            active = (Get.selected_bones(context) + [None])[0]
    elif mode == 'POSE':
        active = context.active_pose_bone
        if active is None:
            active = (Get.selected_pose_bones(context) + [None])[0]
    else:
        utils.debug(
            f"Error: zpy.Get.py > Get.active() > "
            f"does not account for this mode ({mode})"
        )
        active = None

    return active

def cursor(context):
    return context.scene.cursor

def frame_mapped(context, frame=None):
    """Get the time remapped original of the frame # or frame current"""
    scn = context.scene

    if frame is None:
        frame = scn.frame_current_final
    fmn = scn.render.frame_map_new
    fmo = scn.render.frame_map_old

    frame *= (fmn / fmo)
    # if fmo < fmn:
    # elif fmn < fmo:
        # frame *= (fmo / fmn)

    return frame

def frame_from_strip(context, anim_or_strip, frame=None, absolute=False):
    "Convert the keyframe.co inside scaled nla strip to timeline frame"

    if frame is None:
        frame = context.scene.frame_current_final

    anim = anim_or_strip.id_data.animation_data
    strip = anim_or_strip
    tweak = anim.use_tweak_mode
    is_strip = Is.nla_strip(strip)

    if (tweak or is_strip):
        if (tweak and is_strip and strip.active) and hasattr(anim, 'nla_tweak_strip_time_to_scene'):
            value = anim.nla_tweak_strip_time_to_scene(frame, invert=False)
        else:
            if not is_strip:
                strip = Get.active_strip(anim)
            value = Get.strip_co_frame(strip, co=frame)
    else:
        value = frame

    if absolute:
        return value
    else:
        return (round(value, 6))

def frame_to_strip(context, anim_or_strip, frame=None, absolute=False):
    "Convert the keyframe.co/frame to correct frame inside nla strip"

    if frame is None:
        frame = context.scene.frame_current_final

    anim = anim_or_strip.id_data.animation_data
    strip = anim_or_strip
    tweak = anim.use_tweak_mode
    is_strip = Is.nla_strip(strip)

    if (tweak or is_strip):
        if is_strip and strip.active and hasattr(anim, 'nla_tweak_strip_time_to_scene'):
            value = anim.nla_tweak_strip_time_to_scene(frame, invert=True)
        else:
            if not is_strip:
                strip = Get.active_strip(anim)
            value = Get.strip_co_frame(strip, frame=frame)
    else:
        # Null call, so just return scene frame
        value = frame

    if absolute:
        return value
    else:
        return (round(value, 6))

def frame_range(context):
    "return current playback range"

    scn = context.scene
    preview = ("_", "_preview_")[scn.use_preview_range]
    frame_start = eval(f'scn.frame{preview}start')
    frame_end = eval(f'scn.frame{preview}end')

    return (frame_start, frame_end)

def frame_range_from_nla(context, **kargs):
    "return current playback range of selected nla strips"
    # kargs:
    # objects=[]

    objects = kargs.get('objects', Get.objects(context=context))

    def get_sub_strips(strip, sub):
        """Loop through Meta Strips to find Action Strips"""
        nonlocal start, end
        if (getattr(sub, 'strips', None)):
            for sub_sub in sub.strips:
                get_sub_strips(sub, sub_sub)
                # "loop until no more sub-strips"
        elif (getattr(sub, 'action', None)):
            start = (start, strip.frame_start)[start is None]
            end = (end, strip.frame_end)[end is None]

            if (strip.frame_start < start):
                start = strip.frame_start
            if (strip.frame_end > end):
                end = strip.frame_end

    start = end = None
    for obj in objects:
        anim = obj.animation_data
        if anim is None:
            continue

        for track in anim.nla_tracks:
            for strip in track.strips:
                if (not strip.select):
                    continue
                get_sub_strips(strip, strip)

    frame_start = (start, frame_start)[start is None]
    frame_end = (end, frame_end)[end is None]

    return (frame_start, frame_end)

def in_scene(context, *scenes):
    "return list of objects in the scene(s)"

    objects = []
    if not scenes:
        scenes = [context.scene]

    for scene in scenes:
        objects.extend(scene.objects)

    return objects

def in_view(context, *views, **filters):
    """return list of objects/collections in the current view layer"""

    items = []

    if not views:
        views = [context.view_layer]

    if not filters:
        filters = dict(
            object=True,
            collection=False,
        )

    for view_layer in views:
        if filters.get('object'):
            items.extend(view_layer.objects)

        if filters.get('collection'):
            def colls(layer):
                for view in layer.children:
                    if view.is_visible:
                        items.append(view.collection)
                    colls(view)
            colls(view_layer.layer_collection)

    return items

def matrix_constraints(context, src, mat=None, force_constraints=[]):
    """Convert matrix so that it doesn't offset with the item's constraints"""

    from mathutils import Matrix

    mats_cons = list()

    mat_new = Get.matrix(src)
    if mat is None:
        mat = mat_new

    if Is.posebone(src):
        inv = src.id_data.matrix_world.inverted()
        mat = utils.multiply_matrix(inv, mat)
        mat_new = utils.multiply_matrix(inv, mat_new)

    for con in src.constraints:
        if not getattr(con, 'target', None) or \
                con.mute or \
                (con.influence == 0.0 and con not in force_constraints):
            continue
        else:
            pose = con.target.pose
            bone = None
            if pose and getattr(con, 'subtarget', None):
                bone = pose.bones.get(con.subtarget)

        if hasattr(con, 'inverse_matrix'):
            im = con.inverse_matrix
            imi = im.inverted()
            mat0 = utils.multiply_matrix(imi, mat)
            # The constraint with cleared inverse
            mat1 = mat.copy()
            # The goal pose without constraint
            con.inverse_matrix == utils.multiply_matrix(mat1, mat0.inverted())
            # The value set initially for the constraint

            mat_tar = con.target.matrix_world.inverted()
            if bone:
                mat_tar = utils.multiply_matrix(mat_tar, bone.matrix.inverted())

            mat_value = utils.multiply_matrix(imi, mat_tar)

            # # TODO: This get's the standard Child of, but it doesn't get
            # # a constraint without all 3 Rotation enabled

            # This is meant to give a list of matrices to cycle through
            # to try to find the correct mixture. What's wrong appears to
            # be that the target matrix from above needs to be altered,
            # and the current math is already valid
            # mat0 and mat
            # mats = (
            #     Matrix(),               # 0

            #     mat0,                   # 1
            #     mat1,                   # 2
            #     mat_new,                # 3
            #     mat_tar,                # 4
            #     im,                     # 5

            #     mat0.inverted(),        # 6
            #     mat1.inverted(),        # 7
            #     mat_new.inverted(),     # 8
            #     mat_tar.inverted(),     # 9
            #     imi,                    # 10
            # )
            # ac = [' ' + a for a in ['a', 'b', 'c', 'd', 'e', 'f', 'g']]
            # for a in ac:
            #     if src.get(a) is None:
            #         src[a] = 0

            # mat_value = multiply_matrix(
            #     *(
            #         mats[src[a]] for a in ac
            #     ),
            # )

            # Set influence of the result from constraint
            mat_influce = Matrix().lerp(mat_value, con.influence)
            mats_cons.append(mat_influce)
        elif getattr(con, 'use_offset', False):
            mats_cons.append(Matrix().lerp(
                utils.multiply_matrix(mat, mat_new.inverted()),
                con.influence))

    return utils.multiply_matrix(*mats_cons, mat)

def objects(context, collection=None, link=False):
    """
    Return the active scene objects pointer list\\
    (for example, to set the active object)
    """

    objects = context.view_layer.objects  # active scene's objects (all)
    if link:
        # 2.8 can only link to collections
        objects = context.scene.collection.objects
        # active scene's objects without a collection
    if collection:
        objects = collection.objects

    return objects

def objects_nla(context):
    """Find available objects in the NLA / Dopesheet"""

    objects = []

    for obj in Get.objects(context):
        if Is.in_dopesheet(context, obj):
            objects.append(obj)

    return objects

def rig(context, src):
    """
    bone/editbone don't directly link to objects, just the armature.\\
    So, try to find an object by using their armature instead
    """

    is_armature = isinstance(src, bpy.types.Armature)

    if Is.posebone(src) or Is.armature(src.id_data):
        return src.id_data
    elif (is_armature, Is.bone(src) or Is.editbone(src)):
        for object in Get.objects(context):
            if (object.data == src.id_data):
                break
        else:
            object = None
        return object
    else:
        assert None, ("Could not find rig for", src)

def selected(context, src=None):
    """
    return selected objects or selected bones.
    If src is specified, return selected bones from it
    """

    mode = context.mode

    if mode in ('POSE', 'PAINT_WEIGHT'):
        selected = Get.selected_pose_bones(context, src)
    elif mode == 'EDIT_ARMATURE':
        selected = Get.selected_edit_bones(context, src)
    else:
        selected = Get.selected_objects(context, True)

    return selected

def selected_bones(context, src=None):
    "Always return data bones as list, never as None"

    if src:
        bones = src.data.bones
        selected = [b for b in bones if Is.selected(b) and
        Is.visible(context, b)]
    else:
        selected = Get.as_list(context, 'selected_bones')

    return selected

def selected_edit_bones(context, src=None):
    "Always return edit bones as list, never as None"

    if src:
        bones = src.data.edit_bones
        selected = [b for b in bones if Is.selected(b) and
        Is.visible(context, b)]
    else:
        selected = Get.as_list(context, 'selected_editable_bones')

    return selected

def selected_pose_bones(context, src=None, force: "not needed, todelete" = False):
    "Always return pose bones as list, never as None"

    if src:
        selected = [b for b in src.pose.bones if Is.selected(b) and
        Is.visible(context, b)]
    # elif not force and context.mode not in ('POSE', 'PAINT_WEIGHT'):
        # selected = []
    else:
        selected = Get.as_list(context, 'selected_pose_bones')

    return selected

# Disabled because this is supposed to find the "Active object" only
# def selected_pose_bones_from_object(src=None, context=None):
    # if src:
    # selected = [b for b in Get.selected_pose_bones() if b.id_data == src]
    # else:
    # return Get.as_list('selected_pose_bones_from_active_object', context=context)

def selected_objects(context, all_objects=False):
    "Return selected objects if either they're in object mode or object is True"

    selected = [o for o in Get.as_list(context, 'selected_objects')
                if (all_objects) or (o.mode not in ('POSE', 'EDIT'))]

    return selected

def space(context, area):
    "return the active spacedata or a defaults if not found"

    space = getattr(context.space_data, area, None)

    if space is not None:
        pass
    elif area == 'dopesheet':
        space = type('defaults', (), dict(
            # Isolation
            show_hidden=False,
            show_missing_nla=True,
            show_only_selected=False,

            # Main Filter
            # None or a collection
            filter_collection=type('None', (), {'objects': []}),
            show_only_group_objects=False,  # 27
            # None or a group
            filter_group=type('None', (), {'objects': []}),
            filter_fcurve_name='',
            filter_text='',

            # Filters
            show_armatures=True,
            show_cache_files=True,
            show_cameras=True,
            show_curves=True,
            show_datablock_filters=False,
            show_expanded_summary=True,
            show_gpencil=True,
            show_gpencil_3d_only=False,
            show_lattices=True,
            show_lights=True,
            show_lamps=True,  # 27
            show_linestyles=True,
            show_materials=True,
            show_meshes=True,
            show_metaballs=True,
            show_modifiers=True,
            show_nodes=True,
            show_only_errors=False,
            show_only_matching_fcurves=False,  # 27
            show_particles=True,
            show_scenes=True,
            show_shapekeys=True,
            show_speakers=True,
            show_summary=False,
            show_textures=True,
            show_transforms=True,
            show_worlds=True,

            __doc__=None,
            __module__='bpy.types',
            __slots__=(),
            # bl_rna=< bpy_struct, Struct("DopeSheet") >,
            # rna_type=bpy.data.screens['Layout']...Struct,
            source=context.scene,  # 'source': bpy.data.scenes['Scene'],
            use_datablock_sort=True,
            use_filter_text=False,  # 27
            use_multi_word_filter=False
        ))

    return space

def strip_influence(context, strip, frame=None):
    "Return influence of NLA Strip at a given frame (or current frame)"
    if frame is None:
        frame = context.scene.frame_current_final

    (start, end) = (strip.frame_start, strip.frame_end)
    start_in = (start + strip.blend_in)
    end_out = (end - strip.blend_out)

    if (strip.use_animated_influence):
        # Animated Influence
        influence = strip.fcurves.find('influence').evaluate(frame)
    elif (start_in < frame < end_out):
        # Regular Influence
        influence = 1.0
    elif (frame < start):
        # Before strip
        if (strip.extrapolation == 'HOLD') and (start == start_in):
            influence = 1.0
        else:
            influence = 0.0
    elif (end < frame):
        # After strip
        if (strip.extrapolation.startswith('HOLD')) and (end_out == end):
            influence = 1.0
        else:
            influence = 0.0
    elif (start <= frame <= start_in):
        # At Start or Blend In
        if start != start_in:
            influence = utils.scale_range(frame, start, start_in, 0, 1)
        else:
            influence = 1.0
    elif (end_out <= frame <= end):
        # At End or Blend Out
        if end_out != end:
            influence = utils.scale_range(frame, end_out, end, 1, 0)
        else:
            influence = 1.0
    else:
        # Unknown
        influence = None

    return influence

def strip_in_track(context, track, frame=None):
    "Find the currently evaluated strip in the track"

    if frame is None:
        frame = context.scene.frame_current_final
    right_of_strips = None
    active_strip = None
    for strip in track.strips:
        if (frame < strip.frame_start):
            if right_of_strips in {'HOLD', 'HOLD_FORWARD'}:
                # Previous strip extends forward, so use that
                strip = active_strip
            elif right_of_strips in {'NOTHING', None}:
                if (strip.extrapolation == 'HOLD'):
                    # This strip extends backwards
                    pass
                elif (strip.extrapolation in {'NOTHING', 'HOLD_FORWARD'}):
                    # Outside and between two strips. No effect.
                    strip = None
            break
        elif (strip.frame_start <= frame <= strip.frame_end):
            # Inside strip's range
            break
        elif (strip.frame_end < frame):
            # Set strip as active to compare to next strip
            right_of_strips = strip.extrapolation
            active_strip = strip
    else:
        if not active_strip:
            # Reached end of track, and no strip can be even considered
            return

    return strip

def strips_nla(context, selected=True, top_down=True, active_first=True):
    "Return all strips (class pack) available from the NLA editor"

    items = []
    args = dict(items=items, selected=selected,
                top_down=top_down, active_first=active_first)

    for _object_ in Get.objects_nla(context):
        obj = getattr(_object_, 'id_data', _object_)
        data = getattr(_object_, 'data', None)
        shapes = getattr(data, 'shape_keys', None)

        Get.strip_tracks(obj, **args)
        Get.strip_tracks(data, **args)
        Get.strip_tracks(shapes, **args)

    return items

def tail(context, bone):
    # ------ Bones
    if any((Is.posebone(bone), Is.bone(bone), Is.editbone(bone))):
        if (Is.posebone(bone)):
            obj = bone.id_data
        else:
            obj = Get.rig(context, bone)

        if obj is None:
            return
        else:
            mat = Get.matrix(obj)
            return utils.multiply_matrix(mat, bone.tail)

    # # ------ Mesh
    # vp = getattr(obj.data, 'polygons', None)
    # ve = getattr(obj.data, 'edges', None)
    # vt = getattr(obj.data, 'vertices', None)
    # if any((vp, ve, vt)):
    #     if vp and len(vp) > 1:
    #         co = vp[0].center
    #         for poly in vp:
    #             if poly.center != Vector():
    #                 co = poly.center
    #                 break
    #     elif ve:
    #         for edge in ve:
    #             co = Vector()
    #             for i in range(3):
    #                 for vert_i in edge.vertices:
    #                     co[i] += obj.data.vertices[vert_i].co[i]
    #                 co[i] /= len(edge.vertices)
    #             if co != Vector():
    #                 break
    #     elif vt:
    #         co = vt[-1].co
    #         for vert in [vt[-1], vt[0], *reversed(vt)]:
    #             if vert.co != Vector():
    #                 co = vert.co
    #                 break
    #     return multiply_matrix(obj.matrix_world, co)

    # # -------- All other
    # try:
    #     # Get object's dimensions; Maybe get tail for curves/metaballs/etc later
    #     co = Vector((0, obj.dimensions[1], 0))
    #     if self:
    #         self.report({'INFO'}, "Non-Mesh tail finding doesn't work")
    #     return multiply_matrix(obj.matrix_world, co)
    # except:
    #     return None

def value_from_nla_strips(context, src, path, index=-1, frame=None, selected=False, tweak=False):
    # Tries to actually read the strips
    # but it doesn't work with COMBINE quaternions

    anim = getattr(src.id_data, 'animation_data', None)
    if not anim:
        return

    if frame is None:
        frame = context.scene.frame_current_final

    if Is.posebone(src) and not path.startswith('pose.bones'):
        kpath = (f'pose.bones[\"{src.name}\"].' + path)
    else:
        kpath = path

    # Establish default
    if any((
            'default' is 1,
            path.endswith('scale'),
            path.endswith('rotation_quaternion') and index == 0,
            path.endswith('rotation_axis_angle') and index == 2,
    )):
        default = 1.0
    else:
        default = 0.0
    value = default

    default_quat = Quaternion()
    value_quat = default_quat

    first_strip = True
    for track in anim.nla_tracks:
        if track.mute:
            continue
        if tweak and anim.use_tweak_mode:
            for strip in track.strips:
                if strip.active:
                    break
            else:
                strip = Get.strip_in_track(track, frame=frame)
        else:
            strip = Get.strip_in_track(track, frame=frame)

        action = getattr(strip, 'action', None)
        if not action or strip.mute:
            # Either no strip evaluated or strip is meta strips (TODO)
            continue

        # Try to find property in fcurve
        if index == -1:
            fc = action.fcurves.get(kpath)
        else:
            for fc in action.fcurves:
                if fc.data_path == kpath and fc.array_index == index:
                    break
            else:
                fc = None
        if fc is None:
            if path.endswith('rotation_quaternion') and action.fcurves.get(kpath):
                pass
            else:
                continue

        if frame < strip.frame_start:
            frame_strip = strip.frame_start
        elif (frame > strip.frame_end):
            frame_strip = strip.frame_end
        else:
            frame_strip = frame

        if tweak and strip.active:
            ff = frame
            inf = 1.0
        else:
            ff = frame_strip
            inf = Get.strip_influence(context, strip, frame=frame_strip)

        next_quat = Quaternion()

        if path.endswith('rotation_quaternion'):
            for i in range(4):
                # Don't research curves for the already found curve
                if fc and i == index:
                    next = fc.evaluate(Get.frame_to_strip(strip, ff))
                    next_quat[i] = next
                    continue
                for afc in action.fcurves:
                    if afc.data_path == kpath and afc.array_index == i:
                        next = afc.evaluate(Get.frame_to_strip(strip, ff))
                        next_quat[i] = next
                        break
                else:
                    next_quat[i] = default_quat[i]

        if fc:
            next = fc.evaluate(Get.frame_to_strip(context, strip, ff))
        else:
            next = None

        if (first_strip):
            value = next
            value_quat = next_quat
            first_strip = False
        elif (strip.blend_type == 'COMBINE'):
            if (path.endswith('rotation_axis_angle') and index == 2):
                result = value + (next * inf)
            elif (path.endswith('rotation_quaternion') and index == 0):
                result = value + ((default - next) * inf)
                # result = value * pow(next, inf)
            elif path.endswith('scale'):
                result = value * (next * inf)
            else:
                # result = value - ((default - next) * inf)
                result = value + (next - default) * inf
            # utils.debug(f"{path}[{index}]\t{strip.name}")
            # utils.debug(f"\tValue\t{value}")
            # utils.debug(f"\tNext\t{next}")
            # utils.debug(f"\tResult\t{result}")
            # utils.debug(f'\tGoal\t{getattr(src, path)[index]}')
            value = result

            # "Quaternion blending is deferred until all sub-channel values are known."

            #     # next_val = abs(next - default) * (-1 if next < default else 1)
            #     # value += utils.lerp(0.0, next_val, inf)

            # if path.endswith('rotation_quaternion'):
            #     mix_mode = 'QUATERNION'  # return old value
            #     mix_mode = 'AXIS_ANGLE'  # return old_value + (value - base_value) * inf
            #     # value = nla_combine_value(mix_mode, default, value, next, inf)
            #     # nla_invert_combine_quaternion(value_quat, next_quat, inf, value_quat)
            #     nla_combine_quaternion(value_quat, next_quat, inf, value_quat)
            #     value = value_quat[index]
            # else:
            #     # mix_mode = 'ADD'  # pass
            #     mix_mode = 'AXIS_ANGLE'  # return old_value + (value - base_value) * inf
            #     # mix_mode = 'MULTIPLY'  # return old_value * pow(value / base_value, inf)

            #     value = nla_combine_value(mix_mode, default, value, next, inf)
        else:
            value = nla_blend_value(strip.blend_type, value, next, inf)
            # nla_combine_quaternion(value_quat, next_quat, inf, value_quat)

    return value

# endregion
# -------------------------------------------------------------------------


Get = type('', (), globals())
