import bpy
from math import radians

# Version check
is27 = bpy.app.version < (2, 80, 0)
is28 = not is27

# region: utils
class Get:
    "Get various variables"
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

    def bone_group(rig, name='Group'):
        """Returns a named group inside rig; (probably should find a "bone's" group instead)"""

        if Is.posebone(rig):
            rig = rig.id_data

        return rig.pose.bone_groups.get(name)

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
        ""

        scn = context.scene

        if is27:
            class cursor:
                location = scn.cursor_location
                rotation_axis_angle = Quaternion()
                rotation_euler = Euler()
                rotation_mode = 'EULER'
                rotation_quaternion = Quaternion()
        if is28:
            cursor = scn.cursor

        return cursor

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
            if is27: views = [context.scene]
            if is28: views = [context.view_layer]

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

        if is27:
            objects = context.scene.objects
            if collection:  # group
                objects = collection.objects
        if is28:
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


class cpp:
    """C++ classes from Blender internals, converted (mostly) as-is to python"""

    def add_qt_qtqt(result, quat1, quat2, t):
        result[0] = quat1[0] + t * quat2[0]
        result[1] = quat1[1] + t * quat2[1]
        result[2] = quat1[2] + t * quat2[2]
        result[3] = quat1[3] + t * quat2[3]

    def add_v3_v3(r, a):
        r[0] += a[0]
        r[1] += a[1]
        r[2] += a[2]

    def copy_qt_qt(q1, q2):
        q1[0] = q2[0]
        q1[1] = q2[1]
        q1[2] = q2[2]
        q1[3] = q2[3]

    def copy_v3_v3(r, a):
        r[0] = a[0]
        r[1] = a[1]
        r[2] = a[2]

    def dot_qtqt(q1, q2):
        return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]

    def interp_dot_slerp(t, cosom, r_w):
        """
        * Generic function for implementing slerp
        * (quaternions and spherical vector coords).
        *
        * param t: factor in [0..1]
        * param cosom: dot product from normalized vectors/quats.
        * param r_w: calculated weights.
        """
        from math import sin, acos
        eps = 1e-4

        # BLI_assert(IN_RANGE_INCL(cosom, -1.0001, 1.0001))

        # /* within [-1..1] range, avoid aligned axis */
        if (abs(cosom) < (1.0 - eps)):
            omega = acos(cosom)
            sinom = sin(omega)
            r_w[0] = sin((1.0 - t) * omega) / sinom
            r_w[1] = sin(t * omega) / sinom
        else:
            # /* fallback to lerp */
            r_w[0] = 1.0 - t
            r_w[1] = t

    def interp_qt_qtqt(result, quat1, quat2, t):
        quat = [0, 0, 0, 0]
        w = [0, 0]

        cosom = cpp.dot_qtqt(quat1, quat2)

        # /* rotate around shortest angle */
        if (cosom < 0.0):
            cosom = -cosom
            cpp.negate_v4_v4(quat, quat1)
        else:
            cpp.copy_qt_qt(quat, quat1)

        cpp.interp_dot_slerp(t, cosom, w)

        result[0] = w[0] * quat[0] + w[1] * quat2[0]
        result[1] = w[0] * quat[1] + w[1] * quat2[1]
        result[2] = w[0] * quat[2] + w[1] * quat2[2]
        result[3] = w[0] * quat[3] + w[1] * quat2[3]

    def mid_v3_v3v3(v, v1, v2):
        v[0] = 0.5 * (v1[0] + v2[0])
        v[1] = 0.5 * (v1[1] + v2[1])
        v[2] = 0.5 * (v1[2] + v2[2])

    def minmax_v3v3_v3(min, max, vec):
        if (min[0] > vec[0]):
            min[0] = vec[0]
        if (min[1] > vec[1]):
            min[1] = vec[1]
        if (min[2] > vec[2]):
            min[2] = vec[2]

        if (max[0] < vec[0]):
            max[0] = vec[0]
        if (max[1] < vec[1]):
            max[1] = vec[1]
        if (max[2] < vec[2]):
            max[2] = vec[2]

    def mul_v3_fl(r, f):
        r[0] *= f
        r[1] *= f
        r[2] *= f

    def mul_m4_v3(mat, vec):
        x = vec[0]
        y = vec[1]

        vec[0] = x * mat[0][0] + y * mat[1][0] + mat[2][0] * vec[2] + mat[3][0]
        vec[1] = x * mat[0][1] + y * mat[1][1] + mat[2][1] * vec[2] + mat[3][1]
        vec[2] = x * mat[0][2] + y * mat[1][2] + mat[2][2] * vec[2] + mat[3][2]

    def mul_qt_fl(q, f):
        q[0] *= f
        q[1] *= f
        q[2] *= f
        q[3] *= f

    def mul_qt_qtqt(q, q1, q2):
        t0 = [0, 0, 0, 0]
        t1 = [0, 0, 0, 0]
        t2 = [0, 0, 0, 0]

        t0 = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
        t1 = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
        t2 = q1[0] * q2[2] + q1[2] * q2[0] + q1[3] * q2[1] - q1[1] * q2[3]
        q[3] = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]
        q[0] = t0
        q[1] = t1
        q[2] = t2

    def negate_v4_v4(r, a):
        r[0] = -a[0]
        r[1] = -a[1]
        r[2] = -a[2]
        r[3] = -a[3]

    def normalize_qt(q):
        from math import sqrt

        qlen = sqrt(cpp.dot_qtqt(q, q))

        if (qlen != 0.0):
            cpp.mul_qt_fl(q, 1.0 / qlen)
        else:
            q[1] = 1.0
            q[0] = q[2] = q[3] = 0.0

        return qlen

    def normalize_qt_qt(r, q):
        cpp.copy_qt_qt(r, q)
        return cpp.normalize_qt(r)

    def sub_qt_qtqt(q, q1, q2):
        nq2 = [0, 0, 0, 0]

        nq2[0] = -q2[0]
        nq2[1] = q2[1]
        nq2[2] = q2[2]
        nq2[3] = q2[3]

        cpp.mul_qt_qtqt(q, q1, nq2)


class Is:
    "Find if an item 'is' a particular thing"


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
        bool = __builtins__['bool']
        float = __builtins__['float']
        if Is.string(src):
            try:
                return bool(Get.as_float(src))
            except:
                return False
        else:
            return isinstance(src, float)

    def int(src):
        """src is a whole number"""
        bool = __builtins__['bool']
        int = __builtins__['int']
        if Is.string(src):
            try:
                return bool(Get.as_int(src))
            except:
                return False
        else:
            return isinstance(src, int)

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
        # if is27: return False
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
        else:
            assert None, ("Have not calculated for this data type " + repr(src))
            return

        if obj.proxy or obj.library:
            return True
        else:
            return False

    def panel_expanded(context):
        """
        Try to find if the sidebar is stretched enough to see button text\\
        Returns bool and whether 1 or 2 buttons are visible
        """

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
            if is27:
                return src.is_visible(context.scene)

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


class utils:
    """
    Various shortcut/utility functions\\
    Generally either does one specific thing without any context
    """

    @staticmethod
    def clear_console():
        import os
        os.system("cls")

    @staticmethod
    def line():
        from inspect import currentframe

        cf = currentframe()

        return cf.f_back.f_lineno

    @staticmethod
    def file_line(levels=0):
        line = Get.line(levels)
        file = Get.file(levels)
        # return f"line #{line} in {file}"
        function = Get.stack(levels + 2)

        return f"line #{line} in {function}"

    @staticmethod
    def clean_custom(src):
        """Delete empty custom prop entries"""

        from idprop.types import IDPropertyArray, IDPropertyGroup

        for (name, value) in src.items():
            if isinstance(value, IDPropertyGroup) and not value:
                del (src[name])

    @staticmethod
    def copy_driver(driver, src, attr):
        """Insert a driver onto the src.attr and copy the paramaters of driver"""

        new = src.driver_add(attr)

        new.extrapolation = driver.extrapolation
        # new.group = driver.group  # Drivers don't use groups

        ndriver = new.driver
        odriver = driver.driver
        ndriver.type = odriver.type

        while new.keyframe_points:
            new.keyframe_points.remove(new.keyframe_points[0], fast=True)
        while new.modifiers:
            new.modifiers.remove(new.modifiers[0])
        while ndriver.variables:
            ndriver.variables.remove(ndriver.variables[0])

        for ov in odriver.variables:
            nv = ndriver.variables.new()
            nv.name = ov.name
            nv.type = ov.type

            ot = ov.targets[0]
            nt = nv.targets[0]
            nt.bone_target = ot.bone_target
            nt.data_path = ot.data_path
            if nv.type == 'SINGLE_PROP':
                nt.id_type = ot.id_type
            # nt.id_data = ot.id_data  # Not writable
            nt.id = ot.id
            nt.rotation_mode = ot.rotation_mode
            nt.transform_space = ot.transform_space
            nt.transform_type = ot.transform_type

        attribs = (
            'co', 'easing', 'interpolation', 'type',
            'handle_left', 'handle_left_type', 'handle_right', 'handle_right_type',
            'select_control_point', 'select_left_handle', 'select_right_handle',
        )
        new.keyframe_points.add(len(driver.keyframe_points))
        for (newk, k) in zip(new.keyframe_points, driver.keyframe_points):
            # newk = new.keyframe_points.insert(k.co[0] + 1, k.co[1], keyframe_type=k.type)
            # newk.co[0] = k.co[0]
            # Now copy attributes from k, to newk
            for var in attribs:
                setattr(newk, var, getattr(k, var))

        ndriver.use_self = odriver.use_self
        ndriver.expression = odriver.expression

        return new

    @staticmethod
    def debug(*value, sep=' ', end='\n', file=None, flush=False, line=True, **args):
        """
        Method to leave print functions without DEFINITELY leaving them
            As in using an arg labeled "debug" will allow using
            a global variable to determine whether or not to print
        """

        if file is None:
            from sys import stdout
            file = stdout

        if 'debug' in args:
            skip = not args['debug']
        elif 'class' in args and 'level' in args:
            skip = not getattr(args['class'], args['level'], None)
        else:
            skip = False
        if skip:
            return

        def pop(arg, val):
            if arg in args:
                return args.pop(arg)
            else:
                return val

        try:
            # if not value:
            #     print(f"(Line #{Get.line(1)} in {Get.stack(2)}", sep=sep, end=end, flush=flush)
            # elif 'line' in args:
            #     print(f"({file_line(1)})\t", *value, sep=sep, end=end, flush=flush)
            if not value:
                value = [utils.file_line(1)]
            elif line:
                value = [f"({utils.file_line(1)})\t", *value]
            # else:
            print(*value, sep=sep, end=end, flush=flush)
        except:
            error("Bad properties in debugger")

    class debug_timer():
        """create a callable timer, to keep track of time pasted since last call"""

        def __init__(self):
            from datetime import datetime
            self.time = datetime.now()

        def __call__(self):
            from datetime import datetime
            now = datetime.now()
            elapsed = (now - self.time)
            self.time = now

            hour = minute = 0
            second = elapsed.seconds
            micro = elapsed.microseconds
            # seconds and microseconds are the variables of datetime after math
            # otherwise has day/hour/minute/second/microsecond

            while 3600 <= second:  # Hours
                hour += 1
                second -= 3600
            while 60 <= micro:  # Minutes
                minute += 1
                second -= 60

            log = str()
            if hour:
                log += f'{hour}:'
            if minute:
                log += f'{minute}:'
            if second:
                log += f"{second}."

            if log:
                log += f"{micro}"
            elif ms:
                log += f"{micro / 1000} ms"  # micro to milli
            else:
                log = 'None'

            return "Time lapsed: " + log

    class draw_status():
        def __init__(self, draw_function):
            def draw(self, context):
                draw_function(self, context)

            self.draw = draw

        def start(self, context, wipe_text=True):
            if wipe_text:
                context.window.workspace.status_text_set("")
            bpy.types.STATUSBAR_HT_header.prepend(self.draw)
            return self

        def stop(self, context, restore_text=True):
            if restore_text:
                context.window.workspace.status_text_set(None)
            bpy.types.STATUSBAR_HT_header.remove(self.draw)
            return self

    def draw_keymaps(context, layout, km_kmi: "dict / load_modules.keymaps"):
        """"""
        from rna_keymap_ui import draw_kmi

        wm = context.window_manager
        kc = wm.keyconfigs.addon
        # kc = wm.keyconfigs.user

        for keymap, kmis in km_kmi.items():
            layout.context_pointer_set('keymap', keymap)

            row = layout.row()
            row.alignment = 'LEFT'
            row.emboss = 'PULLDOWN_MENU'
            row.prop(keymap, 'show_expanded_items', text=keymap.name)
            # row.prop(keymap, "show_expanded_items", text="", emboss=False)
            # row.label(text=keymap.name)

            if keymap.show_expanded_items:
                col = layout.column()

                # for kmi in keymap.keymap_items:
                    # if kmi in kmis:
                        # draw_kmi(["ADDON", "USER", "DEFAULT"], kc, keymap, kmi, col, 0)
                for kmi in kmis:
                    draw_kmi(['ADDON'], kc, keymap, kmi, col, 0)

    @staticmethod
    def error(*logs):
        from sys import stderr, exc_info, exc_info
        from traceback import TracebackException

        def print_exception(etype, value, tb, limit=None, file=None, chain=True):
            if file is None:
                file = stderr
            on_line = False
            for line in TracebackException(
                    type(value), value, tb, limit=limit).format(chain=chain):
                end = ""
                tb = exc_info()[2]
                if line.startswith('Traceback'):
                    line = f"Error in "
                elif line.startswith('  File "') and __package__ in line:
                    filename = __package__ + line.split('  File "')[1].split('"')[0].split(__package__)[1]
                    (line_no, function) = line.split(f' line ')[1], ''
                    if ', ' in line_no:
                        (line_no, function) = line_no.split(', ', 1)
                    line = f"{filename}\nline #{line_no} {function}"

                print(line, file=file, end="")
            print(*logs)

        def print_exc(limit=None, file=None, chain=True):
            print_exception(*exc_info(), limit=limit, file=file, chain=chain)

        print_exc()

    @staticmethod
    def find_op(idname):
        """Try to find if an operator is valid"""

        import bpy

        try:
            op = eval(f'bpy.ops.{idname}')
            if hasattr(op, 'get_rna_type'):  # 2.8 and 2.7 daily
                op.get_rna_type()
            elif hasattr(op, 'get_rna'):  # 2.7
                op.get_rna()
            else:
                return
                # debug(f"ERROR WARNING: Can't register keymap! "
                #         f"No valid confirmation check for operator [{idname!r}]")
            return op
        except:
            # debug(f"ERROR: Can't register keymap, operator not found [{idname!r}]")
            return

    @staticmethod
    def layer(*ins, max=32):
        """Get a layer array with only the specified layers enabled"""

        layers = [False] * max
        for i in ins:
            layers[i] = True

        return tuple(layers)

    @staticmethod
    def lerp(current, target, factor=1.0, falloff=False):
        """
        Blend between two values\\
        current <to> target
        """

        if falloff:
            factor = utils.proportional(factor, mode=falloff)

        def blend(current, target):
            if (Is.digit(current) and Is.digit(target)) and not \
                    (Is.string(current) or Is.string(target)):
                return (current * (1.0 - factor) + target * factor)
            elif (factor):
                return target
            else:
                return current

        if (Is.matrix(current) and Is.matrix(target)):
            return current.lerp(target, factor)
        elif (Is.iterable(current) and Is.iterable(target)) and not \
                (Is.string(current) or Is.string(target)):
            # Assume the items are tuple/list/set. Not dict (but dicts can merge)
            merge = list()
            for (s, o) in zip(current, target):
                merge.append(blend(s, o))
            return merge
        else:
            return blend(current, target)

    @staticmethod
    def matrix_from_tuple(tuple):
        from mathutils import Matrix
        if len(tuple) == 16:
            return Matrix((tuple[0:4], tuple[4:8], tuple[8:12], tuple[12:16]))

    @staticmethod
    def matrix_to_tuple(matrix):
        return tuple(y for x in matrix.col for y in x)
        # return tuple(y for x in matrix for y in x)

    @staticmethod
    def matrix_to_transforms(matrix, euler='XYZ'):
        location = matrix.to_translation()
        rotation_quaternion = matrix.to_quaternion()
        axis = matrix.to_quaternion().to_axis_angle()
        rotation_axis_angle = (*axis[0], axis[1])
        rotation_euler = matrix.to_euler(euler)
        scale = matrix.to_scale()

        matrix = type('', (), dict(
            location=location,
            rotation_quaternion=rotation_quaternion,
            rotation_axis_angle=rotation_axis_angle,
            rotation_euler=rotation_euler,
            scale=scale,
        ))

        return matrix

    @staticmethod
    def merge_vertex_groups(ob, vg_A_name, vg_B_name,
    remove: 'keep or remove (vg B)' = False):
        """Get both groups and add them into third"""

        vgroup_A = ob.vertex_groups.get(vg_A_name)
        vgroup_B = ob.vertex_groups.get(vg_B_name)

        if not vgroup_B:
            return

        elif not vgroup_A:
            if remove:
                # The goal group doesn't exist, so just rename the other group
                vgroup_B.name = vg_A_name
                return
            else:
                ob.vertex_groups.new(name=vg_A_name)
                vgroup_A = ob.vertex_groups[vg_A_name]

        # The new tmp group to start layering the old weights
        vgroup = ob.vertex_groups.new(name="TMP" + vg_A_name + "+" + vg_B_name)

        for (id, vert) in enumerate(ob.data.vertices):
            available_groups = [vg_elem.group for vg_elem in vert.groups]
            A = B = 0

            if vgroup_A.index in available_groups:
                A = vgroup_A.weight(id)
            if vgroup_B.index in available_groups:
                B = vgroup_B.weight(id)

            # only add to vertex group is weight is > 0
            sum = A + B
            if sum > 0:
                vgroup.add([id], sum, 'REPLACE')

        if remove:
            ob.vertex_groups.remove(vgroup_B)

        # Now that its weights were transferred, replace the target group
        ob.vertex_groups.remove(vgroup_A)
        vgroup.name = vg_A_name

    @staticmethod
    def multiply_matrix(*matrices):
        from mathutils import Matrix, Vector

        merge = Matrix()
        for mat in matrices:
            if is28: sym = '@'
            if is27 or Is.digit(mat): sym = '*'
            merge = eval(f'{merge!r} {sym} {mat!r}')

        return merge

    @staticmethod
    def multiply_list(*vectors):
        """multiply a list of numbers together (such as Vectors)"""

        sets = dict()
        for tup in vectors:
            for (index, val) in enumerate(tup):
                if index not in sets:
                    sets[index] = list()
                sets[index].append(val)

        merge = list()
        for (index, items) in sets.items():
            value = items.pop(0)
            while items:
                value *= items.pop(0)
            merge.append(value)

        return merge

    @staticmethod
    def name_split_hash(name):
        """Get the original name without the added hash"""

        split = name.rsplit('-')
        if Is.digit(split[-1]):
            name = split[0]

        return name

    # @classmethod
    def poll_workspace(self, context):
        # bl_space_type = 'PROPERTIES'
        # bl_region_type = 'WINDOW'
        # bl_context = ".workspace"
        """Only display a panel if it's in the workspace"""

        if is27: return True

        # return context.area.type != 'VIEW_3D'
        return context.area.type == 'PROPERTIES'

    @staticmethod
    def prefs(name: "__package__" = None) ->\
    "return prefs or prefs.addons[name].preferences":
        """
        return the current addon's preferences property
        """

        if hasattr(bpy.utils, '_preferences'):  # 2.8
            prefs = bpy.utils._preferences
            # context.user_preferences
        elif hasattr(bpy.utils, '_user_preferences'):  # 2.7
            prefs = bpy.utils._user_preferences
            # context.preferences
        else:
            prefs = None

        if name is None:
            return prefs
        else:
            if name in prefs.addons:
                return prefs.addons[name].preferences
            else:
                return prefs.addons[name.split('.')[0]].preferences

    @staticmethod
    def proportional(dist, mode: "string or context" = None, rng: "Random Seed" = None):
        """Convert a number (from 0-1) to its proportional equivalent"""
        from math import sqrt
        from random import random

        if not (0 <= dist <= 1):
            return dist

        if not Is.string(mode) and mode is not None:
            mode = mode.scene.tool_settings.proportional_edit_falloff

        if mode == 'SHARP':
            return dist * dist
        elif mode == 'SMOOTH':
            return 3.0 * dist * dist - 2.0 * dist * dist * dist
        elif mode == 'ROOT':
            return sqrt(dist)
        elif mode == 'LINEAR':
            return dist
        elif mode == 'CONSTANT':
            return 1.0
        elif mode == 'SPHERE':
            return sqrt(2 * dist - dist * dist)
        elif mode == 'RANDOM':
            if (rng is None):
                rng = random()
            return rng * dist
        elif mode == 'INVERSE_SQUARE':
            return dist * (2.0 - dist)
        else:
            # default equivalent to constant
            return 1

    @staticmethod
    def register_collection(cls, **kwargs):
        """
        register a class, then return a pointer of a Pointer property\\
        kwargs are optional additional paramenter to insert in the type\\
        """

        if hasattr(cls, 'is_registered') and (not cls.is_registered):
            bpy.utils.register_class(cls)
        cls_registered = bpy.props.CollectionProperty(type=cls, **kwargs)

        return cls_registered

    @staticmethod
    def register_pointer(cls, **kwargs):
        """
        register a class, then return a pointer of a Pointer property\\
        kwargs are optional additional paramenter to insert in the type\\
        """

        if hasattr(cls, 'is_registered') and (not cls.is_registered):
            bpy.utils.register_class(cls)
        cls_registered = bpy.props.PointerProperty(type=cls, **kwargs)

        return cls_registered

    # https://docs.blender.org/api/blender2.8/bpy.app.timers.html
    @staticmethod
    def register_timer(wait, function, *args, use_threading=False, **keywords):
        """
        Start a function on a looped timer\\
        (or run it outside the function it was called in)\\
        Default wait for 2.8 is 0.0\\
        Default in 2.7 = 0.01
        """

        import bpy
        import threading
        import time
        import functools

        is27 = bpy.app.version < (2, 80, 0)
        is28 = not is27

        def looper(*args, **keywords):
            try:
                exit = function(*args, **keywords)  # Digit for new wait
                if is27 or use_threading:
                    while exit is not None:
                        if exit is not None:
                            time.sleep(exit)
                            exit = function(*args, **keywords)  # Digit for new wait
                elif is28:
                    return (exit, None)[exit is None]
            except:
                utils.error(
                    f"\n\tError with register_timer({function}) @ line#" +
                    utils.line(),
                )
                return

        if is27 or use_threading:
            timer = threading.Timer(
                wait, looper, args=args, kwargs=keywords)
            timer.start()
        elif is28:
            bpy.app.timers.register(
                functools.partial(looper, *args, **keywords),
                first_interval=wait)

    @staticmethod
    def rotate_matrix(matrix, angles_in_degrees: "float or tuple (x, y, z)"):
        from mathutils import Matrix

        for (angle, axis) in zip(angles_in_degrees, ('X', 'Y', 'Z')):
            # define the rotation
            rot_mat = Matrix.Rotation(radians(angle), 4, axis)

            # decompose world_matrix's components, and from them assemble 4x4 matrices
            orig_loc, orig_rot, orig_scale = matrix.decompose()
            orig_loc_mat = Matrix.Translation(orig_loc)
            orig_rot_mat = orig_rot.to_matrix().to_4x4()
            orig_scale_mat = utils.multiply_matrix(
                Matrix.Scale(orig_scale[0], 4, (1, 0, 0)),
                Matrix.Scale(orig_scale[1], 4, (0, 1, 0)),
                Matrix.Scale(orig_scale[2], 4, (0, 0, 1)),
            )

            # assemble the new matrix
            matrix = utils.multiply_matrix(orig_loc_mat, rot_mat, orig_rot_mat, orig_scale_mat)

        return matrix

    @staticmethod
    def scale_range(OldValue, OldMin, OldMax, NewMin, NewMax):
        """Convert a number in a range, relatively to a different range"""

        if Is.iterable(OldValue):
            NewValues = list()
            for (index, OldValue) in enumerate(OldValue):
                OldRange = (OldMax[index] - OldMin[index])
                NewRange = (NewMax[index] - NewMin[index])

                if (OldRange == 0):
                    NewValue = NewMin[index]
                else:
                    NewValue = (((OldValue - OldMin[index]) * NewRange) / OldRange) + NewMin[index]

                NewValues.append(NewValue)

            return NewValues

        OldRange = (OldMax - OldMin)
        NewRange = (NewMax - NewMin)

        if (OldRange == 0):
            NewValue = NewMin
        else:
            NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin

        return NewValue

    @staticmethod
    def string(*args):
        """return a list of items as a merged string"""

        string = str()
        for i in args:
            string += str(i)

        return string

    @staticmethod
    def update(context):
        """Try to update the context"""

        scn = getattr(context, 'scene', None)
        view = getattr(context, 'view_layer', None)
        dps = getattr(context, 'evaluated_depsgraph_get', print)()
        if dps is None:
            dps = getattr(context, 'depsgraph', None)

        if hasattr(view, 'update'):
            view.update()
        elif hasattr(scn, 'update'):
            scn.update()
        elif hasattr(dps, 'update'):
            dps.update()
        else:
            # Display error window
            assert None, (
                "utils.update()!"
                "\ncontext can't find an updater for the scene!"
            )

    @staticmethod
    def update_keyframe_points(context):
        """# The select operator(s) are bugged, and can fail to update selected keys, so"""

        # if (context.area.type == 'DOPESHEET_EDITOR'):
            # bpy.ops.transform.transform(mode='TIME_TRANSLATE')
        # else:
            # bpy.ops.transform.transform()

        # Dopesheet's operator doesn't work, so always use graph's
        area = context.area.type
        if area != 'GRAPH_EDITOR':
            context.area.type = 'GRAPH_EDITOR'

        snap = context.space_data.auto_snap
        context.space_data.auto_snap = 'NONE'

        bpy.ops.transform.transform()

        context.space_data.auto_snap = snap
        if area != 'GRAPH_EDITOR':
            context.area.type = area

    @staticmethod
    class progress:
        """Displays the little black box with 4 numbers as counter"""

        def start(context, min=0, max=9999):
            context.window_manager.progress_begin(min, max)

        def update(context, value=1):
            context.window_manager.progress_update(value)

        def end(context):
            context.window_manager.progress_end()

    class Preferences:
        def draw(self, context):
            layout = self.layout
            self.draw_keymaps(context)

        def draw_keymaps(self, context):
            layout = self.layout

            # cls = self.__class__
            if self.keymaps:
                # layout.label(text="Keymaps:")
                layout.prop(self, 'show_keymaps', text="Keymaps:")

                if self.show_keymaps:
                    utils.draw_keymaps(context, layout, self.keymaps)

        keymaps = None
        show_keymaps: bpy.props.BoolProperty()


# endregion utils

# region: functions
class New:
    "Functions to create data"

    class event_timer:
        "Starts a timer function, to continuously update modal operators"

        def __init__(self, context, time_step=0.001, window=None):
            if window is None:
                window = context.window
            wm = context.window_manager

            self.timer = wm.event_timer_add(time_step, window=window)

        def stop(self, context):
            "Stops the timer from updating"

            wm = context.window_manager
            wm.event_timer_remove(self.timer)

    # -------------------------------------------------------------------------
    # region: Animation

    def fcurve(action, path, index=-1, group=""):
        """Create a new fcurve in an action if it doesn't exist"""

        if group is None:
            group = ""

        fc = Get.fcurve(action, path, index)
        if fc is None:
            fc = action.fcurves.new(path, index=index, action_group=group)

        return fc

    def strip(anim, action, name=True, blend_type=None, extrapolation=None, track=None):
        """Create new strip using specifed action"""

        # Get strip blend and extend modes
        if blend_type is None:
            blend_type = anim.action_blend_type
        if extrapolation is None:
            extrapolation = anim.action_extrapolation

        # Get new strip's name
        if not Is.string(name):
            blend = Get.nla_blend_name(anim)

            if name is True:
                name = f"{blend}: {action.name}"
            elif name is None:
                name = f"{blend}: Action"  # Don't use unique names
            else:
                name = action.name

        # Find or create track to place new strip
        (astart, aend) = (action.frame_range)
        if track:
            pass
        elif anim.nla_tracks:
            active_track = anim.nla_tracks[-1]
            for strip in active_track.strips:
                if active_track.lock:
                    track = anim.nla_tracks.new()
                    break
                sstart = strip.frame_start
                send = strip.frame_end
                if (send <= astart) or (aend <= sstart):
                    # Strip does not conflict with action
                    continue
                if any(((sstart <= astart <= send), (sstart <= aend <= send),
                        (astart <= sstart <= aend), (astart <= send <= aend),)):
                    # The strip is in the range of the action
                    track = anim.nla_tracks.new()
                    break
            else:
                track = active_track
        else:
            track = anim.nla_tracks.new()

        # Create and name new strip
        strip = track.strips.new("", astart, action)
        strip.name = name
        strip.blend_type = blend_type
        strip.extrapolation = extrapolation

        return strip

    # endregion
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # region: Object

    def camera(context, name="Camera", size=1.0, link=True):
        "Create a new camera object"

        data = bpy.data.cameras.new(name)
        obj = New.object(context, name=name, data=data, link=link)
        data.display_size = size

        return obj

    def empty(context, name="Empty", type='PLAIN_AXES', size=1.0, link=True):
        "Create a new empty object"

        empty = New.object(context, name, None, link)
        Set.empty_type(empty, type)
        Set.empty_size(empty, size)

        return empty

    def object(context, name='Object', data=None, link=True):
        """
        Create a new object using the specified data.
        If no data is provided, the object will be an empty.
        """

        object = bpy.data.objects.new(name, data)
        if link: Get.objects(context, link=True).link(object)

        return object

    # endregion
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # region: Curves

    def curve(context, name="Spline IK", link=True):
        "Create a curve object"

        # Create curve
        data = bpy.data.curves.new(name, 'CURVE')

        # Setup curve's display settings
        data.dimensions = '3D'
        data.fill_mode = 'FULL'
        data.bevel_depth = 0.01

        curve = New.object(context, name, data, link)

        if hasattr(curve, 'display_type'):  # 2.8
            curve.display_type = 'WIRE'
        elif hasattr(curve, 'draw_type'):  # 2.7
            curve.draw_type = 'WIRE'

        return curve

    def spline(curve):
        """Add an array to create an actual curve from"""
        # ('POLY', 'BEZIER', 'BSPLINE', 'CARDINAL', 'NURBS')

        return curve.data.splines.new('BEZIER')

    def hook(curve, target, index):
        mod = curve.modifiers.new("Modifier Name", 'HOOK')

        mod.object = target.id_data
        if Is.posebone(target):
            mod.subtarget = target.name

        # modifier sets indices (which are the points + their two handles)
        mod_points = (index * 3 + 0, index * 3 + 1, index * 3 + 2)
        mod.vertex_indices_set(mod_points)
        mod.matrix_inverse = Get.matrix(target).inverted_safe()
        # Set the matrix, to allow the hook to stay with the "posed" bones
            # Don't get target's matrix, because it hasn't been updated yet
        mod.show_expanded = False

        return mod

    # endregion
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # region: Rigs

    def armature(context, name="Armature", display_type=None, link=True):
        "Create and return a rig object"

        data = bpy.data.armatures.new(name)
        # armature.show_in_front = True

        if display_type is not None:
            Set.armature_display_type(data, display_type)

        return New.object(context, name, data, link)

    def bone(context, armature, name="", edits=[], poses=[]):
        "Insert a bone into an armature object"

        if getattr(armature, 'type', None) != 'ARMATURE':
            return

        # active = Get.active(context)
        # Set.active(context, armature)
        # Set.select(armature, True)
        # Set.visible(context, armature, True)
        # mode = armature.mode.replace('EDIT_ARMATURE', 'EDIT')
        Set.in_scene(context, armature)
        is_visible = Is.visible(context, armature)
        Set.visible(context, armature, True)

        mode = armature.mode
            # mode = context.mode.replace('EDIT_ARMATURE', 'EDIT')

        # Go into Edit mode and create a new bone
        ebones = armature.data.edit_bones
        if armature.mode != 'EDIT':
            Set.mode(context, armature, 'EDIT')
        mirror = armature.data.use_mirror_x
        armature.data.use_mirror_x = False
        bone = ebones.new(name)
        bone.tail = ((0, 0, 1))
            # If the bone's head AND tail stay at 0,
            # it gets deleted when leaving edit mode
        name = bone.name
        # for tweak in edits:
        #     if tweak:
        #         tweak(bone)
        armature.data.use_mirror_x = mirror

        pbones = armature.pose.bones
        # if poses:
        #     Set.mode(context, armature, 'POSE')
        #     bone = pbones[name]
        #     for tweak in poses:
        #         if tweak:
        #             tweak(bone)

        # Revert mode change
        if mode != armature.mode:
            # Set.active(active)
            Set.mode(context, armature, mode)
        Set.visible(context, armature, is_visible)

        if armature.mode == 'EDIT':
            bone = ebones[name]
        else:
            bone = pbones[name]

        return bone

    def bone_group(rig, name='Group', color=None):
        ""

        if Is.posebone(rig):
            rig = rig.id_data
        group = rig.pose.bone_groups.new(name=name)

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

        return group

    # endregion
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # region: Mesh

    def mesh(context, name='Mesh', type='PLANE', link=True, **kargs):
        "Create a box-mesh object"

        import bmesh
        from mathutils import Matrix
        # from bpy_extras.object_utils import AddObjectHelper

        width = kargs.get('width', 1.0)
        height = kargs.get('height', 1.0)
        depth = kargs.get('depth', 1.0)
        loc = kargs.get('loc', None)
        rot = kargs.get('rot', None)

        def gen_mesh():
            """
            This function takes inputs and returns vertex and face arrays.
            no actual mesh data creation is done here.
            """

            if type == 'CUBE':
                left = -1.0
                right = +1.0
                top = +1.0
                bottom = -1.0
                front = -1.0
                back = +1.0

                verts = [
                    (right, front, top),  # right-front-top
                    (right, front, bottom),  # right-front-bottom
                    (left, front, bottom),  # left-front-bottom
                    (left, front, top),  # left-front-top
                    (right, back, top),  # right-back-top
                    (right, back, bottom),  # right-back-bottom
                    (left, back, bottom),  # left-back-bottom
                    (left, back, top),  # left-back-top
                ]

                faces = [
                    (3, 2, 1, 0),
                    (5, 6, 7, 4),
                    (1, 5, 4, 0),
                    (2, 6, 5, 1),
                    (3, 7, 6, 2),
                    (7, 3, 0, 4),

                    # (3, 2, 1, 0),
                    # # (4, 7, 6, 5),
                    # # (0, 4, 5, 1),
                    # # (1, 5, 6, 2),
                    # # (2, 6, 7, 3),
                    # # (4, 0, 3, 7),

                ]
            elif type == 'PLANE':
                left = -1.0
                right = +1.0
                top = +0.0
                bottom = -0.0
                front = -1.0
                back = +1.0

                # Plane

                verts = [
                    (left, back, bottom),  # left-back-bottom
                    (left, front, bottom),  # left-front-bottom
                    (right, front, bottom),  # right-front-bottom
                    (right, back, bottom),  # right-back-bottom
                ]

                faces = [
                    (0, 1, 2, 3),  # top
                ]
            elif type == 'POINT':
                verts = [(0, 0, 0)]
                faces = []
            else:  # null mesh
                verts = []
                faces = []

            # apply size
            if loc:
                for (dist, axis) in zip(loc, (0, 1, 2)):
                    for i, v in enumerate(verts):
                        verts[i] = list(verts[i])
                        verts[i][axis] += dist
            for i, v in enumerate(verts):
                verts[i] = v[0] * width, v[1] * depth, v[2] * height

            return (verts, faces)

        verts_loc, faces = gen_mesh()

        mesh = bpy.data.meshes.new(name)

        bm = bmesh.new()
        for v_co in verts_loc:
            bm.verts.new(v_co)

        # if loc:
            # bm.transform(Matrix().Translation(loc))
        if rot:
            bm.transform(utils.rotate_matrix(Matrix(), rot))

        bm.verts.ensure_lookup_table()
        for f_idx in faces:
            bm.faces.new([bm.verts[i] for i in f_idx])

        bm.to_mesh(mesh)
        mesh.update()

        # # add the mesh as an object into the scene with this utility module
        # from bpy_extras import object_utils
        # object_utils.object_data_add(context, mesh, operator=self)

        return New.object(context, name, mesh, link)

        # width: FloatProperty(
            # name="Width",
            # description="Box Width",
            # min=0.01, max=100.0,
            # default=1.0,
        # )
        # height: FloatProperty(
            # name="Height",
            # description="Box Height",
            # min=0.01, max=100.0,
            # default=1.0,
        # )
        # depth: FloatProperty(
            # name="Depth",
            # description="Box Depth",
            # min=0.01, max=100.0,
            # default=1.0,
        # )

            # empty.modifiers.new("Skin", 'SKIN')
            # vt = empty_data.vertices
            # vt.add(2)

            # def measure(first, second):
            #     locx = second[0] - first[0]
            #     locy = second[1] - first[1]
            #     locz = second[2] - first[2]
            #     distance = sqrt((locx)**2 + (locy)**2 + (locz)**2)
            #     return distance

            # if type(bone) is bpy.types.Bone:
            #     co = bone.length
            # elif type(bone) is bpy.types.PoseBone:
            #     co = bone.bone.length
            # elif type(bone) is bpy.types.Object and bone.data.polygons:
            #     co = measure(bone.matrix_world.to_translation(), bone.data.polygons[-1].center)
            #     co = measure(bone.matrix_world.to_translation(), get_tail(obj=bone))
            # elif type(bone) is bpy.types.Object and len(bone.data.vertices) > 1:
            #     co = measure(bone.data.vertices[0].co, bone.data.vertices[-1].co)
            #     co = measure(bone.data.vertices[0].co, bone.data.vertices[-1].co)
            #     co = measure(bone.matrix_world.to_translation(), get_tail(obj=bone))
            # else:
            #     co = get_tail(None, bone, bone)
            #     if co is None:
            #         co = 0.2
            #     else:
            #         co = co[1]
            # vt[1].co[1] = co

            #     # if type(bone) is bpy.types.Bone:
            #         # co = bone.length
            #     # elif type(bone) is bpy.types.PoseBone:
            #         # co = bone.bone.length
            #     # elif type(bone) is bpy.types.Object and bone.data.polygons:
            #         # co = measure(bone.matrix_world.to_translation(), bone.data.polygons[-1].center)
            #     # elif type(bone) is bpy.types.Object and len(bone.data.vertices) > 1:
            #         # co = measure(bone.data.vertices[0].co, bone.data.vertices[-1].co)
            #     # else:
            #         # co = 0.2
            #     # vt[1].co[1] = co
            #     # vt[1].co[1] = bone.data.polygons[-1].center[1]

            # # --------- fix tail with a previous empty
            # # if type(bone) is bpy.types.Object and len(bone.data.vertices) > 1:
            #     # co0 = vt[0].co[1]
            #     # co1 = vt[1].co[1]
            #     # vt[0].co[1] = co1
            #     # vt[1].co[1] = co0

            # # ------- Vertex Groups, to mimic bone head/tail
            # vg = empty.vertex_groups
            # head = vg.new('head')
            # tail = vg.new('tail')
            # head.add([0], 1.0, 'REPLACE')
            # tail.add([1], 1.0, 'REPLACE')

            # # ----------- Skin Modifier
            # vd = empty_data.skin_vertices[0].data
            # vd[0].radius = 2 * [0.020]
            # vd[1].radius = 2 * [0.015]
            # for v in vd:
            #     v.use_root = False

            # return empty

    # endregion
    # -------------------------------------------------------------------------


class Set:
    "Functions to apply properties or status to data"

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
            #     Set.mode(context, src, 'OBJECT')
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
            Set.mode(context, target, 'POSE')

    def constraint_toggle(context, srcs, constraints, influence=None, insert_key=None):
        """Disable constraint while maintaining the visual transform."""

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

    def mode(context, target, mode, keep_visiblity=True):
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


class keyframe:
    "shortcuts for inserting keyframes"


    # these prefixes should be avoided, as they are not really bones
    # that animators should be touching (or need to touch)
    badBonePrefixes = (
        'DEF',
        'GEO',
        'MCH',
        'ORG',
        'COR',
        'VIS',
        # ... more can be added here as you need in your own rigs ...
    )

    # file:///C:\Program%20Files\Blender%20Foundation\Blender\2.80\scripts\startup\keyingsets_builtins.py
    # file:///C:\Program%20Files\Blender%20Foundation\Blender\2.80\scripts\modules\keyingsets_utils.py

    def group_name(src):
        if Is.object(src):
            return "Object Transforms"
        else:
            return src.name

    def poll_unlock(src, path, index=-1):

        if path.endswith('location'):
            # Disabled for keyframe removal for when a bone gets keyed anyway
            # if Is.connected(src):
                # return

            if index < 0:
                return (True not in src.lock_location)
            else:
                return (not src.lock_location[index])
        elif path.endswith('scale'):
            if index < 0:
                return (True not in src.lock_scale)
            else:
                return (not src.lock_scale[index])
        elif ('rotation' in path):
            if path == 'rotation':
                path = 'rotation_' + src.rotation_mode.lower()

            # Placed at bottom in case bone's name contains "rotation"
            if path.endswith('quaternion') or path.endswith('axis_angle'):
                unlock_w = ((not src.lock_rotations_4d) or (not src.lock_rotation_w))
                unlock_q = (unlock_w or (False in src.lock_rotation))
                    # One of the locks are disabled, so keyframe all if Combine

                if (not unlock_q):
                    # All locks are active
                    return False

                if path.endswith('quaternion') and (index >= 0):
                    # Check if animation's blend type is COMBINE,
                    # which keys all quaternions

                    anim = getattr(src.id_data, 'animation_data', None)
                    if (anim and anim.use_tweak_mode):
                        found = None
                        for t in anim.nla_tracks:
                            if found:
                                break
                            for s in t.strips:
                                if s.active:
                                    if (s.blend_type == 'COMBINE'):
                                        index = -1
                                    found = True
                                    break
                    elif anim and (anim.action_blend_type == 'COMBINE'):
                        index = -1

                if index < 0:
                    return unlock_q
                elif index == 0:
                    # lock 4d = W unlocked
                    return unlock_w
                else:
                    return (not src.lock_rotation[index - 1])
            else:  # euler, XYZ, ZXY etc
                if index < 0:
                    return (True not in src.lock_rotation)
                else:
                    return (not src.lock_rotation[index])

        return True  # Path isn't in transforms, so assume it's not lockable

    def poll_insert(context, insert_key=None, src=None):
        if (src is not None):
            if Is.string(src):
                name = src
            else:
                name = getattr(src, 'name', '')
            if name.startswith(keyframe.badBonePrefixes):
                return False
        if insert_key is None:
            return keyframe.use_auto(context)

        return insert_key

    def poll_keyingset(context, attribute):
        """
        Determine if an attribute is enabled in the variable keyingset
        """

        # addon to store custom keyingset
        # prefs = utils.prefs()
        # addon = prefs.addons.get('zpy_animation_keyingsets')
        addon = getattr(context.scene, 'keying_set_variable', None)

        # If the addon isn't enabled, or the property isn't listed in it's props
        # there is no variable keyingset, or the property can't "limit" the prop
        # In both cases, return true to run keyframe insertion as intended
        return getattr(addon, attribute, True)

    def use_auto(context, set=None):

        ts = context.scene.tool_settings
        if set is not None:
            ts.use_keyframe_insert_auto = set

        return ts.use_keyframe_insert_auto

    def use_keyingset(context, set=None):

        ts = context.scene.tool_settings
        if set is not None:
            ts.use_keyframe_insert_keyingset = set

        return ts.use_keyframe_insert_keyingset

    # Keyframe insertion functions

    def manual(context, src, path, **kargs):
        "Insert a keyframe manually. Sub is for sub targets, like constraints"
        # kargs:
            # sub=None, index=-1, frame=None, value=None,
            # group=None, insert_key=None, action=None, context=None

        insert_key = kargs.get('insert_key', None)
        sub = kargs.get('sub', None)
        index = kargs.get('index', -1)
        frame = kargs.get('frame', None)
        value = kargs.get('value', None)
        group = kargs.get('group', None)
        action = kargs.get('action', None)
        results = kargs.get('results', list())
        options = kargs.get('options', set())
        delete = kargs.get('delete', False)

        keyframe_type = kargs.get('type', context.scene.tool_settings.keyframe_type)

        if not keyframe.poll_insert(context, insert_key, src=src):
            return results

        # if frame is None:
        #     frame = context.scene.frame_current
        # if group is None:
        #     group = keyframe.group_name(src)
        # src.keyframe_insert(path, index=index, frame=frame, group=group)
        # return

        if group is None:
            group = keyframe.group_name(src)

        obj = src.id_data
        anim = obj.animation_data_create()
        if action is None:
            key_in_action = True
            action = anim.action
            if action is None:
                action = anim.action = bpy.data.actions.new("Action")
        else:
            # When using a specified action, don't insert
            # keyframes later to determine the property's value
            key_in_action = False

        if frame is None:
            frame = context.scene.frame_current_final

        strip = Get.active_strip(anim) if anim.use_tweak_mode else None

        if strip:
            if not (strip.frame_start <= frame <= strip.frame_end):
                # frame outside of the strip's bounds
                key_in_action = False

            tweak_frame = Get.frame_to_strip(context, anim, frame=frame)
        else:
            tweak_frame = frame

        if Is.posebone(src) and not path.startswith('pose.bones'):
            keypath = utils.string('pose.bones[\"', src.name, '\"]', '' if path.startswith('[') else '.', path)
        elif not Is.object(src) and hasattr(src, 'path_from_id'):
            keypath = src.path_from_id(path)
        else:
            keypath = path

        # Find the value(s) to insert keyframes for
        if hasattr(sub, path):
            base = getattr(sub, path)
        elif hasattr(src, path):
            base = getattr(src, path)
        else:
            base = eval(f'{obj!r}.{keypath}')

        if value is None:
            prop = base
        else:
            if Is.iterable(base) and not Is.iterable(value):
                prop = [value for i in base]
            elif not Is.iterable(base) and Is.iterable(value):
                prop = value[(index, 0)[index == -1]]
            else:
                prop = value

        if (not Is.iterable(prop)):
            if index != -1:
                index = -1
            if (not Is.iterable(prop)):
                props = [(index, prop)]
            else:
                props = [(index, prop[index])]
        elif (index == -1):
            props = list(enumerate(prop))
        else:
            props = [(index, prop[index])]

        def save_quats():
            quats = [0, 0, 0, 0]
            for index in range(4):
                fc = Get.fcurve(action, keypath, index=index)
                if fc:
                    quats[index] = len(fc.keyframe_points)
            return quats

        def restore_quats(skip):
            nonlocal quats
            for index in range(4):
                if index == skip:
                    continue
                fc = Get.fcurve(action, keypath, index=index)
                if fc and len(fc.keyframe_points) > quats[index]:
                    for key in fc.keyframe_points:
                        if key.co.x == tweak_frame:
                            fc.keyframe_points.remove(key)
                            break
                    else:
                        # Shouldn't happen but backup in case fail to find key
                        key_args = dict(index=index, frame=frame, group=group)
                        if sub is None:
                            src.keyframe_delete(path, **key_args)
                        else:
                            sub.keyframe_delete(path, **key_args)

        if path.endswith('rotation_quaternion') and (
                (strip and strip.blend_type == 'COMBINE') or
                (not strip and anim.action_blend_type == 'COMBINE')):
            # Combine forces keyframe insertion on all 4 channels, so reset them
            quats = save_quats()
        else:
            quats = None

        # Create curve(s) (if needed) and keyframe(s)
        for (i, v) in props:
            fc = Get.fcurve(action, keypath, i)
            new_fc = not bool(fc)
            if new_fc:
                fc = New.fcurve(action, keypath, index=i, group=group)

            if fc.lock:
                # Internal ops don't allow keyframing locked channels, so :p
                results.append((fc, None))
                continue

            if delete:
                results.append((fc, src.keyframe_delete(path, frame=frame)))
                continue

            count = len(fc.keyframe_points)

            if (value is None) and key_in_action:
                key_args = dict(index=i, frame=frame, group=group, options=options)
                if sub is None:
                    src.keyframe_insert(path, **key_args)
                else:
                    sub.keyframe_insert(path, **key_args)
                v = fc.evaluate(tweak_frame)

            key = fc.keyframe_points.insert(tweak_frame, v, options={'FAST'})
                    # options=set({'REPLACE', 'NEEDED', 'FAST'})
            # src.keyframe_insert(path, index=i, frame=frame, group=group)

            if quats:
                restore_quats(skip=i)
                quats[i] = len(fc.keyframe_points)
                # Update key count for current index, to not remove it later

            # Update keyframe to use default preferences values

            edit = utils.prefs().edit

            key.handle_left_type = key.handle_right_type = \
                edit.keyframe_new_handle_type
            if new_fc:
                # When inserting keyframes, only change their interpolation type if the fcurve is new
                key.interpolation = edit.keyframe_new_interpolation_type

            if len(fc.keyframe_points) > count:
                # New keyframe was added
                key.type = keyframe_type

            results.append((fc, key))

        if kargs.get('index', -1) == -1:
            return results
        else:
            return results[0]

    def prop(context, src, path, **kargs):
        """
        Insert a keyframe using the builtin function
        kargs:
            index=-1, frame=None, group=None, options=set({}),
            insert_key=None
        """
        insert_key = kargs.get('insert_key', None)
        index = kargs.get('index', -1)
        frame = kargs.get('frame', None)
        group = kargs.get('group', None)
        options = kargs.get('options', set({}))
        # Options:
        # - INSERTKEY_NEEDED		Only insert keyframes where they're needed in the relevant F-Curves.
        # - INSERTKEY_VISUAL		Insert keyframes based on 'visual transforms'.
        # - INSERTKEY_XYZ_TO_RGB	Color for newly added transformation F-Curves (Location, Rotation, Scale) is based on the transform axis.
        # - INSERTKEY_REPLACE		Only replace already exising keyframes.
        # - INSERTKEY_AVAILABLE		Only insert into already existing F-Curves.
        # - INSERTKEY_CYCLE_AWARE	Take cyclic extrapolation into account (Cycle-Aware Keying option).

        if not keyframe.poll_insert(context, insert_key, src=src):
            return
        # if (Is.posebone(src) and Is.connected(src)):
            # return

        if frame is None:
            frame = context.scene.frame_current_final

        if group is None:
            group = keyframe.group_name(src)

        src.keyframe_insert(path,
            index=index, frame=frame, group=group, options=options
        )

    def constraints(context, src, **kargs):
        """
        kargs:
            frame=None, group=None, action=None, insert_key=None,
        """

        insert_key = kargs.get('insert_key')
        kargs['results'] = kargs.get('results', list())

        if not keyframe.poll_insert(context, insert_key, src=src):
            return

        for con in src.constraints:
            kargs['sub'] = con
            # kargs['options'] = {'INSERTKEY_NEEDED'}
            if con.name.startswith('Temp'):
                continue
            if src.constraints_relative.locks.find(con.name) != -1:
                continue

            keyframe.manual(context, src, 'influence', **kargs)
            # if hasattr(con, 'chain_count'):
            #     keyframe.manual(context, src, 'chain_count', sub=con, frame=frame,
            #                     group=group, # options={'INSERTKEY_NEEDED'},
            #                     insert_key=insert_key)

        return kargs['results']

    def location(context, src, **kargs):
        """
        kargs:
            frame=None, group=None, action=None, insert_key=None,
            unlock=False,  # keyframe locked channels
            connected=False,  # ignore whether or not a bone is connected
        """

        insert_key = kargs.get('insert_key')
        unlock = kargs.get('unlock', False)
        kargs['results'] = kargs.get('results', list())

        if not keyframe.poll_insert(context, insert_key, src=src):
            return

        if kargs.get('connected') and Is.connected(src):
            return

        args = (context, src, 'location')

        if unlock or (True not in src.lock_location):
            keyframe.manual(*args, index=-1, **kargs)
        else:
            for i in range(3):
                if (not src.lock_location[i]):
                    keyframe.manual(*args, index=i, **kargs)

        return kargs['results']

    def rotation(context, src, **kargs):
        """
        kargs:
            frame=None, group=None, action=None, insert_key=None,
            unlock=False,  # keyframe locked channels
        """
        insert_key = kargs.get('insert_key')
        unlock = kargs.get('unlock', False)
        kargs['results'] = kargs.get('results', list())

        if not keyframe.poll_insert(context, insert_key, src):
            return

        rotation_mode = ('rotation_' + src.rotation_mode.lower())

        if src.rotation_mode in {'QUATERNION', 'AXIS_ANGLE'}:
            unlock_w = ((not src.lock_rotations_4d) or (not src.lock_rotation_w))
            if (src.rotation_mode == 'QUATERNION'):
                unlock_q = (unlock_w or (False in src.lock_rotation))
                prop = 'rotation_quaternion'
                if (not unlock) and unlock_q:
                    # One of the locks are disabled, so keyframe all if Combine

                    anim = getattr(src.id_data, 'animation_data', None)
                    if anim and anim.use_tweak_mode:
                        for t in anim.nla_tracks:
                            for s in t.strips:
                                if s.active:
                                    if (s.blend_type == 'COMBINE'):
                                        unlock = True
                                    break
                    elif anim and anim.action_blend_type == 'COMBINE':
                        unlock = True
            elif (src.rotation_mode == 'AXIS_ANGLE'):
                prop = 'rotation_axis_angle'

            args = (context, src, rotation_mode)
            if unlock or unlock_w:
                keyframe.manual(*args, index=0, **kargs)  # w = 0
            for i in range(3):
                if unlock or (not src.lock_rotation[i]):
                    keyframe.manual(*args, index=i + 1, **kargs)
                        # i + 1, since here x/y/z = 1,2,3, and w=0
        else:  # euler, XYZ, ZXY etc
            args = (context, src, 'rotation_euler')
            if unlock or (True not in src.lock_rotation):
                keyframe.manual(*args, index=-1, **kargs)
            else:
                for i in range(3):
                    if not src.lock_rotation[i]:
                        keyframe.manual(*args, index=i, **kargs)

        return kargs['results']

    def scale(context, src, **kargs):
        """
        # kargs:
            # frame=None, group=None, action=None, insert_key=None,
            unlock=False,  # keyframe locked channels
        """
        insert_key = kargs.get('insert_key')
        unlock = kargs.get('unlock', False)
        kargs['results'] = kargs.get('results', list())

        if not keyframe.poll_insert(context, insert_key, src=src):
            return

        args = (context, src, 'scale')
        if unlock or (True not in src.lock_scale):
            keyframe.manual(*args, index=-1, **kargs)
        else:
            for i in range(3):
                if (not src.lock_scale[i]):
                    keyframe.manual(*args, index=i, **kargs)

        return kargs['results']

    # Macro Keyframe insertion functions


    def all(context, src, **kargs):
        """
        Keyframe Loc/Rot/Scale + Constraints
        # kargs:
            # frame=None, group=None, action=None, insert_key=None,
        """
        insert_key = kargs.get('insert_key')
        kargs['results'] = kargs.get('results', list())

        if keyframe.poll_insert(context, insert_key, src=src):
            if keyframe.poll_keyingset(context, 'key_location'):
                keyframe.location(context, src, **kargs)
            if keyframe.poll_keyingset(context, 'key_rotation'):
                keyframe.rotation(context, src, **kargs)
            if keyframe.poll_keyingset(context, 'key_scale'):
                keyframe.scale(context, src, **kargs)
            if keyframe.poll_keyingset(context, 'key_constraints'):
                keyframe.constraints(context, src, **kargs)

        return kargs['results']

    def transforms(context, src, **kargs):
        """
        Keyframe Loc/Rot/Scale
        # kargs:
            # frame=None, group=None, action=None, insert_key=None,
        """
        insert_key = kargs.get('insert_key')
        kargs['results'] = kargs.get('results', list())

        if keyframe.poll_insert(context, insert_key, src=src):
            if keyframe.poll_keyingset(context, 'key_location'):
                keyframe.location(context, src, **kargs)
            if keyframe.poll_keyingset(context, 'key_rotation'):
                keyframe.rotation(context, src, **kargs)
            if keyframe.poll_keyingset(context, 'key_scale'):
                keyframe.scale(context, src, **kargs)

        return kargs['results']

    def keyingset(context, default=True, **kargs):
        """
        Use bpy.ops to insert keyframe.
        default means to only use the keying set if the keying set button is enabled
        """

        insert_key = kargs.get('insert_key')
        if not keyframe.poll_insert(context, insert_key):
            return

        ks = context.scene.keying_sets_all.active
        if (default and not keyframe.use_keyingset(context)) or \
                ks is None:
            ks = 'LocRotScale'
        else:
            ks = ks.bl_idname

        bones = kargs.get('bones', list())
        objects = kargs.get('objects', list())
        selected = kargs.get('selected', list())

        # if bones is None: bones = Get.selected_pose_bones(context)
        # if objects is None: objects = Get.selected_objects(context)

        for src in selected:
            if Is.posebone(src):
                bones.append(src)
            elif Is.object(src):
                objects.append(src)
            else:
                assert None, ("This is not a bone or object", src)

        if kargs.get('skip_bones', False):
            # This step removes bones from badBonePrefixes
            for src in bones.copy():
                if not keyframe.poll_insert(context, insert_key, src=src):
                    bones.remove(src)

        # if keyframe.use_keyingset():
            # bpy.ops.anim.keyframe_insert(type=ks, confirm_success=False)
        # # else:
            # for src in KSI.iter():
                # keyframe.all(src)


        if (bones or objects):
            try:
                return bpy.ops.anim.keyframe_insert(
                    dict(selected_pose_bones=bones, selected_objects=objects),
                    type=ks, confirm_success=False)
            except Exception as ex:
                pass
        else:
            "Nothing to keyframe"


class popup:
    """shortcuts for invoking popup windows"""

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


# endregion functions

# region: functions that are initialized()
class register_keymaps:
    def __init__(self):
        self.addon_keymaps = dict()
        self.toggle_keymaps = list()
        self.toggle_functions = list()
        self.has_toggles = False

    def get_key(self, name):
        """Find the window name in the active keymaps"""

        # key = bpy.context.window_manager.keyconfigs.active.keymaps.get(name)
        # if key:
            # class key:
                # region_type = key.region_type
                # space_type = key.space_type
        key = self.keymap_list.get(name)
        if key:
            class key:
                space_type = key[0]
                region_type = key[1]
        return key

    def add(self, idname,
        name='Window', type='NONE', value='PRESS',
        any=False, shift=False, ctrl=False, alt=False, oskey=False,
        key_modifier='NONE', head=True,
        # space_type='EMPTY', region_type='WINDOW',
        modal=False, tool=False, **properties):
        "Register a hotkey for an operator"

        import bpy

        if hasattr(idname, 'bl_idname'):
            idname = idname.bl_idname

        if ('properties' in properties and len(properties) == 1) \
                and isinstance(properties['properties'], dict):
            # Using old method
            properties = properties['properties']

        source = self.get_key(name)

        poll_skip = bool(
            not idname or

            # Try to avoid registering bad keymaps
            not isinstance(self.addon_keymaps, dict) or

            not utils.find_op(idname) or
            not source
        )
        if poll_skip:
            if not source:
                print(name, "not a valid keymap space")
            return

        region_type = source.region_type
        space_type = source.space_type

        # Keymaps: https://docs.blender.org/api/blender_python_api_master/bpy.types.KeyMaps.html#bpy.types.KeyMaps.new
        keymaps = bpy.context.window_manager.keyconfigs.addon.keymaps

        # Keymap: https://docs.blender.org/api/blender_python_api_master/bpy.types.KeyMap.html
        keymap = keymaps.new(name=name, space_type=space_type,
            region_type=region_type, modal=modal, tool=tool)

        # Keymap_item: https://docs.blender.org/api/blender_python_api_master/bpy.types.KeyMapItems.html
        kmi = keymap.keymap_items.new(
            idname=idname, type=type, value=value,
            any=any, shift=shift, ctrl=ctrl, alt=alt, oskey=oskey,
            key_modifier=key_modifier, head=head,
        )

        try:
            if isinstance(properties, (list, tuple, set)):
                for prop in properties:
                    setattr(kmi.properties, *prop)
            elif isinstance(properties, dict):
                for prop in properties:
                    setattr(kmi.properties, prop, properties[prop])
            else:
                # debug("KEYMAP SETUP PROPERTIES FAIL", self.type(properties))
                # debug('     ', properties)
                pass
        except:
            # error('Failed to set attrib for propety during Keymap Registry')
            # debug('\t', properties, end="\n\t")
            # debug(properties)
            # debug(prop, properties[prop])
            pass

        if keymap not in self.addon_keymaps:
            self.addon_keymaps[keymap] = list()
        self.addon_keymaps[keymap].append(kmi)

        return kmi.properties

    def remove(self):
        "Revert keymap back to normal"
        for (keymap, kmis) in self.addon_keymaps.items():
            for kmi in kmis:
                # Check to see it wasn't already removed (since 2.90)
                if kmi in keymap.keymap_items[:]:
                    keymap.keymap_items.remove(kmi)
        self.addon_keymaps.clear()

        for (kmi) in self.toggle_keymaps:
            kmi.active = True
        self.toggle_keymaps.clear()

        if self.has_toggles:
            self.has_toggles = False
            self.toggle_functions.clear()
            if self.refresh_toggles in bpy.app.handlers.load_post:
                bpy.app.handlers.load_post.remove(self.refresh_toggles)

    def toggle(self, idname,
        name='Window', type='NONE', value='PRESS',
        any=False, shift=False, ctrl=False, alt=False, oskey=False,
        key_modifier='NONE', addon=False,
        **properties):
        # space_type='EMPTY', region_type='WINDOW',
        "Disable a keymap without deleting it"

        import bpy

        keymap = None
        kmi = None

        # Determine whether or not the Blender window should exist already
        if not self.get_key(name):
            # debug(f"ERROR: Keymap.Toggle can't find specified area name"
            #       f", so it's incorrect or custom\n{name!r} > {idname!r}\n\tkey [{type}]")
            return

        def get_keymap():
            kf = bpy.context.window_manager.keyconfigs
            if addon:
                return kf.addon.keymaps.get(name)
            else:
                return kf.active.keymaps.get(name)

        def get_kmi(keymap):
            # kmi = None
            kmi = []
            for key in keymap.keymap_items:
                if key.idname != idname:
                    continue
                if all((
                        key.type == type,
                        key.value == value,
                        (key.any is any and any) or all((
                            key.shift is shift,
                            key.ctrl is ctrl,
                            key.alt is alt,
                            key.oskey is oskey
                            )),
                        key.key_modifier == key_modifier,
                        )):
                    kmi.append(key)
                    # kmi = key
                    # break
            return kmi

        def toggle_kmi(kmi):
            for key in kmi:
                key.active = False
            # kmi.active = False
            # if type(kmi) is list:
            self.toggle_keymaps.extend(kmi)
            # else:
                # self.toggle_keymaps.append(kmi)

        timer = 50

        @bpy.app.handlers.persistent
        def start(scn=None):
            nonlocal timer, keymap, kmi
            timer -= 1

            if timer < 0:
                # debug(f"ERROR: Keymap.Toggle can't find specified keymap_item"
                #       f", so it must not be designated correctly"
                #       f"\n{name!r} > {idname!r}\n\tkey [{type}]")
                return  # exit

            keymap = get_keymap()
            if not keymap:
                return 0.0  # loop

            kmi = get_kmi(keymap)
            if not kmi:
                return 0.0  # loop

            toggle_kmi(kmi)
            return  # exit

        if not(self.has_toggles):
            @bpy.app.handlers.persistent
            def refresh_toggles(scn):
                def start():
                    for func in self.toggle_functions:
                        func()

                # Run in a timer because when creating a new file,
                # loading preferences takes place after loading file
                # which resets the keymap toggles
                bpy.app.timers.register(start, persistent=True)

            self.has_toggles = True
            self.refresh_toggles = refresh_toggles
            bpy.app.handlers.load_post.append(refresh_toggles)
        self.toggle_functions.append(start)

        # https://docs.blender.org/api/blender2.8/bpy.app.timers.html
        bpy.app.timers.register(start, persistent=True)
            # What peristent does is allow it to run when starting blender from a file

        return kmi

    class enums:
        "List of available items for keymap args"

        type_and_modifier = [
            'NONE',  #
            'LEFTMOUSE',  # Left Mouse, LMB
            'MIDDLEMOUSE',  # Middle Mouse, MMB
            'RIGHTMOUSE',  # Right Mouse, RMB
            'BUTTON4MOUSE',  # Button4 Mouse, MB4
            'BUTTON5MOUSE',  # Button5 Mouse, MB5
            'BUTTON6MOUSE',  # Button6 Mouse, MB6
            'BUTTON7MOUSE',  # Button7 Mouse, MB7
            'ACTIONMOUSE',  # Action Mouse, MBA
            'SELECTMOUSE',  # Select Mouse, MBS
            'PEN',  # Pen
            'ERASER',  # Eraser
            'MOUSEMOVE',  # Mouse Move, MsMov
            'INBETWEEN_MOUSEMOVE',  # In-between Move, MsSubMov
            'TRACKPADPAN',  # Mouse/Trackpad Pan, MsPan
            'TRACKPADZOOM',  # Mouse/Trackpad Zoom, MsZoom
            'MOUSEROTATE',  # Mouse/Trackpad Rotate, MsRot
            'WHEELUPMOUSE',  # Wheel Up, WhUp
            'WHEELDOWNMOUSE',  # Wheel Down, WhDown
            'WHEELINMOUSE',  # Wheel In, WhIn
            'WHEELOUTMOUSE',  # Wheel Out, WhOut
            'EVT_TWEAK_L',  # Tweak Left, TwkL
            'EVT_TWEAK_M',  # Tweak Middle, TwkM
            'EVT_TWEAK_R',  # Tweak Right, TwkR
            'EVT_TWEAK_A',  # Tweak Action, TwkA
            'EVT_TWEAK_S',  # Tweak Select, TwkS
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'ZERO', 'ONE', 'TWO', 'THREE', 'FOUR',
            'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE',
            'LEFT_CTRL', 'LEFT_ALT', 'LEFT_SHIFT',
            'RIGHT_ALT', 'RIGHT_CTRL', 'RIGHT_SHIFT',
            'OSKEY',  # OS Key, Cmd
            'GRLESS',  # Grless
            'ESC',  # Esc
            'TAB',  # Tab
            'RET',  # Return, Enter
            'SPACE',  # Spacebar, Space
            'LINE_FEED',  # Line Feed
            'BACK_SPACE',  # Back Space, BkSpace
            'DEL',  # Delete, Del
            'SEMI_COLON',  # ;
            'PERIOD',  # .
            'COMMA',  # ,
            'QUOTE',  # ”
            'ACCENT_GRAVE',  # `
            'MINUS',  # -
            'PLUS',  # +
            'SLASH',  # /
            'BACK_SLASH',  # \
            'EQUAL',  # =
            'LEFT_BRACKET',  # [
            'RIGHT_BRACKET',  # ]
            'LEFT_ARROW',  # Left Arrow, ←
            'DOWN_ARROW',  # Down Arrow, ↓
            'RIGHT_ARROW',  # Right Arrow, →
            'UP_ARROW',  # Up Arrow, ↑
            'NUMPAD_2',  # Numpad 2, Pad2
            'NUMPAD_4',  # Numpad 4, Pad4
            'NUMPAD_6',  # Numpad 6, Pad6
            'NUMPAD_8',  # Numpad 8, Pad8
            'NUMPAD_1',  # Numpad 1, Pad1
            'NUMPAD_3',  # Numpad 3, Pad3
            'NUMPAD_5',  # Numpad 5, Pad5
            'NUMPAD_7',  # Numpad 7, Pad7
            'NUMPAD_9',  # Numpad 9, Pad9
            'NUMPAD_PERIOD',  # Numpad ., Pad.
            'NUMPAD_SLASH',  # Numpad /, Pad/
            'NUMPAD_ASTERIX',  # Numpad *, Pad*
            'NUMPAD_0',  # Numpad 0, Pad0
            'NUMPAD_MINUS',  # Numpad -, Pad-
            'NUMPAD_ENTER',  # Numpad Enter, PadEnter
            'NUMPAD_PLUS',  # Numpad +, Pad+
            'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
            'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19',
            'PAUSE',  # Pause
            'INSERT',  # Insert, Ins
            'HOME',  # Home
            'PAGE_UP',  # Page Up, PgUp
            'PAGE_DOWN',  # Page Down, PgDown
            'END',  # End
            'MEDIA_PLAY',  # Media Play/Pause, >/||
            'MEDIA_STOP',  # Media Stop, Stop
            'MEDIA_FIRST',  # Media First, |<<
            'MEDIA_LAST',  # Media Last, >>|
            'TEXTINPUT',  # Text Input, TxtIn
            'WINDOW_DEACTIVATE',  # Window Deactivate
            'TIMER',  # Timer, Tmr
            'TIMER0',  # Timer 0, Tmr0
            'TIMER1',  # Timer 1, Tmr1
            'TIMER2',  # Timer 2, Tmr2
            'TIMER_JOBS',  # Timer Jobs, TmrJob
            'TIMER_AUTOSAVE',  # Timer Autosave, TmrSave
            'TIMER_REPORT',  # Timer Report, TmrReport
            'TIMERREGION',  # Timer Region, TmrReg
            'NDOF_MOTION',  # NDOF Motion, NdofMov
            'NDOF_BUTTON_MENU',  # NDOF Menu, NdofMenu
            'NDOF_BUTTON_FIT',  # NDOF Fit, NdofFit
            'NDOF_BUTTON_TOP',  # NDOF Top, Ndof↑
            'NDOF_BUTTON_BOTTOM',  # NDOF Bottom, Ndof↓
            'NDOF_BUTTON_LEFT',  # NDOF Left, Ndof←
            'NDOF_BUTTON_RIGHT',  # NDOF Right, Ndof→
            'NDOF_BUTTON_FRONT',  # NDOF Front, NdofFront
            'NDOF_BUTTON_BACK',  # NDOF Back, NdofBack
            'NDOF_BUTTON_ISO1',  # NDOF Isometric 1, NdofIso1
            'NDOF_BUTTON_ISO2',  # NDOF Isometric 2, NdofIso2
            'NDOF_BUTTON_ROLL_CW',  # NDOF Roll CW, NdofRCW
            'NDOF_BUTTON_ROLL_CCW',  # NDOF Roll CCW, NdofRCCW
            'NDOF_BUTTON_SPIN_CW',  # NDOF Spin CW, NdofSCW
            'NDOF_BUTTON_SPIN_CCW',  # NDOF Spin CCW, NdofSCCW
            'NDOF_BUTTON_TILT_CW',  # NDOF Tilt CW, NdofTCW
            'NDOF_BUTTON_TILT_CCW',  # NDOF Tilt CCW, NdofTCCW
            'NDOF_BUTTON_ROTATE',  # NDOF Rotate, NdofRot
            'NDOF_BUTTON_PANZOOM',  # NDOF Pan/Zoom, NdofPanZoom
            'NDOF_BUTTON_DOMINANT',  # NDOF Dominant, NdofDom
            'NDOF_BUTTON_PLUS',  # NDOF Plus, Ndof+
            'NDOF_BUTTON_MINUS',  # NDOF Minus, Ndof-
            'NDOF_BUTTON_ESC',  # NDOF Esc, NdofEsc
            'NDOF_BUTTON_ALT',  # NDOF Alt, NdofAlt
            'NDOF_BUTTON_SHIFT',  # NDOF Shift, NdofShift
            'NDOF_BUTTON_CTRL',  # NDOF Ctrl, NdofCtrl
            'NDOF_BUTTON_1',  # NDOF Button 1, NdofB1
            'NDOF_BUTTON_2',  # NDOF Button 2, NdofB2
            'NDOF_BUTTON_3',  # NDOF Button 3, NdofB3
            'NDOF_BUTTON_4',  # NDOF Button 4, NdofB4
            'NDOF_BUTTON_5',  # NDOF Button 5, NdofB5
            'NDOF_BUTTON_6',  # NDOF Button 6, NdofB6
            'NDOF_BUTTON_7',  # NDOF Button 7, NdofB7
            'NDOF_BUTTON_8',  # NDOF Button 8, NdofB8
            'NDOF_BUTTON_9',  # NDOF Button 9, NdofB9
            'NDOF_BUTTON_10',  # NDOF Button 10, NdofB10
            'NDOF_BUTTON_A',  # NDOF Button A, NdofBA
            'NDOF_BUTTON_B',  # NDOF Button B, NdofBB
            'NDOF_BUTTON_C',  # NDOF Button C, NdofBC
        ]
        value = [
            'ANY', 'NOTHING', 'PRESS', 'RELEASE', 'CLICK', 'DOUBLE_CLICK',
            'NORTH', 'NORTH_EAST', 'EAST', 'SOUTH_EAST', 'SOUTH',
            'SOUTH_WEST', 'WEST', 'NORTH_WEST'
        ]
        space_types = {
            'Window',
            'Screen',
            'Screen Editing',
            'User Interface',
            'View2D',
            'Header',
            'View2D Buttons List',
            'Frames',
            'Property Editor',
            'Outliner',
            'Markers',
            'Animation',
            'Dopesheet',
            'Dopesheet Generic',
            '3D View Generic',
            'Grease Pencil',
            'Grease Pencil Stroke Edit Mode',
            'Grease Pencil Stroke Paint Mode',
            'Grease Pencil Stroke Paint (Draw brush)',
            'Grease Pencil Stroke Paint (Erase)',
            'Grease Pencil Stroke Paint (Fill)',
            'Grease Pencil Stroke Sculpt Mode',
            'Face Mask',
            'Weight Paint Vertex Selection',
            'Pose',
            'Object Mode',
            'Paint Curve',
            'Curve',
            'Image Paint',
            'Vertex Paint',
            'Weight Paint',
            'Sculpt',
            'Mesh',
            'Armature',
            'Metaball',
            'Lattice',
            'Particle',
            'Font',
            'Object Non-modal',
            '3D View',
            'Image Editor Tool: Uv, Select',
            'Image Editor Tool: Uv, Select Box',
            'Image Editor Tool: Uv, Select Circle',
            'Image Editor Tool: Uv, Select Lasso',
            'Image Editor Tool: Uv, Cursor',
            '3D View Tool: Pose, Breakdowner',
            '3D View Tool: Pose, Push',
            '3D View Tool: Pose, Relax',
            '3D View Tool: Edit Armature, Roll',
            '3D View Tool: Edit Armature, Bone Size',
            '3D View Tool: Edit Armature, Bone Envelope',
            '3D View Tool: Edit Armature, Extrude',
            '3D View Tool: Edit Armature, Extrude to Cursor',
            '3D View Tool: Edit Mesh, Add Cube',
            '3D View Tool: Edit Mesh, Extrude Region',
            '3D View Tool: Edit Mesh, Extrude Along Normals',
            '3D View Tool: Edit Mesh, Extrude Individual',
            '3D View Tool: Edit Mesh, Extrude to Cursor',
            '3D View Tool: Edit Mesh, Inset Faces',
            '3D View Tool: Edit Mesh, Bevel',
            '3D View Tool: Edit Mesh, Loop Cut',
            '3D View Tool: Edit Mesh, Offset Edge Loop Cut',
            '3D View Tool: Edit Mesh, Knife',
            '3D View Tool: Edit Mesh, Bisect',
            '3D View Tool: Edit Mesh, Poly Build',
            '3D View Tool: Edit Mesh, Spin',
            '3D View Tool: Edit Mesh, Spin Duplicates',
            '3D View Tool: Edit Mesh, Smooth',
            '3D View Tool: Edit Mesh, Randomize',
            '3D View Tool: Edit Mesh, Edge Slide',
            '3D View Tool: Edit Mesh, Vertex Slide',
            '3D View Tool: Edit Mesh, Shrink/Fatten',
            '3D View Tool: Edit Mesh, Push/Pull',
            '3D View Tool: Edit Mesh, Shear',
            '3D View Tool: Edit Mesh, To Sphere',
            '3D View Tool: Edit Mesh, Rip Region',
            '3D View Tool: Edit Mesh, Rip Edge',
            '3D View Tool: Edit Curve, Draw',
            '3D View Tool: Edit Curve, Extrude',
            '3D View Tool: Edit Curve, Extrude Cursor',
            '3D View Tool: Edit Curve, Radius',
            '3D View Tool: Edit Curve, Tilt',
            '3D View Tool: Edit Curve, Randomize',
            '3D View Tool: Sculpt, Box Hide',
            '3D View Tool: Sculpt, Box Mask',
            '3D View Tool: Paint Weight, Gradient',
            '3D View Tool: Paint Weight, Sample Weight',
            '3D View Tool: Paint Weight, Sample Vertex Group',
            '3D View Tool: Paint Gpencil, Cutter',
            '3D View Tool: Paint Gpencil, Line',
            '3D View Tool: Paint Gpencil, Arc',
            '3D View Tool: Paint Gpencil, Curve',
            '3D View Tool: Paint Gpencil, Box',
            '3D View Tool: Paint Gpencil, Circle',
            '3D View Tool: Edit Gpencil, Select',
            '3D View Tool: Edit Gpencil, Select Box',
            '3D View Tool: Edit Gpencil, Select Circle',
            '3D View Tool: Edit Gpencil, Select Lasso',
            '3D View Tool: Edit Gpencil, Extrude',
            '3D View Tool: Edit Gpencil, Radius',
            '3D View Tool: Edit Gpencil, Bend',
            '3D View Tool: Edit Gpencil, Shear',
            '3D View Tool: Edit Gpencil, To Sphere',
            'Gizmos',
            'Backdrop Transform Widget',
            'Backdrop Transform Widget Tweak Modal Map',
            'Backdrop Crop Widget',
            'Backdrop Crop Widget Tweak Modal Map',
            'Sun Beams Widget',
            'Sun Beams Widget Tweak Modal Map',
            'Corner Pin Widget',
            'Corner Pin Widget Tweak Modal Map',
            'Spot Light Widgets',
            'Spot Light Widgets Tweak Modal Map',
            'Area Light Widgets',
            'Area Light Widgets Tweak Modal Map',
            'Target Light Widgets',
            'Target Light Widgets Tweak Modal Map',
            'Force Field Widgets',
            'Force Field Widgets Tweak Modal Map',
            'Camera Widgets',
            'Camera Widgets Tweak Modal Map',
            'Camera View Widgets',
            'Camera View Widgets Tweak Modal Map',
            'Armature Spline Widgets',
            'Armature Spline Widgets Tweak Modal Map',
            'View3D Navigate',
            'View3D Navigate Tweak Modal Map',
            'View3D Gesture Circle',
            'Gesture Box',
            'Gesture Zoom Border',
            'Gesture Straight Line',
            'Standard Modal Map',
            'Animation Channels',
            'Grease Pencil Stroke Weight Mode',
            'Knife Tool Modal Map',
            'Custom Normals Modal Map',
            'Bevel Modal Map',
            'UV Editor',
            'UV Sculpt',
            'Paint Stroke Modal',
            'Mask Editing',
            'Eyedropper Modal Map',
            'Eyedropper ColorBand PointSampling Map',
            'Transform Modal Map',
            'View3D Fly Modal',
            'View3D Walk Modal',
            'View3D Rotate Modal',
            'View3D Move Modal',
            'View3D Zoom Modal',
            'View3D Dolly Modal',
            'Graph Editor Generic',
            'Graph Editor',
            'Image Generic',
            'Image',
            'Node Generic',
            'Node Editor',
            'Info',
            'File Browser',
            'File Browser Main',
            'File Browser Buttons',
            'NLA Generic',
            'NLA Channels',
            'NLA Editor',
            'Text Generic',
            'Text',
            'SequencerCommon',
            'Sequencer',
            'SequencerPreview',
            'Console',
            'Clip',
            'Clip Editor',
            'Clip Graph Editor',
            'Clip Dopesheet Editor',
            'UV Transform Gizmo',
            'UV Transform Gizmo Tweak Modal Map',
            'Toolbar Popup',
            'Generic Tool: Annotate',
            'Generic Tool: Annotate Line',
            'Generic Tool: Annotate Polygon',
            'Generic Tool: Annotate Eraser',
            'Image Editor Tool: Sample',
            'Node Tool: Select',
            'Node Tool: Select Box',
            'Node Tool: Select Lasso',
            'Node Tool: Select Circle',
            'Node Tool: Links Cut',
            '3D View Tool: Cursor',
            '3D View Tool: Select',
            '3D View Tool: Select Box',
            '3D View Tool: Select Circle',
            '3D View Tool: Select Lasso',
            '3D View Tool: Transform',
            '3D View Tool: Move',
            '3D View Tool: Rotate',
            '3D View Tool: Scale',
            '3D View Tool: Measure',
            '3D View Tool: Sculpt Gpencil, Select',
            '3D View Tool: Sculpt Gpencil, Select Box',
            '3D View Tool: Sculpt Gpencil, Select Circle',
            '3D View Tool: Sculpt Gpencil, Select Lasso',
            'Toolbar Popup <temp>',
        }
        keymap_list = {
            'Window': 'EMPTY',
            'Pose': 'EMPTY',
            'Object Mode': 'EMPTY',

            'Screen': 'EMPTY',
            'Frames': 'EMPTY',
            'Property Editor': 'PROPERTIES',
            'Outliner': 'OUTLINER',
            'Image Paint': 'EMPTY',
            'Vertex Paint': 'EMPTY',
            'Weight Paint': 'EMPTY',
            'Sculpt': 'EMPTY',
            'Mesh': 'EMPTY',
            'Armature': 'EMPTY',
            'Object Non-modal': 'EMPTY',
            'Dopesheet': 'DOPESHEET_EDITOR',
            'NLA Editor': 'NLA_EDITOR',
            'NLA Generic': 'NLA_EDITOR',
            'Text': 'TEXT_EDITOR',
            'Info': 'INFO',
            'Graph Editor': 'GRAPH_EDITOR',
            'Timeline': 'TIMELINE',
            '3D View': 'VIEW_3D',
            'Animation': 'EMPTY',

            'Screen Editing': 'EMPTY',
            'User Interface': 'EMPTY',
            'View2D': 'EMPTY',
            'Header': 'EMPTY',
            '3D View Generic': 'VIEW_3D',
            'Grease Pencil': 'EMPTY',
            'Grease Pencil Stroke Edit Mode': 'EMPTY',
            'Face Mask': 'EMPTY',
            'Weight Paint Vertex Selection': 'EMPTY',
            'Paint Curve': 'EMPTY',
            'Curve': 'EMPTY',
            'Metaball': 'EMPTY',
            'Lattice': 'EMPTY',
            'Particle': 'EMPTY',
            'Font': 'EMPTY',
            'UV Editor': 'EMPTY',
            'UV Sculpt': 'EMPTY',
            'Mask Editing': 'EMPTY',
            'Markers': 'EMPTY',
            'Graph Editor Generic': 'GRAPH_EDITOR',
            'Image Generic': 'IMAGE_EDITOR',
            'Image': 'IMAGE_EDITOR',
            'Node Generic': 'NODE_EDITOR',
            'Node Editor': 'NODE_EDITOR',
            'Dopesheet Generic': 'DOPESHEET_EDITOR',
            'NLA Channels': 'NLA_EDITOR',
            'Text Generic': 'TEXT_EDITOR',
            'SequencerCommon': 'SEQUENCE_EDITOR',
            'Sequencer': 'SEQUENCE_EDITOR',
            'SequencerPreview': 'SEQUENCE_EDITOR',
            'Console': 'CONSOLE',
            'Clip': 'CLIP_EDITOR',
            'Clip Editor': 'CLIP_EDITOR',
            'Clip Graph Editor': 'CLIP_EDITOR',
            'Clip Dopesheet Editor': 'CLIP_EDITOR',
        }

    keymap_list = {
        'Window': ('EMPTY', 'WINDOW'),
        'Screen': ('EMPTY', 'WINDOW'),
        'Screen Editing': ('EMPTY', 'WINDOW'),
        'User Interface': ('EMPTY', 'WINDOW'),
        'View2D': ('EMPTY', 'WINDOW'),
        'Header': ('EMPTY', 'WINDOW'),
        'View2D Buttons List': ('EMPTY', 'WINDOW'),
        'Frames': ('EMPTY', 'WINDOW'),
        'Property Editor': ('PROPERTIES', 'WINDOW'),
        'Outliner': ('OUTLINER', 'WINDOW'),
        'Markers': ('EMPTY', 'WINDOW'),
        'Animation': ('EMPTY', 'WINDOW'),
        'Dopesheet': ('DOPESHEET_EDITOR', 'WINDOW'),
        'Dopesheet Generic': ('DOPESHEET_EDITOR', 'WINDOW'),
        '3D View Generic': ('VIEW_3D', 'WINDOW'),
        'Grease Pencil': ('EMPTY', 'WINDOW'),
        'Grease Pencil Stroke Edit Mode': ('EMPTY', 'WINDOW'),
        'Grease Pencil Stroke Paint Mode': ('EMPTY', 'WINDOW'),
        'Grease Pencil Stroke Paint (Draw brush)': ('EMPTY', 'WINDOW'),
        'Grease Pencil Stroke Paint (Erase)': ('EMPTY', 'WINDOW'),
        'Grease Pencil Stroke Paint (Fill)': ('EMPTY', 'WINDOW'),
        'Grease Pencil Stroke Sculpt Mode': ('EMPTY', 'WINDOW'),
        'Face Mask': ('EMPTY', 'WINDOW'),
        'Weight Paint Vertex Selection': ('EMPTY', 'WINDOW'),
        'Pose': ('EMPTY', 'WINDOW'),
        'Object Mode': ('EMPTY', 'WINDOW'),
        'Paint Curve': ('EMPTY', 'WINDOW'),
        'Curve': ('EMPTY', 'WINDOW'),
        'Image Paint': ('EMPTY', 'WINDOW'),
        'Vertex Paint': ('EMPTY', 'WINDOW'),
        'Weight Paint': ('EMPTY', 'WINDOW'),
        'Sculpt': ('EMPTY', 'WINDOW'),
        'Mesh': ('EMPTY', 'WINDOW'),
        'Armature': ('EMPTY', 'WINDOW'),
        'Metaball': ('EMPTY', 'WINDOW'),
        'Lattice': ('EMPTY', 'WINDOW'),
        'Particle': ('EMPTY', 'WINDOW'),
        'Font': ('EMPTY', 'WINDOW'),
        'Object Non-modal': ('EMPTY', 'WINDOW'),
        '3D View': ('VIEW_3D', 'WINDOW'),
        'Image Editor Tool: Uv, Select': ('IMAGE_EDITOR', 'WINDOW'),
        'Image Editor Tool: Uv, Select Box': ('IMAGE_EDITOR', 'WINDOW'),
        'Image Editor Tool: Uv, Select Circle': ('IMAGE_EDITOR', 'WINDOW'),
        'Image Editor Tool: Uv, Select Lasso': ('IMAGE_EDITOR', 'WINDOW'),
        'Image Editor Tool: Uv, Cursor': ('IMAGE_EDITOR', 'WINDOW'),
        '3D View Tool: Pose, Breakdowner': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Pose, Push': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Pose, Relax': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Armature, Roll': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Armature, Bone Size': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Armature, Bone Envelope': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Armature, Extrude': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Armature, Extrude to Cursor': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Add Cube': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Extrude Region': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Extrude Along Normals': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Extrude Individual': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Extrude to Cursor': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Inset Faces': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Bevel': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Loop Cut': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Offset Edge Loop Cut': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Knife': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Bisect': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Poly Build': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Spin': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Spin Duplicates': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Smooth': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Randomize': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Edge Slide': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Vertex Slide': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Shrink/Fatten': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Push/Pull': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Shear': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, To Sphere': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Rip Region': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Mesh, Rip Edge': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Curve, Draw': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Curve, Extrude': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Curve, Extrude Cursor': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Curve, Radius': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Curve, Tilt': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Curve, Randomize': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Sculpt, Box Hide': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Sculpt, Box Mask': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Paint Weight, Gradient': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Paint Weight, Sample Weight': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Paint Weight, Sample Vertex Group': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Paint Gpencil, Cutter': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Paint Gpencil, Line': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Paint Gpencil, Arc': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Paint Gpencil, Curve': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Paint Gpencil, Box': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Paint Gpencil, Circle': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Gpencil, Select': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Gpencil, Select Box': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Gpencil, Select Circle': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Gpencil, Select Lasso': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Gpencil, Extrude': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Gpencil, Radius': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Gpencil, Bend': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Gpencil, Shear': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Edit Gpencil, To Sphere': ('VIEW_3D', 'WINDOW'),
        'Gizmos': ('EMPTY', 'WINDOW'),
        'Backdrop Transform Widget': ('NODE_EDITOR', 'WINDOW'),
        'Backdrop Transform Widget Tweak Modal Map': ('EMPTY', 'WINDOW'),
        'Backdrop Crop Widget': ('NODE_EDITOR', 'WINDOW'),
        'Backdrop Crop Widget Tweak Modal Map': ('EMPTY', 'WINDOW'),
        'Sun Beams Widget': ('NODE_EDITOR', 'WINDOW'),
        'Sun Beams Widget Tweak Modal Map': ('EMPTY', 'WINDOW'),
        'Corner Pin Widget': ('NODE_EDITOR', 'WINDOW'),
        'Corner Pin Widget Tweak Modal Map': ('EMPTY', 'WINDOW'),
        'Spot Light Widgets': ('VIEW_3D', 'WINDOW'),
        'Spot Light Widgets Tweak Modal Map': ('EMPTY', 'WINDOW'),
        'Area Light Widgets': ('VIEW_3D', 'WINDOW'),
        'Area Light Widgets Tweak Modal Map': ('EMPTY', 'WINDOW'),
        'Target Light Widgets': ('VIEW_3D', 'WINDOW'),
        'Target Light Widgets Tweak Modal Map': ('EMPTY', 'WINDOW'),
        'Force Field Widgets': ('VIEW_3D', 'WINDOW'),
        'Force Field Widgets Tweak Modal Map': ('EMPTY', 'WINDOW'),
        'Camera Widgets': ('VIEW_3D', 'WINDOW'),
        'Camera Widgets Tweak Modal Map': ('EMPTY', 'WINDOW'),
        'Camera View Widgets': ('VIEW_3D', 'WINDOW'),
        'Camera View Widgets Tweak Modal Map': ('EMPTY', 'WINDOW'),
        'Armature Spline Widgets': ('VIEW_3D', 'WINDOW'),
        'Armature Spline Widgets Tweak Modal Map': ('EMPTY', 'WINDOW'),
        'View3D Navigate': ('VIEW_3D', 'WINDOW'),
        'View3D Navigate Tweak Modal Map': ('EMPTY', 'WINDOW'),
        'View3D Gesture Circle': ('EMPTY', 'WINDOW'),
        'Gesture Box': ('EMPTY', 'WINDOW'),
        'Gesture Zoom Border': ('EMPTY', 'WINDOW'),
        'Gesture Straight Line': ('EMPTY', 'WINDOW'),
        'Standard Modal Map': ('EMPTY', 'WINDOW'),
        'Animation Channels': ('EMPTY', 'WINDOW'),
        'Grease Pencil Stroke Weight Mode': ('EMPTY', 'WINDOW'),
        'Knife Tool Modal Map': ('EMPTY', 'WINDOW'),
        'Custom Normals Modal Map': ('EMPTY', 'WINDOW'),
        'Bevel Modal Map': ('EMPTY', 'WINDOW'),
        'UV Editor': ('EMPTY', 'WINDOW'),
        'UV Sculpt': ('EMPTY', 'WINDOW'),
        'Paint Stroke Modal': ('EMPTY', 'WINDOW'),
        'Mask Editing': ('EMPTY', 'WINDOW'),
        'Eyedropper Modal Map': ('EMPTY', 'WINDOW'),
        'Eyedropper ColorBand PointSampling Map': ('EMPTY', 'WINDOW'),
        'Transform Modal Map': ('EMPTY', 'WINDOW'),
        'View3D Fly Modal': ('EMPTY', 'WINDOW'),
        'View3D Walk Modal': ('EMPTY', 'WINDOW'),
        'View3D Rotate Modal': ('EMPTY', 'WINDOW'),
        'View3D Move Modal': ('EMPTY', 'WINDOW'),
        'View3D Zoom Modal': ('EMPTY', 'WINDOW'),
        'View3D Dolly Modal': ('EMPTY', 'WINDOW'),
        'Graph Editor Generic': ('GRAPH_EDITOR', 'WINDOW'),
        'Graph Editor': ('GRAPH_EDITOR', 'WINDOW'),
        'Image Generic': ('IMAGE_EDITOR', 'WINDOW'),
        'Image': ('IMAGE_EDITOR', 'WINDOW'),
        'Node Generic': ('NODE_EDITOR', 'WINDOW'),
        'Node Editor': ('NODE_EDITOR', 'WINDOW'),
        'Info': ('INFO', 'WINDOW'),
        'File Browser': ('FILE_BROWSER', 'WINDOW'),
        'File Browser Main': ('FILE_BROWSER', 'WINDOW'),
        'File Browser Buttons': ('FILE_BROWSER', 'WINDOW'),
        'NLA Generic': ('NLA_EDITOR', 'WINDOW'),
        'NLA Channels': ('NLA_EDITOR', 'WINDOW'),
        'NLA Editor': ('NLA_EDITOR', 'WINDOW'),
        'Text Generic': ('TEXT_EDITOR', 'WINDOW'),
        'Text': ('TEXT_EDITOR', 'WINDOW'),
        'SequencerCommon': ('SEQUENCE_EDITOR', 'WINDOW'),
        'Sequencer': ('SEQUENCE_EDITOR', 'WINDOW'),
        'SequencerPreview': ('SEQUENCE_EDITOR', 'WINDOW'),
        'Console': ('CONSOLE', 'WINDOW'),
        'Clip': ('CLIP_EDITOR', 'WINDOW'),
        'Clip Editor': ('CLIP_EDITOR', 'WINDOW'),
        'Clip Graph Editor': ('CLIP_EDITOR', 'WINDOW'),
        'Clip Dopesheet Editor': ('CLIP_EDITOR', 'WINDOW'),
        'UV Transform Gizmo': ('IMAGE_EDITOR', 'WINDOW'),
        'UV Transform Gizmo Tweak Modal Map': ('EMPTY', 'WINDOW'),
        'Toolbar Popup': ('EMPTY', 'TEMPORARY'),
        'Generic Tool: Annotate': ('EMPTY', 'WINDOW'),
        'Generic Tool: Annotate Line': ('EMPTY', 'WINDOW'),
        'Generic Tool: Annotate Polygon': ('EMPTY', 'WINDOW'),
        'Generic Tool: Annotate Eraser': ('EMPTY', 'WINDOW'),
        'Image Editor Tool: Sample': ('IMAGE_EDITOR', 'WINDOW'),
        'Node Tool: Select': ('NODE_EDITOR', 'WINDOW'),
        'Node Tool: Select Box': ('NODE_EDITOR', 'WINDOW'),
        'Node Tool: Select Lasso': ('NODE_EDITOR', 'WINDOW'),
        'Node Tool: Select Circle': ('NODE_EDITOR', 'WINDOW'),
        'Node Tool: Links Cut': ('NODE_EDITOR', 'WINDOW'),
        '3D View Tool: Cursor': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Select': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Select Box': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Select Circle': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Select Lasso': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Transform': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Move': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Rotate': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Scale': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Measure': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Sculpt Gpencil, Select': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Sculpt Gpencil, Select Box': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Sculpt Gpencil, Select Circle': ('VIEW_3D', 'WINDOW'),
        '3D View Tool: Sculpt Gpencil, Select Lasso': ('VIEW_3D', 'WINDOW'),
    }

    class reload_toggles:
        functions = list()


class Constraint:
    """Global(constraint) properties that are called from the real Constraint"""

    no_target = (
        # Constraint types without a target,
        # so never generate objects for them
        'CAMERA_SOLVER',
        'FOLLOW_TRACK', 'OBJECT_SOLVER',  # 'target' is 'camera' object
        'LIMIT_LOCATION', 'LIMIT_ROTATION', 'LIMIT_SCALE',
        'MAINTAIN_VOLUME', 'TRANSFORM_CACHE',
    )

    curves = ('CLAMP_TO', 'SPLINE_IK', 'FOLLOW_PATH')

    class types:
        "Separate constraint types by category"

        motion_tracking = [
            ('CAMERA_SOLVER', "Camera Solver", "", 'CON_CAMERASOLVER', 1),
            ('FOLLOW_TRACK', "Follow Track", "", 'CON_FOLLOWTRACK', 2),
            ('OBJECT_SOLVER', "Object Solver", "", 'CON_OBJECTSOLVER', 3),
        ]
        transform = [
            ('COPY_LOCATION', "Copy Location",
                "Copy the location of a target (with an optional offset), "
                "so that they move together",
                'CON_LOCLIKE', 4),
            ('COPY_ROTATION', "Copy Rotation",
                "Copy the rotation of a target (with an optional offset), "
                "so that they rotate together",
                'CON_ROTLIKE', 5),
            ('COPY_SCALE', "Copy Scale",
                "Copy the scale factors of a target (with an optional offset), "
                "so that they are scaled by the same amount",
                'CON_SIZELIKE', 6),
            ('COPY_TRANSFORMS', "Copy Transforms",
                "Copy all the transformations of a target, "
                "so that they move together",
                'CON_TRANSLIKE', 7),
            ('LIMIT_DISTANCE', "Limit Distance",
                "Restrict movements to within a certain distance of a target "
                "(at the time of constraint evaluation only)",
                'CON_DISTLIMIT', 8),
            ('LIMIT_LOCATION', "Limit Location",
                "Restrict movement along each axis within given ranges",
                'CON_LOCLIMIT', 9),
            ('LIMIT_ROTATION', "Limit Rotation",
                "Restrict rotation along each axis within given ranges",
                'CON_ROTLIMIT', 10),
            ('LIMIT_SCALE', "Limit Scale",
                "Restrict scaling along each axis with given ranges",
                'CON_SIZELIMIT', 11),
            ('MAINTAIN_VOLUME', "Maintain Volume",
                "Compensate for scaling one axis by applying suitable "
                "scaling to the other two axes",
                'CON_SAMEVOL', 12),
            ('TRANSFORM', "Transformation",
                "Use one transform property from target to control "
                "another (or same) property on owner",
                'CON_TRANSFORM', 13),
            ('TRANSFORM_CACHE', "Transform Cache",
                "Look up the transformation matrix from an external file",
                'CON_TRANSFORM_CACHE', 14),
        ]
        tracking = [
            ('CLAMP_TO', "Clamp To",
                "Restrict movements to lie along a curve "
                "by remapping location along curve's longest axis",
                'CON_CLAMPTO', 15),
            ('DAMPED_TRACK', "Damped Track",
                "Point towards a target by performing the smallest rotation necessary",
                'CON_TRACKTO', 16),
            ('IK', "Inverse Kinematics",
                "Control a chain of bones by specifying the endpoint target (Bones only",
                'CON_KINEMATIC', 17),
            ('LOCKED_TRACK', "Locked Track",
                "Rotate around the specified ('locked') axis to point towards a target",
                'CON_LOCKTRACK', 18),
            ('SPLINE_IK', "Spline IK",
                "Align chain of bones along a curve (Bones only)",
                'CON_SPLINEIK', 19),
            ('STRETCH_TO', "Stretch To",
                "Stretch along Y-Axis to point towards a target",
                'CON_STRETCHTO', 20),
            ('TRACK_TO', "Track To",
                "Legacy tracking constraint prone to twisting artifacts",
                'CON_TRACKTO', 21),
        ]
        relationship = [
            ('ACTION', "Action",
                "Use transform property of target to "
                "look up pose for owner from an Action",
                'CON_ACTION', 22),
            *[('ARMATURE', "Armature",
                "Apply weight-blended transformation from "
                "multiple bones like the Armature modifier",
                'CON_ARMATURE', 23) for a in [is28] if a],  # Added in 2.80
            ('CHILD_OF', "Child Of",
                "Make target the 'detachable' parent of owner",
                'CON_CHILDOF', 24),
            ('FLOOR', "Floor",
                "Use position (and optionally rotation) of target to define a 'wall'"
                " or 'floor' that the owner can not cross",
                'CON_FLOOR', 25),
            ('FOLLOW_PATH', "Follow Path",
                "Use to animate an object/bone following a path",
                'CON_FOLLOWPATH', 26),
            ('PIVOT', "Pivot",
                "Change pivot point for transforms (buggy)",
                'CON_PIVOT', 27),
            *[('RIGID_BODY_JOINT', "Rigid Body Joint",
                "Use to define a Rigid Body Constraint (for Game Engine use only)",
                'CONSTRAINT_DATA', 28) for a in [is27] if a],  # removed in 2.80
            # ('SCRIPT', "Script",
            #  "Custom constraint(s) written in Python (Not yet implemented)",
            #  'CONSTRAINT_DATA', 29),  # removed around 2.6
            ('SHRINKWRAP', "Shrinkwrap", "Restrict movements to surface of target mesh",
                'CON_SHRINKWRAP', 30),
        ]

        # Custom
        child_of = [
            ('child_of_locrot', "Child of Loc + Rot",
                "Map to Location + Rotation separately", 'CON_TRANSLIKE', 52),
            ('child_of_location', "    Location",
                "Map only location", 'CON_LOCLIKE', 53),
            ('child_of_rotation', "    Rotation",
                "Map only rotation", 'CON_ROTLIKE', 54),
            ('child_of_scale', "    Scale", "Map only Scale", 'CON_SIZELIKE', 55),
        ]
        other = [
            ('ik_transforms', "IK Transforms",
                "IK with Lock Transforms", 'CON_KINEMATIC', 56),
            ('pivot', "Pivot",
                "Pivot around target point, like 3D Cursor (from rigify)", 'PIVOT_CURSOR', 57),
            ('track_rotation', "Damped Rotation",
                "Damped Track with Lock Rotation", 'CON_LOCKTRACK', 58),
            ('stretch_rotation', "Stretched Rotation",
                "Stretch To with Lock Rotation", 'CON_STRETCHTO', 59),
            # ('offset_transforms', "Offset Transforms",
            # "Transform constraint to \"add\" to the current transforms",
            # 'MOD_PARTICLE_INSTANCE', 59),
            # ('', "", "Other constraints"),
            # ('look_at', "(N/A) Look At",
                # "A series of offsetters and a Damped Track, "
                    # "to a single target to focus on"
                    # ".\n""Inspired by Meklab's SFM example / tutorial"
                    # ".\n""https://keythedrawist.tumblr.com/post/180848042768/some-constraint-tutorials",
                # 'CON_TRACKTO', 60),
        ]

    constraint_type_items = [
        ('', "Motion Tracking", ""), *types.motion_tracking,
        ('', "Transform", ""), *types.transform,
        ('', "Tracking", ""), *types.tracking,
        ('', "Relationship", ""), *types.relationship,
        # ('', "None", "Don't add a constraint, just create copy"),
        # Custom constraints
        ('', "Custom", ""),
        *types.child_of,
        *types.other,
    ]
    if is27: constraint_type_items = [
        (a, b, c[0]) for a, b, *c in constraint_type_items]

    # Store the ID of constraint types and custom types
    constraint_types = {x[0]: (x[1], x[2]) for x in (
        *types.motion_tracking,
        *types.transform,
        *types.tracking,
        *types.relationship,
    )}
    custom_types = {x[0]: (x[1], x[2]) for x in (
        *types.child_of,
        *types.other,
    )}


    ###############################################################################
    # region: Functions

    def get_track_axis(owner, axis):
        """
        Re-orient axis for objects\\
        Bones use the axis as-is but objects have a different Y-Z axis
        """

        if Is.object(owner) and axis.endswith('Y'):
            axis = axis[:-1] + 'Z'

        return axis

    def get_dict(src, prop):
        """
        Check if a custom prop is in src.\\
        If not, create it as a dict() and return it
        """

        if src.get(prop) is None:
            src[prop] = dict()

        return src[prop]

    def add_original_relation(context, src, original, use_local_transforms=True):
        """
        Remember the initial pose for new bones/objects\\
        Also remember what the original bone/object was\\
        Note: Only stored in the duplicate, nothing added to the original
        """
        if not hasattr(src, 'base_src'):
            return

        if src.rotation_mode in ('QUATERNION', 'AXIS_ANGLE'):
            euler = 'XYZ'
        else:
            euler = src.rotation_mode

        base = src.base_src
        base.is_duplicate = True
        if use_local_transforms:
            # Bone's default pose is the same as orignial's
            src.base_transforms.matrix_set(Get.matrix(src, basis=True), euler)
        base.target = original.id_data.name
        if Is.posebone(original):
            base.subtarget = original.name
        if original.id_data == src.id_data:
            base.target_self = True

        original.base_src.has_duplicates = True
        copies = original.base_src.copies
        src_hash = str(hash(src))
        if src_hash not in copies:
            copy = copies.add()
            copy.name = src_hash
            copy.label = src.id_data.name.split('||', 1)[0]
        else:
            copy = copies[src_hash]

        copy.target = src.id_data.name
        if Is.posebone(src):
            copy.subtarget = src.name

        return

        # if hasattr(src.id_data, 'constraints_extra'):
            # con_list = src.id_data.constraints_extra
            # if src.name in con_list:
                # con_entry = con_list[src.name]
            # else:
                # con_entry = con_list.add()
                # con_entry.name = src.name

            # con_self = con_entry.self
            # # con_self.bone = Is.posebone(src)
            # con_self['base_matrix'] = Get.matrix(src)

            # con_original = con_entry.original
            # con_original.target = original.id_data.name
            # if Is.posebone(original):
                # con_original.subtarget = original.name

            # # Add copy to the original
            # con_list = original.id_data.constraints_extra
            # if original.name in con_list:
                # con_entry = con_list[original.name]
            # else:
                # con_entry = con_list.add()
                # con_entry.name = original.name

            # con_copies = con_entry.copies
            # cc_entry = con_copies.add()
            # cc_entry.name = utils.string(
            # src.id_data.name, '-', src.name if Is.posebone(src) else '')

            # cc_entry.target = src.id_data.name
            # if Is.posebone(src):
                # cc_entry.subtarget = src.name
        # else:
            # con_list = Constraint.get_dict(src.id_data, 'constraints_extra')
            # con_entry = Constraint.get_dict(con_list, src.name)

            # con_entry['self'] = dict(
                # # bone=Is.posebone(src),
                # base_matrix=Get.matrix(src),
            # )
            # con_entry['original'] = dict(
                # target=original.id_data.name,
                # subtarget=original.name
                # if Is.posebone(original) else '',
            # )

            # # Add copy to the original
            # con_list = Constraint.get_dict(original.id_data, 'constraints_extra')
            # con_entry = Constraint.get_dict(con_list, original.name)

            # con_copies = Constraint.get_dict(con_entry, 'copies')
            # con_copies[utils.string(
            # src.id_data.name, '-', src.name if Is.posebone(src) else ''
            # )] = dict(
                # target=src.id_data.name,
                # subtarget=src.name
                # if Is.posebone(src) else '',
            # )

    # def add_constraint_relation(context, constraint, owner, target):
        # """
        # Add custom property to owner's object, to mark constraint information
        # """

        # # for src in (owner, target):
            # # if not src:
            #     # continue

            # # if not hasattr(src, 'base_src'):
            #     # continue

            # # continue

            # # owner_id = owner.id_data.name
            # # owner_bone = owner.name if Is.posebone(owner) else ''
            # # target_id = target.id_data.name if target else ''
            # # target_bone = target.name if Is.posebone(target) else ''

            # # if hasattr(src.id_data, 'constraints_extra'):
            #     # con_list = src.id_data.constraints_extra
            #     # if src.name in con_list:
            #         # con_entry = con_list[src.name]
            #     # else:
            #         # con_entry = con_list.add()
            #         # con_entry.name = src.name

            #     # # Constraints entries
            #     # con_constraints = con_entry.constraints
            #     # con_item = con_constraints.add()
            #     # con_item.name = constraint.name

            #     # con_item.owner = owner_id
            #     # con_item.subowner = owner_bone
            #     # con_item.target = target_id
            #     # con_item.subtarget = target_bone

            #     # con_item.show_expanded = src is target
            # # else:
            #     # con_list = Constraint.get_dict(src.id_data, 'constraints_extra')
            #     # con_entry = Constraint.get_dict(con_list, src.name)

            #     # # Constraints entries
            #     # con_constraints = Constraint.get_dict(con_entry, 'constraints')

            #     # con_constraints[constraint.name] = dict(
            #         # owner=owner_id,
            #         # subowner=owner_bone,
            #         # target=target_id,
            #         # subtarget=target_bone,
            #         # show_expanded=src is target,
            #     # )

    # endregion
    ###############################################################################


from bpy.props import (BoolProperty, FloatProperty, FloatVectorProperty, EnumProperty, StringProperty)
from mathutils import Matrix, Vector

class Constraint(Constraint):
    class new():
        """
        When called with arg(start=True), this will run like an operator\\
        The operation will generate the constraint type specified\\
        As well as run the related tasks\\
        Otherwise, must call the functions manually
        """
        def __init__(self, context, **kwargs):
            # Defaults
            self.show_expanded = True

            self.type = ''  # 'CAMERA_SOLVER'
            self.target = 'TO_ACTIVE'  # 'NONE'
            self.new_armature = True
            self.hash_name = False

            self.target_space = 'WORLD'
            self.owner_space = 'WORLD'
            self.use_local_location = True
            self.track_axis = 'TRACK_Y'
            self.track_offset = 0
            self.space_object = 'WORLD'

            self.at_tail = False
            self.add_drivers = False
            self.add_relation = False
            self.bake_animation = False
            self.bake_mode = 'range'
            self.display_bake = True
            # self.push_to_nla = False  # Only push if generated new objects
            self.copy_chain = False
            # self.copy_chain_list = dict()
            self.head_tail = 0.0
            self.armature_display_type = 'RANDOM'
            self.map_from = 'LOCATION'
            self.map_to = 'LOCATION'
            self.rotation_x = 0.0
            self.rotation_y = 0.0
            self.rotation_z = 0.0
            self.use_mesh = False
            self.do_physics = False

            self.is_custom = bool(self.type in Constraint.custom_types)
            self.selected = None

            self.relations = list()
            # self.relations = dict()
            self.start = False

            # Overwrite defaults with manual values
            for arg in kwargs:
                setattr(self, arg, kwargs[arg])

            if self.selected is None:
                self.selected = list()
                if context.mode == 'POSE':
                    self.selected.extend(context.selected_pose_bones)
                    if len(self.selected) <= 1 and (self.type not in Constraint.no_target):
                        # When one bone is selected, add the selected meshes/empties
                        for src in context.selected_objects:
                            if not Is.armature(src):
                                self.selected.append(src)
                else:
                    for src in Get.selected(context):
                        if Is.armature(src) and (self.type not in Constraint.no_target):
                            bones = list()
                            for bone in src.pose.bones:
                                if bone.bone.select and Is.in_visible_armature_layers(bone, src.data):
                                    bones.append(bone)
                            if bones:
                                self.selected.extend(bones)
                            else:
                                self.selected.append(src)
                        else:
                            self.selected.append(src)

            if self.start:
                self.init_objects_relations(context)
                self.add_constraints(context, self.relations)

        #####################################
        # Process

        def init_objects_relations(self, context, active=None, new_objects=None):
            """
            Create new bones/objects (if applicable)\\
            Then mark the owners + targets for later use in constraint creation\\
            new_objects is a dictionary containing
                items from selected and their target:
                new_objects[src] = new_control
            """

            if active is None:
                active = Get.active(context)

            if (self.target == 'TO_ACTIVE' and len(self.selected) == 1) or \
                (self.type in Constraint.curves and self.target not in (
                    'TO_ARMATURE', 'TO_OBJECT', 'NONE')
                ):
                if Is.posebone(self.selected[0]):
                    self.target = 'TO_ARMATURE'
                elif Is.object(self.selected[0]):
                    self.target = 'TO_OBJECT'
                else:
                    self.target = 'NONE'
                    # None SHOULD be the default

            # By default, use the bone's rest pose in edit, then update pose in pose mode
            self.use_local_transforms = True

            if (self.type == 'ARMATURE'):
                # Armature modifier starts offsetting from the rest pose
                self.use_local_transforms = False
            elif len({self.target_space, self.owner_space, 'LOCAL_WITH_PARENT', 'LOCAL'}) == 2:
                if self.type in ('COPY_LOCATION', 'COPY_ROTATION', 'COPY_SCALE', 'COPY_TRANSFORMS'):
                    # Use the original's final pose as the copy's default
                    self.use_local_transforms = False

            if (self.type == 'SHRINKWRAP'):
                if self.target == 'NONE':
                    pass
                elif self.target in ('TO_ARMATURE', 'TO_OBJECT'):
                    self.target = 'TO_OBJECT'
                else:
                    for src in self.selected:
                        if Is.mesh(src):
                            break
                    else:
                        self.target = 'TO_OBJECT'
                self.use_mesh = True

            if self.type in Constraint.no_target:
                new_objects = dict()
            elif new_objects is None:
                new_objects = self.generate_new_objects(context,
                    selected=self.selected)

            if self.bake_animation:
                # if no new bones/objects were created, then don't bake anything
                self.bake_animation = bool(new_objects)

            for src in self.selected:
                if self.target == 'NONE':
                    owner = src
                    target = None
                elif self.target == 'TO_ACTIVE':
                    if src == active: continue
                    owner = src
                    target = active
                elif self.target == 'FROM_ACTIVE':
                    if src == active: continue
                    owner = active
                    target = src
                elif self.target in ('TO_ARMATURE', 'TO_OBJECT'):
                    owner = src
                    target = new_objects.get(src)
                    if self.add_relation and target:
                        Constraint.add_original_relation(context, target, owner, self.use_local_transforms)
                elif self.target in ('FROM_ARMATURE', 'FROM_OBJECT'):
                    owner = new_objects.get(src)
                    target = src
                    if self.add_relation and owner:
                        Constraint.add_original_relation(context, owner, target, self.use_local_transforms)
                else:
                    assert None, self.target

                "This is used externally by the operator for the next step"
                self.relations.append((owner, target))

        def add_constraints(self, context, pairs):
            """
            create constraints for the items of (owner, target) in pairs\\
            Owner is the item holding the constraint\\
            Target is the target of the constraint
            """
            assert self.type, (
                self.type, "not valid type for inserting constraints")

            # self.find_prop_desc(kwargs, self.type)

            if self.type == 'SPLINE_IK':
                self.spline_ik(context, pairs)
            else:
                for (owner, target) in pairs:
                    self.add_constraint(context, owner, target)

        ########################################
        # Functions

        def add_constraint(self, context, owner, target):
            # self.find_prop_desc(kwargs, type)
            type = self.type

            limit = None
            limits = ('LIMIT_LOCATION', 'LIMIT_ROTATION', 'LIMIT_SCALE')
            if type not in limits:
                for limit_con in reversed(owner.constraints):
                    if limit_con.type in limits:
                        limit = limit_con
                    else:
                        break
            if limit:
                for (index, limit_con) in enumerate(owner.constraints):
                    if limit_con == limit:
                        iters = index
                        break
            else:
                index = iters = None

            def iterate(count):
                nonlocal index, iters
                if index is None:
                    index = iter(-1 for x in range(count))
                else:
                    index = iter(iters + x for x in range(count))

            if not Is.posebone(owner) and type in {'ik_transforms', 'SPLINE_IK', 'IK'}:
                return

            # Custom constraint setups
            if type == 'ik_transforms':
                iterate(3)

                con = self.new_constraint(owner, 'IK', next(index))
                con.use_tail = self.at_tail
                self.set_constraint(context, con, owner, target)
                if not self.at_tail:
                    con.chain_count -= 1  # IK setup is different, requiring 1 less length

                # mute if bone doesn't have parent, and it's not at the bone's tail
                con.mute = not bool((self.at_tail or owner.parent))

                con = self.new_constraint(owner, 'COPY_ROTATION', next(index))
                self.set_constraint(context, con, owner, target)
                con = self.new_constraint(owner, 'COPY_SCALE', next(index))
                self.set_constraint(context, con, owner, target)
            elif type == 'pivot':
                iterate(2)
                con = self.new_constraint(owner, 'COPY_LOCATION', next(index))
                self.set_constraint(context, con, owner, target)
                con.invert_x = con.invert_y = con.invert_z = True
                con.use_offset = True
                con.target_space = con.owner_space = 'LOCAL'
                loc = target.location.copy()
                target.location = (0, 0, 0)
                con = self.new_constraint(owner, 'CHILD_OF', next(index))
                self.set_constraint(context, con, owner, target)
                target.location = loc
            elif type == 'track_rotation':
                iterate(4)
                if Is.posebone(target) and not Is.connected(owner):
                    con = self.new_constraint(owner, 'COPY_LOCATION', next(index))
                    self.set_constraint(context, con, owner, target)
                    con.head_tail = 0
                con = self.new_constraint(owner, 'COPY_SCALE', next(index))
                self.set_constraint(context, con, owner, target)
                # mimic stretch constraints
                con.use_x = False
                con.use_z = False

                con = self.new_constraint(owner, 'COPY_ROTATION', next(index))
                self.set_constraint(context, con, owner, target)
                con = self.new_constraint(owner, 'DAMPED_TRACK', next(index))
                self.set_constraint(context, con, owner, target)
            elif type == 'stretch_rotation':
                iterate(4)
                if Is.posebone(target) and not Is.connected(owner):
                    con = self.new_constraint(owner, 'COPY_LOCATION', next(index))
                    self.set_constraint(context, con, owner, target)
                    con.head_tail = 0
                con = self.new_constraint(owner, 'COPY_SCALE', next(index))
                self.set_constraint(context, con, owner, target)
                # mimic stretch constraints
                con.use_x = False
                con.use_z = False

                con = self.new_constraint(owner, 'COPY_ROTATION', next(index))
                self.set_constraint(context, con, owner, target)
                con = self.new_constraint(owner, 'STRETCH_TO', next(index))
                self.set_constraint(context, con, owner, target)
            elif type.startswith('child_of_'):
                for t in ('location', 'rotation', 'scale'):
                    if type == f'child_of_{t}':
                        iterate(1)
                        con = self.new_constraint(owner, 'CHILD_OF', next(index))
                        self.set_child_of(
                            con, t == 'location', t == 'rotation', t == 'scale')
                        self.set_constraint(context, con, owner, target)
                        break
                else:
                    if type == 'child_of_locrot':
                        iterate(2)
                        con = self.new_constraint(owner, 'CHILD_OF', next(index))
                        self.set_child_of(con, True, False, False)
                        self.set_constraint(context, con, owner, target)
                        con = self.new_constraint(owner, 'CHILD_OF', next(index))
                        self.set_child_of(con, False, True, True)
                        self.set_constraint(context, con, owner, target)
                    elif type == 'child_of_locscale':
                        iterate(1)
                        con = self.new_constraint(owner, 'CHILD_OF', next(index))
                        self.set_child_of(con, True, False, True)
                        self.set_constraint(context, con, owner, target)
                    elif type == 'child_of_rotscale':
                        iterate(1)
                        con = self.new_constraint(owner, 'CHILD_OF', next(index))
                        self.set_child_of(con, False, True, True)
                        self.set_constraint(context, con, owner, target)
            elif type == 'offset_transforms':
                # for space in ('LOCATION', 'ROTATION', 'SCALE'):
                    iterate(1)
                    con = self.new_constraint(owner, 'TRANSFORM', next(index))
                    self.set_constraint(context, con, owner, target)
                    # con.map_from = con.map_to = space
                    con.map_from = self.map_from
                    con.map_to = self.map_to
                    con.target_space = con.owner_space = 'LOCAL'

            # Default constraints
            elif type == 'SPLINE_IK':
                """Queue Spline, don't add constraints yet"""
                iterate(1)
                # self.ik_splines[owner] = target
                # con = self.new_constraint(owner, type, next(index))
                con = None
            elif type == 'ARMATURE':
                for con in owner.constraints:
                    if con.type == 'ARMATURE':
                        # Reuse existing constraint
                        break
                else:
                    # Create a new constraint
                    iterate(1)
                    con = self.new_constraint(owner, type, next(index))
                self.set_constraint(context, con, owner, target)
            else:
                iterate(1)
                con = self.new_constraint(owner, type, next(index))
                self.set_constraint(context, con, owner, target)

            if self.do_physics:
                self.generate_physic_objects(context, owner, target)

            return con

        def add_constraint_driver(self, con, target, prop=None, desc=None):
            """
            Add custom prop to target and use it as a driver on the constraint
            """

            info = Constraint.custom_types.get(self.type)
            is_custom = bool(info)
            if info is None: info = Constraint.constraint_types[self.type]
            (prop_, desc_) = info
            # if is_custom: prop_ += ' - ' + Constraint.constraint_types[con.type][0]

            if desc is None: desc = desc_
            if prop is None:
                # Use the Constraint Label and a unique number (for this run)
                prop = prop_ + '||' + str(hash(target)) + '||' + str(hash(self))

            label = prop_  # The display text for Constraints Custom Prop UI
            # prop_obj = prop + '||' + str(hash(target))  # Unique id per control

            # if prop in target:
                # name = prop
                # num = 1
                # while prop in target:
                #     # prop = f'{name}.{num:03}'  # .001 incremental padding
                #     prop = f'{name}{num}'  # custom props don't use padding
                #     num += 1
                # # Add incremental number until not duplicate

            # Add an object level reference to the custom prop
            for src in (con, target):
                if (src == target) and (target.id_data == src.id_data):
                    continue
                src = src.id_data

                if hasattr(src, 'constraints_props'):
                    entry_props = src.id_data.constraints_props.props

                    entry = entry_props.get(prop)
                    # print('entry:', getattr(entry, 'name', "No Entry"), 'prop_obj:', prop_obj)
                    # if entry:
                    #     print('target:', target.name)
                    #     print('Subtar:', entry.subtarget)
                    #     print('-------------')
                    #     if target.name == entry.subtarget:
                    #         entry = None
                    # # while entry:
                    # #     if entry.target == target.id_data.name:
                    # #         if Is.posebone(target):
                    # #             if entry.subtarget == target.name:
                    # #                 break
                    # #         else:
                    # #             break
                    # #     entry = None
                    if entry is None:
                        # Don't create duplicates
                        entry = entry_props.add()
                        entry.name = prop
                else:
                    con_props = Constraint.get_dict(src, 'constraints_props')
                    con_props['show_expanded'] = con_props.get('show_expanded', 1)
                    entry_props = Constraint.get_dict(con_props, 'props')

                    entry = entry_props.get(prop)
                    # print('entry:', getattr(entry, 'name', "No Entry"), 'prop:', prop)
                    # if entry:
                    #     print('target:', target.name)
                    #     print('Subtar:', entry.subtarget)
                    #     print('-------------')
                    #     if target.name == entry.subtarget:
                    #         entry = None
                    # # while entry:
                    # #     if entry['target'] == target.id_data.name:
                    # #         if Is.posebone(target):
                    # #             if entry['subtarget'] == target.name:
                    # #                 break
                    # #         else:
                    # #             break
                    # #     entry = None
                    if entry is None:
                        entry_props[prop_obj] = dict(prop='', target='', subtarget='')
                        entry = entry_props[prop]

                entry['target'] = target.id_data.name
                if Is.posebone(target):
                    entry['subtarget'] = target.name
                entry['prop'] = label

            if not target.get(prop):
                # Don't reset it if already used
                target[prop] = 1.0
                target['_RNA_UI'] = target.get('_RNA_UI', dict())
                target['_RNA_UI'][prop] = dict(
                    min=0.0, max=1.0,
                    soft_min=0.0, soft_max=1.0,
                    description=desc,
                    default=0.0,
                )

            driver_ = con.driver_add('influence')
            driver = driver_.driver
            # driver.expression = 'var'
            driver.type = 'AVERAGE'
            variable = driver.variables.new().targets[0]
            variable.id_type = 'OBJECT'
            variable.id = target.id_data

            variable.data_path = target.path_from_id(f'["{prop}"]')

            driver.expression += " "
            driver.expression = driver.expression[:-1]

            return driver_

        def find_prop_desc(self, kwargs, type):
            """
            If using add_drivers, search through constraint list one time\\
            to find the property and description for the constraint type.\\
            (if 'prop' or 'desc' not already defined)
            """

            if kwargs.get('add_drivers', getattr(self, 'add_drivers', False)):
                prop = kwargs.get('prop')
                desc = kwargs.get('desc')

                if None in (prop, desc):
                    for cti in Constraint.constraint_type_items:
                        if cti[0] == type:
                            if prop is None:
                                # prop = f'{cti[1]}||{hash(self)}'
                                prop = cti[1]
                            if desc is None:
                                desc = cti[2]
                            break
                    else:
                        assert None, ("Constraint has type but it's not labeled")
                    kwargs['prop'] = prop
                    kwargs['desc'] = desc

            return kwargs

        def generate_new_objects(self, context, selected=list()):
            """
            Generate new objects/armatures (in relation to each item in selected)\\
            Then link the new items to the old;  (objects[src] = new_src)\\
            Finally return this relation as a dictionary
            """

            first_active = Get.active(context)
            active = None

            new_arm = self.target in ('TO_ARMATURE', 'FROM_ARMATURE')
            new_obj = self.target in ('TO_OBJECT', 'FROM_OBJECT')
            new_bones = new_arm and not self.new_armature
            new_objects = dict()

            if not (new_arm or new_obj or new_bones):
                return new_objects
            # else:
                # # New objects generated, so allow push to nla
                # self.push_to_nla = True


            # Don't use connect on bones for these constraints
            is_not_tracker = self.type not in (
                *Constraint.curves,
                'DAMPED_TRACK', 'IK', 'LOCKED_TRACK',
                'STRETCH_TO', 'TRACK_TO',
                # Custom
                'ik_transforms', 'track_rotation', 'stretch_rotation', 'look_at',
            )

            if context.scene.use_armature_target:
                armature = context.scene.armature_target
                if armature and (context.scene not in armature.users_scene):
                    armature = context.scene.armature_target = None
            else:
                armature = None

            scn_armature = False
            if (new_arm and armature):
                scn_armature = str(hash(armature))
                self.new_armature = False
                new_bones = False
                Set.mode(context, armature, 'EDIT')
            elif (new_arm and not new_bones):
                armature = New.armature(context,
                    display_type=self.armature_display_type)
                armature.name = utils.string(
                    self.type.replace('_', ' ').title(), '||', hash(armature)
                )

                Set.object_display_type(armature, 'WIRE')
                Set.xray(armature, True)

                Set.mode(context, armature, 'EDIT')
            else:
                armature = None

            for src in selected:
                if new_obj:
                    name = utils.string(
                        self.type.replace('_', ' ').title(),
                        '||', src.id_data.name.split('||', 1)[0],
                        '||', src.name,
                        # '||', hash(src),
                    )

                    if self.use_mesh:
                        obj = New.mesh(context, name=name)
                    else:
                        obj = New.object(context, name=name)

                        if Is.posebone(src):
                            width = src.bone.bbone_x
                            depth = src.bone.bbone_z
                            height = src.length
                            size = sum((width, depth, height))
                        elif Is.empty(src):
                            size = 1.0
                        else:
                            size = sum(src.dimensions)
                        Set.empty_size(obj, size)

                    self.init_nla(src.id_data, obj)
                    new_objects[src] = obj
                    self.set_object(context, obj, src)
                    if (context.mode == 'OBJECT'):
                        if (active is None or first_active == src):
                            Set.active(context, obj)
                            Set.select(obj, True)
                            Set.select(src, False)
                            active = obj
                elif new_arm or new_bones:
                    if new_bones:
                        if not Is.posebone(src):
                            continue
                        armature = src.id_data
                        name = utils.string(
                            # self.type.replace('_', ' ').title(), '||',
                            src.name,
                        )
                        if armature.mode != 'EDIT':
                            Set.mode(context, armature, 'EDIT')
                    else:
                        name = utils.string(
                            # self.type.replace('_', ' ').title(), '||',
                            src.name,
                        )

                    if scn_armature:
                        # Using preset object which may not be the "same" armature
                        name = utils.string(
                            src.id_data.name.split('||', 1)[0],
                            '||', self.type.replace('_', ' ').title(),
                            # '||', scn_armature,
                            '||', src.name,
                            # '||', hash(src),
                        )
                    elif not(self.new_armature):
                        # Since bone is added in same armature, it needs unique name
                        if name.lower().endswith(('.l', '.r')):
                            if (self.type == 'ik_transforms'):
                                name = name[:-2] + '.IK' + name[-2:]
                            else:
                                name = name[:-2] + '.' + self.type + name[-2:]
                        else:
                            name = (name + '.' + self.type)

                    self.init_nla(src.id_data, armature)
                    bone = New.bone(context, armature, name=name)
                    self.set_edit_bone(context, bone, src)
                    new_objects[src] = (armature, bone.name)

                    if self.create_widgets and (src.id_data.data != bone.id_data):
                        # Add display bone
                        vis = New.bone(context, armature, name='VIS-' + bone.name)
                        self.set_edit_bone(context, vis, src)

            if new_arm or new_bones:
                armatures = set()  # objects to set back to pose later

                for src in new_objects:
                    (armature, name) = new_objects[src]
                    armatures.add(armature)
                    if self.copy_chain:
                        # self.copy_chain_list[src] =
                        if not src.parent:
                            continue

                        bone = armature.data.edit_bones[name]
                        # mat = Get.matrix(bone)
                        for psrc in src.parent_recursive:
                            if psrc not in new_objects:
                                continue
                            (armature, pname) = new_objects[psrc]
                            pbone = armature.data.edit_bones[pname]
                            bone.parent = pbone
                            if (psrc == src.parent) and (is_not_tracker):
                                bone.use_connect = src.bone.use_connect
                            break

                for o in armatures:
                    Set.select(armature, True)
                    Set.mode(context, armature, 'POSE')

                ordered_objects = list()
                for src in new_objects:
                    if src.parent in ordered_objects:
                        ordered_objects.insert(ordered_objects.index(src.parent) + 1, src)
                    else:
                        for child in src.children:
                            if child in ordered_objects:
                                ordered_objects.insert(ordered_objects.index(child) + 1, src)
                                break
                        else:
                            ordered_objects.append(src)

                for src in reversed(ordered_objects):
                    (armature, name) = new_objects[src]
                    bone = armature.pose.bones[name]
                    self.set_pose_bone(context, bone, src)
                    Set.bone_group(bone, f"{self.type}||{hash(self)}")
                    new_objects[src] = bone
                    Set.select(src, False)
                    Set.select(bone, True)
                    if (active is None or first_active == src):
                        Set.active(context, bone)
                        active = bone

            return new_objects

        def generate_physic_objects(self, context, owner, target):
            """Add two mesh objects and an empty, with physics preset"""
            # """Add constraint to target, containing physics meshes"""

            name = utils.string(
                "Physics",
                '||', target.id_data.name.split('||', 1)[0],
                '||', target.name,
            )
            if Is.posebone(target):
                width = target.bone.bbone_x
                height = target.length / 2
                depth = target.bone.bbone_z
                size = sum((width, depth, height))
            elif Is.empty(target):
                (width, depth, height) = (1, 1, 1)
                size = 1
            else:
                (width, depth, height) = target.dimensions
                size = sum((width, depth, height))

            shape = dict(width=width, depth=depth, height=height)
            if Is.posebone(target):
                shape['loc'] = (0, 0, 1)
                shape['rot'] = (-90,)

            mesh1 = New.mesh(context, name=name + '-Collision', type='CUBE', **shape)
            mesh2 = New.mesh(context, name=name + '-Bind', type='POINT')
            pivot = New.object(context, name=name + '-Pivot')

            # Move the meshes to target
            # mesh1.parent = mesh2.parent = pivot
            mesh1.matrix_world = Get.matrix(target)
            # mesh2.matrix_world = pivot.matrix_world = Get.matrix(target, tail=True)
            mesh2.matrix_world = Get.matrix(target)
            pivot.matrix_world = Get.matrix(target, tail=True)

            # Change the look default look of the meshes
            Set.object_display_type(mesh1, 'WIRE')
            Set.object_display_type(mesh2, 'WIRE')
            Set.empty_size(pivot, size)

            # Add physics to the meshes
            bpy.ops.rigidbody.object_add(dict(object=mesh1))
            bpy.ops.rigidbody.object_add(dict(object=mesh2))
            active = Get.active(context)
            Set.active(context, pivot)
            bpy.ops.rigidbody.constraint_add()
            Set.active(context, active)

            # Set default physics parameters
            mesh1.rigid_body.collision_shape = 'MESH'
            mesh2.rigid_body.kinematic = True
            pivot.rigid_body_constraint.object1 = mesh1
            pivot.rigid_body_constraint.object2 = mesh2

            con = self.new_constraint(pivot, 'CHILD_OF')
            self.set_constraint(context, con, pivot, target)
            con = self.new_constraint(mesh1, 'COPY_TRANSFORMS')
            self.set_constraint(context, con, mesh1, target)
            con = self.new_constraint(mesh2, 'CHILD_OF')
            self.set_constraint(context, con, mesh2, pivot)

            # Add a constraint to lock the original bone/object to the collision mesh
            con = self.new_constraint(owner, 'COPY_TRANSFORMS')
            con.name = 'Physics || Copy Transforms'
            self.set_constraint(context, con, owner, mesh1)

        def new_constraint(self, owner, type, index=-1):
            """Add a constraint to owner, then name and return it"""

            active_index = len(owner.constraints)
            constraint = owner.constraints.new(type)
            if index >= 0:
                owner.constraints.move(active_index, index)

            if self.hash_name:
                constraint.name = f"{type.title()}-{hash(constraint)}"
            constraint.show_expanded = self.show_expanded

            return constraint

        def set_child_of(self, constraint, loc, rot, scale):
            """macro to toggle the loc/rot/scale for a child_of constraint"""
            for v in ('x', 'y', 'z'):
                setattr(constraint, f'use_location_{v}', loc)
                setattr(constraint, f'use_rotation_{v}', rot)
                setattr(constraint, f'use_scale_{v}', scale)

        def set_constraint(self, context, con, owner, target=None):
            """
            Set the default values of the constraint\\
            Also perform "fixes" if needed
            """

            owner_type = ('OBJECT', 'BONE')[Is.posebone(owner)]

            def scan_ik(owner, disable_ik=dict()):
                """log IK chain mute statuses then mute them"""
                if owner.is_in_ik_chain:
                    for bc in [owner] + owner.children_recursive:
                        for bc_con in bc.constraints:
                            if bc_con.type in {'IK', 'SPLINE_IK'}:
                                disable_ik[bc, bc_con] = bc_con.mute
                                bc_con.mute = True
                return disable_ik

            if target:
                fix_inverse = False
                if con.type == 'CHILD_OF':
                    fix_inverse = False

                if fix_inverse:
                    # Remember pose to reset inverse
                    disable_ik = dict()
                    if Is.posebone(owner):
                        if owner.is_in_ik_chain:
                            disable_ik = scan_ik(owner)
                        else:
                            for b in owner.children:
                                disable_ik = scan_ik(b, disable_ik)
                    if disable_ik:
                        utils.update(context)
                    original_matrix = Get.matrix(owner)

                if Is.posebone(target):
                    subtarget = target.name
                else:
                    subtarget = ""

                if hasattr(con, 'target'):
                    con.target = target.id_data
                    if hasattr(con, 'subtarget') and subtarget:
                        con.subtarget = subtarget
                elif hasattr(con, 'targets'):
                    # Armature constraint
                    sub = con.targets.new()
                    sub.target = target.id_data
                    if subtarget:
                        sub.subtarget = subtarget

                if fix_inverse:
                    utils.update(context)
                    reset_matrix = Get.matrix(owner)
                    con.inverse_matrix = utils.multiply_matrix(
                        original_matrix, reset_matrix.inverted()
                    )

                    # try:
                    # except:
                    # if (con.use_location_x and not con.use_rotation_x):
                    #     cc = context.copy()
                    #     cc['constraint'] = con
                    #     cc['object'] = owner.id_data
                    #     bpy.ops.constraint.childof_set_inverse(
                    #         cc, constraint=con.name, owner=owner_type)

                    # unmute muted ik chains
                    for (bc, bc_con) in disable_ik:
                        bc_con.mute = disable_ik[bc, bc_con]

            if hasattr(con, 'target_space'):
                tspace = self.target_space
                if self.target_space != 'WORLD':
                    if not Is.posebone(target):
                        tspace = 'LOCAL'
                    if self.type == 'ARMATURE':
                        # Armature only supports ('WORLD', 'LOCAL')
                        # However Local is not the result I want
                        tspace = 'WORLD'
                con.target_space = tspace

            if hasattr(con, 'owner_space'):
                ospace = self.owner_space
                if self.owner_space != 'WORLD':
                    if not Is.posebone(owner):
                        ospace = 'LOCAL'
                    if self.type == 'ARMATURE':
                        # Armature only supports ('WORLD', 'LOCAL')
                        # However Local is not the result I want
                        ospace = 'WORLD'
                con.owner_space = ospace

            if hasattr(con, 'head_tail'):
                con.head_tail = self.head_tail
                if (self.type in {'track_rotation', 'stretch_rotation'} and
                        not (self.at_tail or self.head_tail)) or \
                    (self.type == 'STRETCH_TO' and
                        self.target in ('TO_ARMATURE', 'TO_OBJECT')):
                    # By default, instead of forcing the bone to the tail, use its tail
                    con.head_tail = 1.0

            if hasattr(con, 'track_axis'):
                con.track_axis = Constraint.get_track_axis(owner, self.track_axis)

            if hasattr(con, 'space_object'):
                con.space_object = self.space_object

            if len({self.target_space, self.owner_space, 'LOCAL_WITH_PARENT', 'LOCAL'}) == 2:
                if con.type == 'COPY_TRANSFORMS':
                    con.mix_mode = 'AFTER'
                    # mix_mode_items = [
                        # ('REPLACE', "Replace", "Replace the original transformation with copied"),
                        # ('BEFORE', "Before Original", "Apply copied transformation before original, as if the constraint target is a parent. "
                                                    #   "Scale is handled specially to avoid creating shear"),
                        # ('AFTER', "After Original", "Apply copied transformation after original, as if the constraint target is a child. "
                                                    # "Scale is handled specially to avoid creating shear"),
                    # ]
                elif con.type == 'COPY_LOCATION':
                    con.use_offset = True
                elif con.type == 'COPY_ROTATION':
                    con.mix_mode = 'AFTER'
                    # mix_mode_items = [
                        # ('REPLACE', "Replace", "Replace the original rotation with copied"),
                        # ('ADD', "Add", "Add euler component values together"),
                        # ('BEFORE', "Before Original", "Apply copied rotation before original, as if the constraint target is a parent"),
                        # ('AFTER', "After Original", "Apply copied rotation after original, as if the constraint target is a child"),
                        # ('OFFSET', "Offset (Legacy)", "Combine rotations like the original Offset checkbox. Does not work well for " "multiple axis rotations"),
                    # ]
                elif con.type == 'COPY_SCALE':
                    con.use_offset = True

            if con.type == 'ARMATURE':
                con.use_deform_preserve_volume = True

            elif con.type == 'IK':
                con.chain_count = 1  # Fixes length for default IK constraint
                prev = owner
                for b in owner.parent_recursive:
                    if Is.connected(prev, pseudo=not owner.bone.use_connect):
                        con.chain_count += 1
                        prev = b
                    else:
                        break

            elif con.type == 'FLOOR':
                # I could use the track_axis property or a separate property
                # but use a default instead
                con.floor_location = 'FLOOR_NEGATIVE_Z'
                con.use_rotation = True

            elif con.type == 'LIMIT_DISTANCE':
                con.limit_mode = 'LIMITDIST_ONSURFACE'
                active = Set.active(context, owner)
                bpy.ops.constraint.limitdistance_reset(
                    dict(constraint=con), constraint=con.name, owner=owner_type)
                Set.active(context, active)
            elif con.type == 'LIMIT_LOCATION':
                for xyz in 'xyz':
                    if getattr(owner.location, xyz) < 0:
                        setattr(con, 'min_' + xyz, getattr(owner.location, xyz))
                        setattr(con, 'max_' + xyz, 0)
                    else:
                        setattr(con, 'min_' + xyz, 0)
                        setattr(con, 'max_' + xyz, getattr(owner.location, xyz))
                con.use_min_x = con.use_min_y = con.use_min_z = True
                con.use_max_x = con.use_max_y = con.use_max_z = True
                con.owner_space = 'LOCAL'
            elif con.type == 'LIMIT_ROTATION':
                # con.min_x = con.min_y = con.min_z = radians(-360)
                # con.max_x = con.max_y = con.max_z = radians(360)
                if owner.rotation_mode == 'QUATERNION':
                    rot = owner.rotation_quaternion.to_euler('XYZ')
                elif owner.rotation_mode == 'AXIS_ANGLE':
                    rot = owner.rotation_quaternion.to_euler('XYZ')
                    # I don't know how to convert axis_angle to anything else, consistently
                else:
                    rot = owner.rotation_euler
                for xyz in 'xyz':
                    if getattr(rot, xyz) < 0:
                        setattr(con, 'min_' + xyz, getattr(rot, xyz))
                        setattr(con, 'max_' + xyz, 0)
                    else:
                        setattr(con, 'min_' + xyz, 0)
                        setattr(con, 'max_' + xyz, getattr(rot, xyz))
                con.use_limit_x = con.use_limit_y = con.use_limit_z = True
                con.owner_space = 'LOCAL'
            elif con.type == 'LIMIT_SCALE':
                for xyz in 'xyz':
                    if getattr(owner.scale, xyz) < 1:
                        setattr(con, 'min_' + xyz, getattr(owner.scale, xyz))
                        setattr(con, 'max_' + xyz, 1)
                    else:
                        setattr(con, 'min_' + xyz, 1)
                        setattr(con, 'max_' + xyz, getattr(owner.scale, xyz))
                con.use_min_x = con.use_min_y = con.use_min_z = True
                con.use_max_x = con.use_max_y = con.use_max_z = True
                con.owner_space = 'LOCAL'

            elif con.type == 'SHRINKWRAP':
                con.wrap_mode = 'OUTSIDE'

            elif con.type == 'STRETCH_TO':
                con.keep_axis = 'SWING_Y'

            # I honestly don't know how to use the transform constraint :p
            elif con.type == 'TRANSFORM':
                con.map_from = self.map_from
                con.map_to = self.map_to
                con.use_motion_extrapolate = True
                for dest in ('from', 'to'):
                    for axis in ('x', 'y', 'z'):
                        for p, v in {
                                ('', 10),
                                ('_rot', radians(10)),  # 180
                                ('_scale', 10)}:
                            setattr(con, f'{dest}_min_{axis}{p}', -v)
                            setattr(con, f'{dest}_max_{axis}{p}', v)
            # elif type == 'TRANSFORM':
                # spaces = self.map_from_to
                # if not spaces:
                    # spaces = {'LOCATION'}
                # for space in ('LOCATION', 'ROTATION', 'SCALE'):
                    # if space not in spaces:
                        # continue
                    # con = self.new_constraint(owner, type)
                    # self.set_constraint(context, con, owner, target)
                    # con = self.new(owner, type)
                    # con.show_expanded = expand_constraints
                    # con.map_from = con.map_to = space
                    # self.set(con, owner, target)
                    # self.fix(con, owner, target, is_custom)
                    # self.add_extra(owner, target, con)

            # if self.add_relation:
                # Constraint.add_constraint_relation(context, con, owner, target)

            if self.add_drivers and hasattr(con, 'influence') and target:
                prop = None  # kwargs.get('prop')
                desc = None  # kwargs.get('desc')
                self.add_constraint_driver(con, target, prop, desc)

        def set_edit_bone(self, context, bone, src):
            """Set the default pose for newly created bones"""

            from math import degrees
            from mathutils import Vector

            # When keyframing, use world space or local space
            if (self.type == 'pivot') and Is.posebone(src):
                bone.use_local_location = src.bone.use_local_location
            else:
                bone.use_local_location = self.use_local_location

            # Set the bone's default transforms
            bone.tail = Vector((0, 0, 1))
            bone.matrix = Get.matrix(src, local=self.use_local_transforms)

            if Is.posebone(src):
                for prop in ('length', 'bbone_x', 'bbone_z'):
                    setattr(bone, prop, getattr(src.bone, prop))

                bone.use_inherit_rotation = src.bone.use_inherit_rotation
                bone.inherit_scale = src.bone.inherit_scale

            bone.use_deform = False

            # if self.at_tail:
                # bone.translate(bone.tail - bone.head)
                # # bone.matrix.translation = bone.matrix.Translation(
                # #     bone.tail - bone.head).translation
            # elif self.head_tail:
                # slide = (bone.tail - bone.head)
                # tran = Vector(lerp(bone.head, slide, self.head_tail))
                # bone.translate(tran)

            # if self.type in {'DAMPED_TRACK'}:
                # head_tail = 0
            # elif self.at_tail or self.type in {'STRETCH_TO'}:
                # head_tail = 1
            # else:
                # head_tail = self.head_tail
            head_tail = self.at_tail

            if head_tail:
                # tail = multiply_matrix(
                #     Vector((0, bone.length, 0)),
                #     bone.matrix.inverted(),
                # )
                slide = Vector(utils.lerp(bone.head, bone.tail, head_tail))
                bone.translate(slide)
                # b.location += multipy_matrix(Vector((0, b.length, 0)) @ b.matrix_basis.inverted()

            # if self.copy_chain and src.parent in self.copy_chain_list:
                # bone.parent = bone.id_data.edit_bones[
                #     self.copy_chain_list[src.parent].name]
                # bone.use_connect = src.bone.use_connect
            if (self.rotation_x or self.rotation_y or self.rotation_z):
                new_matrix = utils.rotate_matrix(bone.matrix, (
                    *[degrees(getattr(self, f'rotation_{_}'))
                        for _ in ('x', 'y', 'z')],
                ))
                # nonlocal difference
                # difference = bone.matrix - new_matrix
                bone.matrix = new_matrix

        def set_pose_bone(self, context, bone, src):
            """Set the visual pose for newly created bones"""

            self.set_locks(bone)

            # Switch back to Pose mode and transfer pose
            bone.rotation_mode = src.rotation_mode
            if self.use_local_transforms:
                # Bone should NOT be a Default 0 space
                bone.matrix = Get.matrix(src)
            if self.copy_chain:
                utils.update(context)
            # bone.rotation_quaternion = (bone.matrix + difference).to_quaternion()

            if (src.id_data.data == bone.id_data):
                # bone created in active armature
                bone.custom_shape_transform = src  # default widgets to the original
            elif self.create_widgets:
                vis = bone.id_data.pose.bones.get('VIS-' + bone.name)
                if vis:
                    self.set_widget(bone, src, vis)

            # if self.at_tail:
                # update()
                # bone.matrix.translation = bone.matrix.Translation(
                #     getattr(src, 'tail', bone.tail)
                #     ).translation
            # elif self.head_tail:
                # update()
                # tran = Vector(lerp(src.head, src.tail, self.head_tail))
                # bone.matrix.translation = bone.matrix.Translation(
                #     tran
                #     ).translation

            # from math import degrees
            # # # # # new_matrix = rotate_matrix(bone.matrix, (
            # # # # #     *[degrees(getattr(self, f'rotation_{_}'))
            # # # # #         for _ in ('x', 'y', 'z')],
            # # # # # ))
            # # # # # bone.matrix = new_matrix

            # if self.type in {'DAMPED_TRACK'}:
                # head_tail = 0
            # elif self.type in {'STRETCH_TO'} and not (self.at_tail or self.head_tail):
                # # By default, instead of forcing the bone to the tail, use its tail
                # head_tail = 0
            # elif self.at_tail:
                # head_tail = 1
            # else:
                # head_tail = self.head_tail
            head_tail = self.at_tail

            if head_tail:
                utils.update(context)

                if bone.bone.use_local_location:
                    tail = bone.location + utils.multiply_matrix(
                        Vector((0, bone.length, 0)),
                        bone.matrix_basis.inverted(),
                    )
                    bone.location = Vector(utils.lerp(bone.location, tail, head_tail))
                else:
                    Set.matrix(bone, Get.matrix(bone, tail=True))

            track_axis = 'None'

            if self.type in ('DAMPED_TRACK', 'track_rotation'):
                track_axis = self.track_axis
                # rm = bone.rotation_mode
                # bone.rotation_mode = 'XYZ'

                if self.track_axis == 'TRACK_X':
                    # bone.rotation_euler.y += radians(90)
                    angle = -90
                    axis = 'Z'
                elif self.track_axis == 'TRACK_Y':
                    # bone.rotation_euler.x += radians(0)
                    angle = 0
                    axis = 'X'
                elif self.track_axis == 'TRACK_Z':
                    # bone.rotation_euler.x += radians(90)
                    angle = 90
                    axis = 'X'
                elif self.track_axis == 'TRACK_NEGATIVE_X':
                    # bone.rotation_euler.y += radians(-90)
                    angle = 90
                    axis = 'Z'
                elif self.track_axis == 'TRACK_NEGATIVE_Y':
                    # bone.rotation_euler.x += radians(180)
                    angle = 180
                    axis = 'X'
                elif self.track_axis == 'TRACK_NEGATIVE_Z':
                    # bone.rotation_euler.x += radians(-90)
                    angle = -90
                    axis = 'X'
                else:  # Impossible
                    angle = 0
                    axis = 'X'

                utils.update(context)
                mat_rotated = utils.multiply_matrix(
                    Get.matrix(bone),
                    Matrix.Rotation(radians(angle), 4, axis),
                    Matrix.Translation(Vector((0, bone.length * self.track_offset / 100, 0))),
                )
                Set.matrix(bone, mat_rotated)
                utils.update(context)
                Set.matrix(bone, Get.matrix(bone, tail=True))
                # bone.rotation_mode = rm
            elif self.type in ('STRETCH_TO', 'stretch_rotation') and self.track_offset:
                utils.update(context)
                mat_rotated = utils.multiply_matrix(
                    Get.matrix(bone),
                    Matrix.Translation(Vector((0, bone.length * self.track_offset / 100, 0)))
                )
                Set.matrix(bone, mat_rotated)
            elif self.type == 'IK':
                utils.update(context)
                Set.matrix(bone, Get.matrix(bone, tail=True))

            # Store the base matrix
            if hasattr(bone, 'base_transforms'):
                base = bone.base_transforms
                base.location = bone.location
                base.rotation_axis_angle = bone.rotation_axis_angle
                base.rotation_euler = bone.rotation_euler
                base.rotation_quaternion = bone.rotation_quaternion
                base.scale = bone.scale
                base.track_axis = track_axis
                base.track_offset = self.track_offset
            else:
                bone['base_transforms'] = dict(
                    location=bone.location,
                    rotation_axis_angle=bone.rotation_axis_angle,
                    rotation_euler=bone.rotation_euler,
                    rotation_quaternion=bone.rotation_quaternion,
                    scale=bone.scale,
                    track_axis=track_axis,
                    track_offset=self.track_offset,
                )

        def set_object(self, context, obj, src):
            """Set the visual pose for newly created objects"""

            self.set_locks(obj)

            # Switch back to Pose mode and transfer pose
            obj.rotation_mode = src.rotation_mode
            obj.matrix_world = Get.matrix(src)

            # if self.type in {'DAMPED_TRACK'}:
                # head_tail = 0
            if self.at_tail or self.type in {'STRETCH_TO', 'stretch_rotation'}:
                head_tail = 1
            else:
                head_tail = self.head_tail

            if head_tail:
                utils.update(context)
                Set.matrix(obj, Get.matrix(src, tail=True))

            track_axis = 'None'

            if self.type in ('DAMPED_TRACK', 'track_rotation'):
                track_axis = self.track_axis

                if self.track_axis == 'TRACK_X':
                    angle = -90
                    axis = 'Z'
                elif self.track_axis == 'TRACK_Y':
                    angle = 0
                    axis = 'X'
                elif self.track_axis == 'TRACK_Z':
                    angle = 90
                    axis = 'X'
                elif self.track_axis == 'TRACK_NEGATIVE_X':
                    angle = 90
                    axis = 'Z'
                elif self.track_axis == 'TRACK_NEGATIVE_Y':
                    angle = 180
                    axis = 'X'
                elif self.track_axis == 'TRACK_NEGATIVE_Z':
                    angle = -90
                    axis = 'X'
                    obj.rotation_euler.x += radians(-90)
                else:  # Impossible
                    angle = 0
                    axis = 'X'

                utils.update(context)
                mat_rotated = utils.multiply_matrix(
                    Get.matrix(src),
                    Matrix.Rotation(radians(angle), 4, axis),
                )
                Set.matrix(obj, mat_rotated)
                utils.update(context)

                y = (src.length if Is.posebone(src) else 1)
                Set.matrix(obj, utils.multiply_matrix(
                    Get.matrix(src, tail=True),
                    Matrix.Translation(Vector((0, y * self.track_offset / 100, 0))),
                ))
            elif self.type in ('STRETCH_TO', 'stretch_rotation') and self.track_offset:
                utils.update(context)
                y = (src.length if Is.posebone(src) else 1)
                Set.matrix(obj, utils.multiply_matrix(
                    Get.matrix(src, tail=True),
                    Matrix.Translation(Vector((0, y * self.track_offset / 100, 0))),
                ))
            elif self.type == 'IK':
                utils.update(context)
                Set.matrix(obj, Get.matrix(src, tail=True))

            # Store the base matrix
            if hasattr(obj, 'base_transforms'):
                base = obj.base_transforms
                base.location = obj.location
                base.rotation_axis_angle = obj.rotation_axis_angle
                base.rotation_euler = obj.rotation_euler
                base.rotation_quaternion = obj.rotation_quaternion
                base.scale = obj.scale
                base.track_axis = track_axis
                base.track_offset = self.track_offset
            else:
                obj['base_transforms'] = dict(
                    location=obj.location,
                    rotation_axis_angle=obj.rotation_axis_angle,
                    rotation_euler=obj.rotation_euler,
                    rotation_quaternion=obj.rotation_quaternion,
                    scale=obj.scale,
                    track_axis=track_axis,
                    track_offset=self.track_offset,
                )

        def set_locks(self, src):
            """When creating new constraint controllers, lock unneeded attributes"""
            type = self.type

            location = rotation = scale = False

            if type in ('COPY_LOCATION', 'IK', 'child_of_location'):
                rotation = True
                scale = True
            elif type in ('COPY_ROTATION', 'child_of_rotation'):
                location = True
                scale = True
            elif type in ('COPY_SCALE', 'child_of_scale'):
                location = True
                rotation = True
            elif type in ('LIMIT_DISTANCE', 'DAMPED_TRACK', 'LOCKED_TRACK', 'TRACK_TO', 'STRETCH_TO', 'PIVOT'):
                if not (self.head_tail or self.at_tail):
                    rotation = True
                    scale = True
            elif type == 'FLOOR':
                scale = True
            else:
                pass
                # 'COPY_TRANSFORMS', 'TRANSFORM',
                # 'ACTION', 'ARMATURE', 'CHILD_OF', 'child_of_locrot', 'ik_transforms', 'pivot',
                # 'track_rotation', 'stretch_rotation'
                # *Constraint.curves, 'SHRINKWRAP',

            if location:
                src.lock_location[:] = (True, True, True)
            if rotation:
                src.lock_rotation[:] = (True, True, True)
                src.lock_rotation_w = True
            if scale:
                src.lock_scale[:] = (True, True, True)

        def set_widget(self, bone, src, vis):
            """Create Widgets for new bones"""

            wgt_op = utils.find_op('bonewidget.create_widget')

            def wgt(bone, **args):
                """
                args:
                    slide=(0.0, 0.0, 0.0)
                    rotate=(0.0, 0.0, 0.0)
                    global_size=1.0
                    scale=(1.0, 1.0, 1.0)
                """

                if not args.get('widget'):
                    # col = bpy.data.collections.get('Widgets')
                    # if not col:
                        # col = bpy.data.collections.new('Widgets')
                    obj = bpy.data.objects.get('WDGT_VIS')
                    if not obj:
                        obj = bpy.data.objects.new('WDGT_VIS',
                            bpy.data.meshes.new('WDGT_VIS'))
                    # if obj not in list(col.all_objects):
                        # col.objects.link(obj)
                    bone.custom_shape = obj
                elif wgt_op:
                    prefs = utils.prefs('zpy__mods').bone_widgets
                    keep = prefs.keep_settings
                    prefs.keep_settings = False
                    wgt_op(dict(active_object=bone.id_data, selected_pose_bones=[bone]),
                        mirror=False, relative_size=True, **args)
                    prefs.keep_settings = keep

            def lock(src, type, name=""):
                con = vis.constraints.new(type)
                con.target = src.id_data
                con.subtarget = src.name
                if name:
                    con.name = name
                con.show_expanded = False
                return con

            vis.bone.hide_select = True

            if self.type in ('ik_transforms',):  # IK + Copy Transforms:
                wgt(vis)
                if 'hand' in bone.name:
                    wgt(bone, widget='Blenrig - Hand')
                elif 'foot' in bone.name:
                    wgt(bone, widget='Blenrig - Foot')
                else:
                    wgt(bone, widget='Blenrig - IK Limb')
                    bone.custom_shape_transform = vis
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
            elif self.type in ('DAMPED_TRACK', 'LOCKED_TRACK', 'TRACK_TO', 'IK'):
                wgt(bone, widget='Sphere')
                wgt(vis)
                lock(bone, 'COPY_TRANSFORMS', "Lock To Copy")
                lock(src, 'COPY_ROTATION', "Lock To Original")
                bone.custom_shape_transform = vis
            elif self.type in ('STRETCH_TO',):
                wgt(bone, widget='Rigify - Shoulder')
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                bone.custom_shape_transform = vis
            elif self.type in ('FLOOR',):
                wgt(bone, widget='Plane', rotate=(radians(90), 0, 0))
                wgt(vis, widget='Blenrig - Box', rotate=(radians(90), 0, 0))
                lock(bone, 'COPY_TRANSFORMS', "Lock To Copy")
                lock(src, 'COPY_LOCATION', "Lock To Original")
            elif self.type in ('PIVOT',):
                wgt(bone, widget='ik_pole')
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
            elif self.type in ('COPY_LOCATION', 'child_of_location'):
                # size = 1 + 0.2 * len([con for con in src.constraints
                                        #  if con.type in ('COPY_LOCATION', 'CHILD_OF')])
                size = 1.0
                wgt(bone, widget='Circle', global_size=size)
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                bone.custom_shape_transform = vis
            elif self.type in ('COPY_ROTATION', 'child_of_rotation'):
                # wgt(bone, widget='', slide=(0, 0.5, 0))
                # size = 0.25 + 0.2 * len([con for con in src.constraints
                                        #  if con.type in ('COPY_ROTATION', 'CHILD_OF')])
                size = 0.25
                wgt(bone, widget='Blenrig - Heel', slide=(-0.5, 0, 0), rotate=(0, 0, -radians(90)), global_size=size)
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                bone.custom_shape_transform = vis
            elif self.type in ('COPY_SCALE', 'child_of_scale'):
                # size = 1 + 0.2 * len([con for con in src.constraints
                                        #  if con.type in ('COPY_SCALE', 'CHILD_OF')])
                size = 1.0
                wgt(bone, widget='Finger', global_size=size)
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                bone.custom_shape_transform = vis
            elif self.type in ('COPY_TRANSFORMS', 'child_of_locrot'):
                # size = 1 + 0.2 * len([con for con in src.constraints
                                        #  if con.type in ('COPY_TRANSFORMS', 'CHILD_OF')])
                size = 1.0
                wgt(bone, widget='Rigify - Arm', global_size=size)
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                bone.custom_shape_transform = vis
            elif self.type in ('ARMATURE', 'CHILD_OF'):
                # size = 1 + 0.2 * len([con for con in src.constraints
                                        #  if con.type in ('ARMATURE', 'CHILD_OF')])
                size = 1.0
                wgt(bone, widget='Box', global_size=size)
                wgt(vis)
                # lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                # bone.custom_shape_transform = vis
            # 'LIMIT_DISTANCE',
            # 'TRANSFORM',
            # 'SPLINE_IK',
            # 'ACTION',
            # 'pivot',  # Child of + Rotation
            # 'track_rotation',  # Damped Track + Rotation
            # 'stretch_rotation',  # Stretch To + Rotation
            elif self.type in ():
                wgt(bone, widget='Circle', slide=(0, 0.5, 0))
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                bone.custom_shape_transform = vis
            else:
                wgt(vis)
                lock(src, 'COPY_TRANSFORMS', "Lock To Original")
                bone.custom_shape_transform = vis

        def spline_ik(self, context, pairs):
            utils.update(context)
            ik_splines = dict()

            for (owner, target) in pairs:
                ik_splines[owner] = target

            spline_cons = set()

            for chain in Get.sorted_chains(ik_splines.keys()):
                if len(chain) == 1 and chain[0].parent:
                    continue
                    # owner = chain[0].parent_recursive[-1]
                    # if self.target == 'TO_ARMATURE':
                    #     target = Constraint.new_bone(self, owner)
                    # if self.target == 'TO_EMPTY':
                    #     target = Constraint.new_empty(self, owner)
                    # Constraint.add(self, owner, target, self.type)
                    # target.id_data.location = target.id_data.location

                    # chain.insert(0, owner)

                curve = New.curve(context, name=f"{Get.name(chain[-1])}_Spline_IK")
                spline = New.spline(curve)
                Set.visible(context, curve, False)

                for bone in chain:
                    target = ik_splines.get(bone, None)
                    if target is None:
                        continue

                    target.base_src.curve_target = curve.name

                    con = self.add_spline(context, bone, target, chain, ik_splines, curve, spline)
                    if con:
                        spline_cons.add(con)

                    if not bone.parent:
                        """Add movement to pelvis bone"""
                        con = self.new_constraint(bone, 'COPY_LOCATION')
                        self.set_constraint(context, con, bone, target)
                        spline_cons.add(con)

                # Keep base of spline attached to the rest of the rig
                bone = chain[0].parent
                target = ik_splines.get(chain[0], None)
                if (bone and target):
                    con = self.new_constraint(target, 'CHILD_OF')
                    self.set_constraint(context, con, target, bone)

            for con in spline_cons:
                con.influence = 1.0

            return

        def add_spline(self, context, bone, target, chain, ik_splines, curve, spline):

            def set_point(spline, index, co):
                point = spline.bezier_points[index]
                # point.handle_left_type = 'VECTOR'
                # point.handle_right_type = 'VECTOR'
                point.handle_left_type = 'AUTO'
                point.handle_right_type = 'AUTO'
                    # ('FREE', 'VECTOR', 'ALIGNED', 'AUTO')
                point.co = co

            # if Is.posebone(target):
                # from ._mods.boneWidget.functions.mainFunctions import createWidget
                # from ._mods.boneWidget.functions.jsonFunctions import readWidgets
                # createWidget(
                    # target,
                    # readWidgets()['Sphere'],
                    # False,  # self.relative_size,
                    # 0.2,  # self.global_size,
                    # (1.0, 1.0, 1.0),  # [*self.scale],  # [1, 1, 1],
                    # (0.0, 0.0, 0.0),  # self.slide,
                    # (0.0, 0.0, 0.0),  # self.rotate,
                # )
            if Is.empty(target):
                Set.empty_type(target, 'SPHERE')
                Set.empty_size(target, 0.02)

            # Store individual bone chains for each curve + chain
            ctxt = (repr(chain))
            spline_is_new = bool(ctxt not in ik_splines)

            if not spline_is_new:
                index = ik_splines[ctxt]
                spline.bezier_points.add(1)

            index = len(spline.bezier_points) - 1
            at_tail = bool(bone == chain[-1])
            ik_splines[ctxt] = index

            # Make curve match the shape of bones
            loc = Get.matrix(bone).to_translation()
            set_point(spline, index, loc)

            # Add hook modifier to curve
            mod = New.hook(curve, target, index)
            mod.name = f"Hook-{target.name}"

            if 1:  # not at_tail:  # or spline_is_new:
                """Add a point for every tail"""
                spline.bezier_points.add(1)

                # Make curve match the shape of bones
                loc = Get.matrix(bone).to_translation() + bone.tail - bone.head
                set_point(spline, index + 1, loc)

                # Add hook modifier to curve
                mod = New.hook(curve, target, index + 1)
                mod.name = f"Hook-{target.name}-Tail"
                utils.update(context)

            if not at_tail:
                return

            """Add Spline IK Constraint"""
            con = self.new_constraint(bone, 'SPLINE_IK')
            self.set_constraint(context, con, bone, curve)

            for b in [chain[-1]] + bone.parent_recursive:
                if b == chain[0]:
                    # con.chain_count += 1  # don't stop at first bone's tail
                    break
                con.chain_count += 1
            con.use_chain_offset = True  # Don't "move" bones to match spline

            con.y_scale_mode = 'BONE_ORIGINAL'  # 2.8
            # con.use_y_stretch = False  # 2.7

            con.xz_scale_mode = 'BONE_ORIGINAL'

            for chain_bone in chain:
                chain_target = ik_splines.get(chain_bone)

            # Keep the curve shape but make them flexible
            # update()
            for point in spline.bezier_points:
                point.handle_left_type = 'FREE'
                point.handle_right_type = 'FREE'
            con.influence = 0.0

            return con

        def init_nla(self, src, copy):
            blend = getattr(src.animation_data, 'action_blend_type', 'REPLACE')
            if blend != 'REPLACE':
                copy.animation_data_create().action_blend_type = blend

    class properties:
        """
        Constraint attributes for preferences and operator
        """
        type: EnumProperty(
            items=Constraint.constraint_type_items,
            name="Constraint Type",
            description="",
            default='COPY_TRANSFORMS',
            options={'SKIP_SAVE'},
        )
        target: EnumProperty(
            items=[
                ('', "To Selected", "", 'MOUSE_LMB', 0),
                ('TO_ACTIVE', "Active",
                    "Selected.constraints.target = Active",
                    'RESTRICT_SELECT_OFF', 2),
                ('TO_ARMATURE', "Armature",
                    "Selected.constraints.target = New_Rig_Bones",
                    'GROUP_BONE', 3),
                ('TO_OBJECT', "Object",
                    "Selected.constraints.target = New_Objects",
                    'OUTLINER_OB_EMPTY', 5),
                ('NONE', "No target",
                    "Add constraints without target",
                    'BLANK1', 1),
                ('', "From Selected", "", 'MOUSE_RMB', 0),
                ('FROM_ACTIVE', "Active",
                    "Active.constraints.target = Selected",
                    'OBJECT_DATAMODE', 6),
                ('FROM_ARMATURE', "Armature (dummies)",
                    "New_Rig_Bones.constraints.target = Selected",
                    'CONSTRAINT_BONE', 7),
                ('FROM_OBJECT', "Object (dummies)",
                    "New_Objects.constraints.target = Selected",
                    'OUTLINER_DATA_EMPTY', 9),
            ],
            name="Target Type",
            description="Default targetting mode for constraint creation",
            default='TO_ACTIVE',
            # options={'ENUM_FLAG'},
        )
        def new_arm_toggle(self, context):
            self.add_relation = self.new_armature
        new_armature: BoolProperty(
            name="Create New Armature Object(s)",
            description="Create new objects for bones, or use the original rig objects",
            default=True,
            update=new_arm_toggle,
        )

        armature_display_type: EnumProperty(
            items=[
                ('RANDOM', "Random",
                    "Randomly select the display type", 'FREEZE', 1),
                ('OCTAHEDRAL', "Octahedral",
                    "Display bones as octahedral shape (default)",
                    # 'PMARKER_SEL', 2),
                    'PMARKER_ACT', 2),
                ('STICK', "Stick",
                    "Display bones as simple 2D lines with dots",
                    'IPO_LINEAR', 3),
                ('BBONE', "B-Bone",
                    "Display bones as boxes, showing subdivision and B-Splines",
                    'MESH_CUBE', 5),
                ('ENVELOPE', "Envelope",
                    "Display bones as extruded spheres, showing deformation influence volume",
                    'PIVOT_MEDIAN', 6),
                ('WIRE', "Wire",
                    "Display bones as thin wires, showing subdivision and B-Splines",
                    'CURVE_DATA', 4),
            ],
            name="Display As",
            description="",
            default='BBONE',
            # options={'HIDDEN', 'SKIP_SAVE'}.
        )

        def set_head_tail(self, context):
            "Function to autoset At_Tail(Float), when setting Head_Tail(Bool)"

            self.head_tail = int(self.at_tail)
        at_tail: BoolProperty(
            name="Create At Tails",
            description="When creating new bones/objects, default them to "
                        "the originals' tail",
            default=False,
            options={'SKIP_SAVE'},
            update=set_head_tail,
        )
        head_tail: FloatProperty(
            name="Head/Tail",
            description="Target along length of bone: Head=0, Tail=1",
            default=0,
            soft_min=0,
            soft_max=1,
            step=3,  # (int) – Step of increment/decrement in UI, in [1, 100], defaults to 3 (* 0.01)
            precision=2,  # (int) – Maximum number of decimal digits to display, in [0, 6].
            options={'SKIP_SAVE'},
            subtype='FACTOR',  # (string) – Enumerator  in ['PIXEL', 'UNSIGNED', 'PERCENTAGE', 'FACTOR', 'ANGLE', 'TIME', 'DISTANCE', 'NONE'].
            # unit='NONE',  # (string) – Enumerator  in ['NONE', 'LENGTH', 'AREA', 'VOLUME', 'ROTATION', 'TIME', 'VELOCITY', 'ACCELERATION'].
        )

        copy_chain: BoolProperty(
            name="Copy Bone Parent Chain",
            description="Duplicate selected bones and "
                        "their parents and mimick their hierachy",
            default=False,
            options={'SKIP_SAVE'},
        )

        use_mesh: BoolProperty(
            name="Use Mesh Objects",
            description="When generating empty objects, create meshes instead",
            default=False,
            # options={'SKIP_SAVE'},
        )
        do_physics: BoolProperty(
            name="Add Phyics Objects",
            description="Create additional mesh objects with physics",
            default=False,
            options={'SKIP_SAVE'},
        )

        add_drivers: BoolProperty(
            name="Add Drivers",
            description="Insert drivers to the new constraints' influence",
            default=False,
            # options={'SKIP_SAVE'},
        )
        add_relation: BoolProperty(
            name="Add Relation",
            description="Insert custom properties to define relation "
                        "between the selected and new bones/objects",
            default=True,
            # options={'SKIP_SAVE'},
        )

        bake_animation: BoolProperty(
            name="Bake Transforms",
            # name="Bake Animation",
            description="Bake selected item's transforms to the new controls "
                        "before setting constraints",
            # description="Bake visual transforms to newly created bones/objects",
            default=False,
            # options={'SKIP_SAVE'},
        )
        bake_mode: EnumProperty(
            items=[
                # # ('frame', "Current Frame", "Store animation data from the current frame", 'PARTICLE_TIP', 0),
                ('range', "Frame Range", "Store animation data from the selected frame range", 'PARTICLE_PATH', 1),
                ('keyframes', "Available Keyframes", "Insert keys only on frames with existing keys (all transforms)", 'PARTICLE_POINT', 2),
            ],
            name="Cache Frames",
            description="Method to determine which frames to process for cache",
            default='range',
        )
        push_nla: BoolProperty(
            name="Push To NLA",
            description="Push baked actions to NLA strips after baking",
            default=False,
        )
        display_bake: BoolProperty(
            name="Display Bake Progress",
            description="Display baking process if baking animation at creation",
            default=True,
        )

        rotation: FloatVectorProperty(
            name="Rotation",
            description="Default rotation offset of new bones/objects",
            default=(0, 0, 0),
            soft_min=radians(-360),
            soft_max=radians(360),
            step=45,
            unit='ROTATION',
            size=3,
        )

        copy_chain: BoolProperty(
            name="Copy Bone Parent Chain",
            description="Duplicate selected bones and "
                        "their parents and mimick their hierachy",
            default=False,
            options={'SKIP_SAVE'},
        )

        def update_space(self, context):
            """Keep the spaces in sync, since I do this anyway"""
            self.owner_space = self.target_space
        target_space: EnumProperty(
            items=[
                ("WORLD", "World Space",
                    "The transformation of the target is evaluated relative to the world "
                    "coordinate system"),
                ("POSE", "Pose Space",
                    "The transformation of the target is only evaluated in the Pose Space, "
                    "the target armature object transformation is ignored"),
                ("LOCAL_WITH_PARENT",
                    "Local With Parent", "The transformation of the target bone is evaluated relative its local "
                    "coordinate system, with the parent transformation added"),
                ("LOCAL", "Local Space",
                    "The transformation of the target is evaluated relative to its local "
                    "coordinate system"),
            ],
            name="Target Space",
            description="",
            default=None,
            # options={},
            update=update_space,
        )
        owner_space: EnumProperty(
            items=[
                ("WORLD", "World Space",
                    "The constraint is applied relative to the world coordinate system"),
                ("POSE", "Pose Space",
                    "The constraint is applied in Pose Space, the object transformation is ignored"),
                ("LOCAL_WITH_PARENT", "Local With Parent",
                "The constraint is applied relative to the local coordinate system of the object, "
                    "with the parent transformation added"),
                ("LOCAL", "Local Space",
                    "The constraint is applied relative to the local coordinate system of the object"),
            ],
            name="Owner Space",
            description="",
            default=None,
            # options={},
        )
        use_local_location: BoolProperty(
            name="Local Location",
            description="When creating animation, use local space instead of world space",
            default=False,
        )

        track_axis: EnumProperty(
            items=[
                ("TRACK_X", "X", "", 'BLANKS1', 1),
                ("TRACK_Y", "Y", "", 'BLANKS1', 2),
                ("TRACK_Z", "Z", "", 'BLANKS1', 3),
                ("TRACK_NEGATIVE_X", "-X", "", 'BLANKS1', 4),
                ("TRACK_NEGATIVE_Y", "-Y", "", 'BLANKS1', 5),
                ("TRACK_NEGATIVE_Z", "-Z", "", 'BLANKS1', 6),
                ],
            name="Track Axis",
            description="",
            default='TRACK_Y',
            # options={},
        )
        track_offset: FloatProperty(
            name="Track Offset",
            description="Distance to offset damped trackers",
            default=0,
            min=0,
            soft_max=500,
            # step=100,
            precision=0,
            # options={'SKIP_SAVE'},
            subtype='PERCENTAGE',  # (string) – Enumerator  in ['PIXEL', 'UNSIGNED', 'PERCENTAGE', 'FACTOR', 'ANGLE', 'TIME', 'DISTANCE', 'NONE'].
            # unit='NONE',  # (string) – Enumerator  in ['NONE', 'LENGTH', 'AREA', 'VOLUME', 'ROTATION', 'TIME', 'VELOCITY', 'ACCELERATION'].
        )

        map_from: EnumProperty(
            items=[
                ('LOCATION', "Loc", ""),
                ('ROTATION', "Rot", ""),
                ('SCALE', "Scale", ""),
                ],
            name="Map From",
            description="Which parameter to set defaults for Transform constraint",
            default=None,
            options={'SKIP_SAVE'},
        )
        map_to: EnumProperty(
            items=[
                ('LOCATION', "Loc", ""),
                ('ROTATION', "Rot", ""),
                ('SCALE', "Scale", ""),
                ],
            name="Map To",
            description="Which parameter to set defaults for Transform constraint",
            default=None,
            options={'SKIP_SAVE'},
        )

        space_object: EnumProperty(
            items=[
                ("WORLD", "World Space",
                "The transformation of the target is evaluated"
                " relative to the world coordinate system"),
                ("LOCAL", "Local Space",
                "The transformation of the target is evaluated"
                " relative to its local coordinate system"),
                ],
            name="Space Object",
            description="",
            default=None,
            # options={},
        )

        show_expanded: BoolProperty(
            name="Expand Constraints",
            description="Leave constraint boxes open after creation",
            default=True,
        )

        create_widgets: BoolProperty(
            name="Create Widgets",
            description="Create Widget controllers for new bones",
            default=False,
        )

    # region: globals
    # Store the text label of constraint types and custom types
    constraint_labels_type = {x[1]: (x[0], x[2]) for x in (
        *Constraint.types.motion_tracking,
        *Constraint.types.transform,
        *Constraint.types.tracking,
        *Constraint.types.relationship,
    )}
    custom_labels_type = {x[1]: (x[0], x[2]) for x in (
        *Constraint.types.child_of,
        *Constraint.types.other,
    )}

    def _get_icons():
        """convert constraint_type_items into dict() of type and icon"""
        icons = dict()
        for entry, b, *c in Constraint.constraint_type_items:
            if len(c) == 1:
                icon = 'NONE'
            else:
                icon = c[1]
            icons[entry] = icon

        return icons
    icons = _get_icons()

    # endregion

    ###############################################################################
    # region: Functions

    def is_at_tail(src, other):
        for con in other.constraints:
            if not hasattr(con, 'target') or ((con.target != src) and (con.target != src.id_data or con.subtarget != src.name)):
                continue

            if (con.type == 'IK' and con.use_tail):
                return True
            elif con.type in ('DAMPED_TRACK', 'STRETCH_TO') and not con.head_tail:
                return True

        return False
    # def is_at_tail(src, other):
        # if other == get.original(src):
        #     return False
        # else:
        #     for con in other.constraints:
        #         if con.type in ('DAMPED_TRACK', 'STRETCH_TO', 'IK'):
        #             if con.target == src:
        #                 return True
        #             elif con.target == src.id_data and con.subtarget == src.name:
        #                 return True
        # if other == get.original(src):
        #     return src.constraints_relative.at_tail

        # connect = Constraint.get.references(src, other)
        # if connect:
        #     for rel in connect:
        #         if rel.at_tail:
        #             return True
        # else:
        #     for con in other.constraints:
        #         if con.type in ('DAMPED_TRACK', 'STRETCH_TO', 'IK'):
        #             if con.target == other.id_data and \
        #                     con.subtarget == other.name:
        #                 return True

    def get_copies(src):
        """
        return list of bones and objects created from the src
        """
        if not hasattr(src, 'base_src'): return

        copies = list()

        for item in src.base_src.copies:
            owner = (src.id_data if item.target_self else item.target)
            (target, subtarget) = Get.object_bone_from_string(owner, item.subtarget)

            copies.append(subtarget if item.subtarget else target)

        return copies

    def get_original(src):
        """
        return the original bone or object for the src
        """
        if not hasattr(src, 'base_src'): return

        base = src.base_src

        owner = (src.id_data if base.target_self else base.target)
        (target, subtarget) = Get.object_bone_from_string(owner, base.subtarget)

        return subtarget if base.subtarget else target

        # if hasattr(src.id_data, 'constraints_extra'):
            # con_list = src.id_data.constraints_extra
            # con_entry = con_list.get(src.name)
            # if con_entry is None:
                # return

            # co = con_entry.original
            # target, subtarget = Get.object_bone_from_string(
            # co.target, co.subtarget)

            # origin = subtarget if co.subtarget else target
        # else:
            # con_list = src.id_data.get('constraints_extra', dict())
            # con_entry = con_list.get(src.name, dict())
            # if con_entry is None:
                # return

            # co = con_entry['original']
            # target, subtarget = Get.object_bone_from_string(
            # co['target'], co['subtarget'])

            # origin = subtarget if co['subtarget'] else target

        # return origin

    # Was used in zpy_constraints\ui_properties_panel.py > draw_extra
    # def get_constraint_relations(context, src):
        # """
        # Add custom property to owner's object, to mark constraint information
        # """

        # # if not hasattr(src, 'base_src'): return

        # # return

        # # if hasattr(src.id_data, 'constraints_extra'):
            # # con_list = src.id_data.constraints_extra
            # # con_entry = con_list.get(src.name)
            # # if src.name in con_list:
            #     # con_entry = con_list[src.name]
            # # else:
            #     # return

            # # class extra:
            #     # constraints = con_entry.constraints
            #     # self = con_entry.self
            #     # copies = con_entry.copies
            #     # original = con_entry.original

            # # return extra

            # # # for con_name in con_constraints:
            #     # # item = con_constraints[con_name]
            #     # # owner = item['owner']
            #     # # target = item['target']
        # # else:
            # # con_list = src.id_data.get('constraints_extra', dict())
            # # con_entry = con_list.get(src.name)
            # # if con_entry is None:
            #     # return

            # # class extra:
            #     # constraints = con_entry.get('constraints', dict())
            #     # self = con_entry.get('self')
            #     # copies = con_entry.get('copies', dict())
            #     # original = con_entry.get('original')

            # # return extra

            # # # for con_name in con_constraints:
            #     # # item = con_constraints[con_name]
            #     # # owner = item['owner']
            #     # # target = item['target']

    # Was used in zpy_constraints\operator_remove_constraints
    # def remove_constraint_relation(context, name, *owners):
        # """
        # Delete custom properties of the constraint, in the owners' objects
        # """

        # # for src in owners:
            # # if (not src):
            #     # continue

            # # if not hasattr(src, 'base_src'): continue

            # # continue

            # # if hasattr(src.id_data, 'constraints_extra'):
                # # con_list = src.id_data.constraints_extra
                # # con_entry = con_list.get(src.name)
                # # if con_entry is None:
                #     # continue
                # # con_constraints = con_entry.constraints
                # # index = con_constraints.find(name)
                # # if index != -1:
                #     # con_constraints.remove(index)
            # # else:
                # # if (src.id_data.get('constraints_extra') is None):
                #     # continue
                # # con_list = Constraint.get_dict(src.id_data, 'constraints_extra')
                # # con_entry = Constraint.get_dict(con_list, src.name)
                # # con_constraints = Constraint.get_dict(con_entry, 'constraints')
                # # con_item = Constraint.get_dict(con_constraints, name)

                # # con_constraints.pop(name)
                # # if not con_constraints: con_entry.pop('constraints')
                # # if not con_entry: con_list.pop(src.name)
                # # if not con_list: src.id_data.pop('constraints_extra')

    # endregion
    ###############################################################################


# __all__ = (
    # # "init",
    # # "register",
    # # "unregister",
    # "load_modules",
# )

class load_modules():
    def __init__(self, file, package):
        self.__file__ = file
        self.__package__ = package
        self.modules = None
        self.ordered_classes = None
        self.prefs_props = list()
        self.keymaps = dict()

    def init(self):
        from pathlib import Path

        self.modules = self.get_all_submodules(Path(self.__file__).parent)
        self.ordered_classes = self.get_ordered_classes_to_register(self.modules)

    def register(self):
        self.delete_zpy()
        self.init()

        for module in self.modules:
            if hasattr(module, "classes"):
                for cls in list(module.classes):
                    if getattr(cls, "is_registered", False):
                        bpy.utils.unregister_class(cls)
                    try:
                        self.register_class(cls)
                    except:
                        pass

        failed_classes = list()
        for cls in self.ordered_classes:
            if not getattr(cls, "is_registered", False):
                try:
                    self.register_class(cls)

                    # Send the keymaps directly to classes
                    if (bpy.types.AddonPreferences in cls.mro()):
                        cls.keymaps = self.keymaps
                except:
                    failed_classes.append(cls)

        for module in self.modules.copy():
            if module.__name__ == __name__:
                continue
            if hasattr(module, "register"):
                try:
                    module.register()
                except Exception as error:
                    print(f"Can't register module:\t{module.__name__}\n{error}")
                    self.modules.remove(module)

                    if hasattr(module, "classes"):
                        for cls in list(module.classes):
                            if getattr(cls, "is_registered", False):
                                bpy.utils.unregister_class(cls)
                    continue
            if hasattr(module, 'km'):
                for keymap, kmis in module.km.addon_keymaps.items():
                    if keymap not in self.keymaps:
                        self.keymaps[keymap] = list()
                    for kmi in kmis:
                        if kmi not in self.keymaps[keymap]:
                            self.keymaps[keymap].append(kmi)

        for cls in failed_classes:
            if not getattr(cls, "is_registered", False):
                try:
                    self.register_class(cls)
                except Exception as error:
                    self.ordered_classes.remove(cls)
                    print("Error: Can't register class: ", cls)
                    print("\t", error)

    def register_class(self, cls):
        # bpy_props = tuple(eval(f'bpy.props.{x}Property') for x in (
        #     'Pointer', 'Collection',
        #     'Bool', 'BoolVector', 'Enum', 'String',
        #     'Float', 'FloatVector', 'Int', 'IntVector',
        # ))
        if bpy.app.version < (2, 80, 0):
            # De-annotate properties for 2.7 backport
            for attribute in getattr(cls, '__annotations__', dict()).copy():
                prop, kwargs = cls.__annotations__[attribute]
                if not hasattr(cls, attribute):
                    setattr(cls, attribute, prop(**kwargs))
                    if attribute not in cls.order:
                        cls.order.append(attribute)
        # else:
            # # Annotate properties for 2.8
            # for attr in dir(cls):
            #     prop = getattr(cls, attr)

            #     if not prop:
            #         continue
            #     elif isinstance(prop, tuple) and prop[0] in bpy_props:
            #         pass
            #     elif prop in bpy_props:
            #         pass
            #     else:
            #         continue

            #     if not hasattr(cls, '__annotations__'):
            #         cls.__annotations__ = dict()

            #     if not cls.__annotations__.get(attr):
            #         # If the property is already annotated, overwriting it may not be desired
            #         cls.__annotations__[attr] = prop
            #         delattr(cls, attr)

        bpy.utils.register_class(cls)

    def unregister(self):
        from sys import modules as sys_modules

        for cls in reversed(self.prefs_props):
            if getattr(cls, 'is_registered', False):
                bpy.utils.unregister_class(cls)
            self.prefs_props.remove(cls)

        for cls in reversed(self.ordered_classes):
            if getattr(cls, 'is_registered', False):
                bpy.utils.unregister_class(cls)

        for module in self.modules:
            if module.__name__ == __name__:
                continue
            if hasattr(module, "unregister"):
                try:
                    module.unregister()
                except Exception as error:
                    print(f"Can't unregister module:\t{module.__name__}\n{error}")

        for module in self.modules:
            if module.__name__ == __name__:
                continue
            if module.__name__ in sys_modules:
                del (sys_modules[module.__name__])

        # Remove the remaining entries for the folder, zpy, and zpy.functions
        for module_name in reversed(list(sys_modules.keys())):
            # if module_name == __name__:  # This should exist anyway
            #     continue
            if module_name.startswith(self.__package__ + '.') or module_name == self.__package__:
                del sys_modules[module_name]

        self.keymaps.clear()

    # Import modules
    #################################################

    def get_all_submodules(self, directory):
        return list(self.iter_submodules(directory, directory.name))

    def iter_submodules(self, path, package_name):
        import importlib
        for name in sorted(self.iter_submodule_names(path)):
            # try:
            yield importlib.import_module("." + name, package_name)
            # except Exception as error:
                # print(error, name, package_name, path)

    def iter_submodule_names(self, path, root=""):
        import pkgutil
        for _, module_name, is_package in pkgutil.iter_modules([str(path)]):
            if is_package:
                sub_path = path / module_name
                sub_root = root + module_name + "."
                yield from self.iter_submodule_names(sub_path, sub_root)
            else:
                yield root + module_name

    def delete_zpy(self):
        "Delete zpy so that it can reload"
        from sys import modules as sys_modules

        if self.modules is not None:
            for module in self.modules:
                if module.__name__ == __name__:
                    continue
                if module.__name__ in sys_modules:
                    del (sys_modules[module.__name__])

        root = None
        # from pathlib import Path
        # root_self = str(Path(self.__file__).parent) + '\\'
        for (name, module) in sys_modules.copy().items():
            if getattr(module, '__file__', None) is None:
                continue

            if name == 'zpy' and hasattr(module, '__path__'):
                root = module.__path__[0] + '\\'

            if (root is not None and module.__file__.startswith(root)):
                del sys_modules[name]

    # Find classes to register
    #################################################

    def get_ordered_classes_to_register(self, modules):
        return self.toposort(self.get_register_deps_dict(modules))

    def get_register_deps_dict(self, modules):
        deps_dict = {}
        classes_to_register = set(self.iter_classes_to_register(modules))
        for cls in classes_to_register:
            deps_dict[cls] = set(self.iter_own_register_deps(cls, classes_to_register))
        return deps_dict

    def iter_own_register_deps(self, cls, own_classes):
        yield from (dep for dep in self.iter_register_deps(cls) if dep in own_classes)

    def iter_register_deps(self, cls):
        import typing
        for value in typing.get_type_hints(cls, {}, {}).values():
            dependency = self.get_dependency_from_annotation(value)
            if dependency is not None:
                yield dependency

    def get_dependency_from_annotation(self, value):
        if isinstance(value, tuple) and len(value) == 2:
            if value[0] in (bpy.props.PointerProperty, bpy.props.CollectionProperty):
                return value[1]["type"]
        return None

    def iter_classes_to_register(self, modules):
        base_types = self.get_register_base_types()
        for cls in self.get_classes_in_modules(modules):
            # if any(base in base_types for base in cls.__bases__):
            if any(base in base_types for base in cls.mro()):
                if not getattr(cls, "is_registered", False):
                    # if bpy.types.AddonPreferences in cls.__bases__:
                    #     if not hasattr(cls, '__annotations__'):
                    #         cls.__annotations__ = dict()
                    #     self.reg_addon_groups(cls)
                    yield cls

    def reg_addon_groups(self, cls):
        "Find PropertyGroup classes and register them as Pointers"
        from inspect import isclass

        for i in vars(cls):
            prop = getattr(cls, i, None)
            if not isclass(prop):
                continue

            if issubclass(prop, bpy.types.PropertyGroup):
                if not getattr(prop, "is_registered", False):
                    register_class(prop)
                    self.prefs_props.append(prop)
                self.reg_addon_groups(prop)
                group = bpy.props.PointerProperty(type=prop)

                # setattr(cls, i, group)
                if hasattr(cls, '__annotations__'):
                    cls.__annotations__[i] = group
                else:
                    setattr(cls, i, group)

    def get_classes_in_modules(self, modules):
        classes = set()
        for module in modules:
            for cls in self.iter_classes_in_module(module):
                classes.add(cls)
        return classes

    def iter_classes_in_module(self, module):
        from inspect import isclass

        for value in module.__dict__.values():
            if isclass(value):
                yield value

    def get_register_base_types(self):
        return set(getattr(bpy.types, name) for name in [
            "Panel", "Operator", "PropertyGroup",
            "AddonPreferences", "Header", "Menu",
            "Node", "NodeSocket", "NodeTree",
            "UIList", "RenderEngine",
            'KeyingSetInfo',
        ])

    # Find order to register to solve dependencies
    #################################################

    def toposort(self, deps_dict):
        sorted_list = []
        sorted_values = set()
        while len(deps_dict) > 0:
            unsorted = []
            for value, deps in deps_dict.items():
                if len(deps) == 0:
                    sorted_list.append(value)
                    sorted_values.add(value)
                else:
                    unsorted.append(value)
            deps_dict = {value: deps_dict[value] - sorted_values for value in unsorted}
        return sorted_list

# endregion functions that are initialized()
