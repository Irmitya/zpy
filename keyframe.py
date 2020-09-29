"shortcuts for inserting keyframes"
import bpy
from zpy import Get, Is, New


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


keyframe = type('', (), globals())
