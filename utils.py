"""
Various shortcut/utility functions\\
Generally either does one specific thing without any context
"""
import bpy
from zpy import Get, Is


def clear_console():
    import os
    os.system("cls")

def line():
    from inspect import currentframe

    cf = currentframe()

    return cf.f_back.f_lineno

def file_line(levels=0):
    line = Get.line(levels)
    file = Get.file(levels)
    # return f"line #{line} in {file}"
    function = Get.stack(levels + 2)

    return f"line #{line} in {function}"

def clean_custom(src):
    """Delete empty custom prop entries"""

    from idprop.types import IDPropertyArray, IDPropertyGroup

    for (name, value) in src.items():
        if isinstance(value, IDPropertyGroup) and not value:
            del (src[name])

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
        while 60 <= second:  # Minutes
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
        elif micro:
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

def flip_name(name, split=False, only_split=False):
    if len(name) < 3:
        if split:
            return (name, "", "", "")
        else:
            return name

    is_set = False
    prefix = replace = previous = suffix = number = str()

    # /* We first check the case with a .### extension, let's find the last period */
    if (Is.digit(name[-1]) and ("." in name)):
        index = name.rsplit(".", 1)
        if Is.digit(index[1]):
            name = index[0]
            number = '.' + index[1]
        del index

    # /* first case; separator . - _ with extensions r R l L  */
    if ((len(name) > 1) and (name[-2] in (' .-_'))):
        is_set = True
        previous = name[-2:]
        sep = name[-2]
        if name[-1] == 'l':
            prefix = name[:-2]
            replace = sep + 'r'
        elif name[-1] == 'r':
            prefix = name[:-2]
            replace = sep + 'l'
        elif name[-1] == 'L':
            prefix = name[:-2]
            replace = sep + 'R'
        elif name[-1] == 'R':
            prefix = name[:-2]
            replace = sep + 'L'
        else:
            is_set = False
            previous = ""

    # /* case; beginning with r R l L, with separator after it */
    if ((not is_set) and (len(name) > 1) and (name[1] in (' .-_'))):
        is_set = True
        previous = name[:2]
        sep = name[1]
        if name[0] == 'l':
            replace = 'r' + sep
            suffix = name[2:]
        elif name[0] == 'r':
            replace = 'l' + sep
            suffix = name[2:]
        elif name[0] == 'L':
            replace = 'R' + sep
            suffix = name[2:]
        elif name[0] == 'R':
            replace = 'L' + sep
            suffix = name[2:]
        else:
            is_set = False
            previous = ""

    if (not is_set):
        prefix = name

    if (not is_set and len(name) > 5):
        # /* hrms, why test for a separator? lets do the rule 'ultimate left or right' */
        if name.lower().startswith("right") or name.lower().endswith("right"):
            index = name.lower().index("right")
            replace = name[index:index + 5]
            previous = replace
            (prefix, suffix) = name.split(replace, 1)
            if replace[0] == "r":
                replace = "left"
            elif replace[1] == "I":
                replace = "LEFT"
            else:
                replace = "Left"
        elif name.lower().startswith("left") or name.lower().endswith("left"):
            index = name.lower().index("left")
            replace = name[index:index + 4]
            previous = replace
            (prefix, suffix) = name.split(replace, 1)
            if replace[0] == "l":
                replace = "right"
            elif replace[1] == "E":
                replace = "RIGHT"
            else:
                replace = "Right"

    if only_split:
        return (prefix, previous, suffix, number)
    elif split:
        return (prefix, replace, suffix, number)
    else:
        return prefix + replace + suffix + number

def layer(*ins, max=32):
    """Get a layer array with only the specified layers enabled"""

    layers = [False] * max
    for i in ins:
        layers[i] = True

    return tuple(layers)

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

def matrix_from_tuple(tuple):
    from mathutils import Matrix
    if len(tuple) == 16:
        return Matrix((tuple[0:4], tuple[4:8], tuple[8:12], tuple[12:16]))

def matrix_to_tuple(matrix):
    return tuple(y for x in matrix.col for y in x)
    # return tuple(y for x in matrix for y in x)

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

def multiply_matrix(*matrices):
    from mathutils import Matrix, Vector

    merge = Matrix()
    for mat in matrices:
        sym = '*' if Is.digit(mat) else '@'
        merge = eval(f'{merge!r} {sym} {mat!r}')

    return merge

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

    # return context.area.type != 'VIEW_3D'
    return context.area.type == 'PROPERTIES'

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

def register_collection(cls, **kwargs):
    """
    register a class, then return a pointer of a Pointer property\\
    kwargs are optional additional paramenter to insert in the type\\
    """

    if hasattr(cls, 'is_registered') and (not cls.is_registered):
        bpy.utils.register_class(cls)
    cls_registered = bpy.props.CollectionProperty(type=cls, **kwargs)

    return cls_registered

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

    def looper(*args, **keywords):
        try:
            exit = function(*args, **keywords)  # Digit for new wait
            if use_threading:
                while exit is not None:
                    if exit is not None:
                        time.sleep(exit)
                        exit = function(*args, **keywords)  # Digit for new wait
            else:
                return (exit, None)[exit is None]
        except:
            utils.error(
                f"\n\tError with register_timer({function}) @ line#" +
                utils.line(),
            )
            return

    if use_threading:
        timer = threading.Timer(
            wait, looper, args=args, kwargs=keywords)
        timer.start()
    else:
        bpy.app.timers.register(
            functools.partial(looper, *args, **keywords),
            first_interval=wait)

def rotate_matrix(matrix, angles_in_degrees: "float or tuple (x, y, z)"):
    from math import radians
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

def string(*args):
    """return a list of items as a merged string"""

    string = str()
    for i in args:
        string += str(i)

    return string

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


utils = type('', (), globals())
