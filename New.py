"Functions to create data"
import bpy
from zpy import Get, Is, New, Set


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

def driver(src, path, **kargs):
    driver_type = kargs.get('driver_type', None)
        # 'AVERAGE', 'Sum Values', 'SCRIPTED', 'Minimum Value', 'Maximum Value
    expression = kargs.get('expression', None)
    frames = kargs.get('frames', list())  # keyframe.co for the driver's fcurve
    name = kargs.get('name', "var")  # Name of the variable added to the driver
    overwrite = kargs.get('overwrite', False)  # Delete the existing driver
    rotation_mode = kargs.get('rotation_mode', 'AUTO')
    target = kargs.get('target', None)
    target_path = kargs.get('target_path', '')
    transform_space = kargs.get('transform_space', 'LOCAL_SPACE')
    transform_type = kargs.get('transform_type', 'LOC_X')
    var_type = kargs.get('var_type', None)
        # 'SINGLE_PROP', 'TRANSFORMS', 'Rotational Difference', 'Distance'
    if var_type is None:
        if target and (not target_path):
            var_type = 'TRANSFORMS'
        else:
            var_type = 'SINGLE_PROP'

    Driver = Get.driver(src, path)

    if not Driver:
        Driver = src.driver_add(path)
        overwrite = True

    if overwrite:
        while Driver.keyframe_points:
            Driver.keyframe_points.remove(Driver.keyframe_points[0])

    if frames:
        if overwrite:
            Driver.extrapolation = 'LINEAR'
            while Driver.modifiers:
                Driver.modifiers.remove(Driver.modifiers[0])
        Driver.keyframe_points.add(len(frames))
        for key, co in zip(Driver.keyframe_points[:], frames):
            key.interpolation = 'LINEAR'
            key.co = co

    driver = Driver.driver

    if overwrite:
        if (expression is None):
            if (driver_type is None):
                driver_type = 'AVERAGE'
            elif (driver.type == 'SCRIPTED'):
                driver.expression = name

        while driver.variables:
            driver.variables.remove(driver.variables[0])

    if expression is not None:
        driver.expression = expression
    if driver_type:
        driver.type = driver_type

    var = driver.variables.new()
    var.name = name
    var.type = var_type
    var_target = var.targets[0]

    if target:
        is_pose = Is.posebone(target)
        is_bone = Is.bone(target) or Is.editbone(target)
        is_obj = Is.object(target)

        if is_obj:
            var_target.id = target
        elif (is_pose or is_bone):
            var_target.id = target.id_data
            var_target.bone_target = target.name
            if target_path and (not target_path.startswith(('pose.bones', 'bones'))):
                if is_pose:
                    text = f'pose.bones["{target.name}"]'
                else:
                    text = f'bones["{target.name}"]'

                if (target_path[0] != '['):
                    text += '.'

                target_path = text + target_path
        else:
            try:
                var_target.id = target
            except:
                var_target.id = target.id_data

    var_target.data_path = target_path
    var_target.rotation_mode = rotation_mode
    var_target.transform_space = transform_space
    var_target.transform_type = transform_type

    return Driver


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

def bone(context, armature, name="", edit=None, pose=None, overwrite=False):
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
    children = list()
    if overwrite and name in ebones:
        for child in ebones[name].children:
            children.append((child, child.use_connect))
            child.use_connect = False
        ebones.remove(ebones[name])
    bone = ebones.new(name)
    for child, child_connect in children:
        child.parent = bone
        child.use_connect = child_connect
    bone.tail = ((0, 0, 1))
        # If the bone's head AND tail stay at 0,
        # it gets deleted when leaving edit mode
    name = bone.name
    if edit:
        edit(bone)
    armature.data.use_mirror_x = mirror

    pbones = armature.pose.bones
    if pose:
        Set.mode(context, armature, 'POSE')
        bone = pbones[name]
        pose(bone)

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
        from zpy import utils
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


New = type('', (), globals())
