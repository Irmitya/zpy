import bpy


class register_keymaps():
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

        from zpy import utils
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
