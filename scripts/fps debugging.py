import bpy
from bpy.app.handlers import frame_change_post
from datetime import datetime
now = datetime.now


def wait_for_end(scn):
    fc = scn.frame_current
    if (scn.use_preview_range):
        fs = scn.frame_preview_start
        fe = scn.frame_preview_end
    else:
        fs = scn.frame_start
        fe = scn.frame_end

    if (fc == fe):  # if (fc in {fs, fe}):
        end = now()
        print("\nTime lapsed:", end - start, "")
        frame_change_post.remove(wait_for_end)


start = now()
frame_change_post.append(wait_for_end)
bpy.ops.screen.animation_play()
