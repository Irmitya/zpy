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


cpp = type('', (), globals())
