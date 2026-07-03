import mujoco
import numpy as np


def mj_name_to_id(model, obj_type, name):
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id == -1:
        raise ValueError(f"MuJoCo object '{name}' was not found.")
    return obj_id


def joint_qpos_ids(model, joint_names):
    return np.array(
        [
            model.jnt_qposadr[
                mj_name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            ]
            for joint_name in joint_names
        ],
        dtype=np.int32,
    )


def joint_dof_ids(model, joint_names):
    return np.array(
        [
            model.jnt_dofadr[
                mj_name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            ]
            for joint_name in joint_names
        ],
        dtype=np.int32,
    )


def actuator_ids(model, actuator_names):
    return np.array(
        [
            mj_name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            for actuator_name in actuator_names
        ],
        dtype=np.int32,
    )


def euler_xyz_to_mat(euler):
    roll, pitch, yaw = np.asarray(euler, dtype=np.float64)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float64,
    )
    rot_y = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return rot_z @ rot_y @ rot_x


def mat_to_quat_wxyz(mat):
    mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    trace = np.trace(mat)

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            [
                0.25 * s,
                (mat[2, 1] - mat[1, 2]) / s,
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[1, 0] - mat[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        quat = np.array(
            [
                (mat[2, 1] - mat[1, 2]) / s,
                0.25 * s,
                (mat[0, 1] + mat[1, 0]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif mat[1, 1] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        quat = np.array(
            [
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[0, 1] + mat[1, 0]) / s,
                0.25 * s,
                (mat[1, 2] + mat[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
        quat = np.array(
            [
                (mat[1, 0] - mat[0, 1]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
                (mat[1, 2] + mat[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )

    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def quat_wxyz_to_mat(quat):
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def orientation_error_vector(current_mat, target_mat):
    rel_mat = target_mat @ current_mat.T
    rel_quat = mat_to_quat_wxyz(rel_mat)
    axis_norm = np.linalg.norm(rel_quat[1:])

    if axis_norm < 1e-8:
        return 2.0 * rel_quat[1:]

    angle = 2.0 * np.arctan2(axis_norm, rel_quat[0])
    return rel_quat[1:] / axis_norm * angle
