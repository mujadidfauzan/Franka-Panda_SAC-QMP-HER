import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco

from panda_rl.controllers.ik_controller import DifferentialIK6DController
from panda_rl.utils.mujoco_utils import (
    euler_xyz_to_mat,
    mat_to_quat_wxyz,
    orientation_error_vector,
    quat_wxyz_to_mat,
)


DEFAULT_MODEL_PATH = PROJECT_ROOT / "franka_emika_panda" / "scene_ik_test.xml"
ARM_HOME_QPOS = np.array(
    [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853],
    dtype=np.float64,
)


def reset_to_home(model, data):
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    if home_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = ARM_HOME_QPOS
        if model.nq >= 9:
            data.qpos[7:9] = 0.04
        if model.nu >= 7:
            data.ctrl[:7] = data.qpos[:7]
        if model.nu >= 8:
            data.ctrl[7] = 255

    mujoco.mj_forward(model, data)


def site_pose(model, data, site_id):
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[site_id].copy()
    mat = data.site_xmat[site_id].reshape(3, 3).copy()
    return pos, mat


def resolve_target_mat(args):
    provided = [
        args.target_quat is not None,
        args.target_rpy_deg is not None,
        args.target_rpy_rad is not None,
    ]
    if sum(provided) != 1:
        raise ValueError(
            "Pilih tepat satu orientasi target: "
            "--target-quat, --target-rpy-deg, atau --target-rpy-rad."
        )

    if args.target_quat is not None:
        return quat_wxyz_to_mat(args.target_quat)

    if args.target_rpy_deg is not None:
        return euler_xyz_to_mat(np.deg2rad(args.target_rpy_deg))

    return euler_xyz_to_mat(args.target_rpy_rad)


def sync_target_marker(viewer, target_pos, target_mat):
    target_quat = mat_to_quat_wxyz(target_mat)
    viewer.user_scn.ngeom = 0
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.025, 0.0, 0.0], dtype=np.float64),
        pos=target_pos,
        mat=np.eye(3).reshape(-1),
        rgba=np.array([0.0, 1.0, 0.0, 0.8], dtype=np.float32),
    )
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[1],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=np.array([0.04, 0.005, 0.005], dtype=np.float64),
        pos=target_pos,
        mat=quat_wxyz_to_mat(target_quat).reshape(-1),
        rgba=np.array([1.0, 0.2, 0.1, 0.45], dtype=np.float32),
    )
    viewer.user_scn.ngeom = 2


def maybe_sync_viewer(viewer, target_pos, target_mat, delay):
    if viewer is None:
        return True
    if not viewer.is_running():
        return False
    with viewer.lock():
        sync_target_marker(viewer, target_pos, target_mat)
    viewer.sync()
    time.sleep(max(delay, 0.0))
    return True


def solve_ik(args):
    model = mujoco.MjModel.from_xml_path(str(args.model_path))
    data = mujoco.MjData(model)
    reset_to_home(model, data)

    target_pos = np.asarray(args.target_pos, dtype=np.float64).reshape(3)
    target_mat = resolve_target_mat(args)

    controller = DifferentialIK6DController(
        model=model,
        site_name=args.site_name,
        pos_weight=args.pos_weight,
        ori_weight=args.ori_weight,
        damping=args.damping,
        max_joint_step=args.max_joint_step,
    )

    final_info = None
    converged = False
    viewer = None

    if args.render:
        import mujoco.viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)

    try:
        maybe_sync_viewer(viewer, target_pos, target_mat, args.render_delay)

        for iteration in range(1, args.max_iters + 1):
            current_pos, current_mat = site_pose(model, data, controller.site_id)
            pos_err = target_pos - current_pos
            ori_err = orientation_error_vector(current_mat, target_mat)
            pos_norm = float(np.linalg.norm(pos_err))
            ori_norm = float(np.linalg.norm(ori_err))

            final_info = {
                "iteration": iteration,
                "current_pos": current_pos,
                "current_mat": current_mat,
                "pos_err": pos_err,
                "ori_err": ori_err,
                "pos_norm": pos_norm,
                "ori_norm": ori_norm,
            }

            if pos_norm <= args.pos_tol and ori_norm <= args.ori_tol:
                converged = True
                break

            target_qpos, ik_info = controller.compute_joint_target(
                data=data,
                target_pos=target_pos,
                target_mat=target_mat,
            )
            data.qpos[controller.qpos_ids] = target_qpos
            data.qvel[controller.dof_ids] = 0.0
            if model.nu >= 7:
                data.ctrl[:7] = target_qpos
            if model.nu >= 8:
                data.ctrl[7] = 255
            mujoco.mj_forward(model, data)
            final_info["dq"] = ik_info["dq"]

            if not maybe_sync_viewer(
                viewer,
                target_pos,
                target_mat,
                args.render_delay,
            ):
                break

        current_pos, current_mat = site_pose(model, data, controller.site_id)
        pos_err = target_pos - current_pos
        ori_err = orientation_error_vector(current_mat, target_mat)
        final_info = {
            **(final_info or {}),
            "current_pos": current_pos,
            "current_mat": current_mat,
            "current_quat": mat_to_quat_wxyz(current_mat),
            "target_pos": target_pos,
            "target_quat": mat_to_quat_wxyz(target_mat),
            "pos_err": pos_err,
            "ori_err": ori_err,
            "pos_norm": float(np.linalg.norm(pos_err)),
            "ori_norm": float(np.linalg.norm(ori_err)),
            "qpos": data.qpos[controller.qpos_ids].copy(),
            "qpos_deg": np.rad2deg(data.qpos[controller.qpos_ids].copy()),
            "converged": converged,
        }

        if viewer is not None:
            start_time = time.time()
            while viewer.is_running():
                maybe_sync_viewer(viewer, target_pos, target_mat, 0.01)
                if (
                    args.hold_seconds > 0.0
                    and time.time() - start_time >= args.hold_seconds
                ):
                    break
    finally:
        if viewer is not None:
            viewer.close()

    return model, data, controller, final_info


def print_result(info):
    status = "CONVERGED" if info["converged"] else "NOT CONVERGED"
    print(f"status: {status}")
    print(f"iteration: {info.get('iteration')}")
    print("target_pos:", np.array2string(info["target_pos"], precision=6))
    print("target_quat_wxyz:", np.array2string(info["target_quat"], precision=6))
    print("final_pos :", np.array2string(info["current_pos"], precision=6))
    print("final_quat_wxyz :", np.array2string(info["current_quat"], precision=6))
    print(f"position_error_norm: {info['pos_norm']:.8f} m")
    print(f"orientation_error_norm: {info['ori_norm']:.8f} rad")
    print("qpos_rad:", np.array2string(info["qpos"], precision=6))
    print("qpos_deg:", np.array2string(info["qpos_deg"], precision=3))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--site-name", type=str, default="ee_site")
    parser.add_argument("--target-pos", nargs=3, type=float, required=True)

    orientation_group = parser.add_mutually_exclusive_group(required=True)
    orientation_group.add_argument("--target-rpy-deg", nargs=3, type=float)
    orientation_group.add_argument("--target-rpy-rad", nargs=3, type=float)
    orientation_group.add_argument("--target-quat", nargs=4, type=float)

    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--pos-tol", type=float, default=1e-3)
    parser.add_argument("--ori-tol", type=float, default=np.deg2rad(2.0))
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--ori-weight", type=float, default=0.35)
    parser.add_argument("--damping", type=float, default=0.08)
    parser.add_argument("--max-joint-step", type=float, default=0.08)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-delay", type=float, default=0.03)
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    model, data, _, info = solve_ik(args)
    print_result(info)


if __name__ == "__main__":
    main()
