import time

import mujoco
import mujoco.viewer
import numpy as np

from panda_rl.controllers.ik_controller import DifferentialIK6DController

MODEL_PATH = "/home/fauzan/Robot/Panda-SAC_QMP/franka_emika_panda/scene_ik_test.xml"


def get_body_pose(model, data, body_name):
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if body_id == -1:
        raise ValueError(f"Body '{body_name}' tidak ditemukan.")

    pos = data.xpos[body_id].copy()
    mat = data.xmat[body_id].reshape(3, 3).copy()

    return pos, mat


def reset_to_home(model, data):
    home_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if home_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)

        data.qpos[:7] = np.array(
            [
                0.0,
                0.0,
                0.0,
                -1.57079,
                0.0,
                1.57079,
                -0.7853,
            ]
        )

        data.qpos[7:9] = 0.04
        data.ctrl[:7] = data.qpos[:7]
        data.ctrl[7] = 255

    mujoco.mj_forward(model, data)


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    reset_to_home(model, data)

    controller = DifferentialIK6DController(
        model=model,
        site_name="ee_site",
        pos_weight=1.0,
        ori_weight=0.35,
        damping=0.08,
        max_joint_step=0.035,
    )

    target_body_name = "target_body"

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_count = 0

        while viewer.is_running():
            step_count += 1

            target_pos, target_mat = get_body_pose(
                model,
                data,
                target_body_name,
            )

            target_qpos, info = controller.compute_joint_target(
                data=data,
                target_pos=target_pos,
                target_mat=target_mat,
            )

            # actuator1-actuator7 untuk arm Panda
            data.ctrl[:7] = target_qpos

            # actuator8 untuk gripper
            # 255 = buka
            # 0 = tutup
            data.ctrl[7] = 255

            for _ in range(5):
                mujoco.mj_step(model, data)

            viewer.sync()

            if step_count % 50 == 0:
                pos_dist = np.linalg.norm(info["pos_err"])
                ori_dist = np.linalg.norm(info["ori_err"])

                print("step:", step_count)
                print("current_pos:", info["current_pos"])
                print("target_pos :", info["target_pos"])
                print("pos_error  :", pos_dist)
                print("ori_error  :", ori_dist)
                print("dq         :", info["dq"])
                print("-" * 60)

            time.sleep(0.01)


if __name__ == "__main__":
    main()
