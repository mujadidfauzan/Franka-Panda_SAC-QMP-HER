import mujoco
import numpy as np

from panda_rl.utils.mujoco_utils import orientation_error_vector


class DifferentialIK6DController:
    """
    Jacobian-based Differential IK untuk Franka Panda.

    Controller ini mengontrol:
    - posisi end-effector: x, y, z
    - orientasi end-effector: roll, pitch, yaw secara differential

    Output akhirnya adalah target joint position untuk actuator1-actuator7.
    """

    def __init__(
        self,
        model,
        site_name="ee_site",
        joint_names=None,
        pos_weight=1.0,
        ori_weight=0.5,
        damping=0.05,
        max_joint_step=0.04,
    ):
        self.model = model
        self.site_name = site_name

        self.pos_weight = pos_weight
        self.ori_weight = ori_weight
        self.damping = damping
        self.max_joint_step = max_joint_step

        if joint_names is None:
            joint_names = [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint7",
            ]

        self.joint_names = joint_names

        self.site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.site_name,
        )

        if self.site_id == -1:
            raise ValueError(f"Site '{self.site_name}' tidak ditemukan.")

        self.qpos_ids = []
        self.dof_ids = []
        self.joint_ranges = []

        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if joint_id == -1:
                raise ValueError(f"Joint '{joint_name}' tidak ditemukan.")

            self.qpos_ids.append(self.model.jnt_qposadr[joint_id])
            self.dof_ids.append(self.model.jnt_dofadr[joint_id])
            self.joint_ranges.append(self.model.jnt_range[joint_id].copy())

        self.qpos_ids = np.array(self.qpos_ids, dtype=np.int32)
        self.dof_ids = np.array(self.dof_ids, dtype=np.int32)
        self.joint_ranges = np.array(self.joint_ranges, dtype=np.float64)

    def compute_joint_target(self, data, target_pos, target_mat):
        """
        Hitung target joint position berdasarkan target pose 6D.

        Parameters
        ----------
        data:
            mujoco.MjData

        target_pos:
            np.array shape (3,)
            Posisi target end-effector dalam world frame.

        target_mat:
            np.array shape (3, 3)
            Orientasi target end-effector dalam world frame.

        Returns
        -------
        target_qpos:
            np.array shape (7,)
            Target posisi joint untuk actuator1-actuator7.

        info:
            dict debug.
        """

        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_mat = np.asarray(target_mat, dtype=np.float64).reshape(3, 3)

        mujoco.mj_forward(self.model, data)

        current_pos = data.site_xpos[self.site_id].copy()
        current_mat = data.site_xmat[self.site_id].reshape(3, 3).copy()

        pos_err = target_pos - current_pos
        ori_err = orientation_error_vector(current_mat, target_mat)

        error_6d = np.concatenate(
            [
                self.pos_weight * pos_err,
                self.ori_weight * ori_err,
            ]
        )

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        mujoco.mj_jacSite(
            self.model,
            data,
            jacp,
            jacr,
            self.site_id,
        )

        J_pos = jacp[:, self.dof_ids]
        J_ori = jacr[:, self.dof_ids]

        J_6d = np.vstack(
            [
                self.pos_weight * J_pos,
                self.ori_weight * J_ori,
            ]
        )

        # Damped Least Squares:
        # dq = J.T @ inv(J @ J.T + lambda^2 I) @ error
        damping_matrix = (self.damping**2) * np.eye(6)

        dq = J_6d.T @ np.linalg.solve(
            J_6d @ J_6d.T + damping_matrix,
            error_6d,
        )

        dq = np.clip(
            dq,
            -self.max_joint_step,
            self.max_joint_step,
        )

        current_qpos = data.qpos[self.qpos_ids].copy()
        target_qpos = current_qpos + dq

        lower = self.joint_ranges[:, 0]
        upper = self.joint_ranges[:, 1]

        target_qpos = np.clip(target_qpos, lower, upper)

        info = {
            "current_pos": current_pos,
            "target_pos": target_pos,
            "pos_err": pos_err,
            "ori_err": ori_err,
            "dq": dq,
            "target_qpos": target_qpos,
        }

        return target_qpos, info
