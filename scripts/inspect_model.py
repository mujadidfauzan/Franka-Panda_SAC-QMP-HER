import mujoco

MODEL_PATH = "/home/fauzan/Robot/Panda-SAC_QMP/franka_emika_panda/scene_rl.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)

print("=== JOINTS ===")
for i in range(model.njnt):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

print("\n=== ACTUATORS ===")
for i in range(model.nu):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

print("\n=== SITES ===")
for i in range(model.nsite):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i))

print("\nnq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)
