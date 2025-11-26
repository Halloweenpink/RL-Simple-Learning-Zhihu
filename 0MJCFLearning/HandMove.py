import time

import mujoco.viewer

# model = mujoco.MjModel.from_xml_path("vx300s.xml")
model = mujoco.MjModel.from_xml_path("HandPose.xml")
# model = mujoco.MjModel.from_xml_path("test_torque.xml")
# model = mujoco.MjModel.from_xml_path("SimpleHand.xml")
data = mujoco.MjData(model)


# 加载关键帧
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")  # 1) 通过名字找到 keyframe 的 id
mujoco.mj_resetDataKeyframe(model, data, key_id)  # 2) 把这个 keyframe 拷贝到 data （qpos/ctrl 等都会被重置）
mujoco.mj_forward(model, data)  # 3) forward，暴力改完位置信息后，重新计算物理量

with mujoco.viewer.launch(model, data) as viewer:
    viewer.renderer.scene.option.bgcolor[:] = [1.0, 1.0, 1.0]

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.1)