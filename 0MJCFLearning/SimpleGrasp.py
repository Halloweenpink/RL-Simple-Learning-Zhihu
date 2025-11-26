import mujoco.viewer

xml = r"""
<mujoco model="grasp_demo_clean">
  <compiler angle="radian" coordinate="local" autolimits="true"/>
  <option gravity="0 0 -9.81"/>
    
  <default>
    <geom type="box" density="500" friction="1.5 0.1 0.1"/>
    <joint damping="1"/>
  </default>

  <worldbody>
    <!-- 桌面 -->
    <geom name="table" type="box" size="0.3 0.3 0.02"
          pos="0 0 -0.02" rgba="0.8 0.8 0.8 1"/>

    <!-- 要被抓的小方块 -->
    <body name="block" pos="0 0 0.02">
      <joint name="block_free" type="free"/>
      <geom name="block_geom" size="0.02 0.02 0.02" rgba="0.9 0.3 0.3 1"/>
    </body>

    <!-- 悬空两指夹爪基座 -->
    <body name="gripper_base" pos="0 0 0.12">
      <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>

      <!-- 上下滑动关节 -->
      <joint name="gripper_z" type="slide" axis="0 0 1"
             limited="true" range="-0.08 0.1"/>

      <!-- 左指：沿 +x 方向滑动 -->
      <body name="finger_left" pos="0.04 0 0">
        <joint name="finger_left_slide" type="slide" axis="1 0 0"
               limited="true" range="-0.1 0.1"/>
        <geom name="finger_left_geom" size="0.005 0.015 0.02"
              pos="0 0 -0.04" rgba="0.2 0.6 0.9 1"/>
      </body>

      <!-- 右指：沿 -x 方向滑动 -->
      <body name="finger_right" pos="-0.04 0 0">
        <joint name="finger_right_slide" type="slide" axis="-1 0 0"
               limited="true" range="-0.1 0.1"/>
        <geom name="finger_right_geom" size="0.005 0.015 0.02"
              pos="0 0 -0.04" rgba="0.2 0.6 0.9 1"/>
      </body>
    </body>
  </worldbody>

  <!-- 三个马达 -->
  <actuator>
    <motor name="act_gripper_z" joint="gripper_z" ctrlrange="-1 1" gear="1"/>
    <motor name="act_finger_left" joint="finger_left_slide" ctrlrange="-5 1" gear="1"/>
    <motor name="act_finger_right" joint="finger_right_slide" ctrlrange="-5 1" gear="1"/>
  </actuator>
</mujoco>
"""

# ======== 载入模型 ========
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)


# ======== 主循环：下压 -> 夹紧 -> 抬起 ========
with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():

        mujoco.mj_step(model, data)
        viewer.sync()
