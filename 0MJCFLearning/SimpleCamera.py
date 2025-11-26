import mujoco
import numpy as np
import cv2

xml = r"""
<mujoco model="cam_demo">
  <worldbody>
    <geom type="plane" size="1 1 0.1" rgba="0.8 0.9 0.8 1"/>

    <body name="ball" pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
    </body>
    
    <body name="stick" pos="0 0 0.1">
      <geom type="capsule" size="0.08" fromto="0 0 0 0.5 0.5 1.5" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
    </body>

    <!-- 固定相机 -->
    <camera name="cam1" pos="1 1 3.1" euler="-18.435 17.5484 30" fovy="60"/>
 
    <!-- 欧拉角，mujoco 默认xyz，右手定则，先后顺序转，默认向下-->
  </worldbody>
</mujoco>
"""
# -------------------------
# 1) 解析模型
# -------------------------
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# -------------------------
# 2) 创建 renderer（高层封装 = GLFW + MjvScene + MjrContext）
# -------------------------
renderer = mujoco.Renderer(model, width=640, height=480)  # 相机的分辨率是 640×480。

# 创建 MjvCamera，指定使用 XML 里的 cam1
camera = mujoco.MjvCamera()

camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
'''mjCAMERA_FREE：自由相机（可以随便平移/旋转）
mjCAMERA_FIXED：使用 MuJoCo 模型里的某个 <camera>
mjCAMERA_TRACKING：跟踪某个 body（用于第三人称跟随）'''
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam1")
# mj_name2id(model, kind, name)：根据名字查 id 的工具函数。cam_id 是一个 int，例如 0、1、2……
# 之后只要你写 camera.fixedcamid = cam_id，这个 MjvCamera 就会使用 XML 里那个 cam1 的位姿（pos/euler/quat/fovy）
camera.fixedcamid = cam_id

# ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
# camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
# camera.trackbodyid = ball_id     # 跟踪 ball

# -------------------------
# 3) 主循环
# -------------------------
while True:
    mujoco.mj_step(model, data)

    # update_scene = 把世界摆到相机面前
    renderer.update_scene(data, camera)

    rgb = renderer.render()   # 注意：不能写 camera_id
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # BGR（Blue-Green-Red，蓝 - 绿 - 红）历史问题
    cv2.imshow("MuJoCo Camera Demo", bgr)

    if cv2.waitKey(1) == 27:  # ESC的 ASCII 码

        break

renderer.close()
cv2.destroyAllWindows()
