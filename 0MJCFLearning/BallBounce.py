import mujoco
import mujoco.viewer

xml = """
<mujoco>
  <!-- 更小的时间步让接触更稳定、更“硬” -->
  <option timestep="0.001" gravity="0 0 -9.81"/>

  <!-- 用 default 让双方（球+地板）都生效 -->
  <default>
    <!-- solref = [timeconst, dampratio]
         更小的 timeconst → 更硬，
         dampratio < 1 → 欠阻尼，出现“弹跳”
    -->

    <geom solref=".0002 0.4" solimp="0.99 0.999 0.0005"/>
  </default>

  <worldbody> 
    <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
      <inertial mass="0.1" pos="0 0 0" diaginertia="0.004 0.004 0.004"/>
    </body>
  </worldbody>
</mujoco>

"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
