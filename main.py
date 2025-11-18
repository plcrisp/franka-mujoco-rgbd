import mujoco
import mujoco.viewer
import numpy as np 

import warnings
warnings.filterwarnings("ignore")

XML_PATH = "model/scene.xml"

try:
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
except Exception as e:
    print(f"❌ Erro ao carregar o modelo: {e}")
    print(f"Verifique se o arquivo '{XML_PATH}' existe e se as referências internas estão corretas.")
    exit()

ready_state = np.array([0., -1/4 * np.pi, 0., -3/4 * np.pi, 0., 1/2 * np.pi, 1/4 * np.pi])
m.qpos0[:7] = -ready_state 
mujoco.mj_resetData(m, d)

print("🚀 Franka Panda carregado. Use o mouse para girar o modelo.")
mujoco.viewer.launch(m, d)
