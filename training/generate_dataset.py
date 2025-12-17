import os
import time
import shutil
import random
import mujoco
import mujoco.viewer
import numpy as np
import cv2

# --- CONFIG ---
XML_PATH = "../model/scene.xml" # Ajuste o caminho relativo!
CAMERA_NAME = "end_effector_camera"
OUTPUT_DIR = "dataset_raw"      # Pasta temporária
FINAL_DIR = "dataset_final"     # Pasta pronta para o YOLO

CLASS_MAPPING = {
    0: {"name": "hammer", "radius": 0.15},
    1: {"name": "mug", "radius": 0.08},
    2: {"name": "bottle", "radius": 0.12}
}

# --- FUNÇÃO DE PROJEÇÃO (Sua lógica original) ---
def get_pixel_coord(cam_pos, cam_mat, obj_pos, f, cx, cy):
    res = obj_pos - cam_pos
    cam_coord = cam_mat.T @ res
    x_cv, y_cv, z_cv = cam_coord[0], -cam_coord[1], -cam_coord[2]
    if z_cv <= 0.01: return None
    u = f * (x_cv / z_cv) + cx
    v = f * (y_cv / z_cv) + cy
    return u, v, z_cv

# --- FUNÇÃO DE SPLIT (Sua lógica original melhorada) ---
def split_data():
    print("🔄 Organizando dataset em Train/Valid...")
    
    # Cria estrutura limpa
    if os.path.exists(FINAL_DIR): shutil.rmtree(FINAL_DIR)
    for split in ['train', 'valid']:
        for kind in ['images', 'labels']:
            os.makedirs(f"{FINAL_DIR}/{split}/{kind}", exist_ok=True)

    # Lista imagens
    img_path = f"{OUTPUT_DIR}/images"
    if not os.path.exists(img_path):
        print("❌ Nenhuma imagem gerada.")
        return

    images = [f for f in os.listdir(img_path) if f.endswith(".png")]
    random.shuffle(images)

    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    def move_files(file_list, split_name):
        for img in file_list:
            # Copia Imagem
            src_img = f"{OUTPUT_DIR}/images/{img}"
            dst_img = f"{FINAL_DIR}/{split_name}/images/{img}"
            shutil.copy(src_img, dst_img)
            
            # Copia Label
            lbl = img.replace(".png", ".txt")
            src_lbl = f"{OUTPUT_DIR}/labels/{lbl}"
            dst_lbl = f"{FINAL_DIR}/{split_name}/labels/{lbl}"
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dst_lbl)

    move_files(train_imgs, "train")
    move_files(val_imgs, "valid")
    print(f"✅ Dataset pronto em '{FINAL_DIR}'! Pode rodar o treino.")

# --- MAIN LOOP ---
def main():
    if not os.path.exists(XML_PATH):
        print(f"Erro: Não achei {XML_PATH}. Rode script da pasta 'training'.")
        return

    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, 720, 1280)

    # Limpa/Cria pasta temporária
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

    cam_id = m.camera(CAMERA_NAME).id
    fovy = m.camera(CAMERA_NAME).fovy 
    H, W = 720, 1280
    f = 0.5 * H / np.tan(0.5 * fovy * np.pi/180)    
    cx, cy = W/2, H/2   
    
    last_photo = time.time()
    print("📸 Iniciando Captura... Feche a janela do MuJoCo para processar os dados.")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(m, d)
            viewer.sync()

            renderer.update_scene(d, camera=CAMERA_NAME)
            rgb = renderer.render()
            rgb_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Lógica de Captura (A cada 0.5s para não repetir muito)
            if time.time() - last_photo >= 0.5:
                cam_pos = d.cam_xpos[cam_id]
                cam_mat = d.cam_xmat[cam_id].reshape(3, 3)
                labels = []
                
                # Desenho Visual
                for cls_id, props in CLASS_MAPPING.items():
                    obj_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, props["name"])
                    if obj_id != -1:
                        res = get_pixel_coord(cam_pos, cam_mat, d.xpos[obj_id], f, cx, cy)
                        if res:
                            u, v, depth = res
                            if 0 <= u < W and 0 <= v < H:
                                box_size = int(f * 2 * props["radius"] / depth)
                                # YOLO Format
                                x_c, y_c = u/W, v/H
                                w_n, h_n = (box_size*2)/W, (box_size*2)/H
                                labels.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
                                
                                # Desenha Box
                                p1 = (int(u - box_size), int(v - box_size))
                                p2 = (int(u + box_size), int(v + box_size))
                                cv2.rectangle(rgb_vis, p1, p2, (0, 255, 0), 2)

                # Salva se detectou algo
                if labels:
                    t = str(time.time()).replace('.', '')
                    cv2.imwrite(f"{OUTPUT_DIR}/images/{t}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    with open(f"{OUTPUT_DIR}/labels/{t}.txt", "w") as f_txt:
                        f_txt.write("\n".join(labels))
                    last_photo = time.time()

            cv2.imshow("Generator View", rgb_vis)
            cv2.waitKey(1)
            
            # Mantém tempo real
            time_until_next = m.opt.timestep - (time.time() - step_start)
            if time_until_next > 0: time.sleep(time_until_next)

    cv2.destroyAllWindows()
    # RODA O SPLIT AO FECHAR
    split_data()

if __name__ == "__main__":
    main()