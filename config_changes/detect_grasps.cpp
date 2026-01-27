#include <string>

#include <gpd/grasp_detector.h>

namespace gpd {
namespace apps {
namespace detect_grasps {

bool checkFileExists(const std::string &file_name) {
  std::ifstream file;
  file.open(file_name.c_str());
  if (!file) {
    std::cout << "File " + file_name + " could not be found!\n";
    return false;
  }
  file.close();
  return true;
}

int DoMain(int argc, char *argv[]) {
  // Read arguments from command line.
  if (argc < 3) {
    std::cout << "Error: Not enough input arguments!\n\n";
    std::cout << "Usage: detect_grasps CONFIG_FILE PCD_FILE [NORMALS_FILE]\n\n";
    std::cout << "Detect grasp poses for a point cloud, PCD_FILE (*.pcd), "
                 "using parameters from CONFIG_FILE (*.cfg).\n\n";
    std::cout << "[NORMALS_FILE] (optional) contains a surface normal for each "
                 "point in the cloud (*.csv).\n";
    return (-1);
  }

  std::string config_filename = argv[1];
  std::string pcd_filename = argv[2];
  if (!checkFileExists(config_filename)) {
    printf("Error: config file not found!\n");
    return (-1);
  }
  if (!checkFileExists(pcd_filename)) {
    printf("Error: PCD file not found!\n");
    return (-1);
  }

  // Read parameters from configuration file.
  util::ConfigFile config_file(config_filename);
  config_file.ExtractKeys();

  // Set the camera position. Assumes a single camera view.
  std::vector<double> camera_position =
      config_file.getValueOfKeyAsStdVectorDouble("camera_position",
                                                 "0.0 0.0 0.0");
  Eigen::Matrix3Xd view_points(3, 1);
  view_points << camera_position[0], camera_position[1], camera_position[2];

  // Load point cloud from file.
  util::Cloud cloud(pcd_filename, view_points);
  if (cloud.getCloudOriginal()->size() == 0) {
    std::cout << "Error: Input point cloud is empty or does not exist!\n";
    return (-1);
  }

  // Load surface normals from file.
  if (argc > 3) {
    std::string normals_filename = argv[3];
    cloud.setNormalsFromFile(normals_filename);
    std::cout << "Loaded surface normals from file: " << normals_filename
              << "\n";
  }

  GraspDetector detector(config_filename);

  // Preprocess the point cloud.
  detector.preprocessPointCloud(cloud);

  // If the object is centered at the origin, reverse all surface normals.
  bool centered_at_origin =
      config_file.getValueOfKey<bool>("centered_at_origin", false);
  if (centered_at_origin) {
    printf("Reversing normal directions ...\n");
    cloud.setNormals(cloud.getNormals() * (-1.0));
  }

  // Detect grasp poses.
  auto grasps = detector.detectGrasps(cloud);

  // MUDANÇA: Loop para imprimir as coordenadas EXATAS para o Python ler
  std::cout << "======== Selected grasps (CUSTOM OUTPUT) ========" << std::endl;
  
  for (size_t i = 0; i < grasps.size(); i++) {
      // Nota: Dependendo da versão do GPD, pode ser grasps[i]-> ou grasps[i].
      // Se der erro de compilação, troque '->' por '.'
      
      std::cout << "Grasp " << i << " Score: " << grasps[i]->getScore() << std::endl;

      // Imprime Posição (X Y Z)
      const auto& pos = grasps[i]->getSample();
      std::cout << "DATA_POS:" 
                << pos.x() << " " << pos.y() << " " << pos.z() << std::endl;

      // Imprime Rotação (Matriz 3x3)
      const auto& rot = grasps[i]->getFrame();
      std::cout << "DATA_ROT:" 
                << rot(0,0) << " " << rot(0,1) << " " << rot(0,2) << " "
                << rot(1,0) << " " << rot(1,1) << " " << rot(1,2) << " "
                << rot(2,0) << " " << rot(2,1) << " " << rot(2,2) << std::endl;
  }

  return 0;
}

}  // namespace detect_grasps
}  // namespace apps
}  // namespace gpd

int main(int argc, char *argv[]) {
  return gpd::apps::detect_grasps::DoMain(argc, argv);
}
