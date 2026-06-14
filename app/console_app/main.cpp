#include "matrix/matrix.hpp"
#include "ml/mnist_loader.hpp"
#include "ml/neural_network.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace ml;
using namespace ml::math;

void print_usage(const char *program) {
  std::cout << "Usage:\n"
            << "  Train:  " << program
            << " train <data_dir> <model_path> [epochs] [lr]\n"
            << "  Predict: " << program
            << " predict <model_path> <image_path>\n"
            << "\nExamples:\n"
            << "  " << program << " train ../data model.bin 20 0.001\n"
            << "  " << program << " predict model.bin digit.pgm\n";
}

Matrix load_pgm_image(const std::string &filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open image: " + filepath);
  }

  std::string format;
  int width, height, max_val;
  file >> format >> width >> height >> max_val;
  file.get(); // skip whitespace

  if (format != "P5") {
    throw std::runtime_error("Only PGM P5 format supported");
  }

  Matrix img(784, 1);
  for (int i = 0; i < 784; ++i) {
    unsigned char pixel;
    file.read(reinterpret_cast<char *>(&pixel), 1);
    img(i, 0) = 1.0 - static_cast<double>(pixel) / 255.0;
  }
  return img;
}

void train(const std::string &data_dir, const std::string &model_path,
           size_t epochs, double lr) {
  std::cout << "Loading MNIST dataset...\n";

  auto dataset =
      MNISTLoader::load_dataset(data_dir + "/train-images-idx3-ubyte",
                                data_dir + "/train-labels-idx1-ubyte", 60000);

  auto targets = MNISTLoader::one_hot_encode(dataset.labels);

  std::cout << "Loaded " << dataset.size() << " examples\n";

  std::vector<LayerConfig> config = {
      {784, 128, "relu"}, {128, 64, "relu"}, {64, 10, "softmax"}};

  NeuralNetwork network(config);
  std::cout << "Parameters: " << network.num_parameters() << "\n";
  std::cout << "Training " << epochs << " epochs, lr=" << lr << "...\n\n";

  network.train(dataset.images, targets, epochs, lr, 64,
                [](const TrainingMetrics &m) {
                  std::cout << "Epoch " << m.epoch << " | Loss: " << m.loss
                            << " | Accuracy: " << m.accuracy * 100 << "%\n";
                });

  network.save(model_path);
  std::cout << "\nModel saved to " << model_path << "\n";
}

void predict(const std::string &model_path, const std::string &image_path) {
  std::vector<LayerConfig> config = {
      {784, 128, "relu"}, {128, 64, "relu"}, {64, 10, "softmax"}};

  NeuralNetwork network(config);
  network.load(model_path);

  Matrix image = load_pgm_image(image_path);
  int prediction = network.predict(image);
  auto probabilities = network.predict_proba(image);

  std::cout << "\nPredicted digit: " << prediction << "\n\n";
  std::cout << "Probabilities:\n";
  for (int i = 0; i < 10; ++i) {
    int bar = static_cast<int>(probabilities[i] * 50);
    std::cout << "  " << i << ": " << std::string(bar, '#')
              << std::string(50 - bar, ' ') << " " << probabilities[i] * 100
              << "%\n";
  }
}

int main(int argc, char *argv[]) {
  try {
    if (argc < 2) {
      print_usage(argv[0]);
      return 1;
    }

    std::string mode = argv[1];

    if (mode == "train") {
      if (argc < 4) {
        print_usage(argv[0]);
        return 1;
      }
      std::string data_dir = argv[2];
      std::string model_path = argv[3];
      size_t epochs = argc > 4 ? std::stoul(argv[4]) : 20;
      double lr = argc > 5 ? std::stod(argv[5]) : 0.001;
      train(data_dir, model_path, epochs, lr);

    } else if (mode == "predict") {
      if (argc < 4) {
        print_usage(argv[0]);
        return 1;
      }
      std::string model_path = argv[2];
      std::string image_path = argv[3];
      predict(model_path, image_path);

    } else {
      std::cerr << "Unknown mode: " << mode << "\n";
      print_usage(argv[0]);
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}