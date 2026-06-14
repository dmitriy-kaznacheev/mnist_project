#include "ml/mnist_loader.hpp"
#include <algorithm>
#include <fstream>
#include <random>
#include <stdexcept>

namespace ml {

MNISTLoader::Dataset MNISTLoader::load_dataset(const std::string &images_path,
                                               const std::string &labels_path,
                                               size_t max_samples) {
  Dataset dataset;

  std::ifstream img_file(images_path, std::ios::binary);
  std::ifstream lbl_file(labels_path, std::ios::binary);

  if (!img_file.is_open() || !lbl_file.is_open()) {
    throw std::runtime_error("Cannot open MNIST files");
  }

  // чтение заголовков
  uint32_t magic_img = read_uint32_big_endian(img_file);
  uint32_t num_images = read_uint32_big_endian(img_file);
  uint32_t rows = read_uint32_big_endian(img_file);
  uint32_t cols = read_uint32_big_endian(img_file);

  uint32_t magic_lbl = read_uint32_big_endian(lbl_file);
  uint32_t num_labels = read_uint32_big_endian(lbl_file);

  size_t count = max_samples > 0
                     ? std::min(static_cast<size_t>(num_images), max_samples)
                     : num_images;

  for (size_t i = 0; i < count; ++i) {
    math::Matrix img(rows * cols, 1);

    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        unsigned char pixel;
        img_file.read(reinterpret_cast<char *>(&pixel), sizeof(pixel));
        img(r * cols + c, 0) =
            (static_cast<double>(pixel) / 255.0 - 0.5) * 2.0; // [-1, 1]
      }
    }

    unsigned char label;
    lbl_file.read(reinterpret_cast<char *>(&label), sizeof(label));

    dataset.images.push_back(std::move(img));
    dataset.labels.push_back(static_cast<int>(label));
  }

  return dataset;
}

std::vector<math::Matrix>
MNISTLoader::one_hot_encode(const std::vector<int> &labels,
                            size_t num_classes) {
  std::vector<math::Matrix> encoded;
  encoded.reserve(labels.size());

  for (int label : labels) {
    math::Matrix vec(num_classes, 1, 0.0);
    vec(label, 0) = 1.0;
    encoded.push_back(std::move(vec));
  }

  return encoded;
}

std::pair<MNISTLoader::Dataset, MNISTLoader::Dataset>
MNISTLoader::train_test_split(const Dataset &dataset, double train_ratio) {
  std::vector<size_t> indices(dataset.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(indices.begin(), indices.end(), gen);

  size_t train_size = static_cast<size_t>(dataset.size() * train_ratio);

  Dataset train, test;

  for (size_t i = 0; i < train_size; ++i) {
    train.images.push_back(dataset.images[indices[i]]);
    train.labels.push_back(dataset.labels[indices[i]]);
  }

  for (size_t i = train_size; i < dataset.size(); ++i) {
    test.images.push_back(dataset.images[indices[i]]);
    test.labels.push_back(dataset.labels[indices[i]]);
  }

  return {std::move(train), std::move(test)};
}

uint32_t MNISTLoader::read_uint32_big_endian(std::ifstream &file) {
  uint32_t value = 0;
  file.read(reinterpret_cast<char *>(&value), sizeof(value));

  // из big-endian в little-endian
  return ((value & 0xFF000000) >> 24) | ((value & 0x00FF0000) >> 8) |
         ((value & 0x0000FF00) << 8) | ((value & 0x000000FF) << 24);
}

} // namespace ml