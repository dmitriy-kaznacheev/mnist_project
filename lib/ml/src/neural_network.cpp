#include "ml/neural_network.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ml {

NeuralNetwork::NeuralNetwork(const std::vector<LayerConfig> &config) {
  if (config.empty()) {
    throw std::invalid_argument("Network must have at least one layer");
  }
  initialize_weights(config);
}

math::Matrix NeuralNetwork::forward(const math::Matrix &input) {
  layer_inputs_.clear();
  layer_outputs_.clear();

  // текущий сигнал
  math::Matrix current = input;

  // вход сети - layer_inputs_[0]
  layer_inputs_.push_back(current);

  // проход по всем слоям
  for (size_t i = 0; i < weights_.size(); ++i) {
    // линейное преобразование: z = W * x + b
    //   weights_[i] — матрица весов (выходы_слоя x входы_слоя)
    //   current — вход слоя (входы_слоя x 1)
    //   biases_[i] — вектор смещений (выходы_слоя x 1)
    //   z — выход до активации (выходы_слоя x 1)
    math::Matrix z = weights_[i] * current + biases_[i];

    // z нужен будет для вычисления производной активации при backward
    layer_outputs_.push_back(z);

    // функция активации: a = f(z)
    current = activate(z, activations_[i]);

    // сохранить как вход для следующего слоя
    layer_inputs_.push_back(current);
  }

  return current;
}

void NeuralNetwork::backward(const math::Matrix &input,
                             const math::Matrix &target, double learning_rate) {
  // выход сети — последний элемент layer_inputs_
  math::Matrix output = layer_inputs_.back();

  // ошибка на выходе: выход - цель
  math::Matrix error = output - target;

  // градиент ошибки, передаваемый от слоя к слою
  math::Matrix delta;

  // от последнего слоя к первому: i = N-1, N-2, ..., 0
  for (int i = static_cast<int>(weights_.size()) - 1; i >= 0; --i) {
    if (i == static_cast<int>(weights_.size()) - 1) {
      // последний слой
      //   delta = ошибка на выходе (без умножения на производную)
      delta = error;
    } else {
      // скрытый слой
      // 1. ошибка передается от следующего слоя: W_{i+1}^T * delta_{i+1}
      //    weights_[i+1].transpose() имеет размер (входы_слоя_i+1 x
      //    выходы_слоя_i+1) delta имеет размер (выходы_слоя_i+1 x 1) результат:
      //    (входы_слоя_i+1 x 1) = (выходы_слоя_i x 1)
      delta = weights_[i + 1].transpose() * delta;

      // 2. поэлементное умножение на производную активации
      //    layer_outputs_[i] — значения ДО активации (z_i)
      //    f'(z_i) для ReLU: 1 если z>0, иначе 0
      //    для сигмоиды: f(z)*(1-f(z))
      delta = hadamard(delta,
                       activate_derivative(layer_outputs_[i], activations_[i]));
    }

    // вход текущего слоя
    //   для i=0: layer_inputs_[0] = вход сети
    //   для i>0: layer_inputs_[i] = выход предыдущего слоя после активации
    const math::Matrix &layer_input = layer_inputs_[i];

    // градиент весов: delta * input^T
    //   delta: (выходы_слоя x 1)
    //   input^T: (1 x входы_слоя)
    //   результат: (выходы_слоя x входы_слоя) — как раз размер матрицы весов
    math::Matrix weight_grad = delta * layer_input.transpose();

    // обновление весов: W -= learning_rate * dW
    weights_[i] -= weight_grad * learning_rate;

    // обновление смещений: b -= learning_rate * delta
    biases_[i] -= delta * learning_rate;
  }
}

TrainingMetrics
NeuralNetwork::train(const std::vector<math::Matrix> &inputs,
                     const std::vector<math::Matrix> &targets, size_t epochs,
                     double learning_rate, size_t batch_size,
                     std::function<void(const TrainingMetrics &)> callback) {
  // индексы для перемешивания данных
  std::vector<size_t> indices(inputs.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::random_device rd;
  std::mt19937 gen(rd());

  TrainingMetrics metrics;

  // по эпохам
  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    std::shuffle(indices.begin(), indices.end(), gen);
    double total_loss = 0.0;
    size_t correct = 0;

    // по батчам
    for (size_t start = 0; start < inputs.size(); start += batch_size) {
      size_t end = std::min(start + batch_size, inputs.size());

      // внутри батча - последовательная обработка
      for (size_t i = start; i < end; ++i) {
        // прямой проход
        math::Matrix output = forward(inputs[indices[i]]);

        // функция потерь
        total_loss += cross_entropy_loss(output, targets[indices[i]]);

        // обратный проход и обновление весов
        backward(inputs[indices[i]], targets[indices[i]], learning_rate);

        // предсказанный класс из текущего forward
        int predicted = static_cast<int>(std::distance(
            output.begin(), std::max_element(output.begin(), output.end())));

        // истинный класс из one-hot вектора
        int actual = static_cast<int>(
            std::distance(targets[indices[i]].begin(),
                          std::max_element(targets[indices[i]].begin(),
                                           targets[indices[i]].end())));

        // сравниваем предсказание с истиной меткой
        if (predicted == actual) {
          ++correct;
        }
      }
    }

    // метрики за эпоху
    metrics.loss = total_loss / inputs.size();
    metrics.accuracy = static_cast<double>(correct) / inputs.size();
    metrics.epoch = epoch;

    // если нужно - вывод прогресса
    if (callback) {
      callback(metrics);
    }
  }

  return metrics;
}

int NeuralNetwork::predict(const math::Matrix &input) {
  // прямой проход
  math::Matrix output = forward(input);

  // индекс максимального элемента = предсказанный класс
  return static_cast<int>(std::distance(
      output.begin(), std::max_element(output.begin(), output.end())));
}

std::vector<double> NeuralNetwork::predict_proba(const math::Matrix &input) {
  // прямой проход
  math::Matrix output = forward(input);

  // возвращает весь вектор
  return std::vector<double>(output.begin(), output.end());
}

double NeuralNetwork::evaluate(const std::vector<math::Matrix> &inputs,
                               const std::vector<int> &labels) {
  size_t correct = 0;

  // сравнение предсказания с истинной меткой для каждого примера
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (predict(inputs[i]) == labels[i]) {
      ++correct;
    }
  }

  // доля правильных ответов (0.0 - 1.0)
  return static_cast<double>(correct) / inputs.size();
}

size_t NeuralNetwork::num_parameters() const {
  size_t count = 0;

  // кол-во весов
  for (const auto &w : weights_) {
    count += w.size();
  }

  // кол-во смещений
  for (const auto &b : biases_) {
    count += b.size();
  }

  return count;
}

void NeuralNetwork::save(const std::string &filepath) const {
  std::ofstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filepath);
  }

  size_t num_layers = weights_.size();
  file.write(reinterpret_cast<const char *>(&num_layers), sizeof(num_layers));

  for (size_t i = 0; i < num_layers; ++i) {
    weights_[i].save_binary(file);
    biases_[i].save_binary(file);

    size_t act_len = activations_[i].size();
    file.write(reinterpret_cast<const char *>(&act_len), sizeof(act_len));
    file.write(activations_[i].c_str(), static_cast<std::streamsize>(act_len));
  }
}

void NeuralNetwork::load(const std::string &filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for reading: " + filepath);
  }

  size_t num_layers;
  file.read(reinterpret_cast<char *>(&num_layers), sizeof(num_layers));

  weights_.clear();
  biases_.clear();
  activations_.clear();

  for (size_t i = 0; i < num_layers; ++i) {
    weights_.push_back(math::Matrix::load_binary(file));
    biases_.push_back(math::Matrix::load_binary(file));

    size_t act_len;
    file.read(reinterpret_cast<char *>(&act_len), sizeof(act_len));

    std::string activation(act_len, '\0');
    file.read(&activation[0], static_cast<std::streamsize>(act_len));
    activations_.push_back(activation);
  }
}

void NeuralNetwork::initialize_weights(const std::vector<LayerConfig> &config) {
  std::random_device rd;
  std::mt19937 gen(rd());

  for (const auto &layer : config) {
    // Xavier/Glorot инициализация
    //   предел: sqrt(6 / (fan_in + fan_out))
    //   для равномерного распределения U(-limit, limit)
    double limit = std::sqrt(6.0 / (layer.input_size + layer.output_size));
    std::uniform_real_distribution<double> dis(-limit, limit);

    // матрица весов (выходы x входы)
    math::Matrix w(layer.output_size, layer.input_size);
    for (auto &val : w) {
      val = dis(gen);
    }
    weights_.push_back(std::move(w));

    // смещения инициализируются нулями
    biases_.push_back(math::Matrix::zeros(layer.output_size, 1));

    // название функции активации
    activations_.push_back(layer.activation);
  }
}

math::Matrix NeuralNetwork::activate(const math::Matrix &x,
                                     const std::string &name) const {
  // свободные функции из библиотеки matrix
  if (name == "sigmoid") {
    // 1/(1+e^(-x))
    return sigmoid(x);
  }

  if (name == "relu") {
    // max(0, x)
    return relu(x);
  }

  if (name == "softmax") {
    // e^x / sum(e^x)
    return softmax(x);
  }

  if (name == "none") {
    // без активации
    return x;
  }

  throw std::invalid_argument("Unknown activation: " + name);
}

math::Matrix NeuralNetwork::activate_derivative(const math::Matrix &x,
                                                const std::string &name) const {
  // производные функций активации
  if (name == "sigmoid") {
    // s*(1-s)
    return sigmoid_derivative(x);
  }

  if (name == "relu") {
    // 1 if x>0 else 0
    return relu_derivative(x);
  }

  if (name == "softmax" || name == "none") {
    // для softmax производная уже учтена в комбинации с cross-entropy
    // для "none" производная = 1
    return math::Matrix::ones(x.rows(), x.cols());
  }

  throw std::invalid_argument("Unknown activation: " + name);
}

double NeuralNetwork::cross_entropy_loss(const math::Matrix &output,
                                         const math::Matrix &target) const {
  double loss = 0.0;
  const double epsilon = 1e-15; // защита от log(0)

  // L = -sum(y_true * log(y_pred)) / N
  //   y_true — target (one-hot)
  //   y_pred — output (вероятности от softmax)

  for (size_t i = 0; i < output.rows(); ++i) {
    // для зажатия выхода в [epsilon, 1-epsilon]
    double p = std::max(epsilon, std::min(1.0 - epsilon, output(i, 0)));

    // target(i,0) = 1 только для правильного класса
    loss -= target(i, 0) * std::log(p);
  }

  return loss;
}

} // namespace ml