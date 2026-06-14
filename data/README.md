# Директория данных

Здесь размещаются файлы датасета MNIST:

## Обучающий набор
- `train-images-idx3-ubyte` (60 000 изображений)
- `train-labels-idx1-ubyte` (60 000 меток)

## Тестовый набор
- `t10k-images-idx3-ubyte` (10 000 изображений)
- `t10k-labels-idx1-ubyte` (10 000 меток)

## Скачивание датасета

Официальный источник: http://yann.lecun.com/exdb/mnist/

На момент разработки официальный сайт был недоступен, поэтому файлы были скачаны с зеркала Google Storage:

```bash
cd data

curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz
```

## Файлы моделей
Обученные модели имеют формат `.bin`.