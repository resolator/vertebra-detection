# vertebra-detection


## Fast start

- Install dependencies by running the `install.sh` from the repository's root.

- Run `demo_app.py` on your data:
```bash
python3 demo/demo_app.py --images PATH_TO_FOLDER_WITH_IMAGES_ONLY --model-path data/model.pth
```

## Постановка задачи

Детекция по принципу **здоров/не здоров** для **каждого** 
межпозвоночного диска **шейного** отдела.


## Анализ данных


#### Анализ 

Датасет содержит квадратные RGB изображения (можно конвертировать в ЧБ так как 
чистый скан является ЧБ изображением). 
Размер каждого изображения: 384х384 или 512х512.

- Размер входного датасета: 891 изображение и 12 файлов разметки которые 
хранят аннотацию для каждого изображения.
- Количество уникальных классов в разметке: 8
- Количество пригодных для разметки семплов: 365
- Количество семплов содержащих хотя бы один межпозвоночный диск шейного отдела: 343
- Медианное количество размеченных межпозвоночных дисков шейного отдела: 5

Так как исходная разметка содержит 8 классов конвертируем их в 2 
(здоров/не здоров).

Распределение изображений по количеству размеченных межпозвоночных дисков шейного отдела:

![raw_hist.png](content/raw_hist.png?raw=true)

Основываясь на этой гистограмме было принято решение отсечь семплы которые 
содержат меньше 4 и больше 6 размеченных дисков. Итоговая гистограмма:

![processed_hist.png](content/processed_hist.png?raw=true) 

- Итоговое количество используемых семплов: 328
- Количество размеченных дисков на итоговом датасете: 1641
- Количество здоровых дисков (1 класс): 920 (56%)
- Количество паталогических дисков (2 класс): 721 (44%)


##### Проблемы датасета:

- Некоторые изображения значительно ярче большинства
- Некоторые изображения имеют странную разметку

Пример странной разметки:

![img_00122.jpg](content/img_00122.jpg?raw=true)
![img_00123.jpg](content/img_00123.jpg?raw=true)

Пояснение: два соседних изображения одного пациента, но в одном случае 
второй диск сверху размечен как здоровый, во втором - как больной. Также 
последний диск снизу либо размечен лишним в одном случае, либо не 
размечен в другом.


#### Пример входных данных

##### 512x512

![img_00005.jpg](content/img_00492.jpg?raw=true)
![img_00005.jpg](content/img_00981.jpg?raw=true)


##### 384x384

![img_00642.jpg](content/img_00642_raw.jpg?raw=true)
![img_00644.jpg](content/img_00644.jpg?raw=true)


#### Пример данных с обработанной разметкой

![img_00005.jpg](content/img_00005.jpg?raw=true)
![img_00005.jpg](content/img_00347.jpg?raw=true)
![img_00005.jpg](content/img_00381.jpg?raw=true)
![img_00005.jpg](content/img_00642.jpg?raw=true)


## Архитектура сети

Выбор происходил между YOLOv3 и FasterRCNN. Изначально была выбрана YOLOv3, 
однако после проблем с её обучением (была использована сторонняя 
имплементация) а так же из-за того что FasterRCNN является более точной 
архитектурой, хоть и более медленной и тяжеловесной, в качестве 
архитектуры была выбрана **FasterRCNN** с **ResNet50** в качестве экстрактора 
фич.

## Подготовка данных

Для обработки данных и подготовки выборок для обучения и теста был разработан 
скрипт `tools/prepare_markup.py`. С помощью него можно:
- подготовить train/test выборки 
- посчитать `mean` и `std` для нормализации картинок во время обучения 
(считается по всем переданным картинкам)
- найти и удалить дубликаты разметки межпозвоночных дисков (дубликат 
определяется с помощью IoU)
 - удалить из итоговой выборки семплы на которых количество размеченных 
 дисков меньше чем N (параметр)
- визуализировать и сохранить изображения с отрисованной разметкой

Для обучения были сгенерированны выборки train/test с параметрами в файле 
`tools/prepare_markup.cfg`.
Скрипт поддерживает запуск с конфиг файлом (параметр `--config`).


## Процесс обучения

Обучение проходило на ноутбуке со следующими параметрами:
- Видеокарта: NVIDIA GeForce 1070
- Процессор: Intel Core i7-8750H
- ОЗУ: 32GB DDR4
- Тип накопителя: SSD

Процесс обучения был разработан на основе фреймворка PyTorch. Скрипт для 
обучения по пути `train/train_pytorch.py`. Было проведено несколько 
экспериментов и лучшие параметры для обучения содержатся в 
`train/train_pytorch.cfg`. 

Особенности процесса обучения:
- Аугментация (1 вариант): случайный flip по `x` и `y`, центральный кроп
- Аугментация (2 вариант):
- Пост обработка фильтрует дублирующиеся bounding box'ы
- Оптимизированный расчет метрик


#### History or training process:

##### augmentation 1

![train_process_aug1](content/train_process_aug1.png?raw=true)

##### augmentation 2

![train_process_aug2](content/train_process_aug2.png?raw=true)



#### Final metrics

##### Results for augmentation 1

|           | train.json | test.json | markup.json |
|-----------|------------|-----------|-------------|
| Precision | 0.84       | 0.698     | 0.793       |
| Recall    | 0.848      | 0.741     | 0.813       |
| F1        | 0.843      | 0.714     | 0.8         |
| mAP       | 0.571      | 0.679     | 0.605       |


##### Results for augmentation 2

|           | train.json | test.json | markup.json |
|-----------|------------|-----------|-------------|
| Precision | 0.998      | 0.772     | 0.923       |
| Recall    | 0.998      | 0.816     | 0.937       |
| F1        | 0.998      | 0.789     | 0.928       |
| mAP       | 0.578      | 0.726     | 0.625       |


#### Prediction examples (augmentation 1)

From test.json (left - GT, right - PD):

![img_00292.jpg](content/img_00292.jpg?raw=true)
![img_00357.jpg](content/img_00357.jpg?raw=true)
![img_01200.jpg](content/img_01200.jpg?raw=true)

From train.sjon (left - GT, right - PD):

![img_00632.jpg](content/img_00632.jpg?raw=true)
![img_00721.jpg](content/img_00721.jpg?raw=true)
![img_00760.jpg](content/img_00760.jpg?raw=true)


#### Очистка модели

Изначалько тренировочный скрипт сохраняет лучшие модели для каждой метрики + 
модель на последней эпохе. Он сохраняет не только веса сети, но и состояние 
оптимизатора и LR_scheduler'а. Это нужно для продолжения обучения, но не нужно 
для production, так как из-за этого модель весит в два раза больше.

Для облегчения модели за счет удаления лишней информации можно воспользоваться 
скриптом `tools/clean_model.py`.


## Демонстрация и оценка качества

Скрипт для демонстрации и оценки качества находится по пути 
`demo/demo_app.py`. Позволяет применить переданную модель как на картинках 
без разметки (с сохранением или непосредственной визуализацией результатов), 
так и на сгенерированной с помощью `tools/prepare_markup.py` выборке 
(в данном случае будет проведена оценка качества).


## Запланированные приемы для улучшения качества:

- Обучение на одноканальных изображениях, вместо трехканальных
- Дополнительная аугментация: случайный поворот на произвольный угол
- Балансировка классов с помощью применения `class_weights` к лоссу
- Балансировка классов на этапе разбиения датасета на train/test выборки
- Ансамблирование моделей
- Провести эксперименты с другими сетями в качестве экстракторов фич
- Возобновление обучения с сохраненного чек-поинта

