# LinearWithConv2d
Реализация FC-слоя через Conv2d

## Настройка среды
Для установки нужных библиотек запустите:
  ```bash
    pip install -r requirements.txt
  ```
## Тестирование
Чтобы запустить проверку введите:
 ```bash
    py.test -s tests.py
 ```
## ONNX
 Чтобы показать различия модели были конвертированы в ONNX:
 ```bash
    python3 convert_to_onnx.py
 ```
после выполнения данного скрипта, появятся два файла `linear_conv.onnx` и `linear.onnx`. Открыть и сравнить их можно на сайте [Netron](https://netron.app/).
* Linear_
  
![Image alt](https://github.com/teektonik/LinearWithConv2d/raw/main/images/fc.jpeg)

* LinearWithConv2d

![Image alt](https://github.com/teektonik/LinearWithConv2d/raw/main/images/fc_conv.jpeg)
