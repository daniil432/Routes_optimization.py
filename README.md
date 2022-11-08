# Оптимизация маршрутов между объектами и потребителями с учетом кусочно-линейных затрат на перевозку
## Описание проекта
Проект предназначен для расчёта затрат на перевозку товара от складов до потребителей с учётом возможных кусочно-линейных затрат, например затрат на CAPEX и OPEX, зависящих от количества товара на складах. Задача является по сути своей задачей о нахождении минимума функции множества переменных, на которые наложены ограничения. В теории программу можно модифицировать для решения задач с нелинейными затратами. Маршруты после расчёта визуализируются на картах, если базы и объекты расположены на территории РФ или Беларуси. Программа реализована с использованием библиотек pyomo и geopandas. Задача о минимизации затрат решается одним из 3-х open-source алгоритмов, cbc или glpk для линейных задач и ipopt для нелинейных. Данные считываются из excel-файла Для удобного выбора считываемого файла с данными используется простой интерфейс, реализованный на PyQt5.

## Требования к установке
- Python версии 3.x
- Anaconda
- Установка пакетов, перечисленных в requirements.txt
- Microsoft Visual C++ 14.0 или выше

## Как использовать
Создать виртуальное окружение conda: ```conda create --name myenv``` и ```conda activate myenv```. Установка пакетов из requirements.txt ```conda install --name myenv -c conda-forge --file requirements.txt```. После установки пакетов из requirements.txt нужно установить один из алгоритмов, решающих задачу о минимизации. Другие алгоритму уже присутствуют в проекте в виде .exe-файлов. Алгоритм glpk должен быть установлен через терминал ```conda install -c conda-forge glpk```. Если при использовании алгоритма ipopt возникают проблемы с памятью (выдаётся ошибка или предупреждение), можно уменьшить параметр mumps_mem_percent на 100 или более (260 стр. кода).
Запускать код можно как через Interface.py ```py Interface.py```, так и через pyomo_piecewise.py ```py pyomo_piecewise.py```.
В первом случае будет запущена программа с интерфейсом, где через интерфейс можно выставить необходимые параметры для задачи.
Во втором случае путь к файлу, файл, алгоритм и отрисовка карт выбираются вручную в конце файла pyomo_piecewise.py (~ 724-749 строчки кода). Файл с данными по формату должен быть подобен файлу ТПС_Тест.xlsx, который лежит в папке support_files.
 На данный момент карты рисуются для России и Беларуси, другие регионы не предусмотрены. Если необходимо отрисовать карты с участием Беларуси, в названии файла должно быть слово "Беларусь".

