# Оптимизация маршрутов между объектами и потребителями с учетом кусочно-линейных затрат на перевозку
## Описание проекта
Проект предназначен для расчёта затрат на перевозку товара от складов до потребителей с учётом возможных кусочно-линейных затрат, например затрат на CAPEX и OPEX, зависящих от количества товара на складах. Задача является по сути своей задачей о нахождении минимума функции множества переменных, на которые наложены ограничения. В теории программу можно модифицировать для решения задач с нелинейными затратами. Маршруты после расчёта визуализируются на картах, если базы и объекты расположены на территории РФ или Беларуси. Программа реализована с использованием библиотек pyomo и geopandas. Задача о минимизации затрат решается одним из 3-х open-source алгоритмов, cbc или glpk для линейных задач и ipopt для нелинейных. Данные считываются из excel-файла Для удобного выбора считываемого файла с данными используется простой интерфейс, реализованный на PyQt5.
