import os
import re
import sys
import time
import copy
import openpyxl
import matplotlib
import numpy as np
import pandas as pd
import geopandas as gpd
from pyomo.environ import *
from datetime import datetime
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.lines import Line2D


matplotlib.use('TkAgg')


class OptimizationRoutes:
    def __init__(self, file_path, filename, add_to_capex):
        # Объявляем все необходимые переменные, необходимые методам класса для работы
        self.filename = filename
        self.add_to_capex = add_to_capex
        self.answer_copy = None
        self.supply = None
        self.secondary_logistics = None
        self.azs_requirements = None
        self.fuel_name = None
        self.azs_name = None
        self.base_name = None
        self.capex = None
        self.OPEX_fix = None
        self.OPEX_var = None
        self.primary_logistics = None
        self.km = None
        self.azs_info = None
        self.channel = None
        self.base_info = None
        self.model = None
        self.all_routes_separated = None
        self.azs_fuel_name = None
        self.full_distribution = None
        self.distribution_summary = None
        self.answer_separated = None
        self.ts = None
        self.file_path = file_path

    def read_data(self):
        print('Программа запушена, начинается чтение файла')
        # Вектор с значениями максимального запаса НП на базах
        self.supply = pd.read_excel(self.file_path, sheet_name='САРЕХ').set_index(
            ['Базис']).loc[:, ['Пропуск.способность']]
        # Индексы баз с нулевой возможной загрузкой
        zero_val = self.supply.index[self.supply['Пропуск.способность'] == 0].tolist()
        self.supply = self.supply.drop(zero_val)['Пропуск.способность'].tolist()
        # Чтение исходных данных из файлов
        # Матрица размерностью кол-во нефтебаз X кол-во потребителей
        self.secondary_logistics = pd.read_excel(self.file_path, sheet_name='ВЛ').set_index(['ID']).drop(
            zero_val, axis=1).values.tolist()
        # Матрица размерностью спрос на разные нефтепродукты X кол-во потребителей
        self.azs_requirements = pd.read_excel(self.file_path, sheet_name='Объемы 2030').set_index(['ID']).drop(
            ['Спрос'], axis=1).values.tolist()
        # Названия типов топлива, АЗС и баз
        self.fuel_name = list(
            pd.read_excel(self.file_path, sheet_name='Объемы 2030').set_index(['ID']).drop(['Спрос'], axis=1).columns)
        self.azs_name = list(
            pd.read_excel(self.file_path, sheet_name='Объемы 2030').set_index(['ID']).drop(['Спрос'], axis=1).index)
        self.base_name = list(
            pd.read_excel(self.file_path, sheet_name='САРЕХ').set_index(['Базис']).drop(zero_val).index)
        # Матрица размерностью кап. затраты на 100*n тонн нефти X кол-во нефтебаз
        self.capex = pd.read_excel(self.file_path, sheet_name='САРЕХ').set_index(['Базис']).drop(
            ['Планируемая мощность', 'Пропуск.способность'], axis=1).drop(zero_val).values.tolist()
        # Матрица размерностью фиксированные операционные затраты на 100*n тонн нефти X кол-во нефтебаз
        self.OPEX_fix = pd.read_excel(self.file_path, sheet_name='ОРЕХ_фикс').set_index(['Базис']).drop(
            zero_val).values.tolist()
        # Вектор с переменными операционными затратами для нефтебаз
        self.OPEX_var = pd.read_excel(self.file_path, sheet_name='ОРЕХ_пер').set_index(['Базис']).drop(
            zero_val)['руб/тн'].tolist()
        # Матрица размерностью тариф на разные нефтепродукты X кол-во нефтебаз
        self.primary_logistics = pd.read_excel(self.file_path, sheet_name='ПЛ').set_index(['Базис']).drop(
            zero_val).values.tolist()
        # Матрица размерностью кол-ва АЗС X кол-во нефтебаз
        self.km = pd.read_excel(self.file_path, sheet_name='Км').drop(zero_val, axis=1).set_index(['ID'])
        # Инфо об АЗС/МОК
        self.azs_info = pd.read_excel(self.file_path, sheet_name='спрОбъект').set_index(['ID']).loc[self.azs_name]
        # Инфо о каналах
        self.channel = pd.read_excel(self.file_path, sheet_name='спрОбъект').loc[
                       0:3, ["Канал", "Канал крупно", "Канал3"]].set_index("Канал")
        # Инфо о базах
        self.base_info = pd.read_excel(
            self.file_path, sheet_name='спрБаза').loc[0:, ["База_подробно", "База", "Тип Базы"]].set_index(
            ['База_подробно']).loc[self.base_name].values.tolist()
        print('Файл прочитан, формируем модель для оптимизации и создаём переменные')

    def create_model(self):
        # Создаём модель pyomo, в которую поместим проблему в виде уравнения и ограничений на переменные
        self.model = ConcreteModel()
        # Формируем списки для переменных возможных маршрутов, их названий и тарифов по этим маршрутам
        all_routes_name_separated = []
        self.all_routes_separated = []
        all_routes_name = []
        all_routes = []
        all_routes_costs = []
        self.azs_fuel_name = []
        # Формируем маршруты и названия маршрутов от одной нефтебазы до всех азс, разделяя маршруты по видам топлива
        for base1 in range(len(self.supply)):
            one_base_name = []
            one_base = []
            one_base_costs = []
            for azs in range(len(self.azs_requirements)):
                for fuel in range(len(self.azs_requirements[azs])):
                    route_name = self.base_name[base1] + '_' + self.azs_name[azs] + '_' + self.fuel_name[fuel]
                    one_base_name.append(route_name)
                    all_routes_name.append(route_name)
                    one_base_costs.append((self.secondary_logistics[azs][base1] + self.primary_logistics[base1][fuel] +
                                           self.OPEX_var[base1]))
                    # Для pyomo-проблемы задаём аттрибут в виде переменной маршрута от НБ до АЗС-топливо
                    setattr(self.model, route_name, Var(bounds=(0, 9999), initialize=0))
                    # Список маршрутов ко всем АЗС от базы
                    one_base.append(getattr(self.model, route_name))
                    # Список всех маршрутов
                    all_routes.append(getattr(self.model, route_name))
            self.all_routes_separated.append(one_base)
            all_routes_name_separated.append(one_base_name)
            all_routes_costs.append(one_base_costs)
        # Имена связок АЗС-топливо
        for azs in range(len(self.azs_requirements)):
            for fuel in range(len(self.azs_requirements[azs])):
                self.azs_fuel_name.append(self.azs_name[azs] + '_' + self.fuel_name[fuel])

        # Задаём ограничения на переменные маршрутов - количество тонн топлива больше или равно нулю
        for variable in range(len(all_routes)):
            setattr(self.model, 'constr_' + all_routes_name[variable], Constraint(expr=all_routes[variable] >= 0))

        # Создаём ограничения - сумма тонн от одной НБ меньше или равна её максимально возможной загрузке
        for base1 in range(len(self.all_routes_separated)):
            constraint = 0
            for route in range(len(self.all_routes_separated[base1])):
                constraint += self.all_routes_separated[base1][route]
            if self.supply[base1] == 0:
                setattr(self.model, f'constr_base_{base1}',
                        Constraint(expr=constraint == self.supply[base1]))
            else:
                setattr(self.model, f'constr_base_{base1}',
                        Constraint(expr=constraint <= self.supply[base1]))

        # Формируем одну матрицу для CAPEX и OPEX, чтобы не рассматривать их отдельно в уравнении
        CAP_OP = []
        for base1 in range(len(self.capex)):
            temp = []
            for limit in range(len(self.capex[base1])):
                temp.append((self.capex[base1][limit] / 15) + (self.OPEX_fix[base1][limit] * 1000))
            CAP_OP.append(temp)

        # Создаём переменную суммы топлива от одной базы
        sum_base = []
        for base1 in range(len(self.supply)):
            setattr(self.model, f'sum_{base1}', Var(bounds=(0, len(self.capex[0])*100)))
            sum_base.append(getattr(self.model, f'sum_{base1}'))

        # Накладываем ограничение - переменная суммы топлива должна быть равна сумме переменных маршрутов от этой базы 
        for base1 in range(len(self.all_routes_separated)):
            setattr(self.model, f'cons_sum_{base1}',
                    Constraint(expr=sum_base[base1] == sum(self.all_routes_separated[base1])))

        # Для создания кусочно-постоянной ф-ии задаём вектор значений сумм топлива, на которых происходит скачок тарифа
        # Первые значения вектора созданы для нулевой загрузки и для скачка с нулевой загрузки до ненулевой
        domains = [0, 0]
        for index in range(len(self.capex[0])):
            if index == len(self.capex[0]) - 1:
                domains.append(100 * (index + 1) + self.add_to_capex)
            else:
                domains.append(100 * (index + 1) + self.add_to_capex)
                domains.append(100 * (index + 1) + self.add_to_capex)

        # Тарифы CAPEX и OPEX на заданных промежутках сумм топлива
        # Первое значение - тариф при нулевой загрузке
        CAP_OP_costs = []
        for base1 in range(len(CAP_OP)):
            temp = [0.]
            for val in range(len(CAP_OP[base1])):
                temp.append(CAP_OP[base1][val])
                temp.append(CAP_OP[base1][val])
            CAP_OP_costs.append(temp)

        # Создаём переменные, которые будут являться выходным значением кусочно-постоянной ф-ии
        pw_list = []
        for base1 in range(len(self.supply)):
            setattr(self.model, f'pw_var_{base1}', Var())
            pw_list.append(getattr(self.model, 'pw_var_' + f'{base1}'))

        # Задаём кусочно-переменные ф-ии - входным значением является переменная суммы топлива от одной базы,
        # выходным значением является тариф для этой суммы топлива, переменные для выходных значений заданы циклом выше.
        for base1 in range(len(self.supply)):
            setattr(self.model, f'pw_{base1}',
                    Piecewise(pw_list[base1], sum_base[base1], pw_pts=domains, f_rule=CAP_OP_costs[base1],
                              pw_constr_type='EQ', pw_repn='CC'))

        def objective_func(*x):
            # Составляем функцию для нашей проблемы.
            # Функция это сумма произведений всех возможных маршрутов на тарифы по этим маршрутам,
            # плюс CAPEX и OPEX для каждой из баз.
            func = quicksum(
                x[base_ind][elem] * (all_routes_costs[base_ind][elem]) for base_ind in
                range(len(all_routes_costs)) for elem in range(len(all_routes_costs[base_ind])))
            # Тут к функции прибавляются CAPEX_OPEX в виде переменных, значения которых зависят от кусочно-постоянной ф.
            for base2 in range(len(all_routes_costs)):
                func += pw_list[base2]
            return func

        # Задаём в нашей проблеме функцию, которую будем минимизировать.
        self.model.obj = Objective(expr=objective_func(*self.all_routes_separated), sense=minimize)

        # Задаём последние ограничения для нашей проблемы.
        # Транспонируем матрицу всех возможных маршрутов, чтобы строки содержали не все маршруты от одной базы,
        # а все возможные маршруты к одной АЗС
        _transposed = [[self.all_routes_separated[j][i] for j in range(len(self.all_routes_separated))] for i in
                       range(len(self.all_routes_separated[0]))]
        # Создаём ограничения - сумма топлива к одной АЗС должна быть равна потребностям этой АЗС.
        for azs in range(len(_transposed)):
            constraint = 0
            for route in range(len(_transposed[azs])):
                constraint += _transposed[azs][route]
            setattr(self.model, f'constr_azs_{azs}',
                    Constraint(expr=constraint == [item for sublist in self.azs_requirements for item in sublist][azs]))

        return self.model, self.all_routes_separated, self.azs_fuel_name

    def solve_problem(self, signal):
        # Решаем проблему одним из доступных алгоритмов. Все алгоритмы должны быть заранее предустановлены.
        # Алгоритмы cbc и ipopt должны находиться в виде .exe файлов в папке support_files.
        # Алгоритм glpk должен быть установлен через терминал (требуется Microsoft Visual c++ ver >=14.0):
        # Наиболее удобный вариант - при наличии Anaconda: conda install -c conda-forge glpk
        # Без Anaconda - pip install glpk , или при ошибке с pep517 - pip install glpk --no-use-pep517
        # Если не получилось через pip - скачать glpk через https://sourceforge.net/projects/winglpk/ ,
        # Далее необходимо скачанную директорию поместить к себе на рабочий диск и добавить эту директорию в PATH,
        # инструкция: https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/
        print('Модель создана, начинается оптимизация')
        if signal == 0:
            SolverFactory("support_files\\cbc.exe").solve(self.model, tee=True)
        elif signal == 1:
            try:
                SolverFactory("glpk").solve(self.model, tee=True)
            except Exception as err1:
                print(err1)
                try:
                    # Если glpk вручную был скачан корректно, добавлен в PATH, но не срабаотывает, возможно нужно
                    # полностью прописать путь до .exe файла в glpk-директории
                    SolverFactory("glpsol.exe").solve(self.model, tee=True)
                except Exception as err2:
                    print(err2)
                    print('Нет установленного glpk или путь к нему при ручной установке был указан неверно')
        else:
            with SolverFactory("support_files\\ipopt") as opt:
                opt.options.option_file_name = "\\support_files\\ipopt.opt"
                with open("\\support_files\\ipopt.opt", "w") as f:
                    # Если ipopt пишет, что ему не хватает памяти, можно понизить mumps_mem_percent на 100 или более.
                    f.write("mumps_mem_percent 500\n")
                opt.solve(self.model, tee=True)
        print('Оптимизация закончилась, начинается сбор результатов из оптимизатора')

    def create_answer(self):
        # После полученного решения проблемы необходимо сформировать ответ.
        answer = []
        self.answer_separated = []
        sum_base = []
        # Получаем ответ от алгоритма в виде значений от всех переменных
        for v in self.model.component_data_objects(Var):
            try:
                answer.append(round(v.value, 6))
            except Exception as error1:
                answer.append(v.value)
                print(error1)

        # Разделяем весь набор переменных на наборы переменных, распределенные по базам.
        count = 0
        for base3 in range(len(self.supply)):
            temp = []
            for azs in range(len(self.all_routes_separated[base3])):
                temp.append(answer[count])
                count += 1
            sum_base.append(sum(temp))
            self.answer_separated.append(temp)

        # Формируем списки с суммой тонн разных типов топлива, суммами первичной и вторичной логистики от каждой из НБ.
        # Заранее создаём списки правильных размеров, заполненные нулями.
        sum_by_fuel = [[0 + 0 * i * j for i in range(len(self.fuel_name))] for j in range(len(self.base_name))]
        secondary_sum = [0 + 0 * i for i in range(len(self.base_name))]
        sum_primary = [0 + 0 * i for i in range(len(self.base_name))]
        for base3 in range(len(self.base_name)):
            count = 0
            for azs in range(len(self.secondary_logistics)):
                for fuel in range(len(self.fuel_name)):
                    sum_by_fuel[base3][fuel] += self.answer_separated[base3][count]
                    secondary_sum[base3] += self.answer_separated[base3][count] * self.secondary_logistics[azs][base3]
                    sum_primary[base3] += self.answer_separated[base3][count] * self.primary_logistics[base3][fuel]
                    count += 1

        def CAPEX_solver(weight, capex):
            # Небольшая реализация кусочно-постоянной функции.
            # Вес от одной нефтебазы сравниваем с значениями, на которых происходят скачки тарифов,
            # после чего получаем тариф из матрицы capex/opex.
            result = float()
            for index in range(len(capex)):
                if index == 0:
                    if weight <= 100 + self.add_to_capex:
                        result = capex[index]
                    else:
                        pass
                elif index == len(self.capex) - 1:
                    if ((100 * index) + self.add_to_capex) <= weight:
                        result = capex[index]
                    else:
                        pass
                else:
                    if ((100 * index) + self.add_to_capex) <= weight <= ((100 * (index + 1)) + self.add_to_capex):
                        result = capex[index]
            return result

        # Формируем полный ответ в виде списка с значениями множества параметров для каждого из маршрутов.
        # Ответ (матрица), разделенный по базам, превращаем в матрицу, разделенную по связкам АЗС-топливо.
        self.answer_separated = list(map(list, zip(*self.answer_separated)))
        # Счётчик нужен, чтобы учитывать связку АЗС-топливо, так как список ответа длиной число АЗС * тип топлива.
        count = 0
        self.full_distribution = []
        print('Ответ сформирован, начинается формирование файла с результатами')
        for azs in range(len(self.azs_name)):
            for fuel in range(len(self.fuel_name)):
                # Получаем информацию о текущей АЗС.
                _info = self.azs_info.loc[self.azs_name[azs]].tolist()
                # Получаем индекс базы, с которой для данной связки АЗС-топливо есть ненулевая отгрузка.
                base_index = [self.answer_separated[count].index(self.answer_separated[count][val]) for val in
                              range(len(self.answer_separated[count])) if self.answer_separated[count][val] != 0]
                # Дополнительная проверка, что отгрузка с базы не равна нулю.
                for ind in base_index:
                    if sum_base[ind] == 0.0:
                        pass
                    else:
                        # Пробуем сформировать список с значениями для одного маршрута.
                        try:
                            temp = [self.azs_name[azs], self.fuel_name[fuel], self.base_name[ind], *self.base_info[ind],
                                    *_info,
                                    self.channel.loc[[_info[0]]].values.tolist()[0][0],
                                    self.channel.loc[[_info[0]]].values.tolist()[0][1],
                                    self.answer_separated[count][ind],
                                    self.km.loc[[self.azs_name[azs]]].values.tolist()[0][ind],
                                    CAPEX_solver(sum_base[ind], self.capex[ind]),
                                    CAPEX_solver(sum_base[ind], self.OPEX_fix[ind]),
                                    self.OPEX_var[ind], self.primary_logistics[ind][fuel],
                                    self.secondary_logistics[azs][ind],
                                    (CAPEX_solver(sum_base[ind], self.capex[ind]) / sum_base[ind] *
                                     self.answer_separated[count][ind]),
                                    (CAPEX_solver(sum_base[ind], self.capex[ind]) / 15 / sum_base[ind] *
                                     self.answer_separated[count][ind]),
                                    (CAPEX_solver(sum_base[ind], self.OPEX_fix[ind]) / sum_base[ind] *
                                     self.answer_separated[count][ind] * 1000),
                                    self.OPEX_var[ind] * self.answer_separated[count][ind],
                                    self.primary_logistics[ind][fuel] * self.answer_separated[count][ind],
                                    self.secondary_logistics[azs][ind] * self.answer_separated[count][ind],
                                    (CAPEX_solver(sum_base[ind], self.capex[ind]) / 15 / sum_base[ind] *
                                     self.answer_separated[count][ind]) +
                                    (CAPEX_solver(sum_base[ind], self.OPEX_fix[ind]) / sum_base[ind] *
                                     self.answer_separated[count][ind] * 1000)
                                    + self.OPEX_var[ind] * self.answer_separated[count][ind] +
                                    self.primary_logistics[ind][fuel]
                                    * self.answer_separated[count][ind] + self.secondary_logistics[azs][ind] *
                                    self.answer_separated[count][ind], ]
                            self.full_distribution.append(temp)
                        except Exception as error2:
                            # В случае ошибки пытаемся все равно выдать ответ.
                            try:
                                print(error2)
                                temp = [self.azs_name[azs], self.fuel_name[fuel], self.base_name[ind],
                                        f'{error2}', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        self.answer_separated[count][ind],
                                        self.km.loc[[self.azs_name[azs]]].values.tolist()[0][ind],
                                        CAPEX_solver(sum_base[ind], self.capex[ind]),
                                        CAPEX_solver(sum_base[ind], self.OPEX_fix[ind]), self.OPEX_var[ind],
                                        self.primary_logistics[ind][fuel], self.secondary_logistics[azs][ind],
                                        0, 0, 0, 0, 0, 0, 0]
                                self.full_distribution.append(temp)
                            except Exception as error2:
                                print(error2)
                                temp = [self.azs_name[azs], self.fuel_name[fuel], self.base_name[ind],
                                        f'{error2}', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        self.answer_separated[count][ind], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                self.full_distribution.append(temp)
                count += 1

        # Полученный полный ответ конвертируем в DataFrame.
        self.full_distribution = pd.DataFrame(self.full_distribution,
                                              columns=["Название Объекта", "Тип топлива", "База_подробно", "Имя базы",
                                                       "Тип Базы", "Канал", "Объект", "Широта", "Долгота", "Кластер",
                                                       "Регион", "Номер объекта", "Макрорегион КПРА",
                                                       "Канал крупно", "Канал3", "Объём поставки", "Км", "исх capex",
                                                       "исх OPEX fix", "исх OPEX var", "исх ПЛ", "исх ВЛ", "capex итог",
                                                       "capex", "OPEX fix", "OPEX var", "ПЛ", "ВЛ", "Итого затраты"])
        # Формируем краткий ответ-сумму по доставке от каждой НБ.
        self.distribution_summary = []
        for base3 in range(len(self.base_name)):
            if sum_base[base3] == 0.0:
                pass
            else:
                current_capex = CAPEX_solver(sum_base[base3], self.capex[base3])
                current_opex = CAPEX_solver(sum_base[base3], self.OPEX_fix[base3])
                self.distribution_summary.append([self.base_name[base3], sum_base[base3], *sum_by_fuel[base3],
                                                  current_capex, current_capex / 15, current_opex * 1000,
                                                  self.OPEX_var[base3] * sum_base[base3], sum_primary[base3],
                                                  secondary_sum[base3], current_capex / 15 + current_opex * 1000 +
                                                  self.OPEX_var[base3] * sum_base[base3] + sum_primary[base3] +
                                                  secondary_sum[base3], current_capex / 15 / sum_base[base3],
                                                  current_opex * 1000 / sum_base[base3], self.OPEX_var[base3],
                                                  *self.base_info[base3]])
        # Полученный полный ответ конвертируем в DataFrame.
        self.distribution_summary = pd.DataFrame(self.distribution_summary,
                                                 columns=["Базис", "Объем перевалки", "Товар_1",
                                                          "Товар_2", "Товар_3", "Товар_4",
                                                          "Товар_5", "Товар_6", "capex", "capex в год",
                                                          "OPEX_fix", "OPEX_var", "ЖД тариф", "Транспорт", "Суммарно",
                                                          "исхСАРЕХ", "исхОРЕХ_фикс", "исхОРЕХ_пер",
                                                          "База", "Тип Базы"])

        # Теперь формируем матрицу всех маршрутов от НБ до АЗС-топливо в виде DataFrame,
        # добавив к матрице столбец с типами топлива.
        count = 0
        for azs in range(len(self.azs_requirements)):
            for fuel in range(len(self.fuel_name)):
                self.answer_separated[count].append(self.fuel_name[fuel])
                count += 1
        self.answer_separated = pd.DataFrame(self.answer_separated, columns=[*self.base_name, 'Тип топлива'])
        # Добавляем к матрице маршрутов в начало матрицы строку с суммированием тонн по каждому столбцу (каждой из НБ).
        sum_base.append(None)
        sum_base = np.array(sum_base)
        self.answer_separated.loc[-1] = sum_base
        self.answer_separated.index = self.answer_separated.index + 1  # shifting index
        # Не используем те столбцы, где сумма тонн равна нулю.
        col_zero = [col for col in self.answer_separated.columns if self.answer_separated[col][0] == 0]
        self.base_info = [ele for ele in self.base_info if self.base_name[self.base_info.index(ele)] not in col_zero]
        self.base_name = [ele for ele in self.base_name if ele not in col_zero]
        self.answer_separated.sort_index(inplace=True)
        self.answer_separated = self.answer_separated.drop(col_zero, axis=1)
        self.azs_fuel_name.insert(0, 'Сумма по топливу')
        # Добавляем отдельно от матрицы значение суммарных затрат на выбранную карту маршрутов.
        obj_val = [None] * (len(self.answer_separated) + 1)
        obj_val[0] = value(self.model.obj)
        obj_val = pd.DataFrame(obj_val, columns=['Стоимость перевозки'])
        self.answer_separated = self.answer_separated.join(obj_val)
        self.answer_separated = self.answer_separated.set_index(pd.Index(np.array(self.azs_fuel_name)))
        # Закончили формировать ответы в виде DataFrame.
        print('Суммарная цена перевозок: ', round(*obj_val.loc[0, ['Стоимость перевозки']].values.tolist(), 3))
        print('Распределение груза по нефтебазам: ', [round(i, 3) for i in sum_base[:len(sum_base) - 1]])
        print('Суммарный вес: ', round(sum(sum_base[:len(sum_base) - 1]), 3))
        return self.answer_separated, self.distribution_summary, self.full_distribution

    def save_answer(self):
        # Формируем директорию и файл с ответом внутри
        self.ts = datetime.now().strftime('%y.%m.%d_%H-%M')
        if not os.path.isdir(f'{self.filename}_{str(self.ts)}'):
            os.makedirs(f'{self.filename}_{str(self.ts)}')
        # Создание файла, затем заполнение этого файла нужными листами с ответмаи.
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'Объемы доставки'
        wb.save(filename=f'{self.filename}_{str(self.ts)}\\output_{self.filename}_{str(self.ts)}.xlsx')
        with pd.ExcelWriter(f'{self.filename}_{str(self.ts)}\\output_{self.filename}_{str(self.ts)}.xlsx',
                            engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:
            self.answer_separated.to_excel(writer, sheet_name="Объемы доставки")
            self.full_distribution.to_excel(writer, sheet_name="Подробно")
            self.distribution_summary.to_excel(writer, sheet_name="Затраты суммарно")
        print('Файлы сформированы и сохранены')

    def routes_map(self):
        # Если выбрана опция рисовать карты, то проверяем, есть ли в названии файла Беларусь.
        # В любом случае для отрисовки регионов считываем geojson файлы, которые содержат информацию о краях регионов
        # geojson файлы лежат в папке support_files
        print("Начинается формирование карт маршрутов")
        if re.search('Беларусь', self.filename) is None:
            # Если Беларусь, то рисуем карту без муниципальных регионов, только административные и города.
            municipal = gpd.read_file(os.curdir + '\\support_files\\admin_level_6.geojson')
            administrative = gpd.read_file(os.curdir + '\\support_files\\admin_level_4.geojson')
            cities = pd.read_csv(os.curdir + "\\support_files\\cities.csv")
        else:
            # Если Россия, то рисуем административные и муниципальные регионы с городами.
            administrative = gpd.read_file(os.curdir + '\\support_files\\belarus.geojson')
            municipal = administrative
            cities = pd.read_csv(os.curdir + "\\support_files\\cities_by.csv")
        # Извлекаем из excel-файла информацию о базах.
        info_base = pd.read_excel(
            self.file_path, sheet_name='спр').loc[
                    0:, ["База_подробно", "База", "Тип Базы", "Долгота", "Широта"]].set_index('База_подробно')

        # Создаём полотно, на котором будут изображаться все НБ и АЗС, участвующие в задаче.
        fig, axs = plt.subplots(dpi=1000)
        count = 0
        all_ann = []
        duplicates = {}
        # Убираем дубликаты баз, т.е. при наличии ЯНОС_АИ92 и ЯНОС_G-95 останется только ЯНОС, маршруты объединяются.
        for base6 in range(len(self.base_name)):
            if info_base.loc[self.base_name[base6], ['База']].item() in duplicates:
                pass
            else:
                temp = []
                for base7 in range(len(self.base_name)):
                    if info_base.loc[self.base_name[base6], ['База']].item() == \
                            info_base.loc[self.base_name[base7], ['База']].item():
                        temp.append(base7)
                duplicates[info_base.loc[self.base_name[base6], ['База']].item()] = temp
        # Цвета, которыми будут обозначаться точки интереса у разных НБ. Для изменения нужно найти палитру matplotlib и
        # заменить или добавить интересующие цвета.
        colors = ['brown', 'red', 'blue', 'orange', 'green', 'darkblue', 'olive', 'goldenrod', 'rosybrown', 'indigo',
                  'aqua', 'teal', 'slategrey', 'black', 'gray', 'firebrick', 'deepskyblue']
        longitude_azs_global = []
        latitude_azs_global = []
        loading_azs_global = []
        km_azs_global = []
        latitude_base_global = []
        longitude_base_global = []
        # Считываем из файлов координаты АЗС и НБ, соответствующих друг другу.
        for base4 in self.base_name:
            latitude_azs = self.full_distribution.set_index('База_подробно').loc[base4].groupby(
                by=['Название Объекта']).first()['Широта'].tolist()
            longitude_azs = self.full_distribution.set_index('База_подробно').loc[base4].groupby(
                by=['Название Объекта']).first()['Долгота'].tolist()
            loading_azs = self.full_distribution.set_index('База_подробно').loc[base4].groupby(
                by=['Название Объекта']).first()['Объём поставки'].tolist()
            km_azs = self.full_distribution.set_index('База_подробно').loc[base4].groupby(
                by=['Название Объекта']).first()['Км'].tolist()
            # Список с координатами НБ должен быть той же длины, что и список с координатами АЗС.
            latitude_base = info_base.loc[base4, ["Широта"]].values.tolist() * len(latitude_azs)
            longitude_base = info_base.loc[base4, ["Долгота"]].values.tolist() * len(longitude_azs)
            # Если есть дубликаты для какой-то из НБ, то эти дубликаты объединяем. Если нет, то ничего не делаем.
            if len(duplicates[info_base.loc[base4, ['База']].item()]) > 1:
                if self.base_name.index(base4) == duplicates[info_base.loc[base4, ['База']].item()][-1]:
                    longitude_azs_global += longitude_azs
                    latitude_azs_global += latitude_azs
                    loading_azs_global += loading_azs
                    km_azs_global += km_azs
                    latitude_base_global += latitude_base
                    longitude_base_global += longitude_base

                    longitude_azs = longitude_azs_global
                    latitude_azs = latitude_azs_global
                    loading_azs = loading_azs_global
                    km_azs = km_azs_global
                    latitude_base = latitude_base_global
                    longitude_base = longitude_base_global
                else:
                    longitude_azs_global += longitude_azs
                    latitude_azs_global += latitude_azs
                    loading_azs_global += loading_azs
                    km_azs_global += km_azs
                    latitude_base_global += latitude_base
                    longitude_base_global += longitude_base
                    continue
            else:
                pass
            # Списки с координатами и информацией о длине пути объединяем в матрицу.
            base_azs = zip(latitude_base, latitude_azs, longitude_base, longitude_azs, loading_azs, km_azs)
            # Координаты АЗС и НБ переводим в формат GeoDataFrame для дальнейших вычислений.
            geocoord_base = gpd.GeoDataFrame(geometry=gpd.points_from_xy(
                longitude_azs + [longitude_base[0]], latitude_azs + [latitude_base[0]]), crs="EPSG:4326")
            # Проверяем, какие регионы содержат в себе выбранные АЗС и НБ. Неиспользуемые регионы убираются.
            poly_mun = gpd.sjoin(municipal, geocoord_base, op='contains')
            # В процессе возникают дубликаты одних и тех же регионов, убираем их чтобы не перегружать лишним список.
            poly_mun = poly_mun.drop_duplicates(subset='name', keep='last')
            poly_adm = gpd.sjoin(administrative, geocoord_base, op='contains')
            poly_adm = poly_adm.drop_duplicates(subset='name', keep='last')
            poly_adm = administrative.loc[poly_adm['name'].index]
            # Копия DataFrame с городами.
            cities_map = copy.copy(cities)
            # Выбираем начальное значение, при котором только города с соответствующим населением и выше рисуются.
            population = 100000
            # Цикл, при котором выбирается максимально возможное количество городов на карте,
            while len(cities_map) >= 7:
                cities_map = copy.copy(cities)
                cities_map['Население'] = cities['Население'].astype(int)
                cities_map = cities_map.loc[(cities['Население'] >= population)].loc[
                             :, ["Н/п", "Регион", "Город", "Долгота", "Широта"]].rename(columns={
                                "Город": "name", "Долгота": "longitude", "Широта": "latitude"})
                cities_map = gpd.GeoDataFrame(
                    cities_map, geometry=gpd.points_from_xy(cities_map.longitude, cities_map.latitude), crs="EPSG:4326")
                # Проверка того, находятся ли города внутри отфильтрованных регионов.
                cities_map = gpd.sjoin(cities_map, poly_adm, op='within')
                # Если городов больше чем нужно, то к минимальному населению прибавка 10000.
                population += 10000

            # Все необходимые данные готовы, теперь начинается отрисовка графиков.
            # Рисуется полотно, на котором будет отображаться один регион.
            f, ax = plt.subplots(dpi=1000)
            # Есть оси axs и ax, соответствующие глобальной карте со всеми НБ и локальной с одной, делаем рабочими - ax.
            plt.sca(ax)
            # Рисуем на разных осях одни и те же регионы и города.
            poly_adm.plot(ax=ax, color="lightgrey", linewidth=1)
            poly_adm.plot(ax=axs, color="lightgrey", linewidth=1, zorder=count)
            count += 1
            poly_mun.plot(ax=ax, edgecolor='grey', color="lightblue", linewidth=0.3)
            poly_mun.plot(ax=axs, edgecolor='grey', color="lightblue", linewidth=0.3, zorder=count)
            count += 1
            poly_adm.plot(ax=ax, edgecolor='dimgrey', facecolor='none', linewidth=1)
            poly_adm.plot(ax=axs, edgecolor='dimgrey', facecolor='none', linewidth=1, zorder=count)
            count += 1
            cities_map.plot(ax=ax, edgecolor='forestgreen', color='limegreen', alpha=0)
            # Чтобы точки городов были видны, alpha нужно поставить как какое-то значение до единицы (это прозрачность)
            cities_map.plot(ax=axs, edgecolor='forestgreen', color='limegreen', alpha=0, zorder=count+1000)
            # ищем максимальные и минимальные координаты по соответствующим осям.
            max_x, max_y = max(*longitude_azs, *longitude_base), max(*latitude_azs, *latitude_base)
            min_x, min_y = min(*longitude_azs, *longitude_base), min(*latitude_azs, *latitude_base)
            # Если разница между макс. и мин. меньше, чем 1.5 широты или 1.5 долготы, то прибавляем к макс. и мин. 0.5
            if max_x - min_x <= 1.5:
                max_x += 0.5
                min_x -= 0.5
            if max_y - min_y <= 1.5:
                max_y += 0.5
                min_y -= 0.5
            # Определяем, какого цвета будут точки, находящиеся на разном удалении от НБ (на ax, неактуально для axs).
            for slat, dlat, slon, dlon, azsload, azskm in base_azs:
                if 0 <= azskm <= 150:
                    color = 'green'
                elif 150 < azskm <= 300:
                    color = 'orange'
                else:
                    color = 'red'
                # Рисуем точки на картах, размер точек задаётся параметром s.
                # В данном случае размер зависит от тонн, доставляемых на АЗС, как тонны * 5
                ax.scatter(dlon, dlat, s=azsload*5, color=color, alpha=0.8)
                # Рисуем эти же точки, но на глобальной карте. В данном случае цвет заранее задан в списке colors выше.
                axs.scatter(dlon, dlat, s=azsload*5, color=colors[
                    list(duplicates).index(info_base.loc[base4, ['НБ']].item())], alpha=0.8, zorder=count+2000)
            # Для осей ax задаём максимальные и минимальные значения широты/долготы для отображения.
            ax.set_xlim(min_x - (max_x - min_x) * 0.05, max_x + (max_x - min_x) * 0.05)
            ax.set_ylim(min_y - (max_y - min_y) * 0.05, max_y + (max_y - min_y) * 0.05)
            # Текущие оси приводим к квадратному виду, также убираем подписи осей (для отображения поставить True).
            plt.gca().set_box_aspect(1)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            # Выбираем, какие города будут отображаться.
            # В любом случае названия НБ будут отрисовываться.
            good_ann = [ax.text(longitude_base[0], latitude_base[0],
                                f'База {self.base_info[self.base_name.index(base4)][0]}')]
            all_ann.append(axs.text(longitude_base[0], latitude_base[0],
                                    f'База {self.base_info[self.base_name.index(base4)][0]}', zorder=4000))
            # Небольшая проверка, не является ли имя города nan-объектом. Если да, то ищем название в других столбцах.
            for x, y, label, safe_1, safe_2 in zip(cities_map.geometry.x, cities_map.geometry.y,
                                                   cities_map['name_left'], cities_map['Н/п'], cities_map["Регион"]):
                if isinstance(label, float):
                    label = safe_1 if isinstance(safe_1, str) else safe_2
                else:
                    pass
                # Чтобы выбрать то, какие города рисовать, проверяем, не вылезает ли название города за границы экрана.
                # Для этого название рисуется в любом случае.
                ann = ax.annotate(
                    label, xy=(x, y), xytext=(0, 0), textcoords="offset points", fontsize=4, weight='bold')
                # Затем проверяются размеры бокса, в котором эта надпись расположена.
                box = ann.get_window_extent(renderer=f.canvas.get_renderer())
                width = box.transformed(ax.transData.inverted())
                # Если размеры бокса больше или меньше пределов рисунка, то такой бокс с названием не рисуем.
                if width.x1 >= max_x + (max_x - min_x) * 0.05 or width.x0 <= min_x - (max_x - min_x) * 0.05:
                    pass
                else:
                    # Если всё нормально, то такой бокс рисовать будем
                    good_ann.append(ax.text(x, y, label))
                # В любом случае удаляем нарисованное название.
                ann.remove()
            # Специальная библиотека, которая рисует названия так, чтобы они не пересекались друг с другом (двигает их).
            adjust_text(good_ann, ax=ax)
            # Рисуем на обеих осях НБ.
            ax.scatter(longitude_base, latitude_base,
                       linewidths=0.1, s=25, marker="*", color='black', alpha=0.8, zorder=100)
            axs.scatter(longitude_base, latitude_base,
                        linewidths=0.1, s=25, marker="*", color='black', alpha=0.7, zorder=count+3000)
            # Название рисунка
            plt.title(f"Маршруты от {self.base_info[self.base_name.index(base4)][0]}")
            # Рисуем легенду для графиков (размер точки в легенде - markersize, размер текста - 'size').
            plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='km < 150',
                                       markerfacecolor='green', markersize=8),
                                Line2D([0], [0], marker='o', color='w', label='150 <= km < 300',
                                       markerfacecolor='orange', markersize=8),
                                Line2D([0], [0], marker='o', color='w', label='300 <= km',
                                       markerfacecolor='red', markersize=8)],
                       prop={'size': 7}, loc='upper center',
                       bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
            # Пробуем сохранить рисунок. Если в названии есть какие-то запрещённые символы, то без названия базы.
            try:
                plt.savefig(f"{self.filename}_{str(self.ts)}\\Маршруты_от_базы_"
                            f"{self.base_info[self.base_name.index(base4)][0].replace('/', '_').replace(':', '_')}.png")
            except Exception as error3:
                print("Ошибка при сохранении файла, пробуем сохранить без названия базы", error3)
                plt.savefig(f"{self.filename}_{str(self.ts)}\\Маршруты_от_{longitude_base[0]}_{latitude_base[0]}.png")
            print(f'Сформирована карта маршрутов от {base4}')
            # Закрываем нарисованные карты, чтобы не перегружать оперативную память.
            plt.close()
            # Обнуляем списки, в которые помещались дубликаты, чтобы при необходимости следующие дубликаты тоже учесть.
            longitude_azs_global = []
            latitude_azs_global = []
            loading_azs_global = []
            km_azs_global = []
            latitude_base_global = []
            longitude_base_global = []
        # Когда все локальные НБ нарисованы, выбираем axs как текущие оси.
        plt.sca(axs)
        # Убираем подписи осей.
        axs.axes.xaxis.set_visible(False)
        axs.axes.yaxis.set_visible(False)
        # Рисуем легенду для рисунка (размер точки в легенде - markersize, размер текста - 'size').
        handles = []
        for col in range(len(list(duplicates))):
            handles.append(Line2D([0], [0], marker='o', color='w', label=list(duplicates)[col],
                                  markerfacecolor=colors[col], markersize=8))
        plt.legend(handles=handles, prop={'size': 7}, loc='upper center',
                   bbox_to_anchor=(0.5, 0.0), fancybox=True, shadow=True, ncol=4)
        # Рисуем названия городов и НБ на глобальной карте.
        adjust_text(all_ann, only_move={'points': 'y', 'texts': 'y'}, ax=axs)
        # Сохраняем карту глобальную.
        plt.savefig(f"{self.filename}_{str(self.ts)}\\Маршруты_от_всех_баз.png")
        # Закрываем её.
        plt.close()


if __name__ == '__main__':
    # Считаем, сколько программа будет работать.
    try:
        start_time = time.time()
        # Выбор имени файла для считывания (в этом случае файл должен лежать в той же папке, что и исполняемый код).
        filename = 'ТПС_Юг.xlsx'
        # Допустимая прибавка к значениям скачка для CAPEX и OPEX.
        add_to_capex = 10
        # Создание экземпляра класса, который должен решать задачу об оптимизации.
        prob = OptimizationRoutes(os.curdir + '/' + filename, filename, add_to_capex)
        # Чтение файлов.
        prob.read_data()
        # Создание рабочей модели для оптимизации.
        prob.create_model()
        # Решение модели (проблемы), алгоритмы: 0 - cbc, 1 - glpk, 2 - ipopt.
        prob.solve_problem(0)
        # Создание файла с ответом на проблему.
        prob.create_answer()
        # Сохранение файла.
        prob.save_answer()
        # Рисование карт с распределением маршрутов. При ненадобности - закомментировать
        prob.routes_map()
        # Показываем время, за которое программа решила всю задачу.
        print('Время, за которое была решена задача: ', round((time.time() - start_time)))
    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fName = os.path.split(exc_tb.tb_frame.ff_code.co_filename)[1]
        print('Какая-то ошибка, информация:')
        print(exc_type, fName, exc_tb.tb_lineno)
