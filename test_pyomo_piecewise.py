import os
import time
import numpy as np
import openpyxl
from pyomo.environ import *
import pandas as pd
from datetime import datetime


class OptimizationRoutes:
    def __init__(self, file_path, ):
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
        self.file_path = file_path

    def read_data(self):
        print('Программа запушена, начинается чтение файла')
        self.supply = pd.read_excel(self.file_path, sheet_name='САРЕХ').set_index(
            ['Базис']).loc[:, ['Пропуск.способность']]
        zero_val = self.supply.index[self.supply['Пропуск.способность'] == 0].tolist()
        self.supply = [item for sublist in self.supply.drop(zero_val).values.tolist() for item in sublist]
        # Чтение исходных данных из файлов
        # Матрица размерностью кол-во нефтебаз X кол-во потребителей
        self.secondary_logistics = pd.read_excel(self.file_path, sheet_name='ВЛ').set_index(['ID']).drop(
            zero_val, axis=1).values.tolist()
        # Матрица размерностью спрос на разные нефтепродукты X кол-во потребителей
        self.azs_requirements = pd.read_excel(self.file_path, sheet_name='Объемы 2030').set_index(['ID']).drop(
            ['Спрос'], axis=1).values.tolist()
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
            zero_val).values.tolist()
        self.OPEX_var = [item for sublist in self.OPEX_var for item in sublist]
        # Матрица размерностью тариф на разные нефтепродукты X кол-во нефтебазы
        self.primary_logistics = pd.read_excel(self.file_path, sheet_name='ПЛ').set_index(['Базис']).drop(
            zero_val).values.tolist()
        self.km = pd.read_excel(self.file_path, sheet_name='Км').drop(zero_val, axis=1).set_index(['ID'])
        self.azs_info = pd.read_excel(self.file_path, sheet_name='спрАЗС').set_index(['ID']).loc[self.azs_name]
        self.channel = pd.read_excel(self.file_path, sheet_name='спр').loc[
                       0:3, ["Канал", "Канал крупно", "Канал3"]].set_index("Канал")
        self.base_info = pd.read_excel(self.file_path, sheet_name='спр').loc[
                         0:, ["НБ_подробно", "НБ", "Тип НБ"]].set_index(['НБ']).loc[self.base_name].values.tolist()
        # тут надо коммент про нефтебазы
        print('Файл прочитан, формируем модель для оптимизации и создаём переменные')
        return self.supply, self.secondary_logistics, self.azs_requirements, self.fuel_name, self.azs_name, \
            self.base_name, self.capex, self.OPEX_fix, self.OPEX_var, self.primary_logistics, self.km, \
            self.azs_info, self.channel, self.base_info

    def create_model(self):
        self.model = ConcreteModel()
        all_routes_name_separated = []
        self.all_routes_separated = []
        all_routes_name = []
        all_routes = []
        all_routes_costs = []
        self.azs_fuel_name = []
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
                    setattr(self.model, route_name, Var(bounds=(0, 9999), initialize=0))
                    globals()[f'{route_name}'] = getattr(self.model, route_name)
                    one_base.append(globals()[f'{route_name}'])
                    all_routes.append(globals()[f'{route_name}'])
            self.all_routes_separated.append(one_base)
            all_routes_name_separated.append(one_base_name)
            all_routes_costs.append(one_base_costs)
        for azs in range(len(self.azs_requirements)):
            for fuel in range(len(self.azs_requirements[azs])):
                self.azs_fuel_name.append(self.azs_name[azs] + '_' + self.fuel_name[fuel])

        for variable in range(len(all_routes)):
            setattr(self.model, 'constr' + '_' + all_routes_name[variable], Constraint(expr=all_routes[variable] >= 0))

        for base1 in range(len(self.all_routes_separated)):
            constraint = 0
            for route in range(len(self.all_routes_separated[base1])):
                constraint += self.all_routes_separated[base1][route]
            if self.supply[base1] == 0:
                setattr(self.model, 'constr' + '_' + 'base' + '_' + f'{base1}',
                        Constraint(expr=constraint == self.supply[base1]))
            else:
                setattr(self.model, 'constr' + '_' + 'base' + '_' + f'{base1}',
                        Constraint(expr=constraint <= self.supply[base1]))

        CAP_OP = []
        for base1 in range(len(self.capex)):
            temp = []
            for limit in range(len(self.capex[base1])):
                temp.append((self.capex[base1][limit] / 15) + (self.OPEX_fix[base1][limit] * 1000))
            CAP_OP.append(temp)

        sum_base = []
        for base1 in range(len(self.supply)):
            setattr(self.model, 'sum_' + f'{base1}', Var(bounds=(0, 3200)))
            sum_base.append(getattr(self.model, 'sum_' + f'{base1}'))

        for base1 in range(len(self.all_routes_separated)):
            setattr(self.model, 'cons_sum_' + f'{base1}',
                    Constraint(expr=sum_base[base1] == sum(self.all_routes_separated[base1])))

        domains = [0, 0.1, 0.1]
        for val in range(len(self.capex[0])):
            if val == len(self.capex[0]) - 1:
                domains.append(100 * (val + 1) + 0)
            else:
                domains.append(100 * (val + 1) + 0)
                domains.append(100 * (val + 1) + 0)

        CAPOP_range = []
        for base1 in range(len(CAP_OP)):
            temp = [0., 0.]
            for val in range(len(CAP_OP[base1])):
                temp.append(CAP_OP[base1][val])
                temp.append(CAP_OP[base1][val])
            CAPOP_range.append(temp)

        var_list = []
        for base1 in range(len(self.supply)):
            setattr(self.model, 'pw_var_' + f'{base1}', Var())
            var_list.append(getattr(self.model, 'pw_var_' + f'{base1}'))

        for base1 in range(len(self.supply)):
            setattr(self.model, 'pw_' + f'{base1}',
                    Piecewise(var_list[base1], sum_base[base1], pw_pts=domains, f_rule=CAPOP_range[base1],
                              pw_constr_type='EQ', pw_repn='CC'))

        def objective_func(*x):
            # Составляем функцию.
            func = quicksum(
                x[base_ind][elem] * (all_routes_costs[base_ind][elem]) for base_ind in 
                range(len(all_routes_costs)) for elem in range(len(all_routes_costs[base_ind])))
            for base2 in range(len(all_routes_costs)):
                func += var_list[base2]
            return func

        self.model.obj = Objective(expr=objective_func(*self.all_routes_separated), sense=minimize)

        transposed = [[self.all_routes_separated[j][i] for j in range(len(self.all_routes_separated))] for i in
                      range(len(self.all_routes_separated[0]))]
        for azs in range(len(transposed)):
            constraint = 0
            for route in range(len(transposed[azs])):
                constraint += transposed[azs][route]
            setattr(self.model, 'constr_' + 'azs_' + f'{azs}',
                    Constraint(expr=constraint == [item for sublist in self.azs_requirements for item in sublist][azs]))
        return self.model, self.all_routes_separated, self.azs_fuel_name

    def solve_problem(self):
        print('Модель создана, начинается оптимизация')
        # with SolverFactory("ipopt") as opt:
        #    opt.options.option_file_name = "ipopt.opt"
        #    with open("ipopt.opt", "w") as f:
        #        f.write("mumps_mem_percent 500\n")
        #    opt.solve(self.model, tee=True)
        # SolverFactory("glpk").solve(self.model, tee=True)
        SolverFactory("cbc.exe").solve(self.model, tee=True)
        print('Оптимизация закончилась, начинается сбор результатов из оптимизатора')

    def create_answer(self):
        answer = []
        self.answer_separated = []
        sum_base = []
        count = 0
        for v in self.model.component_data_objects(Var):
            # print(str(v), v.value,)
            try:
                if v.value <= 0.001:
                    answer.append(0.0)
                else:
                    answer.append(round(v.value, 6))
            except Exception as error:
                answer.append(v.value)
                print(error)

        for base3 in range(len(self.supply)):
            temp = []
            for azs in range(len(self.all_routes_separated[base3])):
                temp.append(answer[count])
                count += 1
            sum_base.append(sum(temp))
            self.answer_separated.append(temp)

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
            # Теперь вес от одной нефтебазы сравниваем с матрицей self.capex.
            result = float()
            for index in range(len(capex)):
                if index == 0:
                    if weight <= 100 + 0:
                        result = capex[index]
                    else:
                        pass
                elif index == len(self.capex) - 1:
                    if ((100 * index) + 0) <= weight:
                        result = capex[index]
                    else:
                        pass
                else:
                    if ((100 * index) + 0) <= weight <= ((100 * (index + 1)) + 0):
                        result = capex[index]
            return result

        self.answer_separated = list(map(list, zip(*self.answer_separated)))
        count = 0
        self.full_distribution = []
        print('Ответ сформирован, начинается формирование файла с результатами')
        for azs in range(len(self.azs_name)):
            for fuel in range(len(self.fuel_name)):
                _info = [item for sublist in self.azs_info.loc[[self.azs_name[azs]]].values.tolist() for item in
                         sublist]
                base_index = [self.answer_separated[count].index(self.answer_separated[count][val]) for val in
                              range(len(self.answer_separated[count])) if self.answer_separated[count][val] != 0]
                for ind in base_index:
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
                                + self.OPEX_var[ind] * self.answer_separated[count][ind] + self.primary_logistics[ind][
                                    fuel]
                                * self.answer_separated[count][ind] + self.secondary_logistics[azs][ind] *
                                self.answer_separated[count][ind], ]
                        self.full_distribution.append(temp)
                    except Exception as error:
                        temp = [self.azs_name[azs], self.fuel_name[fuel],
                                f'error: {error}',
                                *self.base_info[ind], *_info, self.channel.loc[[_info[0]]].values.tolist()[0][0],
                                self.channel.loc[[_info[0]]].values.tolist()[0][1], self.answer_separated[count][ind],
                                self.km.loc[[self.azs_name[azs]]].values.tolist()[0][ind],
                                CAPEX_solver(sum_base[ind], self.capex[ind]),
                                CAPEX_solver(sum_base[ind], self.OPEX_fix[ind]), self.OPEX_var[ind],
                                self.primary_logistics[ind][fuel], self.secondary_logistics[azs][ind], 0, 0, 0, 0, 0, 0,
                                0]
                        self.full_distribution.append(temp)
                count += 1

        self.full_distribution = pd.DataFrame(self.full_distribution,
                                              columns=["Название АЗС", "Тип топлива", "Имя базы", "НБ_подробно",
                                                       "Тип НБ",
                                                       "Канал", "АЗС/объект", "Широта", "Долгота", "Кластер", "Регион",
                                                       "Номер АЗС/объекта", "Макрорегион КПРА", "Канал крупно",
                                                       "Канал3",
                                                       "Объём поставки", "Км", "исх capex", "исх OPEX fix",
                                                       "исх OPEX var",
                                                       "исх ПЛ", "исх ВЛ", "capex итог", "capex", "OPEX fix",
                                                       "OPEX var",
                                                       "ПЛ", "ВЛ", "Итого затраты"])

        self.distribution_summary = []
        for base3 in range(len(self.base_name)):
            self.distribution_summary.append([self.base_name[base3], sum_base[base3], *sum_by_fuel[base3],
                                              CAPEX_solver(sum_base[base3], self.capex[base3]),
                                              CAPEX_solver(sum_base[base3], self.capex[base3]) / 15,
                                              CAPEX_solver(sum_base[base3], self.OPEX_fix[base3]) * 1000,
                                              self.OPEX_var[base3] * sum_base[base3],
                                              sum_primary[base3], secondary_sum[base3],
                                              CAPEX_solver(sum_base[base3], self.capex[base3]) / 15 / sum_base[base3],
                                              CAPEX_solver(sum_base[base3], self.OPEX_fix[base3])
                                              * 1000 / sum_base[base3], self.OPEX_var[base3], *self.base_info[base3]])
        self.distribution_summary = pd.DataFrame(self.distribution_summary,
                                                 columns=["Базис", "Объем перевалки", "Бензин 92",
                                                          "Бензин 95", "Бензин G-95", "Бензин 98",
                                                          "ДТ летнее", "ДТ зимнее", "capex", "capex в год",
                                                          "OPEX_fix", "OPEX_var", "ЖД тариф", "Транспорт",
                                                          "исхСАРЕХ", "исхОРЕХ_фикс", "исхОРЕХ_пер",
                                                          "НБ", "Тип НБ"])
        count = 0
        for azs in range(len(self.azs_requirements)):
            for fuel in range(len(self.fuel_name)):
                self.answer_separated[count].append(self.fuel_name[fuel])
                count += 1
        self.base_name.append('Тип топлива')
        self.answer_separated = pd.DataFrame(self.answer_separated, columns=self.base_name)
        sum_base.append(None)
        sum_base = np.array(sum_base)
        self.answer_separated.loc[-1] = sum_base
        self.answer_separated.index = self.answer_separated.index + 1  # shifting index
        self.answer_separated.sort_index(inplace=True)
        self.azs_fuel_name.insert(0, 'Сумма по топливу')
        obj_val = [None] * (len(self.answer_separated) + 1)
        obj_val[0] = value(self.model.obj)
        obj_val = pd.DataFrame(obj_val, columns=['Стоимость перевозки'])
        self.answer_separated = self.answer_separated.join(obj_val)
        self.answer_separated = self.answer_separated.set_index(pd.Index(np.array(self.azs_fuel_name)))
        print('Суммарная цена перевозок: ', obj_val)
        print('Распределение груза по нефтебазам: ', sum_base[:len(sum_base) - 1])
        print('Суммарный вес: ', sum(sum_base[:len(sum_base) - 1]))
        return self.answer_separated, self.distribution_summary, self.full_distribution,

    def save_answer(self):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'Объемы доставки'
        ts = datetime.now().strftime('%y.%m.%d_%H-%M')
        wb.save(filename=f'output_optimize_{str(ts)}.xlsx')
        with pd.ExcelWriter(f'output_optimize_{str(ts)}.xlsx', engine='openpyxl', mode='a',
                            if_sheet_exists="replace") as writer:
            self.answer_separated.to_excel(writer, sheet_name="Объемы доставки")
            self.full_distribution.to_excel(writer, sheet_name="Подробно")
            self.distribution_summary.to_excel(writer, sheet_name="Затраты суммарно")
        print('Файлы сформированы и сохранены, программа завершает свою работу')


if __name__ == '__main__':
    start_time = time.time()
    prob = OptimizationRoutes(os.curdir + '/ТПС_Тест.xlsx')
    prob.read_data()
    prob.create_model()
    prob.solve_problem()
    prob.create_answer()
    prob.save_answer()
    print('Время, за которое была решена задача: ', (time.time() - start_time))
