import os
import sys
import time
import PyQt5
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtWidgets
from pyomo_piecewise import OptimizationRoutes
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog


if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        loadUi(os.curdir + '\\support_files\\Interface.ui', self)
        self.folderText.setPlaceholderText(os.path.abspath(os.path.curdir) + '\\support_files\\ТПС_Тест.xlsx')
        self.browseButton.clicked.connect(self.browseFiles)
        self.acceptButton.clicked.connect(self.acceptParams)

    def acceptParams(self):
        indexmaps = [self.mapsGroup.buttons()[x].isChecked() for x in range(len(self.mapsGroup.buttons()))].index(True)
        indexsolvers = [
            self.solversGroup.buttons()[x].isChecked() for x in range(len(self.solversGroup.buttons()))].index(True)
        file_path = self.folderText.toPlainText() if self.folderText.toPlainText() != '' \
            else os.path.abspath(os.path.curdir) + '\\support_files\\ТПС_Тест.xlsx'
        filename = os.path.basename(file_path)
        add_to_capex = self.capexBox.value()
        self.hide()
        print(f'Работаем с файлом {filename}')
        try:
            start_time = time.time()
            task = OptimizationRoutes(file_path, filename, add_to_capex)
            task.read_data()
            task.create_model()
            task.solve_problem(indexsolvers)
            task.create_answer()
            task.save_answer()
            if indexmaps == 0 and filename != 'ТПС_Тест.xlsx':
                task.routes_map()
            else:
                pass
            print('Время, за которое была решена задача: ', round((time.time() - start_time)))
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fName = os.path.split(exc_tb.tb_frame.ff_code.co_filename)[1]
            print(exc_type, fName, exc_tb.tb_lineno)
            print("Что-то пошло не так. Проверьте, нет ли ошибок в считываемом файле, "
                  "либо обратитесь к автору для исправления багов, если проблема в коде.")
        self.show()

    def browseFiles(self):
        fpath = QFileDialog.getOpenFileName(self, "Select Excel file to import",
                                            os.path.abspath(os.path.curdir), "Excel (*.xls *.xlsx)")[0]
        self.folderText.setText(fpath)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mood_example = MainWindow()
    mood_example.show()
    sys.exit(app.exec_())
