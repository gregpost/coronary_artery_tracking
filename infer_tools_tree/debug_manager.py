# Модуль отладочных функций, обеспечивает сохранение промежуточных снимков в файлы, продолжение выполнения алгоритма с
# определённого этапа, показ статистики лейболизованных снимоков, показ времени выполнения этапов программы.

import os
import shutil
import time
import SimpleITK as sitk
import itk
import re
import sys
import traceback
import glob
import pickle
import numpy as np
from networkx.classes.graph import Graph
from PIL import ImageFont, ImageDraw, Image
from colorama import Fore
import data_tools as data
import pygorpho as pg
import torch

from typing import List, Tuple, Any
from numpy import long
from datetime import date

from pip._vendor.colorama import ansi

none_image: sitk.Image = sitk.Image([0, 0, 0], sitk.sitkInt8)


def IsNoneImage(_image: sitk.Image) -> bool:
    if _image.GetWidth() == 0:
        return True
    else:
        return False


# noinspection PyMethodMayBeStatic
class DebugManager:
#CONVERT_TO_CPP_OFF
    save_files: bool = False  # Активировать сохранение промежуточных изображений в файлы
    load_files: bool = False  # Брать готовые изображения из файлов, позволяет пролдолжить выполнение с ранее достигнутого этапа
    load_files_exclusions: List[str] = []  # Список методов, исключающихся из автозагрузки файлов, это позволит подгружать только необходимые готовые данные
    use_timer: bool = False  # Отсчёт времени выполнения этапов алгоритма
    clear_old_files: bool = False  # Удалять предыдущие исследования снимка при перезапуске
    use_numbers: bool = True  # Нумеровать выходные изображения
    short_console_output: bool = False  # Флаг упрощённого вывода в консоль указывает, чтобы выводить в консоль только ключевые этапы выполнения программы, при этом полная статистика будет записана в логи
    console_output: bool = True  # Вести вывод лога в консоль
    output_log: bool = True  # Вести запись лога в файл
    output_info: bool = True  # Вести запись информации о статусе выполнения программы для клиента

    def __init__(self, _output_folder: str = '', _study_name: str = ''):
        """
        Инициализация отладочного модуля

        :param _output_folder: полный путь к файлу чтения/записи результирующих снимков
        """
        self.output_data_folder = ''            # Путь к каталогу записи временных файлов
        self.output_info_path = ''              # Путь к каталогу записи лога промежуточных стадий выполнения для клиента
        self.log_path = ''                      # Путь к файлу записи логов
        self._output_file = None                # Указатель на файл записи промежуточных этапов алгоритма для вывода на клиенте
        self.study_name = _study_name           # Название исследования
        self.image_spacing = None     # Спэйсинг исследования
        self.image_origin = None       # Ориджин исследования
        self.current_folder = ''                # Текущий локальный путь к файлу

        # Служебные внутренние переменные таймера
        self.time_list: List[int] = []  # Список чекпоинтов таймера
        self.start_time: float = time.time()  # Точка отсчёта
        self.pause_start_time: float = 0  # Точка начала паузы
        self.pause_duration: float = 0  # Продолжительность паузы
        self.check_point: int = -1

        self.file_index: int = 1   # Номер файла по счёту сохранения, позволяющий индексировать названия файлов, выстраивая тем самым последовательноасть сохранения

        # Запуск таймера
        if self.use_timer:
            self.start_time = time.time()
            self.time_list.clear()
        if _output_folder != '':
            self.SetOutputFolder(_output_folder)
            self.SetLogFile(_output_folder)

        # Удаление предыдущего исслдедования снимков если стоит флаг clear_old_files
        if self.clear_old_files and not self.load_files and os.path.isdir(self.output_data_folder):
            try:
                _old_path: str = self.output_data_folder.rstrip('\\') + '_old\\'
                os.rename(self.output_data_folder, _old_path)
                shutil.rmtree(_old_path)
            except PermissionError:
                self.Print('WARNING (DebugTools.Init): Не удалось очистить старые файлы в папке %s. Отказано в доступе.' % self.output_data_folder, _hard_priority=True)

    def GetOutputFolder(self):
        return self.output_data_folder

    def SetLogFile(self, _ouptut_path: str):
        if self.output_log:
            # Подключение логирования
            log_path = _ouptut_path + "logs/"

            # Проверка существования и создание дирректории для записи логов
            if not os.path.isdir(log_path):
                os.mkdir(log_path)

            today = date.today()
            date_str = today.strftime("%d-%m-%Y")

            full_log_path: str = log_path + 'output_log_' + date_str
            for i in range(1, 100):
                new_full_log_path = full_log_path + '_' + str(i) + '.txt'
                if not os.path.isfile(new_full_log_path):
                    full_log_path = new_full_log_path
                    break
                if i == 99:
                    full_log_path += '_0.txt'

            # Запомнить путь файла запси логов
            self.log_path = full_log_path

    def SetOutputFolder(self, _ouptut_path: str):
        if not os.path.isdir(_ouptut_path):
            try:
                os.mkdir(_ouptut_path)
            except Exception:
                self.Print('ERROR (DebugManager.SetOutputFolder): Could not create temp output folder. Probably wrong path or permisson proplem: %s' % _ouptut_path)
                return

        self.output_data_folder = _ouptut_path + "data/"

        if not os.path.isdir(self.output_data_folder):
            os.mkdir(self.output_data_folder)

    def SetSpacing(self, _image_spacing: List[float]):
        """
        Позволяет задать спэйсинг, который будет задаваться
        прочитанному из файла изображению. Это позволяет избежать искажений спэйсинга
        во время записи/чтения изображения из файла
        Пример ошибки: https://disk.yandex.ru/d/SHV5zbynZ5HMZg

        :param _image_spacing: Спэйсинг изображения
        """
        self.image_spacing = _image_spacing

    def SetOrigin(self, _image_origin: List[float]):
        """
        Позволяет задать ориджин, который будет задаваться
        прочитанному из файла изображению. Это позволяет избежать искажений ориджина
        во время записи/чтения изображения из файла
        Пример ошибки: https://disk.yandex.ru/d/SHV5zbynZ5HMZg

        :param _image_origin: Ориджин изображения
        """
        self.image_origin = _image_origin

    def NormalizeSlashes(self, _source_string: str) -> str:
        """
        Заменить все обтаные слэши на прямые и удалить избыточные слэши

        :param _source_string: Исходная строка
        :return: Строка с заменёнными обатными слэшами на слэши
        """
        result_string: str = _source_string.replace('\\', '/')
        while result_string.find('//') > -1:
            result_string = result_string.replace('//', '/')
        return result_string

    def IsFileExists(self, _file_name: str, _image_extension: str = 'nii.gz') -> Tuple[bool, str]:
        """
        Проверка на наличие файла

        :param _file_name: Имя файла без расширения
        :param _image_extension: Расширение файла: 'nii.gz', 'nii', 'nrrd' и др.
        :return:
        1) флаг, говорящий о том, что файл существует
        2) полный путь к файлу, если он существует
        """
        # Если программист сам включил расширение в строку с путём к файлу, то удаляем его и подставляем расширение
        # из переменной _image_extension
        _file_name = re.sub(r'([^/\\])\..*', r'\1', _file_name)
        _file_name += '.' + _image_extension

        folder_path: str = f'{self.output_data_folder}/{self.current_folder}/'
        folder_path = self.NormalizeSlashes(folder_path)

        # Если был включён номер в название файла, например, 3_binary_mask.nii.gz, но
        # номер неизвестен, то пытаемся с помощью поиска определить номер
        image_names: List[str] = glob.glob(f'{folder_path}/*{_file_name}')
        image_names_count: int = len(image_names)
        if image_names_count > 0:
            for image_name in image_names:
                image_name = self.NormalizeSlashes(image_name)
                if re.search(f'{folder_path}[0-9]*_?{_file_name}', image_name):
                    return True, image_name
        return False, ''

    def ReadNumpyArrayFromFile(self, _path: str, _image_extension: str = 'nii.gz', _force_use: bool = False, _path_is_local: bool = True) -> np.ndarray:
        """
        Чтение файла c SimpleITK изображением в формате трёхмерного массива

        :param _path: Если _path_is_local = True, то этот путь должен содержать только имя файла и расширение, иначе полный путь к файлу
        :param _image_extension: Расширение файла: 'nii.gz', 'nii', 'nrrd' и др.
        :param  _force_use: Флаг принудительной загрузки файла, независимо от других условий
        :param _path_is_local: Определяет способ чтения пути к файлу
        :return: Изображение, прочитанное из файла
        """
        image: sitk.Image = self.ReadImageFromFile(_path, _image_extension=_image_extension, _force_use=_force_use, _path_is_local=_path_is_local)
        if IsNoneImage(image):
            return np.array([])
        else:
            array: np.ndarray = sitk.GetArrayFromImage(image)
            return array

    def ReadImageFromFile(self, _path: str, _image_extension: str = 'nii.gz', _force_use: bool = False, _path_is_local: bool = True) -> sitk.Image:
        """
        Чтение файла c SimpleITK изображением

        :param _path: Если _path_is_local = True, то этот путь должен содержать только имя файла и расширение, иначе полный путь к файлу
        :param _image_extension: Расширение файла: 'nii.gz', 'nii', 'nrrd' и др.
        :param  _force_use: Флаг принудительной загрузки файла, независимо от других условий
        :param _path_is_local: Определяет способ чтения пути к файлу
        :return: Изображение, прочитанное из файла
        """
        if self.output_data_folder == '':
            return none_image

        if self.load_files or _force_use:

            # Если программист сам включил расширение в строку с путём к файлу, то удаляем его и подставляем расширение
            # из переменной _image_extension
            path: str = re.sub(r'[^/\\]\..*', r'', _path)
            path += '.' + _image_extension

            if _path_is_local:
                is_file_exists: bool
                is_file_exists, path = self.IsFileExists(path, _image_extension=_image_extension)
                if not is_file_exists:
                    return none_image

            if not _force_use:
                if len(self.load_files_exclusions) > 0:
                    for excluded_item in self.load_files_exclusions:
                        if path.find(excluded_item) >= 0:
                            return none_image

            if not _force_use:
                if len(self.load_files_exclusions) > 0:
                    for excluded_item in self.load_files_exclusions:
                        if path.find(excluded_item) >= 0:
                            return none_image

            if os.path.isfile(path):
                reader = sitk.ImageFileReader()
                image_io: str = self.GetImageIOFromExtension(_image_extension)
                reader.SetImageIO(image_io)
                reader.SetFileName(path)
                image: sitk.Image = reader.Execute()

                # Иногда бывает, что при записи/чтении изображения из файла теряется точность ориджина
                # в 7-м знаке и появляется подобная ошибка: https://disk.yandex.ru/d/SHV5zbynZ5HMZg
                # Для избежания подобных ошибок установим заданный разработчиком во время создания объекта
                # DebugManager ориджин и спэйсинг для текущего изображения
                if self.image_origin is not None:
                    image.SetOrigin(self.image_origin)
                if self.image_spacing is not None:
                    image.SetSpacing(self.image_spacing)

                self.CheckTimer(False, False)
                return image

        return none_image

    def GetImageIOFromExtension(self, _extension: str):
        """
        Возвращет Image IO по заданному расширению
        Библиотека поддерживает типы файлов из списка:
            BMPImageIO ( *.bmp, *.BMP )
            BioRadImageIO ( *.PIC, *.pic )
            Bruker2dseqImageIO
            GDCMImageIO
            GE4ImageIO
            GE5ImageIO
            GiplImageIO ( *.gipl *.gipl.gz)
            HDF5ImageIO
            JPEGImageIO ( *.jpg, *.JPG, *.jpeg, *.JPEG )
            LSMImageIO ( *.tif, *.TIF, *.tiff, *.TIFF, *.lsm, *.LSM )
            MINCImageIO ( *.mnc, *.MNC )
            MRCImageIO ( *.mrc, *.rec )
            MetaImageIO ( *.mha, *.mhd )
            NiftiImageIO ( *.nia, *.nii, *.nii.gz, *.hdr, *.img, *.img.gz )
            NrrdImageIO ( *.nrrd, *.nhdr )
            PNGImageIO ( *.png, *.PNG )
            StimulateImageIO
            TIFFImageIO ( *.tif, *.TIF, *.tiff, *.TIFF )
            VTKImageIO ( *.vtk )
            DICOM (Если расширение отсутствует или указано значение dicom)
        :param _extension:
        :return: Тип ридера/врайтера, соответствующий переданному расширению
        """
        _reader_type: str = ''
        if _extension == 'bmp':
            _reader_type = 'BMPImageIO'
        elif _extension == 'pic':
            _reader_type = 'BioRadImageIO'
        elif _extension == 'gipl':
            _reader_type = 'GiplImageIO'
        elif _extension == 'lsm':
            _reader_type = 'LSMImageIO'
        elif _extension == 'mnc':
            _reader_type = 'MINCImageIO'
        elif (_extension == 'mrc') or (_extension == 'rec'):
            _reader_type = 'MRCImageIO'
        elif (_extension == 'mha') or (_extension == 'mhd'):
            _reader_type = 'MetaImageIO'
        elif _extension == 'png':
            _reader_type = 'PNGImageIO'
        elif (_extension == 'tif') or (_extension == 'tiff'):
            _reader_type = 'TIFFImageIO'
        elif _extension == 'vtk':
            _reader_type = 'VTKImageIO'
        elif (_extension == 'nii')or (_extension == 'nii.gz') or (_extension == 'nia') or (_extension == 'hdr') or (_extension == 'img'):
            _reader_type = 'NiftiImageIO'
        elif (_extension == 'nrrd') or (_extension == 'nhdr'):
            _reader_type = 'NrrdImageIO'
        elif (_extension == 'jpeg') or (_extension == 'jpg'):
            _reader_type = 'JPEGImageIO'
        elif _extension == 'dcm':
            _reader_type = 'DICOM'
        elif _extension == '':
            raise Exception('Ошибка определения типа изображения! Не задано расширение файла: _extension=\'\'')
        else:
            raise Exception('Ошибка определения типа изображения! Неизвестный тип файла: %s' % _extension)
        return _reader_type

    def ReadCTImageFile(self, _path: str, _extension: str = '', _path_is_local: bool = False) -> sitk.Image:
        """
        Осуществляет чтение файла со снимком компьютерной томографии указанного типа

        :param _path: Полный путь к файлу
        :param _extension: Расширение файла
        :param _path_is_local: Указывает на то, что передан локальный путь к файлу
        :return:
        """
        try:
            # Если расширение не указано явно, то попытка определить его по имени файла
            if _extension == '':
                parsed_path: List[str] = _path.split('.')
                if len(parsed_path) > 1:
                    _extension = parsed_path[1]
                else:
                    _extension = 'dcm'

            _extension = _extension.lower()

            # Основываясь на расширении переданного файла задаётся тип ридера
            reader_type: str = self.GetImageIOFromExtension(_extension)

            if _path_is_local:
                _path = self.output_data_folder + _path

            if reader_type == 'DICOM':
                if os.path.isdir(_path):
                    image: sitk.Image = data.ReadDicomFile(_path)
                    self.SetSpacing(image.GetSpacing())
                    self.SetOrigin(image.GetOrigin())
                    return image
            else:
                if os.path.isfile(_path):
                    reader = sitk.ImageFileReader()
                    reader.SetImageIO(reader_type)
                    reader.SetFileName(_path)
                    reader.SetOutputPixelType(sitk.sitkFloat32)
                    image: sitk.Image = reader.Execute()
                    self.SetSpacing(image.GetSpacing())
                    self.SetOrigin(image.GetOrigin())
                    return image

            self.Print('WARNING (ReadCTImageFile): Снимок не найден по адресу: %s.' % _path)
            return none_image
        except Exception:
            raise Exception('ERROR (ReadCTImageFile): Ошибка чтения снимка: %s.\nКод ошибки: (%s) %s\n%s' % (_path, sys.exc_info()[0], sys.exc_info()[1], traceback.format_exc()))

    def UnselectDirectory(self, _steps_count: int = 1):
        """
        Возвращает указатель записи промежуточных снимков из текущей директории в вышестоящую, но не выше корня

        :param _steps_count: Количество шагов на которые надо подняться от текущей директории записи, если 0 то поднимается до корневой
        :return:
        """
        self.file_index = 1
        if self.save_files:
            if _steps_count == 0:
                self.current_folder = ''
                return
            for i in range(_steps_count):
                if self.current_folder != '':
                    if self.current_folder.find('/') != -1:
                        self.current_folder = self.current_folder.rpartition('/')[0]
                    else:
                        self.current_folder = ''
                else:
                    break

    def SelectDirectory(self, _dir_path: str, _absolute_path: bool = False):
        """
        Задаёт путь к текущей директории для сохранения промежуточных снимков и при необходимости создаёт её

        :param _dir_path: Путь к директории
        :param _absolute_path: если True, то путь задаётся от корня output_folder, если False то от текущей папки записи
        """
        self.file_index = 1
        if self.save_files:
            _dir_path = _dir_path.strip('\\').strip('/')

            if not _absolute_path:
                self.current_folder = f'{self.current_folder}/{_dir_path}'
            else:
                self.current_folder = _dir_path

            if not os.path.isdir(self.output_data_folder + self.current_folder):
                os.mkdir(self.output_data_folder + self.current_folder)

    def SaveImageToPictureFile(self, _image: sitk.Image, _file_name: str, _output_extension: str = 'jpg', _path_is_local: bool = True, _short_flag: bool = False):
        """
        Сохранение исследования в файлы картинок заданного формата

        :param _image: Изображение, которое нужно сохранить в файл
        :param _file_name: Если _path_is_local = True, то нужно указать название папки выходных снимков
        :param _path_is_local: Является ли указанный путь локальным или глобальным
        :param _short_flag: Если True, то результирующее изображение будет конвертировано в однобайтовый формат
        """
        if self.output_data_folder == '':
            self.Print('%s: WARNING (SaveImageToFile) Не удалось сохранить файл. Не задан путь к файлу.' % self.study_name, _hard_priority=True)
            return
        if IsNoneImage(_image):
            self.Print('%s: WARNING (SaveImageToFile) Передано нулевое изображение. Будет сформирован пустой файл: %s' % (self.study_name, _file_name), _hard_priority=True)
            _image = sitk.Image(1, 1, 1, sitk.sitkUInt8)

        if _image is not None:
            self.CheckTimer(False)

            reader_type: str = self.GetImageIOFromExtension(_output_extension)

            # Преобразование изображения в однобайтовый формат для превращения в jpeg
            if _short_flag:
                _image = sitk.RescaleIntensity(_image)
                _image = sitk.Cast(_image, sitk.sitkUInt8)

            writer = sitk.ImageFileWriter()
            writer.SetImageIO(reader_type)
            if _path_is_local:
                _file_name = self.output_data_folder + self.current_folder + '/' + _file_name

            self.Print('Сохранение в файл в формате %s: %s' % (_output_extension, _file_name))

            if not os.path.isdir(_file_name):
                os.mkdir(_file_name)

            for f in os.listdir(_file_name):
                os.remove(os.path.join(_file_name, f))

            for i in range(_image.GetDepth()):
                _image_slice: sitk.Image = _image[:, :, i]
                sitk.WriteImage(_image_slice, _file_name + '/' + str(i) + '.' + _output_extension)

            self.CheckTimer(False, False)
        else:
            self.Print('Ошибка сохранения в файл: %s. Изображение отсутствует!' % _file_name)

    def SaveItkImage(self, _image: itk.Image, _filename: str):
        """
        Сохраняет ITK изображение в файл

        :param _image: ITK изображение
        :param filename: Имя файла или полный путь к файлу
        """
        writer = itk.ImageFileWriter[_image].New()
        writer.SetInput(_image)
        writer.SetFileName(_filename)
        writer.Update()

    def SaveImageToFile(self, _image: sitk.Image, _file_name: str, _image_extension: str = 'nii.gz', _force_use: bool = False,
                        _path_is_local: bool = True, _ignore_numbers: bool = False):
        """
        Сохранение изображения в файл

        :param _image: Изображение, которое нужно сохранить в файл
        :param _file_name: Если _path_is_local = True, то нужно указать имя файла без расширения, иначе полный путь к файлу
        :param _image_extension: Расширение файла: 'nii.gz', 'nii', 'nrrd' и др.
        :param _force_use: Принудительное сохранение файла, независимо от флага save_files
        :param _path_is_local: Является ли указанный путь локальным или глобальным
        :param _ignore_numbers: Принудительное отключение нумерации выходного файла, независимо от флага use_numbers
        """
        if _path_is_local and self.output_data_folder == '':
            self.Print('%s: WARNING (SaveImageToFile) Не удалось сохранить файл. Не задан локальный путь к файлу.' % self.study_name, _hard_priority=True)
            return
        if IsNoneImage(_image):
            self.Print('%s: WARNING (SaveImageToFile) Передано нулевое изображение. Будет сформирован пустой файл: %s' % (self.study_name, _file_name), _hard_priority=True)
            _image = sitk.Image(1, 1, 1, sitk.sitkUInt8)

        if self.save_files or _force_use:
            if _image is not None:
                self.PauseTimer()
                writer = sitk.ImageFileWriter()
                image_io: str = self.GetImageIOFromExtension(_image_extension)
                writer.SetImageIO(image_io)

                # Если программист сам включил расширение в строку с путём к файлу, то удаляем его и подставляем расширение
                # из переменной _image_extension
                _file_name = re.sub(r'\..*', r'', _file_name)
                _file_name += '.' + _image_extension

                if _path_is_local:
                    _file_name = self.output_data_folder + self.current_folder + '/' + _file_name

                self.Print('Сохранение в файл: %s' % _file_name)

                # Добавление к имени файла индекса, показывающего очередность сохранения
                if not _ignore_numbers and self.use_numbers:
                    _file_name = _file_name.replace('\\\\', '/')
                    _file_name = _file_name.replace('\\', '/')
                    _file_name = _file_name.replace('//', '/')
                    _file_name = re.sub('/([^/]+?)(?=.' + _image_extension + ')', '/' + str(self.file_index) + '_' + '\g<1>', _file_name)
                    self.file_index += 1

                writer.SetFileName(_file_name)
                writer.UseCompressionOn()
                writer.Execute(_image)
                self.ContinueTimer()
            else:
                self.Print('Ошибка сохранения в файл: %s. Изображение отсутствует!' % _file_name)

    def SaveNumpyArrayToFile(
            self,
            _array: np.ndarray,
            _file_name: str,
            _reference_image: sitk.Image,
            _pixel_id_value: int = -1,
            _image_extension: str = 'nii.gz',
            _force_use: bool = False,
            _path_is_local: bool = True,
            _ignore_numbers: bool = False,
            _dilation_radius_in_pixels: int = 0.0,
            _image_spacing: List[float] = [],
            _is_normalized_spacing_required: bool = False
    ):
        """
        Сохранение изображения в виде Numpy-массива в файл. ВАЖНО: если тип изображения int32 или float32, то сохранение может длиться около 20 секунд.
        Так что если не нужно выводить дробные знаки или большие чилса, то лучше указать _pixel_id_value=sitk.sitkInt16 и сохранение займёт 5 секунд

        :param _array: Изображение в виде Numpy-массива, которое нужно сохранить в файл
        :param _file_name: Если _path_is_local = True, то нужно указать имя файла без расширения, иначе полный путь к файлу
        :param _reference_image: Эталонное изображение, из которого будут взяты только спэйсинг и другие мета-данные
        :param _pixel_id_value: Позволяет задать тип пикселя (uint8, int32 и т.д.). Если -1 значит тип пикселя определяется
        по оригинальному изображению
        :param _image_extension: Расширение файла: 'nii.gz', 'nii', 'nrrd' и др.
        :param _force_use: Принудительное сохранение файла, независимо от флага save_files
        :param _path_is_local: Является ли указанный путь локальным или глобальным
        :param _ignore_numbers: Принудительное отключение нумерации выходного файла, независимо от флага use_numbers
        :param _dilation_radius_in_pixels: Радиус дилатации. Необходимо для увеличения точек на итоговом изображении для улучшения видимости
        :param _image_spacing: Расстояние между срезами в мм по аксимальной, корональной и сагиттальной осям соответственно
        :param _is_normalized_spacing_required: Если True, то споэйсинг будет единичным = [1.0, 1.0, 1.0]
        """
        new_array: np.ndarray = _array.copy()
        if self.save_files or _force_use:
            reference_image: sitk.Image = sitk.Image(_reference_image)
            if _is_normalized_spacing_required:
                _image_spacing = [1.0, 1.0, 1.0]
            if len(_image_spacing) > 0:
                reference_image.SetSpacing(_image_spacing)

            if _dilation_radius_in_pixels > 0:
                lineSteps: np.ndarray
                lineLens: np.ndarray
                lineSteps, lineLens = pg.strel.flat_ball_approx(_dilation_radius_in_pixels)
                new_array = pg.flat.linear_dilate(new_array, lineSteps, lineLens)
            image: sitk.Image = data.ConvertNumpyArrayToSimpleItkImage(new_array, reference_image, _pixel_id_value=_pixel_id_value)
            self.SaveImageToFile(image, _file_name, _image_extension=_image_extension, _force_use=_force_use, _path_is_local=_path_is_local, _ignore_numbers=_ignore_numbers)

    def SaveObjectToFile(self, _object: Any, _file_name: str):
        """
        Сохранить объект в бинарный файл

        :param _object: Любая переменная или объект, который нужно сохранить
        :param _file_name: Имя файла без расширения
        """
        file_path: str = self.output_data_folder + self.current_folder + '/' + _file_name + '.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(_object, f, pickle.HIGHEST_PROTOCOL)

    def LoadObjectFromFile(self, _file_name: str) -> Any:
        """
        Загрузить объект из бинарного файла

        :param _file_name: Имя файла без расширения
        """
        file_extension: str = 'pkl'
        is_file_exists: bool
        is_file_exists, file_path = self.IsFileExists(_file_name, _image_extension=file_extension)
        if is_file_exists:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            return False

    @staticmethod
    def DrawTextToImage(_path_to_file: str, _text: str):
        """
        Рисует текст на картинке и сохраняет её в файл

        :param _path_to_file:
        :param _text:
        :return:
        """
        text_arr: List[str] = _text.split('\n')
        img = Image.new('RGB', (620, 330), color=(73, 109, 137))
        font = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSerif-Regular.ttf", 20)
        d = ImageDraw.Draw(img)
        for i in range(len(text_arr)):
            d.text((50, 50 + i*40), text_arr[i], fill=(255, 255, 0), font=font)
        img.save(_path_to_file)

    def ShowImageInfo(self, _image: sitk.Image, _title: str = 'image'):
        """
        Выводит подробную информацию о типе и размерности изображения, а также метаданные изображения

        :return:
        """
        im_size: str = str(_image.GetSize())
        im_dim: str = str(_image.GetDimension())
        im_pixel_type: str = _image.GetPixelIDTypeAsString()
        im_origin: str = str(_image.GetOrigin())
        im_spacing: str = str(_image.GetSpacing())
        im_direction: str = str(_image.GetDirection())

        self.Print('')
        self.Print('Сведения об изображении %s:' % _title)
        self.Print('Размер: %s' % im_size)
        self.Print('Количество измерений: %s' % im_dim)
        self.Print('Тип пикселя: %s' % im_pixel_type)
        self.Print('Смещение от начала координат (origin): %s' % im_origin)
        self.Print('Размер вокселя в физических единицах (spacing): %s' % im_spacing)
        self.Print('Угол поворота (direction): %s' % im_direction)
        self.Print('')

    def GetLabelStats(self, _image: sitk.Image, _title: str = 'Image labels statistic',
                      _print_to_file: bool = False, _print_to_console: bool = True, _labeled_image: sitk.Image = None):
        """
        Вывод отладочной информации об отдельных сегментах изображения

        :param _image: Изображение, по которому нужно вывести информацию
        :param _title: Название изображения
        :param _print_to_file: Флаг записи сообщения в файле
        :param _print_to_console: Флаг вывода сообщения в консоль
        :param _labeled_image: Маркированный объём с данными об объектах на изображении _image. Если не задано, то считается автоматически по изображению _image
        """
        if _labeled_image is None:
            connected_component_filter = sitk.ConnectedComponentImageFilter()
            _labeled_image = connected_component_filter.Execute(_image)

        stats_filter = sitk.LabelStatisticsImageFilter()
        stats_filter.Execute(_image, _labeled_image)
        intensity_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
        intensity_stats_filter.Execute(_image, _labeled_image)

        label_id: int = 1
        output_message: List[str] = list()
        output_message.append('-----------------------------------------')
        output_message.append('Статистика объёмов сегментов изображения {0}:'.format(_title))
        while label_id < stats_filter.GetNumberOfLabels():
            pixels_count: long = stats_filter.GetCount(label_id)
            physical_size: float = 0
            if label_id <= intensity_stats_filter.GetNumberOfLabels():
                physical_size = intensity_stats_filter.GetPhysicalSize(label_id)
            output_message.append(
                'Label {0}: Количество пикселей = {1}; Физический размер = {2}'.format(label_id, pixels_count,
                                                                                       physical_size))
            label_id += 1
        output_message.append('-----------------------------------------')

        if _print_to_file and self.output_data_folder != '':
            dir_path = self.output_data_folder + 'logs/'

            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            write_flag: str
            if not os.path.isfile(dir_path + 'ShowLabelStats.txt'):
                write_flag = 'w'
            else:
                write_flag = 'a'
            f = open(dir_path + 'ShowLabelStats.txt', write_flag, encoding='utf-8')

        for text_line in output_message:
            if _print_to_console:
                self.Print(text_line)
            if _print_to_file:
                f.write(text_line + '\r\n')

        if _print_to_file:
            f.close()

        self.CheckTimer(False, False)

    def WriteClientOutput(self, _output_message: str, _new_file: bool = False):
        """
        Производит запись сообщения об этапе выполнения алгоритма для отображения на стороне клиента

        :return:
        """
        self.Print(_output_message, _color=Fore.MAGENTA)
        if self.output_info_path != '' and self.output_info:
            write_flag: str
            if not _new_file and os.path.isfile(self.output_info_path):
                write_flag = 'a'
            else:
                write_flag = 'w'
            self._output_file = open(self.output_info_path, write_flag, encoding='utf-8')
            self._output_file.write(_output_message + '\n')
            self._output_file.close()

    def Print(self, _output_message: str, _hard_priority: bool = False, _color: ansi.AnsiFore = Fore.RESET):
        """
        Логирование программного вывода текста в консоль и опционально в файл

        :param _output_message: Выходная строка, которую надо напечатать
        :param _hard_priority: Указывыат, что данное сообщение будет выведено в консоль даже при наличии флага short_console_output
        :param _color: Цвет текста
        :return:
        """
        if self.output_log and self.log_path != '':
            write_flag: str
            if os.path.isfile(self.log_path):
                write_flag = 'a'
            else:
                write_flag = 'w'
            log_file = open(self.log_path, write_flag, encoding='utf-8')
            log_file.write(_output_message + '\r\n')
            log_file.close()

        if (not self.short_console_output or _hard_priority) and self.console_output:
            print(_color + _output_message)

    def PauseTimer(self):
        self.pause_start_time = time.time()

    def ContinueTimer(self):
        self.pause_duration += time.time() - self.pause_start_time

    def CheckTimer(self, _show: bool = True, _save_value: bool = True, _units: str = 'sec') -> float:
        """
        Запись и отображение времени выполнения операции и последующий рестарт таймера

        :param _show: Выводить данные таймера в консоль
        :param _save_value: Учитывать данные при общем подсчёте времени
        :param _units: Елиницы имерения: 'seconds', 'micro', 'nano'
        """
        if self.use_timer:
            time_step: float = time.time() - self.start_time - self.pause_duration
            self.pause_duration = 0

            if _save_value:
                self.time_list.append(time_step)
            if _show:
                if _units == 'sec':
                    formatted_time_step: str = "{:.2f}".format(time_step)
                    self.Print('--- %s секунд ---' % formatted_time_step, _color=Fore.BLUE)
                elif _units == 'micro':
                    formatted_time_step: str = "{:.2f}".format(time_step * 1000)
                    self.Print('--- %s микросекунд ---' % formatted_time_step, _color=Fore.BLUE)
                elif _units == 'nano':
                    formatted_time_step: str = "{:.2f}".format(time_step * 1000000)
                    self.Print('--- %s наносекунд ---' % formatted_time_step, _color=Fore.BLUE)
            self.start_time = time.time()

            return time_step

    def ResetTimer(self) -> float:
        """
        Сброс таймера
        """
        self.CheckTimer(_show=False)

    def AggregateTimer(self, _print_result: bool = True) -> int:
        """
        Получение суммарного значения времени от начала отчёта

        """
        if self.use_timer:
            self.CheckTimer(False)
            time_sum: float = 0
            for time_step in self.time_list:
                time_sum += time_step
            formatted_time_sum: str = "{:.2f}".format(time_sum)
            self.Print('Суммарное время: %s секунд' % formatted_time_sum, _color=Fore.MAGENTA)
            return time_sum

    def SaveCheckPoint(self):
        """
        Запомнить номер текущего отрезка времени

        :return:
        """
        if self.use_timer:
            self.CheckTimer(False)
            self.check_point = len(self.time_list) - 1

    def AggregateCheckPoint(self) -> float:
        """
        Получение суммарного значения времени от чекпоинта

        :return:
        """
        if self.use_timer and self.check_point >= 0:
            self.CheckTimer(False)
            time_sum: int = 0
            for i in range(self.check_point + 1, len(self.time_list)):
                time_sum += self.time_list[i]
            return round(time_sum, 2)

    def ShowGpuMemoryInfoUsingPyTorch(self):
        total_gpu_memory: int = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
        reserved_gpu_memory: int = round(torch.cuda.memory_reserved(0) / 1024 / 1024)
        allocated_gpu_memory: int = round(torch.cuda.memory_allocated(0) / 1024 / 1024)
        free_inside_reserved_gpu_memory: int = reserved_gpu_memory - allocated_gpu_memory
        self.Print('Данные о видеопамяти:')
        self.Print('Общее количество видеопамяти: %sMB' % total_gpu_memory)
        self.Print('Зарезервированное (reserved) количество видеопамяти: %sMB' % reserved_gpu_memory)
        self.Print('Выделенное (allocated) количество видеопамяти: %sMB' % allocated_gpu_memory)
        self.Print('Свободное количество видеопамяти внутри зарезервированной (free inside reserved): %sMB' % free_inside_reserved_gpu_memory)

    def SaveImageWithPointsToFile(
            self,
            _points: List[List[int]],
            _reference_image: sitk.Image,
            _file_name: str,
            _dilation_radius_in_pixels: int = 1,
            _force_use: bool = False,
            _ignore_numbers: bool = False):
        """
        Сохраняет в файл изображение, на котором изображены точки из списка _points

        :param _points: Список точек для вывода
        :param _reference_image: Эталонное изображение, из которого будут взяты только мета-данные
        :param _file_name: Имя файла
        :param _dilation_radius_in_pixels: Для улучшения видимости отображаемых точек будет применена дилатация с заданным радиусом
        :param _force_use: Принудительное сохранение файла, независимо от флага save_files
        :param _ignore_numbers: Принудительное отключение нумерации выходного файла, независимо от флага use_numbers
        """
        if self.save_files or _force_use:
            reference_image_size: List[int] = _reference_image.GetSize()
            points_array_size: np.ndarray = np.array([reference_image_size[2], reference_image_size[1], reference_image_size[0]])
            points_array: np.ndarray = np.zeros(points_array_size)
            for i, point in enumerate(_points):
                points_array[point[0], point[1], point[2]] = 1
            if _dilation_radius_in_pixels:
                lineSteps: np.ndarray
                lineLens: np.ndarray
                lineSteps, lineLens = pg.strel.flat_ball_approx(_dilation_radius_in_pixels)
                points_array = pg.flat.linear_dilate(points_array, lineSteps, lineLens)
            self.SaveNumpyArrayToFile(points_array, _file_name, _reference_image, _force_use=_force_use, _ignore_numbers=_ignore_numbers)

    def SaveGraphAndGraphNodesToFile(
            self,
            _graph: Graph,
            _reference_image: sitk.Image,
            _graph_file_name: str,
            _dilation_radius_in_pixels: int = 1,
            _force_use: bool = False,
            _ignore_numbers: bool = False
    ):
        """
        Сохраняет два файла с изображениями:
        1) изображение рёбер графа
        2) изображение узлов графа

        :param _graph: Граф
        :param _reference_image: Эталонное изображение, из которого будут взяты только мета-данные
        :param _graph_file_name: Имя файла 
        :param _dilation_radius_in_pixels: Для улучшения видимости отображаемых точек будет применена дилатация с заданным радиусом
        :param _force_use: Принудительное сохранение файла, независимо от флага save_files
        :param _ignore_numbers: Принудительное отключение нумерации выходного файла, независимо от флага use_numbers
        """
        self.SaveGraphToFile(
            _graph,
            _reference_image,
            _graph_file_name,
            _force_use = _force_use,
            _ignore_numbers=_ignore_numbers
        )
        self.SaveGraphNodesToFile(
            _graph,
            _reference_image,
            _graph_file_name + '_nodes',
            _dilation_radius_in_pixels=_dilation_radius_in_pixels,
            _force_use = _force_use,
            _ignore_numbers=_ignore_numbers
        )

    def SaveGraphNodesToFile(
            self,
            _graph: Graph,
            _reference_image: sitk.Image,
            _file_name: str,
            _dilation_radius_in_pixels: int = 1,
            _force_use: bool = False,
            _ignore_numbers: bool = False):
        """
        Сохраняет в файл изображение, на котором изображены узлы графа

        :param _graph: Граф
        :param _reference_image: Эталонное изображение, из которого будут взяты только мета-данные
        :param _file_name: Имя файла
        :param _dilation_radius: Для улучшения видимости отображаемых точек будет применена дилатация с заданным радиусом
        :param _force_use: Принудительное сохранение файла, независимо от флага save_files
        :param _ignore_numbers: Принудительное отключение нумерации выходного файла, независимо от флага use_numbers
        """
        node_coordinates: List[List[int]] = []
        for node in _graph.nodes:
            node_coordinates.append(list(np.round(_graph.nodes[node]['o']).astype(np.int32)))
        self.SaveImageWithPointsToFile(
            node_coordinates,
            _reference_image,
            _file_name,
            _dilation_radius_in_pixels=_dilation_radius_in_pixels,
            _force_use=_force_use,
            _ignore_numbers=_ignore_numbers
        )

    def SaveGraphToFile(
            self,
            _graph: Graph,
            _reference_image: sitk.Image,
            _file_name: str,
            _force_use: bool = False,
            _ignore_numbers: bool = False):
        """
        Сохраняет в файл изображение с графом

        :param _graph: Граф
        :param _reference_image: Эталонное изображение, из которого будут взяты только мета-данные
        :param _file_name: Имя файла
        :param _force_use: Принудительное сохранение файла, независимо от флага save_files
        :param _ignore_numbers: Принудительное отключение нумерации выходного файла, независимо от флага use_numbers
        """
        image_size: List[int] = _reference_image.GetSize()
        graph_array: np.ndarray = np.zeros([image_size[2], image_size[0], image_size[1]], np.int16)
        for first_node in _graph.adj:
            node_point: np.ndarray = np.round(_graph.nodes[first_node]['o']).astype(np.int32)
            graph_array[node_point[0], node_point[1], node_point[2]] = 1
            for second_node in _graph.adj[first_node]:
                edge_coordinates: np.ndarray = _graph.adj[first_node][second_node]['pts']
                previous_point: np.ndarray = edge_coordinates[0]

                for point in edge_coordinates[1:]:
                    x0: int = int(previous_point[0])
                    y0: int = int(previous_point[1])
                    z0: int = int(previous_point[2])
                    x1: int = int(point[0])
                    y1: int = int(point[1])
                    z1: int = int(point[2])

                    graph_array[x0, y0, z0] = 1
                    graph_array[x1, y1, z1] = 1
                    graph_array[x0, y1, z1] = 1
                    graph_array[x1, y0, z1] = 1
                    graph_array[x1, y0, z0] = 1
                    graph_array[x0, y1, z0] = 1
                    graph_array[x0, y0, z1] = 1

                    previous_point = point

        self.SaveNumpyArrayToFile(graph_array, _file_name, _reference_image, _force_use=_force_use, _ignore_numbers=_ignore_numbers)

    def ReadImageMetadataKey(self, _path: str, _key: str, _is_dicom: bool = False) -> str:
        if _is_dicom:
            _path = _path + '/' + os.listdir(_path)[0]

        reader = sitk.ImageFileReader()
        reader.SetFileName(_path)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        value = reader.GetMetaData(_key)
        return value

    def ReadImageMetadata(self, _path: str, _is_dicom: bool = False):
        if _is_dicom:
            _path = _path + '/' + os.listdir(_path)[0]

        reader = sitk.ImageFileReader()
        reader.SetFileName(_path)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        for k in reader.GetMetaDataKeys():
            v = reader.GetMetaData(k)
            print("({0}) = = \"{1}\"".format(k, v))
