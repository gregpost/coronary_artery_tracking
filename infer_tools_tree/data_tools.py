# Данный модуль содержит методы обработки данных: загрузка, конвертация, сохранение данных серий и сегментаций.

import SimpleITK as sitk
import os
import glob
import sys
import math
import json
import traceback
import re
import tempfile
import shutil
import numpy as np
import itk
from typing import List, Tuple, Any, Dict, Union
from numpy import uint32
import cv2
from datetime import datetime, timedelta
from random import random
from time import time

none_image: sitk.Image = sitk.Image([0, 0, 0], sitk.sitkInt8)


def IsNoneImage(_image: sitk.Image) -> bool:
    if _image.GetWidth() == 0:
        return True
    else:
        return False


# Вспомогательный объект, содержащий дополнительные данные серии
class image_data():
    image: sitk.Image = none_image      # Изображение данной серии
    filename: str = ''                  # Название файла данной серии
    height: float = 0                   # Высота данной серии, полученная как произведение толщины среза и количества срезов
    group_number: int = -1              # Номер группы, в которую помещена данная серия по признаку тождественности высоты


def GetImageIOFromExtension(_extension: str):
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
    elif (_extension == 'nii') or (_extension == 'nii.gz') or (_extension == 'nia') or (_extension == 'hdr') or (_extension == 'img'):
        _reader_type = 'NiftiImageIO'
    elif (_extension == 'nrrd') or (_extension == 'nhdr'):
        _reader_type = 'NrrdImageIO'
    elif (_extension == 'jpeg') or (_extension == 'jpg'):
        _reader_type = 'JPEGImageIO'
    elif _extension == 'dcm':
        _reader_type = 'DICOM'
    else:
        raise Exception('Error with identify image IO. Wrong image type: %s' % _extension)
    return _reader_type


def ReadDicomFile(_dicom_path: str) -> sitk.Image:
    """
    Чтение исходного DICOM-файла

    :param _dicom_path: путь к DICOM-файлу
    :return: хаунсфилдовое изображение
    """
    reader = sitk.ImageSeriesReader()
    dicom_names: List[str] = reader.GetGDCMSeriesFileNames(_dicom_path)
    reader.SetFileNames(dicom_names)
    hounsfield_image: sitk.Image = reader.Execute()
    return hounsfield_image


def ReadImagesFromMultiphaseStudy(_path: str) -> dict:
    """
    Считывает функциональное исследование (например сердца), содержащее несколько фаз в одной папке.

    :param _path: Путь к сканам
    :return: Возвращает список изображений с разными фазами функционального исследования
    """
    all_files: List[str] = glob.glob(f'{_path}/*')
    phase_filenames: Dict[str, List[str]] = {}
    file_reader: sitk.ImageFileReader = sitk.ImageFileReader()
    key_name: str = "0018|0022"

    for file in all_files:
        file_reader.SetFileName(file)
        file_reader.ReadImageInformation()

        phase: str = file_reader.GetMetaData(key_name)
        phase_key = re.search('TP(\d*)PC', phase)
        if phase_key is not None and len(phase_key.groups()) > 0:
            phase = phase_key.groups()[0]
        else:
            phase = '00'

        if phase_filenames.get(phase) is None:
            phase_filenames[phase] = [file]
        else:
            phase_filenames[phase].append(file)

    images: Dict[str, sitk.Image] = {}
    reader: sitk.ImageSeriesReader = sitk.ImageSeriesReader()
    for phase in phase_filenames:
        with tempfile.TemporaryDirectory() as tmp_folder:
            for file in phase_filenames[phase]:
                shutil.copy(file, os.path.join(tmp_folder, os.path.basename(file)))

            filenames = reader.GetGDCMSeriesFileNames(tmp_folder)
            reader.SetFileNames(filenames)
            image: sitk.Image = reader.Execute()
            images[phase] = image

    return images


def SortImageFiles(filenames):
    file_names_and_image_position = []
    file_reader = sitk.ImageFileReader()
    for file in filenames:
        file_reader.SetFileName(file)
        file_reader.ReadImageInformation()
        file_names_and_image_position.append((file, float(file_reader.GetMetaData('0020|0032').split('\\')[2])))

    file_names_and_image_position.sort(key=lambda x: x[1])
    sorted_file_names, _ = zip(*file_names_and_image_position)
    return sorted_file_names


def ReadImageFromFile(_path: str, _image_extension: str = '', _is_binary: bool = False) -> sitk.Image:
    """
    Чтение файла с SimpleITK изображением

    :param _path: полный путь к файлу
    :param _image_extension: Расширение файла: 'nii.gz', 'nii', 'nrrd' и др.
    :param _is_binary: Если читаем бинарное изображение, то его нужно привести к UInt8 для дальнейшей обработки
    :return: Изображение, прочитанное из файла
    """
    return ReadCTImageFile(_path, _image_extension, _is_binary)


def GetExtension(_path: str) -> str:
    parsed_path: List[str] = _path.split('.')
    if len(parsed_path) > 1:
        extension = parsed_path[1]
    else:
        extension = 'dcm'
    extension = extension.lower()
    return extension


def ReadCTImageFile(_path: str, _extension: str = '', _is_binary: bool = False) -> sitk.Image:
    """
    Осуществляет чтение файла со снимком компьютерной томографии указанного типа

    :param _path: Полный путь к файлу
    :param _extension: Расширение файла
    :param _is_binary: Если читаем бинарное изображение, то его нужно привести к UInt8 для дальнейшей обработки
    :return:
    """
    try:
        # Если расширение не указано явно, то попытка определить его по имени файла
        if _extension == '':
            _extension = GetExtension(_path)

        # Основываясь на расширении переданного файла задаётся тип ридера
        reader_type: str = GetImageIOFromExtension(_extension)
        image: sitk.Image = none_image
        if reader_type == 'DICOM':
            if os.path.isdir(_path):
                image = ReadDicomFile(_path)
        else:
            if os.path.isfile(_path):
                reader = sitk.ImageFileReader()
                reader.SetImageIO(reader_type)
                reader.SetFileName(_path)
                reader.SetOutputPixelType(sitk.sitkFloat32)
                image = reader.Execute()

        if _is_binary:
            image = sitk.Cast(image, sitk.sitkUInt8)

        return image
    except Exception:
        raise Exception('ERROR (ReadCTImageFile): Image reading exception: %s.\nError code: (%s) %s\n%s' % (_path, sys.exc_info()[0], sys.exc_info()[1], traceback.format_exc()))


def SaveImageToFile(_image: sitk.Image, _path: str, _image_extension: str = ''):
    """
    Сохранение изображения в файл

    :param _image: Изображение, которое нужно сохранить в файл
    :param _image_extension: Расширение файла: 'nii.gz', 'nii', 'nrrd' и др.
    :param _path: полный путь к файлу
    """
    if _image_extension == '':
        _image_extension = GetExtension(_path)

    writer = sitk.ImageFileWriter()
    image_io: str = GetImageIOFromExtension(_image_extension)
    writer.SetImageIO(image_io)
    writer.SetFileName(_path)
    writer.Execute(_image)


def SaveImageToDicomSegmentationObject(_path_to_convertion_script: str, _path_to_meta: str, _image_list_folder: str, _source_dicom_folder: str, _output_path: str):
    """
    Сохраняет переданное изображение в формате dicom seg object

    :param _path_to_convertion_script: Путь к файлу itkimage2segimage
    :param _path_to_meta: Путь к файлу метаданных DSO
    :param _image_list_folder: Путь к каталогу с исходными изображениями сегментаций
    :param _source_dicom_folder: Путь к каталогу с исходными дайком файлами
    :param _output_path: Путь записи выходного DSO файла
    :return:
    """
    image_list_str: str = ''
    for filename in os.listdir(_image_list_folder):
        if filename.endswith(".nii.gz"):
            image_list_str += os.path.join(_image_list_folder, filename)
            image_list_str += ','
        else:
            continue

    # Попробуем прожевать выбранный файлик
    query: str = _path_to_convertion_script + ' ' \
                '--inputImageList ' + image_list_str + ' ' \
                '--inputDICOMDirectory ' + _source_dicom_folder + ' ' \
                '--outputDICOM ' + _output_path + ' ' \
                '--inputMetadata ' + _path_to_meta
    os.system(query)


# todo: ВАЖНО! Починить реализацию для кишечника, перевести её на новый формат (без json)
def SaveDataToDicomStructuredReport(_data_str: str, _path_to_convertion_script: str, _dicom_template_path: str, _source_dicom_folder: str, _output_path: str, _file_name: str = 'out', _data_is_path_to_file: bool = False):
    """
    Сохраняет данные в формате structured report

    :param _data_str: Строка с данными для записи в SR или путь к файлу (нужен флаг data_is_file)
    :param _path_to_convertion_script: Путь к исполняемому файлу генерации SR
    :param _dicom_template_path: Путь к файлу шаблона, который используется в скрипте генерации SR
    :param _source_dicom_folder: Путь к каталогу с исходными дайком файлами
    :param _output_path: Путь записи выходного SR файла
    :param _file_name: Название выходного файла
    :param _data_is_path_to_file: Флаг, указывающий на то, что в _data_str лежит путь к файлу, а не строка
    """
    # Создание json файла на основе переданных данных
    # data = {}
    # data['Measurements'] = []
    # measurementItems = []
    # measurementItems.append({'measurementPopulationDescription': _data_str})
    # data['Measurements'].append({'measurementItems': measurementItems})
    # json_temp_path = _output_path + 'data.json'
    # with open(json_temp_path, 'w') as outfile:
    #     json.dump(data, outfile)
    #     outfile.close()

    # Создание временного текстового файла на основе переданных данных
    if not _data_is_path_to_file:
        routes_temp_path = _output_path + 'routes.txt'
        routes_file = open(routes_temp_path, 'w', encoding='utf-8')
        routes_file.write(_data_str)
        routes_file.close()
    else:
        routes_temp_path = _data_str

    # Сохранение сгенерированных данных в SR
    dicom_source_path = _source_dicom_folder + '/' + os.listdir(_source_dicom_folder)[0]
    query: str = _path_to_convertion_script + ' ' + \
                 _dicom_template_path + ' ' + \
                 routes_temp_path + ' ' + \
                 dicom_source_path + ' ' + \
                 _output_path + _file_name + '.SR.dcm'
    query = query.replace('\\', '/')
    os.system(query)
    os.remove(routes_temp_path)


def ReadSeriesGroupedByZones(_path_to_study: str, _modality: str = 'ct', _min_number_of_slices: int = 0) -> List[List[image_data]]:
    """
    Данный метод принимает на вход путь к корню исследования, считывает все серии в этом исследовании
    и группирует их по однотипным зонам исследования.

    :param _path_to_study: Полный путь к папке, содержащей директории с сериями
    :param _modality: Модальность серий, которые будут включены в целевую выборку
    :param _min_number_of_slices: Минимальное количество срезов в серии для включения в целевую выборку
    :return: Список изображений в формате sitk.Image, сгруппированных по однотипным зонам
    """
    # Чтение всех серий в заданном каталоге
    image_list: List[image_data] = []
    for filename in os.listdir(_path_to_study):
        path_to_series: str = os.path.join(_path_to_study, filename) + '/dcm'
        if os.path.isdir(path_to_series):
            # Чтение метаданных данной серии
            series_data: image_data = image_data()
            series_data.filename = filename
            series_modality: str = ''
            slice_thickness: str = ''
            try:
                series_modality = ReadImageMetadataKey(path_to_series, '0008|0060', _is_dicom=True)
                slice_thickness = ReadImageMetadataKey(path_to_series, '0018|0050', _is_dicom=True)
            except:
                pass

            # Проверка, что серия имеет модальность CT
            if series_modality == 'CT':
                # Чтение изображения серии
                try:
                    series_image = ReadCTImageFile(path_to_series)
                    series_data.image = series_image
                except:
                    continue

                # Определение высоты данной серии
                number_of_slices = series_image.GetSize()[2]
                if number_of_slices >= _min_number_of_slices:
                    try:
                        series_data.height = float(slice_thickness) * int(number_of_slices)
                        image_list.append(series_data)
                    except:
                        continue
                    # image_list.append([series_image, filename, slice_thickness, number_of_slices])

    image_groups: List[List[image_data]] = []    # Список выходных изображений, разделённых на группы
    for new_series in image_list:
        for grouped_series in image_list:
            # Проверка условия, что уже есть серия в группе, для которой высота серии примерно равна текущей (различие не более 1% от средней высоты)
            if grouped_series.group_number != -1 and abs(new_series.height - grouped_series.height) < ((new_series.height + grouped_series.height) / 200):
                image_groups[grouped_series.group_number].append(new_series)
                new_series.group_number = grouped_series.group_number
                break
        # Если не было найдено ни одной походящей группы, то создаём новую
        if new_series.group_number == -1:
            image_groups.append([new_series])
            new_series.group_number = len(image_groups) - 1

    return image_groups


def ReadImageMetadataKey(_path: str, _key: str, _is_dicom: bool = False) -> str:
    """
    Возвращает из метаданных исследования значение заданного тега по ключу

    :param _path: Полный путь к файлу или каталогу целевого исследования
    :param _key: Название ключа тэга в формате xxxx|yyyy
    :param _is_dicom: Если целевое исследование в формате дайком, то должно быть True, иначе False
    :return: Значение заданного тега в формате str
    """
    if _is_dicom:
        _path = _path + '/' + os.listdir(_path)[0]

    reader = sitk.ImageFileReader()
    reader.SetFileName(_path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    value = reader.GetMetaData(_key)
    return value


def PrintImageMetadata(_path: str, _is_dicom: bool = False):
    """
    Читает все метаданные исследования по заданному пути

    :param _path: Полный путь к файлу или каталогу целевого исследования
    :param _is_dicom: Если целевое исследование в формате дайком, то должно быть True, иначе False
    :return: Список пар ключ - значение, со всеми дайком тегами данного исследования
    """
    if _is_dicom:
        _path = _path + '/' + os.listdir(_path)[0]

    reader = sitk.ImageFileReader()
    reader.SetFileName(_path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        print("({0}) = = \"{1}\"".format(k, v))


def WriteQuantitativeAnalysisFile(_path: str, _analysis_data: dict):
    """
    Запись данных количественного анализа в файл (например, степень поражения левого и правого лёгких эмфиземой и т.д.)

    :param _path: Путь записи файла количественного анализа
    :param _analysis_data: Список данных для записи, которые будут записаны в файл через разделитель
    """
    # Запись выходного файла с данными количественного анализа
    titles: str = ''
    values: str = ''
    for key in _analysis_data:
        titles += f'{key};'
        formatted_float: str = "{:.2f}".format(_analysis_data[key])
        values += f'{formatted_float};'
    quantitative_analysis_file = open(_path, 'w', encoding='utf-8')
    quantitative_analysis_file.write(f'{titles}\n{values}')
    quantitative_analysis_file.close()


def SaveListOfListOfIntToFile(_path: str, _save_data: List[List[Any]], _delimiter: str = ' ', _digits_after_dot: int = 0):
    """
    Сохранить cписок списков чисел в файл в формате:
    13б, 453, 567
    ...
    4334, 656, 765

    :param _path: Путь записи файла количественного анализа
    :param _save_data: Список списков целых чисел
    :param _delimiter: Разделитель между числами в строке
    :param _digits_after_dot: Количество знаков после запятой
    """
    output_file = open(_path, 'w', encoding='utf-8')
    for numbers in _save_data:
        for i, number in enumerate(numbers):
            number_str: str
            if _digits_after_dot > 0:
                number_str = "{:.2f}".format(number)
            else:
                number_str = str(number)
            if i != 0:
                output_file.write(_delimiter)
            output_file.write(number_str)
        output_file.write('\n')
    output_file.close()


def ConvertNumpyArrayToSimpleItkImage(_array: np.ndarray, _reference_image: sitk.Image, _pixel_id_value: int = -1) -> sitk.Image:
    """
    Конвертирует Numpy массив в изображение SimpleITK, копируя метаинформацию из этоалонного изображения

    :param _array: Массив Numpy
    :param _reference_image: Эталонное изображение, из которого будет скопирована метаинформация
    :param _pixel_id_value: Позволяет задать тип пикселя (uint8, int32 и т.д.). Если -1 значит тип пикселя определяется
    по оригинальному изображению. Примеры значений: _pixel_id_value=sitk.sitkUInt8
    :return: Изображение SimpleITK
    """
    if _array.dtype == 'uint8':
        _array = _array.astype(int)
    result_image: sitk.Image = sitk.GetImageFromArray(_array)
    result_image = CopyImageMetaInformation(result_image, _reference_image, _pixel_id_value=_pixel_id_value)
    return result_image


def CopyImageMetaInformation(_image: sitk.Image, _reference_image: sitk.Image, _pixel_id_value=-1) -> sitk.Image:
    """
    Копирует мета-информацию из одного изображения SimpleITK в другое SimpleITK изображение

    :param _image: Исходное изображение
    :param _reference_image: Оригинальное изображение, откуда надо скопировать мета-информацию
    :param _pixel_id_value: Позволяет задать тип пикселя (uint8, int32 и т.д.). Если -1 значит тип пикселя определяется
    по оригинальному изображению
    :return: Изображение с новой мета-информацией
    """
    _image.SetOrigin(_reference_image.GetOrigin())
    _image.SetSpacing(_reference_image.GetSpacing())
    _image.SetDirection(_reference_image.GetDirection())

    if _pixel_id_value > -1:
        _image = sitk.Cast(_image, _pixel_id_value)
    else:
        _image = sitk.Cast(_image, _reference_image.GetPixelIDValue())

    meta_data_keys: Tuple[str] = _image.GetMetaDataKeys()
    for key in meta_data_keys:
        meta_data_value: str = _reference_image.GetMetaData(key)
        _image.SetMetaData(key, meta_data_value)

    return _image


def CopyImageMetaInformationFromSimpleItkImageToItkImage(_itk_image: itk.Image, _reference_sitk_image: sitk.Image, _output_pixel_type) -> itk.Image:
    """
    Копирует мета-информацию из SimpleITK изображения в ITK изображение

    :param _itk_image: Исходное ITK изображение
    :param _reference_sitk_image: Оригинальное SimpleITK изображение, откуда надо скопировать мета-информацию
    :param _pixel_type: Тип пикселя в формате ITK (например: itk.F, itk.UC)
    :return: Изображение ITK с новой метаинформацией
    """
    _itk_image.SetOrigin(_reference_sitk_image.GetOrigin())
    _itk_image.SetSpacing(_reference_sitk_image.GetSpacing())

    # Установка direction (косинусов направления координатных осей исследования в пространстве)
    reference_image_direction: np.ndarray = np.eye(3)
    np_dir_vnl = itk.GetVnlMatrixFromArray(reference_image_direction)
    itk_image_direction = _itk_image.GetDirection()
    itk_image_direction.GetVnlMatrix().copy_in(np_dir_vnl.data_block())

    dimension: int = _itk_image.GetImageDimension()
    input_image_type = type(_itk_image)
    output_image_type = itk.Image[_output_pixel_type, dimension]

    castImageFilter = itk.CastImageFilter[input_image_type, output_image_type].New()
    castImageFilter.SetInput(_itk_image)
    castImageFilter.Update()
    result_itk_image: itk.Image = castImageFilter.GetOutput()

    return result_itk_image


def CopyImageMetaInformationFromItkImageToSimpleItkImage(_sitk_image: sitk.Image, _reference_itk_image: itk.Image, _pixel_id_value: int, _direction: List[float]) -> itk.Image:
    """
    Копирует мета-информацию из ITK изображения в SimpleITK изображение

    :param _sitk_image: Исходное SimpleITK изображение
    :param _reference_itk_image: Оригинальное ITK изображение, откуда надо скопировать мета-информацию
    :param _pixel_id_value: Тип пикселя в формате SimpleITK (например: sitk.sitkFloat32, sitk.sitkUInt8)
    :param _direction: Список косинусов, описывающих направление координатных осей исследования в пространстве
    :return: Изображение SimpleITK с новой метаинформацией
    """
    reference_image_origin: List[int] = list(_reference_itk_image.GetOrigin())
    _sitk_image.SetOrigin(reference_image_origin)
    reference_image_spacing: List[int] = list(_reference_itk_image.GetSpacing())
    _sitk_image.SetSpacing(reference_image_spacing)
    _sitk_image.SetDirection(_direction)
    result_sitk_image: sitk.Image = sitk.Cast(_sitk_image, _pixel_id_value)
    return result_sitk_image


def GetFirstDictValue(_dict: dict) -> any:
    """
    Получить значение первого элемента словаря
    _dict: Словарь
    """
    first_key = list(_dict.keys())[0]
    first_value = _dict[first_key]
    return first_value


def GetFirstDictKey(_dict: any) -> any:
    """
    Получить ключ первого элемента словаря
    _dict: Словарь. Возможные типы: dict, NodeView, AdjacencyView и др.
    """
    first_key = list(_dict.keys())[0]
    return first_key


def GetImageSize(_source_image: Union[np.ndarray, sitk.Image]) -> Tuple[int, int, int]:
    """
    Определить размерность изображения в формате [axial, coronal, sagittal]

    :param _image: Исходное изображение
    """
    image_shape: Tuple[int, int, int]
    is_numpy_array: bool = (str(type(_source_image)) == '<class \'numpy.ndarray\'>')
    if is_numpy_array:
        image_shape = _source_image.shape
    else:
        image_size: List[uint32] = _source_image.GetSize()
        image_shape = (
            int(image_size[2]),
            int(image_size[1]),
            int(image_size[0])
        )
    return image_shape


def __writeSlices(self, _writer, _series_tag_values, _new_img, _i, _path_to_file, _volume_id, _slice, _fixed_time, _modality):
    """
    Производит запись DICOM-срезов
    """
    image_slice = _slice
    if IsNoneImage(_slice):
        image_slice = _new_img[:, :, _i]

    # Tags shared by the series.
    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), _series_tag_values))

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    _slice_name: str = os.path.join(_path_to_file + '/', str(_volume_id) + '_' + str(_i) + '.dcm')
    while os.path.isfile(_slice_name):
        _i += 1
        _slice_name: str = os.path.join(_path_to_file + '/', str(_volume_id) + '_' + str(_i) + '.dcm')

    # (0020, 0032) image position patient determines the 3D spacing between slices.
    if IsNoneImage(_slice):
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str, _new_img.TransformIndexToPhysicalPoint((0, 0, _i)))))  # Image Position (Patient)
    else:
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str, (0, 0, _i))))

    # Slice specific tags.
    current_time: datetime = _fixed_time
    if _fixed_time is None:
        current_time = datetime.now()
    else:
        current_time += timedelta(seconds=_i)

    image_slice.SetMetaData("0008|0012", current_time.strftime("%Y%m%d"))  # Instance Creation Date
    image_slice.SetMetaData("0008|0013", current_time.strftime("%H%M%S"))  # Instance Creation Time

    # Setting the type to CT preserves the slice location.
    image_slice.SetMetaData("0008|0060", _modality)  # set the type to CT so the thickness is carried over
    image_slice.SetMetaData("0020|0013", str(_i))  # Instance Number

    _writer.SetFileName(_slice_name)
    _writer.Execute(image_slice)


def SaveImageToDicomSeries(self, _image: sitk.Image,
                           _file_name: str,
                           _path_is_local: bool = True,
                           _patient_id: str = '',
                           _uid_prefix: str = "777777",
                           _study_uid: str = '',
                           _volume_id: int = 1,
                           _fixed_time: datetime = None,
                           _modality: str = "CT",
                           _phase_name: str = '',
                           _gate: str = '',
                           _gate_count: str = ''
                           ):
    """
    Конвертирует изображение в том дайком серий и сохраняет его в файл

    :param _image: Изображение для конвертации и сохранения в файл
    :param _patient_id: ID пациента
    :param _file_name: Если _path_is_local = True, то нужно указать имя и расширение файла, иначе полный путь к файлу
    :param _path_is_local: Является ли указанный путь локальным или глобальным
    :param _phase_name: Если задано, то данное поле определяет название текущей фазы в многофазном исследовании
    :param _gate: Номер фазы биения сердца для многофазного исследования
    :param _gate_count: Общее количество фаз биения сердца для многофазного исследования
    :return:
    """
    if self.save_files:
        if _image is not None:
            self.CheckTimer(False)
            writer = sitk.ImageFileWriter()
            writer.SetImageIO("JPEGImageIO")
            if _path_is_local:
                _file_name = self.output_data_folder + self.current_folder + _file_name

            self.Print('Сохранение в файл в формате DICOM: %s' % _file_name)

            if not os.path.isdir(_file_name):
                os.mkdir(_file_name)

            if _image.GetPixelIDValue() == sitk.sitkFloat32:
                _image = sitk.Cast(_image, sitk.sitkInt32)

            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()

            current_time: datetime = _fixed_time
            if _fixed_time is None:
                current_time = datetime.now()

            modification_time = current_time.strftime("%H%M%S")
            modification_date = current_time.strftime("%Y%m%d")

            series_uid: str = _uid_prefix + "." + str(_volume_id) + "." + modification_date + "." + modification_time

            if _study_uid == '':
                _study_uid = series_uid

            direction = _image.GetDirection()
            dir_param = ''
            for dir_value in direction:
                dir_param = dir_param + str(dir_value) + '\\'
            dir_param = dir_param.rstrip('\\')

            phase_time: str = datetime.now().strftime("%H%M%S")

            if _patient_id == '':
                _patient_id = int(random() * 1000000000)

            series_tag_values = [("0008|0031", modification_time),  # Series Time
                                 ("0008|0030", modification_time),  # Study Time
                                 ("0008|0021", modification_date),  # Series Date
                                 ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
                                 ("0020|000e", series_uid),  # Series Instance UID
                                 ("0020|000d", _study_uid),
                                 ("0020|0037", dir_param),
                                 ("0008|103e", "Created-SimpleITK"),  # Series Description
                                 ("0018|0022", f'TP{_phase_name}PC\\GATE_0{_gate}_OF_0{_gate_count}'),  # Информация о фазе в многофазном исследовании
                                 ("0008|0032", phase_time),  # Момент времени, в который было произведено сканирование (нужно для многофазных исследований)
                                 ("0008|002A", modification_date + phase_time),  # Момент времени, в который было произведено сканирование (нужно для многофазных исследований)
                                 ("0008|0033", phase_time),  # Момент времени, в который было произведено сканирование (нужно для многофазных исследований)
                                 ("0010|0020", _patient_id)]  # ID пациента

            # Write slices to output directory
            if _image.GetDepth() > 1:
                for i in range(_image.GetDepth()):
                    self.__writeSlices(writer, series_tag_values, _image, i, _file_name, _volume_id, none_image, _fixed_time, _modality)
            else:
                self.__writeSlices(writer, series_tag_values, _image, 0, _file_name, _volume_id, _image, _fixed_time, _modality)


def SaveImageToMediaFile(self, _image: sitk.Image, _file_name: str):
    """
    Конвертирует изображение и сохраняет его в видеофайл в выбранном расширении

    :param _image: Исходное изображение
    :param _file_name: Полный путь к файлу результирующего видео
    :return:
    """
    # Сначала нужно сохранить изображение во временную папку в формате jpg
    self.SaveImageToPictureFile(_image, 'img', _short_flag=True)

    # Теперь считаем выходные файлы и конвертируем их в видео с помощью opencv
    img_array = []
    size = ()
    path = self.output_data_folder + 'img/'
    for frame_num in range(len(os.listdir(path))):
        filename = path + str(frame_num) + '.jpg'
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(_file_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def ConvertImageToDICOM(self, _path_to_file: str, _output_path: str = '', _volume_id=1, _study_uid: str = ''):
    """
    Преобразует файл произвольного графического или мультимедийного типа в DICOM и сохраняет в выбранном месте

    :return: Ничего не возвращает. Перегоняет картинку или видео в дайком формат.
    """
    file_extension: str = ''
    try:
        file_extension = _path_to_file.split('.')[1]
        if _output_path == '':
            _output_path = _path_to_file.split('.')[0]
    except IndexError:
        self.Print('%s: ERROR (ConvertImageToDICOM): Невозможно определить тип файла. Не задано расширение: %s' % (self.study_name, _path_to_file), _hard_priority=True)

    if (file_extension == 'mp4') or (file_extension == 'avi'):
        vidcap = cv2.VideoCapture(_path_to_file)
        success, image = vidcap.read()
        count: int = 0
        modification_time = datetime.now()
        while success:
            path_to_frame: str = self.output_data_folder + 'frame_' + str(count) + '.png'
            cv2.imwrite(path_to_frame, image)  # save frame as png file
            success, image = vidcap.read()
            count += 1
            ct_image: sitk.Image = self.ReadCTImageFile(path_to_frame, 'png')
            self.SaveImageToDicomSeries(ct_image, _output_path, _path_is_local=False, _volume_id=_volume_id, _study_uid=_study_uid, _fixed_time=modification_time)
    else:
        ct_image: sitk.Image = self.ReadCTImageFile(_path_to_file, file_extension)
        self.SaveImageToDicomSeries(ct_image, _output_path, _path_is_local=False, _volume_id=_volume_id, _study_uid=_study_uid)


def GetStringWithZerosFromValue(_value: int, _max_number_count: int) -> str:
    """
    Добавляет нули в строку с числом в зависимости от максимально возможного количества чисел в строке
    Например: 91 -> '091' если максмальное количество цифр указано 3

    :param _value: Число
    :param _max_number_count: Максимальное количество цифр в числе
    """
    value_str: str = str(int(round(_value)))
    current_number_count: int = len(value_str)
    zero_count: int = _max_number_count - current_number_count
    value_with_zeros_str: str = ''
    for i in range(zero_count):
        value_with_zeros_str += '0'
    value_with_zeros_str += value_str
    return value_with_zeros_str


def GetBatchSize(_batch_size_for_two_gb_of_free_gpu_memory: int) -> int:
    """
    Получить размер пакета для нейросети на основании количества Гб свободной видеопамяти,
    а также исходя из подобранного опытным путём batch_size для двух Гб свободной памяти GPU

    _batch_size_for_two_gb_of_free_gpu_memory: Значение batch_size для случая с 2 Гб свободной
    памяти GPU
    """
    if 'torch' not in globals():
        import torch

    # Автоматический подбор размера пакета в зависимотсити от количества имеющейся видеопамяти
    total_gpu_memory: int = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024)
    allocated_gpu_memory: int = round(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    reserved_gpu_memory: int = round(torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
    free_gpu_memory: int = total_gpu_memory - reserved_gpu_memory - allocated_gpu_memory
    batch_size: int
    if free_gpu_memory > 0:
        batch_size = round(_batch_size_for_two_gb_of_free_gpu_memory * pow(3.5, math.log2(free_gpu_memory) - 1))
    else:
        batch_size = 1
    return batch_size


def GetBatchSizeFromCache(_algorithm_name: str = '', _cache_filename: str = 'batch_size.json') -> int:
    """
    Возвращает размер пакета из конфигурационного файла.

    :param _algorithm_name: Имя алгоритма. Например, 'lung', 'heart'.
    :param _cache_filename: имя конфигурационного файла.
    :return: Размер пакета, если произошла ошибка, то возвращается -1.
    """
    if 'torch' not in globals():
        import torch

    cuda_device: int = torch.cuda.current_device()
    device_name: str = torch.cuda.get_device_name(cuda_device)

    root_folder: str = os.path.dirname(os.path.abspath(__file__)).rpartition(os.sep)[0]
    filename_abs_path: str = os.path.join(root_folder, _cache_filename)

    try:
        with open(filename_abs_path, 'r') as file:
            records: List[Dict] = json.load(file)
            for record in records:
                if device_name == record['device_name'] and _algorithm_name == record['algorithm_name']:
                    return record['batch_size']
    except FileNotFoundError:
        pass

    return -1


def SaveBatchSizeToCache(_batch_size: int, _algorithm_name: str, _cache_filename: str = 'batch_size.json'):
    """
    Сохраняет размер пакета в конфигурационный файл.

    :param _batch_size: Размер пакета.
    :param _algorithm_name: Имя алгоритма. Например, 'lung', 'heart'.
    :param _cache_filename: Имя конфигурационного файла.
    """
    if 'torch' not in globals():
        import torch

    cuda_device: int = torch.cuda.current_device()
    device_name: str = torch.cuda.get_device_name(cuda_device)

    root_folder: str = os.path.dirname(os.path.abspath(__file__)).rpartition(os.sep)[0]
    filename_abs_path: str = os.path.join(root_folder, _cache_filename)

    data = {
        'device_name': device_name,
        'batch_size': _batch_size,
        'algorithm_name': _algorithm_name
    }

    with open(filename_abs_path, 'w+') as file:
        try:
            records: List[Dict] = json.load(file)
        except json.JSONDecodeError:
            records = []
        is_found: bool = False

        for idx, record in enumerate(records):
            if record['device_name'] == device_name and record['algorithm_name'] == _algorithm_name:
                is_found = True
                records[idx] = data

        if not is_found:
            records.append(data)

        json.dump(records, file)
