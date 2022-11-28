from xml.dom import minidom
import SimpleITK as sitk
import numpy as np

from typing import List


def ConvertPositionToImageCoords(_point: List[float], _image: sitk.Image) -> List[int]:
    """
    Переводит точку из координат размещения на поверхности в MITK Workbench в координаты изображения.

    :param _point: Точка в position координатах.
    :param _image: Исходное изображение.
    :return: Точка в координатах изображения.
    """
    origin: np.ndarray = np.array(_image.GetOrigin())
    spacing: np.ndarray = np.array(_image.GetSpacing())
    point: np.ndarray = np.array(_point)

    coords: np.ndarray = (point - origin) / spacing
    return list(np.round(coords).astype(np.int16))


def ConvertImageCoordsToPosition(_point: List[int], _image: sitk.Image) -> List[float]:
    """
    Переводит точку из координат изображения в координаты размещения на поверхности в MITK Workbench.

    :param _point: Точка в координатах изображения.
    :param _image: Исходное изображение.
    :return: Точка в position координатах.
    """
    origin: np.ndarray = np.array(_image.GetOrigin())
    spacing: np.ndarray = np.array(_image.GetSpacing())
    point: np.ndarray = np.array(_point)

    position: np.ndarray = point * spacing + origin
    return list(position)


def ConvertSITKCoordsToNumpy(_point: List[int]) -> List[int]:
    """
    Переводит координату из системы SITK в систему координат NumPy изображения.

    :param _point: Точка в координатах SITK.
    :return: Точка в координатах NumPy.
    """
    return list(reversed(_point))


def GetPointListFromMPS(_mps_filename: str, _reference_image: sitk.Image) -> List[List[int]]:
    """
    Возвращает список из координат точек, полученных из .mps файла. Данный файл создается при помощи MITK Workbench, для
    разметки точек. Координаты возвращаются в системе NumPy координат.

    :param _mps_filename: Путь к файлу с размеченными точками в виде .mps файлов.
    :param _reference_image: Исходное изображение.
    :return: Список точек в NumPy координатах.
    """
    with open(_mps_filename, 'r') as file:
        points_document: minidom.Document = minidom.parse(file)

    points_tags = points_document.getElementsByTagName('point')
    points: List[List[float]] = []

    for tag in points_tags:
        x = float(tag.getElementsByTagName('x')[0].childNodes[0].data)
        y = float(tag.getElementsByTagName('y')[0].childNodes[0].data)
        z = float(tag.getElementsByTagName('z')[0].childNodes[0].data)
        points.append([x, y, z])

    coords: List[List[int]] = []
    for point in points:
        coord = ConvertSITKCoordsToNumpy(ConvertPositionToImageCoords(point, _reference_image))
        coords.append(coord)

    return coords


def SavePointListAsMPS(_points: List[List[int]], _filename: str,  _image: sitk.Image):
    """
    Сохраняет список точек в координатах NumPy в виде .mps файла.

    :param _points: Список точек в NumPy координатах.
    :param _filename: Имя .mps файла, который будет создан.
    :param _image: Исходное изображение.
    """

    points = [ConvertImageCoordsToPosition(point[::-1], _image) for point in _points]

    document = minidom.Document()
    point_set_file: minidom.Element = document.createElement('point_set_file')
    document.appendChild(point_set_file)

    file_version: minidom.Element = document.createElement('file_version')
    file_version_value = document.createTextNode('0.1')
    file_version.appendChild(file_version_value)
    point_set_file.appendChild(file_version)

    point_set: minidom.Element = document.createElement('point_set')
    point_set_file.appendChild(point_set)

    time_series: minidom.Element = document.createElement('time_series')
    point_set.appendChild(time_series)

    time_series_id: minidom.Element = document.createElement('time_series_id')
    time_series_id_value = document.createTextNode('0')
    time_series_id.appendChild(time_series_id_value)
    time_series.appendChild(time_series_id)

    AddPointNodes(points, time_series, document)

    with open(_filename, 'w') as file:
        document_str = document.toprettyxml()
        file.write(document_str)


def CreatePointNode(_point: List[float], _id: int, _document: minidom.Document) \
        -> minidom.Element:
    """
    Возвращает узел, содержащий информацию о точке в виде XML тега. Это необходимо для создания .mps файла.

    :param _point: Точка в position координатах MITK Workbench.
    :param _id: Порядковый номер точки.
    :param _document: XML документ, на основе которого будет создаваться узел, описывающий точку.
    """
    point_node: minidom.Element = _document.createElement('point')

    id_node: minidom.Element = _document.createElement('id')
    id_node_value = _document.createTextNode(str(_id))
    id_node.appendChild(id_node_value)
    point_node.appendChild(id_node)

    specification: minidom.Element = _document.createElement('specification')
    specification_value = _document.createTextNode('0')
    specification.appendChild(specification_value)
    point_node.appendChild(specification)

    x_node: minidom.Element = _document.createElement('x')
    x_node_value = _document.createTextNode(str(_point[0]))
    x_node.appendChild(x_node_value)
    point_node.appendChild(x_node)

    y_node: minidom.Element = _document.createElement('y')
    y_node_value = _document.createTextNode(str(_point[1]))
    y_node.appendChild(y_node_value)
    point_node.appendChild(y_node)

    z_node: minidom.Element = _document.createElement('z')
    z_node_value = _document.createTextNode(str(_point[2]))
    z_node.appendChild(z_node_value)
    point_node.appendChild(z_node)

    return point_node


def AddPointNodes(_points: List[List[float]], _parent_node: minidom.Element, _document: minidom.Document):
    """
    Добавляет в _parent_node точки. Используется для создания .mps файла.

    :param _points: Список точек в position координатах MITK Workbench.
    :param _parent_node: Родительский узел, в который будут добавлены точки.
    :param _document: Документ, на основе которого будут формироваться точки.
    """
    for idx, point in enumerate(_points):
        point_node = CreatePointNode(point, idx, _document)
        _parent_node.appendChild(point_node)


# image_path = '/home/skilpadd/Job/vessels_ds/dataset11/image11.nii.gz'
# ref_points_path = '/home/skilpadd/Job/vessels_ds/dataset11/seedpoints11.json'
# base_dir = '/home/skilpadd/Job/vessels_ds/dataset11/'
# image = sitk.ReadImage(image_path)
# keys = ['RCA', 'LCX', 'LAD', 'side_vessel']
#
# with open(ref_points_path, 'r') as file:
#     points_document = json.load(file)
#
# for key in keys:
#     for idx, points in enumerate(points_document[key]):
#         points_path = f'{base_dir}{key}{idx + 1}.mps'
#         SavePointListAsMPS(points, points_path, image)
