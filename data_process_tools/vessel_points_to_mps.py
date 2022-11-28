from xml.dom import minidom
import numpy as np
import SimpleITK as sitk
from mps_reader import AddPointNodes
from typing import List


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

    position: np.ndarray = point + origin
    return list(position)


def SavePointListAsMPS(_points: List[List[int]], _filename: str,  _image: sitk.Image):
    """
    Сохраняет список точек в координатах NumPy в виде .mps файла.

    :param _points: Список точек в NumPy координатах.
    :param _filename: Имя .mps файла, который будет создан.
    :param _image: Исходное изображение.
    """

    points = [ConvertImageCoordsToPosition(point, _image) for point in _points]

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


image_path = '/home/skilpadd/Job/vessels_ds/dataset02/image02.nii.gz'
image = sitk.ReadImage(image_path)

reference_points_path = 'train_data/dataset02/vessel0/reference.txt'
ref_points = np.loadtxt(reference_points_path)
points = ref_points[:, :3]

SavePointListAsMPS(points, 'reference.mps', image)
