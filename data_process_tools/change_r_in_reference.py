import os
from glob import glob
import numpy as np
from typing import List, Any


def SaveListOfPointsToFile(
        _path: str,
        _save_data: List[List[Any]],
        _delimiter: str = ' ',
        _digit_number_after_dot: int = 2,
        _line_wrap: bool = True
):
    """
    Сохранить список координат в файл в формате:
    13б, 453, 567
    ...
    4334, 656, 765

    :param _path: Путь записи файла количественного анализа
    :param _save_data: Список списков целых чисел
    :param _delimiter: Разделитель между числами в строке
    :param _digit_number_after_dot: Количество знаков после запятой
    :param _line_wrap: Флаг записи каждой точки в отдельную строку
    """
    output_file = open(_path, 'w', encoding='utf-8')
    for numbers in _save_data:
        for i, number in enumerate(numbers):
            number_str: str
            if _digit_number_after_dot > 0:
                format_str: str = "{:." + str(_digit_number_after_dot) + "f}"
                number_str = format_str.format(number)
            else:
                number_str = str(number)
            if i != 0:
                output_file.write(_delimiter)
            output_file.write(number_str)
        if _line_wrap:
            output_file.write('\n')
        else:
            output_file.write(_delimiter)
    output_file.close()


ds_dir = './train_data/'
dataset_dirs = glob(os.path.join(ds_dir, 'dataset*'))
new_r = 3.0

for dataset_dir in dataset_dirs:
    vessel_dirs = glob(os.path.join(dataset_dir, 'vessel*'))
    for vessel_dir in vessel_dirs:
        reference_path = os.path.join(vessel_dir, 'reference.txt')
        txt_data = np.loadtxt(reference_path, dtype=np.float32)
        txt_data[..., 3] = new_r
        SaveListOfPointsToFile(reference_path, txt_data)
