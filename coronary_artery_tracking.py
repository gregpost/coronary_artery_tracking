import os
import copy
import torch
import sknw
import SimpleITK as sitk
import onnxruntime as rt
import numpy as np
from skimage.morphology import skeletonize, ball
from scipy.ndimage import center_of_mass
import lib.greg.seg_tools as seg
import lib.data_tools as data
from debug_manager import DebugManager
from .infer_tools_tree.utils import resample, prob_terminates, data_preprocess, get_shell, get_spacing_res2, get_angle, crop_heart

from typing import List, Dict, Tuple


"""Нижний порог яркости контрастированных коронарных артерий"""
VESSEL_LOWER_THRESHOLD: int = 100

"""Минимальный размер связной компоненты в пикселях"""
MIN_SMALL_OBJECT_SIZE: int = 100

class TreeNode(object):
    def __init__(self, value, start_point_index):
        self.value = value
        self.start_point_index = start_point_index
        self.child_list = []

    def add_child(self, node):
        self.child_list.append(node)


class CoronaryArteryTracking:
    def __init__(self, _neural_network_weight_folder: str, _debug: DebugManager = DebugManager()):
        """
        Инициализация класса

        :param _neural_network_weight_folder: Абсолютный путь к папке с весами нейросетей
        :param _debug: Объект для вывода отладочной информации
        """
        self.debug = _debug

        self.centerline_radii_model_path = os.path.join(_neural_network_weight_folder,
                                                        'centerline_radii_net_model.onnx')
        self.centerline_directions_model_path = os.path.join(_neural_network_weight_folder,
                                                             'centerline_directions_net_model.onnx')
        self.ostia_model_path = os.path.join(_neural_network_weight_folder, 'ostiapoints_net_model.onnx')
        self.seeds_model_path = os.path.join(_neural_network_weight_folder, 'seedspoints_net_model.onnx')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.execution_providers: List[str] = self.GetExecutionProviders()
        self.prob_thr = 0.75
        self.max_points = 500

        self.radii_infer_model = None
        self.directions_infer_model = None
        self.seeds_model = None
        self.ostia_model = None

        self.re_spacing_img = None
        self.curr_spacing = None
        self.resize_factor = None
        self.spacing = None

        self.points_for_binary_mask: List[List[float]] = []
        self.radii_for_binary_mask: List[float] = []
        self.ostia_points: List[float] = []
        self.original_image = None

        self.deb_points = []

    @staticmethod
    def GetExecutionProviders() -> List[str]:
        """
        Возвращает тип устройства для запуска на нем модели в onnxruntime.
        :return: ExecutionProvider для onnxruntime.
        """
        cuda_provider: str = 'CUDAExecutionProvider'
        cpu_provider: str = 'CPUExecutionProvider'
        providers: List[str] = rt.get_available_providers()

        if cuda_provider in providers:
            return [cuda_provider]
        else:
            return [cpu_provider]

    def CreateVesselsTreeByPoints(self, _ostias: List[List[int]], _seed_points: List[List[int]], _image: sitk.Image):
        """
        Создает дерево коронарных артерий по остьевым и посевным точкам.
    
        :param _ostias: Остьевые точки в координатах SITK.
        :param _seed_points: Посевные точки в координатах SITK.
        :param _image: Исходное SITK изображение.
        :return: Список из коронарных артерий, в координатах NumPy.
        """
        image_array: np.ndarray = sitk.GetArrayFromImage(_image)
        self.original_image = _image
        self.spacing = _image.GetSpacing()
        spacing_x, spacing_y, spacing_z = self.spacing
        self.re_spacing_img, self.curr_spacing, self.resize_factor = resample(image_array, np.array([spacing_z, spacing_x, spacing_y]),
                                                                              np.array([0.5, 0.5, 0.5]))
    
        seeds = np.array(_seed_points)
        root = TreeNode(_ostias, start_point_index=None)
        self.BuildVesselTree(seeds, root=root)
        single_tree = self.DfsSearchTree(root)
        vessel_tree_postprocess = []
        for vessel_list in single_tree:
            vessel_list.pop(0)
            res = np.array([]).reshape(0, 3)
            while vessel_list:
                first_node = vessel_list[0]
                first_res = first_node.value
                vessel_list.pop(0)
                if vessel_list:
                    second_node = vessel_list[0]
                    first_res = first_res[:second_node.start_point_index]
                    res = np.vstack((res, first_res))
                else:
                    res = np.vstack((res, first_res))
                    vessel_tree_postprocess.append(res)
    
        vessels: List[List[List[int]]] = []
        for vessel in vessel_tree_postprocess:
            res_vessel: List[List[int]] = np.flip((vessel / self.spacing).round().astype(np.int16), axis=1).tolist()
            vessels.append(res_vessel)

        # Убираем повторяющиеся точки, они образуются из-за конвертации float в int, так как трекер работает в
        # реальных координатах
        unique_vessels: List[List[List[int]]] = []
        for vessel in vessels:
            vessel_array: np.ndarray = np.array(vessel)
            unique_vessel, indices = np.unique(vessel_array, return_index=True, axis=0)
            unique_vessels.append(vessel_array[np.sort(indices)].tolist())
    
        return unique_vessels

    def CreateVesselsTreeByHeartChambers(self, _heart_chambers: Dict[str, sitk.Image], _image: sitk.Image):
        """
        Создает дерево коронарных артерий на основе бинарных масок сердца и изображения.
    
        :param _heart_chambers: Бинарные маски сердца.
        :param _image: Исходное монохромное изображение.
        :return: Список из коронарных артерий, в координатах NumPy.
        """

        # Если маска в словаре обрезана, приведем ее к исходному размеру
        for part in _heart_chambers:
            if _heart_chambers[part].GetSize() != _image.GetSize():
                _heart_chambers[part] = seg.RestoreImageByRegionOfInterestImage(_heart_chambers[part], _image)

        res_seeds, res_ostia = self.SearchSeedsOstiasUsingHeartChambers(_image, _heart_chambers)
    
        seed_points: List[List[int]] = [point for point, proximity in res_seeds]
        ostia_points: List[List[int]] = [point for point, proximity in res_ostia]
    
        ostias = []
        ostias_thr = 10
        node_first = np.array(ostia_points[0])
        ostias.append(list(node_first))
        for point in ostia_points:
            point_array = np.array(point)
            if np.linalg.norm(point_array - node_first) > ostias_thr:
                ostias.append(list(point_array))
                break

        self.ostia_points = ostias

        if len(ostias) > 1:
            self.debug.Print(f'Найдены устьевые точки: {ostias[0]} и {ostias[1]}')
            vessel_tree = self.CreateVesselsTreeByPoints(ostias, seed_points, _image)
            return vessel_tree
        else:
            print("Not find 2 ostia points")
            return []

    def CreateVesselsTree(self, _image: sitk.Image):
        """
        Создает дерево коронарных артерий на основе изображения.
    
        :param _image: Исходное SITK изображение.
        :return: Список из коронарных артерий.
        """
        self.seeds_model = rt.InferenceSession(self.seeds_model_path, providers=self.execution_providers)
        self.ostia_model = rt.InferenceSession(self.ostia_model_path, providers=self.execution_providers)
    
        image_array: np.ndarray = sitk.GetArrayFromImage(_image)
        self.spacing = _image.GetSpacing()
    
        seed_points, ostia_points = self.SearchSeedsOstias(image_array)
        seed_points = [list(point) for point in seed_points]
    
        ostias = [list(point) for point in ostia_points]

        if len(ostias) > 1:
            vessel_tree = self.CreateVesselsTreeByPoints(ostias, seed_points, _image)
            return vessel_tree
        else:
            self.debug.Print("not find 2 ostia points")
            return []

    def GetOstiaPoints(self) -> List[List[int]]:
        """
        Возвращает список из найденных устьевых точек после трекинга.

        :return: Устьевые точки в формате NumPy.
        """
        # Если не были найдены устьевые точки, тогда возвращаем пустой список
        if len(self.ostia_points) > 0:
            ostia_points: np.ndarray = np.array(self.ostia_points)
            points: List[List[int]] = np.flip((ostia_points / self.spacing).round().astype(np.int16), axis=1).tolist()
            return points
        else:
            return []

    def CreateVesselsBinaryMask(self) -> sitk.Image:
        """
        Возвращает Бинарную маску сосудов, полученную после трекинга.

        :return: Бинарная маска сосудов, построенная на основе трекинга артерий.
        """
        image_array: np.ndarray = sitk.GetArrayFromImage(self.original_image)
        vessels_mask: np.ndarray = np.zeros_like(image_array)
        mask_shape: np.ndarray = np.array(vessels_mask.shape)

        # Переводим точки из реальных координат в координаты Numpy
        points: np.ndarray = np.array(self.points_for_binary_mask)
        points = np.flip((points / self.spacing).astype(np.int16), axis=1)

        # Добавляем смещение по координатам из-за перевода float в int. Добавление ко всем координатам единицы
        # работает качественнее, чем округление
        points = points + 1

        # Если после смещение где-то вышли за границы, то убираем смещение
        if np.any(points >= mask_shape):
            np.place(points, points >= mask_shape, points - 1)

        radii: np.ndarray = (np.array(self.radii_for_binary_mask) / self.spacing[0]).round().astype(np.int16)

        for point, r in zip(points, radii):
            vessels_mask = self.PlaceStructure(vessels_mask, ball(r), point)

        # Расширим маску сосудов по маске с порогом, чтобы маска полностью заполняла сосуды
        threshold_array: np.ndarray = image_array > VESSEL_LOWER_THRESHOLD
        vessels_mask = seg.DilateNumpyArrayByMask(vessels_mask, threshold_array, self.original_image.GetSpacing(), 1, 1)

        # Уберем мелкие связные компоненты, которые могут появиться из-за дилатации
        vessels_mask = seg.RemoveSmallObjects(vessels_mask, MIN_SMALL_OBJECT_SIZE)

        vessels_mask_image: sitk.Image = data.ConvertNumpyArrayToSimpleItkImage(vessels_mask, self.original_image)
        return vessels_mask_image

    @staticmethod
    def PlaceStructure(_image_array: np.ndarray, _structure: np.ndarray, position: List[int]) -> np.ndarray:
        """
        Вырезает кубик по центру координат из _image_array, и на его место вставляет массив _structure.

        :param _image_array: Изображение для замены.
        :param _structure: Структура, которая будет вставляться.
        :param position: Координаты центра, по которым будет вставлена структура в массив.
        :return: Новый массив со вставленной структурой в int8.
        """
        mask_array: np.ndarray = np.zeros_like(_image_array, dtype=np.int8)
        x, y, z = position

        str_shape_x, str_shape_y, str_shape_z = _structure.shape

        half_x_1 = int(np.ceil(str_shape_x / 2))
        half_x_2 = int(np.floor(str_shape_x / 2))

        half_y_1 = int(np.ceil(str_shape_y / 2))
        half_y_2 = int(np.floor(str_shape_y / 2))

        half_z_1 = int(np.ceil(str_shape_z / 2))
        half_z_2 = int(np.floor(str_shape_z / 2))

        mask_array[x - half_x_1:x + half_x_2, y - half_y_1:y + half_y_2, z - half_z_1:z + half_z_2] = _structure
        new_image: np.ndarray = (_image_array | mask_array).astype(np.int8)

        return new_image

    def SearchSeedsOstiasUsingHeartChambers(self, _image: sitk.Image, _heart_chambers: Dict[str, sitk.Image]):
        """
        Возвращает список посевных точек и список устьевых точек, в формате кортежа, где первый элемент -- точка в формате
        SITK, а второй элемент -- значение близости для этой точки.
    
        :param _image: Исходное изображение.
        :param _heart_chambers: Словарь из камер сердца для этого изображения.
        :return: 1) Список посевных точек. 2) Список устьевых точек.
        """
        self.seeds_model = rt.InferenceSession(self.seeds_model_path, providers=self.execution_providers)
        self.ostia_model = rt.InferenceSession(self.ostia_model_path, providers=self.execution_providers)
    
        cut_size = 9
        res_seeds = {}
        res_ostia = {}
        new_patch_list = []
        center_coord_list = []
    
        # Приводим изображение к единичному спейсингу
        image_array = sitk.GetArrayFromImage(_image)
        self.spacing = _image.GetSpacing()
        spacing_x, spacing_y, spacing_z = self.spacing
        self.re_spacing_img, self.curr_spacing, self.resize_factor = resample(image_array, np.array([spacing_z, spacing_x, spacing_y]),
                                                               np.array([1, 1, 1]))
        z, h, w = self.re_spacing_img.shape
        re_spacing_image = sitk.GetImageFromArray(self.re_spacing_img)
    
        # Приводим все камеры сердца к единичному спейсингу
        heart_chambers = _heart_chambers.copy()
        for chamber in heart_chambers:
            chamber_array = sitk.GetArrayFromImage(heart_chambers[chamber])
            chamber_spacing: Tuple[float, float, float] = heart_chambers[chamber].GetSpacing()
            chamber_spacing_x, chamber_spacing_y, chamber_spacing_z = chamber_spacing
            re_spacing_chamber_array, self.curr_spacing, self.resize_factor = resample(
                chamber_array,
                np.array([chamber_spacing_z, chamber_spacing_x, chamber_spacing_y]),
                np.array([1, 1, 1]))
            re_spacing_chamber: sitk.Image = data.ConvertNumpyArrayToSimpleItkImage(re_spacing_chamber_array,
                                                                                    re_spacing_image)
            heart_chambers[chamber] = re_spacing_chamber

        # Объединим маски артерий
        lca_array: np.ndarray = sitk.GetArrayFromImage(heart_chambers['LCA'])
        rca_array: np.ndarray = sitk.GetArrayFromImage(heart_chambers['RCA'])
        vessels_mask_array: np.ndarray = (lca_array | rca_array).astype(np.int8)

        # Получим узлы из скелета артерий для трекинга по ним
        vessels_skel: np.ndarray = skeletonize(vessels_mask_array)
        vessels_graph = sknw.build_sknw(vessels_skel)
        node_points = [vessels_graph.nodes[node]['o'].astype(np.int16) for node in vessels_graph.nodes]

        self.debug.SaveImageWithPointsToFile(node_points, re_spacing_image, 'node_points')
        self.debug.SaveImageToFile(re_spacing_image, 're_spacing_image')

        # Дилатируем аорту по маске и обрежем, чтобы точно найти устьевые точки, если их не нашла нейронная сеть
        vessels_mask: np.ndarray = (self.re_spacing_img > 250).astype(np.int8)
        aorta_mask: np.ndarray = sitk.GetArrayFromImage(heart_chambers['ASA'])
        dilated_aorta: np.ndarray = seg.DilateNumpyArrayByMask(aorta_mask, vessels_mask, re_spacing_image.GetSpacing(), 15, 1, _kernel_type=seg.KernelType.Cross3D)
        self.debug.SaveNumpyArrayToFile(dilated_aorta, 'dilated_aorta', re_spacing_image)

        # Находим центроид аорты и по нему получаем маску аорты
        aorta_centroid: List[int] = list(np.array(center_of_mass(aorta_mask)).round().astype(np.int16))
        aorta_connected_component: np.ndarray = seg.GetNearestToGivenPointImageConnectedComponent(dilated_aorta, aorta_centroid)
        self.debug.SaveNumpyArrayToFile(aorta_connected_component, 'aorta_connected_component', re_spacing_image)

        # Обрезаем дилатированную маску по маске аорты
        cut_aorta_mask: np.ndarray = np.logical_and(aorta_connected_component, np.logical_not(aorta_mask)).astype(np.int16)
        self.debug.SaveNumpyArrayToFile(cut_aorta_mask, 'cut_aorta_mask', re_spacing_image)

        # Применяем закрытие, чтобы срезать лишние с краев маски, чтобы точки не касались напрямую аорту
        closed_aorta: np.ndarray = seg.CloseNumpyArray(cut_aorta_mask, 8, re_spacing_image.GetSpacing())
        self.debug.SaveNumpyArrayToFile(closed_aorta, 'closed_aorta', re_spacing_image)

        # Скелетонизируем получившуюся маску и получаем из нее точки
        aorta_skeleton: np.ndarray = skeletonize(closed_aorta)
        aorta_graph = sknw.build_sknw(aorta_skeleton)
        aorta_points = [aorta_graph.nodes[node]['o'].round().astype(np.int16) for node in aorta_graph.nodes]
        aorta_points = [point[::-1] for point in aorta_points]
        self.debug.SaveImageWithPointsToFile(aorta_points, re_spacing_image, 'aorta_points')

        node_points = [point[::-1] for point in node_points]
        ostia_centroids = node_points.copy()
        ostia_centroids.extend(aorta_points)

        for center_x_pixel, center_y_pixel, center_z_pixel in node_points:
            left_x = center_x_pixel - cut_size
            right_x = center_x_pixel + cut_size
            left_y = center_y_pixel - cut_size
            right_y = center_y_pixel + cut_size
            left_z = center_z_pixel - cut_size
            right_z = center_z_pixel + cut_size
            if left_x >= 0 and right_x < h and left_y >= 0 and right_y < w and left_z >= 0 and right_z < z:
                new_patch = np.zeros((cut_size * 2 + 1, cut_size * 2 + 1, cut_size * 2 + 1))
                for ind in range(left_z, right_z + 1):
                    src_temp = self.re_spacing_img[ind].copy()
                    new_patch[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]
                input_data = data_preprocess(new_patch)
                new_patch_list.append(input_data)
                center_coord_list.append((center_x_pixel, center_y_pixel, center_z_pixel))
    
        input_data = torch.cat(new_patch_list, axis=0)
        inputs = {self.seeds_model.get_inputs()[0].name: input_data.float().numpy()}
        seeds_outputs = torch.from_numpy(self.seeds_model.run(None, inputs)[0])
        seeds_outputs = seeds_outputs.view((len(input_data)))
        seeds_proximity = seeds_outputs.cpu().detach().numpy()
        for point, seed_proximity in zip(center_coord_list, seeds_proximity):
            res_seeds[point] = seed_proximity
        new_patch_list.clear()
        center_coord_list.clear()
        del input_data
        del inputs
        del seeds_outputs
    
        for center_x_pixel, center_y_pixel, center_z_pixel in ostia_centroids:
            left_x = center_x_pixel - cut_size
            right_x = center_x_pixel + cut_size
            left_y = center_y_pixel - cut_size
            right_y = center_y_pixel + cut_size
            left_z = center_z_pixel - cut_size
            right_z = center_z_pixel + cut_size
            if left_x >= 0 and right_x < h and left_y >= 0 and right_y < w and left_z >= 0 and right_z < z:
                new_patch = np.zeros((cut_size * 2 + 1, cut_size * 2 + 1, cut_size * 2 + 1))
                for ind in range(left_z, right_z + 1):
                    src_temp = self.re_spacing_img[ind].copy()
                    new_patch[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]
                input_data = data_preprocess(new_patch)
                new_patch_list.append(input_data)
                center_coord_list.append((center_x_pixel, center_y_pixel, center_z_pixel))
    
        input_data = torch.cat(new_patch_list, axis=0)
        inputs = {self.ostia_model.get_inputs()[0].name: input_data.float().numpy()}
        ostia_outputs = torch.from_numpy(self.ostia_model.run(None, inputs)[0])
        ostia_outputs = ostia_outputs.view(len(input_data))
        ostia_proximity = ostia_outputs.cpu().detach().numpy()
        for point, proximity in zip(center_coord_list, ostia_proximity):
            res_ostia[point] = proximity
        new_patch_list.clear()
        center_coord_list.clear()
        del input_data
        del inputs
        del ostia_outputs
    
        positive_seeds = []
        for x, y, z in res_seeds:
            if res_seeds[(x, y, z)] > 0:
                positive_seeds.append([[x, y, z], res_seeds[(x, y, z)]])
    
        res_ostia = sorted(res_ostia.items(), key=lambda item: item[1], reverse=True)
        ostia_points = []
        for point, proximity in res_ostia:
            x, y, z = point
            ostia_points.append([[x, y, z], proximity])
    
        bitmap = np.zeros_like(self.re_spacing_img)
        for point, proximity in ostia_points:
            x, y, z = point
            bitmap[z, y, x] = int(proximity)
        self.debug.SaveNumpyArrayToFile(bitmap, 'ostia_bitmap', re_spacing_image)
    
        bitmap = np.zeros_like(self.re_spacing_img)
        for x, y, z in res_seeds:
            bitmap[z, y, x] = int(res_seeds[(x, y, z)])
        self.debug.SaveNumpyArrayToFile(bitmap, 'seeds_bitmap', re_spacing_image)
    
        bitmap = np.zeros_like(self.re_spacing_img)
        for point, proximity in positive_seeds:
            x, y, z = point
            bitmap[z, y, x] = int(proximity)
        self.debug.SaveNumpyArrayToFile(bitmap, 'positive_seeds', re_spacing_image)

        return positive_seeds, ostia_points

    def GetHeartVesselsUsingHeartChambers(self, _heart_phase_image: sitk.Image, _heart_chambers: dict) -> sitk.Image:
        """
        Получить списки точек центральных линий сосудов сердца
        1) ударный объём левого желудочка сердца, мл
        2) фракция выброса левого желудочка сердца, %
    
        :param _heart_phase_image: Исходное изображение из DICOM-файла с одной фазой сердца
        :param _heart_chambers: Бинарные маски камер сердца
        :return: Списки точек центральных линий сосудов:
            'LAD': ПМЖВ – передняя межжелудочковая ветвь. Другие названия: передняя нисходящая артерия, левая передняя нисходящая артерия, левая передняя межжелудочковая артерия, left anterior descending artery (LAD), anterior interventricular artery (AIA), anterior descending coronary artery
            'LCX': ОВ – огибающая ветвь левой коронарной артерии. Другие названия: огибающая артерия, left circumflex coronary artery (LCX), circumflex artery (CX, CA)
            'RCA': ПКА - правая коронарная артерия. Другие названия: ПВА - правая венечная артерия, right coronary artery (RCA), right main coronary artery
            'side_vessel': боковая ветвь одного из трёх сосудов: ПМЖВ, ОВ или ПКА
        """
        self.debug.SelectDirectory('heart_vessels')
    
        # Сегментированое изображение разбивается на самостоятельные изображения
        img_arr = sitk.GetArrayFromImage(_heart_phase_image)
        heart_205 = sitk.GetArrayFromImage(_heart_chambers['MLV'])
        heart_500 = sitk.GetArrayFromImage(_heart_chambers['LVBC'])
        heart_420 = sitk.GetArrayFromImage(_heart_chambers['LABC'])
        heart_600 = sitk.GetArrayFromImage(_heart_chambers['RVBC'])
        heart_550 = sitk.GetArrayFromImage(_heart_chambers['RABC'])
        heart_820 = sitk.GetArrayFromImage(_heart_chambers['ASA'])
        heart_850 = sitk.GetArrayFromImage(_heart_chambers['PUA'])
    
        # Расширяется ЛЖ (HU схоже с сосудами, чтобы края не захватывались)
        # и сужается миокард (по краям миокарда проходят сосуды)
        radius: float = 2.5
        image_spacing: List[float] = _heart_phase_image.GetSpacing()
        dilated_heart_500: np.ndarray = seg.DilateNumpyArray(heart_500, radius, image_spacing)
        eroded_heart_205: np.ndarray = seg.ErodeNumpyArray(heart_205, radius, image_spacing)
        eroded_heart_600: np.ndarray = seg.ErodeNumpyArray(heart_600, radius, image_spacing)
        self.debug.CheckTimer()
        self.debug.SaveNumpyArrayToFile(dilated_heart_500, 'dilated_heart_500', _heart_phase_image,
                                        _pixel_id_value=sitk.sitkUInt8)
        self.debug.SaveNumpyArrayToFile(eroded_heart_205, 'eroded_heart_205', _heart_phase_image,
                                        _pixel_id_value=sitk.sitkUInt8)
        self.debug.SaveNumpyArrayToFile(eroded_heart_600, 'eroded_heart_600', _heart_phase_image,
                                        _pixel_id_value=sitk.sitkUInt8)
    
        # Берется изображение без сегментированных областей кроме 820 (аорта)
        without_heart = np.where(np.logical_and(eroded_heart_600 != 1, (np.logical_and(dilated_heart_500 != 1, (
            np.logical_and(heart_420 != 1,
                           (np.logical_and(heart_550 != 1, (np.logical_and(eroded_heart_205 != 1, heart_850 != 1))))))))),
                                 img_arr, 0)
        self.debug.CheckTimer()
        self.debug.SaveNumpyArrayToFile(without_heart, 'where_without_heart', _heart_phase_image,
                                        _pixel_id_value=sitk.sitkInt16)
    
        # Бинаризация по области в кторой находятся сосуды
        without_heart_threshold_array = seg.Threshold(without_heart, 150, 600)
        without_heart_threshold_mask = data.ConvertNumpyArrayToSimpleItkImage(without_heart_threshold_array, _heart_phase_image)
        self.debug.CheckTimer()
        self.debug.SaveImageToFile(without_heart_threshold_mask, 'threshold_without_heart')
    
        # Выделение трубчатых объектов (отделяются кости)
        self.debug.Print('Применение фильтра трубкообразных объектов')
        tubular_obj = self.debug.ReadImageFromFile('tubular_objects')
        if seg.IsNoneImage(tubular_obj):
            tubular_obj = seg.GetTubularOrPlaneOrEllipsoidalObjects(without_heart_threshold_mask, seg.ObjectShapeType.Tubular,
                                                                    _sigma_minimum=1.0, _sigma_maximum=2.0,
                                                                    _number_of_sigma_steps=2)
            self.debug.CheckTimer()
            self.debug.SaveImageToFile(tubular_obj, 'tubular_objects')
        else:
            self.debug.Print('Получение результата фильтрации трубкообразных объектов из файла')
    
        tubular_obj = sitk.Cast(tubular_obj, sitk.sitkInt16)
        tubular_obj = seg.Threshold(tubular_obj, 30, 255)
        self.debug.CheckTimer()
        self.debug.SaveImageToFile(tubular_obj, 'rso')
    
        # Выделение области легких
        self.debug.Print('Поиск лёгких')
        lung = np.where(np.logical_and(img_arr > -9999, img_arr < -200), 1, 0)
        lung = seg.CloseNumpyArray(lung, 5.0, image_spacing)
        self.debug.CheckTimer()
        self.debug.SaveNumpyArrayToFile(lung, 'lung', _heart_phase_image, _pixel_id_value=sitk.sitkUInt8)
    
        # Вычитание сосудов легких из общей картины
        self.debug.Print('Вычитание лёгких')
        tubular_obj = sitk.GetArrayFromImage(tubular_obj)
        res = np.where(np.logical_and(tubular_obj == 1, lung == 0), 1, 0)
        res = data.ConvertNumpyArrayToSimpleItkImage(res, _heart_phase_image, _pixel_id_value=sitk.sitkUInt8)
        self.debug.CheckTimer()
        self.debug.SaveImageToFile(res, 'remove_lung')
        res = sitk.GetArrayFromImage(res)
    
        # Добавление аорты (820)
        self.debug.Print('Добавление аорты')
        radius: float = 2.5
        heart_820 = seg.DilateNumpyArray(heart_820, radius, image_spacing)
        res = seg.DilateNumpyArray(res, radius, image_spacing)
        self.debug.CheckTimer()
        self.debug.SaveNumpyArrayToFile(res, 'plus_820_and_dilate', _heart_phase_image, _pixel_id_value=sitk.sitkUInt8)
    
        # Заливка из посевной точки внутри аорты (выделятся нужные сосуды)
        self.debug.Print('Заливка из посевной точки внутри аорты (выделятся нужные сосуды)')
        res = np.where(np.logical_or(res == 1, heart_820 == 1), 1, 0)
        res = data.ConvertNumpyArrayToSimpleItkImage(res, _heart_phase_image, _pixel_id_value=sitk.sitkUInt8)

        # Получение маски всех сосудов серца путём наращивания из аорты
        heart_vessel_mask = seg.ConfidenceConnectedThreshold(res, _number_of_iterations=1)
        heart_vessel_mask = sitk.GetArrayFromImage(heart_vessel_mask)
        heart_vessel_mask = np.where(np.logical_and(heart_vessel_mask == 1, heart_820 == 0), 1, 0)
        heart_vessel_mask = seg.ErodeNumpyArray(heart_vessel_mask, 2.5, image_spacing)
        heart_vessel_mask = data.ConvertNumpyArrayToSimpleItkImage(heart_vessel_mask, _heart_phase_image,
                                                                   _pixel_id_value=sitk.sitkUInt8)
        self.debug.CheckTimer()
        self.debug.SaveImageToFile(heart_vessel_mask, 'heart_vessels_mask', _ignore_numbers=True)
    
        self.debug.UnselectDirectory()
    
        return heart_vessel_mask

    def SearchSeedsOstias(self, src_array: np.ndarray, max_size=(200, 10)):
        '''
        find seeds points arr and ostia points arr
        :param max_size: The first max_size[0] seed points and the first max_size[1] ostia points were selected
        :return:
        '''
        spacing_x = self.spacing[0]
        spacing_y = self.spacing[1]
        spacing_z = self.spacing[2]
    
        self.re_spacing_img, self.curr_spacing, self.resize_factor = resample(src_array, np.array([spacing_z, spacing_x, spacing_y]),
                                                               np.array([1, 1, 1]))
        self.re_spacing_img, meam_minc, mean_minr, mean_maxc, mean_maxr = crop_heart(self.re_spacing_img)
        cut_size = 9
        res_seeds = {}
        res_ostia = {}
        count = 0
        random_point_size = 80000
        batch_size = 1000
        new_patch_list = []
        center_coord_list = []
        z, h, w = self.re_spacing_img.shape
        offset_size = 10
        x_list = np.random.random_integers(meam_minc - offset_size, mean_maxc + offset_size, (random_point_size, 1))
        y_list = np.random.random_integers(mean_minr - offset_size, mean_maxr + offset_size, (random_point_size, 1))
        z_list = np.random.random_integers(0, z, (random_point_size, 1))
    
        index = np.concatenate([x_list, y_list, z_list], axis=1)
    
        index = list(set(tuple(x) for x in index))
        for i in index:
            center_x_pixel = i[0]
            center_y_pixel = i[1]
            center_z_pixel = i[2]
            left_x = center_x_pixel - cut_size
            right_x = center_x_pixel + cut_size
            left_y = center_y_pixel - cut_size
            right_y = center_y_pixel + cut_size
            left_z = center_z_pixel - cut_size
            right_z = center_z_pixel + cut_size
            if left_x >= 0 and right_x < h and left_y >= 0 and right_y < w and left_z >= 0 and right_z < z:
                new_patch = np.zeros((cut_size * 2 + 1, cut_size * 2 + 1, cut_size * 2 + 1))
                for ind in range(left_z, right_z + 1):
                    src_temp = self.re_spacing_img[ind].copy()
                    new_patch[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]
                count += 1
                input_data = data_preprocess(new_patch)
                new_patch_list.append(input_data)
                center_coord_list.append((center_x_pixel, center_y_pixel, center_z_pixel))
                if count % batch_size == 0:
                    input_data = torch.cat(new_patch_list, axis=0)
                    inputs = input_data.to(self.device)
                    seeds_outputs = self.seeds_model(inputs.float())
                    seeds_outputs = seeds_outputs.view((len(input_data)))  # view
                    seeds_proximity = seeds_outputs.cpu().detach().numpy()
                    ostia_outputs = self.ostia_model(inputs.float())
                    ostia_outputs = ostia_outputs.view(len(input_data))
                    ostia_proximity = ostia_outputs.cpu().detach().numpy()
                    for i in range(batch_size):
                        res_seeds[center_coord_list[i]] = seeds_proximity[i]
                        res_ostia[center_coord_list[i]] = ostia_proximity[i]
                    new_patch_list.clear()
                    center_coord_list.clear()
                    del input_data
                    del inputs
                    del seeds_outputs
                    del ostia_outputs
    
        positive_count = 0
        for i in res_seeds.values():
            if i > 0:
                positive_count += 1
        res_seeds = sorted(res_seeds.items(), key=lambda item: item[1], reverse=True)
        res_ostia = sorted(res_ostia.items(), key=lambda item: item[1], reverse=True)
        res_seeds = res_seeds[:max_size[0]]
        res_ostia = res_ostia[:max_size[1]]
        return res_seeds, res_ostia

    def Infer(self, start: list):
        """
        :param start: Initial point
        :return: Moving position, the index of maximum confidence direction, Current termination probability
        """
        self.radii_infer_model = rt.InferenceSession(self.centerline_radii_model_path,
                                                     providers=self.execution_providers)
        self.directions_infer_model = rt.InferenceSession(self.centerline_directions_model_path,
                                                          providers=self.execution_providers)
    
        max_z = self.re_spacing_img.shape[0]
        max_x = self.re_spacing_img.shape[1]
        max_y = self.re_spacing_img.shape[2]
    
        cut_size = 9
        spacing_x = self.spacing[0]
        spacing_y = self.spacing[1]
        spacing_z = self.spacing[2]
    
        center_x_pixel = get_spacing_res2(start[0], spacing_x, self.resize_factor[1])
        center_y_pixel = get_spacing_res2(start[1], spacing_y, self.resize_factor[2])
        center_z_pixel = get_spacing_res2(start[2], spacing_z, self.resize_factor[0])
    
        left_x = center_x_pixel - cut_size
        right_x = center_x_pixel + cut_size
        left_y = center_y_pixel - cut_size
        right_y = center_y_pixel + cut_size
        left_z = center_z_pixel - cut_size
        right_z = center_z_pixel + cut_size
    
        new_patch = np.zeros((cut_size * 2 + 1, cut_size * 2 + 1, cut_size * 2 + 1))
    
        if not (
                left_x < 0 or right_x < 0 or left_y < 0 or right_y < 0 or left_z < 0 or right_z < 0 or left_x >= max_x or right_x >= max_x or left_y >= max_y or right_y >= max_y or left_z >= max_z or right_z >= max_z):
            for ind in range(left_z, right_z + 1):
                src_temp = self.re_spacing_img[ind].copy()
                new_patch[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]

            # Подготавливаем данные для загрузки в нейросети
            input_data = data_preprocess(new_patch)

            # Запускаем инференс на модели, которая предназначена для определения направления движения
            direction_inputs = {self.directions_infer_model.get_inputs()[0].name: input_data.float().numpy()}
            direction_outputs = torch.from_numpy(self.directions_infer_model.run(None, direction_inputs)[0])
            directions = direction_outputs.view((len(input_data), self.max_points + 1))
            directions = directions[:, :len(directions[0]) - 1]
            directions = torch.nn.functional.softmax(directions, 1)

            # Запускаем инференс на модели, которая предназначена для определения радииуса в точке
            radii_inputs = {self.radii_infer_model.get_inputs()[0].name: input_data.float().numpy()}
            radii_outputs = torch.from_numpy(self.radii_infer_model.run(None, radii_inputs)[0])
            radii = radii_outputs.view((len(input_data), self.max_points + 1))
            radii = radii[:, -1]

            indexes = np.argsort(directions.cpu().detach().numpy()[0])[::-1]
            curr_prob = prob_terminates(directions, self.max_points).cpu().detach().numpy()[0]

            curr_r = radii.cpu().detach().numpy()[0]
            sx, sy, sz = get_shell(self.max_points, curr_r)
            return [sx, sy, sz], indexes, curr_r, curr_prob
        else:
            return None

    def SearchFirstNode(self, start: list, prob_records: list):
        """
        :param start: Initial point
        :return: Next direction vector, Probability record, Current radius
        """
        try:
            s_all, indexs, curr_r, curr_prob = self.Infer(start=start)
        except TypeError:
            return None
        start_x, start_y, start_z = start
        prob_records.pop(0)
        prob_records.append(curr_prob)
        sx, sy, sz = s_all
        forward_x = sx[indexs[0]] + start_x
        forward_y = sy[indexs[0]] + start_y
        forward_z = sz[indexs[0]] + start_z
        forward_move_direction_x = sx[indexs[0]]
        forward_move_direction_y = sy[indexs[0]]
        forward_move_direction_z = sz[indexs[0]]
        for i in range(1, len(indexs)):
            curr_angle = get_angle(np.array([sx[indexs[i]], sy[indexs[i]], sz[indexs[i]]]),
                                   np.array([forward_move_direction_x, forward_move_direction_y, forward_move_direction_z]))
            # To determine two initial opposing directions of the tracker, two local maxima d0 and d0′ separated by an angle ≥ 90°
            if curr_angle >= 90:
                backward_move_direction_x = copy.deepcopy(sx[indexs[i]])
                backward_move_direction_y = copy.deepcopy(sy[indexs[i]])
                backward_move_direction_z = copy.deepcopy(sz[indexs[i]])
                break
        backward_x = backward_move_direction_x + start_x
        backward_y = backward_move_direction_y + start_y
        backward_z = backward_move_direction_z + start_z
        direction = {}
        direction["forward"] = [forward_x, forward_y, forward_z]
        direction["forward_vector"] = [forward_move_direction_x, forward_move_direction_y, forward_move_direction_z]
        direction["backward"] = [backward_x, backward_y, backward_z]
        direction["backward_vector"] = [backward_move_direction_x, backward_move_direction_y, backward_move_direction_z]
        return direction, prob_records, curr_r

    def SearchOneDirection(self, start: list, move_direction: list, prob_records: list, point_list: list, r_list: list,
                           root: TreeNode, find_node=None):
        """
        :param start: start point
        :param move_direction: last move direction
        :param prob_records: record of termination probability
        :param point_list:
        :param r_list: radius arr
        :return:
        """
        find_node_initial = None
        prob_mean = sum(prob_records) / len(prob_records)
        while prob_mean <= self.prob_thr and find_node_initial is None:
            result = self.Infer(start=start)
            if result is not None:
                shell_arr, indexs, curr_r, curr_prob = result
                r_list.append(curr_r)
                point_list.append(start)
                prob_records.pop(0)
                prob_records.append(curr_prob)
                prob_mean = sum(prob_records) / len(prob_records)
                move_direction, start = self.Move(start=start, shell_arr=shell_arr, indexs=indexs,
                                                  move_direction=move_direction)
                self.deb_points.append(start)
    
                if find_node is None:
                    find_node_initial = self.SearchTree(root, start)
            else:
                break
        return find_node_initial

    def SearchLine(self, start, curr_r, direction, prob_records, root: TreeNode):
        '''
        Search from the initial point to the direction of d0 and d0',
        :param start:
        :param curr_r:
        :param direction:
        :param prob_records:
        :param root:
        :return:
        '''
        point_list = []
        r_list = []
        point_list.append(start)
        r_list.append(curr_r)
        point_forward_list = copy.deepcopy(point_list)
        r_forward_list = copy.deepcopy(r_list)
        prob_forward_records = copy.deepcopy(prob_records)
        point_backward_list = copy.deepcopy(point_list)
        r_backward_list = copy.deepcopy(r_list)
        prob_backward_records = copy.deepcopy(prob_records)
        find_node_forward = self.SearchOneDirection(start=direction["forward"],
                                                 move_direction=direction["forward_vector"],
                                                 prob_records=prob_forward_records,
                                                 r_list=r_forward_list, point_list=point_forward_list, root=root)
        find_node_backward = self.SearchOneDirection(start=direction["backward"],
                                                  move_direction=direction["backward_vector"],
                                                  prob_records=prob_backward_records, r_list=r_backward_list,
                                                  point_list=point_backward_list, find_node=find_node_forward, root=root)
        find = True
    
        # If the current point is within 200 points from the end of the centerline,
        # it will be spliced with the current centerline, otherwise it will be set as a new branch
        add_thr = 200
        if find_node_forward is not None:
            point_forward_list.reverse()
            r_forward_list.reverse()
            point_list = point_forward_list + point_backward_list
            r_list = r_forward_list + r_backward_list
            res_arr = self.Interpolation(point_list, r_list)
            start_point_index = find_node_forward[1]
            start_node = find_node_forward[0]
            start_coord = start_node.value[start_point_index]
            tmp_arr = np.linspace(start_coord, res_arr[0], num=100)
            res_arr = np.vstack((tmp_arr, res_arr))

            # Добавление точек и радиусов для создания бинарной маски
            self.points_for_binary_mask.extend(point_list)
            self.radii_for_binary_mask.extend(r_list)

            if start_node != root and start_point_index > start_node.value.shape[0] - add_thr:
                start_node.value = np.vstack((start_node.value[:start_point_index], res_arr))
            else:
                start_node.add_child(TreeNode(res_arr, start_point_index=start_point_index))
        elif find_node_backward is not None:
            point_backward_list.reverse()
            r_backward_list.reverse()
            point_list = point_backward_list + point_forward_list
    
            r_list = r_backward_list + r_forward_list
            res_arr = self.Interpolation(point_list, r_list)
            start_point_index = find_node_backward[1]
            start_node = find_node_backward[0]
            start_coord = start_node.value[start_point_index]
            tmp_arr = np.linspace(start_coord, res_arr[0], num=100)
            res_arr = np.vstack((tmp_arr, res_arr))

            # Добавление точек и радиусов для создания бинарной маски
            self.points_for_binary_mask.extend(point_list)
            self.radii_for_binary_mask.extend(r_list)

            if start_node != root and start_point_index > start_node.value.shape[0] - add_thr:
                start_node.value = np.vstack((start_node.value[:start_point_index], res_arr))
            else:
                start_node.add_child(TreeNode(res_arr, start_point_index=start_point_index))
        else:
            # This vessel is added to the final record only when the ostia point is found
            find = False
        return find

    def BuildVesselTree(self, seeds: np.ndarray, root: TreeNode):
        '''
        :param seeds:seeds arr
        :param root: tree root
        :return:
        '''
        prob_records = [0] * 3
        seeds_unused = []
        for seed in seeds:
            if self.SearchTree(root, seed) is None:
                try:
                    direction, prob_records, curr_r = self.SearchFirstNode(start=seed, prob_records=prob_records)
                except TypeError:
                    continue
                find = self.SearchLine(start=seed, curr_r=curr_r, prob_records=prob_records, direction=direction, root=root)
                if not find:
                    seeds_unused.append(seed)
        for seed in seeds_unused:
            if self.SearchTree(root, seed) is None:
                try:
                    direction, prob_records, curr_r = self.SearchFirstNode(start=seed, prob_records=prob_records)
                except TypeError:
                    continue
                self.SearchLine(start=seed, curr_r=curr_r, prob_records=prob_records, direction=direction, root=root)

    def Interpolation(self, point_list: list, r_list: list):
        # Interpolate according to 0.03 mm
        p1 = point_list[0]
        p2 = point_list[1]
        res_arr = np.linspace(p1, p2, num=int(r_list[0] / 0.03))
        for i in range(1, len(point_list) - 1):
            p1 = point_list[i]
            p2 = point_list[i + 1]
            tmp_arr = np.linspace(p1, p2, num=int(r_list[i] / 0.03))
            res_arr = np.vstack((res_arr, tmp_arr))
        return res_arr

    def SearchTree(self, root: TreeNode, point):
        '''
        BFS tree, determine whether the current input point is close to the existing centerline
        :param root:
        :param point:
        :return:
        '''
        queue = []
        queue.append(root)
        while queue:
            vertex = queue.pop(0)
            point = np.array(point)
            dis_all = np.linalg.norm(point - np.array(vertex.value), axis=1)
            dis = dis_all.min()
            index = dis_all.argmin()
            if dis < 2:
                return vertex, index
            nodes = vertex.child_list
            for w in nodes:
                queue.append(w)
        return None

    def DfsSearchTree(self, root: TreeNode):
        '''
        DFS, build single vessel
        :param root:
        :return: single vessel
        '''
        stack_list = []
        visited = []
        stack_list.append(root)
        visited.append(root)
        res = [root]
        single_vessel = []
        while len(stack_list) > 0:
            temp = []
            x = stack_list[-1]
            for w in x.child_list:
                if w not in visited:
                    temp.append(w)
                    visited.append(w)
                    stack_list.append(w)
                    break
            if len(temp) > 0:
                res.append(temp[0])
            if stack_list[-1] == x:
                single_vessel.append(res[:])
                res.pop()
                stack_list.pop()
        return single_vessel

    def Move(self, start: list, shell_arr: list, indexs: list, move_direction: list):
        """
        Moving ball
        :param start: start point
        :param shell_arr: shell arr
        :param indexs: index of next direction
        :param move_direction: last move direction
        :param curr_r: radius
        :return: direction vector, move to next point
        """
        start_x, start_y, start_z = start
        sx, sy, sz = shell_arr
        move_direction_x, move_direction_y, move_direction_z = move_direction
        for i in range(len(indexs)):
            curr_angle = get_angle(np.array([sx[indexs[i]], sy[indexs[i]], sz[indexs[i]]]),
                                   np.array([move_direction_x, move_direction_y, move_direction_z]))
            # Only directions with an angle ≤ 60°to the previously followed direction are considered.
            if curr_angle <= 60:
                new_x = sx[indexs[i]] + start_x
                new_y = sy[indexs[i]] + start_y
                new_z = sz[indexs[i]] + start_z
                move_direction_x = sx[indexs[i]]
                move_direction_y = sy[indexs[i]]
                move_direction_z = sz[indexs[i]]
                break

        return [move_direction_x, move_direction_y, move_direction_z], [new_x, new_y, new_z]
