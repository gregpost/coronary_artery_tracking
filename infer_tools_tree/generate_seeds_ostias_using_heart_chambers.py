import numpy as np
import SimpleITK as sitk
from setting import seeds_model, ostia_model, device, setting_info
from utils import data_preprocess, resample, crop_heart
from mps_reader import SavePointListAsMPS, GetPointListFromMPS
from scipy.ndimage import center_of_mass
import sknw
import torch
import skimage.measure as skime
from skimage.morphology import skeletonize
import data_tools as data
import seg_tools as seg
import tubular_analysis as tub
from debug_manager import DebugManager
from typing import List, Dict, Tuple

debug = DebugManager('./')
debug.save_files = False


def SearchSeedsOstiasUsingHeartChambers(_image: sitk.Image, _heart_chambers: Dict[str, sitk.Image]):
    """
    Возвращает список посевных точек и список устьевых точек, в формате кортежа, где первый элемент -- точка в формате
    SITK, а второй элемент -- значение близости для этой точки.

    :param _image: Исходное изображение.
    :param _heart_chambers: Словарь из камер сердца для этого изображения.
    :return: 1) Список посевных точек. 2) Список устьевых точек.
    """
    cut_size = 9
    res_seeds = {}
    res_ostia = {}
    new_patch_list = []
    center_coord_list = []

    # Приводим изображение к единичному спейсингу
    image_array = sitk.GetArrayFromImage(_image)
    spacing = image.GetSpacing()
    spacing_x, spacing_y, spacing_z = spacing
    re_spacing_img, curr_spacing, resize_factor = resample(image_array, np.array([spacing_z, spacing_x, spacing_y]),
                                                           np.array([1, 1, 1]))
    z, h, w = re_spacing_img.shape
    re_spacing_image = sitk.GetImageFromArray(re_spacing_img)

    # Приводим все камеры сердца к единичному спейсингу
    heart_chambers = _heart_chambers.copy()
    for chamber in heart_chambers:
        chamber_array = sitk.GetArrayFromImage(heart_chambers[chamber])
        chamber_spacing: Tuple[float, float, float] = heart_chambers[chamber].GetSpacing()
        chamber_spacing_x, chamber_spacing_y, chamber_spacing_z = chamber_spacing
        re_spacing_chamber_array, curr_spacing, resize_factor = resample(
            chamber_array,
            np.array([chamber_spacing_z, chamber_spacing_x, chamber_spacing_y]),
            np.array([1, 1, 1]))
        re_spacing_chamber: sitk.Image = data.ConvertNumpyArrayToSimpleItkImage(re_spacing_chamber_array,
                                                                                re_spacing_image)
        heart_chambers[chamber] = re_spacing_chamber

    heart_array, meam_minc, mean_minr, mean_maxc, mean_maxr = crop_heart(re_spacing_img)
    heart_image = data.ConvertNumpyArrayToSimpleItkImage(heart_array, re_spacing_image)
    heart_image = sitk.Resample(heart_image, re_spacing_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, re_spacing_image.GetPixelID())

    vessels_mask = GetHeartVesselsUsingHeartChambers(heart_image, heart_chambers)
    vessels_mask_array = sitk.GetArrayFromImage(vessels_mask)

    dilated_aorta_mask = seg.DilateNumpyArray(asa_array, 10, curr_spacing)
    aorta_vessels_mask = (vessels_mask_array | dilated_aorta_mask).astype(np.int16)

    properties: List = skime.regionprops(asa_array)
    aorta_point: List[int] = list(properties[0]['coords'][0])

    aorta_vessels = seg.GetConnectedComponentByPoint(aorta_vessels_mask, aorta_point)
    vessels = (aorta_vessels & vessels_mask_array).astype(np.int16)

    vessels_skel = skeletonize(vessels)
    vessels_graph = sknw.build_sknw(vessels_skel)
    vessels_graph = tub.RemoveGraphNodesWithTwoNeigbours(vessels_graph)
    node_points = [vessels_graph.nodes[node]['o'].astype(np.int16) for node in vessels_graph.nodes]

    # Найдем устьевые точки
    vessels_theshold = heart_array > 150
    dilated_aorta_mask = seg.DilateNumpyArray(asa_array, 5, curr_spacing)
    dilated_aorta_by_mask = seg.DilateNumpyArrayByMask(asa_array, vessels_theshold, curr_spacing, 20, 1,  _kernel_type=seg.KernelType.Cross3D)
    ostia_connected_components = (dilated_aorta_by_mask & ~mlv_array & ~labc_array & ~lvbc_array & ~rabc_array & ~rvbc_array & ~dilated_aorta_mask & ~pua_array)
    ostia_label_array = skime.label(ostia_connected_components)
    properties = skime.regionprops(ostia_label_array)
    ostia_centroids = []
    for property in properties:
        centroid = list(np.array(property['centroid']).astype(np.int16))
        ostia_centroids.append(centroid)

    debug.SaveImageWithPointsToFile(ostia_centroids, re_spacing_image, 'ostia_centroids')
    debug.SaveNumpyArrayToFile(ostia_connected_components, 'ostia_connected_components', re_spacing_image)
    debug.SaveNumpyArrayToFile(vessels_mask_array, 'vessels_mask_array', re_spacing_image)
    debug.SaveNumpyArrayToFile(dilated_aorta_mask, 'dilated_aorta_mask', re_spacing_image)
    debug.SaveNumpyArrayToFile(aorta_vessels, 'aorta_vessels', re_spacing_image)
    debug.SaveGraphToFile(vessels_graph, re_spacing_image, 'vessels_graph')
    debug.SaveImageWithPointsToFile(node_points, re_spacing_image, 'node_points')
    debug.SaveImageToFile(re_spacing_image, 're_spacing_image')

    node_points = [point[::-1] for point in node_points]
    ostia_centroids = [point[::-1] for point in ostia_centroids]

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
                src_temp = re_spacing_img[ind].copy()
                new_patch[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]
            input_data = data_preprocess(new_patch)
            new_patch_list.append(input_data)
            center_coord_list.append((center_x_pixel, center_y_pixel, center_z_pixel))

    input_data = torch.cat(new_patch_list, axis=0)
    inputs = input_data.to(device)
    seeds_outputs = seeds_model(inputs.float())
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
                src_temp = re_spacing_img[ind].copy()
                new_patch[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]
            input_data = data_preprocess(new_patch)
            new_patch_list.append(input_data)
            center_coord_list.append((center_x_pixel, center_y_pixel, center_z_pixel))

    input_data = torch.cat(new_patch_list, axis=0)
    inputs = input_data.to(device)
    ostia_outputs = ostia_model(inputs.float())
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

    bitmap = np.zeros_like(re_spacing_img)
    for point, proximity in ostia_points:
        x, y, z = point
        bitmap[z, y, x] = int(proximity)
    debug.SaveNumpyArrayToFile(bitmap, 'ostia_bitmap', re_spacing_image)

    bitmap = np.zeros_like(re_spacing_img)
    for x, y, z in res_seeds:
        bitmap[z, y, x] = int(res_seeds[(x, y, z)])
    debug.SaveNumpyArrayToFile(bitmap, 'seeds_bitmap', re_spacing_image)

    bitmap = np.zeros_like(re_spacing_img)
    for point, proximity in positive_seeds:
        x, y, z = point
        bitmap[z, y, x] = int(proximity)
    debug.SaveNumpyArrayToFile(bitmap, 'positive_seeds', re_spacing_image)

    print('ostia', ostia_points)
    print('positive_seeds', positive_seeds)

    return positive_seeds, ostia_points


def GetHeartVesselsUsingHeartChambers(_heart_phase_image: sitk.Image, _heart_chambers: dict) -> sitk.Image:
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
    debug.SelectDirectory('heart_vessels')

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
    debug.CheckTimer()
    debug.SaveNumpyArrayToFile(dilated_heart_500, 'dilated_heart_500', _heart_phase_image,
                                    _pixel_id_value=sitk.sitkUInt8)
    debug.SaveNumpyArrayToFile(eroded_heart_205, 'eroded_heart_205', _heart_phase_image,
                                    _pixel_id_value=sitk.sitkUInt8)
    debug.SaveNumpyArrayToFile(eroded_heart_600, 'eroded_heart_600', _heart_phase_image,
                                    _pixel_id_value=sitk.sitkUInt8)

    # Берется изображение без сегментированных областей кроме 820 (аорта)
    without_heart = np.where(np.logical_and(eroded_heart_600 != 1, (np.logical_and(dilated_heart_500 != 1, (
        np.logical_and(heart_420 != 1,
                       (np.logical_and(heart_550 != 1, (np.logical_and(eroded_heart_205 != 1, heart_850 != 1))))))))),
                             img_arr, 0)
    debug.CheckTimer()
    debug.SaveNumpyArrayToFile(without_heart, 'where_without_heart', _heart_phase_image,
                                    _pixel_id_value=sitk.sitkInt16)

    # Бинаризация по области в кторой находятся сосуды
    without_heart_threshold_array = seg.Threshold(without_heart, 150, 600)
    without_heart_threshold_mask = data.ConvertNumpyArrayToSimpleItkImage(without_heart_threshold_array, _heart_phase_image)
    debug.CheckTimer()
    debug.SaveImageToFile(without_heart_threshold_mask, 'threshold_without_heart')

    # Выделение трубчатых объектов (отделяются кости)
    debug.Print('Применение фильтра трубкообразных объектов')
    tubular_obj = debug.ReadImageFromFile('tubular_objects')
    if seg.IsNoneImage(tubular_obj):
        tubular_obj = seg.GetTubularOrPlaneOrEllipsoidalObjects(without_heart_threshold_mask, seg.ObjectShapeType.Tubular,
                                                                _sigma_minimum=1.0, _sigma_maximum=2.0,
                                                                _number_of_sigma_steps=2)
        debug.CheckTimer()
        debug.SaveImageToFile(tubular_obj, 'tubular_objects')
    else:
        debug.Print('Получение результата фильтрации трубкообразных объектов из файла')

    tubular_obj = sitk.Cast(tubular_obj, sitk.sitkInt16)
    tubular_obj = seg.Threshold(tubular_obj, 30, 255)
    debug.CheckTimer()
    debug.SaveImageToFile(tubular_obj, 'rso')

    # Выделение области легких
    debug.Print('Поиск лёгких')
    lung = np.where(np.logical_and(img_arr > -9999, img_arr < -200), 1, 0)
    lung = seg.CloseNumpyArray(lung, 5.0, image_spacing)
    debug.CheckTimer()
    debug.SaveNumpyArrayToFile(lung, 'lung', _heart_phase_image, _pixel_id_value=sitk.sitkUInt8)

    # Вычитание сосудов легких из общей картины
    debug.Print('Вычитание лёгких')
    tubular_obj = sitk.GetArrayFromImage(tubular_obj)
    res = np.where(np.logical_and(tubular_obj == 1, lung == 0), 1, 0)
    res = data.ConvertNumpyArrayToSimpleItkImage(res, _heart_phase_image, _pixel_id_value=sitk.sitkUInt8)
    debug.CheckTimer()
    debug.SaveImageToFile(res, 'remove_lung')
    res = sitk.GetArrayFromImage(res)

    # Добавление аорты (820)
    debug.Print('Добавление аорты')
    radius: float = 2.5
    heart_820 = seg.DilateNumpyArray(heart_820, radius, image_spacing)
    res = seg.DilateNumpyArray(res, radius, image_spacing)
    debug.CheckTimer()
    debug.SaveNumpyArrayToFile(res, 'plus_820_and_dilate', _heart_phase_image, _pixel_id_value=sitk.sitkUInt8)

    # Заливка из посевной точки внутри аорты (выделятся нужные сосуды)
    debug.Print('Заливка из посевной точки внутри аорты (выделятся нужные сосуды)')
    res = np.where(np.logical_or(res == 1, heart_820 == 1), 1, 0)
    res = data.ConvertNumpyArrayToSimpleItkImage(res, _heart_phase_image, _pixel_id_value=sitk.sitkUInt8)

    # properties: List = skime.regionprops(heart_820)
    # seed_point: List[int] = list(properties[0]['coords'][0])
    # seed_point = [int(point) for point in seed_point]

    # Получение маски всех сосудов серца путём наращивания из аорты
    # debug.Print(f'Координаты посевной точки: {seed_point}')
    heart_vessel_mask = seg.ConfidenceConnectedThreshold(res, _number_of_iterations=1)  # for 22
    heart_vessel_mask = sitk.GetArrayFromImage(heart_vessel_mask)
    heart_vessel_mask = np.where(np.logical_and(heart_vessel_mask == 1, heart_820 == 0), 1, 0)
    heart_vessel_mask = seg.ErodeNumpyArray(heart_vessel_mask, 2.5, image_spacing)
    heart_vessel_mask = data.ConvertNumpyArrayToSimpleItkImage(heart_vessel_mask, _heart_phase_image,
                                                               _pixel_id_value=sitk.sitkUInt8)
    debug.CheckTimer()
    debug.SaveImageToFile(heart_vessel_mask, 'heart_vessels_mask', _ignore_numbers=True)

    debug.UnselectDirectory()

    return heart_vessel_mask
