#!/usr/bin/env python # [1]
"""
Generate painting by numbers from any image along with the color palette.
Author: Maksim Lukashkou (lukashkoum1@gmail.com)
Usage: paint-by-numbers.py -i input_image.jpg -o output_image.jpg
Additional arguments:
-w width of the output image (height is scaled automatically)
-k number of colors to be used (number of clusters for kmeans algorithm)
-b blur of gaussian blur to be applied before quantization (smaller value results in less segments)
-ma to set the minimum value for polygon area (polygons with smaller area will be merged with the largest neighbour)
"""
import math
import os

import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from shapely import Polygon, union, MultiPolygon
from shapely.algorithms.polylabel import polylabel

# Private setting for merging small polygons
buffer_size = 1
# KMeans parameters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 10, 1.0)
# Color palette parameters
square_size = 75
spacing = 50
# Font for palette labels
font_palette = ImageFont.truetype("arialbd.ttf", size=16)
# Polygon reduce parameters
smallest_polygon_area = 10 * 100


class ColoredPolygon:
    def __init__(self, polygon, color):
        self.polygon = polygon
        self.color = color

    def __setattr__(self, name, value):
        if name == 'polygon' and not isinstance(value, (Polygon, MultiPolygon)):
            raise TypeError('A.polygon must be a shapely Polygon')
        super().__setattr__(name, value)


def image_resize(img_to_resize: Image, height: int = None, width: int = None) -> Image:
    if height is None and width is None:
        return img_to_resize
    ratio = width / float(img_to_resize.width)
    return img_to_resize.resize((width, int(img_to_resize.height * ratio)), Image.Resampling.LANCZOS)


def read_image(path: str, width: int) -> Image:
    image = Image.open(path)
    if width is not None and isinstance(width, int) and width > 0:
        image = image_resize(image, width=width)
    return image


def save_images(images: list, name_suffixes: list, path: str) -> None:
    directory, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)
    for idx, image in enumerate(images):
        img_path = os.path.join(directory, name + '-' + str(name_suffixes[idx]) + ext)
        image.save(img_path, "PNG")


def show_images(images: list) -> None:
    for image in images:
        image.show()


def flatten(in_list: list) -> list:
    return [el for in_l in in_list for el in in_l]


def generate_color_palette(colors: list) -> Image:
    # Creates an image with numbered colored rectangles that represent a color palette for given colors
    img_width = len(colors) * (square_size + spacing) - spacing
    img_height = square_size
    img_palette = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img_palette)
    x = 0
    for i, color in enumerate(colors, 1):
        draw.rectangle((x, 0, x + square_size, square_size), fill=color)
        text = str(i)
        text_width = draw.textlength(text, font=font_palette)
        text_height = font_palette.size
        text_x = x + (square_size - text_width) / 2
        text_y = (square_size - text_height) / 2
        draw.text((text_x, text_y), text, fill="black", font=font_palette)
        x += square_size + spacing
    return img_palette


def get_painting_outline_with_numbers(colored_polygons: list, outline: Image, centers: list, img_width: int) -> Image:
    # Adds numbers that indicate color to provided outline by using visual centers of polygons and
    # a color ID of a polygon. Computes the font size based on the polygon area.
    draw = ImageDraw.Draw(outline)
    areas = [cl_pl.polygon.area for cl_pl in colored_polygons]
    for idx, colored_polygon in enumerate(colored_polygons):
        font_size = min(math.ceil(areas[idx] / 25), 30) * img_width / 1920
        font_painting = ImageFont.truetype("arialbd.ttf", size=font_size)
        x = round(centers[idx].x)
        y = round(centers[idx].y)
        try:
            color_id = str(colored_polygon.color + 1)
        except ValueError:
            continue
        text_width = draw.textlength(color_id, font=font_painting)
        text_height = font_painting.size
        text_x = x - (text_width / 2)
        text_y = y - (text_height / 2)
        draw.text((text_x, text_y), color_id, fill="black", font=font_painting)
    return outline


def find_visual_centers(polygons: list) -> list:
    # Finds the best position for placing a label inside a polygon. Uses 'poly label' function from shapely.
    visual_centers = map(lambda p: polylabel(p, 1), polygons)
    return list(visual_centers)


def subtract_holes(polygons: Polygon | MultiPolygon | list,
                   holes: Polygon | MultiPolygon | list) -> Polygon | MultiPolygon | list:
    if isinstance(polygons, (Polygon, MultiPolygon)):
        if isinstance(holes, (Polygon, MultiPolygon)):
            polygons = polygons.difference(holes)
        elif isinstance(holes, list):
            for hole in holes:
                polygons = polygons.difference(hole)
    elif isinstance(polygons, list):
        if isinstance(holes, (Polygon, MultiPolygon)):
            polygons = [poly.difference(holes) for poly in polygons]
        elif isinstance(holes, list):
            for hole in holes:
                polygons = [poly.difference(hole) for poly in polygons]
    return polygons


def generate_polygons(outline: np.ndarray) -> list:
    # Generate polygons based on contours for a given outline/mask. Works by using cv2 findContours function
    kernel = np.ones((3, 3), np.uint8)
    outline = cv2.dilate(outline, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(outline, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = np.squeeze(hierarchy)
    ret = list()
    for i, contour in enumerate(contours):
        contour = np.squeeze(contour)
        to_append = None
        if len(contour) > 2:  # Valid contours have at least 3 points
            if hierarchy.ndim == 1:
                to_append = make_polygon_valid(Polygon(contour))
            elif hierarchy[i][3] == -1:  # Parent case (external contour)
                to_append = make_polygon_valid(Polygon(contour))
                next_child = int(hierarchy[i][2])
                while next_child != -1:
                    next_contour = np.squeeze(contours[next_child])
                    if len(next_contour) > 2:
                        hole_to_subtract = make_polygon_valid(Polygon(next_contour))
                        to_append = subtract_holes(to_append, hole_to_subtract)
                    next_child = hierarchy[next_child][0]
                to_append = make_polygon_valid(to_append)
            if to_append is not None and isinstance(to_append, Polygon):
                ret.append(to_append)
            elif to_append is not None and isinstance(to_append, list):
                ret.extend(to_append)
    return ret


def make_polygon_valid(polygon: Polygon | MultiPolygon | list) -> Polygon | MultiPolygon | list:
    if polygon is None or (isinstance(polygon, Polygon) and polygon.is_empty):
        return None
    if isinstance(polygon, list):
        ret = list()
        for poly in polygon:
            valid = make_polygon_valid(poly)
            if isinstance(valid, Polygon):
                ret.append(valid)
            if isinstance(valid, list):
                ret.extend(valid)
        return ret
    fixed_polygons = list()
    if isinstance(polygon, Polygon):
        if polygon.is_valid:
            return polygon
        else:
            fixed_polygon = polygon.buffer(0)
            if isinstance(fixed_polygon, Polygon):
                return fixed_polygon
            if isinstance(fixed_polygon, MultiPolygon):
                fixed_polygons = [make_polygon_valid(poly) for poly in fixed_polygon.geoms]
    elif isinstance(polygon, MultiPolygon):
        for polygon in polygon.geoms:
            valid_poly = make_polygon_valid(polygon)
            if isinstance(valid_poly, Polygon):
                fixed_polygons.append(valid_poly)
            elif isinstance(valid_poly, list):
                for poly in valid_poly:
                    if isinstance(poly, Polygon):
                        fixed_polygons.append(poly)
    fixed_polygons = [poly for poly in fixed_polygons if (poly is not None) and (not poly.is_empty)]
    return fixed_polygons


def generate_colored_polygons_for_all_colors(labels: np.ndarray, colors: list) -> list:
    # Generates polygons for each color by masking every pixel of other color and
    # calling generate_polygons on the mask
    colored_polygons_per_color = list()
    for idx, color in enumerate(colors):
        colored_polygons_per_color.append(list())
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == idx] = 255
        polygons = generate_polygons(mask)
        colored_polygons = [ColoredPolygon(polygon, idx) for polygon in polygons]
        colored_polygons_per_color[idx].extend(colored_polygons)
    return colored_polygons_per_color


def centred_dist_matrix(m, n):
    # Create coordinate grids
    x = np.linspace(-m // 2, m // 2, m)
    y = np.linspace(-n // 2, n // 2, n)
    x, y = np.meshgrid(x, y)

    # Calculate distances from the center
    dists = np.sqrt(x ** 2 + y ** 2)
    return dists


def merge_small_colored_polygons(colored_polygons: list) -> list:
    # Iterates over all the polygons and merges polygons that have area smaller than --merge-area argument.
    # It does so by finding the largest neighbour and computing the union of two polygons. Assigns colorID
    # of a larger polygon.
    ret_colored_polygons = colored_polygons.copy()
    for pl in ret_colored_polygons:
        idx = ret_colored_polygons.index(pl)
        if pl is None:
            continue
        if pl.polygon.area < smallest_polygon_area:
            polygons = [colored_poly.polygon if colored_poly is not None else None for colored_poly in
                        ret_colored_polygons]
            polygon_to_merge_id = get_largest_neighbour_id(pl.polygon, polygons)
            res_polygon = ColoredPolygon(union(ret_colored_polygons[polygon_to_merge_id].polygon, pl.
                                               polygon.buffer(buffer_size)),
                                         ret_colored_polygons[polygon_to_merge_id].color)
            ids_to_delete = [polygon_to_merge_id, idx]
            for idx in ids_to_delete:
                ret_colored_polygons[idx] = None
            if isinstance(res_polygon.polygon, MultiPolygon):
                ret_colored_polygons.extend(
                    [ColoredPolygon(polygon, res_polygon.color) for polygon in res_polygon.polygon.geoms])
                continue
            ret_colored_polygons.append(res_polygon)
    ret_polygons = [pl for pl in ret_colored_polygons if pl is not None and pl.polygon is not None]
    return ret_polygons


def get_largest_neighbour_id(polygon: Polygon | MultiPolygon, polygons: list) -> int:
    # Finds the largest neighbour of the given polygon by iterating over all the polygons and finding the largest
    # among those that intersect or contain the given polygon.
    neighbours_ids = list()
    polygon_test = polygon.buffer(buffer_size)
    biggest_polygon_id = polygons.index(polygon)
    for idx, polygon in enumerate(polygons):
        if polygon is None:
            continue
        if polygon_test.intersects(polygon) or polygon.contains(polygon_test):
            if polygon != polygon_test:
                neighbours_ids.append(idx)
    for idx in neighbours_ids:
        if polygons[biggest_polygon_id].area < polygons[idx].area:
            biggest_polygon_id = idx
    return biggest_polygon_id


def plot_outline_from_polygons(polygons: list) -> None:
    for polygon in polygons:
        plt.plot(*polygon.exterior.xy)
    plt.gca().invert_yaxis()
    plt.show()


def generate_inverted_outline_from_polygons(polygons: list, shape: tuple) -> np.ndarray:
    outline = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        coords = list(zip(polygon.exterior.xy[0], polygon.exterior.xy[1]))
        cv2.polylines(outline, pts=np.int32([coords]), color=(255,), isClosed=True, thickness=2)
    outline_inverted = cv2.bitwise_not(outline)
    return outline_inverted


def generate_color_image_from_polygons(colored_polygons: list, colors: list, shape: tuple) -> Image:
    # Generate resulting image from colored polygons by taking exterior coordinates and coloring them.
    res_img = Image.new('RGBA', (shape[1], shape[0]), (255, 255, 255, 0))
    for colored_polygon in colored_polygons:
        polygon = colored_polygon.polygon
        mask = Image.new('L', (shape[1], shape[0]), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygon.exterior.coords, fill=255, outline=255)
        interiors = [list(interior.coords) for interior in polygon.interiors]
        for interior in interiors:
            draw.polygon(interior, fill=0, outline=255)
        res_img.paste(Image.new('RGBA', (shape[1], shape[0]), colors[colored_polygon.color]), mask=mask)
    return res_img


def apply_gaussian_filter(image, sigma):
    dists = centred_dist_matrix(image.shape[1], image.shape[0])
    gaussian_ker = np.exp(-(dists ** 2) / (2 * sigma ** 2))
    image_fft = np.fft.fftshift(np.fft.fft2(image))
    res = np.multiply(gaussian_ker, image_fft)
    res = np.fft.ifft2(np.fft.ifftshift(res))
    res = np.uint8(np.real(res))
    return res


def preprocess_image(input_img: np.ndarray, blur_k: int) -> np.ndarray:
    if blur_k > 0:
        r, g, b = input_img[:, :, 0], input_img[:, :, 1], input_img[:, :, 2]
        r_filtered = apply_gaussian_filter(r, blur_k)
        g_filtered = apply_gaussian_filter(g, blur_k)
        b_filtered = apply_gaussian_filter(b, blur_k)
        res = np.stack((r_filtered, g_filtered, b_filtered), axis=-1)
    else:
        res = input_img
    return res


def generate_painting(input_img: np.ndarray, kmeans_k: int, blur_k: int) -> tuple[Image, Image, Image]:
    img_array = np.array(input_img)
    img_blured = preprocess_image(img_array, blur_k)
    plt.imshow(img_blured)
    plt.show()
    img_blured = cv2.cvtColor(img_blured, cv2.COLOR_RGB2LAB)
    _, labels, centers = cv2.kmeans(img_blured.reshape((-1, 3)).astype(np.float32), K=kmeans_k,
                                    bestLabels=None, criteria=criteria, attempts=10,
                                    flags=cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(img_array.shape[:2])
    centers = cv2.cvtColor(np.uint8([centers]), cv2.COLOR_LAB2RGB)[0]
    colors = [(r, g, b) for (r, g, b) in centers]
    colored_polygons = generate_colored_polygons_for_all_colors(labels, colors)
    colored_polygons = flatten(colored_polygons)
    colored_polygons = merge_small_colored_polygons(colored_polygons)
    outline_inverted = generate_inverted_outline_from_polygons([pl.polygon for pl in colored_polygons], labels.shape)
    outline_img = Image.fromarray(outline_inverted)
    centers = find_visual_centers([pl.polygon for pl in colored_polygons])
    outline_with_numbers = get_painting_outline_with_numbers(colored_polygons, outline_img, centers, labels.shape[1])
    resulting_painting = generate_color_image_from_polygons(colored_polygons, colors, labels.shape)
    palette_img = generate_color_palette(colors)
    return outline_with_numbers, resulting_painting, palette_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input-image',
        type=str,
        required=True)
    parser.add_argument(
        '-o',
        '--output-image',
        type=str,
        required=False,
        default="./output/output.jpg")
    parser.add_argument(
        '-w',
        '--image-width',
        type=int,
        required=False,
        default=1024,
        help='Width of the output')
    parser.add_argument(
        '-k',
        '--num-clusters',
        type=int,
        required=False,
        default=16,
        help='Number of colors for an image (kmeans clusters). Default is 16')
    parser.add_argument(
        '-b',
        '--blur-intensity',
        type=int,
        required=False,
        default=20,
        help='Parameter for preprocessing the image. Used to reduce number of small segments in a final result.'
             'The lower the value, the stronger blur will be applied. Default is 20')
    parser.add_argument(
        '-ma',
        '--merge-area',
        type=int,
        required=False,
        default=1,
        help="Merge polygons with area smaller than this argument (area is relative, try different values). Default "
             "is 1")

    args = parser.parse_args()
    smallest_polygon_area = args.merge_area * 100
    img = read_image(args.input_image, args.image_width)
    outline_with_numbers_img, result_img, color_palette_img = generate_painting(input_img=img,
                                                                                kmeans_k=args.num_clusters,
                                                                                blur_k=args.blur_intensity)
    save_images([outline_with_numbers_img, result_img, color_palette_img],
                ['outline', 'result', 'palette'], args.output_image)
