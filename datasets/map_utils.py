
import cv2
import math
import imageio
import numpy as np
import geopy.distance
import matplotlib.pyplot as plt
from staticmap import StaticMap, Polygon


def _lon_to_x(lon, zoom):
    if not (-180 <= lon <= 180): lon = (lon + 180) % 360 - 180
    return ((lon + 180.) / 360) * pow(2, zoom)


def _lat_to_y(lat, zoom):
    if not (-90 <= lat <= 90): lat = (lat + 90) % 180 - 90
    return (1 - math.log(math.tan(lat * math.pi / 180) + 1 / math.cos(lat * math.pi / 180)) / math.pi) / 2 * pow(2, zoom)


def _download_map_image(min_lat=45.0, min_lon=7.6, max_lat=45.1, max_lon=7.7, size=2000):
    """"Download a map of the chosen area as a numpy image"""
    mean_lat = (min_lat + max_lat) / 2
    mean_lon = (min_lon + max_lon) / 2
    static_map = StaticMap(size, size)
    static_map.add_polygon(
        Polygon(((min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat)), None, '#FFFFFF'))
    zoom = static_map._calculate_zoom()

    # print(((min_lat, min_lon), (max_lat, max_lon)))
    dist = geopy.distance.geodesic((min_lat, min_lon), (max_lat, max_lon)).m
    if dist < 50:
        zoom = 22
    else:
        zoom = 20  # static_map._calculate_zoom()
    static_map = StaticMap(size, size)
    image = static_map.render(zoom, [mean_lon, mean_lat])
    # print(f"You can see the map on Google Maps at this link www.google.com/maps/place/@{mean_lat},{mean_lon},{zoom - 1}z")
    min_lat_px, min_lon_px, max_lat_px, max_lon_px = \
        static_map._y_to_px(_lat_to_y(min_lat, zoom)), \
        static_map._x_to_px(_lon_to_x(min_lon, zoom)), \
        static_map._y_to_px(_lat_to_y(max_lat, zoom)), \
        static_map._x_to_px(_lon_to_x(max_lon, zoom))
    assert 0 <= max_lat_px < min_lat_px < size and 0 <= min_lon_px < max_lon_px < size
    return np.array(image)[max_lat_px:min_lat_px, min_lon_px:max_lon_px], static_map, zoom


def get_edges(coordinates, enlarge=0):
    """
    Send the edges of the coordinates, i.e. the most south, west, north and
        east coordinates.
    :param coordinates: A list of numpy.arrays of shape (Nx2)
    :param float enlarge: How much to increase the coordinates, to enlarge
        the area included between the points
    :return: a tuple with the four float
    """
    min_lat, min_lon, max_lat, max_lon = (*np.concatenate(coordinates).min(0), *np.concatenate(coordinates).max(0))
    diff_lat = (max_lat - min_lat) * enlarge
    diff_lon = (max_lon - min_lon) * enlarge
    inc_min_lat, inc_min_lon, inc_max_lat, inc_max_lon = \
        min_lat - diff_lat, min_lon - diff_lon, max_lat + diff_lat, max_lon + diff_lon
    return inc_min_lat, inc_min_lon, inc_max_lat, inc_max_lon


def create_map(coordinates, colors=None, dot_sizes=None, legend_names=None, map_intensity=0.6):

    dot_sizes = dot_sizes if dot_sizes is not None else [10] * len(coordinates)
    colors = colors if colors is not None else ["r"] * len(coordinates)
    assert len(coordinates) == len(dot_sizes) == len(colors), \
        f"The number of coordinates must be equals to the number of colors and dot_sizes, but they're " \
        f"{len(coordinates)}, {len(colors)}, {len(dot_sizes)}"
    
    # Add two dummy points to slightly enlarge the map
    min_lat, min_lon, max_lat, max_lon = get_edges(coordinates, enlarge=0.1)
    coordinates.append(np.array([[min_lat, min_lon], [max_lat, max_lon]]))
    # Download the map of the chosen area
    map_img, static_map, zoom = _download_map_image(min_lat, min_lon, max_lat, max_lon)
    
    scatters = []
    fig = plt.figure(figsize=(map_img.shape[1] / 100, map_img.shape[0] / 100), dpi=1000)
    for i, coord in enumerate(coordinates):
        for i in range(len(coord)):  # Scale latitudes because of earth's curvature
            coord[i, 0] = -static_map._y_to_px(_lat_to_y(coord[i, 0], zoom))
    for coord, size, color in zip(coordinates, dot_sizes, colors):
        scatters.append(plt.scatter(coord[:, 1], coord[:, 0], s=size, color=color))
    
    if legend_names != None:
        plt.legend(scatters, legend_names, scatterpoints=1, loc='best',
                   ncol=1, framealpha=0, prop={"weight": "bold", "size": 20})
    
    min_lat, min_lon, max_lat, max_lon = get_edges(coordinates)
    plt.ylim(min_lat, max_lat)
    plt.xlim(min_lon, max_lon)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    fig.canvas.draw()
    plot_img = np.array(fig.canvas.renderer._renderer)
    plt.close()
    
    plot_img = cv2.resize(plot_img[:, :, :3], map_img.shape[:2][::-1], interpolation=cv2.INTER_LANCZOS4)
    map_img[(map_img.sum(2) < 444)] = 188  # brighten dark pixels
    map_img = (((map_img / 255) ** map_intensity) * 255).astype(np.uint8)  # fade map
    mask = (plot_img.sum(2) == 255 * 3)[:, :, None]  # mask of plot, to find white pixels
    final_map = map_img * mask + plot_img * (~mask)
    return final_map


if __name__ == "__main__":
    # Create a map containing major cities of Italy, Germany and France
    coordinates = [
        np.array([[41.8931, 12.4828], [45.4669, 9.1900], [40.8333, 14.2500]]),
        np.array([[52.5200, 13.4050], [48.7775, 9.1800], [48.1375, 11.5750]]),
        np.array([[48.8567, 2.3522], [43.2964, 5.3700], [45.7600, 4.8400]])
    ]
    map_img = create_map(
        coordinates,
        colors=["green", "black", "blue"],
        dot_sizes=[1000, 1000, 1000],
        legend_names=[
            "Main Italian Cities",
            "Main German Cities",
            "Main French Cities",
        ])
    
    imageio.imsave("cities.png", map_img)

