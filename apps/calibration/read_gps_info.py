import argparse, os, math
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from tqdm import tqdm
from glob import glob

def get_exif(filename):
    image = Image.open(filename)
    image.verify()
    return image._getexif()

def get_gps_info(exif):
    if 'GPSInfo' in exif:
        gps_info = {}
        for key in exif['GPSInfo'].keys():
            decode = GPSTAGS.get(key, key)
            gps_info[decode] = exif['GPSInfo'][key]
        return gps_info
    else:
        return None
    
def to_decimal_coordinates(gps_info):
    def convert_to_degrees(value):
        d, m, s = value
        return d + (m / 60.0) + (s / 3600.0)

    lat = gps_info['GPSLatitude']
    lat_ref = gps_info['GPSLatitudeRef']
    lon = gps_info['GPSLongitude']
    lon_ref = gps_info['GPSLongitudeRef']

    lat = convert_to_degrees(lat)
    if lat_ref != "N":
        lat = -lat

    lon = convert_to_degrees(lon)
    if lon_ref != "E":
        lon = -lon

    assert 'GPSAltitude' in gps_info
    altitude = gps_info['GPSAltitude']
    if 'GPSAltitudeRef' in gps_info and gps_info['GPSAltitudeRef'] == 1:
        altitude = -altitude

    return (lat, lon, altitude)

def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='./images/')
    parser.add_argument('--output_path', default='./gps.npy')
    parser.add_argument('--multifolder', action='store_true')
    args = parser.parse_args()

    image_path = args.image_path
    output_path = args.output_path

    coordinates_3d = dict()

    ref_point = None

    if args.multifolder:
        img_f_list = []
        for sub in sorted(os.listdir(image_path)):
            img_f_list.extend(
                [os.path.join(sub, filename) for filename in sorted(os.listdir(os.path.join(image_path, sub)))]
            )
    else:
        img_f_list = sorted(os.listdir(image_path))

    for img_f in tqdm(img_f_list):
        exif = get_exif(os.path.join(image_path, img_f))
        if exif:
            labeled_exif = {TAGS.get(key): val for key, val in exif.items() if key in TAGS}
            gps_info = get_gps_info(labeled_exif)
            if gps_info:
                lat, lon, alt = to_decimal_coordinates(gps_info)
                if ref_point is None:
                    ref_point = [0, 0]
                distance_x = calculate_distance((ref_point[0], ref_point[1]), (lat, ref_point[1]))
                distance_y = calculate_distance((ref_point[0], ref_point[1]), (ref_point[0], lon))
                distance_z = float(alt)
                coordinates_3d[img_f] = (distance_x, distance_y, distance_z)
        else:
            print(f'No exif for {img_f}')

    np.save(output_path, coordinates_3d)