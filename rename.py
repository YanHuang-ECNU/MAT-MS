# encoding=utf-8
import os
from datetime import datetime, timedelta
from osgeo import gdal
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from Utils import draw, mkdir, get_tif_info, cfg, Multiprocessor
from PIL import Image, ImageDraw, ImageFont


def new_name(fn):
    if not fn.startswith('MATMS_'):
        year, month, day = int(fn[0:4]), int(fn[4:6]), int(fn[6:8])
        day_num = (datetime(year, month, day) - datetime(year, 1, 1)).days + 1
        return f"MATMS_TP_{year}{day_num:03d}.tif"
    else:
        date_name = fn.split('_')[-1]
        date_name = date_name.split('.')[0]
        return f"MATMS_TP_{date_name}.tif"


def rename():
    # folder = 'H:/HuaRY/Data/output_tif'
    folder = 'E:\\HRY\\Data\\FINAL\\output_tif_may_err'
    for filename in os.listdir(folder):
        fp = os.path.join(folder, filename)
        new_n = new_name(filename)
        if new_n:
            new_fp = os.path.join(folder, new_n)
            # print(fp, new_fp)
            os.rename(fp, new_fp)


def load_tif(data_path):
    fptr = gdal.Open(str(data_path))
    data = fptr.ReadAsArray()
    geo_transform = fptr.GetGeoTransform()
    projection = fptr.GetProjection()
    fptr = None
    return data, (geo_transform, projection)


def rerange():
    # dir_data = Path('H:/HuaRY/Data/output_tif')
    dir_data = Path('E:\\HRY\\Data\\FINAL\\output_tif_may_err')
    tif_info = get_tif_info(cfg.DEM.tif_path)
    for filepath in dir_data.glob('*.tif'):
        # if filepath.stem <= 'MATMS_TP_2006108.tif':
        #     continue
        tif_data, _ = load_tif(filepath)
        if tif_data is None:
            print(filepath, 'is None.')
            continue
        # print(tif_data.dtype, tif_data.shape)
        # print(tif_data[..., 0, 0])
        # break
        # new_tif_data = tif_data / 10000
        # print(np.sum(np.isnan(tif_data) == np.isnan(new_tif_data)), 3149*6908)
        # print(np.max(tif_data[~np.isnan(tif_data)]), np.min(tif_data[~np.isnan(tif_data)]))
        # break
        # max_v = np.max(tif_data[~np.isnan(tif_data)])
        # print(max_v)
        if True:    # max_v > 2:
            tif_data = tif_data / 10000

            geo_transform, projection = tif_info
            driver = gdal.GetDriverByName("GTiff")
            out_raster = driver.Create(str(filepath), tif_data.shape[1], tif_data.shape[0], 1, gdal.GDT_Float32)
            out_raster.SetGeoTransform(geo_transform)
            out_raster.SetProjection(projection)
            out_band = out_raster.GetRasterBand(1)
            out_band.WriteArray(tif_data)
            out_band.FlushCache()
            out_raster.FlushCache()


def add_round(pic, line_width, color):
    ph, pw = pic.shape[:2]
    if isinstance(color, int):
        color = [color, color, color]
    c_arr = np.array(color, dtype=np.uint8).reshape((1, 1, 3))
    c_arr = np.tile(c_arr, (ph + 2 * line_width, pw + 2 * line_width, 1))
    c_arr[line_width:-line_width, line_width:-line_width] = pic
    return c_arr


def single_check(filepath, pic_save_path):
    tif_data, _ = load_tif(filepath)
    nan_mask = np.isnan(tif_data)
    scale = 0.1
    p1 = draw(tif_data, cmap='ndsi', normed=True, scale=scale, mask_color_lst=[(nan_mask, (255, 255, 255))])
    p1 = add_round(p1, 5, 255)
    p1 = add_round(p1, 2, 0)
    ph, pw = p1.shape[:2]
    # th, tw = int(0.1 * ph), int(0.1 * pw)
    # font, fs = cv2.FONT_HERSHEY_COMPLEX, 2
    # p1 = cv2.putText(p1, "Gap-filled NDSI", (tw, th), font, fontScale=fs, color=(0, 0, 0))
    date_name = filepath.stem.split('_')[-1]
    # p1 = cv2.putText(p1, date_name, (pw - 2 * tw, th), font, fontScale=fs, color=(0, 0, 0))
    year = int(date_name[:4])
    date = datetime(year, 1, 1) + timedelta(int(date_name[4:7]) - 1)
    month, day = date.month, date.day

    ori_ndsi_path = Path('H:/HuaRY/Data/NDSI') / f"{year}{month:02d}{day:02d}.npz"
    try:
        npz_data = np.load(str(ori_ndsi_path))
        ori_ndsi, ori_mask = npz_data['data'], npz_data['mask']
    except PermissionError:
        print(ori_ndsi_path)
        ori_ndsi, ori_mask = None, None
        exit(1)
    p2 = draw(ori_ndsi, cmap='ndsi', normed=False, scale=scale,
              mask_color_lst=[(ori_mask, (90, 90, 90)), (nan_mask, (255, 255, 255))])
    p2 = add_round(p2, 5, 255)
    p2 = add_round(p2, 2, 0)
    p2 = p2[:-2]
    # p2 = cv2.putText(p2, "Original NDSI", (tw, th), font, fontScale=fs, color=(0, 0, 0))

    cat_pic = np.concatenate([p2, p1], axis=0)
    cat_w = cat_pic.shape[1]
    bar_pic = np.full((80, cat_w, 3), fill_value=255)
    cmap_arr = np.arange(int(1 * cat_w))
    cmap_arr = cmap_arr / np.max(cmap_arr)
    cmap_arr = 2 * cmap_arr - 1
    cmap_arr = np.tile(cmap_arr.reshape(1, -1), (24, 1))
    cmap_pic = draw(cmap_arr, cmap='ndsi', normed=True)
    bw = 0
    bar_pic[16:16 + cmap_pic.shape[0], bw:bw + cmap_pic.shape[1]] = cmap_pic

    cat_pic = np.concatenate([cat_pic, bar_pic], axis=0)
    cat_pic = add_round(cat_pic, 12, 255)
    cv2.imwrite(str(pic_save_path), cat_pic)


def check():
    dir_data = Path('H:/HuaRY/Data/output_tif')
    dir_pic_view = mkdir('H:/HuaRY/Data/output_tif_vis')
    pool = Multiprocessor(processes=1)
    for filepath in dir_data.rglob('*.tif'):
        if not filepath.stem.endswith('2008154'):
            continue
        year = filepath.stem.split('_')[-1][:4]
        pic_save_path = mkdir(dir_pic_view / year) / (filepath.stem + '.png')
        if pic_save_path.exists():
            continue
        pool.add(single_check, (filepath, pic_save_path))
        # break
    pool.close_and_join()


def single_redraw(filepath, pic_save_path):
    img = Image.open(str(filepath))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/timesbd.ttf", size=28)
    d.text((30, 260), "Original NDSI", font=font, fill=(0, 0, 0))
    date = filepath.stem.split('_')[-1]
    d.text((220, 40), date, font=font, fill=(0, 0, 0))
    d.text((30, 280 + 7 + 315), "Gap-filled NDSI", font=font, fill=(0, 0, 0))
    font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", size=24)
    d.text((15, 710), "-1", font=font, fill=(0, 0, 0))
    d.text((345, 710), "0", font=font, fill=(0, 0, 0))
    d.text((700, 710), "1", font=font, fill=(0, 0, 0))
    font = ImageFont.truetype("C:/Windows/Fonts/timesbd.ttf", size=22)
    d.text((55, 300), "Data gaps", font=font, fill=(0, 0, 0))
    d.rectangle([(30, 305), (45, 320)], fill=(90, 90, 90))
    img.save(str(pic_save_path))


def redraw():
    dir_pic_view = mkdir('H:/HuaRY/Data/output_tif_vis')
    pool = Multiprocessor(processes=1)
    for filepath in dir_pic_view.rglob('*.png'):
        if not filepath.stem.endswith('2008154'):
            continue
        # pic_save_path = Path('H:/HuaRY/Data/test_font.png')
        pool.add(single_redraw, (filepath, filepath))
        # break
    pool.close_and_join()


if __name__ == '__main__':
    # rename()
    # rerange()
    check()
    redraw()
