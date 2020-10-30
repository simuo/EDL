import os
import json
from cfg import *

jsonfilepath = r'D:\VIDI\ExploreDLInIndustry\Betel_nut\outputs'
jsonstr = [os.path.join(jsonfilepath, x) for x in os.listdir(jsonfilepath)]
ff = open('label.txt', 'w')
for str in jsonstr:
    with open(str, 'r', encoding='utf-8') as f:
        strs = json.loads(f.read())
        # print(strs)
        imgpath = strs['path']
        print(strs)
        infos = strs['outputs']['object']
        cls = [nameinfo['name'] for nameinfo in infos]
        boxlist = [box['bndbox'] for box in infos]
        boxes = [[box['xmin'], box['ymin'], box['xmax'], box['ymax']] for box in boxlist]

        w, h = strs['size']['width'], strs['size']['height']
        # print(w, h)
        w_scale, h_scale = w / IMG_WIDTH, h / IMG_HEIGHT

        ff.write(imgpath.split('\\')[-1])
        for clsname, box in zip(cls, boxes):
            # print(clsname, box)
            clsnumber = COCO_CLASS.index(clsname)
            _x1, _y1, _x2, _y2 = box
            _w, _h = _x2 - _x1, _y2 - _y1
            _w0_5, _h0_5 = _w / 2, _h / 2
            _cx, _cy = _x1 + _w0_5, _y1 + _h0_5
            x1, y1, w, h = int(_cx / w_scale), int(_cy / h_scale), int(_w / w_scale), int(_h / h_scale)
            ff.write(" {} {} {} {} {}".format(clsnumber, x1, y1, w, h))
        ff.write("\n")
        ff.flush()

