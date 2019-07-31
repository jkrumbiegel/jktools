from svg.path import parse_path
from svg.path.path import Path, CubicBezier, Arc, QuadraticBezier, Move
from matplotlib.path import Path as mPath
import xmltodict
import numpy as np
from jktools.geometry import remove_redundant_movetos
from collections import OrderedDict


def read_svg_paths(svg_file):

    with open(svg_file, 'r') as f:
        xml = f.read()
    xml_dict = xmltodict.parse(xml)
    path_dicts = xml_dict['svg']['path']
    paths = OrderedDict((pd['@id'], pd['@d']) for pd in path_dicts)
    return paths

def convert_svg_path_to_mpl(path_string):
    parsed = parse_path(path_string)

    vertices = []
    codes = []

    i_to_list = lambda i: [i.real, i.imag]

    for part in parsed:
        if isinstance(part, Move):
            vertices.append(i_to_list(part.start))
            codes.append(mPath.MOVETO)

        elif isinstance(part, CubicBezier):
            vertices.extend([i_to_list(part.start), i_to_list(part.control1), i_to_list(part.control2), i_to_list(part.end)])
            codes.extend([mPath.MOVETO, mPath.CURVE4, mPath.CURVE4, mPath.CURVE4])

        elif isinstance(part, QuadraticBezier):
            raise Exception('QuadraticBezier not implemented.')
        elif isinstance(part, Arc):
            raise Exception('Arc not implemented.')
        else:
            raise Exception(f'{type(part)} not implemented.')
    if parsed.closed:
        vertices.append([np.nan, np.nan])
        codes.append(mPath.CLOSEPOLY)

    path = mPath(np.array(vertices), np.array(codes, dtype=np.uint8))
    return remove_redundant_movetos(path)

# svg_paths = read_svg_paths('/Users/juliuskrumbiegel/Dropbox/Uni/Mind and Brain/Rolfslab Master/Visuals/head-top-down.svg')
# paths = dict((key, convert_svg_path_to_mpl(value)) for key, value in svg_paths.items())
#
#
# import matplotlib.pyplot as plt
# from matplotlib.patches import PathPatch
#
# fig, ax = plt.subplots(1)
#
# for name, path in paths.items():
#     ax.add_patch(PathPatch(path, edgecolor='k', facecolor='none'))
#
# plt.xlim(0, 300)
# plt.ylim(0, 300)
# plt.axis('equal')
#
# plt.show()
