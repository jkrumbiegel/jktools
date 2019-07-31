import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as pCircle, Arc as pArc, PathPatch, Wedge, Rectangle
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from matplotlib.bezier import concatenate_paths
# import jktools.plot as jplt


class P:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item == slice(None, None, None):
                return self.x, self.y
            else:
                return [self[i] for i in range(*item.indices(2))]
        elif item == 0:
            return self.x
        elif item == 1:
            return self.y
        else:
            Exception('You cannot index a point with anything but 0 or 1')

    def __iter__(self):
        return iter([self.x, self.y])

    def __add__(self, other):
        if isinstance(other, P):
            return P(self.x + other.x, self.y + other.y)
        else:
            return P(self.x + other, self.y + other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, P):
            return P(self.x - other.x, self.y - other.y)
        else:
            return P(self.x - other, self.y - other)

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        return P(self.x * other, self.y * other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return P(self.x / other, self.y / other)

    def __repr__(self):
        return f'|{self.x:.5f}, {self.y:.5f}|'

    def to(self, vector):
        return vector - self

    def distance_to(self, point):
        return self.to(point).magnitude()

    def magnitude(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self):
        return self / self.magnitude()

    def scale_to(self, length):
        return self.normalize() * length

    def xy(self):
        return self.x, self.y

    def plot(self, *args, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        ax.scatter(*self, *args, **kwargs)

    def rotate(self, angle, around=None, degrees=True):

        if around is None:
            around = P(0, 0)
        if degrees:
            angle = np.deg2rad(angle)

        vector = around.to(self)
        rotated_vector = P(
            np.cos(angle) * vector.x - np.sin(angle) * vector.y,
            np.sin(angle) * vector.x + np.cos(angle) * vector.y)
        return rotated_vector + around

    def angle(self, degrees=True):
        radians = np.arctan2(self.y, self.x)
        return radians if not degrees else np.rad2deg(radians)

    def signed_angle_to(self, other, degrees=True):
        radians = np.arctan2(self.cross(other), self.dot(other))
        return radians if not degrees else np.rad2deg(radians)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def angle_to(self, other, degrees=True):
        radians = np.arccos(self.normalize().dot(other.normalize()))
        return radians if not degrees else np.rad2deg(radians)

    def cross(self, other):
        return self.x * other.y - self.y * other.x

    def mirror_y(self):
        return P(-self.x, self.y)

    def mirror_x(self):
        return P(self.x, -self.y)

    def mirror_o(self):
        return P(-self.x, -self.y)


class X(P):

    def __init__(self, x):
        super(X, self).__init__(x, 0)


class Y(P):

    def __init__(self, y):
        super(Y, self).__init__(0, y)


class Line:
    def __init__(self, frm: P, to: P):
        self.frm = frm
        self.to = to

    def __add__(self, p: P):
        return Line(self.frm + p, self.to + p)

    def __radd__(self, p: P):
        return Line(self.frm + p, self.to + p)

    def center(self):
        return self.fraction(0.5)

    def vector(self):
        return self.frm.to(self.to)

    def direction(self):
        return self.vector().normalize()

    def fraction(self, fraction):
        return fraction * self.vector() + self.frm

    def xs(self):
        return self.frm.x, self.to.x

    def ys(self):
        return self.frm.y, self.to.y

    def length(self):
        return self.vector().magnitude()

    def reversed(self):
        return Line(self.to, self.frm)

    def perpendicular(self, through):
        x1, x2 = self.xs()
        y1, y2 = self.ys()
        x3, y3 = through.xy()
        k = ((y2 - y1) * (x3 - x1) - (x2 - x1) * (y3 - y1)) / ((y2 - y1) ** 2 + (x2 - x1) ** 2)
        x4 = x3 - k * (y2 - y1)
        y4 = y3 + k * (x2 - x1)
        return P(x4, y4)

    def intersection(self, line):
        x1, x2 = self.xs()
        y1, y2 = self.ys()
        x3, x4 = line.xs()
        y3, y4 = line.ys()
        t = (((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) /
             ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)))
        # u = -(((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) /
        #     ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)))

        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        return P(px, py)

    def move_fraction(self, fraction):
        direction = self.direction()
        new_frm = self.frm + direction * fraction
        new_to = self.to + direction * fraction
        return Line(new_frm, new_to)

    def scale(self, scalar):
        vector = self.vector()
        movement = (scalar - 1) * vector / 2
        new_frm = self.frm - movement
        new_to = self.to + movement
        return Line(new_frm, new_to)

    def scale_to(self, length):
        scalar = length / self.length()
        return self.scale(scalar)

    def pad(self, padding):
        direction = self.direction()
        return Line(self.frm - padding * direction, self.to + padding * direction)

    def path(self):
        vertices = np.array([
            self.frm.xy(),
            self.to.xy(),
            # [0, 0]
        ])

        codes = np.array([
            Path.MOVETO,
            Path.LINETO,
        ])

        return Path(vertices, codes=codes)

    def plot(self, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        line = Line2D(self.xs(), self.ys(), **kwargs)
        ax.add_line(line)
        return line

    def rotate(self, angle, degrees=True, around=None):
        if around is None:
            around = self.center()
        new_frm = self.frm.rotate(angle, around=around, degrees=degrees)
        new_to = self.to.rotate(angle, around=around, degrees=degrees)
        return Line(new_frm, new_to)

    def angle(self, degrees=True):
        return self.vector().angle(degrees=degrees)


class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def area(self):
        return np.pi * (self.radius ** 2)

    def circumference(self):
        return 2 * np.pi * self.radius

    def closest_to(self, vector):
        return self.center.to(vector).normalize() * self.radius

    def plot(self, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        patch = pCircle(self.center, self.radius, **kwargs)
        ax.add_patch(patch)
        return patch

    def tangents(self, frm):
        center_to_from = self.center - frm

        a = np.arcsin(self.radius / center_to_from.magnitude())
        b = np.arctan2(center_to_from.y, center_to_from.x)

        t_1 = b - a
        tangent_1 = P(np.sin(t_1), -np.cos(t_1)) * self.radius + self.center
        t_2 = b + a
        tangent_2 = P(-np.sin(t_2), np.cos(t_2)) * self.radius + self.center

        return tangent_1, tangent_2

    def outer_tangents(self, circle):
        big_circle, small_circle = (self, circle) if self.radius > circle.radius else (circle, self)
        radius_difference = big_circle.radius - small_circle.radius
        gamma = -np.arctan2(big_circle.center.y - small_circle.center.y, big_circle.center.x - small_circle.center.x)

        beta = np.arcsin(radius_difference / small_circle.center.to(big_circle.center).magnitude())
        alpha1 = gamma - beta
        alpha2 = gamma + beta

        small_tangent_point_1 = small_circle.center + P(
            small_circle.radius * np.cos(np.pi / 2 - alpha1),
            small_circle.radius * np.sin(np.pi / 2 - alpha1),
        )

        big_tangent_point_1 = big_circle.center + P(
            big_circle.radius * np.cos(np.pi / 2 - alpha1),
            big_circle.radius * np.sin(np.pi / 2 - alpha1),
        )

        small_tangent_point_2 = small_circle.center + P(
            small_circle.radius * np.cos(-np.pi / 2 - alpha2),
            small_circle.radius * np.sin(-np.pi / 2 - alpha2),
        )

        big_tangent_point_2 = big_circle.center + P(
            big_circle.radius * np.cos(-np.pi / 2 - alpha2),
            big_circle.radius * np.sin(-np.pi / 2 - alpha2),
        )

        return Line(small_tangent_point_1, big_tangent_point_1), Line(small_tangent_point_2, big_tangent_point_2)

    def point_at_angle(self, angle, degrees=True):
        angle = np.deg2rad(angle) if degrees else angle
        return self.center + P(np.cos(angle), np.sin(angle)) * self.radius

    def lineintersection(self, line):
        # algorithm works for circle at (0, 0)
        frm_moved = line.frm - self.center
        to_moved = line.to - self.center
        dx, dy = frm_moved.to(to_moved)
        dr = np.sqrt(dx ** 2 + dy ** 2)
        D = frm_moved.cross(to_moved)
        sgn = lambda x: -1 if x < 0 else 1
        delta = (self.radius ** 2) * (dr ** 2) - (D ** 2)

        if np.isclose(delta, 0):
            # tangent
            x = D * dy / (dr ** 2)
            y = -D * dx / (dr ** 2)
            return P(x, y) + self.center
        elif delta < 0:
            return None
        else:
            xplusminus = sgn(dy) * dx * np.sqrt(delta)
            yplusminus = np.abs(dy) * np.sqrt(delta)

            x1 = (D * dy + xplusminus) / (dr ** 2)
            x2 = (D * dy - xplusminus) / (dr ** 2)
            y1 = (-D * dx + yplusminus) / (dr ** 2)
            y2 = (-D * dx - yplusminus) / (dr ** 2)
            return P(x1, y1) + self.center, P(x2, y2) + self.center


    @classmethod
    def through(cls, p1, p2, p3):
        l1 = Line(p1, p2)
        c1 = l1.fraction(1 / 2)
        l2 = Line(p2, p3)
        c2 = l2.fraction(1 / 2)

        dl1 = l1.vector()
        dl2 = l2.vector()
        cl1 = Line(c1, c1 + P(dl1.y, -(dl1.x)))
        cl2 = Line(c2, c2 + P(dl2.y, -(dl2.x)))

        center = cl1.intersection(cl2)
        radius = p1.to(center).magnitude()
        return cls(center, radius)


class Arc:
    def __init__(self, center, radius, start_angle, end_angle):
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle

    @classmethod
    def through(cls, p1, p2, p3):
        circle = Circle.through(p1, p2, p3)
        start_angle = circle.center.to(p1).angle()
        end_angle = circle.center.to(p3).angle()

        if p1.to(p3).signed_angle_to(p1.to(p2)) > 0:
            start_angle, end_angle = end_angle, start_angle
        return cls(circle.center, circle.radius, start_angle, end_angle)

    @classmethod
    def center_and_points(cls, center, p_start, p_end):
        start_angle = center.to(p_start).angle()
        end_angle = center.to(p_end).angle()
        radius = center.to(p_start).magnitude()
        if not np.isclose(radius, center.to(p_end).magnitude()):
            Exception('Start and end point do not lie on the same circular arc.')
        return cls(center, radius, start_angle, end_angle)

    def path(self):
        path = Path.arc(self.start_angle, self.end_angle)
        transform = Affine2D().scale(self.radius).translate(*self.center.xy())
        return path.transformed(transform)

    def patch(self, **kwargs):
        shorter_angle, larger_angle = (self.start_angle, self.end_angle) if self.end_angle >= self.start_angle else (self.end_angle, self.start_angle)
        return pArc(self.center.xy(), self.radius * 2, self.radius * 2, angle=0, theta1=shorter_angle,
                    theta2=larger_angle, **kwargs)

    def plot(self, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        patch = self.patch(**kwargs)
        ax.add_patch(patch)
        return patch

    def angle(self):
        return self.end_angle - self.start_angle

    def length(self):
        return np.abs(self.radius * 2 * np.pi / 360 * self.angle())

    def padlength(self, pad_start=0, pad_end=0):
        length = self.length()
        angle = self.angle()
        new_start_angle = self.start_angle + pad_start * -angle / length # -angle so positive pads are larger arcs
        new_end_angle = self.end_angle + pad_end * angle / length
        return Arc(self.center, self.radius, new_start_angle, new_end_angle)

    def fractionpoint(self, fraction):
        return self.center + X(self.radius).rotate(self.start_angle + (self.end_angle - self.start_angle) * fraction)

    def fractionangle(self, fraction):
        return self.start_angle + self.angle() * fraction

    def plotarrow(self, tiplength=3, tipwidth=None, arckwargs=None, tipkwargs=None, **kwargs):
        ax = kwargs.pop('ax', plt.gca())

        if tipwidth is None:
            tipwidth = tiplength

        endpoint_arc = self.fractionpoint(1)
        shorter_arc = self.padlength(pad_end=-tiplength)
        endpoint_shorter_arc = shorter_arc.fractionpoint(1)
        endpoint_tangent = X(1).rotate(shorter_arc.end_angle)
        right_flankpoint = endpoint_shorter_arc + endpoint_tangent * (tipwidth / 2)
        left_flankpoint = endpoint_shorter_arc - endpoint_tangent * (tipwidth / 2)

        tipvertices = np.array([
            endpoint_arc.xy(),
            left_flankpoint.xy(),
            right_flankpoint.xy(),
            [np.nan, np.nan]
        ])

        codes = np.array([
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY
        ], dtype=np.uint8)

        default_tipargs = {
            'edgecolor': 'none',
            'facecolor': 'k'
        }

        default_arcargs = {
            'edgecolor': 'k'
        }

        tipkwargs = tipkwargs if tipkwargs is not None else {}
        arckwargs = arckwargs if arckwargs is not None else {}

        tippath = Path(tipvertices, codes)
        tippatch = PathPatch(tippath, **{**default_tipargs, **tipkwargs})
        arcpatch = shorter_arc.plot(**{**default_arcargs, **arckwargs})
        ax.add_patch(tippatch)
        return arcpatch, tippatch

    def scale(self, scale):
        return Arc(self.center, self.radius * scale, self.start_angle, self.end_angle)

    def lineintersection(self, line):
        circle = Circle(self.center, self.radius)
        intersections = circle.lineintersection(line)

        larger_angle = self.start_angle if self.start_angle > self.end_angle else self.end_angle
        smaller_angle = self.start_angle if self.start_angle < self.end_angle else self.end_angle
        if intersections is None:
            return None
        elif isinstance(intersections, P):
            return intersections if smaller_angle <= intersections.angle() <= larger_angle else None
        else:
            intersections_in_arc = [i if smaller_angle <= i.angle() <= larger_angle else None for i in intersections]
            only_valid = tuple(filter(lambda i: i is not None, intersections_in_arc))
            if len(only_valid) == 2:
                return only_valid
            elif len(only_valid) == 1:
                return only_valid[0]
            else:
                return None



    # def rotate(self, angle, around=P(0, 0), degrees=True):
    #     angle = angle if not degrees else np.deg2rad(angle)


class Rect:
    def __init__(self, center: P, width, height, angle=0, degrees=True, roundradius=None):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.degrees = degrees
        self.roundradius = roundradius

    def path(self):
        if self.roundradius is None:
            vertices = np.array([
                self.upperleft().xy(),
                self.lowerleft().xy(),
                self.lowerright().xy(),
                self.upperright().xy(),
                [0, 0]
            ])
            codes = np.array([
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY], dtype=np.uint8)
        else:
            up_radius = Y(1).rotate(self.angle, degrees=self.degrees) * self.roundradius
            right_radius = X(1).rotate(self.angle, degrees=self.degrees) * self.roundradius

            topline = self.topline().pad(-self.roundradius)
            bottomline = self.bottomline().pad(-self.roundradius)
            leftline = self.leftline().pad(-self.roundradius)
            rightline = self.rightline().pad(-self.roundradius)

            upperleftarc = Arc.center_and_points(self.upperleft() - up_radius + right_radius, topline.frm, leftline.to)
            lowerleftarc = Arc.center_and_points(self.lowerleft() + up_radius + right_radius, leftline.frm, bottomline.frm)
            lowerrightarc = Arc.center_and_points(self.lowerright() + up_radius - right_radius, bottomline.to, rightline.frm)
            upperrightarc = Arc.center_and_points(self.upperright() - up_radius - right_radius, rightline.to, topline.to)

            combined_path = concatenate_paths_without_redundant_movetos(
                topline.reversed().path(),
                upperleftarc.path(),
                leftline.reversed().path(),
                lowerleftarc.path(),
                bottomline.path(),
                lowerrightarc.path(),
                rightline.path(),
                upperrightarc.path()
            )

            return combined_path

        return Path(vertices, codes)

    def patch(self, **kwargs):
        return PathPatch(self.path(), **kwargs)

    def plot(self, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        ax.add_patch(self.patch(**kwargs))

    def lowerleft(self):
        return self.center + P(-self.width, -self.height).rotate(self.angle, degrees=self.degrees) / 2

    def upperleft(self):
        return self.center + P(-self.width, self.height).rotate(self.angle, degrees=self.degrees) / 2

    def lowerright(self):
        return self.center + P(self.width, -self.height).rotate(self.angle, degrees=self.degrees) / 2

    def upperright(self):
        return self.center + P(self.width, self.height).rotate(self.angle, degrees=self.degrees) / 2

    def topline(self):
        return Line(self.upperleft(), self.upperright())

    def bottomline(self):
        return Line(self.lowerleft(), self.lowerright())

    def rightline(self):
        return Line(self.lowerright(), self.upperright())

    def leftline(self):
        return Line(self.lowerleft(), self.upperleft())

    def normpoint(self, point: P):
        x, y = point
        new_x = x * self.width / 2
        new_y = y * self.height / 2
        return P(new_x, new_y).rotate(self.angle)

    # @classmethod
    # def around(cls, *points):

def concatenate_paths_without_redundant_movetos(*paths):
    return remove_redundant_movetos(concatenate_paths(paths))


def remove_redundant_movetos(path):
    movetos = path.codes == Path.MOVETO
    # remove only movetos that follow after the same coordinate
    same_coords = np.concatenate(([False], np.all(np.isclose(np.diff(path.vertices, axis=0), 0), axis=1)))
    movetos[0] = False
    movetos = movetos & same_coords

    new_vertices = path.vertices[~movetos, :]
    new_codes = path.codes[~movetos]

    return Path(new_vertices, codes=new_codes)
