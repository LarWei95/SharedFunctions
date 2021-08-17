'''
Created on 07.12.2020

@author: Lars
'''
import numpy as np
from shapely.geometry import Polygon
from collections import defaultdict
from pprint import pprint
from copy import deepcopy
from datetime import datetime as dt
import matplotlib.pyplot as plt
import datetime as dt
from abc import ABC, abstractmethod

class LinearAlgebra2D():
    @classmethod
    def rotate_vectors_direction_angle (cls, vectors, angle):
        angle = angle * (np.pi / 180)
        
        vec_shapes = vectors.shape
        shape_length = len(vec_shapes)
        
        if shape_length == 2:
            r = vectors[0]
            v = vectors[1]
        else:
            r = vectors[:,0]
            v = vectors[:,1]
        
        vectors = np.matmul(np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ]), v)
            
        if shape_length == 2:
            vectors = np.stack((r, vectors))
        else:
            vectors = np.concatenate([
                    r,
                    vectors
                ], axis=1)
            
        
        return vectors
    
    @classmethod
    def get_degrees_between_vectors (cls, v1, v2):
        v1_shape = v1.shape
        v2_shape = v2.shape
        
        if len(v1_shape) == 2 and len(v2_shape) == 2:
            v1 = v1[1] / np.linalg.norm(v1[1])
            v2 = v2[1] / np.linalg.norm(v2[1])
            
            degs = np.arccos(np.dot(v1, v2))
            
            return degs / (np.pi / 180)
        elif len(v1_shape) == 2 and len(v2_shape) == 3:
            v1 = v1[1] / np.linalg.norm(v1[1])
            v2 = v2[:,1] / np.repeat(np.linalg.norm(v2[:,1], axis=1)[:,np.newaxis], 2, axis=1)
            
            v1 = np.repeat([v1], len(v2), axis=0)
            
            degs = np.arccos(np.sum(v1 * v2, axis=1)) / (np.pi / 180)
            return degs
        elif len(v1_shape) == 3 and len(v2_shape) == 2:
            return cls.get_degrees_between_vectors(v2, v1)
        else:
            raise ValueError("Unaccepted shapes: {:s} {:s}".format(str(v1_shape), str(v2_shape)))
        
    @classmethod
    def get_signed_degrees_between_vectors (cls, v1, v2):
        v1_shape = v1.shape
        v2_shape = v2.shape
        
        if len(v1_shape) == 2 and len(v2_shape) == 2:
            v1 = v1[1] / np.linalg.norm(v1[1])
            v2 = v2[1] / np.linalg.norm(v2[1])
            
            degs = np.arccos(np.dot(v1, v2)) / (np.pi / 180)
            
            cb = np.cross(v1, v2)
            
            if cb < 0:
                degs = -degs
            
            return degs
        elif len(v1_shape) == 2 and len(v2_shape) == 3:
            v1 = v1[1] / np.linalg.norm(v1[1])
            v2 = v2[:,1] / np.repeat(np.linalg.norm(v2[:,1], axis=1)[:,np.newaxis], 2, axis=1)
            
            v1 = np.repeat([v1], len(v2), axis=0)
            
            # TODO Source for NaNs
            degs = np.sum(v1 * v2, axis=1)
            degs = np.round(degs, decimals=8)
            degs = np.arccos(degs) / (np.pi / 180)
            cb = np.cross(v1, v2)
            sel = cb < 0
            
            degs[sel] = -degs[sel] 
            
            return degs
        elif len(v1_shape) == 3 and len(v2_shape) == 2:
            return cls.get_signed_degrees_between_vectors(v2, v1)
        else:
            raise ValueError("Unaccepted shapes: {:s} {:s}".format(str(v1_shape), str(v2_shape)))
    
    @classmethod
    def convert_linegroups_to_graph (cls, linegroups, decimals=6):
        # linegroups: (Group, Lines, XY)
        # vectors: (Group, Lines, a+lamdba*b)
        vectors = LinearAlgebra2D.get_vectors_from_linepoints_groups(linegroups)
        # vectors: (Lines, a+lambda*b (2, 2))
        vectors = np.concatenate(vectors, axis=0)
        vectors_count = len(vectors)
        vectors_range = np.arange(vectors_count)
        
        points = []
        connections = []
        connections_set = set()
        point_dict = {}
        
        for main_line_index in vectors_range:
            main_line = vectors[main_line_index]
            # Lambda, Gamma (BOTH must be 0 < x < 1)
            coefs = LinearAlgebra2D.get_line_crossing_coefs_for_lines(main_line, vectors)            
            rounded_coefs = np.round(coefs, decimals)
            
            start_point = main_line[0]
            end_point = start_point + main_line[1]
            
            valids = np.logical_and(rounded_coefs >= 0, rounded_coefs <= 1)
            valids = np.logical_and(valids[:,0], valids[:,1])
            valids = np.sort(coefs[valids, 0])
            valids = np.repeat(valids[:,np.newaxis], 2, axis=1)
            
            a = np.repeat([main_line[0]], len(valids), axis=0)
            b = np.repeat([main_line[1]], len(valids), axis=0)
            
            middle_cuts = a + valids * b
            
            # ALL POINTS
            if len(middle_cuts) != 0:
                middle_cuts = np.insert(middle_cuts, 0, [start_point], axis=0)
                middle_cuts = np.append(middle_cuts, [end_point], axis=0)
            else:
                middle_cuts = np.array([start_point, end_point])
            
            middle_cuts = list(map(tuple, np.round(middle_cuts, decimals)))
            
            last_point = None
            
            for point in middle_cuts:
                if point not in point_dict:
                    points.append(point)
                    indx = len(points) - 1
                    point_dict[point] = indx
                else:
                    indx = point_dict[point]
                    
                if last_point is not None:
                    if last_point != indx:
                        tn = (last_point, indx)
                        tt = (indx, last_point)
                        
                        if tn not in connections_set and tt not in connections_set:
                            connections.append(tn)
                            connections_set.add(tn)
                
                last_point = indx
                
        points = np.array(points)
        connections = np.array(connections)
                
        return points, connections
        
    @classmethod
    def convert_vectors_to_linegroups (cls, vectors):
        vectors_shape = vectors.shape
        shape_length = len(vectors_shape)
        
        if shape_length == 2:
            return np.array([
                    vectors[0],
                    vectors[0] + vectors[1]
                ])
        else:
            vectors_count = vectors_shape[0]
            
            points = np.full((vectors_count, 2), 0, dtype=np.float)
            points[0] = vectors[0, 0]
            
            for i in range(vectors_count):
                v = vectors[i, 0] + vectors[i, 1]
                points[i+1] = v
                
            return points
            
            
            
        
        
        
    @classmethod
    def _shoelace_formula_base (cls, polygon):
        s = np.sum(polygon[:-1,0] * polygon[1:,1]) 
        s += polygon[-1, 0] * polygon[0, 1]
        s -= np.sum(polygon[1:, 0] * polygon[:-1, 1]) 
        s -= polygon[0, 0] * polygon[-1, 1]
        return s
        
                
    @classmethod
    def get_polygon_point_order (cls, polygon):
        '''
        Decides and returns the polygon point order.
        If it is clockwise, 1 is returned. If counter-
        clockwise, -1 is returned.
        '''
        # 1 -> clockwise
        # -1 -> counter-clockwise
        if isinstance(polygon, Polygon):
            return cls.get_polygon_point_order(
                    cls.polygon_to_points(polygon)
                )
        else:
            s = cls._shoelace_formula_base(polygon)
            return -np.sign(s)
            
    @classmethod
    def get_normal_vector (cls, dirvec):
        dimcount = len(dirvec.shape)
        
        if dimcount == 1:
            x = dirvec[0]
            y = dirvec[1]            
            
            if y != 0:
                # Ay > By -> By - Ay < 0
                ny = 0.5 * np.sign(x)
                
                if x != 0:
                    nx = - y * ny / x
                else:
                    ny = 0
                    nx = -0.5 * np.sign(y)
            else:
                # Ay == By
                nx = 0
                
                ny = np.sign(x)
                
            point = np.array([nx, ny])
            point /= np.sqrt(nx ** 2 + ny ** 2)
                    
            return point
        else:
            x = dirvec[:,0]
            y = dirvec[:,1]
            
            normal_vectors = np.full((len(dirvec), 2), np.nan, dtype=np.float)
            xnotzero = x != 0
            ynotzero = y != 0
            
            # y != 0 (MAIN)
            # x != 0
            sel = ynotzero & xnotzero
            
            normal_vectors[sel, 1] = 0.5 * np.sign(x[sel])
            normal_vectors[sel, 0] = -y[sel] * normal_vectors[sel, 1] / x[sel]
            
            # x == 0
            sel = ynotzero & ~xnotzero
            normal_vectors[sel, 1] = 0
            normal_vectors[sel, 0] = -0.5 * np.sign(y[sel])
            
            # y == 0 (MAIN)
            sel = ~ynotzero
            normal_vectors[sel, 0] = 0
            normal_vectors[sel, 1] = np.sign(x[sel])
            
            sel = normal_vectors[:,0] ** 2 + normal_vectors[:,1] ** 2
            sel = np.sqrt(sel)
            sel = np.repeat(sel[:,np.newaxis], 2, axis=1)
            
            normal_vectors /= sel
            
            return normal_vectors

    @classmethod
    def get_normal_vector_line_family (cls, line_vector):
        '''
        Finds the normal vector for the given line vector
        and returns a line family with the normal vector
        attached at the end. Line vector has the shape
        of (2, 2) with the root and direction vector.
        The returned numpy array has the shape (3, 2)
        with the same root and direction vector, but with
        the normal vector attached.
        '''
        multi = len(line_vector.shape) == 3
        
        if multi:
            normal_vector = cls.get_normal_vector(line_vector[:,1])
        else:
            normal_vector = cls.get_normal_vector(line_vector[1])
        
        if multi:
            return np.append(line_vector, normal_vector[:,np.newaxis], axis=1)
        else:
            return np.concatenate([line_vector, [normal_vector]], axis=0)
            
            
        
    @classmethod
    def polygon_to_points (cls, polygon):
        return np.concatenate([
                np.array(polygon.exterior.coords.xy[0])[:,np.newaxis],
                np.array(polygon.exterior.coords.xy[1])[:,np.newaxis]
            ], axis=1)
    
    @classmethod
    def polygons_to_points (cls, polygons):
        return [
                cls.polygon_to_points(x)
                for x in polygons
            ]
    
    @classmethod
    def get_vectors_from_linepoints (cls, linepoints):
        '''
        Returns line vectors for the given line points.
        The given numpy array of shape (X, 2) only contains
        points. The returned numpy array has the shape
        (X, 2, 2). The first dimension is the count of line
        vectors. The second contains the root vector (0) and
        the direction vector (1).
        '''
        return np.array([
                    [
                        [linepoints[i, 0],  linepoints[i, 1]], 
                        [
                            linepoints[i+1, 0] - linepoints[i, 0], 
                            linepoints[i+1, 1] - linepoints[i, 1]
                        ]
                    ]
                    for i in range(0, len(linepoints) - 1)
                 ])
    
    @classmethod
    def get_vectors_from_linepoints_groups (cls, linepoints_groups):
        '''
        Returns line vectors for each given list of line points.
        The given list of numpy arrays of shape (X, 2) only contains
        points. The returned numpy arrays have the shape (X, 2, 2).
        The first dimension is the count of line vectors. The second
        contains the root vector (0) and the direction vector (1).
        '''        
        linepoints = [
                cls.get_vectors_from_linepoints(x)
                for x in linepoints_groups
            ]
        return linepoints
    
    @classmethod
    def get_minimum_line_distance (cls, vec1, vec2, decimals=9):
        vec1 = np.round(vec1, decimals=decimals)
        vec2 = np.round(vec2, decimals=decimals)        
        
        cross_coefs = np.array([
                cls.get_line_crossing_coefs_for_lines(x, vec2)
                for x in vec1
            ])
        sel = (cross_coefs[:,:,0] >= 0) & (cross_coefs[:,:,0] < 1)
        sel &= (cross_coefs[:,:,1] >= 0) & (cross_coefs[:,:,1] < 1)
        
        if np.any(sel):
            return 0.0
        else:
            line1_point2_coefs = np.array([
                    cls.get_line_family_coefficients_for_point(x, vec2[:,0])
                    for x in vec1
                ])
            line1_point2_coefs_rnd = np.round(line1_point2_coefs, decimals=decimals)
            sel = (line1_point2_coefs_rnd[:,:,0] >= 0) & (line1_point2_coefs_rnd[:,:,0] <= 1)
            sel = line1_point2_coefs[sel, 1]
            dist1 = np.min(np.abs(sel)) if len(sel) != 0 else np.inf
            
            line2_point1_coefs = np.array([
                    cls.get_line_family_coefficients_for_point(x, vec1[:,0])
                    for x in vec2
                ])
            line2_point1_coefs_rnd = np.round(line2_point1_coefs, decimals=decimals)
            sel = (line2_point1_coefs_rnd[:,:,0] >= 0) & (line2_point1_coefs_rnd[:,:,0] <= 1)
            sel = line2_point1_coefs[sel, 1]
            dist2 = np.min(np.abs(sel)) if len(sel) != 0 else np.inf
            return np.min([dist1, dist2])
    
    @classmethod
    def get_line_crossing_coefs (cls, lines1, lines2):
        # lines1 / lines2: np.array!! (X, 4)
        lines1_count = len(lines1)
        lines2_count = len(lines2)
        
        coefs = np.full((lines1_count, lines2_count, 2), np.nan, dtype=float)
        
        for i in range(lines1_count):
            # [x0, y0, x1, y1]
            line1 = lines1[i]
            a1 = line1[0]
            a2 = line1[1]
            b1 = line1[2]
            b2 = line1[3]
            
            nonparallels = np.logical_not(np.equal(b1 * lines2[:,3], b2 * lines2[:,2]))
            nonparallels = np.arange(lines2_count)[nonparallels]
            
            for j in nonparallels:
                line2 = lines2[j]
                c1 = line2[0]
                c2 = line2[1]
                d1 = line2[2]
                d2 = line2[3]
                
                params_set = True
                
                if b1 != 0:
                    gam = (b1 * (c2 - a2) - b2 * (c1 - a1)) / (b2 * d1 - b1 * d2)
                    lam = (c1 - a1 + gam * d1) / b1
                elif d1 != 0:
                    lam = (d1 * (c2 - a2) - d2 * (c1 - a1)) / (b2 * d1 - b1 * d2)
                    gam = -(c1 - a1 - lam * b1) / d1
                elif b2 != 0:
                    gam = (b2 * (c1 - a1) - b1 * (c2 - a2)) / (b1 * d2 - b2 * d1)
                    lam = (c2 - a2 + gam * d2) / b2
                elif d2 != 0:
                    lam = (d2 * (c1 - a1) - d1 * (c2 - a2)) / (b1 * d2 - b2 * d1)
                    gam = -(c2 - a2 - lam * b2) / d2
                else:
                    params_set = False
                    
                if params_set:
                    coefs[i, j, 0] = lam
                    coefs[i, j, 1] = gam
                    
        return coefs
            
    @classmethod
    def get_line_crossing_coefs_for_points (cls, line, points):
        # One line, multiple points
        
        coefs = np.full(len(points), np.nan, dtype=np.float)
        
        if line[1, 0] != 0:
            lambdas = (points[:,0] - line[0, 0]) / line[1, 0]
            fits = np.round(line[0, 1] + lambdas * line[1, 1], 6) == np.round(points[:,1], 6)
        elif line[1, 1] != 0:
            lambdas = (points[:,1] - line[0, 1]) / line[1, 1]
            fits = np.round(line[0, 0] + lambdas * line[1, 0], 6) == np.round(points[:,0], 6)
        else:
            fits = np.full(len(points), False)
            
        # print(fits)
            
        if np.any(fits):
            coefs[fits] = lambdas[fits]
            
        return coefs
        
        
                
    @classmethod
    def get_line_crossing_coefs_for_lines (cls, line1, lines2):
        # lines1 / lines2: np.array!! (X, 4)
        
        # [x0, y0, x1, y1]
        a1 = line1[0, 0]
        a2 = line1[0, 1]
        b1 = line1[1, 0]
        b2 = line1[1, 1]
        
        c1 = lines2[:,0, 0]
        c2 = lines2[:,0, 1]
        d1 = lines2[:,1, 0]
        d2 = lines2[:,1, 1]
        
        nonp = np.logical_not(np.equal(b1 * d2, b2 * d1))
        
        coefs = np.full((len(lines2), 2), np.nan, dtype=float)
        todo = nonp
        
        b1nzero = b1 != 0
        b2nzero = b2 != 0
        
        if b1nzero or b2nzero:
            # d1 != 0
            sel = np.logical_and(np.not_equal(d1, 0), todo)
            
            coefs[sel,0] = (d1[sel] * (c2[sel] - a2) - d2[sel] * (c1[sel] - a1)) / (b2 * d1[sel] - b1 * d2[sel])
            coefs[sel,1] = -(c1[sel] - a1 - coefs[sel,0] * b1) / d1[sel]
            
            todo = np.logical_and(todo, np.logical_not(sel))
            
            # d2 != 0
            sel = np.logical_and(np.not_equal(d2, 0), todo)
            
            coefs[sel,0] = (d2[sel] * (c1[sel] - a1) - d1[sel] * (c2[sel] - a2)) / (b1 * d2[sel] - b2 * d1[sel])
            coefs[sel,1] = -(c2[sel] - a2 - coefs[sel,0] * b2) / d2[sel]
            
            todo = np.logical_and(todo, np.logical_not(sel))
            
            if np.any(todo):
                if b1nzero:
                    coefs[todo,1] = (b1 * (c2[todo] - a2) - b2 * (c1[todo] - a1)) / (b2 * d1[todo] - b1 * d2[todo])
                    coefs[todo,0] = (c1[todo] - a1 + coefs[todo,1] * d1[todo]) / b1
                elif b2nzero:
                    coefs[todo,1] = (b2 * (c1 - a1) - b1 * (c2 - a2)) / (b1 * d2 - b2 * d1)
                    coefs[todo,0] = (c2 - a2 + coefs[todo,1] * d2) / b2
                    
        return coefs
    
    @classmethod
    def get_line_crossing_coefs_for_linegroups (cls, line1, linegroups2):
        coefs = (
                cls.get_line_crossing_coefs_for_lines(line1, lines2)
                for lines2 in linegroups2
            )
        return coefs
    
    @classmethod
    def get_line_family_coefficients_for_point (cls, lf, point):
        # lf: (3, 2)
        # point: (2, )
        if len(lf.shape) == 2:
            b = lf[1]
            c = lf[2]
            
            prd1 = b[0] * c[1]
            prd2 = b[1] * c[0]
            
            rdiff = point - lf[0]
            
            if len(point.shape) == 1:
                if prd1 != prd2:
                    # b1 != 0
                    if b[0] != 0:
                        gam = (b[0] * rdiff[1] - b[1] * rdiff[0]) / (prd1 - prd2)
                        lam = (rdiff[0] - gam * c[0]) / b[0]
                    # c1 != 0
                    elif c[0] != 0:
                        lam = (c[0] * rdiff[1] - c[1] * rdiff[0]) / (prd2 - prd1)
                        gam = (rdiff[0] - lam * b[0]) / c[0]
                    # b2 != 0
                    elif b[1] != 0:
                        gam = (b[1] * rdiff[0] - b[0] * rdiff[1]) / (prd2 - prd1)
                        lam = (rdiff[1] - gam * c[1]) / b[1]
                    # c2 != 0
                    else:
                        lam = (c[1] * rdiff[0] - c[0] * rdiff[1]) / (prd1 - prd2)
                        gam = (rdiff[1] - lam * b[1]) / c[1]
                else:
                    lam = np.nan
                    gam = np.nan
                    
                coefs = np.array([lam, gam])
            else:
                if prd1 != prd2:
                    # b1 != 0
                    if b[0] != 0:
                        gam = (b[0] * rdiff[:,1] - b[1] * rdiff[:,0]) / (prd1 - prd2)
                        lam = (rdiff[:,0] - gam * c[0]) / b[0]
                    # c1 != 0
                    elif c[0] != 0:
                        lam = (c[0] * rdiff[:,1] - c[1] * rdiff[:,0]) / (prd2 - prd1)
                        gam = (rdiff[:,0] - lam * b[0]) / c[0]
                    # b2 != 0
                    elif b[1] != 0:
                        gam = (b[1] * rdiff[:,0] - b[0] * rdiff[:,1]) / (prd2 - prd1)
                        lam = (rdiff[:,1] - gam * c[1]) / b[1]
                    # c2 != 0
                    else:
                        lam = (c[1] * rdiff[:,0] - c[0] * rdiff[:,1]) / (prd1 - prd2)
                        gam = (rdiff[:,1] - lam * b[1]) / c[1]
                        
                    coefs = np.concatenate([
                            lam[:,np.newaxis],
                            gam[:,np.newaxis]
                        ], axis=1)
                else:
                    coefs = np.full((len(point), 2), np.nan)
        else:
            if len(point.shape) == 1:
                b = lf[:,1]
                c = lf[:,2]
                
                prd1 = b[:,0] * c[:,1]
                prd2 = b[:,1] * c[:,0]
                
                prd1m2 = prd1 - prd2
                prd2m1 = prd2 - prd1
                
                nonparallel_selector = prd1 != prd2
                bx_nonzero = b[:,0] != 0
                cx_nonzero = c[:,0] != 0
                by_nonzero = b[:,1] != 0
                rdiff = point - lf[:,0]
                
                coefs = np.full((len(lf), 2), np.nan, dtype=np.float)
                
                sel = nonparallel_selector & bx_nonzero
                coefs[sel, 1] = (b[sel, 0] * rdiff[sel, 1] - b[sel, 1] * rdiff[sel, 0]) / (prd1m2[sel])
                coefs[sel, 0] = (rdiff[sel, 0] - coefs[sel, 1] * c[sel, 0]) / b[sel, 0]
                
                sel = nonparallel_selector & cx_nonzero
                coefs[sel, 0] = (c[sel, 0] * rdiff[sel, 1] - c[sel, 1] * rdiff[sel, 0]) / (prd2m1[sel])
                coefs[sel, 1] = (rdiff[sel, 0] - coefs[sel, 0] * b[sel, 0]) / c[sel, 0]
                
                sel = nonparallel_selector & by_nonzero
                coefs[sel, 1] = (b[sel, 1] * rdiff[sel, 0] - b[sel, 0] * rdiff[sel, 1]) / (prd2m1[sel])
                coefs[sel, 0] = (rdiff[sel, 1] - coefs[sel, 1] * c[sel, 1]) / b[sel, 1]
                
                sel = nonparallel_selector & (~(bx_nonzero | cx_nonzero | by_nonzero))
                coefs[sel, 0] = (c[sel, 1] * rdiff[sel, 0] - c[sel, 0] * rdiff[sel, 1]) / (prd1m2[sel])                
                coefs[sel, 1] = (rdiff[sel, 1] - coefs[sel, 0] * b[sel, 1]) / c[sel, 1]
            else:
                lfcount = len(lf)
                pointcount = len(point)
                
                lf = np.repeat(lf, pointcount, axis=0)
                point = np.tile(point, (lfcount, 1))
                
                b = lf[:,1]
                c = lf[:,2]
                
                prd1 = b[:,0] * c[:,1]
                prd2 = b[:,1] * c[:,0]
                
                prd1m2 = prd1 - prd2
                prd2m1 = prd2 - prd1
                
                nonparallel_selector = prd1 != prd2
                bx_nonzero = b[:,0] != 0
                cx_nonzero = c[:,0] != 0
                by_nonzero = b[:,1] != 0
                rdiff = point - lf[:,0]
                
                coefs = np.full((len(lf), 2), np.nan, dtype=np.float)
                
                sel = nonparallel_selector & bx_nonzero
                
                if np.any(sel):                    
                    coefs[sel, 1] = (b[sel, 0] * rdiff[sel, 1] - b[sel, 1] * rdiff[sel, 0]) / (prd1m2[sel])
                    coefs[sel, 0] = (rdiff[sel, 0] - coefs[sel, 1] * c[sel, 0]) / b[sel, 0]
                
                sel = nonparallel_selector & cx_nonzero
                
                if np.any(sel):
                    coefs[sel, 0] = (c[sel, 0] * rdiff[sel, 1] - c[sel, 1] * rdiff[sel, 0]) / (prd2m1[sel])
                    coefs[sel, 1] = (rdiff[sel, 0] - coefs[sel, 0] * b[sel, 0]) / c[sel, 0]
                
                sel = nonparallel_selector & by_nonzero
                
                if np.any(sel):
                    coefs[sel, 1] = (b[sel, 1] * rdiff[sel, 0] - b[sel, 0] * rdiff[sel, 1]) / (prd2m1[sel])
                    coefs[sel, 0] = (rdiff[sel, 1] - coefs[sel, 1] * c[sel, 1]) / b[sel, 1]
                
                sel = nonparallel_selector & (~(bx_nonzero | cx_nonzero | by_nonzero))
                
                if np.any(sel):
                    coefs[sel, 0] = (c[sel, 1] * rdiff[sel, 0] - c[sel, 0] * rdiff[sel, 1]) / (prd1m2[sel])                
                    coefs[sel, 1] = (rdiff[sel, 1] - coefs[sel, 0] * b[sel, 1]) / c[sel, 1]
                
                coefs = np.reshape(coefs, (lfcount, pointcount, 2))
                
        return coefs
    
    @classmethod
    def get_corner_mean_normal_vectors (cls, polygon, return_order=False):
        if isinstance(polygon, Polygon):
            return cls.get_corner_mean_normal_vectors(
                    cls.polygon_to_points(polygon), return_order=return_order
                )
        else:
            order = LinearAlgebra2D.get_polygon_point_order(polygon[:-1])
            
            lines = LinearAlgebra2D.get_vectors_from_linepoints(polygon)
            lines = LinearAlgebra2D.get_normal_vector_line_family(lines)
            line_count = len(lines)
            
            lines = np.array([
                    [
                        lines[i, 0],
                        (lines[(i - 1) % line_count, 2] + lines[i, 2]) / 2
                    ]
                    for i in range(0, line_count)
                ])
            
            lengths = np.sqrt(lines[:,1, 0] ** 2 + lines[:,1, 1] ** 2)
            lines[:,1] /= np.repeat(lengths[:,np.newaxis], 2, axis=1)
            
            if return_order is False:
                return lines
            else:
                return lines, order
            
    @classmethod
    def resize_polygon_vectors (cls, vectors, order, size_change):
        fitted_vectors = vectors[:,0] + vectors[:,2] * size_change
        fitted_vectors = np.concatenate([
                fitted_vectors[:,np.newaxis],
                vectors[:,1][:,np.newaxis]
            ], axis=1)
        return fitted_vectors
    
    @classmethod
    def _create_new_linegroup_for_resize (cls, fitted_vectors):
        coefs = [
                cls.get_line_crossing_coefs_for_lines(vec, fitted_vectors)
                for vec in fitted_vectors
            ]
        
        vector_count = len(fitted_vectors)
        
        new_points = []
        
        for i in range(vector_count):#
            mod_i = i - 1
            
            v = fitted_vectors[mod_i]
            ni = (mod_i + 1) % vector_count
            coef = coefs[mod_i][ni, 0]
            
            p = v[0] + coef * v[1]
            new_points.append(p)
            
        new_points.append(new_points[0])
        new_points = np.array(new_points)
        return new_points
        
    @classmethod
    def _create_new_resized_polygons (cls, vectors,
                                      new_vectors,
                                      points, size_change):
        if np.any(np.isnan(vectors)) | np.any(np.isnan(new_vectors)) | np.any(np.isnan(points)):
            return []
        
        if size_change == 0:
            return [points]
        
        vectors = np.round(vectors, decimals=10)
        new_vectors = np.round(new_vectors, decimals=10)
        
        coefs = np.array([
                cls.get_line_crossing_coefs_for_lines(vec, new_vectors)
                for vec in vectors
            ])
        coefs = (coefs >= 0) & (coefs < 1)
        coefs = coefs[:,:,0] & coefs[:,:,1]
        
        if np.any(coefs):
        # if False:
            return []
        else:
            if size_change == -1:
                bigger_poly = Polygon(vectors[:,0])
                smaller_poly = Polygon(points)
                
                if smaller_poly.within(bigger_poly):
                    return [points]
                else:
                    return []
                
            elif size_change == 1:
                bigger_poly = Polygon(points)
                smaller_poly = Polygon(vectors[:,0])
                
                if smaller_poly.within(bigger_poly):
                    return [points]
                else:
                    return []
            else:
                return [points]
            
    @classmethod
    def _create_new_resized_polygons_v2 (cls, vectors,
                                      new_vectors,
                                      points, size_change):
        
        size_change_sn = np.sign(size_change)
        
        if size_change_sn == -1:
            if Polygon(points).within(Polygon(vectors[:,0])) == False:
                return []
            
            new_coefs = np.array([
                    cls.get_line_crossing_coefs_for_lines(new_vec, new_vectors)
                    for new_vec in new_vectors
                ])
            new_coefs = (new_coefs >= 0) & (new_coefs < 1)
            new_coefs = new_coefs[:,:,0] & new_coefs[:,:,1]
            
            if np.any(new_coefs):
                filtered_polys = cls.get_inner_polygons_of_linegroups([points])
            else:
                filtered_polys = [points]
        else:
            filtered_polys = [points]
        
        new_filtered_polys = filtered_polys
        '''
        new_filtered_polys = []
        
        for filtered_poly in filtered_polys:
            poly_vecs = cls.get_normal_vector_line_family(cls.get_vectors_from_linepoints(filtered_poly))
            
            dists = np.round(cls.get_minimum_line_distance(vectors, poly_vecs), decimals=3)
            
            print(dists)
            
            if np.min(np.abs(dists)) >= np.abs(size_change):
                new_filtered_polys.append(filtered_poly)
        '''
            
            
            
        return new_filtered_polys
        
    @classmethod
    def resize_polygon (cls, polygon, size_change):
        order = cls.get_polygon_point_order(polygon)
        
        if order == -1:
            polygon = np.flip(polygon, axis=0)
        
        
        vectors = cls.get_vectors_from_linepoints(polygon)
        vectors = cls.get_normal_vector_line_family(vectors)
        
        fitted_vectors = cls.resize_polygon_vectors(vectors, order, size_change)
        new_points = cls._create_new_linegroup_for_resize(fitted_vectors)
        
        new_vectors = cls.get_normal_vector_line_family(
                cls.get_vectors_from_linepoints(new_points)
            )
        
        # TODO Fuer Testzwecke
        # new_polys = cls._create_new_resized_polygons_v2(vectors, new_vectors, new_points, size_change)
        new_polys = [new_points]
        return new_polys
        
        
    
    @classmethod
    def offset_segment_vectors (cls, vectors, maxoffset):
        '''
        Linearly segments vectors and offsets them to the side.
        vectors: (X, 2, 2) Root + Direction
        maxoffset: Maximum sideways offset
        '''
        vectors = LinearAlgebra2D.get_normal_vector_line_family(vectors)
        return cls.offset_segment_normal_vector_line_families(vectors, maxoffset)
        
    
    @classmethod
    def offset_segment_normal_vector_line_families (cls, normal_line_families, maxoffset):
        '''
        Linearly segments normal line families and offsets them to the side.
        vectors: (X, 3, 2) Root + Direction + Normal line
        maxoffset: Maximum sideways offset
        '''
        segstep = maxoffset
        
        lengths = np.sqrt(np.sum(normal_line_families[:,1] ** 2, axis=1))
        segs = np.floor(lengths / segstep).astype(int)
        
        lineparams = [
                np.linspace(0, 1, seg, endpoint=False)[1:]
                if seg > 0
                else []
                for seg in segs
            ]
        normalparams = [
                np.random.rand(len(lineparam)) * maxoffset * 2 - maxoffset
                if len(lineparam) > 0
                else []
                for lineparam in lineparams
            ]
        
        newpoints = []
        
        for vector, lineparam, normalparam in zip(normal_line_families, lineparams, normalparams):
            newpoints.append(vector[0])
            
            for lparam, nparam in zip(lineparam, normalparam):
                newpoints.append(vector[0] + lparam * vector[1] + nparam * vector[2])
            
            newpoints.append(vector[0] + vector[1])
        
        newpoints = np.array(newpoints)
        return newpoints

    @classmethod
    def initialize_spline_interpolation_matrices_cubic (cls, linepoints):
        '''
        Initializes the cubic bezier curve system of linear equations
        for spline interpolation. The given line is treated as non-
        closing. Two matrices are returned.
        '''
        
        points_count = len(linepoints)
        matrix_cr = points_count - 1
        
        xs = np.zeros((matrix_cr, matrix_cr), dtype=np.float)
        ys = np.zeros((matrix_cr, 2), dtype=np.float)
        
        xs[0, [0, 1]] = np.array([2, 1])
        ys[0] = linepoints[0] + 2 * linepoints[1]
        
        cx = np.array([1, 4, 1])
        
        for i in range(1, matrix_cr - 1):
            xinds = np.arange(i-1, i+2)
            
            xs[i, xinds] = cx
            ys[i] = 4 * linepoints[i] + 2 * linepoints[i + 1]
        
        xs[matrix_cr - 1, [matrix_cr - 2, matrix_cr - 1]] = np.array([2, 7])
        ys[matrix_cr - 1] = 8 * linepoints[points_count - 2] + linepoints[points_count - 1]
        
        
        return xs, ys

    @classmethod
    def get_spline_interpolation_control_points_cubic (cls, linepoints):
        xs, ys = cls.initialize_spline_interpolation_matrices_cubic(linepoints)
        matrix_rows = len(xs)
        last_index = matrix_rows - 1
        
        for i in range(matrix_rows):
            c_index = i + 1
            b_c = xs[i, i]
            
            if i == 0:
                # Done
                xs[i, c_index] /= b_c
                ys[i] /= b_c
            else:
                a_c = xs[i, i - 1]
                c_p = xs[i - 1, i]
                d_p = ys[i - 1]
                d_c = ys[i]
                
                if i != last_index:
                    xs[i, c_index] /= b_c - c_p * a_c
                    ys[i] = (d_c - d_p * a_c) / (b_c - c_p * a_c)
                else:
                    ys[i] = (d_c - d_p * a_c) / (b_c - c_p * a_c)
               
        first_cpoints = np.empty((matrix_rows, 2), dtype=np.float)
        
        for i in range(last_index, -1, -1):
            if i == last_index:
                first_cpoints[i] = ys[i]
            else:
                c_index = i + 1
                
                first_cpoints[i] = ys[i] - xs[i, c_index] * first_cpoints[i + 1]
               
        second_cpoints = np.empty((matrix_rows, 2), dtype=np.float)
        
        for i in range(matrix_rows):
            if i == last_index:
                second_cpoints[i] = (linepoints[-1] + first_cpoints[-1]) / 2
            else:
                second_cpoints[i] = 2 * linepoints[i + 1] - first_cpoints[i + 1]
               
        all_points = np.concatenate([
                first_cpoints[:,np.newaxis],
                second_cpoints[:,np.newaxis]
            ], axis=1)
        
        return all_points
    
    @classmethod
    def bezier_curve_cubic (cls, points, step_count):
        p1 = points[0]
        p2 = points[1]
        p3 = points[2]
        p4 = points[3]
        
        steps = np.linspace(0, 1, step_count)
        
        new_line = []
        
        for step in steps:
            t1 = ((1 - step) ** 3) * p1
            t2 = 3 * (step - 2 * (step ** 2) + step ** 3) * p2
            t3 = 3 * (step ** 2 - step ** 3) * p3
            t4 = (step ** 3) * p4
            
            
            t = t1 + t2 + t3 + t4
            new_line.append(t)
            
        return np.array(new_line)    
    
    @classmethod
    def spline_interpolate_cubic (cls, linepoints, step_count):
        points = LinearAlgebra2D.get_spline_interpolation_control_points_cubic(linepoints)
        # bezier = LinearAlgebra2D.bezier_curve(line, 10)
        
        interpolated = []
        
        for i in range(len(points)):
            p1 = linepoints[i]
            p4 = linepoints[i + 1]
            p2 = points[i, 0]
            p3 = points[i, 1]
            
            l = np.array([p1, p2, p3, p4])
            l = LinearAlgebra2D.bezier_curve_cubic(l, step_count)
            interpolated.extend(l)
            
        return np.array(interpolated)
        
    @classmethod
    def linear_interpolate_linepoints (cls, linepoints, step_count):
        step_count = (np.max([0, step_count]) + 1).astype(int)
        coefs = np.repeat(np.linspace(0, 1, num=step_count, endpoint=False)[:,np.newaxis], 2, axis=1)
        
        vectors = cls.get_vectors_from_linepoints(linepoints)
        vectors_count = len(vectors)
        
        new_points = []
        
        for i in range(vectors_count):
            vector = vectors[i]
            root_vector = np.repeat(vector[0][np.newaxis], step_count, axis=0)
            
            new_dirs = np.repeat(vector[1][np.newaxis], step_count, axis=0) * coefs
            new_dirs = root_vector + new_dirs
            
            new_points.extend(new_dirs)
        
        new_points.append(vectors[-1, 0] + vectors[-1, 1])
        
        new_points = np.array(new_points)
        
        return new_points
    
    @classmethod
    def linear_interpolate_linepoints_equal_length (cls, linepoints, segment_length):        
        vectors = cls.get_vectors_from_linepoints(linepoints)
        vectors_count = len(vectors)
        vectors_length = np.sqrt(np.sum(vectors[:,1], axis=1) ** 2)
        
        new_points = []
        
        for i in range(vectors_count):
            vector = vectors[i]
            vector_length = vectors_length[i]
            coefs = np.arange(0, vector_length, segment_length) / vector_length
            coefs = np.repeat(coefs[:,np.newaxis], 2, axis=1)
            step_count = len(coefs)
            
            root_vector = np.repeat(vector[0][np.newaxis], step_count, axis=0)
            
            new_dirs = np.repeat(vector[1][np.newaxis], step_count, axis=0) * coefs
            new_dirs = root_vector + new_dirs
            
            new_points.extend(new_dirs)
        
        new_points.append(vectors[-1, 0] + vectors[-1, 1])
        
        new_points = np.array(new_points)
        
        return new_points
    
    @classmethod
    def get_inner_polygons_of_linegroups_old (cls, linegroups, substitute=True):
        points, graph = cls.convert_linegroups_to_graph(linegroups)
        graph_adjlist = GraphTheory.edges_to_adjacency_list(len(points), graph, symmetric=True)
    
        lengths = GraphTheory.length_of_pointed_adjacency_list(points, graph_adjlist)
        
        # SUBSTITUTE
        cycle_edges = GraphTheory.minimum_cycle_basis_polygons(graph_adjlist, lengths, points, substitute=substitute)
        # DESUBSTITUTION ON EDGES
        
        areas = [
                Polygon(points[x[:,0]])
                for x in cycle_edges
            ]
        
        return areas
    
    @classmethod
    def filter_minimal_polygons (cls, linegroups):
        polygons = [
                Polygon(x)
                for x in linegroups
            ]
        polygon_areas = [
                x.area
                for x in polygons
            ]
        
        area_sorted_indices = np.flip(np.argsort(polygon_areas))
        
        
        containment_dicts = defaultdict(set)
        
        for poly_i in area_sorted_indices:
            for poly_j in area_sorted_indices:
                if poly_i != poly_j:
                    overlaps = polygons[poly_i].overlaps(polygons[poly_j])
                    overlaps |= polygons[poly_i].within(polygons[poly_j])
                    
                    if overlaps:
                        containment_dicts[poly_i].add(poly_j)
            
        valid_indices = set(area_sorted_indices) - set(containment_dicts.keys())
        return [
                polygons[i]
                for i in valid_indices
            ]
        '''
        overlappings = np.array([
                np.any([
                    g1.overlaps(g2) | g1.within(g2)
                    if g1 != g2
                    else False                    
                    for g2 in polygons
                ])
                for g1 in polygons
            ])
        # print(overlappings)
        non_overlapping = np.arange(len(polygons))[np.logical_not(overlappings)]
        return [
                polygons[i]
                for i in non_overlapping
            ]
        '''
    
    
    @classmethod
    def _line_profile_coordinates(cls, src, dst, linewidth=1):
        """Return the coordinates of the profile of an image along a scan line.
        Parameters
        ----------
        src : 2-tuple of numeric scalar (float or int)
            The start point of the scan line.
        dst : 2-tuple of numeric scalar (float or int)
            The end point of the scan line.
        linewidth : int, optional
            Width of the scan, perpendicular to the line
        Returns
        -------
        coords : array, shape (2, N, C), float
            The coordinates of the profile along the scan line. The length of the
            profile is the ceil of the computed length of the scan line.
        Notes
        -----
        This is a utility method meant to be used internally by skimage functions.
        The destination point is included in the profile, in contrast to
        standard numpy indexing.
        """
        src_row, src_col = src = np.asarray(src, dtype=float)
        dst_row, dst_col = dst = np.asarray(dst, dtype=float)
        d_row, d_col = dst - src
        theta = np.arctan2(d_row, d_col)
    
        length = int(np.ceil(np.hypot(d_row, d_col) + 1))
        # we add one above because we include the last point in the profile
        # (in contrast to standard numpy indexing)
        line_col = np.linspace(src_col, dst_col, length)
        line_row = np.linspace(src_row, dst_row, length)
    
        # we subtract 1 from linewidth to change from pixel-counting
        # (make this line 3 pixels wide) to point distances (the
        # distance between pixel centers)
        col_width = (linewidth - 1) * np.sin(-theta) / 2
        row_width = (linewidth - 1) * np.cos(theta) / 2
        perp_rows = np.stack([np.linspace(row_i - row_width, row_i + row_width,
                                          linewidth) for row_i in line_row])
        perp_cols = np.stack([np.linspace(col_i - col_width, col_i + col_width,
                                          linewidth) for col_i in line_col])
        return np.stack([perp_rows, perp_cols])
    
    
    
    @classmethod
    def vector_to_image_indices (cls, vec, width):
        s = vec[0]
        e = s + vec[1]
        
        cx, cy = cls._line_profile_coordinates(tuple(s), tuple(e), width)
        cx = cx.flatten()
        cy = cy.flatten()
        
        cxc = np.ceil(cx).astype(int)
        cxf = np.floor(cx).astype(int)
        
        cyc = np.ceil(cy).astype(int)
        cyf = np.floor(cy).astype(int)
        
        indsx = np.concatenate([cxc, cxf, cxc, cxf])
        indsy = np.concatenate([cyc, cyf, cyf, cyc])
        return indsx, indsy
    
    @classmethod
    def vectors_to_image_indices (cls, vectors, width):
        indsx = []
        indsy = []
        
        for vec in vectors:
            sx, sy = cls.vector_to_image_indices(vec, width)
            indsx.append(sx)
            indsy.append(sy)
        
        indsx = np.concatenate(indsx)
        indsy = np.concatenate(indsy)
        return indsx, indsy
    
    @classmethod
    def vectors_to_mask_image (cls, vectors, width, image_shape):
        mask = np.full(image_shape, False, dtype=np.bool)
        
        for vector in vectors:
            indsx, indsy = cls.vector_to_image_indices(vector, width)
            
            indsx[indsx >= image_shape[1]] = image_shape[1] - 1
            indsy[indsy >= image_shape[0]] = image_shape[0] - 1
            mask[indsy, indsx] = True
            
        return mask
            
    @classmethod
    def box_linegroups (cls, root, width, height):
        vectors = np.array([
                [width, 0],
                [0, 0],
                [0, height],
                [width, height],
                [width, 0]
            ]) + root
        vectors = cls.get_vectors_from_linepoints(vectors)
        vectors = cls.get_normal_vector_line_family(vectors)
        return vectors
    
    @classmethod
    def polygon_polygon_reachability_mean (cls, polygons):
        # (Index i, Index j) -> Line
        reachabilities = {}
        # (Index i, Index j) -> (i Entry point, j Entry point)
        points = {}
        
        
        polygon_means = [
                np.mean(x, axis=0)
                for x in polygons
            ]
        
        polygons = [
                np.concatenate([
                        polygon,
                        [polygon[0]]
                    ], axis=0)
                for polygon in polygons
            ]
        
        polygon_vecfams = [
                cls.get_normal_vector_line_family(
                        cls.get_vectors_from_linepoints(x)
                    )
                for x in polygons
            ]
        
        polygon_indices = np.arange(len(polygons))
        
        for i, p1 in enumerate(polygons):
            p1_mean = polygon_means[i]
            p1_vecfams = polygon_vecfams[i]
            
            for j, p2 in enumerate(polygons):
                if i < j:
                    p2_mean = polygon_means[j]
                    p2_vecfams = polygon_vecfams[j]
                    
                    # IN BETWEEN STEP: FINDING RELEVANT POINTS OF POLYS
                    p1p2_line = np.array([
                            p1_mean,
                            p2_mean - p1_mean
                        ])
                    
                    p1_c = cls.get_line_crossing_coefs_for_lines(p1p2_line, p1_vecfams)
                    sel = (p1_c[:,1] >= 0) & (p1_c[:,1] <= 1)
                    p1_c = p1_c[sel]
                    p1_c = np.max(p1_c[:,0])
                    p1_c = p1p2_line[0] + p1_c * p1p2_line[1]
                    
                    p2_c = cls.get_line_crossing_coefs_for_lines(p1p2_line, p2_vecfams)
                    sel = (p2_c[:,1] >= 0) & (p2_c[:,1] <= 1)
                    p2_c = p2_c[sel]
                    p2_c = np.min(p2_c[:,0])
                    p2_c = p1p2_line[0] + p2_c * p1p2_line[1]
                    
                    
                    p1p2_line = np.array([
                            p1_c,
                            p2_c - p1_c
                        ])
                    
                    check_indices = np.delete(polygon_indices, [i, j])
                    
                    if len(check_indices) != 0:
                        # TODO ValueError: need at least one array to concatenate
                        checkables = np.concatenate([
                                polygon_vecfams[i]
                                for i in check_indices
                            ], axis=0)
                        
                        check_coefs = cls.get_line_crossing_coefs_for_lines(p1p2_line, checkables)
                        sel = (check_coefs < 0) | (check_coefs > 1)
                        sel = sel[:,0] | sel[:,1]
                        
                        if np.all(sel):
                            reachabilities[(i, j)] = p1p2_line
                            points[(i, j)] = (p1_c, p2_c)
                    else:
                        reachabilities[(i, j)] = p1p2_line
                        points[(i, j)] = (p1_c, p2_c)
                    
                    
        return reachabilities, points
    
    @classmethod
    def get_radius_between_points (cls, p1, p2, p1_direction=None):
        if p1_direction is None:
            median_point = np.mean([p1, p2], axis=1)
            return np.linalg.norm(median_point - p1)
        else:
            p1_normvec = LinearAlgebra2D.get_normal_vector_line_family(np.stack((p1, p1_direction)))[2]
            
            median_line = np.stack((p1, p1_normvec))
            
            x = median_line[0, 0]
            y = median_line[0, 1]
            d1 = median_line[1, 0]
            d2 = median_line[1, 1]
            p1 = p2[0]
            p2 = p2[1]
            
            div = 2 * (x * d1 - p1 * d1 + y * d2 - p2 * d2)
            
            if div != 0:
                lam = -((x ** 2) + (p1 ** 2) + (y ** 2) - 2*p2*y - 2 * p1 * x + (p2 ** 2))
                lam /= div
            else:
                lam = np.nan
                
            dirvec = median_line[1] * lam
    
            radius = np.linalg.norm(dirvec)
            return radius
        
    @classmethod
    def get_intersection_points_of_circles (cls, p1, r1, p2, r2):
        vecdiff = p2 - p1
        lengthdiff = np.linalg.norm(vecdiff)
        
        part1 = (p1 + p2) / 2
        part2 = ((r1 ** 2 - r2 ** 2) / (2 * lengthdiff ** 2)) * (p2 - p1)
        part3 = 2* ((r1 ** 2 + r2 ** 2) / (lengthdiff ** 2)) - (((r1 ** 2 - r2 ** 2) ** 2) / (lengthdiff ** 4)) - 1
        
        if part3 >= 0.0:
            part3 = np.sqrt(part3) / 2
            part4 = np.array([
                    p2[1] - p1[1],
                    p1[0] - p2[0]
                ])
            
            newpoint1 = part1 + part2 + part3 * part4
            newpoint2 = part1 + part2 - part3 * part4
            
            if np.all(newpoint1 == newpoint2):
                return np.array([newpoint1])
            else:
                return np.array([
                        newpoint1,
                        newpoint2
                    ])
        else:
            return np.array([])
    
    @classmethod
    def get_counterclockwise_neighbor_dict (cls, vectors):
        vectors_count = len(vectors)
        
        neighbor_dict = {}
        
        for i in range(vectors_count):
            angles = cls.get_signed_degrees_between_vectors(vectors[i], vectors)
            angles[i] = np.nan
            angles[angles < 0] = 360 + angles[angles < 0]
            minindex = np.nanargmin(angles)
            
            neighbor_dict[i] = minindex
            
        return neighbor_dict
    
    @classmethod
    def _get_crossing_neighbor_dict (cls, crossings, points, adjlist):
        neighbor_dicts = {}
        
        for crossing_index in crossings:
            neighbor_dict = {}
            
            crossing_point = points[crossing_index]
            adjs = np.array(list(adjlist[crossing_index]))
            vectors = np.array([
                    [
                        crossing_point,
                        points[adj_index] - crossing_point
                    ]
                    for adj_index in adjs
                ])
            neighbor_dict = LinearAlgebra2D.get_counterclockwise_neighbor_dict(vectors)
            neighbor_dict = {
                    adjs[x] : adjs[neighbor_dict[x]]
                    for x in neighbor_dict            
                }
            neighbor_dicts[crossing_index] = neighbor_dict
            
        return neighbor_dicts
    
    @classmethod
    def _walk (cls, crossing, start_point, adjlist, neighbor_dicts):    
        path = [crossing, start_point]
        visited = {crossing, start_point}
        
        walk_start = crossing
        
        last_point = crossing
        c_point = start_point
        
        while True:
            c_adjs_unfiltered_set = adjlist[c_point]
            c_adjs_set = adjlist[c_point] - visited
            
            if last_point != walk_start and walk_start in c_adjs_unfiltered_set:
                path.append(walk_start)
                break
            elif c_point not in neighbor_dicts:
                c_adjs = list(c_adjs_set)
                
                if len(c_adjs) == 1:
                    next_point = list(c_adjs)[0]
                else:
                    '''
                    Polygon inside polygon, connected with just
                    one line.
                    '''
                    path = None
                    return path
                    '''
                    raise ValueError("""The path has suddenly stopped in a filament, containing
                    no more visitable nodes.
                    Previous index: {:d}
                    Point index: {:d}
                    Unfiltered adjacents: {:s}
                    Filtered adjacents: {:s}
                    Created path: {:s}""".format(last_point, c_point, str(c_adjs_unfiltered_set),
                                                       str(c_adjs_set), str(path)))
                    '''
            else:
                next_point = neighbor_dicts[c_point][last_point]
            
            last_point = c_point
            path.append(next_point)
            visited.add(next_point)
            c_point = next_point
            
        return np.array(path)    
    
    @classmethod
    def _hash_path (cls, path):
        c = np.stack((
                path[:-1],
                path[1:]
            ), axis=1) ** 5
        c = c[:,0] * c[:,1]
        return np.sum(c) + 31 * len(c) ** 3
    
    @classmethod
    def _filter_polygons (cls, polygons):
        poly_count = len(polygons)
        
        poly_objs = [
                Polygon(x)
                for x in polygons
            ]
        polygon_valids = np.array([
                [
                    poly_objs[j].within(poly_objs[i])
                    if i != j
                    else False
                    for j in range(poly_count)
                ]
                for i in range(poly_count)
            ])
        polygon_valids = np.any(polygon_valids, axis=1)
        polygon_valids = ~polygon_valids
        
        polygons = [
                polygons[i]
                for i in range(poly_count)
                if polygon_valids[i]
            ]
        return polygons
                    
    @classmethod
    def _get_all_inner_polygons (cls, neighbor_dicts, points, adjlist, filter_polygons):
        polygons = []
        polygons_hash = set()
        
        for crossing in neighbor_dicts:
            for start_point in neighbor_dicts[crossing]:
                path = cls._walk(crossing, start_point, adjlist, neighbor_dicts)
                
                if path is not None:
                    
                    path_hash = cls._hash_path(path)
                    # print("Path:",path, path_hash)
                    if path_hash not in polygons_hash:
                        # print("Path has does not exist. Adding.")
                        path = points[path]
                        
                        order = LinearAlgebra2D.get_polygon_point_order(path)
                        
                        if order == -1:
                            path = np.flip(path, axis=0)
                        
                       
                        
                        polygons.append(path)
                        polygons_hash.add(path_hash)
        
        if filter_polygons:
            if len(polygons) != 0:
                polygons = cls._filter_polygons(polygons)
                
        return polygons
    
    @classmethod
    def _delete_dead_ends (cls, endpoints, adjlist):
        for endpoint in endpoints:
            c_point = endpoint
            
            while True:
                old_adj = adjlist[c_point]
                
                if len(old_adj) == 1:
                    adj = list(old_adj)[0]
                    
                    adjlist[c_point].clear()
                    adjlist[adj].discard(c_point)
                    c_point = adj
                else:
                    break
    
    @classmethod
    def get_inner_polygons_of_linegroups (cls, linegroups, filter_polygons=True):
        points, graph = LinearAlgebra2D.convert_linegroups_to_graph(linegroups)
        adjlist = GraphTheory.edges_to_adjacency_list(len(points), graph, symmetric=True)
        
        endpoints = {
                x
                for x in adjlist
                if len(adjlist[x]) == 1
            }
        cls._delete_dead_ends(endpoints, adjlist)
        '''
        TODO
        There can be no crossings, but still one polygon at max!
        '''
        crossings = {
                    x
                    for x in adjlist
                    if len(adjlist[x]) >= 3
                }
        
        neighbor_dicts = cls._get_crossing_neighbor_dict(crossings, points, adjlist)
        
        polygons = cls._get_all_inner_polygons(neighbor_dicts, points, adjlist, filter_polygons)
        return polygons
    
    @classmethod
    def get_basic_distance_corner_vectors_for_polygon (cls, normvecs):
        corner_vecs = []
        count = len(normvecs)
        
        for i in range(count):
            cur = normvecs[i]
            nex = normvecs[(i + 1) % count]
            
            a = cur[0]
            b = cur[1]
            c = cur[2]
            d = nex[1]
            e = nex[2]
            
            d1b2 = d[0] * b[1]
            d2b1 = d[1] * b[0]
            
            if d1b2 != d2b1:
                if d[0] != 0:
                    # !!! c[0] = -c[1] !!!
                    const = (2*b[0]*d[0]*(((c[0] - e[0]) * d[1] / d[0]) - (c[1] - e[1]))) / ((c[0] + e[0]) * (d1b2 - d2b1))
                    const += 2 * c[0] / (c[0] + e[0])
                else:
                    const = (2*b[1]*d[0]*(((c[1] - e[1]) * d[0] / d[1]) - (c[0] - e[0]))) / ((c[0] + e[0]) * (d2b1 - d1b2))
                    const += (2*d[0]*(c[1] - e[1])) / ((c[0] + e[0]) * d[1])
                    const += (2*e[0]) / (c[0] + e[0])
                
                newroot = a + b
                newdir = ((c + e) / 2) * const
            else:
                newroot = a + b
                newdir = e
            
            
            corner_vec = np.array([
                        newroot,
                        newdir
                    ])
            corner_vecs.append(corner_vec)
            
        corner_vecs = np.array(corner_vecs)
        return corner_vecs
        
    @classmethod
    def get_distance_corner_vectors_for_polygon (cls, normvecs):
        corner_vecs = cls.get_basic_distance_corner_vectors_for_polygon(normvecs)
        
        coefs = np.array([
                cls.get_line_crossing_coefs_for_lines(corner_vec, normvecs)
                for corner_vec in corner_vecs
            ])
        coefs_base_valid = (coefs[:,:,1] >= 0) & (coefs[:,:,1] < 1)
        coefs_max_valid = coefs_base_valid & (coefs[:,:,0] > 0)
        coefs_min_valid = coefs_base_valid & (coefs[:,:,0] < 0)
        
        # coefs_valid[indices, (indices + 1) % len(corner_vecs)] = False
        
        max_coefs = [
                x[valid][:,0]
                for valid, x in zip(coefs_max_valid, coefs)
            ]
        min_coefs = [
                x[valid][:,0]
                for valid, x in zip(coefs_min_valid, coefs)
            ]
        
        max_coefs = np.array([
                np.min(x)
                if len(x) != 0
                else np.inf
                for x in max_coefs
            ])
        min_coefs = np.array([
                np.max(x)
                if len(x) != 0
                else -np.inf
                for x in min_coefs
            ])
        
        coef_ranges = np.stack((min_coefs, max_coefs), axis=1)
        
        return corner_vecs, coef_ranges
    
    @classmethod
    def resize_polygon2 (cls, polygon, size_change):
        order = cls.get_polygon_point_order(polygon)
        
        if order == -1:
            polygon = np.flip(polygon, axis=0)
        
        
        vectors = cls.get_vectors_from_linepoints(polygon)
        vectors = cls.get_normal_vector_line_family(vectors)
        
        corner_vecs, coef_ranges = cls.get_distance_corner_vectors_for_polygon(vectors)
        
        coefs = np.repeat(size_change, len(corner_vecs))
        coefs = np.max([coefs, coef_ranges[:,0] / 2], axis=0)
        coefs = np.min([coefs, coef_ranges[:,1]], axis=0)
        
        coefs = np.repeat(coefs[:,np.newaxis], 2, axis=1)
        
        newpoints = corner_vecs[:,0] + coefs * corner_vecs[:,1]
        newpoints = np.concatenate([newpoints, [newpoints[0]]], axis=0)
        return [newpoints]
    
    @classmethod
    def points_on_vectors (cls, points, vectors):
        coefs = np.array([
                LinearAlgebra2D.get_line_crossing_coefs_for_points(l, points)
                for l in vectors
            ])
        coefs = np.transpose(coefs, (1, 0))
        # print("POINTS ON VECTORS COEFS:\n"+str(coefs))
        valids = [
                np.where((x > 0) & (x < 1))[0]
                for x in coefs
            ]
        
        return_value = []
        
        for coef, valid in zip(coefs, valids):
            perm = np.argsort(coef[valid])
            v = valid[perm]
        
            return_value.append(v)
            
        return return_value
    
class LinearEquationSystems ():
    @classmethod
    def get_coef_sorting_permutation (cls, xs):
        maxs = xs != 0
        maxs = np.argmax(maxs, axis=1)
        perm = np.argsort(maxs, axis=0)
        
        backperm = np.argsort(perm)
        
        return perm, backperm
    
    @classmethod
    def gaussian_elimination (cls, xs, ys):
        perm, reperm = cls.get_coef_sorting_permutation(xs)
        # NOT READY
        permxs = xs[perm]
        permys = ys[perm]
        
        count = len(permxs)
        lastind = count - 1
        
        for i in range(lastind):
            # i: Coef to set to 0
            cxs = permxs[i]
            cys = permys[i]
            
            if cxs[i] != 0:
                for j in range(i+1, count):
                    if permxs[j, i] != 0:
                        mult = permxs[j, i] / cxs[i]
                        
                        permxs[j] = np.round(permxs[j] - cxs * mult, 16)
                        permys[j] = np.round(permys[j] - cys * mult, 16)
            
        degree = xs.shape[1]
            
        coefs = np.full(degree, 0, dtype=np.float)
        coefs[degree-1] = permys[degree-1] / permxs[degree-1, degree-1] 
        
        for i in range(degree-2, -1, -1):
            cxs = permxs[i]
            cys = permys[i]
            
            found_coefs = coefs[i+1:]
            addables = cxs[i+1:]
            additions = np.sum(addables * found_coefs)
                
            new_y = cys - additions
            
            coef = new_y / cxs[i]
            coefs[i] = coef
        
        return coefs
    
    @classmethod    
    def create_polynomial_function (cls, x, y):
        degree = len(x)
        
        exponents = np.flip(np.arange(0, degree))
        
        liny = np.array(y, dtype=np.float)
        linx = np.array([
                i ** exponents
                for i in x
            ])
        
        coefs = cls.gaussian_elimination(linx, liny)
        return coefs
    
    @classmethod
    def inference_polynomial_function (cls, x, coefs):
        degree = len(coefs)
        
        exponents = np.flip(np.arange(0, degree))
        x = np.repeat(x[:,np.newaxis], degree, axis=1)
        
        x = x ** exponents
        y = coefs * x
        
        y = np.sum(y, axis=1)
        return y

class GraphTheory ():
    @classmethod
    def order_cycle_edges (cls, edges):
        point_sets = [
                set(x)
                for x in edges
            ]
        
        done = set([0])
        last_edge = edges[0]
        last_endpoint = last_edge[1]
        start_point = last_edge[0]
        
        new_edges = [last_edge]
        
        while True:
            next_edge_index = None
            
            for i, ps in enumerate(point_sets):
                if i in done:
                    continue
                else:
                    if last_endpoint in ps:
                        next_edge_index = i
                        break
            
            new_edge = edges[next_edge_index]
            done.add(next_edge_index)
            
            if new_edge[0] != last_endpoint:
                new_edge = np.flip(new_edge)
                
            new_edges.append(new_edge)
            last_edge = new_edge
            last_endpoint = new_edge[1]
            
            if last_endpoint == start_point:
                break
            
        new_edges = np.array(new_edges)
        return new_edges
            
    @classmethod
    def order_path_edges (cls, edges):
        verts = np.unique(edges)
        
        occurence_counts = np.array([
                np.sum(edges == x)
                for x in verts
            ])
        verts = verts[occurence_counts == 1]
        
        if len(verts) == 2:
            last_vertex = verts[0]
            end_vertex = verts[1]
            
            point_sets = [
                    set(x)
                    for x in edges
                ]
            
            new_edges = []
            
            while last_vertex != end_vertex:
                next_edge_index = None
                
                for i, ps in enumerate(point_sets):
                    if last_vertex in ps:
                        next_edge_index = i
                        break
                    
                edge = np.array(list(point_sets[next_edge_index]))
                
                if edge[0] != last_vertex:
                    edge = np.flip(edge)
                
                del point_sets[next_edge_index]
                        
                last_vertex = edge[1]
                new_edges.append(edge)
            
            new_edges = np.array(new_edges)
            
            return new_edges
        else:
            raise ValueError("Not a simple path.\n"+str(edges)+" "+str(len(verts)))
        
    @classmethod
    def index_order_edges (cls, edges):
        new_edges = []
        
        for i in range(len(edges)):
            tup = edges[i]
            
            if tup[0] > tup[1]:
                tup = (tup[1], tup[0])
                
            new_edges.append(tup)
                
        return new_edges
            
    @classmethod
    def length_of_pointed_edges (cls, points, edges):
        dist =  points[edges[:,1]] - points[edges[:,0]]
        dist = np.linalg.norm(dist, axis=1)
        return dist
    
    @classmethod
    def length_of_pointed_adjacency_list (cls, points, adjlist):
        distdict = {}
        
        for k1 in adjlist:
            for k2 in adjlist[k1]:
                dist = points[k2] - points[k1]
                dist = np.linalg.norm(dist)
                distdict[(k1, k2)] = dist
                
        return distdict
        
    @classmethod
    def edges_to_singular_linegroups (cls, points, edges):
        linegroups = np.empty((len(edges), 2, 2))
        
        starts = points[edges[:,0]]
        ends = points[edges[:,1]]
        
        linegroups[:,0] = starts
        linegroups[:,1] = ends
        
        return linegroups
        
    @classmethod
    def point_list_to_edges (cls, point_list):
        count = len(point_list)
        
        edges = []
        
        for i in range(1, count):
            s = point_list[i-1]
            e = point_list[i]
            edges.append((s, e))
            
        return edges
    
    @classmethod
    def pointed_cycle_to_linegroup (cls, points, cycle_edges):
        new_points = []
        
        for edge in cycle_edges:
            edge = edge[0]
            new_points.append(points[edge])
            
        new_points.append(points[cycle_edges[-1][1]])
        new_points = np.array(new_points)
        return new_points
        
    @classmethod
    def euclidean_weights_adjlist (cls, points, adjlist):
        weights_adjlist = {}
        
        for p1 in adjlist:
            for p2 in adjlist:
                dist = np.linalg.norm(points[p1] - points[p2])
                weights_adjlist[(p1, p2)] = dist
                
        return weights_adjlist
        
        
    @classmethod
    def pointed_cycles_to_linegroups (cls, points, cycle_edges_groups):
        linegroups = []
        
        for cycle_edges_group in cycle_edges_groups:
            linegroup = cls.pointed_cycle_to_linegroup(points, cycle_edges_group)
            linegroups.append(linegroup)
            
        return linegroups
            
    @classmethod
    def edges_to_adjacency_matrix (cls, point_count, edges):
        # Edges: (X, 2)
        
        adjmat = np.full((point_count, point_count), False, dtype=np.bool)
        
        adjmat[edges[:,0], edges[:,1]] = True
        adjmat[edges[:,1], edges[:,0]] = True
        
        return adjmat
    
    @classmethod
    def adjacency_matrix_xor (cls, m1, m2):
        return np.logical_xor(m1, m2)
    
    @classmethod
    def edges_to_adjacency_list (cls, point_count, edges, symmetric=True):
        # Edges: (X, 2)
        
        adjlists = {
                i : set()
                for i in range(point_count)
            }
        
        for edge in edges:
            adjlists[edge[0]].add(edge[1])
            
            if symmetric:
                adjlists[edge[1]].add(edge[0])
        
        return adjlists
    
    @classmethod
    def edges_to_vertex_occurence (cls, edges):
        vertex_occurences = set()
        
        for edge in edges:
            vertex_occurences.add(edge[0])
            vertex_occurences.add(edge[1])
            
        return vertex_occurences
    
    @classmethod
    def symmetrize_adjacency_list (cls, adjlist):
        adjlist = deepcopy(adjlist)
        
        for k1 in adjlist:
            for k2 in adjlist[k1]:
                adjlist[k2].add(k1)
                
        return adjlist
    
    @classmethod
    def desymmetrize_adjacency_list (cls, adjlist):
        adjlist = deepcopy(adjlist)
        
        for k1 in adjlist:
            for k2 in adjlist[k1]:
                adjlist[k2].discard(k1)
                
        return adjlist
    
    @classmethod
    def adjacency_lists_xor (cls, l1, l2):
        l1k = l1.keys()
        
        if l1k == l2.keys():            
            new_adjlist = {}
            
            for k in l1k:
                s1 = l1[k]
                s2 = l2[k]
                
                sn = s1.symmetric_difference(s2)
                new_adjlist[k] = sn
                
            return new_adjlist
        else:
            errmsg = "The keys are not equal."
            raise ValueError(errmsg)
        
    @classmethod
    def adjacency_lists_or (cls, l1, l2):
        l1k = l1.keys()
        
        if l1k == l2.keys():            
            new_adjlist = {}
            
            for k in l1k:
                s1 = l1[k]
                s2 = l2[k]
                
                sn = s1.union(s2)
                new_adjlist[k] = sn
                
            return new_adjlist
        else:
            errmsg = "The keys are not equal."
            raise ValueError(errmsg)
    
    @classmethod
    def adjacency_matrix_to_edges (cls, adjmat):
        adjmat = np.array(adjmat)
        edges = []
        point_count = len(adjmat)
        
        for i in range(point_count):
            adjacents = np.where(adjmat[i] == True)[0]
            adjmat[adjacents, i] = False
            
            for adj in adjacents:
                edges.append([i, adj])
                
        edges = np.array(edges)
        return edges
        
    @classmethod
    def adjacency_list_to_edges (cls, adjlist):
        point_count = len(adjlist)
        edges = []
        edges_set = set()
        
        for i in range(point_count):
            adjacents = adjlist[i]
            
            for adj in adjacents:
                tup = (i, adj)
                invtup = (adj, i)
                
                if tup not in edges_set and invtup not in edges_set:
                    edges_set.add(tup)
                    edges_set.add(invtup)
                    
                    edges.append([i, adj])
                    
        edges = np.array(edges)
        return edges
    
    @classmethod
    def bfs (cls, point_count, edges):
        adjmat = cls.edges_to_adjacency_matrix(point_count, edges)
        visited = np.full(point_count, False, dtype=np.bool)
        visited[0] = True
        
        jobs = [0]
        jcount = len(jobs)
        
        tree = []
        
        while jcount != 0:
            ci = jobs.pop(0)
            
            adjacents = np.where(adjmat[ci] == True)[0]
            adjacents = adjacents[~visited[adjacents]]
            
            for adj in adjacents:
                tree.append([ci, adj])
                visited[adj] = True
                
                jobs.append(adj)
                
            jcount = len(jobs)
            
        tree = np.array(tree)
        return tree
    
    @classmethod
    def backtrack_inverted_spanning_tree_to_root (cls, start_point, inverted_tree):
        edges = []
        
        current_point = start_point
        adj = list(inverted_tree[current_point])
        
        while len(adj) != 0:
            upwards = adj[0]
            edges.insert(0, (upwards, current_point))
            
            current_point = upwards
            adj = list(inverted_tree[current_point])
        
        return np.array(edges)
        
    @classmethod
    def _cycle_split (cls, cycle_adjlist, cycle_edges, xor):        
        splitter_lines = cls._get_edges_with_common_points(cycle_adjlist, xor)
        
        if len(splitter_lines) == 0:
            splitter_line = None
        else:
            splitter_line = splitter_lines[0]
            
        return cls._split_cycle_with_splitter_line(cycle_edges, splitter_line)
        
    @classmethod
    def _get_edges_with_common_points (cls, cycle_adjlist, other_graph):
        splitter_line = None
        
        found_edges = []
        edge_hashes = set()
        
        for k1 in other_graph:
            for k2 in other_graph[k1]:
                verts_active = len(cycle_adjlist[k1]) > 0 and len(cycle_adjlist[k2]) > 0
                no_connection = (k1 not in cycle_adjlist[k2]) and (k2 not in cycle_adjlist[k1])
                
                if verts_active and no_connection:
                    splitter_line = (k1, k2)
                    h = cls._hash_undirected_edges([splitter_line])
                    
                    if h not in edge_hashes:
                        found_edges.append(splitter_line)
                        edge_hashes.add(h)
                        
        return found_edges
            
    @classmethod
    def _split_cycle_with_splitter_line (cls, cycle_edges, splitter_line):
        if splitter_line is None:
            return [cycle_edges]
        else:
            k1 = splitter_line[0]
            k2 = splitter_line[1]
            cycle2_start = None
            cycle2_stop = None
            
            cycle1 = []
            cycle1_started = False
            cycle1_continued = False
            
            for i in range(0, len(cycle_edges)):
                edge = cycle_edges[i]
                
                if cycle1_started == False:
                    if edge[0] == k1:
                        cycle2_start = i
                        cycle1_started = True
                        
                        cycle1.append((k1, k2))                        
                    elif edge[0] == k2:
                        cycle2_start = i
                        cycle1_started = True
                        
                        k1 = splitter_line[1]
                        k2 = splitter_line[0]
                        
                        cycle1.append((k1, k2))    
                    else:
                        cycle1.append(edge)
                else:
                    if cycle1_continued == False:
                        if edge[0] == k2:
                            cycle1_continued = True
                            cycle2_stop = i
                            cycle1.append(edge)
                    else:
                        cycle1.append(edge)
            
            cycle1 = np.array(cycle1)            
            cycle2 = [(k2, k1)]
            
            for i in range(cycle2_start, cycle2_stop):
                edge = cycle_edges[i]
                cycle2.append(edge)
                
            cycle2 = np.array(cycle2)
            
            return [
                    cycle1, cycle2
                ]
    
    @classmethod
    def bfs_cycles (cls, point_count, edges):
        bfs = cls.bfs(point_count, edges)
        bfs_xor = cls.adjacency_lists_xor(
                cls.edges_to_adjacency_list(point_count, bfs, symmetric=True),
                cls.edges_to_adjacency_list(point_count, edges, symmetric=True)
            )
        edges_adjlist = cls.edges_to_adjacency_list(point_count, edges, symmetric=False)
        desym_bfs_xor = cls.desymmetrize_adjacency_list(bfs_xor)
        
        inverted_bfs = np.array([
                [edge[1], edge[0]]
                for edge in bfs
            ])
        
        inverted_bfs = cls.edges_to_adjacency_list(point_count, inverted_bfs, symmetric=False)
        cycles = []
        cycle_hashes = set()
        
        for k1 in desym_bfs_xor:
            for k2 in desym_bfs_xor[k1]:
                c1 = cls.edges_to_adjacency_list(
                        point_count, 
                        cls.backtrack_inverted_spanning_tree_to_root(k1, inverted_bfs), 
                        symmetric=False
                    )
                
                c2 = cls.edges_to_adjacency_list(
                        point_count, 
                        cls.backtrack_inverted_spanning_tree_to_root(k2, inverted_bfs), 
                        symmetric=False
                    )
        
                xord = cls.adjacency_lists_xor(c1, c2)
                xord[k1].add(k2)
                xord_edges = cls.adjacency_list_to_edges(xord)
                xord_edges = cls.order_cycle_edges(xord_edges)
                
                xord_edges = cls._cycle_split(xord, xord_edges, bfs_xor)
                # xord_edges = cls._cycle_split(xord, xord_edges, edges_adjlist)
                
                for cycle in xord_edges:
                    h = cls._hash_undirected_edges(cycle)
                    
                    if h not in cycle_hashes:
                        cycles.append(cycle)
                        cycle_hashes.add(h)
                
        return cycles
        
    @classmethod
    def get_path_for_inverse_tree (cls, node_list, start_index):
        path = [start_index]
        
        indx = start_index
        
        while True:
            new_indx = node_list[indx]["p"]
            
            if indx != new_indx:
                path.insert(0, new_indx)
                indx = new_indx
            else:
                break
            
        return path
            
    @classmethod
    def path_to_adjacency_list (cls, point_count, path, symmetry=True):
        adjlist = {
                i : set()
                for i in range(point_count)
            }
        
        last_p = None
        
        for p in path:
            if last_p is not None:
                adjlist[last_p].add(p)
                
                if symmetry:
                    adjlist[p].add(last_p)
            
            last_p = p
            
        return adjlist
        
        
        
    @classmethod
    def fundamental_cycles (cls, point_count, edges):
        # edges = cls.bfs(point_count, edges)
        adjlist = cls.edges_to_adjacency_list(point_count, edges, symmetric=True)
        
        contree = {}
        node_stack = [0]
        parent_stack = [None]
        fundamentals = []
        
        for i in range(point_count):
            d = {
                    "p" : i,
                    "i" : i
                }
            contree[i] = d
            
        stack_size = len(node_stack)
        
        while stack_size > 0:
            c_i = node_stack.pop(0)
            c_p = parent_stack.pop(0)
            c_adjs = set(adjlist[c_i])
            
            for adj in c_adjs:
                if adj == c_p:
                    continue
                
                adj_node = contree[adj]
                
                if adj_node["p"] != adj:
                    path_i = cls.get_path_for_inverse_tree(contree, c_i)
                    path_j = cls.get_path_for_inverse_tree(contree, adj)
                    
                    path_i = cls.path_to_adjacency_list(point_count, path_i)
                    path_j = cls.path_to_adjacency_list(point_count, path_j)
                    
                    path_i[c_i].add(adj)
                    
                    path_xor = cls.adjacency_lists_xor(path_i, path_j)
                    fundamentals.append(path_xor)
                else:
                    adj_node["p"] = c_i
                    node_stack.insert(0, adj)
                    parent_stack.insert(0, c_i)
                
                adjlist[c_i].remove(adj)
                
            stack_size = len(node_stack)
            
        return fundamentals
        
    @classmethod
    def get_paths_to_tree_leaves (cls, point_count, edges, start_index=0):
        adjlist = cls.edges_to_adjacency_list(point_count, edges, symmetric=False)
        
        paths = []
        
        job_stack = [start_index]
        leaf_stack = [len(adjlist[job_stack[0]]) == 0]
        
        stack_size = len(job_stack)
        
        while stack_size != 0:
            indx = job_stack[0]
            is_leaf = leaf_stack[0]
            
            if is_leaf:
                path = list(reversed(list(job_stack)))
                paths.append(path)
                
                job_stack.pop(0)
                leaf_stack.pop(0)
            else:
                if len(adjlist[indx]) != 0:
                    nextind = adjlist[indx].pop()
                    is_leaf = len(adjlist[nextind]) == 0
                    
                    job_stack.insert(0, nextind)
                    leaf_stack.insert(0, is_leaf)                    
                else:
                    job_stack.pop(0)
                    leaf_stack.pop(0)
                    
            stack_size = len(job_stack)
        
        return paths
        
    @classmethod
    def fundamental_cycles_v2 (cls, point_count, edges):
        adjlist = cls.edges_to_adjacency_list(point_count, edges, symmetric=True)
        cycles = []
        cyclehashes = set()
        cycleends = set()
        
        bfs = cls.bfs(point_count, edges)
        bfs_xor = cls.adjacency_lists_xor(
                cls.edges_to_adjacency_list(point_count, bfs, symmetric=True),
                cls.edges_to_adjacency_list(point_count, edges, symmetric=True)
            )
        bfs_symadjlist = cls.edges_to_adjacency_list(point_count, bfs, symmetric=True)
        
        leafs = [
                x
                for x in bfs_symadjlist
                if len(bfs_symadjlist[x]) == 1
            ]
        
        leaf_xor_connects = {
                x : bfs_xor[x]
                for x in leafs
                if len(bfs_xor[x]) != 0
            }         
        
        leaf_indices = np.array(list(leaf_xor_connects.keys()))
        leaf_lengths = np.array([len(leaf_xor_connects[x]) for x in leaf_indices])
        
        while True:
            if len(leaf_indices) == 0:
                break
            
            lowest = np.min(leaf_lengths)
            lowest_indices = np.where(leaf_lengths == lowest)[0]
            
            for lowest_index in lowest_indices:                
                node_main_index = leaf_indices[lowest_index]
                
                xor_adj = leaf_xor_connects[node_main_index].pop()
                cycle, cyclehash = cls._find_exclusive_path(node_main_index, xor_adj, adjlist, cyclehashes)
                
                if cycle is not None:
                    ends = (cycle[0][0], cycle[0][1], cycle[-1][0], cycle[-1][1])
                    
                    if ends not in cycleends:
                        cycles.append(cycle)
                        cyclehashes.add(cyclehash)
                        cycleends.add(ends)
                    
                leaf_lengths[lowest_index] -= 1
                    
            dead_indices = np.where(leaf_lengths == 0)[0]
            
            if len(dead_indices) > 0:
                leaf_indices = np.delete(leaf_indices, dead_indices)
                leaf_lengths = np.delete(leaf_lengths, dead_indices)
            
        return cycles
        
    @classmethod
    def fundamental_cycles_v3 (cls, point_count, edges):
        adjlist = cls.edges_to_adjacency_list(point_count, edges, symmetric=True)
        
        bfs = cls.bfs(point_count, edges)
        bfs_xor = cls.adjacency_lists_xor(
                cls.edges_to_adjacency_list(point_count, bfs, symmetric=True),
                cls.edges_to_adjacency_list(point_count, edges, symmetric=True)
            )
        
        
    @classmethod
    def _concatenate_cycle (cls, hit, first_index, second_index, node_list):
        edges = [(hit, first_index), (first_index, second_index)]
        last_index = hit
        
        while True:
            indx = node_list[last_index]["p"]
            
            if indx == first_index:
                break
            
            edges.insert(2, (indx, last_index))
            
            last_index = indx
            
        h = cls._hash_undirected_edges(edges)
            
        return edges, h
    
    @classmethod
    def _hash_undirected_edges (cls, edges):
        h = len(edges) * 1171
        
        for edge in edges:
            eh = (edge[0] + edge[1]) ** 3
            eh += 179
            h += eh
            
        return h
            
        
    @classmethod
    def _find_exclusive_path (cls, first_index, second_index, adjlist, cyclehashes):
        node_list = {
                x : {
                        "p" : None
                    }
                for x in range(len(adjlist))
            }
        node_list[second_index]["p"] = first_index
        
        cparents = [first_index]
        cindices = [second_index]
        
        while True:
            new_cparents = []
            new_cindices = []
            
            break_loop = False
            
            for cparent, cindex in zip(cparents, cindices):
                adjs = adjlist[cindex] - set([cparent])
                
                if first_index in adjs:
                    
                    edges, h = cls._concatenate_cycle(cindex, first_index, second_index, node_list)
                    
                    if h not in cyclehashes:
                        break_loop = True
                        break
                else:
                    for adj in adjs:
                        if node_list[adj]["p"] is None:
                            new_cparents.append(cindex)
                            new_cindices.append(adj)
                            
                            node_list[adj]["p"] = cindex
                        
            if break_loop:
                break
            
            cparents = new_cparents
            cindices = new_cindices
            
            if len(cindices) == 0:
                edges = None
                h = None
                break
            
        return edges, h
    
    @classmethod
    def get_vertex_degrees (cls, adjlist):
        degrees = {}
        
        for v in adjlist:
            degrees[v] = len(adjlist[v])
            
        return degrees
    
    @classmethod
    def remove_filaments_from_adjacency_list (cls, adjlist):
        degrees = cls.get_vertex_degrees(adjlist)
        adjlist = deepcopy(adjlist)
        cls._remove_simple_filaments(adjlist, degrees)
        cls._remove_complex_filaments(adjlist, degrees)
        
        return adjlist
        
    @classmethod
    def _remove_simple_filaments (cls, adjlist, degrees):
        verts_with_one = [
                k 
                for k in degrees
                if degrees[k] == 1
            ]
        
        for k in verts_with_one:
            cv = k
            
            while True:
                adjs = adjlist[cv]
                
                if len(adjs) == 1:
                    adj = adjs.pop()
                    degrees[cv] -= 1
                    
                    if cv in adjlist[adj]:
                        adjlist[adj].remove(cv)
                        degrees[adj] -= 1
                    
                    cv = adj
                else:
                    break
            
    @classmethod
    def _remove_complex_filaments (cls, adjlist, degrees):
        branch_verts = [
                k
                for k in degrees
                if degrees[k] > 2
            ]
        
        filament_list = cls._create_filament_candidate_list(branch_verts, adjlist)
        
        for filament in filament_list:
            filament_length = len(filament)
            start_vertex = filament[0][0]
            end_vertex = filament[-1][1]
            
            if start_vertex != end_vertex:
                forbidden_vertices = set()
                
                for i in range(1, filament_length):
                    forbidden_vertices.add(filament[i][0])
                    
                has_cycle = cls._has_path_from_to(start_vertex, end_vertex, adjlist, forbidden_vertices)
                
                if not has_cycle:
                    for edge in filament:
                        v1 = edge[0]
                        v2 = edge[1]
                        
                        if v2 in adjlist[v1]:
                            adjlist[v1].remove(v2)
                            degrees[v1] -= 1
                            
                        if v1 in adjlist[v2]:
                            adjlist[v2].remove(v1)
                            degrees[v2] -= 1
            
            
    @classmethod
    def _create_filament_candidate_list (cls, branch_verts, adjlist):
        filament_list = []
        filament_hash_set = set()
        
        for branch_vert in branch_verts:
            adjs = adjlist[branch_vert]
            cls._find_possible_complex_filaments(branch_vert, adjs, adjlist, filament_list, filament_hash_set)
            
        return filament_list
            
    @classmethod
    def _find_possible_complex_filaments (cls, branch_vert, branch_adjs, adjlist, filament_list, filament_hash_set):
        '''
        Finds possible non-isolated filaments with the given information.
        The variable branch_vert is a vertex that has a degree of at least
        3. ls_adjs are the adjacents of the branch_vert that have a degree
        of 2. adjlist is the adjacency list. The filament_list and
        filament_hash_set are global containers for informations about
        found candidates for non-isolated filaments.
        '''
        
        for ls_vertex in branch_adjs:
            edges = [(branch_vert, ls_vertex)]
            
            cv = ls_vertex
            last_v = branch_vert
            
            while True:
                c_adjs = adjlist[cv] - set([last_v])
                c_adjs_len = len(c_adjs)
                
                if c_adjs_len > 1:
                    break
                elif c_adjs_len == 1:
                    last_v = cv
                    cv = list(c_adjs)[0]
                    edges.append((last_v, cv))
                else:
                    break
                
            h = cls._hash_undirected_edges(edges)
            
            if h not in filament_hash_set:
                filament_list.append(edges)
                filament_hash_set.add(h)
                
    @classmethod
    def _has_path_from_to (cls, start_vertex, end_vertex, adjlist, forbidden_vertices):
        '''
        SPECIAL FUNCTION FOR FILAMENT DETECTION
        '''
        c_indices = [start_vertex]
        visited = set()
        
        first = len(forbidden_vertices) == 0
        
        while len(c_indices) != 0:
            new_c_indices = []
            
            for c_index in c_indices:
                visited.add(c_index)
                
                c_adjs = (adjlist[c_index] - forbidden_vertices) - visited
                
                if first:
                    c_adjs.discard(end_vertex)
                    first = False
                
                if end_vertex in c_adjs:
                    return True
                else:
                    new_c_indices.extend(c_adjs)
                    
            c_indices = new_c_indices
                    
        return False
                
    @classmethod
    def _remap_indices_by_adjacency_list_degrees (cls, adjlist):
        '''
        ADJLIST MUST BE SYMMETRIC!
        '''
        
        remap = {}
        index_counter = 0
                
        for k in adjlist:
            adjs = adjlist[k]
            
            if len(adjs) != 0:
                remap[k] = index_counter
                index_counter += 1
            
        new_adjlist = defaultdict(set)
            
        for k in adjlist:
            adjs = adjlist[k]
            
            if len(adjs) != 0:
                new_k = remap[k]
                
                for adj in adjs:
                    new_adj = remap[adj]
                    new_adjlist[new_k].add(new_adj)
                    
        return remap, new_adjlist
                
    @classmethod
    def _get_minimal_point (cls, cor_points):
        '''
        SPECIAL FUNCTION FOR CYCLE BASIS DETECTION
        '''
        xminind = np.argmin(cor_points[:,0])
        xmin = cor_points[xminind, 0]
        xminindices = np.where(cor_points[:,0] == xmin)[0]
        
        if len(xminindices) == 1:
            minind = xminind
        else:
            yminind = np.argmin(cor_points[xminindices, 1])
            minind = xminindices[yminind]
        
                
    @classmethod
    def cycle_basis (cls, linegroups):
        points, graph = LinearAlgebra2D.convert_linegroups_to_graph(linegroups)
        adjlist = cls.edges_to_adjacency_list(len(points), graph, symmetric=True)
        
        cor_adjlist = cls.remove_filaments_from_adjacency_list(adjlist)
        remap, cor_adjlist = cls._remap_indices_by_adjacency_list_degrees(cor_adjlist)
        cor_points = points[np.sort(list(remap.keys()))]
        
        start_index = cls._get_minimal_point(cor_points)
        support_line = np.array([0, -1])
        
    @classmethod
    def get_clockwise_most (cls, vprev_point, vprev_index, vcurr_point, vcurr_index, vprev_adjs, vcurr_adjs, points):
        if len(vcurr_adjs - set([vprev_index])) == 0:
            return None
        
        dcurr = vcurr_point - vprev_point
        vnext_index = list(vcurr_adjs - set([vprev_index]))[0]
        vnext = points[vnext_index]
        dnext = vnext - vcurr_point
        vcurrIsConves = (np.dot(dnext, LinearAlgebra2D.get_normal_vector(dcurr))) <= 0
        
        
        
        '''
        vnext = adjacent vertex of vcurr not equal to vpres
        dnext = vnext - vcurr
        vcurrIsConvex = (Dot(dnext, Perp(dcurr)) <= 0)
        '''
                
    @classmethod
    def _find_filaments (cls, adjlist):
        filament_indices = set()
        
        for i in adjlist:
            count = len(adjlist[i])
            
            if count == 2:
                filament_indices.add(i)
            
        
        return filament_indices
                
    @classmethod
    def _connect_filaments (cls, adjlist, filament_indices):
        filaments_dict = {}
        
        while len(filament_indices) != 0:
            start_index = filament_indices.pop()
            
            adjs = adjlist[start_index].intersection(filament_indices)
            
            edges = []
            
            for adj in adjs:
                subedges = [(start_index, adj)]
                last = adj
                filament_indices.discard(last)
                
                new_adjs = adjlist[last].intersection(filament_indices)
                
                while len(new_adjs) != 0:
                    next_vertex = new_adjs.pop()
                    filament_indices.discard(next_vertex)
                    
                    subedges.append((last, next_vertex))
                    last = next_vertex
                    
                    new_adjs = adjlist[last].intersection(filament_indices)
                    
                edges.extend(subedges)
                    
            if len(edges) > 1:
                edges = np.array(edges)
                edges = cls.order_path_edges(edges)
                
                filaments_dict[(edges[0][0], edges[-1][1])] = edges
                
        return filaments_dict
            
                
    @classmethod
    def substitution_dict_for_adjacency_list (cls, adjlist):
        # Sub-Tuble -> List of edges
        filament_indices = cls._find_filaments(adjlist)
        
        filaments = cls._connect_filaments(adjlist, filament_indices)
        update_dict = {}
        
        for sub_k in filaments:
            flipped = np.flip(filaments[sub_k], axis=[1, 0])
            new_sub_k = (sub_k[1], sub_k[0])
            update_dict[new_sub_k] = flipped
        
        filaments.update(update_dict)
        
        return filaments
    
    @classmethod
    def substitution_dict_for_points (cls, adjlist):
        sub_dict = {}
        
        counter = 0
        
        for adjlist_k in adjlist:
            adjs = adjlist[adjlist_k]
            
            if len(adjs) != 0:
                sub_dict[adjlist_k] = counter
                counter += 1
                
        return sub_dict
                
        
                
    @classmethod
    def substitute_adjacency_list (cls, adjlist, subdict):
        # Subdict: Subtuple -> List of edges
        adjlist = deepcopy(adjlist)
        
        for sub_tuple in subdict:
            for edge in subdict[sub_tuple]:
                adjlist[edge[0]].remove(edge[1])
                
            adjlist[sub_tuple[0]].add(sub_tuple[1])
            
        return adjlist
    
    @classmethod
    def substitute_points (cls, adjlist, subdict_points):
        new_adjlist = {}
        
        for subdict_k in subdict_points:
            new_key = subdict_points[subdict_k]
            adjs = adjlist[subdict_k]
            
            new_adjs = set()
            
            for adj in adjs:
                new_adjs.add(subdict_points[adj])
            
            new_adjlist[new_key] = new_adjs
            
        return new_adjlist
        
    @classmethod
    def desubstitute_points_in_adjacency_list (cls, adjlist, subdict_points, point_count):
        subdict_points = {
                subdict_points[x] : x
                for x in subdict_points
            }
        
        new_adjlist = {}
        
        for subdict_k in subdict_points:
            new_key = subdict_points[subdict_k]
            adjs = adjlist[subdict_k]
            
            new_adjs = set()
            
            for adj in adjs:
                new_adjs.add(subdict_points[adj])
            
            new_adjlist[new_key] = new_adjs
            
        for i in range(point_count):
            if i not in new_adjlist:
                new_adjlist[i] = set()
            
        return new_adjlist
    
    @classmethod
    def desubstitute_points_in_edges (cls, edges, subdict_points):
        subdict_points = {
                subdict_points[x] : x
                for x in subdict_points
            }
        
        for i in range(len(edges)):
            edge = edges[i]
            new_edge = (subdict_points[edge[0]], subdict_points[edge[1]])
            edges[i] = new_edge
            
        return edges
            
    
    @classmethod
    def desubstitute_adjacency_list (cls, sub_adjlist, subdict):
        sub_adjlist = deepcopy(sub_adjlist)        

        # PRUEFT NICHT, OB DIE TUPEL AUCH WIRKLICH DRIN SIND!!!
        
        for sub_tuple in subdict:
            if sub_tuple[1] in sub_adjlist[sub_tuple[0]]:
                for edge in subdict[sub_tuple]:
                    sub_adjlist[edge[0]].add(edge[1])
                    
                sub_adjlist[sub_tuple[0]].remove(sub_tuple[1])
        
        return sub_adjlist
    
    @classmethod
    def desubstitute_edges (cls, edges, subdict):
        i = 0
        
        while i < len(edges):
            
            i += 1 
        
                
    @classmethod
    def substitute_length_adjacency_list (cls, length_adjlist, subdict):
        # Length adjlist: Edge -> Length
        length_adjlist = deepcopy(length_adjlist)
        
        for sub_tuple in subdict:
            length_sum = 0.0
            
            for edge in subdict[sub_tuple]:
                edge = tuple(edge)
                length_sum += length_adjlist[edge]
                del length_adjlist[edge]
                
            length_adjlist[sub_tuple] = length_sum
            
        return length_adjlist
    
    @classmethod
    def substitute_points_in_length_adjacency_list (cls, length_adjlist, subdict):
        new_adjlist = {}
        
        for length_tuple in length_adjlist:
            new_tuple = (subdict[length_tuple[0]], subdict[length_tuple[1]])
            new_val = length_adjlist[length_tuple]
            
            new_adjlist[new_tuple] = new_val
        
        return new_adjlist
        
    @classmethod
    def _dijkstra_min_distance (cls, distances, visited):
        argsorted = np.argsort(distances)
        
        for i in argsorted:
            if visited[i] == False:
                return i
            
        return None
        
    @classmethod
    def _dijkstra_shortest_paths (cls, parents, source_index):
        paths = []
        
        for i in range(len(parents)):
            if i == source_index:
                continue
            
            l = i
            p = parents[i]
            
            path = [(p, l)]
            
            while True:
                l = p
                p = parents[l]
                
                if p == -1:
                    break
                
                tup = (p, l)
                
                path.insert(0, tup)
                
            paths.append(np.array(path))
            
        return paths
        
    @classmethod
    def dijkstra (cls, adjlist, weights_adjlist, source_vertex):
        visited = np.full(len(adjlist), False, dtype=np.bool)
        indices = np.arange(len(adjlist))
        
        distances = np.full(len(adjlist), np.inf, dtype=np.float)
        distances[source_vertex] = 0
        
        parents = np.full(len(adjlist), -1, dtype=np.int)
        
        for _ in indices:
            u = cls._dijkstra_min_distance(distances, visited)
            adjs = adjlist[u]
            
            visited[u] = True
            
            non_visited = indices[visited == False]
            
            for v in non_visited:
                if (v in adjs):
                    if distances[v] > distances[u] + weights_adjlist[(u, v)]:
                        distances[v] = distances[u] + weights_adjlist[(u, v)]
                        parents[v] = u
        
        path_dict = {
                (path[0][0], path[-1][1]) : path
                for path in cls._dijkstra_shortest_paths(parents, source_vertex)
            }
        distances = {
                (source_vertex, i) : x
                for i, x in enumerate(distances)
                if i != source_vertex
            }
        
        return path_dict, distances
    
    @classmethod
    def dijkstra_all (cls, adjlist, weights_adjlist):
        path_dict = {}
        distances = {}
        
        for i in range(len(adjlist)):
            pd, dists = cls.dijkstra(adjlist, weights_adjlist, i)
            
            path_dict.update(pd)
            distances.update(dists)
            
        return path_dict, distances
    
    @classmethod
    def length_of_cycle (cls, edges, length_adjlist):
        lengths = [
                length_adjlist[tuple(edge)]
                for edge in edges
            ]
        return np.sum(lengths)
    
    @classmethod
    def filter_independent_cycles_by_sizes (cls, point_count, cycles_edges, cycle_sizes):
        cycles_adjlists = [
                cls.edges_to_adjacency_list(point_count, x, symmetric=True)
                for x in cycles_edges
            ]
        
        cycle_tuples_set = set()
        filtered_cycles_set = set()
        filtered_cycles = []
        
        for cycle_edges in cycles_edges:
            for edge in cycle_edges:
                n1 = edge[0]
                n2 = edge[1]
                
                if (n1, n2) not in cycle_tuples_set and (n2, n1) not in cycle_tuples_set:
                    cycle_candidate = None
                    cycle_length = np.inf
                    
                    for i, cycle_adjlist in enumerate(cycles_adjlists):
                        if n1 in cycle_adjlist[n2] or n2 in cycle_adjlist[n1]:
                            length = cycle_sizes[i]
                            
                            if length < cycle_length:
                                cycle_candidate = i
                                cycle_length = length
                                
                    cycle_tuples_set.add((n1, n2))
                    
                    if cycle_candidate is not None:
                        if cycle_candidate not in filtered_cycles_set:
                            filtered_cycles.append(cycle_candidate)
                            filtered_cycles_set.add(cycle_candidate)
                        
        filtered_cycles = [
                cycles_edges[i]
                for i in filtered_cycles
            ]
        return filtered_cycles
        
    @classmethod
    def mcb_base (cls, adjlist, weights_adjlist):
        mincycles = []
        mincycles_hashes = set()
        paths, _ = cls.dijkstra_all(adjlist, weights_adjlist)
        
        vertices = np.arange(len(adjlist))
        paths_vertoccs = {
                x : cls.edges_to_vertex_occurence(paths[x])
                for x in paths
            }
        
        for v in vertices:
            for e1 in adjlist:
                for e2 in adjlist[e1]:
                    p1 = paths_vertoccs.get((e1, v), None)
                    p2 = paths_vertoccs.get((v, e2), None)
                    
                    if p1 is not None and p2 is not None:
                        if p1.intersection(p2) == set([v]):
                            c = np.concatenate([
                                    paths[(e1, v)],
                                    paths[(v, e2)],
                                    np.array([[e1, e2]])
                                ], axis=0)
                            
                            h = cls._hash_undirected_edges(c)
                            
                            if h not in mincycles_hashes:
                                mincycles.append(c)
                                mincycles_hashes.add(h)
                                
        return mincycles
    
    @classmethod
    def minimum_cycle_basis (cls, adjlist, weights_adjlist):
        mincycles = cls.mcb_base(adjlist, weights_adjlist)
                                
        lengths = [
                cls.length_of_cycle(x, weights_adjlist)
                for x in mincycles
            ]
        sorted_by_lengths = np.argsort(lengths)
        mincycles = [
                mincycles[i]
                for i in sorted_by_lengths
            ]
        lengths = [
                lengths[i]
                for i in sorted_by_lengths
            ]
            
        mincycles = cls.filter_independent_cycles_by_sizes(len(adjlist), mincycles, lengths)
            
        return mincycles
            
    @classmethod
    def minimum_cycle_basis_polygons (cls, adjlist, weights_adjlist, points, substitute=False):
        if substitute:
            # NOT A SIMPLE PATH-PROBLEM -> order_path_edges
            sub_dict = cls.substitution_dict_for_adjacency_list(adjlist)
            
            weights_adjlist = cls.substitute_length_adjacency_list(weights_adjlist, sub_dict)
            
            adjlist = cls.substitute_adjacency_list(adjlist, sub_dict)
            
            sub_dict_points = cls.substitution_dict_for_points(adjlist)
            
            weights_adjlist = cls.substitute_points_in_length_adjacency_list(weights_adjlist, sub_dict_points)
            
            adjlist = cls.substitute_points(adjlist, sub_dict_points)
            
        mincycles = cls.mcb_base(adjlist, weights_adjlist)
                             
        if substitute:           
            mincycles = [
                    cls.desubstitute_points_in_edges(x, sub_dict_points)
                    for x in mincycles
                ]
              
            mincycles = [
                    cls.edges_to_adjacency_list(len(points), cycle, symmetric=False)
                    for cycle in mincycles
                ]
             
            mincycles = [
                    cls.desubstitute_adjacency_list(cycle, sub_dict)
                    for cycle in mincycles
                ]
              
            mincycles = [
                    cls.adjacency_list_to_edges(cycle)
                    for cycle in mincycles
                ]
        
        mincycles = [
                cls.order_cycle_edges(cycle)
                for cycle in mincycles
            ]
                                
        areas = [
                Polygon(points[x[:,0]]).area
                for x in mincycles
            ]
        sorted_by_areas = np.argsort(areas)
        mincycles = [
                mincycles[i]
                for i in sorted_by_areas
            ]
        areas = [
                areas[i]
                for i in sorted_by_areas
            ]
            
        mincycles = cls.filter_independent_cycles_by_sizes(len(points), mincycles, areas)
            
        return mincycles
    
    @classmethod
    def get_all_possible_paths (cls, adjlist, start_index, end_index, maxdepth):
        final_paths = []
        
        paths = [list([start_index])]
        paths_sets = [set([start_index])]
        
        all_paths = False
        
        for _ in range(maxdepth):
            old_length = len(paths)
            
            if old_length == 0:
                print("No more paths to evaluate!")
                all_paths = True
                break
            
            rng = range(len(paths))
            
            for i in rng:
                path = paths[i]
                path_set = paths_sets[i]
                
                lastnode = path[-1]
                adjacents = adjlist[lastnode] - path_set
                                    
                if end_index in adjacents:
                    final_path = list(path)
                    final_path.append(end_index)
                    adjacents.remove(end_index)
                    
                    final_paths.append(final_path)
                    
                for adj in adjacents:
                    final_path = list(path)
                    final_path.append(adj)
                    final_path_set = set(final_path)
                    
                    paths.append(final_path)
                    paths_sets.append(final_path_set)
                    
            paths = paths[old_length : ]
            paths_sets = paths_sets[old_length : ]
                    
        return final_paths, all_paths
                    
    @classmethod
    def get_all_possible_paths_auto (cls, adjlist, start_index, end_index, mincount, depth_limit):
        last_count = -1
        
        for i in range(1, depth_limit):
            paths, all_paths = cls.get_all_possible_paths(adjlist, start_index, end_index, i)
            
            if all_paths:
                return paths, all_paths
            
            if len(paths) >= mincount:
                return paths, all_paths
            else:
                if last_count == len(paths):
                    i += 1
                else:
                    last_count = len(paths)
                    
        return paths, all_paths
            
    @classmethod
    def get_all_possible_paths_multi_auto (cls, adjlist, mincount, depth_limit, evaluate_indices=None):
        if evaluate_indices is None:
            points = sorted(list(adjlist.keys()))
        else:
            points = np.sort(evaluate_indices)
            
        points_count = len(points)
        
        all_paths = {}
        
        for i in range(points_count - 1):
            for j in range(i+1, points_count):
                all_paths[(points[i], points[j])] = cls.get_all_possible_paths_auto(adjlist, points[i], points[j], mincount, depth_limit)[0]
                
        return all_paths
    
    @classmethod
    def get_essentials_for_key (cls, paths, key_tuple):
        paths = paths[key_tuple]
        
        if len(paths) == 0:
            return set()
        elif len(paths) == 1:
            return paths[0]
        else:
            t = paths[0]
            t = t.intersection(*paths[1:])
            
        return t
        
        
        
    class RemovalLogicBlock (ABC):
        def __init__ (self):
            pass
        
        @abstractmethod
        def reduce (self, adjlist, deleteables_set):
            pass
        
    class CandidatesRemovalLogicBlock (RemovalLogicBlock):
        def __init__ (self, edges):
            self.__edges = {
                    (tup[0], tup[1])
                    if tup[0] < tup[1]
                    else (tup[1], tup[0])
                    for tup in edges
                }
            
        def reduce (self, adjlist, deleteables_set):
            return deleteables_set.intersection(self.__edges)
        
    class MinEdgeRemovalLogicBlock (RemovalLogicBlock):
        def __init__ (self, min_edges):
            self.__min_edges = min_edges
            
        def reduce (self, adjlist, deleteables_set):
            removables = set()
            
            for deleteable in deleteables_set:
                s = deleteable[0]
                e = deleteable[1]
                
                s_adjs = adjlist[s]
                e_adjs = adjlist[e]
                
                s_side = len(s_adjs) >= self.__min_edges + 1
                e_side = len(e_adjs) >= self.__min_edges + 1
                
                if s_side and e_side:
                    removables.add(deleteable)
                    
            if len(removables) == 0:
                return None
            else:
                return removables
                
                
                
            
        
    class RemovalANDLogic (RemovalLogicBlock):
        def __init__ (self, attempts):
            self.__attempts = attempts
            
        def reduce (self, adjlist, deleteables_set):
            returnset = deleteables_set
            
            for attempt in self.__attempts:
                reduced = attempt.reduce(adjlist, returnset)
                
                if reduced is not None:
                    returnset = returnset.intersection(reduced)
                else:
                    returnset = set()
                    break
                    
            if len(returnset) == 0:
                return None
            else:
                return returnset
            
    @classmethod
    def remove_nonessential_connections (cls, adjlist, keeper_points, removal_logic=None):
        adjlist = deepcopy(adjlist)
        
        all_edges = set([
                (i, j)
                if i < j
                else (j, i)
                for i in adjlist
                for j in adjlist[i]
            ])
        tupled_paths = cls.get_all_possible_paths_multi_auto(adjlist, 50, 13, keeper_points)
        
        tupled_paths = {
                k : [
                        set(cls.index_order_edges(cls.point_list_to_edges(x)))
                        for x in tupled_paths[k]
                    ]
                for k in tupled_paths
            }
        
        while True:
            essentials = {
                    k : cls.get_essentials_for_key(tupled_paths, k)
                    for k in tupled_paths
                }
            essentials = [
                    essentials[k]
                    for k in essentials
                ]
            
            if len(essentials) != 0:
                essentials = essentials[0].union(*essentials[1:])
            else:
                essentials = set()
                
            deleteables = all_edges - essentials
            
            if len(deleteables) == 0:
                break
            
            if removal_logic is not None:
                deleteables = removal_logic.reduce(adjlist, deleteables)
                
                if deleteables is not None:
                    deleteables = deleteables.pop()
                else:
                    break
            else:
                deleteables = deleteables.pop()
            
            if deleteables is None:
                break
            
            all_edges.remove(deleteables)
            
            adjlist[deleteables[0]].remove(deleteables[1])
            adjlist[deleteables[1]].remove(deleteables[0])
            
            for k in tupled_paths:
                tupled_paths[k] = [
                        x
                        for x in tupled_paths[k]
                        if deleteables not in x
                    ]
                
        return adjlist
    
class SteinerTree2D ():
    class SubstitutionVariables ():
        def __init__ (self, points, reachabilities, cut_points, moveable_points):
            self.points = points
            self.reachabilities = reachabilities
            self.cut_points = cut_points
            self.moveable_points = moveable_points
            
            self.cp_keys = list(cut_points.keys())
            self.cp_count = len(self.cp_keys)
            self.mp_count = len(moveable_points)
            
            self.actual_cut_points = None
            self.moveable_points_translation = None
        
            self.l1sel = None
            self.l2sel = None
            self.mvp1 = None
            self.mvp2 = None
            self.both_mvps = None
            
            self.newpoint = None
            self.substituted_connections = None
            
            self.new_points = None
            self.new_reachabilities = None
            
            
        def set_selectors (self, l1sel, l2sel):
            self.l1sel = l1sel
            self.l2sel = l2sel
            
            self.mvp1 = self.l1sel >= self.cp_count
            self.mvp2 = self.l2sel >= self.cp_count
            
            self.both_mvps = self.mvp1 and self.mvp2
    
    @classmethod
    def calculate_connections (cls, points, reachabilities):
        base_lines = [
                np.array([
                        points[p1],
                        points[p2]
                    ])
                for p1, p2 in reachabilities
            ]
        
        # (Reach Index, Reach Index) : Point
        cut_points = cls.get_cut_points(base_lines)
        return base_lines, cut_points
    
    @classmethod
    def get_cut_points (cls, base_lines):
        base_vectors = np.array(LinearAlgebra2D.get_vectors_from_linepoints_groups(base_lines))
        base_vectors = np.reshape(base_vectors, (-1, 2, 2))
        
        coefs = np.array([
                LinearAlgebra2D.get_line_crossing_coefs_for_lines(main_vector, base_vectors)
                for main_vector in base_vectors
            ])
        coefs = np.round(coefs, 12)
        coefs_valids = [
                np.where(np.all((main_coefs > 0) & (main_coefs < 1), axis=1))[0]
                for main_coefs in coefs
            ]
        
        reachability_pairs = {}
        
        for i, main_line_valid in enumerate(coefs_valids):
            for other in main_line_valid:       
                if i < other:
                    tup = (i, other)
                    
                    coef = coefs[i, other]
                    point = base_vectors[i, 0] + coef[0] * base_vectors[i, 1]
                    
                    reachability_pairs[tup] = point
        
        return reachability_pairs
    
    @classmethod
    def _get_selectors (cls, sv):
        distances = [
                    np.linalg.norm(sv.actual_cut_points[i] - sv.actual_cut_points[i+1:], axis=1)
                    for i in range(len(sv.actual_cut_points)-1)
            ]
        l1_mindists = [
                np.min(x)
                for x in distances
            ]
        l1sel = np.argmin(l1_mindists)
        l2sel = np.argmin(distances[l1sel]) + l1sel + 1
        
        sv.set_selectors(l1sel, l2sel)
    
    @classmethod
    def _get_basics (cls, sv):
        if len(sv.cp_keys) != 0:
            sv.actual_cut_points = np.array([
                    sv.cut_points[x]
                    for x in sv.cp_keys
                ])
            
            sv.moveables_inds = np.array(list(sv.moveable_points))
                
            if len(sv.moveables_inds) != 0:
                sv.moveable_points_translation = np.arange(len(sv.points))[sv.moveables_inds]
                sv.actual_cut_points = np.concatenate([
                    sv.actual_cut_points,
                    sv.points[sv.moveables_inds]
                ], axis=0)
            else:
                sv.moveable_points_translation = []
        else:
            sv.moveables_inds = np.array(list(sv.moveable_points))
            sv.moveable_points_translation = sv.moveables_inds
            sv.actual_cut_points = sv.points[sv.moveables_inds]
    
    @classmethod
    def _get_substitutables_and_median (cls, sv):
        if sv.mvp2 == False:
            cpk1 = sv.cp_keys[sv.l1sel]
            cp1 = sv.cut_points[cpk1]   
            
            cpk2 = sv.cp_keys[sv.l2sel]
            cp2 = sv.cut_points[cpk2]
            
            substituted_connections = set([
                    sv.reachabilities[x]
                    for y in [cpk1, cpk2]
                    for x in y
                ])
        else:    
            sv.l2sel = sv.moveable_points_translation[sv.l2sel - sv.cp_count]
            cp2 = sv.points[sv.l2sel]    
                   
            substituted_connections = set([
                    reachability
                    for reachability in sv.reachabilities
                    if sv.l2sel in reachability
                ])
            
            if sv.mvp1 == False:
                cpk1 = sv.cp_keys[sv.l1sel]
                cp1 = sv.cut_points[cpk1]   
                
                substituted_connections = substituted_connections.union(set([
                    sv.reachabilities[x]
                    for x in cpk1
                ]))
            else:
                sv.l1sel = sv.moveable_points_translation[sv.l1sel - sv.cp_count]
                cp1 = sv.points[sv.l1sel]  
                
                substituted_connections = substituted_connections.union(set([
                        reachability
                        for reachability in sv.reachabilities
                        if sv.l1sel in reachability
                    ]))
                
                if sv.l1sel > sv.l2sel:
                    x = sv.l2sel
                    sv.l2sel = sv.l1sel
                    sv.l1sel = x
                    
        sv.newpoint = np.mean([cp1, cp2], axis=0)
        sv.substituted_connections = substituted_connections
        
    @classmethod
    def _reconfigure_moveable_points (cls, sv):
        old_moveable_points = sv.moveable_points
        new_moveable_points = set()
        
        for old_moveable in old_moveable_points:
            if old_moveable == sv.l2sel:
                continue
            
            if old_moveable > sv.l2sel:
                old_moveable = old_moveable - 1
            
            new_moveable_points.add(old_moveable)
        
        sv.moveable_points = new_moveable_points
        
    @classmethod
    def _reconnect (cls, sv):
        if sv.mvp2 == False:
            con_index = len(sv.points)
            sv.new_points = np.append(sv.points, [sv.newpoint], axis=0)
            sv.moveable_points.add(con_index)
            
            new_reachabilities = set(sv.reachabilities) - sv.substituted_connections
            
            for s, t in sv.substituted_connections:
                new_reachabilities.add((s, con_index))
                new_reachabilities.add((t, con_index))
        else:
            if sv.both_mvps == False:
                con_index = sv.l2sel
                sv.new_points = sv.points
                sv.new_points[con_index] = sv.newpoint
                
                new_reachabilities = set(sv.reachabilities) - sv.substituted_connections
                
                for s, t in sv.substituted_connections:
                    new_reachabilities.add((s, con_index))
                    new_reachabilities.add((t, con_index))
            else:
                con_index = sv.l1sel
                sv.new_points = np.delete(sv.points, sv.l2sel, axis=0)
                sv.new_points[con_index] = sv.newpoint
                '''
                Indizes ueber l2sel um 1 reduzieren
                Indizes gleich l1sel gleich l1sel lassen
                
                Direkte Verbindung zwischen l1sel und l2sel nicht hinzufuegen
                '''
                
                new_reachabilities = set()
                
                for s, t in set(sv.reachabilities):
                    tup = np.sort([s, t])
                    sel = tup >= sv.l2sel
                    tup[sel] = tup[sel] - 1
                    
                    if tup[0] != tup[1]:
                        tup = tuple(tup)
                        new_reachabilities.add(tup)
                
                cls._reconfigure_moveable_points(sv)
                
        new_reachabilities = list(new_reachabilities)
        sv.new_reachabilities = new_reachabilities
    
    
    @classmethod
    def substitute_closest (cls, points, reachabilities, cut_points, moveable_points):
        sv = cls.SubstitutionVariables(points, reachabilities, cut_points, moveable_points)
        
        if sv.cp_count + sv.mp_count > 1:
            cls._get_basics (sv)
            cls._get_selectors(sv)
            cls._get_substitutables_and_median (sv)
            cls._reconnect (sv) 
            return sv.new_points, sv.new_reachabilities, sv.moveable_points
        else:
            return None, None, None
    
    @classmethod    
    def reduce_network_track (cls, points, reachabilities, iters):
        moveable_points = set()
        
        points_track = [np.copy(points)]
        reachabilities_track = [reachabilities]
        moveables = [set()]
        
        for i in range(iters):
            base_lines, cut_points = cls.calculate_connections (points, reachabilities)
            points, reachabilities, moveable_points = cls.substitute_closest (points, reachabilities, cut_points, moveable_points)
            
            print(i,"Changes:\n-Points:")
            pprint(points)
            print("-Reachabilities:")
            pprint(reachabilities)
            print("-Moveables:")
            pprint(moveable_points)
            
            
            if points is not None:
                points_track.append(np.copy(points))
                reachabilities_track.append(reachabilities)
                moveables.append(set(moveable_points))
            else:
                points_track.append(points_track[-1])
                reachabilities_track.append(reachabilities_track[-1])
                moveables.append(moveables[-1])  
        
        return points_track, reachabilities_track, moveables
    
    @classmethod
    def reduce_network (cls, points, reachabilities, iters):
        moveable_points = set()
        
        for _ in range(iters):
            base_lines, cut_points = cls.calculate_connections (points, reachabilities)
            new_points, new_reachabilities, new_moveable_points = cls.substitute_closest (points, reachabilities, cut_points, moveable_points)
            
            if new_points is None:
                break
            else:
                points = new_points
                reachabilities = new_reachabilities
                moveable_points = new_moveable_points
        
        return points, reachabilities
            
class PathFind ():
    @classmethod
    def _angle_raycast_point_inside_polygon (cls, point, normvec, target_vectors, polygon_normvecs, iteration, iteration_limit, cast_count, maxrot):
        if iteration == iteration_limit:
            return None
        
        rotations = np.linspace(-maxrot, maxrot, cast_count, dtype=np.float)
        
        mod_normvec = np.array([
                point,
                normvec
            ])
        rotated_normvecs = np.array([
                LinearAlgebra2D.rotate_vectors_direction_angle(mod_normvec, x)
                for x in rotations
            ])
        
        # POLYGON CHECK
        polyhit_coefs = np.array([
                LinearAlgebra2D.get_line_crossing_coefs_for_lines(x, polygon_normvecs)
                for x in rotated_normvecs
            ])
        
        
        sel = (polyhit_coefs[:,:,1] > 0) & (polyhit_coefs[:,:,1] < 1) & (polyhit_coefs[:,:,0] > 0)
        
        polyhit_args = np.array([
                np.argmin(x[s, 0])
                for x, s in zip(polyhit_coefs, sel)
            ])
        polyhit_normvecs = np.array([
                polygon_normvecs[s,2][arg]
                for s, arg in zip(sel, polyhit_args) 
            ])
        polyhit_coefs = np.array([
                x[s, 0][arg]
                for x, s, arg in zip(polyhit_coefs, sel, polyhit_args)
            ])
        
        # BASE CHECK
        basehit_coefs = np.array([
                LinearAlgebra2D.get_line_crossing_coefs_for_lines(x, target_vectors)
                for x in rotated_normvecs
            ])
        sel = (basehit_coefs[:,:,1] >= 0) & (basehit_coefs[:,:,1] < 1) & (basehit_coefs[:,:,0] > 0)
        basehit_coefs = np.array([
                np.min(x[s, 0])
                if len(x[s, 0]) != 0
                else np.inf
                for x, s in zip(basehit_coefs, sel)
            ])
        
        basehit_truth = polyhit_coefs > basehit_coefs
        
        paths = []
        
        for i in range(len(rotated_normvecs)):
            hitted_base = basehit_truth[i]
            
            if hitted_base:
                p = rotated_normvecs[i, 0] + rotated_normvecs[i, 1] * basehit_coefs[i]
                paths.append(np.array([point, p]))
            else:
                p = rotated_normvecs[i, 0] + rotated_normvecs[i, 1] * polyhit_coefs[i]
                n = -polyhit_normvecs[i]
                
                # List of lists (paths)
                subpaths = cls._angle_raycast_point_inside_polygon(p, n, target_vectors, polygon_normvecs, 
                                                             iteration+1, iteration_limit, cast_count, maxrot)
                
                if subpaths is None:
                    paths.append(None)
                else:
                    for subpath in subpaths:
                        if subpath is not None:
                            subpath = np.concatenate([[point], subpath], axis=0)
                            paths.append(subpath)
        return paths
            
    @classmethod
    def angle_raycast_point_inside_polygon (cls, point, normvec, target_vectors, polygon_normvecs, iteration_limit, cast_count, maxrot):
        paths = PathFind._angle_raycast_point_inside_polygon(point, normvec, target_vectors, polygon_normvecs, 0, iteration_limit, cast_count, maxrot)
        
        if paths is None:
            paths = []
        else:
            paths = [
                    x
                    for x in paths
                    if x is not None
                ]
        
        return paths
    
    @classmethod
    def _normal_raycast_point_inside_polygon (cls, point, normvec, target_vectors, polygon_normvecs, iteration, iteration_limit, coef_count):
        if iteration == iteration_limit:
            return None
        
        full_normvec = np.array([point, normvec])
        # print("Full normvec:\n"+str(full_normvec))
        new_normvec = LinearAlgebra2D.get_normal_vector(normvec)
        
        # (PolyNormvecs, 2)
        poly_maxcoef = LinearAlgebra2D.get_line_crossing_coefs_for_lines(full_normvec, polygon_normvecs)
        poly_maxcoef = poly_maxcoef[(poly_maxcoef[:,1] > 0) & (poly_maxcoef[:,1] < 1) & (poly_maxcoef[:,0] > 0)]
        # print("POLY MAXCOEF:\n"+str(poly_maxcoef))
        
        if len(poly_maxcoef) == 0:
            return None
            
        poly_maxcoef = np.min(poly_maxcoef[:,0])
        
        target_coefs = LinearAlgebra2D.get_line_crossing_coefs_for_lines(full_normvec, target_vectors)
        target_coefs = target_coefs[(target_coefs[:,1] > 0) & (target_coefs[:,1] < 1) & (target_coefs[:,0] > 0)]
        target_coefs = np.min(target_coefs[:,0]) if len(target_coefs) != 0 else np.inf
        
        if target_coefs <= poly_maxcoef:
            ret = [np.array([point, point + target_coefs * normvec])]
            return ret
        
        test_coefs = np.linspace(0, poly_maxcoef, coef_count+1, endpoint=False)[1:]
        
        paths = []
        
        new_iteration = iteration + 1
        
        for c in test_coefs:
            new_point = full_normvec[0] + c * full_normvec[1]
            
            subpaths = []
            current_subpaths = cls._normal_raycast_point_inside_polygon(new_point, new_normvec, target_vectors, 
                                                                        polygon_normvecs, new_iteration, iteration_limit, coef_count)
            
            if current_subpaths != None:
                subpaths.extend(current_subpaths)
                
            current_subpaths = cls._normal_raycast_point_inside_polygon(new_point, -new_normvec, target_vectors, 
                                                                        polygon_normvecs, new_iteration, iteration_limit, coef_count)
            
            if current_subpaths != None:
                subpaths.extend(current_subpaths)
            
            for subpath in subpaths:
                if subpath is not None:
                    subpath = np.concatenate([[point], subpath], axis=0)
                    paths.append(subpath)
                    
        return paths
    
    @classmethod
    def normal_raycast_point_inside_polygon (cls, point, normvec, target_vectors, polygon_normvecs, iteration_limit, cast_count):
        paths = PathFind._normal_raycast_point_inside_polygon(point, normvec, target_vectors, polygon_normvecs, 0, iteration_limit, cast_count)
        
        if paths is None:
            paths = []
        else:
            paths = [
                    x
                    for x in paths
                    if x is not None
                ]
        
        return paths
    
    @classmethod
    def get_shortest_path (cls, paths):
        length = [
                np.sum([
                    np.linalg.norm(p[i] - p[i+1])
                    for i in range(len(p) - 1)    
                ])
                for p in paths
            ]
        
        if len(length) == 0:
            return None
        else:
            shortest = np.argmin(length)
            return paths[shortest]
        
    @classmethod
    def get_paths_with_least_points (cls, paths):
        counts = np.array([
                len(x)
                for x in paths
            ])
        mincount = np.min(counts)
        sel = np.where(counts == mincount)[0]
        
        return [paths[x] for x in sel]
        
    @classmethod
    def _prettify_path_in_polygon (cls, path, target_index, segment_start_point, poly_normvecs, coefs):
        vector = np.array([
                segment_start_point,
                path[target_index] - segment_start_point
            ], dtype=np.float)
        
        cut_coefs = LinearAlgebra2D.get_line_crossing_coefs_for_lines(vector, poly_normvecs)
        sel = np.all((cut_coefs > 0) & (cut_coefs < 1), axis=1)
        cut_coefs = cut_coefs[sel]
        
        cut_coefs = np.inf if len(cut_coefs) == 0 else np.min(cut_coefs[:,0])
        
        if (target_index == (len(path) - 1)):
            if cut_coefs >= 1:
                '''
                print("CPoint: {:s} | Target: {:s} | Coef: {:f}".format(
                    str(segment_start_point),
                    str(path[target_index]),
                    cut_coefs
                ))
                '''
                return [[segment_start_point, path[target_index]]]
            else:
                return [None]
        
        coefs = coefs[coefs < cut_coefs]
        paths = []
        
        new_target_index = target_index+1
        
        for c in coefs:
            p = vector[0] + c * vector[1]
            
            subpaths = cls._prettify_path_in_polygon(path, new_target_index, p, poly_normvecs, coefs)
            
            for subpath in subpaths:
                if subpath is not None:
                    subpath = np.concatenate([[segment_start_point], subpath], axis=0)
                    paths.append(subpath)
            
        return paths
            
        
    @classmethod
    def prettify_path_in_polygon (cls, path, poly_normvecs, test_count):
        if len(path) == 2:
            return [np.array(path)]
        elif len(path) < 2:
            return None
        else:
            coefs = np.linspace(0, 1, test_count+1, endpoint=False)[1:]
            return cls._prettify_path_in_polygon(path, 1, path[0], poly_normvecs, coefs)
        
class RandomLinearAlgebra2D ():
    @classmethod
    def _generate_polygon (cls, base_directions, minbounds, maxbounds, angles, seglength, root, max_angle):
        points = []
        all_vectors = []
        
        spoint = root
        start_point = None
        stage = 0
        stage_change = False
        
        while True:
            base_direction = base_directions[stage]
            
            vec = np.array([spoint, base_direction])
            possible_vecs = np.array([
                    LinearAlgebra2D.rotate_vectors_direction_angle(vec, x)
                    for x in angles
                ])
            seglengths = np.tile(seglength, len(possible_vecs))
            possible_vecs = np.repeat(possible_vecs, len(seglength), axis=0)        
            
            target_points = possible_vecs[:,0] + np.repeat(seglengths[:,np.newaxis], 2, axis=1) * possible_vecs[:,1]
            possible_vecs[:,1] *= np.repeat(seglengths[:,np.newaxis], 2, axis=1)
            
            point_sel = np.all((target_points >= minbounds) & (target_points <= maxbounds), axis=1)
            
            if np.any(point_sel):
                if len(all_vectors) == 0:
                    coefs_sel = np.full(len(point_sel), True)
                else:
                    t = np.array(all_vectors)
                    
                    coefs = np.array([
                            LinearAlgebra2D.get_line_crossing_coefs_for_lines(x, t)
                            for x in possible_vecs
                        ])
                    coefs_sel = (coefs > 0) & (coefs <= 1)
                    coefs_sel = np.all(coefs_sel, axis=2)
                    coefs_sel = np.any(coefs_sel, axis=1)
                    coefs_sel = ~coefs_sel
                
                if stage_change:
                    if len(all_vectors) != 0:
                        angle_sel = LinearAlgebra2D.get_degrees_between_vectors(all_vectors[-1], possible_vecs)
                        angle_sel = angle_sel <= max_angle
                    else:       
                        angle_sel = np.full(len(point_sel), True)
                else:
                    angle_sel = np.full(len(point_sel), True)
                                     
                sel = np.where(point_sel & coefs_sel & angle_sel)[0]
                
                if len(sel) != 0:
                    sel = sel[np.random.randint(0, len(sel))]
                    point = target_points[sel]
                    vec = possible_vecs[sel]
                
                    if start_point is None:
                        start_point = point
                        
                    spoint = point
                    points.append(spoint)
                    all_vectors.append(vec)
                    
                    stage_change = True
                else:
                    spoint = root
                    start_point = None
                    stage = 0
                    points = []
                    all_vectors = []    
                    stage_change = False
            else:
                stage += 1
                
                stage_change = True
                
                if stage == 4:
                    break
                    
        points.append(start_point)
        
        points = np.array(points)
        return points
    
    @classmethod
    def generate_polygon_in_bounds (cls, root, seglength, required_dimensions, angles):    
        angles = np.unique(
                np.concatenate([
                        angles,
                        -angles
                    ], axis=0)
            )
        
        max_angle = np.max(angles)
        
        minbounds = root
        maxbounds = root + required_dimensions
        
        rolled = np.roll(np.arange(0, 4), np.random.randint(-3, 4))
        first = rolled[0]
        
        base_directions = np.array([
                [0, 1],
                [1, 0],
                [0, -1],
                [-1, 0]
            ])
        base_directions = base_directions[rolled]
        
        if first == 1:
            root = np.array([root[0], maxbounds[1] - root[1]])
        elif first == 2:
            root = np.array([maxbounds[0] - root[0], maxbounds[1] - root[1]])
        elif first == 3:
            root = np.array([maxbounds[0] - root[0], root[1]])
        
        while True:
            points = cls._generate_polygon (base_directions, minbounds, maxbounds, angles, seglength, root, max_angle)
            vecs = LinearAlgebra2D.get_vectors_from_linepoints(points)
            
            coefs = np.array([
                    LinearAlgebra2D.get_line_crossing_coefs_for_lines(x, vecs)
                    for x in vecs
                ])
            coefs = (coefs > 0) & (coefs < 1)
            coefs = np.all(coefs, axis=2)
            coefs = np.any(coefs)
            
            if coefs == False:
                break
            
        return points
        
class Combinatorics ():
    @classmethod
    def cartesian (cls, *iterables):
        count = len(iterables)
        
        iterables = [
                x[:,np.newaxis]
                for x in iterables
            ]
        
        cart = iterables[0]
        
        for i in range(1, count):            
            current = iterables[i]
        
            cart_shape = cart.shape
            current_shape = current.shape
            
            cart = np.concatenate([
                    np.repeat(cart, current_shape[0], axis=0),
                    np.tile(current, (cart_shape[0], 1))
                ], axis=1)
            
        return cart
    
class Polynomial ():
    class PolynomialFunction ():
        def __init__ (self, coefficients):
            self._coefficients = coefficients
            self._degree = len(self._coefficients)
            self.__exponents = np.flip(np.arange(0, self._degree))
    
        def degree (self):
            return self._degree
        
        def coefficients (self):
            return self._coefficients
        
        def __call__ (self, x):
            x = np.repeat(x[:,np.newaxis], self._degree, axis=1)
            
            x = x ** self.__exponents
            y = np.sum(self._coefficients * x, axis=1)
            return y
            
    @classmethod
    def create (cls, x, y):
        coefs = LinearEquationSystems.create_polynomial_function(x, y)
        return cls.PolynomialFunction(coefs)