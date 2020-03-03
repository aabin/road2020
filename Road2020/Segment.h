//
// Created by joey on 2019/10/6.
//

#ifndef RV_SEGMENT_H
#define RV_SEGMENT_H

#include "structs.h"
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/algorithm.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Cartesian.h>
#include <CGAL/bounding_box.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_2_algorithms.h>
#include <CGAL/Projection_traits_xy_3.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Projection_traits_xy_3<K> Gt;
typedef CGAL::Delaunay_triangulation_2<Gt> Delaunay;
typedef CGAL::Polygon_2<K> Polygon_2;
typedef Delaunay::Finite_faces_iterator Finite_faces_iterator;
typedef Delaunay::Face_circulator Face_circulator;
typedef Delaunay::Finite_vertices_iterator Finite_vertices_iterator;
typedef Delaunay::Finite_edges_iterator Finite_edges_iterator;
typedef Delaunay::Face_handle Face_handle;
typedef Delaunay::Vertex_handle Vertex_handle;
typedef Delaunay::Locate_type Locate_type;
typedef K::Vector_3 Vector_3;
typedef K::FT FT;
typedef K::Segment_2  Segment_CGAL;
typedef K::Segment_3 Segment_3;
typedef CGAL::Alpha_shape_vertex_base_2<K> Vb;
typedef CGAL::Cartesian<FT> KC;
typedef CGAL::Alpha_shape_face_base_2<K>  Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds> Triangulation_2;
typedef CGAL::Alpha_shape_2<Triangulation_2>  Alpha_shape_2;
typedef Alpha_shape_2::Alpha_shape_edges_iterator Alpha_shape_edges_iterator;
typedef Alpha_shape_2::Alpha_shape_vertices_iterator Alpha_shape_vertices_iterator;


using std::vector;

class Segment {

public:
    void segmentToGrid(const vector<Point3DI> &localPts, const Point3D &center, const float &size, const Point3D &minB, const Point3D &maxB, vector<vector<Grid> > &output);

    void ccltGridInfo(vector<vector<Grid> > &chess);

    void regionGrow(const vector<vector<Grid> > &chess, const float &planar_threshold, const float &angle_threshold, const float &height_threshold, vector<vector<uint> > &regions);

    //fill holes and sort regions by size
    void adjustRegions(vector<vector<uint> > &regions, const vector<vector<Grid> > &chess);

    void chooseRegions(const vector<vector<uint> > &regions, const ushort &largest_label, vector<vector<uint> > &alpha);

    void refineRoad(const vector<vector<Grid> > &chess, vector<vector<uint> > &alpha, const float &distance_threshold, vector<Point3DI> &refine_points);

    void extractBoundary(const float &alpha, const vector<Point3DI> &cloud, const vector<vector<Grid> > &chess, vector<zBoundary> &boundarys, Point3D minB, vector<Point3DI> &edges);

private:

};

#endif //RV_SEGMENT_H
