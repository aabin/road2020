#include <string>
#include <iostream>
#include "Segment.h"
#include "lasio.h"
#include "structs.h"
#include "Markings.h"
using std::string;
using std::cout;
using std::endl;

int main()
{
    Time timer;
    //input parameters
    //const string test_file_path  = "E:\\BaiduNetdiskDownload\\hwsegment\\curve_ground.las";
	//const string road_cloud_path = "E:\\BaiduNetdiskDownload\\hwsegment\\curve_road.las";
	//const string edge_cloud_path = "E:\\BaiduNetdiskDownload\\hwsegment\\curve_edge.las";

	const string test_file_path = "D:/2020exp/cross1_ground.las";
	const string road_cloud_path = "road.las";
	const string edge_cloud_path = "edge.las";

    float grid_size, planar_threshold, angle_threshold, height_threshold, refine_distance_threshold, alpha_shape;

    unsigned int type; // 0-ALS, 1-MLS

    type = 0;

    if (type == 0)
    {
        grid_size = 1.5;
        planar_threshold = 0.006;
        angle_threshold = 10;
        height_threshold = 0.05;
        refine_distance_threshold = 0.035;
        alpha_shape = 5;
    }
    else if (type == 1)
    {
        grid_size = 0.5;
        planar_threshold = 0.0004;
        angle_threshold = 5.0;
        height_threshold = 0.025;
        refine_distance_threshold = 0.025;
        alpha_shape = 5;
    }

	cout << "planar_threshold(4-6mm): " << endl;
	cin >> angle_threshold;
	planar_threshold;

	cout << "angle_threshold(5-20degree): " << endl;
	cin >> angle_threshold;

	cout << "height_threshold(1-10cm): " << endl;
	cin >> height_threshold;
	height_threshold /= 100;

    //medium files
    vector<Point3DI> local_point_repo, refined_cloud, edge;
    Point3D global_shift, local_minB, local_maxB;
    vector<vector<Grid> > chess;
    vector<vector<uint> > regions, alpha;
    vector<zBoundary> boundarys;

    Lasio lasio;
    lasio.readLasFile(test_file_path, local_point_repo, global_shift, local_minB, local_maxB);

	Segment segment;
    segment.segmentToGrid(local_point_repo, global_shift, grid_size, local_minB, local_maxB, chess);

    //release local-coord point set
    vector<Point3DI>().swap(local_point_repo);

    segment.ccltGridInfo(chess);

    segment.regionGrow(chess, planar_threshold, angle_threshold, height_threshold, regions);

    segment.adjustRegions(regions, chess);

    vector<Point3DI> opoint, rpoint;
	vector<Point3DI> m_int, s_int;
    for (int i = 0; i < chess.size(); ++i) {
        for (int j = 0; j < chess[0].size(); ++j) {
            for (int k = 0; k < chess[i][j].point_number; ++k) {
                Point3DI cpt = chess[i][j].vPoints[k];
                cpt.inten = chess[i][j].info.residual * 10000;
                opoint.push_back(cpt);
                cpt.inten = regions[i][j];
                rpoint.push_back(cpt);
				
				cpt.inten = chess[i][j].info.average.inten;
				m_int.push_back(cpt);
				cpt.inten = chess[i][j].info.stv_i;
				s_int.push_back(cpt);
            }
        }
    }
    lasio.writeLasFile("display.las", opoint, global_shift);
    lasio.writeLasFile("regions.las", rpoint, global_shift);

	lasio.writeLasFile("m_int.las", m_int, global_shift);
	lasio.writeLasFile("s_int.las", s_int, global_shift);

    //manually input the largest label
    ushort largest_label = 0;
    do{
        cout << "#input the largest region label:  ";
        std::cin >> largest_label;
    }
    while(largest_label <= 0 || largest_label >= 10);

    segment.chooseRegions(regions, largest_label, alpha);

    segment.refineRoad(chess, alpha, refine_distance_threshold, refined_cloud);

    segment.extractBoundary(alpha_shape, refined_cloud, chess, boundarys, local_minB, edge);

    lasio.writeLasFile(road_cloud_path, refined_cloud, global_shift);
    lasio.writeLasFile(edge_cloud_path, edge, global_shift);

	//Lasio lasio_m;
	//vector<Point3DI> road_point;
	//Point3D global_shift_m, local_minB_m, local_maxB_m;
	//lasio_m.readLasFile("road.las", road_point, global_shift_m, local_minB_m, local_maxB_m);

	Markings markings;
	vector<Point3DI> sali = markings.ccltSaliencyPoint(refined_cloud);
	lasio.writeLasFile("sali.las", sali, global_shift);

	vector<Point3DI> mrk, cluster;
	// ≈∑ Ωæ€¿‡
	vector<PointCloud<PointXYZ> > clusters_pcl = markings.extractMarkingPointsEUDistance(sali, 500, mrk, cluster);
	lasio.writeLasFile("marking.las", mrk, global_shift);
	lasio.writeLasFile("cluster.las", cluster, global_shift);
	vector<Point3DI> inl, otl;
	// ransac…∏—°inliers
	markings.filterOutliersByRansac(clusters_pcl, otl, inl);
	lasio.writeLasFile("outliers.las", otl, global_shift);
	lasio.writeLasFile("inliers.las", inl, global_shift);

    cout << timer.timeSpent() << endl;
	system("pause");
    return 0;
}