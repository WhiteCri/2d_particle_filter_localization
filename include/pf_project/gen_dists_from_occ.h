#include <queue>
#include <cmath>
#include <vector>
#include <nav_msgs/OccupancyGrid.h>
#include <ros/ros.h>

/* the structure of the code is from amcl : http://wiki.ros.org/amcl */
#define MAP_INDEX(width, x, y) (y*width + x)
    
using std::vector;

static int width;
static vector<double>* dists_ptr;



class CellData
{
public:
    unsigned int i_, j_; 
    unsigned int src_i_, src_j_;
};

class CachedDistanceMap
{
public:
    CachedDistanceMap(double scale, double max_dist) : 
        scale_(scale), max_dist_(max_dist)
    { 
        cell_radius_ = max_dist / scale;
        distances_.resize(cell_radius_+2);
        for(int i=0; i<=cell_radius_+1; i++)
        {
            distances_[i] = vector<double>(cell_radius_+2);
            for(int j=0; j<=cell_radius_+1; j++){ 
                distances_[i][j] = std::sqrt(i*i + j*j);
            }
        }
    }
    vector<vector<double>> distances_;
    double scale_;
    double max_dist_;
    int cell_radius_;
};

bool operator<(const CellData& a, const CellData& b)
{
    return dists_ptr->at(MAP_INDEX(width, a.i_, a.j_)) > dists_ptr->at(MAP_INDEX(width, b.i_, b.j_));
}

void enqueue(vector<double>& dists, int i, int j,
         int src_i, int src_j,
         std::priority_queue<CellData>& Q,
         CachedDistanceMap& cdm,
         vector<bool>& marked,
         int map_width, float resolution)
{
    if(marked[MAP_INDEX(map_width, i, j)])
        return;

    int di = abs(i - src_i);
    int dj = abs(j - src_j);
    double distance = cdm.distances_[di][dj];

    if(distance > cdm.cell_radius_)
        return;

    dists[MAP_INDEX(map_width, i, j)] = distance * resolution;

    CellData cell;
    cell.i_ = i;
    cell.j_ = j;
    cell.src_i_ = src_i;
    cell.src_j_ = src_j;

    Q.push(cell);

    marked[MAP_INDEX(map_width, i, j)] = true;
}

// Update the cspace distance values
void find_minimum_distances(vector<double>& dists, const nav_msgs::OccupancyGrid& occ, double max_occ_dist)
{
    dists.resize(occ.info.width * occ.info.height);

    width = occ.info.width;
    dists_ptr = &dists;


    std::priority_queue<CellData> Q;

    vector<bool> marked(occ.info.width * occ.info.height);
    std::fill(marked.begin(), marked.end(), false);

    CachedDistanceMap cdm(occ.info.resolution, max_occ_dist);

    // Enqueue all the obstacle cells

    CellData cell;
    for(int i=0; i<occ.info.width; i++){
        cell.src_i_ = cell.i_ = i;
        for(int j=0; j<occ.info.height; j++)
        {
            if(occ.data[MAP_INDEX(occ.info.width, i, j)] == 100) // obj
            {
                dists[MAP_INDEX(occ.info.width, i, j)] = 0.0;
                cell.src_j_ = cell.j_ = j;
                marked[MAP_INDEX(occ.info.width, i, j)] = true;
                Q.push(cell);
            }
            else dists[MAP_INDEX(occ.info.width, i, j)] = max_occ_dist;
        }
    }

    while(!Q.empty()){
        CellData current_cell = Q.top();
        if(current_cell.i_ > 0)
            enqueue(dists, current_cell.i_-1, current_cell.j_, 
                current_cell.src_i_, current_cell.src_j_,
                Q, cdm, marked, occ.info.width, occ.info.resolution);
        if(current_cell.j_ > 0)
            enqueue(dists, current_cell.i_, current_cell.j_-1, 
                current_cell.src_i_, current_cell.src_j_,
                Q, cdm, marked, occ.info.width, occ.info.resolution);
        if((int)current_cell.i_ < occ.info.width - 1)
            enqueue(dists, current_cell.i_+1, current_cell.j_, 
                current_cell.src_i_, current_cell.src_j_,
                Q, cdm, marked, occ.info.width, occ.info.resolution);
        if((int)current_cell.j_ < occ.info.height - 1)
            enqueue(dists, current_cell.i_, current_cell.j_+1, 
                current_cell.src_i_, current_cell.src_j_,
                Q, cdm, marked, occ.info.width, occ.info.resolution);

        Q.pop();
    }
    dists_ptr = nullptr;
}

nav_msgs::OccupancyGrid genOccupancyGridWithDists
    (const vector<double>& dists, const nav_msgs::OccupancyGrid& occ){
    static nav_msgs::OccupancyGrid dists_occ;
    auto dists_copy = dists;

    //normalize dists from 0 to 100
    auto minmax = std::minmax_element(std::begin(dists_copy), std::end(dists_copy));
    auto min = *minmax.first;
    auto max = *minmax.second;
    std::for_each(dists_copy.begin(), dists_copy.end(), [=](auto& v){v = 1.0*(v-min)/(max-min)*100;});

    dists_occ = occ;
    auto cur = dists_copy.begin();
    std::for_each(dists_occ.data.begin(), dists_occ.data.end(), [&](auto& v){v = int(*cur); cur++;});

    return dists_occ;
}