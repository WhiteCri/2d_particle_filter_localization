#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <thread>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <algorithm>
#include <tf/transform_broadcaster.h>
#include <queue>
#include <pf_project/gen_dists_from_occ.h>
#include <eigen3/Eigen/Dense>
#include <random>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Twist.h>

enum {OCC_OBJ, OCC_FREE, OCC_UNKNOWN};
static constexpr int8_t OCC_OBJ_VALUE=100;
static constexpr int8_t OCC_FREE_VALUE=0;
static constexpr int8_t OCC_UNKNOWN_VALUE=-1;
static constexpr int N_STATES = 6; //x, x', y, y', z, z'
static constexpr int N_INPUTS = 3;

inline int REAL_XY_TO_MAP_INDEX(const nav_msgs::OccupancyGrid& map, double x, double y){
    
    auto& pos_ori = map.info.origin.position;
    double x_min = pos_ori.x; // origin represents left bottom point
    double y_min = pos_ori.y;

    int x_idx = std::floor((x - x_min) / map.info.resolution);
    int y_idx = std::floor((y - y_min) / map.info.resolution);

    int ret = y_idx*map.info.width + x_idx;
    if (ret < 0 || map.data.size() < ret) return -1;
    else return ret;
}
inline int INT_XY_TO_MAP_INDEX(const nav_msgs::OccupancyGrid& map, int x, int y){
    int ret = y*map.info.width + x;
    if (ret < 0 || map.data.size() < ret) return -1;
    else return ret;
}   

template <typename T>
void load_param(ros::NodeHandle& nh, std::string name, T& v){
    if (!nh.getParam(name, v)) throw std::runtime_error(std::string() + "param " + name + " is not set");
}

class LikelihoodViewer{
public:
    LikelihoodViewer() : map_ok(false){

        //pub, sub
        map_sub = nh.subscribe("/map", 10, &LikelihoodViewer::mapCB, this);
        odom_sub = nh.subscribe("/odom", 10, &LikelihoodViewer::odomCB, this);
        laser_sub = nh.subscribe("/scan", 10, &LikelihoodViewer::laserScanCB, this);
        cmd_sub = nh.subscribe("/cmd_vel", 10, &LikelihoodViewer::cmdCB, this);
        
        pose_ary_pub = nh.advertise<geometry_msgs::PoseArray>("/pose_array", 10);
        preprocessed_dist_map_pub = nh.advertise<nav_msgs::OccupancyGrid>("/preprocessed_dist_map", 1);
        likelihood_field_map_pub = nh.advertise<nav_msgs::OccupancyGrid>("/likelihood_field", 1);
        estimated_odom_pub = nh.advertise<nav_msgs::Odometry>("estimated_odom", 1);

        //load param
        load_param(nh, "likelihood_field_particle_filter/visualize_rate", visualize_rate);
        load_param(nh, "likelihood_field_particle_filter/z_hit", z_hit);
        load_param(nh, "likelihood_field_particle_filter/z_max", z_max);
        load_param(nh, "likelihood_field_particle_filter/z_random", z_random);
        load_param(nh, "likelihood_field_particle_filter/sigma_hit", sigma_hit);
        load_param(nh, "likelihood_field_particle_filter/max_occ_dist", max_occ_dist);
        load_param(nh, "likelihood_field_particle_filter/n_particles", n_particles);
        load_param(nh, "likelihood_field_particle_filter/initial_x", initial_x);
        load_param(nh, "likelihood_field_particle_filter/initial_y", initial_y);
        load_param(nh, "likelihood_field_particle_filter/initial_th", initial_th);
        load_param(nh, "likelihood_field_particle_filter/state_sigmas", state_sigmas);// must have same length with N_STATES
        if (state_sigmas.size() != N_STATES) throw std::runtime_error("plz set state sigmas well");

        for(int i = 0 ; i < N_STATES; ++i)
            state_noise_distributions.push_back(std::normal_distribution<double>(0, state_sigmas[i]*state_sigmas[i]));
        init_pf(); //must be called after subscribing odom
        
        //create thread
        std::thread(&LikelihoodViewer::run, this).detach();
    }

    void init_pf(){
        particles = Eigen::MatrixXd(6, n_particles);
        /*
            x1  x2  x3...
            x1' x2' x3'...
            y1  y2  y3...
            y1' y2' y3'...
            th1 th2 th3...
            th1'th2'th3'...
        */
        for(int i = 0 ; i < n_particles; ++i){
            particles(0, i) = initial_x + state_noise_distributions[0](rd_generator); 
            particles(1, i) = 0         + state_noise_distributions[1](rd_generator);
            particles(2, i) = initial_y + state_noise_distributions[2](rd_generator);
            particles(3, i) = 0         + state_noise_distributions[3](rd_generator);
            particles(4, i) = initial_th+ state_noise_distributions[4](rd_generator);
            particles(5, i) = 0         + state_noise_distributions[5](rd_generator);
        }
        weights.resize(n_particles);
        std::fill(weights.begin(), weights.end(), 1.0/n_particles);
    }
    void publishParticlePosition(){
        static geometry_msgs::PoseArray pose_ary;
        pose_ary.poses.resize(n_particles);
        pose_ary.header.stamp = ros::Time::now();
        pose_ary.header.frame_id = "map";
        for(int i = 0 ; i < n_particles; ++i){
            pose_ary.poses[i].position.x = particles(0, i);
            pose_ary.poses[i].position.y = particles(2, i);
            pose_ary.poses[i].orientation = tf::createQuaternionMsgFromYaw(particles(4, i));
        }
        pose_ary_pub.publish(pose_ary);
    }

    void mapCB(const nav_msgs::OccupancyGridConstPtr& ptr){
        map = *ptr;
        genLikelihoodField();
        map_ok = true;
        ROS_WARN("map received!");
    }

    void odomCB(const nav_msgs::OdometryConstPtr& ptr){
        odom = *ptr;
        odom_ok = true;
    }

    void laserScanCB(const sensor_msgs::LaserScanConstPtr& ptr){
        scan = *ptr;
        scan_ok = true;
    }

    void cmdCB(const geometry_msgs::TwistConstPtr& ptr){
        cmd_input = *ptr;
    }

    void publishEstimatedOdom(){
        Eigen::VectorXd state = Eigen::VectorXd::Zero(6);
        for(int i = 0 ; i < n_particles; ++i)
            state += weights[i] * particles.col(i);
        
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(state(0), state(2), 0.0));
        tf::Quaternion q;
        q.setRPY(0, 0, state(4));
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "base_link"));

        // odometry
        static nav_msgs::Odometry odom;
        odom.header.frame_id = "map";
        odom.header.stamp = ros::Time::now();
        odom.child_frame_id = "base_link";

    }

    int get_gridtype_from_xy(double x, double y){
        auto& pos_ori = map.info.origin.position;
        double x_min = pos_ori.x; // origin represents left bottom point
        double y_min = pos_ori.y;

        int x_idx = std::floor((x - x_min) / map.info.resolution);
        int y_idx = std::floor((y - y_min) / map.info.resolution);

        if (x_idx < 0) return OCC_UNKNOWN;
        if (x_idx >= map.info.width) return OCC_UNKNOWN;
        if (y_idx < 0) return OCC_UNKNOWN;
        if (y_idx >= map.info.height) return OCC_UNKNOWN;

        int cell_type = map.data[y_idx*map.info.width + x_idx];
        switch(cell_type){
        case OCC_OBJ_VALUE: return OCC_OBJ;
        case OCC_FREE_VALUE: return OCC_FREE;
        case OCC_UNKNOWN_VALUE: return OCC_UNKNOWN;
        default : 
            ROS_ERROR("invalid cell type %d in function [%s]", cell_type, __func__); 
            return OCC_UNKNOWN;
        }
    }

    void genLikelihoodField(){ 
        /* step 1 : gen closest distance map from occupancy map */
        std::vector<double> dists(map.info.width*map.info.height);
        find_minimum_distances(dists, map, max_occ_dist);

        preprocessed_dist_map = genOccupancyGridWithDists(dists, map);
        preprocessed_dist_map_pub.publish(preprocessed_dist_map);

        /* step 2 : gen likelihood map */
        likelihood_field.resize(dists.size());
        auto cur = dists.begin();
        std::for_each(likelihood_field.begin(), likelihood_field.end(), [&](auto& v){
            v = 1 / (sigma_hit * std::sqrt(2*M_PI)) * std::exp(-0.5*(*cur/sigma_hit)*(*cur/sigma_hit));
            cur++;
        });

        //gen occmap for lh map visualizing
        //scaling from 0 ~ 100
        std::vector<double> likelihood_field_copy = likelihood_field;
        auto minmax = std::minmax_element(likelihood_field_copy.begin(), likelihood_field_copy.end());
        auto min = *minmax.first;
        auto max = *minmax.second;
        std::for_each(likelihood_field_copy.begin(), likelihood_field_copy.end(), [=](auto& v){v = 1.0*(v-min)/(max-min)*100;});

        auto cur2 = likelihood_field_copy.begin();
        likelihood_field_map = map;
        std::for_each(likelihood_field_map.data.begin(), likelihood_field_map.data.end(), 
            [&](auto& v){if (v != OCC_UNKNOWN_VALUE) v = int(*cur2); cur2++;});
        likelihood_field_map_pub.publish(likelihood_field_map);
    }

    void run(){
        ros::Rate r(visualize_rate);
        
        while(map_ok==false) {
            ROS_WARN("map is not advertised....");
            ros::Rate(1).sleep();
        }

        while(odom_ok==false) {
            ROS_WARN("odom is not advertised....");
            ros::Rate(1).sleep();
        }

        while(scan_ok==false) {
            ROS_WARN("scan is not advertised....");
            ros::Rate(1).sleep();
        }

        double x_sensor = 0; //sensor is identical to the robot
        double y_sensor = 0;
        double th_sensor = 0;

        std::uniform_real_distribution<double> uniform_distribution(0.0,1.0); //random number generator for resampling
        while(ros::ok()){
            /* Algorithm particle filter using likelihood_field laser model*/
            ROS_INFO("-----------------------------------------");
            
            /* step 1 : update states */
            //helper functions to calc matrix A and B, which are used when calculating dynamics
            //batch processing
            static auto mat_A = [](double dt){
                Eigen::MatrixXd A(N_STATES, N_STATES);
                A << 1, dt/2, 0, 0, 0, 0,
                    0, 1, 0, 0, 0, 0, 
                    0, 0, 1, dt/2, 0, 0,
                    0, 0, 0, 1, 0, 0,
                    0, 0, 0, 0, 1, dt/2,
                    0, 0, 0, 0, 0, 1;
                return A; 
            };
            static auto mat_B = [](double dt){
                Eigen::MatrixXd B(N_STATES, N_INPUTS);
                B <<    dt/2,   0,  0,
                        1   ,   0,  0,
                        0   ,dt/2,  0,
                        0   ,   1,  0,
                        0   ,   0,dt/2,
                        0   ,   0,  1;
                return B;
            };
            
            Eigen::MatrixXd input_mat(N_INPUTS, n_particles); 
            Eigen::MatrixXd noise_mat(N_STATES, n_particles);
            for(int i = 0 ; i < n_particles; ++i){
                input_mat(0, i) = cmd_input.linear.x * std::cos(particles(4, i));
                input_mat(1, i) = cmd_input.linear.x * std::sin(particles(4, i));
                input_mat(0, i) = particles(5, i);

                noise_mat(0, i) = state_noise_distributions[0](rd_generator);
                noise_mat(1, i) = state_noise_distributions[1](rd_generator);
                noise_mat(2, i) = state_noise_distributions[2](rd_generator);
                noise_mat(3, i) = state_noise_distributions[3](rd_generator);
                noise_mat(4, i) = state_noise_distributions[4](rd_generator);
                noise_mat(5, i) = state_noise_distributions[5](rd_generator);
            }
            particles = mat_A(1.0/visualize_rate)*particles + mat_B(1.0/visualize_rate)*input_mat + noise_mat;

            /* step 2 : calculate pdf of each particle*/
            static std::vector<double> pdfs(n_particles);
            for(int particle_idx = 0; particle_idx < n_particles; ++particle_idx){
                double pdf = 1;
                double normalizer;
                for(size_t scan_idx = 0 ; scan_idx < scan.ranges.size(); ++scan_idx){
                    //skip if inf
                    if (std::isinf(scan.ranges[scan_idx])) continue;

                    //extract states that is needed
                    const auto& cur_particle = particles.col(particle_idx);
                    double x_p  = cur_particle(0);
                    double y_p  = cur_particle(2);
                    double th_p = cur_particle(4);
                    
                    //cur yaw of laser
                    double yaw_beam = scan.angle_min + scan_idx*scan.angle_increment;

                    //calculate beam's position(LINE 5, 6 in the algorithm)
                    double beam_x = x_p
                        + x_sensor*std::cos(th_p) 
                        - y_sensor*std::sin(th_p)
                        + scan.ranges[scan_idx]*std::cos(th_p + yaw_beam);
                    double beam_y = y_p
                        + y_sensor*std::cos(th_p) 
                        + x_sensor*std::sin(th_p)
                        + scan.ranges[scan_idx]*std::sin(th_p + yaw_beam);
                    
                    //calculate pdf with likelihood_field_range_finder_model
                    int map_idx = REAL_XY_TO_MAP_INDEX(map, beam_x, beam_y);
                    if (map_idx == -1) pdf *= z_max; //beam position out of map
                    //else if (map.data[map_idx] == OCC_UNKNOWN_VALUE) pdf *= 1/z_max; // beam position is unknown
                    else pdf *= (z_hit * likelihood_field[map_idx] + z_random/z_max);
                }
                if (particle_idx == 0) normalizer = pdf; //prevent overflow
                pdf /= normalizer;
                pdfs[particle_idx] = pdf;
                weights[particle_idx] *=pdf; 
            }
            /* step 3 : normalize weights */
            double weights_total = std::accumulate(weights.begin(), weights.end(), 0.0);
            ROS_INFO("weights total : %lf", weights_total);
            std::for_each(weights.begin(), weights.end(), [=](auto& v){
                v /= weights_total;
            });
            publishEstimatedOdom();            

            /* step 4 : resample weights and particles. equations are from lecture note */
            std::vector<double> weights_acc(n_particles);
            auto weights_acc_iter = weights_acc.begin();
            *weights_acc_iter++ = weights[0];
            for(size_t i = 1; i < n_particles - 1; ++i){
                *weights_acc_iter = *(weights_acc_iter - 1) + weights[i];
                weights_acc_iter++;
            }
            *weights_acc_iter = 1;

            Eigen::MatrixXd particles_new(N_STATES, n_particles);
            std::vector<double> u(n_particles);
            u[0] = uniform_distribution(rd_generator);
            u[0] /= n_particles;
            size_t i = 0;
            for(size_t j = 0; j < n_particles; ++j){
                u[j] = u[0] + 1.0 * j/n_particles;
                while(u[j] > weights_acc[i] && (i < n_particles-1)) i++;
                particles_new.col(j) = particles.col(i);
                weights[j] = 1.0 / n_particles;
            }
            particles = particles_new;


            publishParticlePosition();
            r.sleep();
        }
    }

    void printOccupancyGridData(){ /* only for debug */
        ROS_INFO("----------------------------------");
        ROS_INFO("stamp, frame_id : %lf %s", map.header.stamp.toSec(), map.header.frame_id.c_str());
        ROS_INFO("map_load_time : %lf", map.info.map_load_time.toSec());
        ROS_INFO("resolution : %f", map.info.resolution);
        ROS_INFO("width : %u", map.info.width);
        ROS_INFO("height : %u", map.info.height);
        auto& pos = map.info.origin.position;
        ROS_INFO("pose : %lf %lf %lf", pos.x, pos.y, pos.z);

        double x = -1;
        for (double y = -10; y < 10; y +=0.05)
            ROS_INFO("%.2lf, %.2lf : %s ", x, y, get_gridtype_from_xy(x, y)==OCC_OBJ ? "OBJ" : "NOT_OBJ");
    }

    void printLaserData(){
        ROS_INFO("----------------------------------");
        ROS_INFO("angle_min :           %f", scan.angle_min);
        ROS_INFO("angle_max :           %f", scan.angle_max);
        ROS_INFO("angle_increment :     %f", scan.angle_increment);
        ROS_INFO("time_increment :      %f", scan.time_increment);
        ROS_INFO("scan_time :           %f", scan.scan_time);
        ROS_INFO("range_min :           %f", scan.range_min);
        ROS_INFO("range_max :           %f\n", scan.range_max);
        
        ROS_INFO("len : %lu", scan.ranges.size());
        ROS_INFO("scan : %f", scan.ranges[0]);
        ROS_INFO("scan : %f", scan.ranges[45]);
        ROS_INFO("scan : %f", scan.ranges[90]);
        ROS_INFO("scan : %f", scan.ranges[135]);
        ROS_INFO("scan : %f", scan.ranges[180]);
        ROS_INFO("scan : %f", scan.ranges[225]);
        ROS_INFO("scan : %f", scan.ranges[270]);
        ROS_INFO("scan : %f", scan.ranges[315]);
        ROS_INFO("scan : %f", scan.ranges[359]);
        ROS_INFO("isinf : %d", std::isinf(scan.ranges[359]));
    }

private:
    ros::Publisher estimated_odom_pub, preprocessed_dist_map_pub, likelihood_field_map_pub, pose_ary_pub;
    ros::Subscriber map_sub, odom_sub, laser_sub, cmd_sub;
    ros::NodeHandle nh;

    nav_msgs::OccupancyGrid map, preprocessed_dist_map, likelihood_field_map;
    nav_msgs::Odometry odom;
    sensor_msgs::LaserScan scan;
    int visualize_rate;
    bool map_ok, odom_ok, scan_ok;
    geometry_msgs::Twist cmd_input;
    
    //likelihood field parameters
    double z_max, z_random, z_hit, max_occ_dist, sigma_hit;
    std::vector<double> likelihood_field;

    //particle filter parameters [x, x', y, y', th, th']
    int n_particles;
    std::vector<double> noise;
    Eigen::MatrixXd particles;
    double initial_x, initial_y, initial_th;
    std::vector<double> state_sigmas;
    std::vector<double> weights;
    tf::TransformBroadcaster br;

    std::default_random_engine rd_generator;
    std::vector<std::normal_distribution<double>> state_noise_distributions;
};

int main(int argc, char *argv[]){
    ros::init(argc, argv, "likelihood_field_particle_filter");
    LikelihoodViewer l;

    ros::spin();
}