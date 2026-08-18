// Copyright 2023 Tier IV, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <rclcpp/rclcpp.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

#include <cmath>
#include <yaml-cpp/yaml.h>

using Trajectory = autoware_auto_planning_msgs::msg::Trajectory;
using TrajectoryPoint = autoware_auto_planning_msgs::msg::TrajectoryPoint;

class CSVToTrajectory : public rclcpp::Node
{
public:
  CSVToTrajectory() : Node("csv_to_trajectory_node")
  {
    const auto rb_qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
    pub_ = this->create_publisher<Trajectory>("trajectory", rb_qos);
    set_parameter_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&CSVToTrajectory::on_parameter_event, this, std::placeholders::_1));


    declare_parameter("csv_path", "");
    z_= declare_parameter<float>("z");
    std::string csv_path = get_parameter("csv_path").as_string();
    
    // declare mpc_config file
    const std::string mpc_config_path = get_parameter("mpc_config_path").as_string();

    // if no mpc_config_path specified
    if (!mpc_config_path.empty()) {
      csv_path = get_csv_path_from_mpc_config(mpc_config_path);
    }

    if (csv_path.empty()) {
      RCLCPP_ERROR(get_logger(), "CSV path is not specified");
      return;
    }
    
    if (!loadCSVTrajectory(csv_path)) {
      RCLCPP_ERROR(get_logger(), "Failed to load CSV file: %s", csv_path.c_str());
      return;
    }

    current_csv_path_ = csv_path;
    
    RCLCPP_INFO(get_logger(), "Loaded trajectory from CSV: %s with %zu points",
                csv_path.c_str(), csv_trajectory_.points.size());

    timer_ = rclcpp::create_timer(
      this, get_clock(), std::chrono::seconds(1),
      std::bind(&CSVToTrajectory::publish_trajectory, this));

  }

private:
  bool loadCSVTrajectory(const std::string & csv_path)
  {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
      return false;
    }
    
    std::string line;
    std::getline(file, line);
    
    csv_trajectory_.header.stamp = this->now();
    csv_trajectory_.header.frame_id = "map";

    csv_trajectory_.points.clear();
    
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string token;
      std::vector<double> values;
      
      while (std::getline(ss, token, ',')) {
        values.push_back(std::stod(token));
      }

      TrajectoryPoint point;  
      if (values.size() == 8) {
        // raceline: x, y, z, x_quat, y_quat, z_quat, w_quat, speed
        point.pose.position.x = values[0];
        point.pose.position.y = values[1];
        point.pose.orientation.x = values[3];
        point.pose.orientation.y = values[4];
        point.pose.orientation.z = values[5];
        point.pose.orientation.w = values[6];
        point.longitudinal_velocity_mps = values[7];
      } else if (values.size() == 7) {
        // MPC: s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2
        // Note: in mpc
        // x: values[1]
        // y: values[2]
        // psi: values[3]
        // x_quat, y_quat = 0 (no pitch and roll)
        // z_quat = sin(psi/2)
        // w_quat = cos(psi/2)
        // v = values[5]
        point.pose.position.x = values[1];
        point.pose.position.y = values[2];
        const double psi = values[3];
        point.pose.orientation.x = 0.0;
        point.pose.orientation.y = 0.0;
        point.pose.orientation.z = std::sin(psi * 0.5);
        point.pose.orientation.w = std::cos(psi * 0.5);
        point.longitudinal_velocity_mps = values[5];
      } else {
        RCLCPP_WARN(get_logger(), "Invalid CSV line format, expected 7 or 8 values");
        continue;
      }

      point.pose.position.z = z_;
      point.lateral_velocity_mps = 0.0;
      point.acceleration_mps2 = 0.0;
      point.heading_rate_rps = 0.0;
      csv_trajectory_.points.push_back(point);
    }
    return !csv_trajectory_.points.empty();
  }
  
  void publish_trajectory()
  {
    if (csv_trajectory_.points.empty()) {
      RCLCPP_WARN(get_logger(), "No trajectory points to publish");
      return;
    }
    
    csv_trajectory_.header.stamp = this->now();
    pub_->publish(csv_trajectory_);
    RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(), 60000 /*ms*/, "Published trajectory with %zu points", csv_trajectory_.points.size());
  }

  rcl_interfaces::msg::SetParametersResult on_parameter_event(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "";

    for (const auto & param : parameters) {
      if (param.get_name() == "csv_path") {
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
          std::string new_csv_path = param.as_string();
          // new_csv_pathがFileSystemのパスであることを確認
          if (!std::filesystem::exists(new_csv_path)) {
            RCLCPP_ERROR(get_logger(), "File does not exist: '%s'", new_csv_path.c_str());
            result.successful = false;
            result.reason = "File does not exist.";
            continue;
          }

          if (new_csv_path != current_csv_path_) {
            RCLCPP_INFO(get_logger(), "csv_path parameter changed from '%s' to '%s'", 
                        current_csv_path_.c_str(), new_csv_path.c_str());
            
            // 新しいCSVファイルの読み込みを試みる
            if (loadCSVTrajectory(new_csv_path)) {
              current_csv_path_ = new_csv_path;
              RCLCPP_INFO(get_logger(), "Successfully loaded new trajectory from CSV: %s with %zu points", 
                          current_csv_path_.c_str(), csv_trajectory_.points.size());
            } else {
              RCLCPP_ERROR(get_logger(), "Failed to load new CSV file: %s. Keeping old trajectory.", new_csv_path.c_str());
              result.successful = false;
              result.reason = "Failed to load new CSV file.";
            }
          }
        } else {
          RCLCPP_WARN(get_logger(), "Parameter 'csv_path' received with wrong type. Expected string.");
          result.successful = false;
          result.reason = "Invalid type for csv_path parameter.";
        }
      } else if (param.get_name() == "mpc_config_path") {
      if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
        const std::string new_config = param.as_string();
        const std::string new_csv = get_csv_path_from_mpc_config(new_config);
        if (new_csv.empty()) {
          result.successful = false;
          result.reason = "Failed to get csv_path from mpc config.";
          continue;
        }
        if (new_csv != current_csv_path_ && loadCSVTrajectory(new_csv)) {
          current_csv_path_ = new_csv;
          RCLCPP_INFO(get_logger(), "Loaded trajectory from MPC config CSV: %s with %zu points",
                      current_csv_path_.c_str(), csv_trajectory_.points.size());
        } else if (new_csv != current_csv_path_) {
          RCLCPP_ERROR(get_logger(), "Failed to load new CSV file: %s", new_csv.c_str());
          result.successful = false;
          result.reason = "Failed to load new CSV file.";
        }
      }
      } else if (param.get_name() == "z") {
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE || param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
          z_ = static_cast<float>(param.as_double());
          RCLCPP_INFO(get_logger(), "z parameter changed to %f", z_);
        } else {
          RCLCPP_WARN(get_logger(), "Parameter 'z' received with wrong type. Expected float/double.");
          result.successful = false;
          result.reason = "Invalid type for z parameter.";
        }
      }
    }
    return result;
  }

  // get csv path from mpc config
  std::string get_csv_path_from_mpc_config(const std::string & mpc_config_path)
  {
    try {
      const YAML::Node config = YAML::LoadFile(mpc_config_path);
      const YAML::Node reference_path = config["reference_path"];
      if (!reference_path || !reference_path["csv_path"]) {
        RCLCPP_ERROR(
          get_logger(), "'reference_path.csv_path' not found in %s", mpc_config_path.c_str());
        return "";
      }
      const std::string rel_csv = reference_path["csv_path"].as<std::string>();
      if (rel_csv.empty()) {
        RCLCPP_ERROR(
          get_logger(), "'reference_path.csv_path' is empty in %s", mpc_config_path.c_str());
        return "";
      }
      // config.yaml lives in <share>/config/, reference paths are relative to <share>/
      const auto share_dir = std::filesystem::path(mpc_config_path).parent_path().parent_path();
      const std::string full_csv = (share_dir / rel_csv).string();
      RCLCPP_INFO(get_logger(), "Using MPC reference path: %s", full_csv.c_str());
      return full_csv;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Failed to parse MPC config '%s': %s",
                   mpc_config_path.c_str(), e.what());
      return "";
    }
  }
  
  rclcpp::Publisher<Trajectory>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  Trajectory csv_trajectory_;
  float z_;
  std::string current_csv_path_;
  OnSetParametersCallbackHandle::SharedPtr set_parameter_callback_handle_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CSVToTrajectory>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
