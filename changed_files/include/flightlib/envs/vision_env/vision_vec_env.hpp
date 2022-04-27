#pragma once

// yaml cpp
#include <yaml-cpp/yaml.h>
	
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <random>

#include "flightlib/envs/vec_env_base.hpp"
#include "flightlib/envs/vision_env/vision_env.hpp"
#include "flightlib/objects/unity_object.hpp"

namespace flightlib {

template<typename EnvBaseName>
class VisionVecEnv : public VecEnvBase<EnvBaseName> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VisionVecEnv();
  VisionVecEnv(const std::string& cfg, const bool from_file = true);
  VisionVecEnv(const YAML::Node& cfg_node);
  ~VisionVecEnv();

  using VecEnvBase<EnvBaseName>::configEnv;

  bool reset(Ref<MatrixRowMajor<>> obs) override;
  bool reset(Ref<MatrixRowMajor<>> obs, bool random);
  bool step(Ref<MatrixRowMajor<>> act, Ref<MatrixRowMajor<>> obs,
            Ref<MatrixRowMajor<>> reward, Ref<BoolVector<>> done,
            Ref<MatrixRowMajor<>> extra_info) override;


  bool getQuadAct(Ref<MatrixRowMajor<>> quadact);
  bool getQuadState(Ref<MatrixRowMajor<>> quadstate);
  inline std::vector<std::string> getRewardNames(void) {
    return this->envs_[0]->getRewardNames();
  };

  bool configDynamicObjects(const std::string &yaml_file, std::string obstacle_cfg_path_, std::vector<std::shared_ptr<UnityObject>> &dynamic_objects_);
  bool configStaticObjects(const std::string &yaml_file, std::vector<std::shared_ptr<UnityObject>> &static_objects_);
  bool all_done = false;
  bool change_env_on_reset = false;
  
  std::vector<std::vector<std::shared_ptr<UnityObject>>> all_envs_dynamic_objects_;
  std::vector<std::vector<std::shared_ptr<UnityObject>>> all_envs_static_objects_;

 private:
  void perAgentStep(int agent_id, Ref<MatrixRowMajor<>> act,
                    Ref<MatrixRowMajor<>> obs,
                    Ref<MatrixRowMajor<>> reward_units, Ref<BoolVector<>> done,
                    Ref<MatrixRowMajor<>> extra_info) override;
  // yaml configurations
  bool random_reset_;
  //
  YAML::Node cfg_;
  std::vector<std::string> reward_names_;
};

}  // namespace flightlib
