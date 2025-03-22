This is a **forked repository**. To run it, use `run_multi_times.sh` or `run.sh` in directory `/examples` so as to effectively run `1014-example.yaml`(some parameters in the file, such as path, need to be modified, and [tmuxp](https://github.com/tmux-python/tmuxp) is required):
```shell
# To run Kimera-Multi for multiple times
bash run_multi_times.sh

# To run Kimera-Multi
bash run.sh

# To run KimeraVIO for single robot
bash run.sh 0
```

Here are main changes or new features(still updating):

1. Add trajectory evaluation and visualization by `evo` (In `/examples/evo_*` )
2. Add an option to run single robot simultaneously
3. Add a [branch](https://github.com/RonghaiHe/Kimera-Multi/tree/ubuntu18) to deploy it in ubuntu 18.04
4. Deploy one-key start (In `/examples/slam_front`)
5. Update `kimera-multi.drawio` file (In `/images`)
6. Add evaluation for loop closures (Find GT poses, output corresponding images, statistics, etc. in `/evaluation`)
7. Modify for less randomized by setting random seed in a thread-safe way
  - By setting `deterministic_random_number_generator` true in `/params/D455/Pipeline.flags`
  - By setting `ransac_randomize` 0 in `/params/D455/LcdParams.yaml`
  - By setting `ransac_randomize` 0 in `/params/visual_loopclosure_Jackal.yaml` in [Kimera-Distributed](https://github.com/RonghaiHe/Kimera-Distributed)

<details>
  <summary>See Changelog for more details:</summary>
  2025/02
  - Add files for running EuRoC dataset (`euroc_single.yaml`, `run_euroc.sh` and `kimera_vio_ros_euroc_multi.launch`)

  2025/01
  - Add more evaluation for loop closures (In `/evalution`):
    - Count monocular inliers and stereo inliers w.r.t. threshold (`10` & `5`) (`analyze_inliers.py`)
  - Modify for less randomized:
    - Modify the repository [Kimera-VIO-ROS](https://github.com/RonghaiHe/Kimera-VIO-ROS) to read the pipeline flags for determinical random seed
    - Modify repositories [dpgo_ros](https://github.com/RonghaiHe/dpgo_ros), [dpgo](https://github.com/RonghaiHe/dpgo) and [Kimera-Distributed](https://github.com/RonghaiHe/Kimera-Distributed) to set random seed for random number generators, then add the input in `1014-example.yaml` (In `/examples`)
    - Modify the repository [OpenGV](https://github.com/RonghaiHe/opengv) to be the thread-safe random seed using `thread_local`
    - Modify the repository [Kimera-Multi-LCD](https://github.com/RonghaiHe/Kimera-Multi-LCD) to judge if using the random seed randomizedly


  2024/12
  - Deploy one-key start (In `/example/slam_front`)
  - Update `kimera-multi.drawio` file (In `/images`)
  - Add evaluation for loop closures (In `/evalution`):
    - Find GT poses and output them as files (`lc_result.py`)
    - Output images from loop closures whose distances > 30 meters (`extract_lc_images.py` & `extract_lc_images.sh`)

  2024/10
  - Modify files for running single robot and logging trajectory.
  - Add trajectory comparison and visualization (In `/examples/evo_*`)
  - Automatically run it multiple times based on `run_multi_times.sh`

  2024/09:
  - Change output format(TUM) in this [commit](https://github.com/MIT-SPARK/Kimera-Distributed/commit/c4c8e51462d23b72413548e19eadf9ffefcdc4b7) in [Kimera-Distributed](https://github.com/MIT-SPARK/Kimera-Distributed/)
  - Add evo evaluation in examples

  2024/08:
  - Add new branch [ubuntu18](https://github.com/RonghaiHe/Kimera-Multi/tree/ubuntu18) to deploy in Ubuntu 18.04

  2024/05/03:
  - Add `examples/del_poses_files.sh` to retain the latest and oldest `kimera_distributed_poses_xxx.csv`.
  - Modify 1014-example.yaml to run `examples/del_poses_files.sh` simutaneously
</details>

TODO:
- [x] Modify codes about output format to run `evo`
- [x] Add `.yaml` file to run euroc dataset. A [reference](https://github.com/MIT-SPARK/Kimera-Multi/issues/9).
- [ ] Asynchronous operation for evaluation by `evo`
- [x] Modify `lc_result.py` to run parallelly

Blogs about the [installation](https://blog.csdn.net/Ben__Ho/article/details/137350202)([complement](https://blog.csdn.net/Ben__Ho/article/details/142219177)) and the [running](https://blog.csdn.net/Ben__Ho/article/details/138171249) in Simplified Chinese.

---
<div align="center">
  <a href="https://mit.edu/sparklab/">
    <img align="left" src="images/spark_logo.png" height="80" alt="sparklab">
  </a>
  <a href="https://mit.edu">
    <img align="center" src="images/mit_logo.png" height="80" alt="mit">
  </a>
  <a href="http://acl.mit.edu/">
    <img align="right" src="images/acl_logo.jpeg" height="80" alt="acl">
  </a>
</div>

# Kimera-Multi

Kimera-Multi is a multi-robot system that 
(i) is robust and capable of identifying and rejecting incorrect inter and intra-robot loop closures resulting from perceptual aliasing, 
(ii) is fully distributed and only relies on local (peer-to-peer) communication to achieve distributed localization and mapping,
and (iii) builds a globally consistent metric-semantic 3D mesh model of the environment in real-time, where faces of the mesh are annotated with semantic labels. Kimera-Multi is implemented by a team of robots equipped with visual-inertial sensors. Each robot builds a local trajectory estimate and a local mesh using Kimera. When communication is available, robots initiate a distributed place recognition and robust pose graph optimization protocol based on a novel distributed graduated non-convexity algorithm. The proposed protocol allows the robots to improve their local trajectory estimates by leveraging inter-robot loop closures while being robust to outliers. Finally, each robot uses its improved trajectory estimate to correct the local mesh using mesh deformation techniques.

<p align="center">
    <a href="https://youtu.be/G8PktlQ82uw">
    <img src="images/kimera_multi.png" alt="Kimera-Multi">
    </a>
</p>

## Installation

**Note**
The experiments described by the authors in the paper were done on Ubuntu 18.04 and ROS Melodic.
System has since been updated and now assumes Ubuntu 20.04 and ROS Noetic.
```
# Create workspace
mkdir -p catkin_ws/src
cd catkin_ws/src/
git clone git@github.com:MIT-SPARK/Kimera-Multi.git kimera_multi

# If you do not have these dependencies already
sudo bash kimera_multi/install/dependencies.sh

# For full install
vcs import . --input kimera_multi/kimera_multi.repos --recursive

cd ..
# Configure build options and build!
catkin config -a --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGTSAM_TANGENT_PREINTEGRATION=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF -DOPENGV_BUILD_WITH_MARCH_NATIVE=OFF
catkin build --continue -s
```

## System Architecture & Breakdown
<p align="center">
    <a href="https://arxiv.org/abs/2106.14386">
    <img src="images/system_arch.png" alt="Kimera-Multi System">
    </a>
</p>

For more in depth details about the system, we point the reader to our [paper](https://arxiv.org/abs/2106.14386).
Each robot runs an onboard system using the Robot Operating System (ROS). 
Inter-robot communication is performed in a fully peer-to-peer manner using a lightweight communication layer on top of the UDP protocol using the Remote Topic Manager. 
Kimera-VIO and Kimera-Semantics provide the odometric pose estimates and a reconstructed 3D mesh. 
The distributed front-end detects inter-robot loop closures by communicating visual Bag-of-Words (BoW) vectors and selected keyframes that contain keypoints and descriptors for geometric verification. 
The front-end is also responsible for incorporating the odometry and loop closures into a coarsened pose graph. 
The distributed back-end periodically optimizes the coarse pose graph using robust distributed optimization. Lastly, the optimized trajectory is used by each robot to correct its local 3D mesh.

### Module-to-Repository Directory
- [Kimera-Semantics](https://github.com/MIT-SPARK/Kimera-Semantics)
- [Kimera-VIO](https://github.com/MIT-SPARK/Kimera-VIO)
- Distributed Front-End: [Kimera-Distributed](https://github.com/MIT-SPARK/Kimera-Distributed) which handles the communication protocol and pose graph coarsening and [Kimera-Multi-LCD](https://github.com/MIT-SPARK/Kimera-Multi-LCD) which contains the functionalities for Bag-of-Words matching and keyframe registration.
- Remote Topic Manager: Coming Soon (Still in approval process).
- Robust Distributed PGO: [DPGO](https://github.com/mit-acl/dpgo) and [DPGO-ROS](https://github.com/mit-acl/dpgo_ros).
- Local Mesh Optimization: [Kimera-PGMO](https://github.com/MIT-SPARK/Kimera-PGMO).

## Examples & Usage

To test Kimera-Multi on a single machine, we provided an example in the examples folder.
First, install [tmuxp](https://github.com/tmux-python/tmuxp),
then, download the [Campus-Outdoor data](https://github.com/MIT-SPARK/Kimera-Multi-Data).
Lastly, run the following to launch all the processes for all 6 robots:
```bash
CATKIN_WS=<path-to-catkin-ws> DATA_PATH=<path-to-campus-outdoor-data-folder> LOG_DIR=<path-to-log-folder> tmuxp load 1014-example.yaml
```
Note that this example only uses a single ROS master, and will most likely not work with intermittent communication.
To run with separate ROS masters on separate machines, we will need to use the Remote Topic Manager,
which is currently under-going the approval process for release.
We will provide an additional example once the module is public.

## Docker

A docker image for deploying Kimera-Multi is also provided. Please refer to the README in the `docker` subdirectory for details.

## Citation

If you found Kimera-Multi to be useful, we would really appreciate if you could cite our work:

- [1] Y. Chang, Y. Tian, J. P. How and L. Carlone, "Kimera-Multi: a System for Distributed Multi-Robot Metric-Semantic Simultaneous Localization and Mapping," 2021 IEEE International Conference on Robotics and Automation (ICRA), Xi'an, China, 2021, pp. 11210-11218, doi: 10.1109/ICRA48506.2021.9561090.

```bibtex
@INPROCEEDINGS{chang21icra_kimeramulti,
  author={Chang, Yun and Tian, Yulun and How, Jonathan P. and Carlone, Luca},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Kimera-Multi: a System for Distributed Multi-Robot Metric-Semantic Simultaneous Localization and Mapping}, 
  year={2021},
  volume={},
  number={},
  pages={11210-11218},
  doi={10.1109/ICRA48506.2021.9561090}
}

```

- [2] Y. Tian, Y. Chang, F. Herrera Arias, C. Nieto-Granda, J. P. How and L. Carlone, "Kimera-Multi: Robust, Distributed, Dense Metric-Semantic SLAM for Multi-Robot Systems," in IEEE Transactions on Robotics, vol. 38, no. 4, pp. 2022-2038, Aug. 2022, doi: 10.1109/TRO.2021.3137751.
```bibtex
@ARTICLE{tian22tro_kimeramulti,
  author={Tian, Yulun and Chang, Yun and Herrera Arias, Fernando and Nieto-Granda, Carlos and How, Jonathan P. and Carlone, Luca},
  journal={IEEE Transactions on Robotics}, 
  title={Kimera-Multi: Robust, Distributed, Dense Metric-Semantic SLAM for Multi-Robot Systems}, 
  year={2022},
  volume={38},
  number={4},
  pages={2022-2038},
  doi={10.1109/TRO.2021.3137751}
}

```

- [3] Y. Tian, Y. Chang, L. Quang, A. Schang, C. Nieto-Granda, J. P. How, and L. Carlone, "Resilient and Distributed Multi-Robot Visual SLAM: Datasets, Experiments, and Lessons Learned," arXiv preprint arXiv:2304.04362, 2023.
```bibtex
@ARTICLE{tian23arxiv_kimeramultiexperiments,
  author={Yulun Tian and Yun Chang and Long Quang and Arthur Schang and Carlos Nieto-Granda and Jonathan P. How and Luca Carlone},
  title={Resilient and Distributed Multi-Robot Visual SLAM: Datasets, Experiments, and Lessons Learned},
  year={2023},
  eprint={2304.04362},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

## Datasets

We have also released some [real-life datasets](https://github.com/MIT-SPARK/Kimera-Multi-Data) collected on 8 robots on the MIT campus.

## Acknowledgements
Kimera-Multi was supported in part by ARL [DCIST](https://www.dcist.org), [ONR](https://www.nre.navy.mil/), and [MathWorks](https://www.mathworks.com/).
