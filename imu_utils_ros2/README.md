# imu_utils_ros2
## ROS2 version of [imu_utils](https://github.com/gaowenliang/imu_utils)

## Prerequisites
- **System**
  - ROS2 (Jazzy or Humble distributions recommended)
- **Libraries**
  - OpenCV
  - Ceres Solver
  - Eigen3

## Usage
- Clone this [repository](https://github.com/DarrenCaiimu_utils_ros2).
- Build
  - ```Bash
    cd <path_to_imu_utils_ros2>
    source /opt/ros/<distro>/setup.bash
    colcon build
    ```
- Analyze the Allan Variance for the IMU data.
  - Record you IMU data bag
    ```Bash
    ros2 bag record <your_imu_topic>
    ```
    Collect the data while the IMU is Stationary, with a two hours duration.
  - Generate your launch xml<br>
    Take [A3.xml](launch/A3.xml) as example
    ```xml
    <launch>
      <node pkg="imu_utils" exec="imu_allan" name="imu_allan">
        <param name="imu_topic" value="/djiros/imu"/>
        <param name="imu_name" value="A3"/>
        <param name="data_save_path" value="$(find-pkg-share imu_utils)/../../../../data/"/>
        <param name="max_time_min" value="120"/>
        <param name="max_cluster" value="100"/>
      </node>
    </launch>
    ```
  - Open two terminals, one to run imu_tools, one to play your rosbag2
    ```Bash
    cd <path_to_imu_utils_ros2>
    source /opt/ros/<distro>/setup.bash
    source <path_to_imu_utils_ros2>/install/setup.bash
    ros2 launch imu_utils <your_launch_xml>
    ```
    ```Bash
    source /opt/ros/<distro>/setup.bash
    ros2 bag play <path_to_your_rosbag2> -r 200
    ```
   - The result should be like this:
     ```yaml
     %YAML:1.0
     ---
     type: IMU
     name: A3
     Gyr:
        unit: " rad/s"
        avg-axis:
           gyr_n: 1.0922514245261136e-04
           gyr_w: 3.0407639130588035e-05
        x-axis:
           gyr_n: 1.1712350336249066e-04
           gyr_w: 3.6395480767077183e-05
        y-axis:
           gyr_n: 1.0957986890514727e-04
           gyr_w: 3.1881226725483150e-05
        z-axis:
           gyr_n: 1.0097205509019619e-04
           gyr_w: 2.2946209899203768e-05
     Acc:
        unit: " m/s^2"
        avg-axis:
           acc_n: 1.4268671413807624e-03
           acc_w: 6.3698303145662391e-04
        x-axis:
           acc_n: 1.2219441138445304e-03
           acc_w: 5.3750668357445538e-04
        y-axis:
           acc_n: 1.2229466839249080e-03
           acc_w: 6.0460048331990467e-04
        z-axis:
           acc_n: 1.8357106263728492e-03
           acc_w: 7.6884192747551168e-04
     ```

## Try dataset **DJI A3: `400Hz`**
Download link: [`百度网盘`](https://pan.baidu.com/s/1jJYg8R0 "DJI A3")

Install [rosbags-convert](https://gitlab.com/ternaris/rosbags) to convert the bag file recorded at ROS1 to rosbag2
```Bash
pip install rosbags
rosbags-convert imu_A3.bag
```

<br><br>

# Original Readme:
<br>

# imu_utils

A ROS package tool to analyze the IMU performance. C++ version of Allan Variance Tool. 
The figures are drawn by Matlab, in `scripts`.

Actually, just analyze the Allan Variance for the IMU data. Collect the data while the IMU is Stationary, with a two hours duration.

## refrence

Refrence technical report: [`Allan Variance: Noise Analysis for Gyroscopes`](http://cache.freescale.com/files/sensors/doc/app_note/AN5087.pdf "Allan Variance: Noise Analysis for Gyroscopes"), [`vectornav gyroscope`](https://www.vectornav.com/support/library/gyroscope "vectornav gyroscope") and 
[`An introduction to inertial navigation`](http://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-696.html "An introduction to inertial navigation").

```
Woodman, O.J., 2007. An introduction to inertial navigation (No. UCAM-CL-TR-696). University of Cambridge, Computer Laboratory.
```
Refrence Matlab code: [`GyroAllan`](https://github.com/XinLiGitHub/GyroAllan "GyroAllan")

## IMU Noise Values

Parameter | YAML element | Symbol | Units
--- | --- | --- | ---
Gyroscope "white noise" | `gyr_n` | $$\sigma_g$$ | $$\frac{rad}{s}\frac{1}{\sqrt{Hz}}$$
Accelerometer "white noise" | `acc_n` | $$\sigma_a$$ | $$\frac{m}{s^2}\frac{1}{\sqrt{Hz}}$$
Gyroscope "bias Instability" | `gyr_w` | $$\sigma_{bg}$$ | $$\frac{rad}{s}\sqrt{Hz}$$
Accelerometer "bias Instability" | `acc_w` | $$\sigma_{ba}$$  | $$\frac{m}{s^2}\sqrt{Hz}$$

* White noise is at tau=1;

* Bias Instability is around the minimum;

(according to technical report: [`Allan Variance: Noise Analysis for Gyroscopes`](http://cache.freescale.com/files/sensors/doc/app_note/AN5087.pdf "Allan Variance: Noise Analysis for Gyroscopes"))

## sample test

<img src="imu_utils/figure/gyr.jpg">
<img src="imu_utils/figure/acc.jpg">

* blue  : Vi-Sensor, ADIS16448, `200Hz`
* red   : 3dm-Gx4, `500Hz`
* green : DJI-A3, `400Hz`
* black : DJI-N3, `400Hz`
* circle : xsens-MTI-100, `100Hz`

## How to build and run?

### to build

```
sudo apt-get install libdw-dev
```

* download required [`code_utils`](https://github.com/gaowenliang/code_utils "code_utils");

* put the ROS package `imu_utils` and `code_utils` into your workspace, usually named `catkin_ws`;

* cd to your workspace, build with `catkin_make`;


### to run

* collect the data while the IMU is Stationary, with a two hours duration;

* (or) play rosbag dataset;

```
 rosbag play -r 200 imu_A3.bag
```

* roslaunch the rosnode;

```
roslaunch imu_utils A3.launch
```

Be careful of your roslaunch file:

```
<launch>
    <node pkg="imu_utils" type="imu_an" name="imu_an" output="screen">
        <param name="imu_topic" type="string" value= "/djiros/imu"/>
        <param name="imu_name" type="string" value= "A3"/>
        <param name="data_save_path" type="string" value= "$(find imu_utils)/data/"/>
        <param name="max_time_min" type="int" value= "120"/>
        <param name="max_cluster" type="int" value= "100"/>
    </node>
</launch>
```

### sample output:

```
type: IMU
name: A3
Gyr:
   unit: " rad/s"
   avg-axis:
      gyr_n: 1.0351286977809465e-04
      gyr_w: 2.9438676109223402e-05
   x-axis:
      gyr_n: 1.0312669892959053e-04
      gyr_w: 3.3765827874234673e-05
   y-axis:
      gyr_n: 1.0787155789128671e-04
      gyr_w: 3.1970693666470835e-05
   z-axis:
      gyr_n: 9.9540352513406743e-05
      gyr_w: 2.2579506786964707e-05
Acc:
   unit: " m/s^2"
   avg-axis:
      acc_n: 1.3985049290745563e-03
      acc_w: 6.3249251509920116e-04
   x-axis:
      acc_n: 1.1687799474421937e-03
      acc_w: 5.3044554054317266e-04
   y-axis:
      acc_n: 1.2050535351630543e-03
      acc_w: 6.0281218607825414e-04
   z-axis:
      acc_n: 1.8216813046184213e-03
      acc_w: 7.6421981867617645e-04
```

## dataset

DJI A3: `400Hz`

Download link: [`百度网盘`](https://pan.baidu.com/s/1jJYg8R0 "DJI A3")


DJI A3: `400Hz`

Download link: [`百度网盘`](https://pan.baidu.com/s/1pLXGqx1 "DJI N3")


ADIS16448: `200Hz`
 
Download link:[`百度网盘`](https://pan.baidu.com/s/1dGd0mn3 "ADIS16448")

3dM-GX4: `500Hz`

Download link:[`百度网盘`](https://pan.baidu.com/s/1ggcan9D "GX4")

xsens-MTI-100: `100Hz`

Download link:[`百度网盘`](https://pan.baidu.com/s/1i64xkgP "MTI-100")
