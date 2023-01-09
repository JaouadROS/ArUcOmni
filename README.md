# ArUcOmni
-------------------------------------------------------------------
Detection of highly reliable fiducial markers in panoramic images. This is an adaptation of ArUco 3.0.12 for panoramic cameras such as catadioptric and fisheye sensors. Two main steps were adapted are:
* Homography estimation to get the canonical form of the potential marker.
* Warping.
* Pose estimation.
## Installation

Follow the steps:

```bash
#Clone the repository
git clone https://github.com/JaouadROS/ArUcOmni

#Create build folder
cd ArUcOmni && mkdir build && cd build

#Finally compile the library
cmake .. && make -j4
```

## Usage
After compilation, it is time to run the code.
```bash
#cd to utils folder
cd ArUcOmni/build/utils

# and run aruco_test to show the inputs needed
./aruco_test

# Here is an example of running aruco_test using our open-source dataset (check the next section to download the dataset)
./aruco_test ArUcOmni_dataset/acquired_images/catadioptric/3_plane_marker/image_20.png -c intrinsics_barreto_catadioptric.yaml -s 24.4 -d OPENCV_4X4_1000.dict
```
As an output, you should get the input image with marker detection, as well as the projection of a 3D frame XYZ that demonstrates that the estimated pose is visually correct.

![ArUcOmni example](https://i.imgur.com/piwHjCu.png)

## Open source dataset
To download our open source dataset, please visit [the link](https://home.mis.u-picardie.fr/~g-caron/en/index.php?page=8#ArUcOmni
).

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Reference
If you use this repo please cite the following work:

Hajjami, J., Caracotte, J., Caron, G., & Napoleon, T. (2020). ArUcOmni: detection of highly reliable fiducial markers in panoramic images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 634-635).

Here is the [PDF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w38/Hajjami_ArUcOmni_Detection_of_Highly_Reliable_Fiducial_Markers_in_Panoramic_Images_CVPRW_2020_paper.pdf) version.

