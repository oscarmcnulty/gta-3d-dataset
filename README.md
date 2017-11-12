# GTA-3D Dataset
A dataset of 2D imagery, 3D point cloud data, and 3D vehicle bounding box labels all generated using the Grand Theft Auto 5 game engine. The dataset contains image and depth map data captured at 1680x1050 resolution and oriented 3D bounding box labels of all vehicles. It is 55GB in total.

![alt text](https://raw.githubusercontent.com/oscarmcnulty/gta-3d-dataset/master/3fd50f7b-658b-4ef4-bb17-dfc1f287def8_00000134.jpg)

![alt text](https://raw.githubusercontent.com/oscarmcnulty/gta-3d-dataset/master/3fd50f7b-658b-4ef4-bb17-dfc1f287def8_00000134_preds.png)

## Using the dataset
Helper classes to consume the data with are provided in `gta.py`. To read a single example use
```
from gta import GTAData

filename = 'data/gta_test/3fd50f7b-658b-4ef4-bb17-dfc1f287def8_00000819'
data = GTAData(filename)

img = data.load_rgb() # 1680 x 1050 x 3 ndarray of pixel color intensities
print(len(data.vehicles)) # Number of vehicles in the scene
print(data.vehicles[0].get_bbox_oriented_birdseye()) # Coordinates of the oriented birdseye bounding box of a vehicle in the scene
print(data.vehicles[0].get_bbox_2d()) # Coordinates of a bounding box in the image plane for a vehicle in the scene
```
An example 3D point cloud visualisation with bounding boxes is shown in `test_vis.py`

## Getting the dataset
  
The data totalling 55GB is split across multiple files. Each zip file is a self contained data set segment with both features and labels.

https://s3-us-west-2.amazonaws.com/gtav-captures/67b90283-627b-45cf-9ff2-63dcb95bfc67.zip

https://s3-us-west-2.amazonaws.com/gtav-captures/7007b0bf-503c-4eb7-9b58-19e123ef40e0.zip

https://s3-us-west-2.amazonaws.com/gtav-captures/782579db-da70-492e-a119-4e5bf1241698.zip

https://s3-us-west-2.amazonaws.com/gtav-captures/9bac3205-32d1-4e24-8bc3-7591dbbfac34.zip

https://s3-us-west-2.amazonaws.com/gtav-captures/bcac5255-a6aa-402b-9b75-4d9c422b8ae8.zip

https://s3-us-west-2.amazonaws.com/gtav-captures/e121fb4d-2b4f-40e5-9a34-2658e7647afd.zip

https://s3-us-west-2.amazonaws.com/gtav-captures/e14e4ede-d064-46ae-b513-bab61ca3259f.zip

https://s3-us-west-2.amazonaws.com/gtav-captures/ebecc37a-77ea-46a2-bd54-f67740a411a9.zip

https://s3-us-west-2.amazonaws.com/gtav-captures/ee16f4b5-07f1-4d96-a5b0-92b7de2eee17.zip

https://s3-us-west-2.amazonaws.com/gtav-captures/fd10222e-d26b-4c47-8118-98c8ea545bb4.zip

https://s3-us-west-2.amazonaws.com/gtav-captures/fdf4ad8d-d9b8-49a7-b9a6-c597b8876e0f.zip
