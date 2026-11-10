### Run run_all.sh

``` bash
docker start viw-dev

cd /home/nio/code/VIW-Fusion
bash run/run_all.sh
```

### Run->Rviz

```bash
docker exec -it viw-dev bash -c 

source /opt/ros/melodic/setup.bash 
source /root/catkin_ws/devel/setup.bash 
roslaunch vins vins_rviz.launch
```