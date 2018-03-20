# mapr-streams-mxnet-face
implement mxnet face and insightface with mapr streams for near real time face detection and recognition with deep learning models

# Get pre-trained models
After clone the repo, get the face detection from [mxnet-face](https://github.com/tornadomeet/mxnet-face), whereas the model is stored at [dropbox](https://www.dropbox.com/sh/yqn8sken82gpmfr/AAC8WNSaA1ADVuUq8yaPQF0da?dl=0). Also the face recogition model from [insightface](https://github.com/deepinsight/insightface) whereas the model is stored at [google drive](https://drive.google.com/file/d/1x0-EiYX9jMUKiq-n1Bd9OCK4fVB3a54v/view)

put model-0000.params under "consumer/models/", put mxnet-face-fr50-0000.params under "consumer/deploy"

# Pre-requisite
A GPU MapR cluster, some installation could be referred from [this blog](https://mengdong.github.io/2017/07/14/kubernetes-1.7-gpu-on-mapr-distributed-deep-learning/)

Your laptop, with a camera if you want the content from your camera, should have MapR MAC/Linux/Windows Client installed and tested to be able to connect to your GPU MapR cluster, the installation is [here](https://maprdocs.mapr.com/52/AdvancedInstallation/SettingUptheClient-install-mapr-client.html) 

# Produce the content into a GPU MapR cluster
Producer code is straightforward, also hardcoded. Run "python mapr-producer-video.py", will read the Three Billboards trailer and produce it to a stream on the cluster:'/mapr/DLcluster/tmp/rawvideostream'. Also, you could use the camera, it is similar. 

Before run the producer code, Stream should be created on the cluster, or through the client. Simple commands to create the stream and topics:
```
maprcli stream delete -path /tmp/rawvideostream
maprcli stream create -path /tmp/rawvideostream
maprcli stream edit -path /tmp/rawvideostream -produceperm p -consumeperm p -topicperm p
maprcli stream topic create -path /tmp/rawvideostream -topic topic1 -partitions 1
```


# Consumer the content in the GPU MapR cluster
After making sure the stream in on GPU cluster, we can run the consumer which contains the facial recognition code to process the stream: "python mapr\_consumer.py". We will read from stream '/tmp/rawvideostream', get the face embedding vector and bounding boxes, and write them to stream '/tmp/processedvideostream', also, we will write all identified faces into stream '/tmp/identifiedstream'

Similarly, the stream should be pre-created:
```
maprcli stream delete -path /tmp/processedvideostream
maprcli stream create -path /tmp/processedvideostream
maprcli stream edit -path /tmp/processedvideostream -produceperm p -consumeperm p -topicperm p
maprcli stream topic create -path /tmp/processedvideostream -topic topic1 -partitions 1

maprcli stream create -path /tmp/identifiedstream
maprcli stream edit -path /tmp/identifiedstream -produceperm p -consumeperm p -topicperm p
maprcli stream topic create -path /tmp/identifiedstream -topic sam -partitions 1
maprcli stream topic create -path /tmp/identifiedstream -topic frances -partitions 1
maprcli stream topic create -path /tmp/identifiedstream -topic all -partitions 1
```

# Identify new person in the stream with a picture and a docker run command on your laptop

# Demo the processed stream from a running docker on your laptop
```
docker pull mengdong/mapr-pacc-mxnet:5.2.2_3.0.1_ubuntu16_yarn_fuse_hbase_streams_flask_client_arguments

docker run -it --privileged --cap-add SYS_ADMIN --cap-add SYS_RESOURCE --device /dev/fuse -e MAPR_CLUSTER=DLcluster  \
-e MAPR_CLDB_HOSTS=10.0.1.74 -e MAPR_CONTAINER_USER=mapr -e MAPR_CONTAINER_UID=5000 -e MAPR_CONTAINER_GROUP=mapr  \
-e MAPR_CONTAINER_GID=5000 -e MAPR_MOUNT_PATH=/mapr \
-e GROUPID=YOUGROUPNAME -e STREAM=/tmp/identifiedstream -e TOPIC=all(choose from all/frances/sam) \
-e TIMEOUT=0.035(0.035 if reading from topic all, 0.2 from frances/sam, can be flexible) -e PORT=5010(choose a new port) \
-p 5010:5010(match the port you chose before) mengdong/mapr-pacc-mxnet:5.2.2_3.0.1_ubuntu16_yarn_fuse_hbase_streams_flask_client_arguments
```

The video will show up at the port you chose (go to 'http://localhost:5010') 
