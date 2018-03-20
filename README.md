# mapr-streams-mxnet-face
implement mxnet face and insightface with mapr streams for near real time face detection and recognition with deep learning models

# Get pre-trained models
After clone the repo, get the face detection from [mxnet-face](https://github.com/tornadomeet/mxnet-face), whereas the model is stored at [dropbox](https://www.dropbox.com/sh/yqn8sken82gpmfr/AAC8WNSaA1ADVuUq8yaPQF0da?dl=0). Also the face recogition model from [insightface](https://github.com/deepinsight/insightface) whereas the model is stored at [google drive](https://drive.google.com/file/d/1x0-EiYX9jMUKiq-n1Bd9OCK4fVB3a54v/view)

put model-0000.params under "consumer/models/", put mxnet-face-fr50-0000.params under "consumer/deploy"

# Pre-requisite
A GPU MapR cluster, some installation could be referred from [this blog](https://mengdong.github.io/2017/07/14/kubernetes-1.7-gpu-on-mapr-distributed-deep-learning/)

Your laptop, with a camera if you want the content from your camera, should have MapR MAC/Linux/Windows Client installed and tested to be able to connect to your GPU MapR cluster, the installation is [here](https://maprdocs.mapr.com/52/AdvancedInstallation/SettingUptheClient-install-mapr-client.html) 

# Produce the content into a GPU MapR cluster
Producer code is straightforward, also hardcoded. Run "python mapr-producer-video.py", will read the Three Billboards trailer and produce it to a stream on the cluster:'/mapr/DLcluster/tmp/rawvideostream'. 

Before run the producer code, Stream should be created on the cluster, or through the client. Simple commands to create the stream and topics:
```maprcli stream delete -path /tmp/rawvideostream
maprcli stream create -path /tmp/rawvideostream
maprcli stream edit -path /tmp/rawvideostream -produceperm p -consumeperm p -topicperm p
maprcli stream topic create -path /tmp/rawvideostream -topic topic1 -partitions 1
```




