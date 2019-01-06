python -u ../../convert.py \
      mobilenet_v1.prototxt \
      --caffemodel mobilenet_v1.caffemodel \
      --data-output-path mobilenet_v1_caffe2tf.npy \
      --code-output-path mobilenet_v1_caffe2tf.py \
      |& tee mobilenet_v1_convert.log
