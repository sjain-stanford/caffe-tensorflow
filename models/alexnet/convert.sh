python -u ../../convert.py \
      alexnet.prototxt \
      --caffemodel alexnet.caffemodel \
      --data-output-path alexnet_caffe2tf.npy \
      --code-output-path alexnet_caffe2tf.py \
      |& tee alexnet_convert.log
