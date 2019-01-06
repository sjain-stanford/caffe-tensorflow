python -u ../../convert.py \
      squeezenet_1p1.prototxt \
      --caffemodel squeezenet_1p1.caffemodel \
      --data-output-path squeezenet_1p1_caffe2tf.npy \
      --code-output-path squeezenet_1p1_caffe2tf.py \
      |& tee squeezenet_1p1_convert.log
