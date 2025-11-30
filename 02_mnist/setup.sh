pip install tensorflowjs protobuf==3.20.3

python3 -m tensorflowjs.converters.converter \
    --input_format=keras \
    mnist_model.h5 \
    ./tfjs_model/

// python3 -m http.server 8000

// http://[::]:8000/digit_recognition.html

