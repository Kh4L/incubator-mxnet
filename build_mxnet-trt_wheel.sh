set -ex
BUILDER=mxnet_trt:builder
git submodule update --init --recursive

docker build -t ${BUILDER} -f Dockerfile.wheel .
mkdir -p wheelhouse
docker run --rm -v $(pwd)/wheelhouse:/wheelhouse ${BUILDER} /bin/bash -c "cp /opt/mxnet/python/dist/* /wheelhouse"

# comment this line out when debugging
docker rmi ${BUILDER}
