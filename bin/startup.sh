#!/bin/bash

set -e
set -o pipefail

echo "executing startup script (first run)"

py_major_version=${1:-2}
py_minor_version=${2:-7}
[ "$py_major_version" == "3" ] && py_minor_version="4"

py="python${py_major_version}.${py_minor_version}"

rm -fr /opt/amazon/ApolloCmd
rm -fr /opt/amazon/config
rm -fr /opt/amazon/configuration
rm -fr /opt/amazon/container-tests
rm -fr /opt/amazon/images
rm -fr /opt/amazon/include
rm -fr /opt/amazon/jruby*
rm -fr /opt/amazon/jython*
rm -fr /opt/amazon/man
rm -fr /opt/amazon/model
rm -fr /opt/amazon/perl
rm -fr /opt/amazon/python*
rm -fr /opt/amazon/ruby*
rm -fr /opt/amazon/sbin
rm -fr /opt/amazon/schema
rm -fr /opt/amazon/share
rm -fr /opt/amazon/ssl
rm -fr /opt/amazon/test

find /opt/amazon/lib/$py/site-packages -mindepth 1 -maxdepth 1 \
    | grep -v 'SageMakerContainerSupport' \
    | grep -v 'SageMakerMXNetContainer' \
    | grep -v 'SageMakerTensorflowContainer' \
    | grep -v 'container_support' \
    | grep -v 'mxnet_container' \
    | grep -v 'tf_container' \
    | xargs rm -fr

# move ours to a safe place
rm -fr /tmp/site-packages
mv /opt/amazon/lib/$py/site-packages /tmp/site-packages

# clean up the rest of lib
rm -fr /opt/amazon/lib

# move our packages back to PYTHONPATH location
mkdir -p /opt/amazon/lib/$py
mv /tmp/site-packages /opt/amazon/lib/$py/site-packages

# empty /opt/amazon/bin, except for entry.py
# note that this will delete this script as well
find /opt/amazon/bin/ -mindepth 1 -maxdepth 1  ! -name 'entry.py'  | xargs rm -fr

# write a harmless new startup script
echo '#!/bin/bash
echo "executing startup script"
' > /opt/amazon/bin/startup.sh

chmod a+x /opt/amazon/bin/startup.sh

exit 0
