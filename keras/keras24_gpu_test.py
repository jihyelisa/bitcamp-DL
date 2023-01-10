import tensorflow as tf
print(tf.__version__)  # 2.7.4

# experimental: 아직 test 버전인 기능
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

if(gpus):   # 작동 중인 gpu가 있을 경우
    print("gpu 돈다")
else:
    print("gpu 안 돈다")