def tensor_and(t1, t2):
  return (t1 + t2).eq(2)

def convLayerOutputSize(in_channels_dim, kernel_size, stride=1, padding=0):
  return int((in_channels_dim-kernel_size+padding*2)/stride +1)