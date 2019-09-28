import matplotlib.pyplot as plt

def numpy_array_image(img):
  """
  Shows image for np array
  """
  plt.figure(figsize=(20,30))
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()

def pytorch_tensor_image(img):
  """
  Shows image for pytorch tensor
  """
  numpy_array_image(img.permute(1, 2, 0))
