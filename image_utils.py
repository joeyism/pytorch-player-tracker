import numpy as np

def extract_player_mask(img, mask, box):
  """
  Extract image of a player
  """
  player_only_img = np.multiply(img, mask)
  cropped_player = player_only_img[:, int(box[0][1]):int(box[1][1]), int(box[0][0]):int(box[1][0])]
  return cropped_player

def extract_player_masks(img, masks, boxes):
  """
  Extract image of all the players
  """
  return [extract_player_mask(img, mask, box) for mask, box in zip(masks, boxes)]
