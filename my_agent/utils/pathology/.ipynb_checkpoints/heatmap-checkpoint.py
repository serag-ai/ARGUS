{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "389cf0b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "def scale_coords(coords, orig_size, target_size):\n",
    "    orig_w, orig_h = orig_size\n",
    "    new_w, new_h = target_size\n",
    "    scaled = []\n",
    "    for (x, y) in coords:\n",
    "        sx = int(x * new_w / orig_w)\n",
    "        sy = int(y * new_h / orig_h)\n",
    "        scaled.append((sx, sy))\n",
    "    return scaled\n",
    "\n",
    "def generate_gaussian_heatmap(coords, img_shape=(224, 224), sigma=3):\n",
    "    \"\"\"\n",
    "    coords: list of (x,y)\n",
    "    img_shape: (H,W)\n",
    "    \"\"\"\n",
    "    H, W = img_shape\n",
    "    heatmap = np.zeros((H, W), dtype=np.float32)\n",
    "\n",
    "    if coords is None or len(coords) == 0:\n",
    "        return heatmap\n",
    "\n",
    "    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing=\"ij\")\n",
    "    for (x, y) in coords:\n",
    "        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))\n",
    "        heatmap = np.maximum(heatmap, g)\n",
    "\n",
    "    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)\n",
    "    return heatmap\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
