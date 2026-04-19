{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3357a4d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from skimage.feature import peak_local_max\n",
    "\n",
    "def extract_centroids_from_heatmap(heatmap, threshold=0.5, min_distance=5):\n",
    "    heatmap = heatmap.squeeze()\n",
    "    peaks = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=threshold)\n",
    "    return peaks  # [[y,x], ...]\n",
    "\n",
    "def match_centroids(pred, true, tol=5):\n",
    "    pred = np.asarray(pred)\n",
    "    true = np.asarray(true)\n",
    "\n",
    "    if len(pred) == 0 and len(true) == 0:\n",
    "        return 1, 1, 1\n",
    "    if len(pred) == 0 or len(true) == 0:\n",
    "        return 0, 0, 0\n",
    "\n",
    "    matched = 0\n",
    "    for t in true:\n",
    "        dists = np.linalg.norm(pred - t, axis=1)\n",
    "        if np.any(dists <= tol):\n",
    "            matched += 1\n",
    "\n",
    "    precision = matched / (len(pred) + 1e-8)\n",
    "    recall = matched / (len(true) + 1e-8)\n",
    "    f1 = 2 * precision * recall / (precision + recall + 1e-8)\n",
    "    return precision, recall, f1\n"
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
