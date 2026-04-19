# In-memory cache keyed by (image_path, model_path, orientation, device, batch_size)
_SEGMENT_CACHE: dict = {}

def segment_cache_key(image_path, model_path, orientation, device, batch_size):
    return (image_path, model_path, orientation, device, batch_size)

def get_cache():
    return _SEGMENT_CACHE
