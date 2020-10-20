from .voc import VOC, VOCAug

def get_dataset(name):
    return {
        "voc": VOC,
        "vocaug": VOCAug,
    }[name]
