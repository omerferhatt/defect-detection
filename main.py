from data.dataset import get_ds_pipeline
from models.mobilenet_v2 import get_model

if __name__ == '__main__':
    train_ds, test_ds = get_ds_pipeline(batch_size=64)
    model = get_model()
    model.fit(train_ds, epochs=20, validation_data=test_ds)
