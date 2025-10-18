import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator


@dataclass
class TrainParams:
    data_dir: str = os.path.join(os.path.dirname(__file__), 'input_data')
    train_subdir: str = 'Training'
    test_subdir: str = 'Testing'
    img_size: Tuple[int, int] = (128, 128)
    batch_size: int = 8
    epochs: int = 6
    learning_rate: float = 0.001
    base_model_name: str = 'MobileNetV2'  # kept same as notebook for speed on CPU
    output_model_path: str = os.path.join(os.path.dirname(__file__), 'trained-model.keras')
    # for quick CPU demo we can optionally subset per-class; None -> use all
    per_class_limit: Optional[int] = 80  # was 80 in the notebook script
    validation_split_from_test: float = 0.5  # split Testing directory into valid/test


def _flow_from_df_subset(df, n):
    if n is None:
        return df
    # sample up to n per class
    return (df.groupby('Class', group_keys=False)
              .apply(lambda g: g.sample(n=min(len(g), n), random_state=42))
              .reset_index(drop=True))


def _make_generators(params: TrainParams):
    import pandas as pd
    import os
    from sklearn.model_selection import train_test_split

    def _df_from_dir(path: str):
        classes, class_paths = zip(*[(label, os.path.join(path, label, image))
                                     for label in os.listdir(path)
                                     if os.path.isdir(os.path.join(path, label))
                                     for image in os.listdir(os.path.join(path, label))])
        return pd.DataFrame({'Class Path': class_paths, 'Class': classes})

    tr_path = os.path.join(params.data_dir, params.train_subdir)
    ts_path = os.path.join(params.data_dir, params.test_subdir)

    tr_df = _df_from_dir(tr_path)
    ts_df = _df_from_dir(ts_path)

    valid_df, ts_df = train_test_split(ts_df, train_size=params.validation_split_from_test,
                                       random_state=20, stratify=ts_df['Class'])

    tr_df = _flow_from_df_subset(tr_df, params.per_class_limit)
    # keep validation/test at a small subset if per_class_limit is set
    v_limit = None if params.per_class_limit is None else max(20, params.per_class_limit // 4)
    valid_df = _flow_from_df_subset(valid_df, v_limit)
    ts_df = _flow_from_df_subset(ts_df, v_limit)

    gen = ImageDataGenerator(rescale=1/255.0)
    ts_only = ImageDataGenerator(rescale=1/255.0)

    tr_gen = gen.flow_from_dataframe(tr_df, x_col='Class Path', y_col='Class',
                                     batch_size=params.batch_size, target_size=params.img_size, shuffle=True)
    valid_gen = gen.flow_from_dataframe(valid_df, x_col='Class Path', y_col='Class',
                                        batch_size=params.batch_size, target_size=params.img_size)
    ts_gen = ts_only.flow_from_dataframe(ts_df, x_col='Class Path', y_col='Class',
                                         batch_size=params.batch_size, target_size=params.img_size, shuffle=False)

    class_dict = tr_gen.class_indices
    classes = list(class_dict.keys())
    return tr_gen, valid_gen, ts_gen, classes


def _build_model(num_classes: int, img_size: Tuple[int, int], base_model_name: str, learning_rate: float) -> Model:
    img_shape = (img_size[0], img_size[1], 3)
    base_name = base_model_name.lower()
    if base_name == 'mobilenetv2':
        base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet',
                                                       input_shape=img_shape, pooling='avg')
    elif base_name == 'efficientnetb0':
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet',
                                                          input_shape=img_shape, pooling='avg')
    else:
        # default fallback
        base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet',
                                                       input_shape=img_shape, pooling='avg')

    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(Adamax(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
    return model


def train_model_service(params: TrainParams) -> Dict[str, Any]:
    """Train a fresh model from scratch and save it. Returns training/eval details."""
    tr_gen, valid_gen, ts_gen, classes = _make_generators(params)
    num_classes = len(classes)
    model = _build_model(num_classes, params.img_size, params.base_model_name, params.learning_rate)

    hist = model.fit(
        tr_gen,
        epochs=params.epochs,
        validation_data=valid_gen,
        shuffle=True,
        verbose=1,
    )
    history = hist.history

    train_score = model.evaluate(tr_gen, verbose=0)
    valid_score = model.evaluate(valid_gen, verbose=0)
    test_score = model.evaluate(ts_gen, verbose=0)

    # Save model
    out_path = params.output_model_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save(out_path)

    # Pack metrics in a readable dict
    metrics_names = model.metrics_names  # e.g., ['loss','accuracy','precision','recall']

    def _pack_score(names, values):
        return {name: float(val) for name, val in zip(names, values)}

    result: Dict[str, Any] = {
        'classes': classes,
        'output_model_path': out_path,
        'history': {k: [float(x) for x in v] for k, v in history.items()},
        'final_epoch': len(next(iter(history.values()))) if history else 0,
        'train_score': _pack_score(metrics_names, train_score),
        'valid_score': _pack_score(metrics_names, valid_score),
        'test_score': _pack_score(metrics_names, test_score),
        'used_params': asdict(params),
    }
    return result
