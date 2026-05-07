from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

class ModelFactory:
    @staticmethod
    def get_model(architecture='EfficientNetB0', include_top=False, input_shape=(224, 224, 3)):
        if architecture == 'EfficientNetB0':
            base_model = EfficientNetB0(weights='imagenet', include_top=include_top, input_shape=input_shape)
        elif architecture == 'MobileNetV2':
            base_model = MobileNetV2(weights='imagenet', include_top=include_top, input_shape=input_shape)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        return base_model

    @staticmethod
    def build_e2e_model(architecture='EfficientNetB0', num_classes=1):
        base_model = ModelFactory.get_model(architecture, include_top=False)
        # Freeze base model
        base_model.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        activation = 'sigmoid' if num_classes == 1 else 'softmax'
        outputs = Dense(num_classes, activation=activation)(x)
        
        model = Model(inputs=base_model.input, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
