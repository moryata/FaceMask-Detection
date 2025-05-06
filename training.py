import os
# Suppress oneDNN informational messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Set TensorFlow log level to suppress INFO and WARNING messages before other imports
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
from rich.console import Console
import matplotlib.pyplot as plt

# Import TensorFlow and Keras
import tensorflow as tf

# Import all required Keras components from tf.keras
# This is the recommended approach for newer versions of TensorFlow/Keras
keras = tf.keras
EfficientNetB0 = keras.applications.EfficientNetB0
preprocess_input = keras.applications.efficientnet.preprocess_input
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Model = keras.models.Model
Adam = keras.optimizers.Adam
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
EarlyStopping = keras.callbacks.EarlyStopping
ModelCheckpoint = keras.callbacks.ModelCheckpoint

# Inisialisasi Console
console = Console()

# Configuration dengan parameter yang dioptimalkan
CONFIG = {
    'learning_rate': 5e-5,  # Learning rate yang lebih kecil untuk konvergensi yang lebih stabil
    'epochs': 10,  # Epoch untuk pelatihan awal
    'batch_size': 32,  # Batch size yang lebih besar untuk generalisasi yang lebih baik
    'image_size': (224, 224),  # Ukuran input yang direkomendasikan untuk EfficientNetB0
    'dataset_dir': 'dataset',
    'model_dir': 'models',
    # Nama model yang lebih sederhana
    'model_name': 'mask_detector_efficientnetb0_modular.h5',
    'validation_split': 0.2,
    'dropout_rate': 0.5,  # Dropout yang lebih tinggi untuk mencegah overfitting
    'patience_lr': 3,  # Lebih sabar dengan penurunan learning rate
    'patience_early_stopping': 5,  # Lebih sabar dengan early stopping
    'random_seed': 42,  # Seed untuk reproduktifitas generator data
    'fine_tuning': True,  # Aktifkan fine-tuning
    'fine_tuning_epochs': 8,  # Jumlah epoch untuk fine-tuning
    'fine_tuning_lr': 1e-6,  # Learning rate yang lebih kecil untuk fine-tuning
    'fine_tuning_batch_size': 16  # Batch size yang lebih kecil untuk fine-tuning
}

def check_environment():
    """Memverifikasi versi Python dan keberadaan dataset."""
    required_min_version = (3, 8)
    required_max_version = (3, 12)
    current_version = sys.version_info
    if not (required_min_version <= current_version < required_max_version):
        console.print(f"[yellow]Warning: Rekomendasi versi Python {required_min_version[0]}.{required_min_version[1]} - "
                      f"{required_max_version[0]}.{required_max_version[1]-1}. "
                      f"Versi Anda: {current_version.major}.{current_version.minor}. Lanjutkan dengan hati-hati.[/]")

    if not os.path.exists(CONFIG['dataset_dir']):
        console.print(f"[bold red][ERROR] Direktori dataset tidak ditemukan: {CONFIG['dataset_dir']}[/]")
        sys.exit(1)

    # Verifikasi bahwa TensorFlow terinstal
    try:
        tf_version = tf.__version__
        keras_version = keras.__version__
        console.print(f"[green]TensorFlow terdeteksi, versi: {tf_version}[/]")
        console.print(f"[green]Keras terdeteksi, versi: {keras_version}[/]")
    except ImportError:
        console.print("[bold red][ERROR] TensorFlow tidak terinstal. Jalankan 'pip install tensorflow'.[/]")
        sys.exit(1)

    console.print("[green]Lingkungan dan dataset OK.[/]")

def create_data_generators():
    """Membuat dan mengkonfigurasi generator data untuk training dan validasi dengan augmentasi yang lebih kuat."""
    # Generator untuk data training dengan augmentasi yang lebih kuat
    train_datagen = ImageDataGenerator(
        # Gunakan preprocess_input yang sesuai untuk EfficientNetB0
        preprocessing_function=preprocess_input,
        validation_split=CONFIG['validation_split'],
        rotation_range=20,  # Rotasi yang lebih besar
        width_shift_range=0.2,  # Pergeseran yang lebih besar
        height_shift_range=0.2,  # Pergeseran yang lebih besar
        shear_range=0.15,  # Shear yang lebih besar
        zoom_range=0.2,  # Zoom yang lebih besar
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],  # Variasi kecerahan
        channel_shift_range=20.0,  # Variasi warna
        fill_mode='nearest'
    )

    # Generator untuk data validasi hanya dengan preprocessing
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=CONFIG['validation_split']
    )

    common_args = {
        'directory': CONFIG['dataset_dir'],
        'target_size': CONFIG['image_size'],
        'batch_size': CONFIG['batch_size'],
        'class_mode': 'categorical',
        'seed': CONFIG['random_seed']
    }

    # Generator untuk data training
    train_gen = train_datagen.flow_from_directory(
        subset='training',
        shuffle=True,
        **common_args
    )

    # Generator untuk data validasi
    val_gen = val_datagen.flow_from_directory(
        subset='validation',
        shuffle=False,
        **common_args
    )

    console.print(f"[blue]Ditemukan {train_gen.num_classes} kelas:[/]")
    for cls, idx in train_gen.class_indices.items():
        console.print(f"  - {cls} (index: {idx})")

    console.print("[green]Data augmentasi yang lebih kuat diterapkan pada data training[/]")

    return train_gen, val_gen

def build_model(num_classes):
    """Membuat dan mengkompilasi arsitektur model dengan arsitektur yang lebih baik."""
    # --- Menggunakan EfficientNetB0 ---
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*CONFIG['image_size'], 3)
    )
    # ------------------------------

    # Freeze base model
    base_model.trainable = False

    # Tambahkan lapisan klasifikasi yang lebih kompleks
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)

    # Tambahkan lapisan Dense dengan lebih banyak neuron
    x = Dense(256, activation='relu', name='dense_1')(x)
    x = Dropout(CONFIG['dropout_rate'], name='dropout_1')(x)

    # Tambahkan lapisan Dense kedua
    x = Dense(128, activation='relu', name='dense_2')(x)
    x = Dropout(CONFIG['dropout_rate']/2, name='dropout_2')(x)

    # Lapisan output
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=outputs, name='EfficientNetB0_mask_detector')

    # Kompilasi model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        metrics=['accuracy']
    )

    console.print("[blue]Ringkasan Model:[/]")
    model.summary(print_fn=lambda x: console.print(x))

    return model

def plot_training_history(history, file_path='training_history.png'):
    """Memplot metrik training dan validasi."""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
         console.print("[yellow]Warning: Style 'seaborn-v0_8-darkgrid' tidak ditemukan, menggunakan default Matplotlib.[/]")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Akurasi
    ax1.plot(history.history['accuracy'], label='Akurasi Training')
    ax1.plot(history.history['val_accuracy'], label='Akurasi Validasi')
    ax1.set_title('Akurasi Model')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Akurasi')
    ax1.legend()
    ax1.grid(True)

    # Plot Loss
    ax2.plot(history.history['loss'], label='Loss Training')
    ax2.plot(history.history['val_loss'], label='Loss Validasi')
    ax2.set_title('Loss Model')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(file_path)
    console.print(f"[green]Plot riwayat training disimpan sebagai [bold]{file_path}[/bold][/]")
    plt.close(fig)

def fine_tune_model(model, train_gen, val_gen, best_model_path):
    """Fine-tune model dengan unfreeze beberapa layer terakhir dari base model."""
    console.print("\n[bold cyan]Memulai proses fine-tuning...[/]")

    # Cari base model (EfficientNetB0) di antara layer model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # EfficientNetB0 adalah tf.keras.Model
            base_model = layer
            break

    if base_model is None:
        console.print("[bold yellow]WARNING: Tidak dapat menemukan base model untuk fine-tuning. Menggunakan seluruh model.[/]")
        # Jika tidak dapat menemukan base model, aktifkan seluruh model
        model.trainable = True
    else:
        console.print(f"[green]Base model ditemukan: {base_model.name}[/]")
        # Unfreeze base model
        base_model.trainable = True

        # Freeze beberapa layer awal (hanya unfreeze 30% layer terakhir)
        total_layers = len(base_model.layers)
        fine_tune_at = int(total_layers * 0.7)  # Freeze 70% layer awal

        console.print(f"[blue]Total layer di base model: {total_layers}[/]")
        console.print(f"[blue]Fine-tuning dari layer ke-{fine_tune_at} sampai {total_layers}[/]")

        # Freeze layer awal
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    # Recompile model dengan learning rate yang lebih kecil
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=CONFIG['fine_tuning_lr']),
        metrics=['accuracy']
    )

    console.print("[blue]Model setelah unfreezing beberapa layer:[/]")
    # Hitung jumlah parameter yang dapat dilatih
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    console.print(f"[blue]Parameter yang dapat dilatih: {trainable_params:,}[/]")
    console.print(f"[blue]Parameter yang tidak dapat dilatih: {non_trainable_params:,}[/]")

    # Callbacks untuk fine-tuning
    reduce_lr_ft = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        verbose=1,
        min_lr=1e-7
    )

    early_stopping_ft = EarlyStopping(
        monitor='val_loss',
        patience=4,
        verbose=1,
        restore_best_weights=True
    )

    model_checkpoint_ft = ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    callbacks_ft = [reduce_lr_ft, early_stopping_ft, model_checkpoint_ft]

    # Buat generator data baru dengan batch size yang lebih kecil untuk fine-tuning
    if 'fine_tuning_batch_size' in CONFIG and CONFIG['fine_tuning_batch_size'] != CONFIG['batch_size']:
        console.print(f"[blue]Menggunakan batch size yang lebih kecil untuk fine-tuning: {CONFIG['fine_tuning_batch_size']}[/]")

        # Buat generator data baru untuk fine-tuning
        # Generator untuk data training dengan augmentasi yang lebih kuat
        ft_train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=CONFIG['validation_split'],
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            channel_shift_range=20.0,
            fill_mode='nearest'
        )

        # Generator untuk data validasi hanya dengan preprocessing
        ft_val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=CONFIG['validation_split']
        )

        # Generator untuk data training
        ft_train_gen = ft_train_datagen.flow_from_directory(
            directory=CONFIG['dataset_dir'],
            target_size=CONFIG['image_size'],
            batch_size=CONFIG['fine_tuning_batch_size'],
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=CONFIG['random_seed']
        )

        # Generator untuk data validasi
        ft_val_gen = ft_val_datagen.flow_from_directory(
            directory=CONFIG['dataset_dir'],
            target_size=CONFIG['image_size'],
            batch_size=CONFIG['fine_tuning_batch_size'],
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=CONFIG['random_seed']
        )

        # Fine-tune model dengan batch size yang lebih kecil
        history_ft = model.fit(
            ft_train_gen,
            validation_data=ft_val_gen,
            epochs=CONFIG['fine_tuning_epochs'],
            callbacks=callbacks_ft,
            verbose=1
        )
    else:
        # Fine-tune model dengan generator yang sama
        history_ft = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=CONFIG['fine_tuning_epochs'],
            callbacks=callbacks_ft,
            verbose=1
        )

    console.print("\n[bold green]Fine-tuning selesai.[/]")

    return model, history_ft

def main():
    console.print(f"[bold blue]Memulai training model deteksi masker ({CONFIG['model_name']})...[/]")
    console.print(f"Konfigurasi: {CONFIG}")

    check_environment()
    train_gen, val_gen = create_data_generators()

    if not train_gen.num_classes > 0:
        console.print("[bold red][ERROR] Tidak ada kelas yang ditemukan dalam generator data. Periksa struktur direktori dataset.[/]")
        sys.exit(1)

    model = build_model(num_classes=train_gen.num_classes)

    # Callbacks untuk training awal
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=CONFIG['patience_lr'],
        verbose=1,
        min_lr=1e-7
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=CONFIG['patience_early_stopping'],
        verbose=1,
        restore_best_weights=True
    )

    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    best_model_path = os.path.join(CONFIG['model_dir'], f"best_{CONFIG['model_name']}")

    model_checkpoint = ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    callbacks = [reduce_lr, early_stopping, model_checkpoint]

    console.print(f"\n[bold cyan]Memulai proses training awal untuk {CONFIG['epochs']} epochs...[/]")

    # Training awal
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    console.print("\n[bold green]Training awal selesai.[/]")

    # Plot history training awal
    plot_path = os.path.join(CONFIG['model_dir'], 'training_initial_history.png')
    plot_training_history(history, file_path=plot_path)

    # Fine-tuning jika diaktifkan
    if CONFIG['fine_tuning']:
        model, history_ft = fine_tune_model(model, train_gen, val_gen, best_model_path)

        # Plot history fine-tuning
        plot_path_ft = os.path.join(CONFIG['model_dir'], 'training_finetune_history.png')
        plot_training_history(history_ft, file_path=plot_path_ft)

    # Simpan model final
    final_model_path = os.path.join(CONFIG['model_dir'], CONFIG['model_name'])
    model.save(final_model_path)
    console.print(f"[bold green]Model final disimpan ke {final_model_path}[/]")
    console.print(f"[bold yellow]Model terbaik (berdasarkan val_loss) disimpan sebagai {best_model_path}[/]")

if __name__ == '__main__':
    main()
