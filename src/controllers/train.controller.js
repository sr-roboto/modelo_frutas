import * as tf from '@tensorflow/tfjs-node';
import sharp from 'sharp';
import fs from 'fs/promises';
import path from 'path';

class ModelController {
  constructor() {
    this.dataset = [];
    this.model = null;
    this.etiquetas = [];
  }

  /**
   * Carga el dataset desde archivos locales
   */
  async cargarDatasetLocal() {
    console.log('📁 Cargando dataset local...');
    const datasetPath = './dataset/frutas';

    try {
      // Verificar si existe la carpeta del dataset
      await fs.access(datasetPath);
    } catch (error) {
      console.log('❌ No se encontró la carpeta del dataset:', datasetPath);
      return false;
    }

    this.dataset = [];
    const carpetas = await fs.readdir(datasetPath);

    for (const carpeta of carpetas) {
      const carpetaPath = path.join(datasetPath, carpeta);
      const stats = await fs.stat(carpetaPath);

      if (stats.isDirectory()) {
        console.log(`📂 Procesando carpeta: ${carpeta}`);
        const archivos = await fs.readdir(carpetaPath);

        for (const archivo of archivos) {
          if (this.esImagenValida(archivo)) {
            const archivoPath = path.join(carpetaPath, archivo);
            await this.procesarImagenLocal(archivoPath, carpeta);
          }
        }
      }
    }

    console.log(`✅ Dataset cargado: ${this.dataset.length} imágenes`);
    return this.dataset.length > 0;
  }

  /**
   * Verifica si el archivo es una imagen válida
   */
  esImagenValida(nombreArchivo) {
    const extensiones = ['.jpg', '.jpeg', '.png', '.bmp', '.gif'];
    return extensiones.some((ext) => nombreArchivo.toLowerCase().endsWith(ext));
  }

  /**
   * Procesa una imagen desde el sistema de archivos local
   */
  async procesarImagenLocal(rutaArchivo, etiqueta) {
    try {
      const buffer = await fs.readFile(rutaArchivo);

      // Redimensionar y normalizar imagen
      const imageBuffer = await sharp(buffer).resize(64, 64).raw().toBuffer();

      // Convertir a tensor
      const tensorImg = tf
        .tensor3d(new Uint8Array(imageBuffer), [64, 64, 3])
        .toFloat()
        .div(255.0);

      this.dataset.push({ tensor: tensorImg, label: etiqueta.toLowerCase() });
    } catch (error) {
      console.error(`❌ Error procesando ${rutaArchivo}:`, error.message);
    }
  }

  /**
   * Procesa imágenes subidas desde el frontend (mantener para compatibilidad)
   */
  async procesarImagenes(files) {
    this.dataset = [];

    for (const file of files) {
      try {
        const buffer = await sharp(file.buffer).resize(64, 64).raw().toBuffer();

        const tensorImg = tf
          .tensor3d(new Uint8Array(buffer), [64, 64, 3])
          .toFloat()
          .div(255.0);

        const label = this.extraerEtiqueta(file.originalname);
        this.dataset.push({ tensor: tensorImg, label });
      } catch (error) {
        console.error(`Error procesando imagen ${file.originalname}:`, error);
      }
    }
  }

  /**
   * Extrae la etiqueta del nombre del archivo
   */
  extraerEtiqueta(filename) {
    return filename.split('_')[0].toLowerCase();
  }

  /**
   * Entrena el modelo con el dataset cargado
   */
  async entrenarModelo() {
    if (this.dataset.length === 0) {
      throw new Error('No hay imágenes para entrenar');
    }

    console.log('🧠 Iniciando entrenamiento del modelo...');

    // Extraer etiquetas únicas
    this.etiquetas = [...new Set(this.dataset.map((d) => d.label))];
    const etiquetaToIndex = Object.fromEntries(
      this.etiquetas.map((e, i) => [e, i])
    );

    console.log(`📊 Clases detectadas: ${this.etiquetas.join(', ')}`);

    // Preparar tensores
    const xs = this.dataset.map((d) => d.tensor);
    const ys = this.dataset.map((d) => etiquetaToIndex[d.label]);

    const xsStacked = tf.stack(xs);
    const ysOneHot = tf.oneHot(tf.tensor1d(ys, 'int32'), this.etiquetas.length);

    // Crear modelo
    this.model = tf.sequential();

    this.model.add(
      tf.layers.conv2d({
        inputShape: [64, 64, 3],
        filters: 16,
        kernelSize: 3,
        activation: 'relu',
      })
    );

    this.model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    this.model.add(tf.layers.flatten());
    this.model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    this.model.add(
      tf.layers.dense({
        units: this.etiquetas.length,
        activation: 'softmax',
      })
    );

    this.model.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    // Entrenar modelo
    const history = await this.model.fit(xsStacked, ysOneHot, {
      epochs: 10,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(
            `📈 Época ${epoch + 1}: Pérdida = ${logs.loss.toFixed(
              4
            )}, Precisión = ${logs.acc.toFixed(4)}`
          );
        },
      },
    });

    // Guardar modelo
    await this.model.save('file://./models/modelo-frutas');

    // Guardar mapeo de etiquetas
    await fs.writeFile(
      './models/etiquetas.json',
      JSON.stringify({
        etiquetas: this.etiquetas,
        etiquetaToIndex,
      })
    );

    // Limpiar memoria
    xsStacked.dispose();
    ysOneHot.dispose();
    xs.forEach((tensor) => tensor.dispose());

    console.log('✅ Modelo entrenado y guardado exitosamente');

    return {
      etiquetas: this.etiquetas,
      history: history.history,
      totalImagenes: this.dataset.length,
    };
  }

  /**
   * Carga un modelo existente
   */
  async cargarModeloExistente() {
    try {
      // Verificar si existe el modelo
      await fs.access('./models/modelo-frutas/model.json');

      // Cargar modelo
      this.model = await tf.loadLayersModel(
        'file://./models/modelo-frutas/model.json'
      );

      // Cargar etiquetas
      const etiquetasData = JSON.parse(
        await fs.readFile('./models/etiquetas.json', 'utf8')
      );
      this.etiquetas = etiquetasData.etiquetas;

      console.log('✅ Modelo existente cargado correctamente');
      console.log(`📊 Clases disponibles: ${this.etiquetas.join(', ')}`);

      return true;
    } catch (error) {
      console.log('ℹ️ No se encontró modelo existente');
      return false;
    }
  }

  /**
   * Inicializa el modelo (carga existente o entrena nuevo)
   */
  async inicializar() {
    console.log('🚀 Inicializando sistema de clasificación...');

    // Intentar cargar modelo existente
    const modeloExistente = await this.cargarModeloExistente();

    if (modeloExistente) {
      return true;
    }

    // Si no existe modelo, intentar entrenar uno nuevo
    console.log('🔄 Entrenando nuevo modelo...');
    const datasetCargado = await this.cargarDatasetLocal();

    if (datasetCargado) {
      await this.entrenarModelo();
      return true;
    } else {
      console.log(
        '⚠️ No se pudo cargar dataset. El servidor funcionará sin modelo pre-entrenado.'
      );
      return false;
    }
  }

  /**
   * Predice la clase de una nueva imagen
   */
  async predecir(imageBuffer) {
    if (!this.model) {
      throw new Error('No hay modelo disponible para predicción');
    }

    // Procesar imagen
    const buffer = await sharp(imageBuffer).resize(64, 64).raw().toBuffer();

    const tensorImg = tf
      .tensor3d(new Uint8Array(buffer), [64, 64, 3])
      .toFloat()
      .div(255.0)
      .expandDims(0);

    // Realizar predicción
    const prediccion = this.model.predict(tensorImg);
    const probabilidades = await prediccion.data();

    // Encontrar la clase con mayor probabilidad
    const maxIndex = probabilidades.indexOf(Math.max(...probabilidades));

    tensorImg.dispose();
    prediccion.dispose();

    return {
      clase: this.etiquetas[maxIndex],
      probabilidad: probabilidades[maxIndex],
      todasLasProbabilidades: this.etiquetas.map((etiqueta, i) => ({
        etiqueta,
        probabilidad: probabilidades[i],
      })),
    };
  }

  /**
   * Obtiene información del modelo actual
   */
  getInfo() {
    return {
      modeloEntrenado: this.model !== null,
      etiquetas: this.etiquetas,
      totalClases: this.etiquetas.length,
    };
  }
}

export default new ModelController();
