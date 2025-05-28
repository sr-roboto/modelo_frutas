import * as tf from '@tensorflow/tfjs';
import sharp from 'sharp';
import fs from 'fs/promises';
import path from 'path';

// Variables del estado del modelo
let dataset = [];
let model = null;
let etiquetas = [];
let modeloYaEntrenado = false; // ✅ Bandera para evitar re-entrenamiento

// Configuración de frutas según el trabajo
const FRUTAS_REQUERIDAS = ['manzana', 'banana', 'pera', 'naranja', 'uva'];

/**
 * Carga el dataset desde archivos locales (según traer-frutas.py)
 */
const cargarDatasetLocal = async () => {
  console.log('📁 Cargando dataset local...');
  const datasetPath = './dataset/frutas';

  try {
    await fs.access(datasetPath);
  } catch (error) {
    console.log('❌ No se encontró la carpeta del dataset:', datasetPath);
    console.log('💡 Ejecuta primero: python traer-frutas.py');
    return false;
  }

  dataset = [];
  const carpetas = await fs.readdir(datasetPath);
  const contadorPorClase = {};

  // Verificar que existan las frutas requeridas
  for (const fruta of FRUTAS_REQUERIDAS) {
    if (!carpetas.includes(fruta)) {
      console.log(`⚠️ Falta la carpeta: ${fruta}`);
    }
  }

  for (const carpeta of carpetas) {
    const carpetaPath = path.join(datasetPath, carpeta);
    const stats = await fs.stat(carpetaPath);

    if (stats.isDirectory() && FRUTAS_REQUERIDAS.includes(carpeta)) {
      console.log(`📂 Procesando carpeta: ${carpeta}`);
      const archivos = await fs.readdir(carpetaPath);

      let imagenesEnCarpeta = 0;
      for (const archivo of archivos) {
        if (esImagenValida(archivo)) {
          const archivoPath = path.join(carpetaPath, archivo);
          await procesarImagenLocal(archivoPath, carpeta);
          imagenesEnCarpeta++;

          if (imagenesEnCarpeta % 5 === 0) {
            console.log(
              `   📸 Procesadas ${imagenesEnCarpeta} imágenes de ${carpeta}`
            );
          }
        }
      }

      contadorPorClase[carpeta] = imagenesEnCarpeta;
      console.log(`   ✅ ${carpeta}: ${imagenesEnCarpeta} imágenes procesadas`);
    }
  }

  console.log(`✅ Dataset cargado: ${dataset.length} imágenes total`);
  console.log('📊 Distribución por clase:');
  Object.entries(contadorPorClase).forEach(([clase, cantidad]) => {
    console.log(`   ${clase}: ${cantidad} imágenes`);
  });

  return dataset.length > 0;
};

/**
 * Verifica si el archivo es una imagen válida
 */
const esImagenValida = (nombreArchivo) => {
  const extensiones = ['.jpg', '.jpeg', '.png', '.bmp', '.gif'];
  return extensiones.some((ext) => nombreArchivo.toLowerCase().endsWith(ext));
};

/**
 * Procesa una imagen desde el sistema de archivos local
 */
const procesarImagenLocal = async (rutaArchivo, etiqueta) => {
  try {
    const buffer = await fs.readFile(rutaArchivo);
    const imageBuffer = await sharp(buffer).resize(64, 64).raw().toBuffer();

    const tensorImg = tf
      .tensor3d(new Uint8Array(imageBuffer), [64, 64, 3])
      .toFloat()
      .div(255.0);

    dataset.push({ tensor: tensorImg, label: etiqueta.toLowerCase() });
  } catch (error) {
    console.error(`❌ Error procesando ${rutaArchivo}:`, error.message);
  }
};

/**
 * Procesa imágenes subidas desde el frontend (mantener para compatibilidad)
 */
const procesarImagenes = async (files) => {
  console.log('📤 Procesando imágenes subidas desde frontend...');
  dataset = [];

  for (const file of files) {
    try {
      const buffer = await sharp(file.buffer).resize(64, 64).raw().toBuffer();

      const tensorImg = tf
        .tensor3d(new Uint8Array(buffer), [64, 64, 3])
        .toFloat()
        .div(255.0);

      const label = extraerEtiqueta(file.originalname);
      dataset.push({ tensor: tensorImg, label });
    } catch (error) {
      console.error(`Error procesando imagen ${file.originalname}:`, error);
    }
  }
};

/**
 * Extrae la etiqueta del nombre del archivo
 */
const extraerEtiqueta = (filename) => {
  return filename.split('_')[0].toLowerCase();
};

/**
 * Entrena el modelo (ajustado a los requisitos del trabajo)
 */
const entrenarModelo = async () => {
  if (dataset.length === 0) {
    throw new Error('No hay imágenes para entrenar');
  }

  console.log('🧠 Iniciando entrenamiento del modelo...');
  console.log(`📊 Total de imágenes: ${dataset.length}`);

  // Extraer etiquetas únicas
  etiquetas = [...new Set(dataset.map((d) => d.label))];
  const etiquetaToIndex = Object.fromEntries(etiquetas.map((e, i) => [e, i]));

  console.log(`📊 Clases detectadas: ${etiquetas.join(', ')}`);
  console.log(`🔢 Número de clases: ${etiquetas.length}`);

  // Verificar que tenemos todas las frutas requeridas
  const frutasFaltantes = FRUTAS_REQUERIDAS.filter(
    (f) => !etiquetas.includes(f)
  );
  if (frutasFaltantes.length > 0) {
    console.log(
      `⚠️ Frutas faltantes en el dataset: ${frutasFaltantes.join(', ')}`
    );
  }

  // Preparar tensores
  console.log('⚙️ Preparando datos para entrenamiento...');
  const xs = dataset.map((d) => d.tensor);
  const ys = dataset.map((d) => etiquetaToIndex[d.label]);

  const xsStacked = tf.stack(xs);
  const ysOneHot = tf.oneHot(tf.tensor1d(ys, 'int32'), etiquetas.length);

  // Crear modelo con arquitectura mejorada
  console.log('🏗️ Construyendo arquitectura del modelo...');
  model = tf.sequential();

  // Capa convolucional 1
  model.add(
    tf.layers.conv2d({
      inputShape: [64, 64, 3],
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
    })
  );

  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.dropout({ rate: 0.25 }));

  // Capa convolucional 2
  model.add(
    tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
    })
  );

  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.dropout({ rate: 0.25 }));

  // Capas densas
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.5 }));
  model.add(
    tf.layers.dense({
      units: etiquetas.length,
      activation: 'softmax',
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  console.log('📋 Resumen del modelo:');
  model.summary();

  // Configuración de entrenamiento según trabajo
  const EPOCHS = 50;
  console.log(`🚀 Iniciando entrenamiento (${EPOCHS} épocas)...`);
  const startTime = Date.now();

  const history = await model.fit(xsStacked, ysOneHot, {
    epochs: EPOCHS,
    validationSplit: 0.2,
    batchSize: 32,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(1);
        console.log(
          `📈 Época ${epoch + 1}/${EPOCHS}: ` +
            `Pérdida = ${logs.loss.toFixed(4)}, ` +
            `Precisión = ${(logs.acc * 100).toFixed(2)}%, ` +
            `Val_Pérdida = ${logs.val_loss.toFixed(4)}, ` +
            `Val_Precisión = ${(logs.val_acc * 100).toFixed(2)}% ` +
            `(${elapsedTime}s)`
        );
      },
    },
  });

  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`⏱️ Entrenamiento completado en ${totalTime} segundos`);

  // Guardar modelo
  console.log('💾 Guardando modelo...');

  try {
    // Guardar para el servidor
    await model.save('file://./models/modelo-frutas');
    console.log('✅ Modelo principal guardado');

    // Guardar para descarga
    await model.save('file://./models/modelo-frutas-descarga');
    console.log('✅ Modelo de descarga guardado');

    // Guardar metadatos
    const modeloInfo = {
      etiquetas,
      etiquetaToIndex,
      fechaEntrenamiento: new Date().toISOString(),
      totalImagenes: dataset.length,
      epocas: EPOCHS,
      arquitectura: 'CNN Secuencial',
      frutasRequeridas: FRUTAS_REQUERIDAS,
      precision: history.history.val_acc[history.history.val_acc.length - 1],
    };

    await fs.writeFile(
      './models/modelo-info.json',
      JSON.stringify(modeloInfo, null, 2)
    );
    await fs.writeFile(
      './models/etiquetas.json',
      JSON.stringify({ etiquetas, etiquetaToIndex })
    );

    console.log('✅ Metadatos guardados');

    // ✅ Marcar como entrenado para evitar bucle
    modeloYaEntrenado = true;
  } catch (error) {
    console.error('❌ Error guardando modelo:', error);
    throw error;
  }

  // Limpiar memoria
  xsStacked.dispose();
  ysOneHot.dispose();
  xs.forEach((tensor) => tensor.dispose());

  console.log('✅ Modelo entrenado y guardado exitosamente');
  console.log('📁 Archivos generados:');
  console.log('   ./models/modelo-frutas/ (para servidor)');
  console.log('   ./models/modelo-frutas-descarga/ (para descarga)');
  console.log('   ./models/modelo-info.json (metadatos)');

  return {
    etiquetas,
    history: history.history,
    totalImagenes: dataset.length,
    precision: modeloInfo.precision,
  };
};

/**
 * Carga un modelo existente
 */
const cargarModeloExistente = async () => {
  try {
    await fs.access('./models/modelo-frutas/model.json');

    console.log('📥 Cargando modelo existente...');
    model = await tf.loadLayersModel(
      'file://./models/modelo-frutas/model.json'
    );

    const etiquetasData = JSON.parse(
      await fs.readFile('./models/etiquetas.json', 'utf8')
    );
    etiquetas = etiquetasData.etiquetas;

    // ✅ Marcar como ya entrenado
    modeloYaEntrenado = true;

    console.log('✅ Modelo existente cargado correctamente');
    console.log(`📊 Clases disponibles: ${etiquetas.join(', ')}`);
    return true;
  } catch (error) {
    console.log('ℹ️ No se encontró modelo existente');
    modeloYaEntrenado = false; // ✅ Reset de la bandera
    return false;
  }
};

/**
 * Inicializa el modelo (carga existente o entrena nuevo) - ✅ CON PROTECCIÓN ANTI-BUCLE
 */
const inicializar = async () => {
  console.log('🚀 Inicializando sistema de clasificación...');
  console.log(`🍎 Frutas requeridas: ${FRUTAS_REQUERIDAS.join(', ')}`);

  // ✅ Verificar si ya se inicializó
  if (modeloYaEntrenado && model !== null) {
    console.log('✅ Modelo ya está inicializado y listo');
    return true;
  }

  // Intentar cargar modelo existente
  const modeloExistente = await cargarModeloExistente();
  if (modeloExistente) {
    return true;
  }

  // ✅ Solo entrenar si no hay modelo y no se ha entrenado ya
  if (!modeloYaEntrenado) {
    console.log('🔄 Entrenando nuevo modelo...');
    const datasetCargado = await cargarDatasetLocal();

    if (datasetCargado) {
      await entrenarModelo();
      return true;
    } else {
      console.log('⚠️ No se pudo cargar dataset.');
      console.log('💡 Pasos para solucionar:');
      console.log('   1. Ejecuta: python traer-frutas.py');
      console.log('   2. Agrega imágenes a cada carpeta de fruta');
      console.log('   3. Reinicia el servidor');
      return false;
    }
  }

  return true;
};

/**
 * Predice la clase de una nueva imagen
 */
const predecir = async (imageBuffer) => {
  if (!model) {
    throw new Error('No hay modelo disponible para predicción');
  }

  const buffer = await sharp(imageBuffer).resize(64, 64).raw().toBuffer();
  const tensorImg = tf
    .tensor3d(new Uint8Array(buffer), [64, 64, 3])
    .toFloat()
    .div(255.0)
    .expandDims(0);

  const prediccion = model.predict(tensorImg);
  const probabilidades = await prediccion.data();
  const maxIndex = probabilidades.indexOf(Math.max(...probabilidades));

  tensorImg.dispose();
  prediccion.dispose();

  return {
    clase: etiquetas[maxIndex],
    probabilidad: probabilidades[maxIndex],
    todasLasProbabilidades: etiquetas.map((etiqueta, i) => ({
      etiqueta,
      probabilidad: probabilidades[i],
    })),
  };
};

/**
 * Obtiene información del modelo actual
 */
const getInfo = () => {
  return {
    modeloEntrenado: model !== null,
    etiquetas,
    totalClases: etiquetas.length,
    frutasRequeridas: FRUTAS_REQUERIDAS,
    modeloYaEntrenado, // ✅ Info adicional para debugging
  };
};

export default {
  inicializar,
  cargarDatasetLocal,
  procesarImagenes, // ✅ Agregar esta función
  entrenarModelo,
  predecir,
  getInfo,
};
