import express from 'express';
import multer from 'multer';
import modelController from '../controllers/train.controller.js';
import archiver from 'archiver';
import fs from 'fs/promises';

const router = express.Router();

// Configurar multer para manejar subida de archivos
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Solo se permiten archivos de imagen'), false);
    }
  },
});

// Ruta para entrenar el modelo
router.post('/entrenar', upload.array('imagenes', 100), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({
        error: 'No se proporcionaron imágenes para entrenar',
      });
    }

    console.log(`Procesando ${req.files.length} imágenes...`);

    // Procesar imágenes
    await modelController.procesarImagenes(req.files);

    // Entrenar modelo
    const resultado = await modelController.entrenarModelo();

    res.json({
      mensaje: 'Modelo entrenado exitosamente',
      ...resultado,
    });
  } catch (error) {
    console.error('Error entrenando modelo:', error);
    res.status(500).json({
      error: 'Error entrenando el modelo',
      detalle: error.message,
    });
  }
});

// Ruta para hacer predicciones
router.post('/predecir', upload.single('imagen'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: 'No se proporcionó una imagen',
      });
    }

    const resultado = await modelController.predecir(req.file.buffer);

    res.json(resultado);
  } catch (error) {
    console.error('Error en predicción:', error);
    res.status(500).json({
      error: 'Error realizando la predicción',
      detalle: error.message,
    });
  }
});

// Ruta para obtener información del modelo
router.get('/info', async (req, res) => {
  try {
    const info = modelController.getInfo();
    res.json(info);
  } catch (error) {
    res.status(500).json({
      error: 'Error obteniendo información del modelo',
    });
  }
});

// Ruta para descargar el modelo entrenado
router.get('/descargar', async (req, res) => {
  try {
    console.log('🔍 Intentando descargar modelo...');

    // Verificar qué archivos existen
    const archivosModelo = [
      './models/modelo-frutas/model.json',
      './models/modelo-frutas-descarga/model.json',
      './models/modelo-info.json',
      './models/etiquetas.json',
    ];

    console.log('📂 Verificando archivos del modelo:');
    for (const archivo of archivosModelo) {
      try {
        await fs.access(archivo);
        console.log(`   ✅ ${archivo} - existe`);
      } catch (e) {
        console.log(`   ❌ ${archivo} - no existe`);
      }
    }

    // Intentar con modelo principal si no existe el de descarga
    let modeloPath = './models/modelo-frutas-descarga';
    try {
      await fs.access(modeloPath);
      console.log('✅ Usando modelo de descarga');
    } catch (e) {
      console.log('⚠️ No existe modelo de descarga, usando modelo principal');
      modeloPath = './models/modelo-frutas';
      await fs.access(modeloPath);
    }

    console.log('📦 Creando archivo ZIP...');

    // Crear archivo ZIP con el modelo
    const archive = archiver('zip', {
      zlib: { level: 9 }, // Máxima compresión
    });

    // Configurar headers para descarga
    res.attachment('modelo-frutas.zip');
    res.setHeader('Content-Type', 'application/zip');

    // Manejar errores del archiver
    archive.on('error', (err) => {
      console.error('❌ Error creando ZIP:', err);
      if (!res.headersSent) {
        res.status(500).json({ error: 'Error creando archivo de descarga' });
      }
    });

    // Pipe del archivo al response
    archive.pipe(res);

    // Agregar archivos del modelo al ZIP
    archive.directory(modeloPath, 'modelo');
    console.log(`📁 Agregado directorio: ${modeloPath}`);

    // Agregar archivos de metadatos si existen
    try {
      await fs.access('./models/modelo-info.json');
      archive.file('./models/modelo-info.json', { name: 'info.json' });
      console.log('📄 Agregado: modelo-info.json');
    } catch (e) {
      console.log('⚠️ No se encontró modelo-info.json');
    }

    try {
      await fs.access('./models/etiquetas.json');
      archive.file('./models/etiquetas.json', { name: 'etiquetas.json' });
      console.log('📄 Agregado: etiquetas.json');
    } catch (e) {
      console.log('⚠️ No se encontró etiquetas.json');
    }

    // Finalizar el archivo ZIP
    console.log('🏁 Finalizando archivo ZIP...');
    await archive.finalize();
    console.log('✅ Descarga completada');
  } catch (error) {
    console.error('❌ Error en descarga:', error);
    if (!res.headersSent) {
      res.status(404).json({
        error: 'Modelo no encontrado. Entrena el modelo primero.',
        detalle: error.message,
      });
    }
  }
});

export default router;
