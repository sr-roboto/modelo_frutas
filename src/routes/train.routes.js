import express from 'express';
import multer from 'multer';
import modelController from '../controllers/train.controller.js';

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

export default router;
